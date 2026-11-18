"""Protein sweep (tune/protein.py) of a run recipe. Runs until stopped: ctrl-c, a closed terminal or SIGTERM is a pause,
and the same command continues the run in flight.
  python -m tune configs/sweeps/protein-2.json [--hours H] [--background | --duty D] [--host NAME] [--max-hours H]
                                               [--device DEVICE]
  make tune protein-2 -- --duty 0.5                     the same through make (caffeinate)
Everything is in <runs_dir>/<name>/:
  observations-<host>.jsonl   one line per scored point, appended only by that PC
  pending-<host>.json         that PC's run in flight
  <host>-NNN/                 the runs (config, metrics, logs, final checkpoint)
Several PCs: give each its own --host (default: the hostname) and now and then move a bundle between them
(tools/export.sh, tools/import.sh; a CUDA machine: tools/worker.sh). Every pick reads all observations-*.jsonl."""
from dataclasses import dataclass, field
from functools import reduce
from pathlib import Path
from run import Run
from tune.protein import Param, Protein
from typing import Any
import argparse
import json
import math
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
import zlib

STOP_CODES = (-2, -15, -1, 130, 143, 129) # SIGINT, SIGTERM, SIGHUP: a pause, not a failure (SIGKILL -9 = a crash)

@dataclass
class SweepConfig:
  name: str
  recipe: str # the run config every run starts from
  space: dict[str, dict[str, Any]] # name -> {path, default, lo, hi, kind, integer, scale} (+ share_of, min, max)
  score: dict[str, Any] # {positions, labels, sims, lanes} (tune/score.py; scored on the run's device)
  step_time: str # reference machine's step-time parameters
  runs_dir: str = "runs/sweeps"
  points: int = 5 # scored at 1/points .. 100% of each run
  max_hours: float = 8.0 # cap on a run's predicted clean cost
  alpha: float = 1.25
  fixed: dict[str, Any] = field(default_factory=dict) # config path -> value, for every run
  cost_dims: list[str] = field(default_factory=list) # settings that change speed (inputs of the cost GP)
  snap: dict[str, int] = field(default_factory=dict) # name -> multiple (batch_size 16: Metal tiles; steps 5: points)

def run_config(
  sweep: SweepConfig,
  recipe: dict,
  values: dict[str, Any],
  seed: int = 0
) -> dict:
  "settings -> a full run config"
  config = json.loads(json.dumps(recipe)) # deep copy

  def put(path: str, value: Any) -> None:
    *parents, leaf = path.split(".")
    reduce(dict.__getitem__, parents, config)[leaf] = value

  for path, value in sweep.fixed.items(): put(path, value)
  # counts (e.g. fast_sims = share x n_sims, clamped) after the settings they are a share of
  for name, spec in sorted(sweep.space.items(), key=lambda item: "share_of" in item[1]):
    if "share_of" not in spec:
      put(spec["path"], values[name])
      continue
    total = reduce(dict.__getitem__, spec["share_of"].split("."), config)
    put(spec["path"], min(max(round(values[name] * total), spec.get("min", 1)), spec.get("max", total), total))

  train = config["train"]
  train.update(decay_length=train["max_steps"], checkpoint_interval=train["max_steps"] // sweep.points)
  train["fast_m"] = min(train["fast_m"], train["fast_sims"])
  config["node_cap"] = max(config["node_cap"], train["n_sims"] + 1)
  config["seed"] = seed
  return config

def run_child(cmd: list[str], log: Path) -> None:
  "own process, output appended to log; a crash fails the run (no retry, as Protein)"
  with open(log, "a") as out:
    process = subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT)
    try:
      code = process.wait()
    except KeyboardInterrupt: # the child got the same signal: let it finish (the trainer saves a checkpoint)
      process.wait()
      raise
  if code in STOP_CODES: raise KeyboardInterrupt
  assert code == 0, f"exit code {code}, see {log}"

def main() -> None:
  parser = argparse.ArgumentParser(prog="python -m tune")
  parser.add_argument("config")
  parser.add_argument("--hours", type=float, default=math.inf, help="start no new run after this many hours")
  parser.add_argument("--background", action="store_true", help="low priority, rests (--duty 0.25)")
  parser.add_argument("--duty", type=float, help="share of the time the trainer works, (0, 1]; implies --background")
  parser.add_argument("--host", default=socket.gethostname().split(".")[0].lower(), help="this PC's name in the sweep")
  parser.add_argument("--max-hours", type=float, help="this PC picks only runs predicted under this many clean hours")
  parser.add_argument("--device", help="train and score on this device (default: the recipe's)")
  args = parser.parse_args()
  if args.duty is None: args.duty = 0.25 if args.background else 1.0
  assert 0 < args.duty <= 1, "--duty must be in (0, 1]"
  if args.background or args.duty < 1: os.nice(15)
  # a closed terminal or a shutdown (SIGTERM) is a pause, like ctrl-c
  for sig in (signal.SIGHUP, signal.SIGTERM): signal.signal(sig, signal.default_int_handler)

  sweep = SweepConfig(**json.loads(Path(args.config).read_text()))
  if args.device: sweep.fixed["device"] = args.device
  if args.max_hours is None: args.max_hours = sweep.max_hours # e.g. a slower PC takes the cheaper runs
  recipe = json.loads(Path(sweep.recipe).read_text())
  # clean seconds of a logged step from its counted work (cost is the same on any PC and has no pauses). Self-play is
  # double buffered: per GPU batch the step pays max(CPU, GPU), smoothed with power K. Parameters: d extra share of
  # batches, s1 / s2 search seconds per node (+ x ln n_sims), s0 / book search / game bookkeeping per step, l1 encoding
  # per batch, g GPU per batch, t0 / t1 training per step / per row, o other per step
  st = json.loads(Path(sweep.step_time).read_text())
  folder = Path(sweep.runs_dir) / sweep.name
  folder.mkdir(parents=True, exist_ok=True)
  own, pending = folder / f"observations-{args.host}.jsonl", folder / f"pending-{args.host}.json"

  def basis(values: dict[str, Any]) -> tuple[float, list[float]]:
    "cost per step = a + b x sims per move + c x batch (search + training)"
    train = run_config(sweep, recipe, values)["train"]
    per_move = train["pcr_p"] * train["n_sims"] + (1 - train["pcr_p"]) * train["fast_sims"]
    return values["steps"], [1.0, per_move, train["batch_size"] / 1000]

  protein = Protein(
    {
      k: Param(**{f: s[f] for f in ("lo", "hi", "kind", "integer", "scale") if f in s})
      for k, s in sweep.space.items()
    },
    {k: s["default"] for k, s in sweep.space.items()},
    basis, sweep.cost_dims, args.max_hours * 3600, sweep.alpha,
    seed=zlib.crc32(args.host.encode()) # each PC draws its own candidates
  )
  snap = lambda values: {
    k: sweep.snap[k] * round(v / sweep.snap[k]) if k in sweep.snap else v
    for k, v in values.items()
  }
  end = time.time() + args.hours * 3600
  print(f"=== {sweep.name} on {args.host}: {folder} ===", flush=True)

  while True:
    observations = [
      json.loads(line)
      for path in sorted(folder.glob("observations-*.jsonl"))
      for line in path.read_text().splitlines()
    ]
    job = json.loads(pending.read_text()) if pending.exists() else None
    # a pending run whose results were recorded before a stop is done
    resumed = job is not None and all(o["run"] != job["run"] for o in observations)
    if not resumed:
      picked = protein.suggest(observations, snap) if time.time() < end else None
      if picked is None: break
      n = len({o["run"] for o in observations if o["host"] == args.host})
      job = dict(zip(("values", "predicted_cost", "predicted_score"), picked), run=f"{args.host}-{n:03d}")
      pending.write_text(json.dumps(job))

    v, run_dir = job["values"], folder / job["run"]
    guess = "no prediction (first run)" if job["predicted_cost"] is None else (
      f"predicted {job['predicted_cost'] / 60:.0f} clean min, play loss {-job['predicted_score']:.2f}"
    )
    print(
      f"{job['run']} | {'continuing' if resumed else 'starting'}: {v['steps']} steps, {v['n_sims']} sims, "
      f"batch {v['batch_size']}, lr {v['lr']:.2g}, wd {v['weight_decay']:.2g} / {v['adamw_weight_decay']:.2g} "
      f"| {guess}",
      flush=True
    )
    try:
      if not run_dir.exists():
        (run_dir / Run.CHECKPOINTS_DIR).mkdir(parents=True)
        config = run_config(sweep, recipe, v, zlib.crc32(job["run"].encode()) % 1_000_000)
        (run_dir / Run.CONFIG_FILE).write_text(json.dumps(config, indent=1))
      if not (run_dir / Run.COMPLETED_FILE).exists():
        run_child([
          sys.executable, "-u", "-m", "train", job["run"],
          "--configs-dir", str(folder), "--runs-dir", str(folder), "--duty", str(args.duty)
        ], run_dir / "train.log")
      steps = [v["steps"] * k // sweep.points for k in range(1, sweep.points + 1)]
      run_child([
        sys.executable, "-u", "-m", "tune.score", args.config, str(run_dir), *map(str, steps)
      ], run_dir / "score.log")
      play = json.loads((run_dir / "score.log").read_text().splitlines()[-1])

      train, seconds = json.loads((run_dir / Run.CONFIG_FILE).read_text())["train"], {}
      # after a pause a step can be logged twice: the last row wins
      for row in map(json.loads, (run_dir / Run.METRICS_FILE).read_text().splitlines()):
        batches = row["nn_evals"] / train["eval_batch"] * (1 + st["d"])
        search = row["nodes"] * (st["s1"] + st["s2"] * math.log(train["n_sims"]))
        cpu = search + st["s0"] + st["book"] + st["l1"] * batches
        selfplay = batches * ((cpu / batches) ** st["K"] + st["g"] ** st["K"]) ** (1 / st["K"])
        trained = st["t0"] + st["t1"] * train["batch_size"] if "policy_loss" in row else 0.0
        seconds[row["step"]] = selfplay + trained + st["o"]
      curve = [{
        "values": {**v, "steps": s}, "score": -play[str(s)], "steps": s, "play_loss": play[str(s)],
        "cost": sum(x for step, x in seconds.items() if step <= s)
      } for s in steps]
      # as PufferLib: mid-run points under the minimum length are not observations; the last one always is
      points = [p for p in curve[:-1] if p["steps"] >= sweep.space["steps"]["lo"]] + curve[-1:]
      print(f"{job['run']} | done: {v['steps']} steps, cost {curve[-1]['cost'] / 60:.1f} clean min | "
            f"play loss {curve[-1]['play_loss']:.2f} | {len(points)} point(s)", flush=True)
    except Exception as error: # ctrl-c is not an Exception: it stops the sweep, and the run continues next time
      print(f"{job['run']} | FAILED: {error!r}"[:400], flush=True)
      points = [{"values": v, "score": None, "cost": 0.0, "failed": True, "error": repr(error)[-400:]}]

    with open(own, "a") as f:
      f.write("".join(json.dumps({**p, "run": job["run"], "host": args.host}) + "\n" for p in points))
    pending.unlink()
    if "failed" in points[0]: continue
    # results first, then the clean-up: only the final checkpoint stays
    for path in (run_dir / Run.CHECKPOINTS_DIR).iterdir():
      if path.name != str(v["steps"]): shutil.rmtree(path)
    (run_dir / "buffer.npz").unlink(missing_ok=True)
  print("stopped: --hours is up, the first run failed, or every candidate is over max_hours", flush=True)

if __name__ == "__main__":
  main()
