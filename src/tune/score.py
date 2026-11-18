"""play loss of a run's checkpoints on positions with Stockfish labels: the expected-score loss (x100, lower = better)
of the move the bot's play search picks. Every run plays with the recipe's play settings at `sims` simulations, so runs
differ only by their weights.
  python -m tune.score SWEEP_CONFIG RUN_DIR STEP [STEP ...]     last stdout line = json {step: play_loss}"""
from bot_base import load_bot
from engine.game import GameBatch
from pathlib import Path
import json
import sys

import numpy as np

PROMOTION = {3: "n", 4: "b", 5: "r", 6: "q"}

def uci(row: np.ndarray, k: int, black: bool) -> str:
  "move k of a lane's encoded legal moves"
  f, t, kind = (int(x) for x in row[k, :3])
  if black: f, t = f ^ 56, t ^ 56
  return "".join(chr(97 + s % 8) + str(s // 8 + 1) for s in (f, t)) + PROMOTION.get(kind, "")

def main() -> None:
  sweep_path, run_dir, *steps = sys.argv[1:]
  sweep, run_dir = json.loads(Path(sweep_path).read_text()), Path(run_dir).resolve()
  s = sweep["score"]
  positions = {r["id"]: r["moves"] for r in map(json.loads, Path(s["positions"]).expanduser().read_text().splitlines())}
  labels = sorted(map(json.loads, Path(s["labels"]).expanduser().read_text().splitlines()), key=lambda r: r["id"])
  games = [positions[r["id"]] for r in labels]
  recipe = json.loads(Path(sweep["recipe"]).read_text())
  play = {k: v for k, v in recipe.items() if k not in ("version", "model", "train")}
  device = json.loads((run_dir / "config.json").read_text())["device"] # scored on the device the run trained on
  play.update(n_lanes=s["lanes"], n_sims=s["sims"], device=device, node_cap=max(recipe["node_cap"], s["sims"] + 1))

  # one lane per game, replayed to the end of its move list
  batch = GameBatch(len(games), sim_cap=play["node_cap"])
  for ply in range(max(map(len, games))):
    *_, moves, counts = batch.encode()
    black = batch.to_moves() == 1
    mask = np.array([ply < len(game) for game in games])
    actions = [
      next(k for k in range(counts[i]) if uci(moves[i], k, black[i]) == game[ply]) if live else 0
      for i, (game, live) in enumerate(zip(games, mask))
    ]
    batch.play_batch(np.array(actions, np.int32), mask=mask)
  *_, moves, counts = batch.encode()
  black = batch.to_moves() == 1
  legal = [[uci(moves[i], k, black[i]) for k in range(counts[i])] for i in range(len(games))]

  expected = lambda cp: 1 / (1 + np.exp(-0.00368208 * cp)) # expected score from centipawns (Lichess)
  out = {}
  for step in steps:
    bot = load_bot(run_dir.name, step, configs_dir=run_dir.parent, runs_dir=run_dir.parent, overrides=play)
    batch.reset_trees()
    bot.refresh()
    chosen, _ = bot.select_moves(batch, mask=np.ones(len(games), bool))
    losses = [ # nan: a move Stockfish did not label
      expected(max(r["cp"].values())) - expected(r["cp"][u]) if (u := legal[i][int(chosen[i])]) in r["cp"] else np.nan
      for i, r in enumerate(labels)
    ]
    out[step] = float(np.nanmean(losses) * 100)
  print(json.dumps(out))

if __name__ == "__main__":
  main()
