#!/bin/bash
# write a bundle: what this machine knows about the sweeps, to take to another machine (tools/import.sh there)
#   tools/export.sh [--weights] BUNDLE.tar.gz
# in it: data/scoring/ (the scoring positions and labels), every runs/sweeps/<sweep>/observations-*.jsonl, and the folder
# of every run with results (finished or failed; runs in flight are left out), without checkpoints. --weights: also the
# final checkpoint of each run with a point on its sweep's Pareto front (play loss vs cost)
set -euo pipefail
weights=0
[ "${1:-}" = --weights ] && { weights=1; shift; }
[ $# -eq 1 ] || { echo "usage: tools/export.sh [--weights] BUNDLE.tar.gz"; exit 1; }
out=$(cd "$(dirname "$1")" && pwd)/$(basename "$1")
cd "$(dirname "$0")/.."
venv/bin/python - "$out" $weights <<'EOF'
import json, pathlib, sys, tarfile
out, weights = pathlib.Path(sys.argv[1]), sys.argv[2] == "1"
files = sorted(pathlib.Path("data/scoring").glob("*.jsonl"))
n_runs = n_weights = 0
for sweep in sorted(p for p in pathlib.Path("runs/sweeps").glob("*") if p.is_dir()):
  observations = sorted(sweep.glob("observations-*.jsonl"))
  points = []
  for path in observations:
    for line in path.read_text().splitlines():
      try: points.append(json.loads(line))
      except ValueError: break # half a line: a result was being written
  front, best = set(), float("inf")
  for cost, loss, run in sorted((p["cost"], p["play_loss"], p["run"]) for p in points if not p.get("failed")):
    if loss < best: front.add(run); best = loss
  files += observations
  for run in sorted({p["run"] for p in points}):
    folder = sweep / run
    if not folder.is_dir(): continue
    n_runs += 1
    files += sorted(p for p in folder.rglob("*") if p.is_file() and "checkpoints" not in p.parts and p.suffix != ".npz")
    steps = [p for p in (folder / "checkpoints").glob("*") if p.name.isdigit() and (p / "state.pt").is_file()]
    if weights and run in front and steps:
      files.append(max(steps, key=lambda p: int(p.name)) / "state.pt")
      n_weights += 1

part = out.with_name(out.name + ".part")
with tarfile.open(part, "w:gz") as tar:
  for path in files: tar.add(path)
part.replace(out)
print(f"wrote {out}: {n_runs} runs, {n_weights} checkpoints, {out.stat().st_size / 1e6:.0f} MB")
EOF
