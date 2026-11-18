#!/bin/bash
# set up a Linux machine with NVIDIA GPUs (bf16: RTX 30xx / A100 or newer) for a sweep and start its tuners in the tmux
# session "chessbot": WORKERS_PER_GPU per GPU, each its own host NAME-g<gpu>w<n>. Copy this file and a bundle
# (tools/export.sh) to the machine, then
#   bash worker.sh BUNDLE.tar.gz NAME       env: WORKERS_PER_GPU (2), MAX_HOURS (8), SWEEP (protein-2), BRANCH (main)
# NAME: one per machine, never reused. Run it again with a newer bundle while the tuners run: imports it, nothing else.
# Not running (a reboot, or tmux kill-session -t chessbot, which pauses the tuners with a checkpoint): pulls the newest
# code, rebuilds, imports the bundle and starts the tuners; each continues its run in flight.
# results back: bash ~/chessbot/tools/export.sh --weights out.tar.gz here, tools/import.sh out.tar.gz there
set -euo pipefail
[ $# -eq 2 ] || { echo "usage: bash worker.sh BUNDLE.tar.gz NAME"; exit 1; }
bundle=$(realpath "$1") name=$2
repo=~/chessbot sweep=${SWEEP:-protein-2} per_gpu=${WORKERS_PER_GPU:-2}
if tmux has-session -t chessbot 2>/dev/null; then exec bash "$repo/tools/import.sh" "$bundle"; fi

{ command -v git && command -v tmux && command -v c++ && command -v curl; } >/dev/null ||
  { apt-get update -qq && apt-get install -yqq git tmux build-essential curl >/dev/null; }
[ -d "$repo" ] || git clone -q -b "${BRANCH:-main}" https://github.com/sselvakumaran/chessbot.git "$repo"
cd "$repo"
git pull -q
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null
export PATH=~/.local/bin:$PATH
[ -d venv ] || uv venv -q --python 3.11 venv # the repo needs python 3.11+; many images have an older one
uv pip install -q --python venv/bin/python torch==2.13.0 numpy==2.4.6 tornado -e .
gpus=$(venv/bin/python -c "
import torch
n = torch.cuda.device_count()
assert n and all(torch.cuda.get_device_capability(i) >= (8, 0) for i in range(n)), 'needs NVIDIA GPUs with bf16 (RTX 30xx / A100 or newer)'
print(n)")
make -s

shopt -s nullglob
own=(runs/sweeps/"$sweep"/observations-"$name"-g*.jsonl)
if [ ${#own[@]} -eq 0 ] && tar -tzf "$bundle" | grep -q "^runs/sweeps/$sweep/observations-$name-g"; then
  echo "the bundle already has results from $name-g*: another machine uses the name $name, pick another"
  exit 1
fi
bash tools/import.sh "$bundle"
# a crash while a result was being appended can leave half a line, which the tuner cannot read
for f in "${own[@]}"; do venv/bin/python -c "
import sys
text = open(sys.argv[1]).read()
open(sys.argv[1], 'w').write(text[:text.rfind(chr(10)) + 1])" "$f"; done

mkdir -p runs/sweeps/"$sweep"
for ((g = 0; g < gpus; g++)); do
  for ((w = 0; w < per_gpu; w++)); do
    host=$name-g${g}w$w
    cmd="CUDA_VISIBLE_DEVICES=$g OMP_NUM_THREADS=1 venv/bin/python -m tune configs/sweeps/$sweep.json --host $host"
    cmd+=" --device cuda --max-hours ${MAX_HOURS:-8} 2>&1 | tee -a runs/sweeps/$sweep/tune-$host.log"
    if tmux has-session -t chessbot 2>/dev/null; then tmux new-window -t chessbot -c "$repo" -n "$host" "$cmd"
    else tmux new-session -d -s chessbot -c "$repo" -n "$host" "$cmd"; fi
  done
done
echo "started $((gpus * per_gpu)) tuners: tmux attach -t chessbot (one window each; logs in runs/sweeps/$sweep/tune-*.log)"
