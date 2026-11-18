#!/bin/bash
# merge a bundle (tools/export.sh) into this machine. It only adds: an observations file only grows (a copy that is older
# or does not continue the one here is skipped), any other file is added when missing and never replaced, and a
# checkpoint must load as plain tensors (torch weights_only) or it is dropped. Safe while the tuners run, and to repeat
#   tools/import.sh BUNDLE.tar.gz
set -euo pipefail
[ $# -eq 1 ] || { echo "usage: tools/import.sh BUNDLE.tar.gz"; exit 1; }
bundle=$(cd "$(dirname "$1")" && pwd)/$(basename "$1")
cd "$(dirname "$0")/.."
mkdir -p runs
venv/bin/python - "$bundle" <<'EOF'
import json, pathlib, shutil, sys, tarfile, tempfile
import torch

def lines(path):
  "complete lines only (a copy taken while a result was being written can end in half a line)"
  out = []
  for line in path.read_text().splitlines():
    try: json.loads(line)
    except ValueError: break
    out.append(line)
  return out

def plain_tensors(path):
  "a checkpoint from another machine is loaded only if it is plain data (a full torch.load could run code)"
  try: torch.load(path, map_location="cpu", weights_only=True)
  except Exception: return False
  return True

tmp =pathlib.Path(tempfile.mkdtemp(dir="runs"))
added = grown = dropped = 0
try:
  with tarfile.open(sys.argv[1]) as tar:
    tar.extractall(tmp, filter="data") # no paths outside tmp, no links out
  for src in sorted(p for p in tmp.rglob("*") if p.is_file()):
    dst = src.relative_to(tmp)
    if dst.parts[:2] != ("runs", "sweeps") and dst.parts[:2] != ("data", "scoring"): continue
    if dst.name.startswith("observations-") and dst.suffix == ".jsonl":
      new, old = lines(src), lines(dst) if dst.exists() else []
      if len(new) <= len(old) or new[:len(old)] != old: continue
      src.write_text("".join(line + "\n" for line in new))
      grown += 1
    elif dst.exists():
      continue
    elif dst.name == "state.pt" and not plain_tensors(src):
      print(f"dropped {dst}: not plain tensors")
      dropped += 1
      continue
    else:
      added += 1
    dst.parent.mkdir(parents=True, exist_ok=True)
    part = dst.with_name(dst.name + ".part")
    shutil.copyfile(src, part)
    part.replace(dst)
finally:
  shutil.rmtree(tmp)
print(f"imported {sys.argv[1]}: {grown} observation files grew, {added} files added, {dropped} checkpoints dropped")
EOF
