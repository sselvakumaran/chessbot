import numpy as np
import torch
from engine.game import MAX_MOVES_CAPACITY

# handle chunking: the model has max_moves dense move slots. positions with
# more legal moves are run again with the next block of moves. the FIRST chunk
# runs every row and produces the value; OVERFLOW chunks run only the rows that
# actually have moves past that offset (typically a handful out of hundreds).
# exact, not approximate: move tokens never attend to one another and fixed
# tokens never attend to moves, so a row's fixed-token computation and each
# move's logit are the same no matter which other moves share the chunk.

def chunk_forward(
  model,
  board, castling, ep,
  moves, num_moves,
  device
):
  B, M = len(board), model.max_moves
  fixed = [
    torch.from_numpy(np.ascontiguousarray(x)).to(device)
    for x in (board, castling, ep)
  ]
  logits = torch.full((B, MAX_MOVES_CAPACITY), float('-inf'), device=device)
  first_wdl = None
  max_moves = int(num_moves.max())

  for offset in range(0, max_moves, M):
    rows = np.arange(B) if offset == 0 else np.flatnonzero(num_moves > offset)
    count = min(M, max_moves - offset)
    packed = np.zeros((len(rows), M, 4), dtype=np.uint8)
    packed[:, :count] = moves[rows, offset:offset + count]
    local_counts = np.clip(num_moves[rows] - offset, 0, M).astype(np.int32)

    rows_t = torch.from_numpy(rows).to(device)
    inputs = fixed if offset == 0 else [x[rows_t] for x in fixed]
    p, _, wdl = model(
      *inputs,
      torch.from_numpy(packed).to(device),
      torch.from_numpy(local_counts).to(device)
    )
    logits[rows_t, offset:offset + count] = p[:, :count]
    if first_wdl is None:
      first_wdl = wdl

  return logits, first_wdl
