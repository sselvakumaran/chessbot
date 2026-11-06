import numpy as np
import torch

BOARD_OFFSET, CASTLING_OFFSET, EP_OFFSET, COUNT_OFFSET, MOVES_OFFSET = 0, 64, 65, 66, 67

def pack(board, castling, ep, moves, num_moves, B, M):
  buffer = np.empty((B, MOVES_OFFSET + 4 * M), np.uint8)
  buffer[:, BOARD_OFFSET:BOARD_OFFSET + 64] = board
  buffer[:, CASTLING_OFFSET] = castling
  # convert to -1 -> 0
  buffer[:, EP_OFFSET] = (ep.astype(np.int16) + 1).astype(np.uint8) 
  buffer[:, COUNT_OFFSET] = num_moves
  buffer[:, MOVES_OFFSET:] = moves.reshape(B, -1)
  return buffer


def unpack(buffer, B, M):
  return (
    buffer[:, BOARD_OFFSET:BOARD_OFFSET + 64].long(),
    buffer[:, CASTLING_OFFSET].long(),
    buffer[:, EP_OFFSET].long() - 1,
    buffer[:, MOVES_OFFSET:].view(B, M, 4).long(),
    buffer[:, COUNT_OFFSET].long()
  )


M_BUCKETS = (32, 48, 64, 96, 128, 256)


def row_bucket(n):
  # 1 / 2 / 4 / 8 / 16 / 32 rows, then multiples of 32: few distinct shapes, and a single game's small batches are not
  # padded to 32 rows (~0.6 ms per call at batch 1)
  return 1 << max(n - 1, 0).bit_length() if n <= 32 else -(-n // 32) * 32

def pack_batch(
  board, castling, ep,
  moves, num_moves,
  device,
  pad=False,
  rows=None
):
  """-> (packed (Bp, 67 + 4M) uint8 rows on device, B, M). pad: rows up to row_bucket and move width to a bucket
  (fewer distinct shapes on MPS); rows: a fixed row count instead (the double-buffered self-play always sends exactly
  this many). Padding rows repeat row 0."""
  B, M = len(board), max(int(num_moves.max()), 1)
  Bp = B # bp = batch + padding

  if pad or rows:
    Bp = rows if rows else row_bucket(B)
    M = next(m for m in M_BUCKETS if m >= M)

    board, castling, ep, moves, num_moves = map(
      lambda a: np.concatenate(
        [a, np.repeat(a[:1], Bp - B, 0)]
      ) if Bp > B else a, 
      (board, castling, ep, moves, num_moves)
    )

  moves_padded = np.zeros((Bp, M, 4), np.uint8)
  n_cols = min(M, moves.shape[1])
  moves_padded[:, :n_cols] = moves[:, :n_cols]

  packed = torch.from_numpy(
    pack(board, castling, ep, moves_padded, num_moves, Bp, M)
  ).to(device)
  return packed, B, M


def forward_batch(
  model,
  board, castling, ep,
  moves, num_moves,
  device,
  pad=False,
  with_aux=False,
  rows=None
):
  packed, B, M = pack_batch(board, castling, ep, moves, num_moves, device, pad=pad, rows=rows)
  p, v, wdl, aux = model(
    *unpack(packed, len(packed), M), with_aux=with_aux
  )

  return p[:B], v[:B], wdl[:B], {k: x[:B] for k, x in aux.items()}