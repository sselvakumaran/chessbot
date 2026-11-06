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


def forward_batch(
  model,
  board, castling, ep,
  moves, num_moves,
  device
):
  B, M = len(board), max(int(num_moves.max()), 1)
  moves = np.ascontiguousarray(moves[:, :M])

  packed = torch.from_numpy(
    pack(board, castling, ep, moves, num_moves, B, M)
  ).to(device)

  p_logits, _, wdl_logits = model(
    *unpack(packed, B, M)
  )

  return p_logits, wdl_logits
