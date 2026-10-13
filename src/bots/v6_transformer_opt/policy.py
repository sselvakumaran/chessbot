import numpy as np
import torch
from engine.game import MAX_MOVES_CAPACITY

BOARD_OFFSET, CASTLING_OFFSET, EP_OFFSET, COUNT_OFFSET, MOVES_OFFSET = 0, 64, 65, 66, 67

def pack(board, castling, ep, num_moves, moves, max_moves):
  B = len(board)
  buffer = np.empty((B, MOVES_OFFSET + 4 * max_moves), np.uint8)
  buffer[:, BOARD_OFFSET:BOARD_OFFSET + 64] = board
  buffer[:, CASTLING_OFFSET] = castling
  # convert to -1 -> 0
  buffer[:, EP_OFFSET] = (ep.astype(np.int16) + 1).astype(np.uint8) 
  buffer[:, COUNT_OFFSET] = num_moves
  buffer[:, MOVES_OFFSET:] = moves.reshape(B, -1)
  return buffer


def unpack(buffer, max_moves):
  return (
    buffer[:, BOARD_OFFSET:BOARD_OFFSET + 64].long(),
    buffer[:, CASTLING_OFFSET].long(),
    buffer[:, EP_OFFSET].long() - 1,
    buffer[:, COUNT_OFFSET].long(),
    buffer[:, MOVES_OFFSET:].view(len(buffer), max_moves, 4).long()
  )


def chunk_forward(
  model,
  board, castling, ep,
  moves, num_moves,
  device
):
  B, M = len(board), model.max_moves
  logits = torch.full((B, MAX_MOVES_CAPACITY), float('-inf'), device=device)
  first_wdl = None
  moves_in_batch = int(num_moves.max())

  for offset in range(0, moves_in_batch, M):
    rows = np.arange(B) if offset == 0 else np.flatnonzero(num_moves > offset)
    total_move_count = min(M, moves_in_batch - offset)
    move_block = np.zeros((len(rows), M, 4), dtype=np.uint8)
    move_block[:, :total_move_count] = moves[rows, offset:offset + total_move_count]
    local_num_moves = np.clip(num_moves[rows] - offset, 0, M).astype(np.uint8)

    if offset == 0:
      buffer = torch.from_numpy(pack(
        board, castling, ep, local_num_moves, move_block, M
      )).to(device)
      b, c, e, n, mv = unpack(buffer, M)
      fixed = (b, c, e)
      rows_t = None
    else:
      row_ids = rows.astype('<u2').view(np.uint8).reshape(len(rows), 2)
      small = np.concatenate([
        row_ids,
        local_num_moves[:, None],
        move_block.reshape(len(rows), -1)
      ], axis=1)
      buffer = torch.from_numpy(small).to(device)
      rows_t = buffer[:, 0].long() | (buffer[:, 1].long() <<  8)
      n = buffer[:, 2].long()
      mv = buffer[:, 3:].view(len(rows), M, 4).long()
      b, c, e = (x[rows_t] for x in fixed)

    p, _, wdl = model(b, c, e, mv, n)
    # target = (
    #   logits[:, offset:offset + total_move_count]
    #   if rows_t is None else
    #   logits[rows_t, offset:offset + total_move_count]
    # )
    if rows_t is None:
      logits[:, offset:offset + total_move_count] = p[:, :total_move_count].to(logits.dtype)
    else:
      logits[rows_t, offset:offset + total_move_count] = p[:, :total_move_count].to(logits.dtype)
    if first_wdl is None:
      first_wdl = wdl

  return logits, first_wdl
