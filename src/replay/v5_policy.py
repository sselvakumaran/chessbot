# circular buffer of positions with their final-outcome label.
# z: game outcome from the position's mover's perspective (+1/0/-1);
# distance: plies from this position to the end of the game (for discounting).
# targets are built at training time from (z, distance) -- nothing search-
# derived is stored, so the data can't inherit a search's biases.

import numpy as np
from engine.game import MAX_MOVES_CAPACITY


class ReplayBuffer:
  def __init__(self, capacity: int, move_capacity: int = None):
    self.capacity, self.size, self.ptr = capacity, 0, 0
    self.move_capacity = move_capacity or capacity * 48
    self.move_ptr = self.moves_size = 0
    self.board = np.zeros((capacity, 64), np.uint8)
    self.castling = np.zeros((capacity,), np.uint8)
    self.ep = np.zeros((capacity,), np.int8)
    self.num_moves = np.zeros(capacity, np.int32)
    self.z = np.zeros((capacity,), np.float32)
    self.distance = np.zeros((capacity,), np.int32)

    self.move_offsets = np.zeros(capacity, np.int64)
    self.moves = np.zeros((self.move_capacity, 4), np.uint8)
    self.policy = np.zeros(self.move_capacity, np.float32)


  def add(self, board, castling, ep, moves, num_moves, policy, z, distance) -> None:
    """Append n positions (arrays of matching length)."""
    n = len(board)
    if n == 0: return

    move_counts = np.asarray(num_moves, dtype=np.int64)
    total_moves = int(move_counts.sum())
    assert n <= self.capacity and total_moves <= self.move_capacity

    # probably can make faster
    while self.size + n > self.capacity or self.moves_size + total_moves > self.move_capacity:
      oldest = (self.ptr - self.size) % self.capacity
      self.moves_size -= int(self.num_moves[oldest])
      self.size -= 1

    idx = (self.ptr + np.arange(n)) % self.capacity
    move_offset = np.cumsum(move_counts) - move_counts
    self.board[idx] = board
    self.castling[idx] = castling
    self.ep[idx] = ep
    self.num_moves[idx] = move_counts
    self.z[idx] = z
    self.distance[idx] = distance
    self.move_offsets[idx] = (self.move_ptr + move_offset) % self.move_capacity

    nonpadded_moves = np.arange(moves.shape[1])[None, :] < move_counts[:, None]
    move_addresses = (self.move_ptr + np.arange(total_moves)) % self.move_capacity
    self.moves[move_addresses] = moves[nonpadded_moves]
    self.policy[move_addresses] = policy[nonpadded_moves]
    self.move_ptr = (self.move_ptr + total_moves) % self.move_capacity
    self.moves_size += total_moves
    self.size += n
    self.ptr = (self.ptr + n) % self.capacity
    

  def sample(self, batch_size: int, rng: np.random.Generator):
    """board, castling, ep, moves, num_moves policy, z, distance"""
    idx = (self.ptr - self.size + rng.integers(self.size, size=batch_size)) % self.capacity
    return self._gather(idx)

  def _gather(self, idx):
    num_moves = self.num_moves[idx]
    columns = np.arange(int(num_moves.max()))
    nonpadded_moves = columns[None, :] < num_moves[:, None]
    addresses = (self.move_offsets[idx, None] + columns[None, :]) % self.move_capacity
    moves = self.moves[addresses]
    policy = self.policy[addresses]
    moves[~nonpadded_moves] = 0
    policy[~nonpadded_moves] = 0
    return (
      self.board[idx], self.castling[idx], self.ep[idx],
      moves, num_moves, policy,
      self.z[idx], self.distance[idx],
    )


  def save(self, path) -> None:
    idx = (self.ptr - self.size + np.arange(self.size)) % self.capacity
    move_idx = (self.move_ptr - self.moves_size) % self.move_capacity
    move_addresses = (move_idx + np.arange(self.moves_size)) % self.move_capacity
    np.savez_compressed(
      path,
      board=self.board[idx], castling=self.castling[idx], ep=self.ep[idx],
      moves=self.moves[move_addresses], 
      num_moves=self.num_moves[idx], 
      policy=self.policy[move_addresses],
      z=self.z[idx], distance=self.distance[idx],
      meta=np.array([self.capacity, self.move_capacity], dtype=np.uint64),
    )

  # fields = ('board', 'castling', 'ep', 'num_moves', 'z', 'distance')
  @classmethod
  def load(cls, path) -> "ReplayBuffer":
    with np.load(path) as data:
      out = cls(*(int(x) for x in data["meta"]))
      out.size = len(data['num_moves'])
      out.ptr = out.size % out.capacity

      for key in ("board", "castling", "ep", "num_moves", "z", "distance"):
        getattr(out, key)[:out.size] = data[key]

      out.moves_size = len(data['moves'])
      out.move_ptr = out.moves_size % out.move_capacity
      out.moves[:out.moves_size] = data['moves']
      out.policy[:out.moves_size] = data['policy']

      if out.size:
        out.move_offsets[0] = 0
        out.move_offsets[1:out.size] = np.cumsum(
          out.num_moves[:out.size - 1], dtype=np.int64
        )
    return out
