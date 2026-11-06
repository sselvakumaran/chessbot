# circular buffer of positions with their final-outcome label.
# z: game outcome from the position's mover's perspective (+1/0/-1);
# distance: plies from this position to the end of the game (for discounting).
# targets are built at training time from (z, distance) -- nothing search-
# derived is stored, so the data can't inherit a search's biases.

import numpy as np


class ReplayBuffer:
  def __init__(self, capacity: int):
    self.capacity, self.size, self.ptr = capacity, 0, 0
    self.board = np.zeros((capacity, 64), np.uint8)
    self.castling = np.zeros((capacity,), np.uint8)
    self.ep = np.zeros((capacity,), np.int8)
    self.z = np.zeros((capacity,), np.float32)
    self.distance = np.zeros((capacity,), np.int32)

  def add(self, board, castling, ep, z, distance) -> None:
    """Append n positions (arrays of matching length)."""
    n = len(board)
    idx = (self.ptr + np.arange(n)) % self.capacity
    self.board[idx] = board
    self.castling[idx] = castling
    self.ep[idx] = ep
    self.z[idx] = z
    self.distance[idx] = distance
    self.ptr = (self.ptr + n) % self.capacity
    self.size = min(self.size + n, self.capacity)

  def sample(self, batch_size: int, rng: np.random.Generator):
    idx = rng.integers(0, self.size, size=batch_size)
    return (
      self.board[idx], self.castling[idx], self.ep[idx],
      self.z[idx], self.distance[idx],
    )

  def save(self, path) -> None:
    n = self.size
    np.savez_compressed(
      path,
      board=self.board[:n], castling=self.castling[:n], ep=self.ep[:n],
      z=self.z[:n], distance=self.distance[:n],
      meta=np.array([self.capacity, n, self.ptr], dtype=np.uint64),
    )

  @classmethod
  def load(cls, path) -> "ReplayBuffer":
    with np.load(path) as data:
      capacity, n, ptr = (int(x) for x in data["meta"])
      out = cls(capacity)
      out.size, out.ptr = n, ptr
      for key in ("board", "castling", "ep", "z", "distance"):
        getattr(out, key)[:n] = data[key]
    return out
