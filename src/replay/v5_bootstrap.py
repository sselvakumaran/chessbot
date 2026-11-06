# v5.1 buffer: the v2 schema plus per-position columns for bootstrapped value
# targets. root_value: the search's root Q at that position, mover perspective
# (nan = unknown, e.g. rows loaded from an older buffer); trusted: the game
# ended by a rule-decided outcome or resignation, so z is evidence about the
# position; value_weight: per-position value-loss weight.

import numpy as np
from replay.v5_policy import ReplayBuffer as PolicyBuffer

EXTRA = ("value_weight", "root_value", "trusted")


class ReplayBuffer(PolicyBuffer):
  def __init__(self, capacity: int, move_capacity: int = None):
    super().__init__(capacity, move_capacity)
    self.value_weight = np.ones(capacity, np.float32)
    self.root_value = np.full(capacity, np.nan, np.float32)
    self.trusted = np.ones(capacity, np.uint8)

  def add(self, board, castling, ep, moves, num_moves, policy, z, distance,
          value_weight=None, root_value=None, trusted=None) -> None:
    n = len(board)
    if n == 0: return
    super().add(board, castling, ep, moves, num_moves, policy, z, distance)
    idx = (self.ptr - n + np.arange(n)) % self.capacity  # rows just written
    self.value_weight[idx] = 1.0 if value_weight is None else value_weight
    self.root_value[idx] = np.nan if root_value is None else root_value
    self.trusted[idx] = 1 if trusted is None else trusted

  def sample(self, batch_size: int, rng: np.random.Generator):
    """v2 columns, then value_weight, root_value, trusted"""
    idx = (self.ptr - self.size + rng.integers(self.size, size=batch_size)) % self.capacity
    return self._gather(idx) + (self.value_weight[idx], self.root_value[idx], self.trusted[idx])

  def save(self, path) -> None:
    idx = (self.ptr - self.size + np.arange(self.size)) % self.capacity
    move_idx = (self.move_ptr - self.moves_size) % self.move_capacity
    move_addresses = (move_idx + np.arange(self.moves_size)) % self.move_capacity
    np.savez_compressed(
      path,
      board=self.board[idx], castling=self.castling[idx], ep=self.ep[idx],
      moves=self.moves[move_addresses], num_moves=self.num_moves[idx],
      policy=self.policy[move_addresses], z=self.z[idx], distance=self.distance[idx],
      **{key: getattr(self, key)[idx] for key in EXTRA},
      meta=np.array([self.capacity, self.move_capacity], dtype=np.uint64),
    )

  @classmethod
  def load(cls, path) -> "ReplayBuffer":
    out = super().load(path)
    with np.load(path) as data:
      for key in EXTRA:
        if key in data:  # buffers written by v5 predate these columns
          getattr(out, key)[:out.size] = data[key]
    return out
