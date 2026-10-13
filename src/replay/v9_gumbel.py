import numpy as np
from replay.v5_policy import ReplayBuffer as PolicyBuffer

EXTRA = ("root_value",)


class ReplayBuffer(PolicyBuffer):
  def __init__(self, capacity: int, move_capacity: int = None):
    super().__init__(capacity, move_capacity)
    self.root_value = np.zeros(capacity, np.float32)


  def add(self, board, castling, ep, moves, num_moves, policy, z, distance,
          root_value) -> None:
    n = len(board)
    if n == 0: return
    super().add(board, castling, ep, moves, num_moves, policy, z, distance)

    idx = (self.ptr - n + np.arange(n)) % self.capacity  # rows just written
    self.root_value[idx] = root_value


  def sample_indices(self, n: int, rng: np.random.Generator):
    """n uniform rows from the live buffer"""
    return (self.ptr - self.size + rng.integers(self.size, size=n)) % self.capacity


  def gather(self, idx):
    return self._gather(idx) + (self.root_value[idx],)


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
        getattr(out, key)[:out.size] = data[key]
    return out
