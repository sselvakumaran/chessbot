from functools import cached_property
from dataclasses import dataclass, field

import numpy as np

from engine.game import GameBatch
from bot_base import ChessbotBase
from search.v1_alpha_beta import AlphaBeta

# Uses material value as the sole heuristic.
# Uses minimax-search to find the best move.
# Has a training loop of self-play to create a simple learned position table.

@dataclass(frozen=True)
class TrainingConfig:
  num_games: int = 1024
  discount: float = 0.99
  max_plies: int = 200
  max_depth: int = 3
  epsilon: float = 0.1
  checkpoint_interval: int = 100

@dataclass(frozen=True)
class Config:
  # mailbox codes: 0 empty, 1-6 P N B R Q K, 7-12 opponent.
  material_values: tuple[float, ...] = (1, 3, 3, 5, 9)
  mate_v: float = 1000.0
  pst_weight: float = 1.0
  max_depth: int = 6
  max_search_nodes: int = 1_000_000 # max number of allowed nodes to search
  seed: int = 42 # for e-greedy
  epsilon: float = 0.0 # for e-greedy
  train: TrainingConfig = field(default_factory=TrainingConfig)

class Chessbot(ChessbotBase):
  VERSION = "v1_heuristic"

  def __init__(
    self, 
    config: Config, 
    pst: np.ndarray | None = None
  ):
    self.config = config
    self.rng = np.random.default_rng(self.config.seed)
    self.pst: np.ndarray | None = pst
    self.search = AlphaBeta(
      evaluate=self._calculate_value,
      order_actions=self._ordered_actions,
      max_depth=self.config.max_depth,
      max_nodes=self.config.max_search_nodes,
      mate_value=self.config.mate_v,
      epsilon=self.config.epsilon,
      rng=self.rng,
    )

  @cached_property
  def piece_values(self) -> np.ndarray:
    vals = np.zeros(13, dtype=np.float32)
    for i in range(5):
      vals[i+1] = self.config.material_values[i]
      vals[i+7] = -self.config.material_values[i]
    return vals

  def _calculate_value(self, tree: GameBatch, state: int) -> float:
    "calculate material value of current player"
    board, *_ = tree.encode(states=np.array([state], np.int32))
    v = float(self.piece_values[board[0]].sum())
    if self.pst is not None:
      v += self.config.pst_weight * float(self.pst[board[0], np.arange(64)].sum())
    return v
  
  def _ordered_actions(self, tree: GameBatch, state: int) -> np.ndarray:
    "order actions by prioritizing captures"
    board, *_, moves, num_moves = tree.encode(
      states=np.array([state], np.int32)
    )
    captured = board[0][moves[0, :num_moves[0], 1]]
    return np.argsort(self.piece_values[captured], kind="stable")

  def _select_moves(
    self,
    batch: GameBatch,
    mask: np.ndarray
  ):
    return self.search.select_moves(batch, mask)

def build(config: Config, state: dict) -> Chessbot:
  pst = np.asarray(
    state["pst"],
    dtype=np.float32
  )
  return Chessbot(config, pst=pst)
