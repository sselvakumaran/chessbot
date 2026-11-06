from dataclasses import dataclass
import numpy as np

from engine.game import GameBatch
from bot_base import ChessbotBase

@dataclass(frozen=True)
class Config:
  seed: int = 42

class Chessbot(ChessbotBase):
  VERSION = "v0_random"

  def __init__(self, config: Config):
    self.rng = np.random.default_rng(config.seed)

  def _select_moves(
    self,
    batch: GameBatch,
    mask: np.ndarray
  ):
    moves = np.zeros(len(batch), dtype=np.int32)
    num_moves = batch.num_moves_all()

    for lane in np.flatnonzero(mask):
      moves[lane] = self.rng.integers(
        num_moves[lane]
      )

    return moves, {}
  
def build(config: Config, _state: dict) -> Chessbot:
  return Chessbot(config)
