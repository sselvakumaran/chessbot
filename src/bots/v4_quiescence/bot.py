# v4 family: v3's model with alpha-beta, MVV-LVA, quiescence and an eval cache.
# root_prefetch enables v4.2's iterative-deepening search.
# See CHANGELOG.md for budget and exploration-score semantics.

from dataclasses import dataclass, field
from functools import cached_property

import numpy as np

from bot_base import ChessbotBase
from engine.game import GameBatch
import torch

from models.v3_mlp_nnue import Model
from models.v3_mlp_nnue import Config as ModelConfig
from search.v4_alpha_beta_quiescence import AlphaBetaQuiescence

from bots.v3_nnue_inspired.bot import TrainingConfig as V3TrainingConfig


@dataclass(frozen=True)
class TrainingConfig(V3TrainingConfig):
  # self-play searches shallower than inference: depth 4 is ~10x the evals
  max_depth: int = 3
  n_lanes: int = 32


@dataclass(frozen=True)
class Config:
  model: ModelConfig = field(default_factory=ModelConfig)
  train: TrainingConfig = field(default_factory=TrainingConfig)
  max_depth: int = 4
  tt_capacity: int = 0  # bounded search-result table; v4.2 enables it
  root_prefetch: bool = False  # v4.2: initial coverage + iterative deepening
  q_depth: int = 8           # max forcing plies past the horizon
  max_nodes: int = 12_000    # net evals per move (bounds q blowups)
  tree_cap: int = 32768      # search's own arena; ~states materialized/move
  sim_cap: int = 64          # caller batches: the search brings its own tree
  material_values: tuple[float, ...] = (1, 3, 3, 5, 9)  # P N B R Q (MVV-LVA)
  mate_value: float = 1000.0
  seed: int = 42
  temperature: float = 0.0
  device: str = "cpu"


class Chessbot(ChessbotBase):
  VERSION = "v4_quiescence"

  def __init__(self, config: Config, model: Model):
    self.config = config
    self.rng = np.random.default_rng(config.seed)
    self.model = model
    self.model.eval()
    self.search = AlphaBetaQuiescence(
      evaluate=self._evaluate,
      piece_values=self.piece_values,
      max_depth=config.max_depth,
      q_depth=config.q_depth,
      max_nodes=config.max_nodes,
      mate_value=config.mate_value,
      tree_cap=config.tree_cap,
      temperature=config.temperature,
      rng=self.rng,
      root_prefetch=config.root_prefetch,
      tt_capacity=config.tt_capacity,
    )

  @cached_property
  def piece_values(self) -> np.ndarray:
    "Value per mailbox code, both colors positive (MVV-LVA lookups)."
    values = np.zeros(13, np.float32)
    for i, v in enumerate(self.config.material_values):
      values[i + 1] = values[i + 7] = v
    values[6] = values[12] = 20.0  # king: worst attacker; never a victim
    return values

  def refresh(self) -> None:
    "Drop the eval cache (REQUIRED after the net's weights change)."
    self.search.clear_cache()

  def _evaluate(self, tree: GameBatch, state: int) -> float:
    board, castling, ep, *_ = tree.encode(states=np.array([state], np.int32))
    device = self.config.device
    with torch.inference_mode():
      _, value = self.model(
        torch.from_numpy(board).to(device),
        torch.from_numpy(castling).to(device),
        torch.from_numpy(ep).to(device),
      )
    return float(value[0])

  def _select_moves(self, batch: GameBatch, mask: np.ndarray):
    return self.search.select_moves(batch, mask)


def build(config: Config, state: dict) -> Chessbot:
  model = Model(config.model)
  model.load_state_dict(state["model"])
  model.to(config.device)
  return Chessbot(config, model=model)
