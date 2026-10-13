"""
Basic bot that uses a neural network to evaluate the board.
Uses alpha-beta pruning.
Training is just backed up via final outcome with discounting.
Values not scored off of cut-off games.
"""

# TODO: update v1_heuristic for better alpha-beta search (maybe split into own class)
# TODO: write this
# TODO: write v2 train.py. 
# TODO: change everything to torch instead of np

from dataclasses import dataclass, field

import numpy as np

import torch
from bot_base import ChessbotBase
from engine.game import GameBatch
from models.v2_mlp_onehot import Model
from models.v2_mlp_onehot import Config as ModelConfig
from search.v2_alpha_beta_temperature import AlphaBetaTemperature


@dataclass(frozen=True)
class TrainingConfig:
  max_steps: int = 10000
  lr: float = 3e-4
  temperature: float = 0.05
  discount: float = 0.986
  max_plies: int = 200
  weight_decay: float = 1e-4
  checkpoint_interval: int = 100
  betas: tuple[float, float] = (0.9, 0.95)
  eps: float = 1e-8


@dataclass(frozen=True)
class Config:
  model: ModelConfig = field(default_factory=ModelConfig)
  train: TrainingConfig = field(default_factory=TrainingConfig)
  max_depth: int = 3
  max_search_nodes: int = 1_048_576
  mate_value: float = 1000.0
  seed: int = 42
  temperature: float = 0.0
  device: str = "cpu"
  compile: bool = False


class Chessbot(ChessbotBase):
  VERSION = "v2_basic_nn"

  def __init__(
    self,
    config: Config,
    model: Model # assume model is compiled beforehand
  ):
    self.config = config
    self.rng = np.random.default_rng(self.config.seed)
    self.model = model
    self.model.eval()
    if self.config.compile:
      self.model.compile()

    self.search = AlphaBetaTemperature(
      evaluate=self._calculate_value,
      max_depth=self.config.max_depth,
      max_nodes=self.config.max_search_nodes,
      mate_value=self.config.mate_value,
      temperature=self.config.temperature,
      rng=self.rng
    )

  def _calculate_value(self, tree: GameBatch, state: int) -> float:
    board, *_ = tree.encode(
      states=np.array([state], np.int32)
    )
    board = torch.from_numpy(board).to(self.config.device)

    with torch.inference_mode():
      _, value = self.model(board)
    return value.item()

  def _select_moves(
    self,
    batch: GameBatch,
    mask: np.ndarray
  ):
    return self.search.select_moves(batch, mask)

def build(config: Config, state: dict) -> Chessbot:
  model = Model(config.model)
  model.load_state_dict(state["model"])
  model.to(config.device)
  return Chessbot(config, model=model)
