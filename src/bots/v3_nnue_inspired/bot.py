# v3: nnue-flavored eval (sparse piece/castling/ep features -> embedding sum
# -> tiny head) driven by best-first minimax with batched child evaluation.
# the search runs on the caller's GameBatch, so whoever builds it sizes
# sim_cap for ~n_sims * branching states (trainer uses config.sim_cap).

from dataclasses import dataclass, field

import numpy as np
import torch

from bot_base import ChessbotBase
from engine.game import GameBatch
from models.v3_mlp_nnue import Model
from models.v3_mlp_nnue import Config as ModelConfig
from search.v3_best_first_minimax import BestFirstMinimax


@dataclass(frozen=True)
class TrainingConfig:
  max_steps: int = 10000
  lr: float = 3e-4
  weight_decay: float = 1e-4
  betas: tuple[float, float] = (0.9, 0.95)
  eps: float = 1e-8
  checkpoint_interval: int = 25

  # self-play
  temperature: float = 0.05  # over root values, whole game
  max_plies: int = 300
  n_lanes: int = 128         # games per round

  # replay buffer / targets
  buffer_capacity: int = 500_000
  min_buffer: int = 5_000
  batch_size: int = 512
  grad_steps: int = 32       # per round
  discount: float = 0.99     # target = discount**plies_to_end * onehot(z) + rest draw


@dataclass(frozen=True)
class Config:
  model: ModelConfig = field(default_factory=ModelConfig)
  train: TrainingConfig = field(default_factory=TrainingConfig)
  n_sims: int = 48           # expansions per move (each evaluates all children)
  sim_cap: int = 4096        # C++ arena per lane; ~n_sims * branching + slack
  mate_value: float = 1000.0
  seed: int = 42
  temperature: float = 0.0
  device: str = "cpu"


class Chessbot(ChessbotBase):
  VERSION = "v3_nnue_inspired"

  def __init__(self, config: Config, model: Model):
    self.config = config
    self.rng = np.random.default_rng(config.seed)
    self.model = model
    self.model.eval()
    self.search = BestFirstMinimax(
      evaluate=self._evaluate_states,
      n_sims=config.n_sims,
      mate_value=config.mate_value,
      temperature=config.temperature,
      rng=self.rng,
    )

  def refresh(self) -> None:
    "Called after the model's weights change (nothing cached here)."

  def _evaluate_states(self, batch: GameBatch, lanes: np.ndarray,
                       states: np.ndarray) -> np.ndarray:
    # encode buffers hold len(batch) rows -> encode in chunks, ONE forward
    B = len(batch)
    chunks = [
      [a.copy() for a in batch.encode(lanes[i:i + B], states[i:i + B])[:3]]
      for i in range(0, len(lanes), B)
    ]
    board, castling, ep = (np.concatenate(rows) for rows in zip(*chunks))
    device = self.config.device
    with torch.inference_mode():
      _, value = self.model(
        torch.from_numpy(board).to(device),
        torch.from_numpy(castling).to(device),
        torch.from_numpy(ep).to(device),
      )
    return value.float().cpu().numpy()

  def _select_moves(self, batch: GameBatch, mask: np.ndarray):
    return self.search.select_moves(batch, mask)


def build(config: Config, state: dict) -> Chessbot:
  model = Model(config.model)
  model.load_state_dict(state["model"])
  model.to(config.device)
  return Chessbot(config, model=model)
