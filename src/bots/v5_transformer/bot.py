# v5: shallow transformer (policy + WDL value) with AlphaZero-style PUCT MCTS.
# See bots/v3_nnue_inspired/CHANGELOG.md ("V5 PLAN") for the search contract.

from collections import OrderedDict
from dataclasses import dataclass, field
from functools import cached_property

import numpy as np

from bot_base import ChessbotBase
from bots.v5_transformer.policy import chunk_forward
from engine.game import MAX_MOVES_CAPACITY, GameBatch
import torch

from models.v5_transformer import Model
from models.v5_transformer import Config as ModelConfig
from search.v5_mcts import MCTS, Evaluate


@dataclass(frozen=True)
class TrainingConfig:
  max_steps: int = 10000
  lr: float = 3e-4
  weight_decay: float = 1e-4
  betas: tuple[float, float] = (0.9, 0.95)
  eps: float = 1e-8
  grad_clip: float = 1.0     # max grad norm; 0 disables
  checkpoint_interval: int = 25

  # self-play. lanes are refilled the moment a game ends, so a step is a fixed
  # number of waves (one wave = one move on every lane), not "until all games
  # finish". positions per step = n_lanes * waves_per_step.
  max_plies: int = 300
  n_lanes: int = 16
  waves_per_step: int = 128
  # search / action selection
  n_sims: int = 128
  c_puct: float = 1.5
  temperature: float = 0.5
  # root exploration: prior <- (1-eps) * prior + eps * Dirichlet(alpha)
  dirichlet_alpha: float = 0.3
  dirichlet_epsilon: float = 0.25

  # replay buffer / targets
  buffer_capacity: int = 500_000
  min_buffer: int = 5_000
  batch_size: int = 128
  grad_steps: int = 32       # per round
  discount: float = 0.995     # target = discount**plies_to_end * onehot(z) + rest draw


@dataclass(frozen=True)
class Config:
  model: ModelConfig = field(default_factory=ModelConfig)
  train: TrainingConfig = field(default_factory=TrainingConfig)

  node_cap: int = 8192      # search's own arena; ~states materialized/move
  n_sims: int = 128
  c_puct: float = 1.5
  temperature: float = 0.0
  dirichlet_alpha: float = 0.3
  dirichlet_epsilon: float = 0.0  # evaluation: no root noise
  n_lanes: int = 1 # for caller GameBatch
  batch_size: int = 128 # for inference
  cache_capacity: int = 4096

  seed: int = 42
  device: str = "cpu"


class Chessbot(ChessbotBase):
  VERSION = "v5_transformer"

  def __init__(self, config: Config, model: Model):
    self.config = config
    self.rng = np.random.default_rng(config.seed)
    self.model = model.to(config.device).eval()
    self.search = MCTS(
      evaluate=self._evaluate, lane_count=config.n_lanes,
      max_nodes=config.node_cap + 1, max_sims=config.n_sims
    )
    self._cache = [OrderedDict() for _ in range(config.n_lanes)]
    self._stats = {}

  def refresh(self):
    for cache in self._cache:
      cache.clear()

  def reset_games(self, mask=None) -> None:
    if mask is None:
      self.refresh()
    else:
      for lane in np.flatnonzero(mask):
        self._cache[lane].clear()

  @torch.inference_mode()
  def _evaluate(self, tree: GameBatch, lanes: np.ndarray, states: np.ndarray):
    hashes = tree.hashes(lanes, states)
    priors = np.zeros((len(lanes), MAX_MOVES_CAPACITY), np.float32)
    values = np.zeros(len(lanes), np.float32)
    entries = [None for _ in range(len(lanes))]
    eval_queue = []
    for i, lane in enumerate(lanes):
      entries[i] = self._cache[lane].get(int(hashes[i]))
      if entries[i] is None:
        eval_queue.append(i)
      else:
        self._stats['cache_hits'] += 1

    for offset in range(0, len(eval_queue), self.config.batch_size):
      rows = np.asarray(eval_queue[offset:offset + self.config.batch_size], np.int32)
      board, castling, ep, _, _, moves, counts = tree.encode(lanes[rows], states[rows])

      p, wdl = chunk_forward(self.model, board, castling, ep, moves, counts, self.config.device)
      ps = wdl.float().softmax(-1)
      v = (ps[:, 0] - ps[:, 2]).cpu().numpy()
      p = p.float().cpu().numpy() # ROUGH!
      self._stats['nn_evals'] += len(rows)
      self._stats['model_calls'] += (int(counts.max()) + self.model.max_moves - 1) // self.model.max_moves
      for j, i in enumerate(rows):
        entry = p[j, :counts[j]].copy(), float(v[j])
        entries[i] = entry
        if self.config.cache_capacity:
          cache = self._cache[lanes[i]]
          cache[int(hashes[i])] = entry
          if len(cache) > self.config.cache_capacity:
            cache.popitem(last=False)
    for i, (logits, value) in enumerate(entries):
      probabilities = np.exp(logits - logits.max())
      priors[i, :len(logits)] = probabilities / probabilities.sum()
      values[i] = value
    return priors, values

  def _select_moves(self, batch: GameBatch, mask: np.ndarray):
    self.model.eval()
    self._stats = dict(cache_hits=0, nn_evals=0, model_calls=0)
    moves, policy, root_value = self.search.run(
      batch, self.rng, n_sims=self.config.n_sims, c_puct=self.config.c_puct,
      temperature=self.config.temperature, mask=mask,
      noise_alpha=self.config.dirichlet_alpha,
      noise_epsilon=self.config.dirichlet_epsilon,
    )
    return moves, {'policy': policy, 'value': root_value,
                   'nodes': self.search.node_count.copy(), **self._stats}

def build(config: Config, state: dict) -> Chessbot:
  model = Model(config.model)
  model.load_state_dict(state["model"])
  model.to(config.device)
  return Chessbot(config, model=model)