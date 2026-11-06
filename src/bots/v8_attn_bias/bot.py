# v5: shallow transformer (policy + WDL value) with AlphaZero-style PUCT MCTS.
# See IDEAS.md ("V5 PLAN") for the search contract.

from dataclasses import dataclass, field
import copy

import numpy as np

from bot_base import ChessbotBase
from bots.v8_attn_bias.policy import forward_batch
from engine.game import MAX_MOVES_CAPACITY, GameBatch
import torch

from models.v8_attn_bias import Model
from models.v8_attn_bias import Config as ModelConfig
from search.v6_mcts_lazy import MCTS


@dataclass(frozen=True)
class TrainingConfig:
  max_steps: int = 10000
  lr: float = 3e-4
  weight_decay: float = 1e-4
  betas: tuple[float, float] = (0.9, 0.95)
  eps: float = 1e-8
  grad_clip: float = 1.0     # max grad norm; 0 disables
  checkpoint_interval: int = 25

  max_plies: int = 300
  n_lanes: int = 16
  waves_per_step: int = 128
  # search / action selection
  n_sims: int = 128
  c_puct: float = 1.5
  temperature: float = 0.5
  dirichlet_alpha: float = 0.3
  dirichlet_epsilon: float = 0.25

  # replay buffer / targets
  buffer_capacity: int = 500_000
  min_buffer: int = 5_000
  batch_size: int = 128
  grad_steps: int = 32       # per round
  discount: float = 0.995     # target = discount**plies_to_end * onehot(z) + rest draw

  value_frac_trained: float = 1.0 # reduce over-learning value
  value_weight: float = 1.0 # value loss weight; <1 keeps value's noisy gradient from swamping the trunk (v8.1: 0.25)


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

  use_fpu: bool = False
  fpu: float = 0.4
  fpu_root: float = 0.0

  seed: int = 42
  device: str = "cpu"
  autocast: bool = False
  cache_capacity: int = 4096 # capacity
  cache_slots: int = 1 << 17 # global slots



class Chessbot(ChessbotBase):
  VERSION = "v8_attn_bias"

  def __init__(self, config: Config, model: Model):
    self.config = config
    self.rng = np.random.default_rng(config.seed)
    self.model = model.to(config.device).eval()

    self.infer_model = self.model
    if config.autocast:
      self.infer_model = (
        copy.deepcopy(self.model)
        .to(torch.bfloat16).eval()
        .requires_grad_(False)
      )
    
    self.search = MCTS(
      evaluate=self._evaluate, lane_count=config.n_lanes,
      max_nodes=config.node_cap + 1, max_sims=config.n_sims,
      use_fpu=config.use_fpu
    )

    self._slots = config.cache_slots if config.cache_capacity else 0
    self._c_priors = np.zeros((self._slots, MAX_MOVES_CAPACITY), np.float32)
    self._c_values = np.zeros(self._slots, np.float32)
    self._c_hashes = np.zeros(self._slots, np.int64)
    self._c_used = np.zeros(self._slots, bool)
    self._c_index = dict() # hash -> row
    self._c_ptr = 0

    self._stats = dict()


  def refresh(self):
    if self.infer_model is not self.model:
      self.infer_model.load_state_dict(self.model.state_dict())

    self._c_index.clear()
    self._c_used[:] = False


  @torch.inference_mode()
  def _evaluate(self, tree: GameBatch, lanes: np.ndarray, states: np.ndarray):
    N = len(lanes)
    hashes = tree.hashes(lanes, states).view(np.int64)
    priors = np.zeros((N, MAX_MOVES_CAPACITY), np.float32)
    values = np.zeros(N, np.float32)

    hit = np.zeros(N, bool)
    if self._slots:
      slots = np.fromiter(
        (self._c_index.get(k, -1) for k in hashes.tolist()), 
        np.int64, N
      )
      hit = slots >= 0
      hit[hit] = (
        self._c_used[slots[hit]]
        & (self._c_hashes[slots[hit]] == hashes[hit])
      )
      priors[hit] = self._c_priors[slots[hit]]
      values[hit] = self._c_values[slots[hit]]
    self._stats['cache_hits'] += int(hit.sum())

    miss = np.flatnonzero(~hit)

    for offset in range(0, len(miss), self.config.batch_size):
      rows = miss[offset:offset + self.config.batch_size]
      board, castling, ep, _, _, moves, counts = tree.encode(
        lanes[rows], states[rows]
      )

      p, _, wdl, _ = forward_batch(
        self.infer_model, 
        board, castling, ep, 
        moves, counts, 
        self.config.device,
        pad=True
      )

      probs = p.float().softmax(-1)
      ps = wdl.float().softmax(-1)
      v = ps[:, 0] - ps[:, 2]
      out = torch.cat([probs, v[:, None]], dim=1).cpu().numpy()
      M = probs.shape[1]
      priors[rows, :M] = out[:, :M]
      values[rows] = out[:, M]
      self._stats['nn_evals'] += len(rows)

    if self._slots and len(miss):
      n = min(len(miss), self._slots)
      c_hashes = hashes[miss][-n:]
      c_priors = priors[miss][-n:]
      c_values = values[miss][-n:]
      c_slots = (self._c_ptr + np.arange(n)) % self._slots
      self._c_ptr = int((self._c_ptr + n) % self._slots)
      self._c_priors[c_slots], self._c_values[c_slots] = c_priors, c_values
      self._c_hashes[c_slots], self._c_used[c_slots] = c_hashes, True
      self._c_index.update(zip(c_hashes.tolist(), c_slots.tolist()))
      if len(self._c_index) > 2 * self._slots:
        live_slots = np.flatnonzero(self._c_used)
        self._c_index = dict(
          zip(self._c_hashes[live_slots].tolist(), live_slots.tolist())
        )

    return priors, values

  def _select_moves(self, batch: GameBatch, mask: np.ndarray):
    self.model.eval()
    self._stats = dict(cache_hits=0, nn_evals=0, model_calls=0)
    moves, policy, root_value = self.search.run(
      batch, self.rng, n_sims=self.config.n_sims, c_puct=self.config.c_puct,
      temperature=self.config.temperature, mask=mask,
      noise_alpha=self.config.dirichlet_alpha,
      noise_epsilon=self.config.dirichlet_epsilon,
      fpu=self.config.fpu, fpu_root=self.config.fpu_root
    )
    return moves, {'policy': policy, 'value': root_value,
                   'nodes': self.search.node_count.copy(), **self._stats}



def build(config: Config, state: dict) -> Chessbot:
  model = Model(config.model)
  model.load_state_dict(state["model"])
  model.to(config.device)
  return Chessbot(config, model=model)