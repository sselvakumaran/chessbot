# v10: v9 (Gumbel search, PCR, root-Q targets) with the search in C++ (search/v10_gumbel_cpp) and double-buffered
# self-play (train.py). The network and the training targets are the same as v9.

from dataclasses import dataclass, field
import copy
import weakref

import numpy as np
import torch

from bot_base import ChessbotBase
from bots.v10_double_buffering.policy import forward_batch
from engine.game import MAX_MOVES_CAPACITY, GameBatch
from models.v8_attn_bias import Model
from models.v8_attn_bias import Config as ModelConfig
from search.v10_gumbel_cpp import GumbelSearch


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
  waves_per_step: int = 128  # moves per RL step = n_lanes * waves_per_step (lanes no longer move in lockstep)
  eval_batch: int = 128      # self-play GPU batch: always exactly this many rows; keep n_lanes >= 2 * eval_batch
  # search / action selection
  n_sims: int = 128
  c_puct: float = 1.5

  # replay buffer / targets
  buffer_capacity: int = 500_000
  min_buffer: int = 5_000
  batch_size: int = 128
  grad_steps: int = 32       # per round
  discount: float = 0.995     # target = discount**plies_to_end * onehot(z) + rest draw

  value_weight: float = 1.0 # value loss weight; <1 keeps value's noisy gradient from swamping the trunk (v8.1: 0.25)
  rootq_alpha: float = 0.5 # value target mixing w/ search
  pcr_p: float = 0.25 # playout cap randomization: chance a move gets a full (recorded) search, per lane per move
  fast_sims: int = 16
  fast_m: int = 8
  opening_noise_plies: int = 0  # fast searches also get gumbel noise for the first this-many plies of each game


@dataclass(frozen=True)
class Config:
  model: ModelConfig = field(default_factory=ModelConfig)
  train: TrainingConfig = field(default_factory=TrainingConfig)

  node_cap: int = 8192      # tree states per lane; the caller's GameBatch needs sim_cap >= n_sims
  n_sims: int = 128
  c_puct: float = 1.5
  n_lanes: int = 1 # for caller GameBatch
  batch_size: int = 128 # for inference

  fpu: float = 0.4
  gumbel_m: int = 16
  c_visit: float = 50.0
  c_scale: float = 1.0
  proven_override: bool = False
  forcing: bool = False     # every check / capture / promotion is a root candidate (night-3 skel_root)
  forcing_cap: int = 32

  seed: int = 42
  device: str = "cpu"
  autocast: bool = False
  cache_log2: int = 16      # eval cache slots per search = 2 ** cache_log2 (1 KB each); 0 = off



class Chessbot(ChessbotBase):
  VERSION = "v10_double_buffering"

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

    # the C++ search is bound to one GameBatch (it searches that batch's games), so one per batch
    self._searches = weakref.WeakKeyDictionary()


  def search_for(self, batch: GameBatch) -> GumbelSearch:
    search = self._searches.get(batch)
    if search is None:
      config = self.config
      search = GumbelSearch(
        batch, 
        cache_log2=config.cache_log2,
        c_puct=config.c_puct, fpu=config.fpu, 
        c_visit=config.c_visit, c_scale=config.c_scale,
        proven_override=config.proven_override, 
        forcing=config.forcing, forcing_cap=config.forcing_cap
      )
      self._searches[batch] = search
    return search


  def refresh(self):
    if self.infer_model is not self.model:
      self.infer_model.load_state_dict(self.model.state_dict())
    for search in self._searches.values():
      search.clear_cache()


  @torch.inference_mode()
  def launch(self, 
    batch: GameBatch,lanes: np.ndarray, states: np.ndarray, rows=None
  ):
    board, castling, ep, _, _, moves, counts = batch.encode(lanes, states)
    p, _, wdl, _ = forward_batch(
      self.infer_model, 
      board, castling, ep, 
      moves, counts, 
      self.config.device,
      pad=True, rows=rows
    )
    ps = wdl.float().softmax(1)

    out = torch.cat([p.float().softmax(-1), (ps[:, 0] - ps[:, 2])[:, None]], dim=1)
    if out.device.type != "mps":
      return out, None, len(lanes)

    # double buffering: copy back without blocking and mark the point with an event, so finish() waits for THIS batch
    # only. a blocking .cpu() would also wait for every batch queued after it, and the GPU would sit idle
    host = torch.empty(out.shape, dtype=out.dtype)
    host.copy_(out, non_blocking=True)
    event = torch.mps.event.Event()
    event.record()
    return host, event, len(lanes)


  @staticmethod
  def finish(handle):
    out, event, n = handle
    if event is not None:
      event.synchronize()
    out = out.cpu().numpy()
    return out[:n, :-1], out[:n, -1]
  

  def _evaluate(self, batch: GameBatch, lanes: np.ndarray, states: np.ndarray):
    N = len(lanes)
    priors = np.zeros((N, MAX_MOVES_CAPACITY), np.float32)
    values = np.zeros(N, np.float32)
    size = self.config.batch_size

    for offset in range(0, N, size):
      p, v = self.finish(
        self.launch(
          batch, 
          lanes[offset:offset + size],          
          states[offset:offset + size]
        )
      )

      priors[offset:offset + size, :p.shape[1]] = p
      values[offset:offset + size] = v
    return priors, values


  # separate for playout cap randomization
  def run_search(self, batch: GameBatch, n_sims, m, gumbel_scale, mask=None):
    # the C++ search does not bounds-check: a lane's tree has room for sim_cap states only
    assert n_sims <= batch.sim_cap, "the GameBatch's sim_cap must be >= n_sims"
    self.model.eval()
    search = self.search_for(batch)
    moves, policy, root_value, nodes = search.search(
      lambda lanes, states: self._evaluate(batch, lanes, states),
      self.rng, n_sims=n_sims, m=m,
      gumbel_scale=gumbel_scale, mask=mask
    )
    stats = search.stats()
    return moves, {
      'policy': policy, 
      'value': root_value,
      'nodes': nodes, 
      **stats
    }


  def _select_moves(self, batch: GameBatch, mask: np.ndarray):
    return self.run_search(batch, self.config.n_sims, self.config.gumbel_m, 0.0, mask)



def build(config: Config, state: dict) -> Chessbot:
  model = Model(config.model)
  model.load_state_dict(state["model"])
  model.to(config.device)
  return Chessbot(config, model=model)