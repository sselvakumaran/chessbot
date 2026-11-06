# v11: v10 (C++ Gumbel search, double-buffered self-play) trained with Muon (optimizer.py) and opening noise on by default
# (fast searches sample their move for the first 30 plies). The network and the training targets are the same as v10.
# Search: search/v11_gumbel_cpp (v10's, plus several leaves per lane in flight: max_pending / vl_mode, for single-game
# play); inference: Model.infer (fused kernels) on MPS bf16.

from dataclasses import dataclass, field
import copy
import weakref

import numpy as np
import torch

from bot_base import ChessbotBase
from bots.v11_muon_metal.policy import forward_batch, pack_batch
from engine.game import MAX_MOVES_CAPACITY, GameBatch
from models.v11_muon_metal import Model
from models.v11_muon_metal import Config as ModelConfig
from search.v11_gumbel_cpp import GumbelSearch


@dataclass(frozen=True)
class TrainingConfig:
  max_steps: int = 10000
  lr: float = 1.2e-3          # Muon (AdamW-sized updates) and AdamW alike; lr / weight_decay sets the equilibrium weight norm
  weight_decay: float = 0.4
  warmup_steps: int = 50      # lr ramps up over the first RL steps of a run started from scratch, then stays constant
  decay_length: int = 0       # 0 = constant lr; N = linear decay to ~0 over the last N steps before max_steps
  betas: tuple[float, float] = (0.9, 0.95)
  eps: float = 1e-8
  u_normuon: bool = False     # Muon updates of tall matrices get equal row RMS (stops SwiGLU units dying, README 68)
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
  batch_size: int = 2048
  grad_steps: int = 1        # grad steps per RL step (v11: literal; v10 used round(grad_steps * pcr_p)): 1 x 2048 = reuse 1
  discount: float = 0.995     # target = discount**plies_to_end * onehot(z) + rest draw

  value_weight: float = 1.0 # value loss weight; <1 keeps value's noisy gradient from swamping the trunk (v8.1: 0.25)
  rootq_alpha: float = 0.5 # value target mixing w/ search
  pcr_p: float = 0.25 # playout cap randomization: chance a move gets a full (recorded) search, per lane per move
  fast_sims: int = 16
  fast_m: int = 8
  opening_noise_plies: int = 30 # fast searches also get gumbel noise for the first this-many plies of each game
  autocast: bool = False # bf16 autocast for the grad step's forward/backward; weights, optimizer state and losses stay fp32
  # resignation: the side to move resigns when a full search's root value < -resign_threshold (0 = off), from RL step
  # resign_after on (the early value is uncalibrated); no_resign_frac of games never resign and log how often a resign
  # would have been wrong (resigning side did not lose)
  resign_threshold: float = 0.0
  resign_after: int = 0
  no_resign_frac: float = 0.1


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
  forcing_append: bool = False # forcing moves on top of the top gumbel_m (at most m / 2 extra), not taking its slots
  root_value_policy: bool = False # root value (value target, resign) = improved-policy-weighted completed Q, not the
                                  # mean of every root visit (which counts candidates cut by sequential halving)
  # several leaves per lane in flight, so one game's search fills a network call (single-game play); 1 = v10's search
  max_pending: int = 1
  vl_mode: int = 0          # 0: one leaf per root candidate (same moves as v10); 1: virtual loss; 2: virtual visit
                            # valued mean - virtual_loss (play: max_pending 64, vl_mode 2, virtual_loss 0.2, sync)
  virtual_loss: float = 1.0
  sync_phases: bool = False # vl_mode 1 / 2: nothing past a halving step while leaves are in flight
  fused: bool = True        # inference through Model.infer (fused kernels) where it can run, else forward_batch

  seed: int = 42
  device: str = "cpu"
  autocast: bool = False
  cache_log2: int = 16      # eval cache slots per search = 2 ** cache_log2 (1 KB each); 0 = off



class Chessbot(ChessbotBase):
  VERSION = "v11_muon_metal"

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
        forcing=config.forcing, forcing_cap=config.forcing_cap,
        max_pending=config.max_pending, vl_mode=config.vl_mode,
        virtual_loss=config.virtual_loss, sync_phases=config.sync_phases,
        forcing_append=config.forcing_append, root_value_policy=config.root_value_policy
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
    if self.config.fused and self.infer_model.can_infer():
      packed, _, M = pack_batch(board, castling, ep, moves, counts, self.config.device, pad=True, rows=rows)
      out = self.infer_model.infer(packed, M)       # (rows, M + 1): probabilities, then value
    else:
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