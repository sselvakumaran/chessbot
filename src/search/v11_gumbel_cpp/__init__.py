import ctypes
import weakref
import numpy as np
from engine.game import GameBatch, MAX_MOVES_CAPACITY, _ptr, _i32
from engine.native import lib

# v11: the v10 wrapper (search.v10_gumbel_cpp) for search.cpp's v11 search: several leaves per lane in flight
# (max_pending, vl_mode, virtual_loss, sync_phases; see search.cpp). Defaults = the v10 search.
# async: begin -> run / cache_evals -> take_finished + results
# sync: search()

P, I32, F64 = ctypes.c_void_p, ctypes.c_int, ctypes.c_double
lib.v11_gumbel_cpp_new.restype = P
lib.v11_gumbel_cpp_new.argtypes = [P, I32, I32, I32]
lib.v11_gumbel_cpp_free.argtypes = [P]
lib.v11_gumbel_cpp_set_params.argtypes = [P, F64, F64, F64, F64, I32, I32, I32, I32, F64, I32, I32, I32]
lib.v11_gumbel_cpp_begin.argtypes = [P, P, I32, P, P, P]
lib.v11_gumbel_cpp_run.restype = I32
lib.v11_gumbel_cpp_run.argtypes = [P, I32, P, P]
lib.v11_gumbel_cpp_cache_evals.argtypes = [P, P, P, I32, P, P]
lib.v11_gumbel_cpp_take_finished.restype = I32
lib.v11_gumbel_cpp_take_finished.argtypes = [P, P]
lib.v11_gumbel_cpp_results.argtypes = [P, P, I32, P, P, P, P]
lib.v11_gumbel_cpp_root_flags.argtypes = [P, P, I32, P]
lib.v11_gumbel_cpp_clear_cache.argtypes = [P]
lib.v11_gumbel_cpp_stats.argtypes = [P, P]

class GumbelSearch:
  def __init__(
    self,
    batch: GameBatch,
    cache_log2: int = 19, # 2 ^ cache_log2 cache slots
    c_puct: float = 1.5,
    fpu: float = 0.4,
    c_visit: float = 50.0,
    c_scale: float = 1.0,
    proven_override: bool = False,
    forcing: bool = False,
    forcing_cap: int = 32,
    max_pending: int = 1, # leaves in flight per lane
    vl_mode: int = 0, # 0 one per root candidate, 1 virtual loss, 2 virtual visit at mean - virtual_loss
    virtual_loss: float = 1.0,
    sync_phases: bool = False, # vl_mode 1 / 2: drain before each halving step
    forcing_append: bool = False, # forcing moves on top of the top m (at most m / 2 extra) instead of in its slots
    root_value_policy: bool = False # root value = improved-policy-weighted completed Q instead of the visit mean
  ):
    # weak: the bot caches one search per batch in a WeakKeyDictionary keyed by the batch; a strong reference here
    # would keep every batch (and its search) alive forever
    self._batch = weakref.ref(batch)
    self.B = len(batch)
    self.max_pending = max(int(max_pending), 1)
    self._search = lib.v11_gumbel_cpp_new(batch._gamebatch, batch.sim_cap, cache_log2, self.max_pending)
    lib.v11_gumbel_cpp_set_params(
      self._search,
      c_puct, fpu, c_visit, c_scale,
      int(proven_override), int(forcing), forcing_cap,
      int(vl_mode), float(virtual_loss), int(sync_phases),
      int(forcing_append), int(root_value_policy)
    )

    # one request per leaf in flight: up to B * max_pending per run
    self._req_lanes = np.zeros(self.B * self.max_pending, np.int32)
    self._req_states = np.zeros(self.B * self.max_pending, np.int32)
    self._finished = np.zeros(self.B, np.int32)

  @property
  def batch(self) -> GameBatch:
    return self._batch()

  def __del__(self):
    if getattr(self, "_search", None):
      lib.v11_gumbel_cpp_free(self._search)
      self._search = None

  def begin(self, lanes, n_sims, m, gumbel):
    lanes = _i32(lanes)
    n = len(lanes)
    rows = np.zeros((n, MAX_MOVES_CAPACITY), np.float32)
    rows[:, :gumbel.shape[1]] = gumbel
    n_sims = _i32(np.broadcast_to(n_sims, n))
    m = _i32(np.broadcast_to(m, n))
    lib.v11_gumbel_cpp_begin(
      self._search, _ptr(lanes), n, _ptr(n_sims), _ptr(m), _ptr(rows)
    )

  def run(self, max_requests: int):
    n = lib.v11_gumbel_cpp_run(
      self._search, min(max_requests, len(self._req_lanes)), _ptr(self._req_lanes), _ptr(self._req_states)
    )
    return self._req_lanes[:n].copy(), self._req_states[:n].copy()

  def cache_evals(self, lanes, states, priors, values):
    lanes, states = _i32(lanes), _i32(states)
    rows = np.zeros((len(lanes), MAX_MOVES_CAPACITY), np.float32)
    rows[:, :priors.shape[1]] = priors
    values = np.ascontiguousarray(values, np.float32)
    lib.v11_gumbel_cpp_cache_evals(
      self._search, _ptr(lanes), _ptr(states), len(lanes), _ptr(rows), _ptr(values)
    )

  def take_finished(self):
    n = lib.v11_gumbel_cpp_take_finished(self._search, _ptr(self._finished))
    return self._finished[:n].copy()

  def results(self, lanes):
    lanes = _i32(lanes)
    n = len(lanes)
    chosen = np.zeros(n, np.int32)
    root_value = np.zeros(n, np.float32)
    nodes = np.zeros(n, np.int32)
    policy = np.zeros((n, MAX_MOVES_CAPACITY), np.float32)
    lib.v11_gumbel_cpp_results(
      self._search, _ptr(lanes), n, _ptr(chosen), _ptr(root_value), _ptr(policy), _ptr(nodes)
    )
    return chosen, policy, root_value, nodes

  def root_flags(self, lanes):
    lanes = _i32(lanes)
    out = np.zeros((len(lanes), MAX_MOVES_CAPACITY), np.uint8)
    lib.v11_gumbel_cpp_root_flags(self._search, _ptr(lanes), len(lanes), _ptr(out))
    return out

  def clear_cache(self):
    lib.v11_gumbel_cpp_clear_cache(self._search)

  def stats(self):
    out = np.zeros(3, np.int64)
    lib.v11_gumbel_cpp_stats(self._search, _ptr(out))
    return {"cache_hits": int(out[0]), "cache_misses": int(out[1]), "collisions": int(out[2])}

  def search(self, 
    evaluate, 
    rng: np.random.Generator, 
    n_sims=96, m=16, gumbel_scale=1.0, mask=None
  ):
    B = self.B
    active = self.batch.results() == 0
    if mask is not None:
      active &= np.asarray(mask, bool)
    lanes = np.flatnonzero(active).astype(np.int32)
    chosen = np.full(B, -1, np.int32)
    policy = np.zeros((B, MAX_MOVES_CAPACITY), np.float32)
    root_value = np.zeros(B, np.float32)
    nodes = np.zeros(B, np.int32)
    if not len(lanes):
      return chosen, policy, root_value, nodes

    # same random stream as GumbelMCTS.run: one (L, K) draw, K = most legal moves among the active roots
    K = int(self.batch.num_moves_all()[lanes].max())
    gumbel = (rng.gumbel(size=(len(lanes), K)) * gumbel_scale).astype(np.float32)
    self.begin(lanes, n_sims, m, gumbel)
    while True:
      req_lanes, req_states = self.run(len(self._req_lanes))
      if not len(req_lanes):
        break
      priors, values = evaluate(req_lanes, req_states)
      self.cache_evals(req_lanes, req_states, priors, values)
    done = self.take_finished()
    assert len(done) == len(lanes), (len(done), len(lanes))
    c, p, v, n = self.results(lanes)
    chosen[lanes], policy[lanes], root_value[lanes], nodes[lanes] = c, p, v, n
    return chosen, policy, root_value, nodes
