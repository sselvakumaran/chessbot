import math
import numpy as np
from engine.game import GameBatch, GameResult, MAX_MOVES_CAPACITY
from helpers.evaluatorv1 import Evaluator, Encoding

# handle all the stuff within the simulation / expansion phase.
# (evaluator is inputted — it owns the model, chunking and device)
# gumbel sampling + selective halvening
#
# stats are dense numpy keyed [lane, state, action] (states/actions are the
# C++ arena's LOCAL indices; root is state 0). every lane descends in
# lockstep: one gamebatch.get_children call per depth level, one evaluator
# call per wave of new leaves.

ONGOING = int(GameResult.ONGOING)
CHECKMATE = int(GameResult.CHECKMATE)

def np_softmax(x: np.ndarray, axis=-1):
  x_max = np.max(x, axis=axis, keepdims=True)
  exp_x = np.exp(x - x_max)
  return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

class SearchTree:
  def __init__(
    self,
    gamebatch: GameBatch,
    c_visit: float = 50.0, # sigma: how fast visit counts grow Q's weight
    c_scale: float = 1.0,  # sigma: overall trust in Q vs the prior
  ):
    self.gamebatch = gamebatch
    self.c_visit, self.c_scale = c_visit, c_scale

    gamebatch_size = gamebatch.gamebatch_size
    node_cap = gamebatch.states_cap
    action_cap = MAX_MOVES_CAPACITY

    self.eval_prior = np.zeros( # cache of model's policy logits
      (gamebatch_size, node_cap, action_cap), dtype=np.float32)
    self.eval_v = np.zeros( # cache of model's V
      (gamebatch_size, node_cap), dtype=np.float32)
    self.N = np.zeros( # visit count per child / move
      (gamebatch_size, node_cap, action_cap), dtype=np.float32)
    self.W = np.zeros( # cumulative backup value
      (gamebatch_size, node_cap, action_cap), dtype=np.float32)
    self.n_actions = np.zeros( # legal moves per materialized state
      (gamebatch_size, node_cap), dtype=np.int32)
    self.result = np.zeros( # GameResult per materialized state
      (gamebatch_size, node_cap), dtype=np.int8)
    self.expanded = np.zeros( # True once eval_prior/eval_v are filled
      (gamebatch_size, node_cap), dtype=bool)

    self._lanes = np.arange(gamebatch_size)
    self._action_arange = np.arange(action_cap)

  def reset_tree(self):
    self.eval_prior.fill(0)
    self.eval_v.fill(0)
    self.N.fill(0)
    self.W.fill(0)
    self.n_actions.fill(0)
    self.result.fill(0)
    self.expanded.fill(False)

  # ---- gumbel machinery, batched over lanes at per-lane states (B,) ----
  def sigma(self, q, max_N):
    return (self.c_visit + max_N) * self.c_scale * q

  def _valid(self, states):
    return self._action_arange[None, :] < self.n_actions[self._lanes, states][:, None]

  def v_mix(self, states):
    N = self.N[self._lanes, states]
    W = self.W[self._lanes, states]
    prior = self.eval_prior[self._lanes, states]
    value = self.eval_v[self._lanes, states]

    visited = N > 0
    policy = np_softmax(np.where(self._valid(states), prior, -1e9))
    q = np.where(visited, W / np.maximum(N, 1.0), 0.0)
    policy_visited_sum = (policy * visited).sum(axis=1)
    policy_mean_value = (policy * q * visited).sum(axis=1) \
      / np.maximum(policy_visited_sum, 1e-9)
    node_N = N.sum(axis=1)
    return np.where(
      visited.any(axis=1),
      (value + node_N * policy_mean_value) / (1 + node_N),
      value,
    )

  def completed_Q(self, states):
    N = self.N[self._lanes, states]
    W = self.W[self._lanes, states]
    return np.where(
      N > 0, W / np.maximum(N, 1.0), self.v_mix(states)[:, None]
    )

  def improved_policy(self, states):
    N = self.N[self._lanes, states]
    prior = self.eval_prior[self._lanes, states]
    scored = prior + self.sigma(self.completed_Q(states), N.max(axis=1, keepdims=True))
    return np_softmax(np.where(self._valid(states), scored, -1e9))

  def select(self, states):
    N = self.N[self._lanes, states]
    node_N = N.sum(axis=1, keepdims=True)
    score = self.improved_policy(states) - N / (1 + node_N)
    return np.where(self._valid(states), score, -np.inf).argmax(axis=1).astype(np.int32)

  # ---- one wave: every lane runs one simulation in lockstep ----
  def _run_wave(self, evaluator: Evaluator, first_action: np.ndarray):
    gamebatch = self.gamebatch
    lanes = self._lanes
    node_cap = gamebatch.states_cap

    current = np.zeros(len(lanes), np.int32) # every lane starts at its root
    action = first_action
    alive = action >= 0
    participating = alive.copy()
    leaf = np.zeros(len(lanes), np.int32)
    leaf_depth = np.zeros(len(lanes), np.int32)
    path_state = np.full((len(lanes), node_cap), -1, np.int32)
    path_action = np.full((len(lanes), node_cap), -1, np.int32)

    depth = 0
    while alive.any():
      masked_action = np.where(alive, action, -1)
      path_state[:, depth] = np.where(alive, current, -1)
      path_action[:, depth] = masked_action
      child, child_n, child_result = gamebatch.get_children(current, masked_action)

      self.n_actions[lanes[alive], child[alive]] = child_n[alive]
      self.result[lanes[alive], child[alive]] = child_result[alive]

      # a lane stops on a terminal or never-evaluated child: that's its leaf
      stop = alive & (
        (child_result != ONGOING) | ~self.expanded[lanes, np.maximum(child, 0)]
      )
      leaf[stop] = child[stop]
      leaf_depth[stop] = depth + 1
      alive &= ~stop
      current = np.where(alive, child, current)
      action = self.select(current)
      depth += 1

    # evaluate the new (non-terminal) leaves in one batch
    need = participating \
      & (self.result[lanes, leaf] == ONGOING) & ~self.expanded[lanes, leaf]
    if need.any():
      need_lanes = np.where(need)[0].astype(np.int32)
      need_states = leaf[need]
      p_logits, values = evaluator.eval_states(gamebatch, need_lanes, need_states)
      self.eval_prior[need_lanes, need_states] = p_logits
      self.eval_v[need_lanes, need_states] = values
      self.expanded[need_lanes, need_states] = True

    # backup: leaf value from the leaf player's perspective, sign flips per ply
    leaf_result = self.result[lanes, leaf]
    v = np.where(leaf_result == CHECKMATE, -1.0,
        np.where(leaf_result != ONGOING, 0.0, self.eval_v[lanes, leaf]))
    for depth in range(int(leaf_depth.max()) - 1, -1, -1):
      on_path = participating & (depth < leaf_depth)
      v = np.where(on_path, -v, v)
      self.N[lanes[on_path], path_state[on_path, depth], path_action[on_path, depth]] += 1
      self.W[lanes[on_path], path_state[on_path, depth], path_action[on_path, depth]] += v[on_path]

  def run_search(self,
    evaluator: Evaluator,
    rng: np.random.Generator,
    k_top_actions: int = 8,
    n_sims: int = 64, # total simulation budget across all halving phases
    active: np.ndarray | None = None, # bool (B,); None = all ongoing games
  ): # -> root Encoding, improved policy (B, action_cap), chosen moves (B,)
    gamebatch = self.gamebatch
    gamebatch_size = gamebatch.gamebatch_size
    assert n_sims <= gamebatch.sim_cap, "n_sims exceeds the C++ arena"
    if active is None:
      active = np.asarray(gamebatch.active)
    self.reset_tree()
    gamebatch.reset_trees() # C++ arenas only otherwise reset on play/reset

    # snapshot roots BEFORE leaf evals reuse the encode buffers
    root_encoding = Encoding(*[a.copy() for a in gamebatch.get_encoding()])
    root_p_logits, root_values = evaluator.eval_logits(root_encoding)
    roots = np.zeros(gamebatch_size, np.int32)
    self.eval_prior[:, 0] = root_p_logits
    self.eval_v[:, 0] = root_values
    self.n_actions[:, 0] = root_encoding.num_moves
    self.expanded[:, 0] = True

    # gumbel noise selection
    gumbel = rng.gumbel(size=root_p_logits.shape).astype(np.float32)
    root_score = np.where(self._valid(roots), gumbel + root_p_logits, -np.inf)
    order = np.argsort(-root_score, axis=1)
    candidates = [
      list(order[i, :max(1, min(k_top_actions, int(root_encoding.num_moves[i])))])
      if active[i] else []
      for i in range(gamebatch_size)
    ]

    # sequential halving
    num_phases = max(1, math.ceil(math.log2(k_top_actions)))
    for phase in range(num_phases):
      # schedule each surviving candidate's sims (each item = an initial action)
      to_eval = [[] for _ in range(gamebatch_size)]
      for i in range(gamebatch_size):
        if not active[i] or len(candidates[i]) <= 1: continue
        sims_per_cand = max(1, n_sims // (num_phases * len(candidates[i])))
        to_eval[i] = [a for a in candidates[i] for _ in range(sims_per_cand)]

      for wave in range(max((len(t) for t in to_eval), default=0)):
        first_action = np.array([
          t[wave] if wave < len(t) else -1 for t in to_eval
        ], np.int32)
        self._run_wave(evaluator, first_action)

      # eliminate bottom half by g + prior + sigma(Q)
      root_max_N = self.N[:, 0].max(axis=1)
      root_v_mix = self.v_mix(roots)
      for i in range(gamebatch_size):
        if not active[i] or len(candidates[i]) <= 1: continue
        def score(a):
          q = self.W[i, 0, a] / self.N[i, 0, a] \
            if self.N[i, 0, a] > 0 else root_v_mix[i]
          return gumbel[i, a] + root_p_logits[i, a] + self.sigma(q, root_max_N[i])
        candidates[i] = sorted(
          candidates[i], key=score, reverse=True
        )[:max(1, len(candidates[i]) // 2)]

    policy = np.where(active[:, None], self.improved_policy(roots), 0.0) \
      .astype(np.float32)
    moves = np.array([
      int(candidates[i][0]) if candidates[i] else 0
      for i in range(gamebatch_size)
    ], np.int32)
    return root_encoding, policy, moves
