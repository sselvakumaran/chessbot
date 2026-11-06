import math
import numpy as np
from engine.game import MAX_MOVES_CAPACITY
from search.v6_mcts_lazy import MCTS, ONGOING, CHECKMATE, Evaluate

def _sequential_halving_schedule(m, n_sims) -> list: 
  # produces seq[idx] -> sim idx must have seq[idx]
  if m <= 1:
    return list(range(n_sims))
  seq, visits = [], [0 for _ in range(m)]
  num_phases = math.ceil(math.log2(m))
  num_candidates = m

  while len(seq) < n_sims:
    for _ in range(max(1, int(n_sims / (num_phases * num_candidates)))):
      seq.extend(visits[:num_candidates])
      for i in range(num_candidates):
        visits[i] += 1
    num_candidates = max(2, num_candidates // 2)
  return seq[:n_sims]

class GumbelMCTS(MCTS):
  def __init__(
    self,
    evaluate: Evaluate,
    lane_count: int = 16,
    max_nodes: int = 2048,
    max_sims: int = 1024,
    c_puct=1.5,
    fpu=0.4,
    c_visit=50.0,
    c_scale=1.0,
    proven_override=False
  ):
    super().__init__(evaluate, lane_count, max_nodes, max_sims, use_fpu=True)
    self.c_puct, self.fpu, self.c_visit, self.c_scale = c_puct, fpu, c_visit, c_scale
    self.proven_override = proven_override
    self._tables = dict()

  def _table(self, m, n_sims):
    key = (m, n_sims)
    if key not in self._tables:
      self._tables[key] = np.array(
        [
          _sequential_halving_schedule(max(k, 1), n_sims) 
         for k in range(m + 1)
        ], np.int32
      )
    return self._tables[key]

  def _sigma(self, root_v, N, Q, prior):
    visited = N > 0
    sum_n = N.sum(1)
    p_vis = (prior * visited).sum(1)
    pq_vis = (prior * Q * visited).sum(1)
    v_mix = (
      root_v 
      + np.where(p_vis > 0, sum_n / np.maximum(p_vis, 1e-9) * pq_vis, 0)
    ) / (1 + sum_n)
    completed = np.where(visited, Q, v_mix[:, None])
    return (self.c_visit + N.max(1, keepdims=True)) * self.c_scale * (completed + 1) / 2

  def run(
    self,
    batch,
    rng: np.random.Generator,
    n_sims: int = 96,
    m: int = 16,
    gumbel_scale: float = 1.0,
    mask = None
  ):
    self.batch = batch
    self._reset(n_sims)
    B = len(self.batch)

    active = self.result[:, 0] == ONGOING
    if mask is not None:
      active &= mask
    lanes = np.flatnonzero(active).astype(np.int32)

    chosen = np.full(B, -1, np.int32)
    policy = np.zeros((B, MAX_MOVES_CAPACITY), np.float32)
    root_value = np.zeros(B, np.float32)

    L = len(lanes)
    if not L:
      return chosen, policy, root_value

    root_values = self._expand(lanes, np.zeros(L, np.int32))
    self.visits[lanes, 0] = 1
    self.value_sum[lanes, 0] = root_values

    n_moves = self.n_actions[lanes, 0]
    K = int(n_moves.max()) # max num of moves in batch
    root_rows = self.to_children_idx[lanes, 0]
    root_prior = self.edge_prior[lanes, root_rows, :K]

    legal = np.arange(K)[None] < n_moves[:, None]
    logits = np.where(
      legal,
      np.log(np.maximum(root_prior, 1e-9)),
      -np.inf
    )
    gumbel = (rng.gumbel(size=(L,K)) * gumbel_scale).astype(np.float32)
    n_considered = np.minimum(n_moves, m)
    schedule = self._table(m, n_sims)

    root_N = np.zeros((L, K), np.int32)
    root_Q = np.zeros((L, K), np.float32)
    rows = np.arange(L)

    for sim in range(n_sims):
      allowed = root_N == schedule[n_considered, sim][:, None]
      score = gumbel + logits + self._sigma(root_values, root_N, root_Q, root_prior)
      root_actions = np.where(allowed, score, -np.inf).argmax(axis=1).astype(np.int32)

      paths, depths, edge_parent, edge_action = self._select_below_root(
        lanes, root_actions, n_sims
      )
      self._add_children(lanes, paths, depths, edge_parent, edge_action)
      values = self._leaf_values(lanes, paths, depths)
      self._backup_all(lanes, paths, depths, values)

      children = self.children[lanes, root_rows, root_actions]
      root_N[rows, root_actions] = self.visits[lanes, children]
      root_Q[rows, root_actions] = self.value_sum[lanes, children] / self.visits[lanes, children]

    root_value[lanes] = self.value_sum[lanes, 0] / self.visits[lanes, 0]

    sigma = self._sigma(root_values, root_N, root_Q, root_prior)
    improved = np.exp(logits + sigma - (logits + sigma).max(axis=1, keepdims=True))
    policy[lanes, :K] = improved / improved.sum(axis=1, keepdims=True)

    # play most visited candidate + score
    most_visited = root_N == root_N.max(axis=1, keepdims=True)
    chosen[lanes] = np.where(most_visited, gumbel+logits+sigma, -np.inf).argmax(axis=1)

    if self.proven_override:
      self._override_mates(lanes, root_rows, K, chosen, policy, root_value)
    return chosen, policy, root_value


  def _select_below_root(self, lanes, root_actions, n_sims):
    # force root action, use PUCT w/ fpu
    L = len(lanes)
    paths = np.full((L, n_sims + 2), -1, np.int32)
    paths[:, 0] = 0
    depths = np.zeros(L, np.int32)
    edge_parent = np.full(L, -1, np.int32)
    edge_action = np.full(L, -1, np.int32)

    first = self.children[lanes, self.to_children_idx[lanes, 0], root_actions]
    new = first < 0  # roots have no children
    edge_parent[new] = 0
    edge_action[new] = root_actions[new]

    alive = ~new
    nodes = np.where(new, 0, first)
    depths[alive] = 1
    paths[alive, 1] = first[alive]

    while True:
      a = np.flatnonzero(alive) # active
      if not len(a):
        break

      a_lanes, a_nodes = lanes[a], nodes[a]
      stop = (
        (self.result[a_lanes, a_nodes] != ONGOING)
        | ~self.expanded[a_lanes, a_nodes]
      )
      alive[a[stop]] = False
      a, a_lanes, a_nodes = a[~stop], a_lanes[~stop], a_nodes[~stop]
      if not len(a):
        break

      n_actions = self.n_actions[a_lanes, a_nodes]
      K = int(n_actions.max())
      row = self.to_children_idx[a_lanes, a_nodes]
      child = self.children[a_lanes, row, :K]
      valid = np.arange(K)[None, :] < n_actions[:, None]
      in_tree_mask = child >= 0
      in_tree = np.where(in_tree_mask, child, 0)
      a_lanes_col = a_lanes[:, None]
      visits = np.where(in_tree_mask, self.visits[a_lanes_col, in_tree], 0)
      value_sum = np.where(in_tree_mask, self.value_sum[a_lanes_col, in_tree], np.float32(0))
      q = np.divide(
        value_sum,
        visits,
        out=np.zeros_like(value_sum),
        where=visits > 0
      )

      # unvisited child: q = q(node) - fpu * sqrt(prior mass already visited)
      prior = self.edge_prior[a_lanes, row, :K]
      parent_visits = np.maximum(1, self.visits[a_lanes, a_nodes])
      node_q = -self.value_sum[a_lanes, a_nodes] / parent_visits # flip
      fpu_q = node_q - self.fpu * np.sqrt(np.where(visits > 0, prior, 0).sum(1))
      q = np.where(visits > 0, q, fpu_q[:, None]).astype(np.float32)

      u = (
        self.c_puct * prior
        * np.sqrt(parent_visits)[:, None] / (1 + visits)
      )
      score = np.where(valid, q + u, -np.inf)
      action = score.argmax(axis=1)
      picked = child[np.arange(len(a)), action]

      new = picked < 0
      edge_parent[a[new]] = a_nodes[new]
      edge_action[a[new]] = action[new]

      alive[a[new]] = False
      idx = a[~new]
      nodes[idx] = picked[~new]
      depths[idx] += 1
      paths[idx, depths[idx]] = picked[~new]
    return paths, depths, edge_parent, edge_action


  def _add_children(self, lanes, paths, depths, edge_parent, edge_action):
    has_edge = edge_action >= 0
    if not has_edge.any():
      return

    B = len(self.batch)
    active_lanes = lanes[has_edge]
    parents = np.zeros(B, np.int32)
    actions = np.full(B, -1, np.int32)
    parents[active_lanes] = edge_parent[has_edge]
    actions[active_lanes] = edge_action[has_edge]
    child_states, counts, results = self.batch.get_children(parents, actions)

    child = child_states[active_lanes]
    rows = self.to_children_idx[active_lanes, edge_parent[has_edge]]
    self.children[active_lanes, rows, edge_action[has_edge]] = child
    self.n_actions[active_lanes, child] = counts[active_lanes]
    self.result[active_lanes, child] = results[active_lanes]
    self.node_count[active_lanes] += 1
    depths[has_edge] += 1
    paths[has_edge, depths[has_edge]] = child


  def _leaf_values(self, lanes, paths, depths):
    leaves = paths[np.arange(len(lanes)), depths]
    results = self.result[lanes, leaves]
    values = np.where(results == CHECKMATE, -1.0, 0.0).astype(np.float32)
    ongoing = results == ONGOING
    if ongoing.any():
      values[ongoing] = self._expand(lanes[ongoing], leaves[ongoing])
    return values


  def _override_mates(self, lanes, root_rows, K, chosen, policy, root_value):
    children = self.children[lanes[:, None], root_rows[:, None], np.arange(K)[None]]
    in_tree = children >= 0
    mates = in_tree & (self.result[lanes[:, None], np.where(in_tree, children, 0)] == CHECKMATE)
    has_mate = mates.any(axis=1)
    if not has_mate.any():
      return

    mate_lanes = lanes[has_mate]
    target = mates[has_mate].astype(np.float32)
    policy[mate_lanes] = 0.0
    policy[mate_lanes, :K] = target / target.sum(axis=1, keepdims=True)
    chosen[mate_lanes] = mates[has_mate].argmax(axis=1)
    root_value[mate_lanes] = 1.0
