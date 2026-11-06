from collections.abc import Callable
import numpy as np
from engine.game import GameBatch, GameResult, MAX_MOVES_CAPACITY

# lazy add children
# vectorized _select
# 


# in: batch, lane indices, state IDs
# out: priors, values
Evaluate = Callable[
  [GameBatch, np.ndarray, np.ndarray],
  tuple[np.ndarray, np.ndarray]
]

ONGOING = int(GameResult.ONGOING)
CHECKMATE = int(GameResult.CHECKMATE)

class MCTS:
  def __init__(
    self,
    evaluate: Evaluate,
    lane_count: int = 16,
    max_nodes: int = 2048,
    max_sims: int = 1024,
    use_fpu: bool = False
  ):
    self.B = lane_count
    self.evaluate = evaluate
    self.max_sims = max_sims
    self.max_nodes = max_nodes
    self.use_fpu = use_fpu

    assert max_nodes >= max_sims + 2
    B, R = lane_count, max_sims + 1

    self.visits = np.zeros((B, max_nodes), dtype=np.int32) # number of visits
    self.network_value = np.zeros((B, max_nodes), dtype=np.float32) 
    self.value_sum = np.zeros((B, max_nodes), dtype=np.float32)

    self.expanded = np.zeros((B, max_nodes), dtype=bool)
    self.n_actions = np.zeros((B, max_nodes), np.int32)
    self.result = np.zeros((B, max_nodes), dtype=np.int8)
    self.node_count = np.ones(B, dtype=np.int32)
    
    # keep separate child array for expanded parents
    self.to_children_idx = np.full((B, max_nodes), -1, np.int32)
    self.children = np.full((B, max_sims + 1, MAX_MOVES_CAPACITY), -1, np.int32)
    self.edge_prior = np.zeros((B, R, MAX_MOVES_CAPACITY), np.float32)
    self.row_count = np.zeros(B, np.int32)


  def run(
    self,
    batch: GameBatch,
    rng: np.random.Generator,
    n_sims: int = 128,
    c_puct: float = 1.5,
    temperature: float = 0.0,
    mask = None,
    noise_alpha: float = 0.3,
    noise_epsilon: float = 0.0,
    fpu: float = 0, 
    fpu_root: float = 0,
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

    root_values = self._expand(lanes, np.zeros(len(lanes), np.int32))
    self.visits[lanes, 0] = 1
    self.value_sum[lanes, 0] = root_values
    if noise_epsilon > 0:
      self._add_root_noise(lanes, rng, noise_alpha, noise_epsilon)

    parents = np.zeros(B, np.int32)
    actions = np.full(B, -1, np.int32)
    for _ in range(n_sims):
      path, depth, edge_parent, edge_action = self._select_all(lanes, c_puct, n_sims, fpu, fpu_root)

      has_edge = edge_action >= 0

      if has_edge.any():
        active_lanes = lanes[has_edge]
        actions.fill(-1)
        parents[active_lanes] = edge_parent[has_edge]
        actions[active_lanes] = edge_action[has_edge]
        child_states, counts, results = self.batch.get_children(parents, actions)
        child = child_states[active_lanes]
        assert (child >= 0).all() and (child < self.max_nodes).all()
        rows = self.to_children_idx[active_lanes, edge_parent[has_edge]]
        self.children[active_lanes, rows, edge_action[has_edge]] = child
        self.n_actions[active_lanes, child] = counts[active_lanes]
        self.result[active_lanes, child] = results[active_lanes]
        self.node_count[active_lanes] += 1
        depth[has_edge] += 1
        path[has_edge, depth[has_edge]] = child

      leaves = path[np.arange(L), depth]
      results = self.result[lanes, leaves]
      values = np.where(results == CHECKMATE, -1.0, 0.0).astype(np.float32)

      ongoing = results == ONGOING
      if ongoing.any():
        values[ongoing] = self._expand(lanes[ongoing], leaves[ongoing])
      self._backup_all(lanes, path, depth, values)

    # root Q after search, from the root mover's perspective (root value_sum is
    # seeded with the network's own estimate and accumulates backups as-is)
    root_value[lanes] = self.value_sum[lanes, 0] / self.visits[lanes, 0]

    for lane in lanes:
      visits = self._child_visits(lane, 0).astype(np.float32)
      count = self.n_actions[lane, 0]
      policy[lane, :count] = visits / visits.sum()

      if temperature == 0:
        chosen[lane] = int(visits.argmax())
      else:
        visited = np.flatnonzero(visits)
        weights = np.log(visits[visited])
        weights = np.exp((weights - weights.max()) / temperature)
        chosen[lane] = int(rng.choice(visited, p=weights / weights.sum()))
    return chosen, policy, root_value
  

  def _reset(self, n_sims):
    assert 1 <= n_sims <= self.max_sims

    self.batch.reset_trees()

    self.visits.fill(0)
    self.network_value.fill(0)
    self.value_sum.fill(0)

    self.expanded.fill(0)
    self.n_actions.fill(0)
    self.result.fill(0)

    self.node_count.fill(1)
    self.row_count.fill(0)

    self.to_children_idx.fill(-1)

    self.children[:, :n_sims + 1].fill(-1)
    self.edge_prior[:, :n_sims + 1].fill(0)
    self.n_actions[:, 0] = self.batch.num_moves_all()
    self.result[:, 0] = self.batch.results()


  def _expand(self, lanes, states):
    priors, values = self.evaluate(self.batch, lanes, states)
    rows = self.row_count[lanes].copy()
    self.row_count[lanes] += 1
    self.to_children_idx[lanes, states] = rows
    self.edge_prior[lanes, rows] = priors
    self.network_value[lanes, states] = values
    self.expanded[lanes, states] = True
    return values


  def _child_visits(self, lane, node):
    row = self.to_children_idx[lane, node]
    child = self.children[lane, row, :self.n_actions[lane, node]]
    known = child >= 0
    return np.where(known, self.visits[lane, np.where(known, child, 0)], 0)


  def _add_root_noise(self, lanes, rng, alpha, epsilon):
    # dirichlet noise mixed into the root's edge
    for lane in lanes:
      count = self.n_actions[lane, 0]
      row = self.to_children_idx[lane, 0]
      noise = rng.dirichlet(np.full(count, alpha, np.float64))
      self.edge_prior[lane, row, :count] = (
        (1 - epsilon) * self.edge_prior[lane, row, :count] + epsilon * noise
      ).astype(np.float32)


  def _select_all(self, lanes, c_puct, n_sims, fpu=0, fpu_root=0):
    L = len(lanes)
    nodes = np.zeros(L, np.int32)
    depths = np.zeros(L, np.int32)
    paths = np.full((L, n_sims + 2), -1, np.int32)
    paths[:, 0] = 0
    edge_parent = np.full(L, -1, np.int32)
    edge_action = np.full(L, -1, np.int32)
    alive = np.ones(L, bool)

    while True:
      a = np.flatnonzero(alive)
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

      prior = self.edge_prior[a_lanes, row, :K]
      parent_visits = np.maximum(1, self.visits[a_lanes, a_nodes])
      if self.use_fpu:
        # unvisited child gets q = q(parent) - reduction * sqrt(prior)
        parent_q = self.value_sum[a_lanes, a_nodes] / parent_visits
        parent_q = np.where(a_nodes == 0, parent_q, -parent_q)
        reduction = np.where(a_nodes == 0, fpu_root, fpu)
        fpu_q = parent_q - reduction * np.sqrt(np.where(visits > 0, prior, 0).sum(1))
        q = np.where(visits > 0, q, fpu_q[:, None]).astype(np.float32)

      u = (
        c_puct * prior
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


  def _backup_all(self, lanes, paths, depths, values):
    for d in range(int(depths.max()), 0, -1):
      m = depths >= d
      a_lanes, a_nodes = lanes[m], paths[m, d]
      flips = depths[m] - d + 1
      v = np.where(flips % 2 == 0, values[m], -values[m])
      self.visits[a_lanes, a_nodes] += 1
      self.value_sum[a_lanes, a_nodes] += v
      
    # root
    self.visits[lanes, 0] += 1
    self.value_sum[lanes, 0] += np.where(depths % 2 == 0, values, -values)