# monte carlo tree search
# + puct (continuous upper confidence bounds)
# + tree caching (transposition table)


from collections.abc import Callable
import numpy as np
from engine.game import GameBatch, GameResult, MAX_MOVES_CAPACITY


# in: batch, lane indices, state IDs
# out: priors, values
Evaluate = Callable[
  [GameBatch, np.ndarray, np.ndarray],
  tuple[np.ndarray, np.ndarray]
]


class MCTS:
  def __init__(
    self,
    evaluate: Evaluate,
    lane_count: int = 16,
    max_nodes: int = 2048,
    max_sims: int = 1024
  ):
    self.B = lane_count
    self.evaluate = evaluate
    self.max_sims = max_sims

    B = lane_count

    self.visits = np.zeros((B, max_nodes), dtype=np.int32) # number of visits
    self.network_value = np.zeros((B, max_nodes), dtype=np.float32) 
    self.value_sum = np.zeros((B, max_nodes), dtype=np.float32)
    self.prior = np.zeros((B, max_nodes), dtype=np.float32)

    self.expanded = np.zeros((B, max_nodes), dtype=bool)
    self.n_actions = np.zeros((B, max_nodes), np.int32)
    self.result = np.zeros((B, max_nodes), dtype=np.int8)
    self.child_count = np.zeros(B, dtype=np.int32)
    self.node_count = np.ones(B, dtype=np.int32)
    
    # keep separate child array for expanded parents
    self.to_children_idx = np.full((B, max_nodes), -1, np.int32)
    self.children = np.full((B, max_sims + 1, MAX_MOVES_CAPACITY), -1, np.int32)


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
  ):
    self.batch = batch
    self._reset(n_sims)
    B = len(self.batch)
    active = self.result[:, 0] == GameResult.ONGOING
    if mask is not None:
      active &= mask
    lanes = np.flatnonzero(active).astype(np.int32)
    chosen = np.full(B, -1, np.int32)
    policy = np.zeros((B, MAX_MOVES_CAPACITY), np.float32)
    root_value = np.zeros(B, np.float32)
    if not len(lanes):
      return chosen, policy, root_value

    root_values = self._expand(lanes, np.zeros(len(lanes), np.int32))
    self.visits[lanes, 0] = 1
    self.value_sum[lanes, 0] = root_values
    if noise_epsilon > 0:
      self._add_root_noise(lanes, rng, noise_alpha, noise_epsilon)

    for _ in range(n_sims):
      paths = [self._select(int(lane), c_puct) for lane in lanes]
      states = np.array([path[-1] for path in paths], np.int32)
      results = self.result[lanes, states]
      values = np.where(results == GameResult.CHECKMATE, -1.0, 0.0).astype(np.float32)

      ongoing = results == GameResult.ONGOING
      if ongoing.any():
        values[ongoing] = self._expand(lanes[ongoing], states[ongoing])

      for lane, path, value in zip(lanes, paths, values):
        self._backup(int(lane), path, float(value))

    # root Q after search, from the root mover's perspective (root value_sum is
    # seeded with the network's own estimate and accumulates backups as-is)
    root_value[lanes] = self.value_sum[lanes, 0] / self.visits[lanes, 0]

    for lane in lanes:
      count = self.n_actions[lane, 0]
      children_idx = self.to_children_idx[lane, 0]
      children = self.children[lane, children_idx, :count]
      visits = self.visits[lane, children].astype(np.float32)
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
    self.prior.fill(0)

    self.expanded.fill(0)
    self.n_actions.fill(0)
    self.result.fill(0)
    self.child_count.fill(0)
    self.node_count.fill(1)

    self.to_children_idx.fill(-1)
    self.children[:, :n_sims + 1].fill(-1)

    self.n_actions[:, 0] = self.batch.num_moves_all()
    self.result[:, 0] = self.batch.results()


  def _expand(self, lanes, states):
    priors, values = self.evaluate(self.batch, lanes, states)
    action_counts = self.n_actions[lanes, states]
    self.network_value[lanes, states] = values

    assert np.all(self.node_count[lanes] + action_counts <= self.batch.states_cap)

    children_idx = self.child_count[lanes].copy()
    self.child_count[lanes] += 1
    self.to_children_idx[lanes, states] = children_idx

    parents = np.zeros(len(self.batch), np.int32)
    parents[lanes] = states
    actions = np.full(len(self.batch), -1, np.int32)

    # have to go by action over all batches
    for action in range(int(action_counts.max())):
      has_action = action < action_counts
      games = lanes[has_action]
      actions.fill(-1)
      actions[games] = action

      children, counts, results = self.batch.get_children(parents, actions)
      child_ids = children[games]
      self.children[games, children_idx[has_action], action] = child_ids
      self.prior[games, child_ids] = priors[has_action, action]
      self.n_actions[games, child_ids] = counts[games]
      self.result[games, child_ids] = results[games]

    self.node_count[lanes] += action_counts
    self.expanded[lanes, states] = True
    return values


  def _add_root_noise(self, lanes, rng, alpha, epsilon):
    # self-play exploration: fresh dirichlet noise mixed into the root's edge
    # priors, rebuilt from the base priors every search. only self.prior (a
    # per-search array) is touched, so cached network predictions stay clean
    # and evaluation (epsilon = 0) sees the base priors.
    for lane in lanes:
      count = self.n_actions[lane, 0]
      children = self.children[lane, self.to_children_idx[lane, 0], :count]
      noise = rng.dirichlet(np.full(count, alpha, np.float64))
      self.prior[lane, children] = (
        (1 - epsilon) * self.prior[lane, children] + epsilon * noise
      ).astype(np.float32)

  def _select(self, lane, c_puct):
    path = [0]
    parent = 0
    while self.result[lane, parent] == GameResult.ONGOING \
        and self.expanded[lane, parent]:
      row = self.to_children_idx[lane, parent]
      children = self.children[lane, row, :self.n_actions[lane, parent]]
      visits = self.visits[lane, children]
      q = np.divide(
        self.value_sum[lane, children], 
        visits,
        out = np.zeros(len(children), np.float32),
        where = visits > 0
      )
      exploration = (
        c_puct * self.prior[lane, children] 
        * np.sqrt(max(1, self.visits[lane, parent])) / (1 + visits)
      )
      parent = int(children[np.argmax(q + exploration)])
      path.append(parent)
    return path


  def _backup(self, lane, path, value):
    for child in reversed(path[1:]):
      value = -value
      self.visits[lane, child] += 1
      self.value_sum[lane, child] += value

    # root
    self.visits[lane, 0] += 1
    self.value_sum[lane, 0] += value