from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from engine.game import GameBatch, GameResult


Evaluate = Callable[[GameBatch, int], float]
OrderActions = Callable[[GameBatch, int], np.ndarray]


@dataclass
class _SearchState:
  n_nodes: int = 0
  n_in_storage: int = 1
  path_actions: list[int] = field(default_factory=list)
  path_states: list[int] = field(default_factory=lambda: [0])
  n_truncated: int = 0
  truncated_depth: int = 0

  def pop(self) -> None:
    self.path_actions.pop()
    self.path_states.pop()


class AlphaBetaTemperature:
  """Negamax alpha-beta search with temperature-based root selection."""

  def __init__(
    self,
    evaluate: Evaluate,
    *,
    max_depth: int,
    max_nodes: int,
    mate_value: float,
    temperature: float,
    rng: np.random.Generator,
    order_actions: OrderActions | None = None,
    sim_cap: int = 512
  ):
    self.evaluate = evaluate
    self.max_depth = max_depth
    self.max_nodes = max_nodes
    self.mate_value = mate_value
    self.temperature = temperature
    self.rng = rng
    self.order_actions = order_actions

    # The C++ tree is finite. If it fills, _flush_tree reconstructs only the
    # current path and search continues without changing the result.
    self.tree = GameBatch(1, sim_cap=max(sim_cap, max_depth + 2))

  def _ordered_actions(self, state: int) -> np.ndarray:
    if self.order_actions is not None:
      return self.order_actions(self.tree, state)
    n_moves = int(self.tree.encode(
      states=np.array([state], dtype=np.int32)
    )[-1][0])
    return np.arange(n_moves, dtype=np.int32)

  def _flush_tree(self, search: _SearchState) -> None:
    self.tree.reset_trees()
    state = 0
    for i, action in enumerate(search.path_actions):
      child, _, _ = self.tree.get_children(
        np.array([state], dtype=np.int32),
        np.array([action], dtype=np.int32),
      )
      state = int(child[0])
      search.path_states[i + 1] = state
    search.n_in_storage = 1 + len(search.path_actions)

  def _expand(self, search: _SearchState, action: int) -> int:
    if search.n_in_storage + 1 > self.tree.states_cap:
      self._flush_tree(search)

    child, _, result = self.tree.get_children(
      np.array([search.path_states[-1]], dtype=np.int32),
      np.array([action], dtype=np.int32),
    )
    search.n_nodes += 1
    search.n_in_storage += 1
    search.path_actions.append(action)
    search.path_states.append(int(child[0]))
    return int(result[0])

  def _negamax(
    self,
    search: _SearchState,
    result: int,
    depth: int,
    alpha: float,
    beta: float,
  ) -> float:
    if result == int(GameResult.CHECKMATE):
      return -(self.mate_value - len(search.path_actions))
    if result != int(GameResult.ONGOING):
      return 0.0
    if depth == 0 or search.n_nodes >= self.max_nodes:
      if depth > 0:
        search.n_truncated += 1
        search.truncated_depth += depth
      return float(self.evaluate(self.tree, search.path_states[-1]))

    best = -np.inf
    for action in self._ordered_actions(search.path_states[-1]):
      child_result = self._expand(search, int(action))
      value = -self._negamax(
        search, child_result, depth - 1, -beta, -alpha
      )
      search.pop()
      best = max(best, value)
      alpha = max(alpha, value)
      if alpha >= beta:
        break
    return float(best)

  def _select_one(
    self,
    batch: GameBatch,
    lane: int,
  ) -> tuple[int, float, _SearchState]:
    self.tree.copy_lane_from(batch, src_lane=lane)
    if int(self.tree.results()[0]) != int(GameResult.ONGOING):
      raise ValueError("cannot select move on finished game")

    search = _SearchState()
    actions = self._ordered_actions(0)
    values = []
    for action in actions:
      child_result = self._expand(search, int(action))
      value = -self._negamax(
        search,
        child_result,
        self.max_depth - 1,
        -np.inf,
        np.inf,
      )
      search.pop()
      values.append(value)

    values = np.asarray(values)
    if self.temperature == 0:
      selected = int(np.argmax(values))
    else:
      logits = (values - values.max()) / self.temperature
      probabilities = np.exp(logits)
      probabilities /= probabilities.sum()
      selected = int(self.rng.choice(len(actions), p=probabilities))

    return int(actions[selected]), float(values[selected]), search

  def select_moves(
    self,
    batch: GameBatch,
    mask: np.ndarray,
  ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    moves = np.zeros(len(batch), dtype=np.int32)
    values = np.zeros(len(batch), dtype=np.float32)
    nodes = np.zeros(len(batch), dtype=np.int64)
    truncated = np.zeros(len(batch), dtype=np.int64)
    avg_truncation = np.zeros(len(batch), dtype=np.float32)

    for lane in np.flatnonzero(mask):
      action, value, search = self._select_one(batch, int(lane))
      moves[lane] = action
      values[lane] = value
      nodes[lane] = search.n_nodes
      truncated[lane] = search.n_truncated
      if search.n_truncated:
        avg_truncation[lane] = (
          search.truncated_depth / search.n_truncated
        )

    return moves, {
      "value": values,
      "nodes": nodes,
      "truncated": truncated,
      "avg_truncation": avg_truncation,
    }
