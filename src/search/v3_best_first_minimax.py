# TODO: review
# changes: replaced the alpha-beta copy with best-first minimax per the
# changelog. descend from root by argmax backed-up value to an unexpanded
# node, expand = materialize + evaluate ALL its children, back up max
# (negamax) along the path. all lanes run in lockstep: one get_children
# sweep + ONE evaluate call per wave. terminal children never hit the net.
# runs directly on the caller's GameBatch (reset_trees each search) instead
# of copying lanes -- that's what makes cross-lane batching possible.
# ideas: tiny noise on the descent argmax if the PV thrashes on a noisy net;
# zobrist eval cache hooks in inside evaluate() later; value stats here are
# what a prioritized buffer would want.

from collections.abc import Callable

import numpy as np

from engine.game import GameBatch, GameResult, MAX_MOVES_CAPACITY

# batched: values for (lanes[i], states[i]) from each state's own
# side-to-move perspective, in [-1, 1]
Evaluate = Callable[[GameBatch, np.ndarray, np.ndarray], np.ndarray]

ONGOING = int(GameResult.ONGOING)
CHECKMATE = int(GameResult.CHECKMATE)


class BestFirstMinimax:
  """Greedy best-first minimax with temperature-based root selection.

  V[lane, state] is the state's value from its own mover's perspective:
  NN eval when it's a frontier leaf, exact for terminals, max(-V[children])
  once expanded. Expanded nodes get a slot holding their children's state
  indices; descent is argmax over -V[children]. One sim = one expansion per
  lane. A lane stops early if its PV ends in a terminal or its C++ arena
  is nearly full (size sim_cap for ~n_sims * branching states).
  """

  def __init__(
    self,
    evaluate: Evaluate,
    *,
    n_sims: int,
    mate_value: float,
    temperature: float,
    rng: np.random.Generator,
  ):
    self.evaluate = evaluate
    self.n_sims = n_sims
    self.mate_value = mate_value
    self.temperature = temperature
    self.rng = rng
    self._shape = None

  def _reset(self, batch: GameBatch) -> None:
    B, cap = len(batch), batch.states_cap
    if self._shape != (B, cap):
      self._shape = (B, cap)
      self.V = np.zeros((B, cap), np.float32)
      self.result = np.zeros((B, cap), np.int8)
      self.n_actions = np.zeros((B, cap), np.int32)
      self.slot_of = np.full((B, cap), -1, np.int32)
      self.child_state = np.full(
        (B, self.n_sims + 1, MAX_MOVES_CAPACITY), -1, np.int32)
      self._lanes = np.arange(B)
    else:
      self.V.fill(0.0)
      self.result.fill(0)
      self.n_actions.fill(0)
      self.slot_of.fill(-1)
      self.child_state.fill(-1)
    self.n_slots = np.zeros(B, np.int32)
    self.states_used = np.ones(B, np.int64)  # root

  def _child_values(self, states: np.ndarray) -> np.ndarray:
    """-V of every child of each lane's (expanded) state; -inf padding."""
    rows = self.child_state[self._lanes, self.slot_of[self._lanes, states]]
    vals = -self.V[self._lanes[:, None], rows.clip(min=0)]
    return np.where(rows >= 0, vals, -np.inf)

  def _expand(self, batch: GameBatch, expand: np.ndarray,
              frontier: np.ndarray, depth: np.ndarray) -> int:
    """Materialize + evaluate all children of each expanding lane's frontier."""
    lanes = self._lanes
    slot = self.n_slots.copy()
    self.slot_of[lanes[expand], frontier[expand]] = slot[expand]
    self.n_slots[expand] += 1

    n = self.n_actions[lanes, frontier]
    need_lanes, need_states = [], []
    for a in range(int(n[expand].max())):
      act = np.where(expand & (a < n), a, -1).astype(np.int32)
      child, child_n, child_res = batch.get_children(frontier, act)
      live = act >= 0
      self.child_state[lanes[live], slot[live], a] = child[live]
      self.n_actions[lanes[live], child[live]] = child_n[live]
      self.result[lanes[live], child[live]] = child_res[live]
      self.states_used[live] += 1

      # terminal children are exact and never hit the net. checkmate = the
      # child's mover is mated; nearer mates score higher for the parent
      mate = live & (child_res == CHECKMATE)
      self.V[lanes[mate], child[mate]] = \
        -(self.mate_value - depth[mate].astype(np.float32))
      draw = live & (child_res != ONGOING) & ~mate
      self.V[lanes[draw], child[draw]] = 0.0
      ongoing = live & (child_res == ONGOING)
      need_lanes.append(lanes[ongoing])
      need_states.append(child[ongoing])

    need_lanes = np.concatenate(need_lanes)
    need_states = np.concatenate(need_states)
    if len(need_lanes):
      values = self.evaluate(
        batch,
        need_lanes.astype(np.int32),
        need_states.astype(np.int32),
      )
      self.V[need_lanes, need_states] = values
    return len(need_lanes)

  def _backup(self, on_path: np.ndarray, path: np.ndarray,
              path_len: np.ndarray) -> None:
    """Recompute V = max(-V[children]) bottom-up along each lane's path."""
    for d in range(int(path_len[on_path].max()) - 1, -1, -1):
      at = on_path & (d < path_len)
      s = path[:, d]
      best = self._child_values(s.clip(min=0)).max(axis=1)
      self.V[self._lanes[at], s[at]] = best[at]

  def select_moves(
    self,
    batch: GameBatch,
    mask: np.ndarray | None = None,
  ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    B = len(batch)
    lanes = np.arange(B)
    if mask is None:
      mask = np.asarray(batch.active)
    mask = np.asarray(mask, dtype=bool)
    if (np.asarray(batch.results())[mask] != ONGOING).any():
      raise ValueError("cannot select move on finished game")

    self._reset(batch)
    batch.reset_trees()  # arena only otherwise resets on play/reset
    self.n_actions[:, 0] = batch.num_moves_all()
    nodes = np.zeros(B, np.int64)

    done = ~mask
    path = np.full((B, self.n_sims + 2), -1, np.int32)
    for _ in range(self.n_sims):
      if done.all():
        break

      # ---- descend the current best line to the frontier ----
      current = np.zeros(B, np.int32)
      path[:, 0] = 0
      path_len = np.ones(B, np.int32)
      expand = np.zeros(B, bool)
      descending = ~done
      while descending.any():
        # PV reached a terminal: its value is exact, greedy search is done
        terminal = descending & (self.result[lanes, current] != ONGOING)
        done |= terminal
        descending &= ~terminal

        frontier = descending & (self.slot_of[lanes, current] < 0)
        expand |= frontier
        descending &= ~frontier

        if descending.any():
          best = self._child_values(current).argmax(axis=1)
          slot = self.slot_of[lanes, current]
          nxt = self.child_state[lanes, slot.clip(min=0), best]
          current = np.where(descending, nxt, current)
          path[lanes[descending], path_len[descending]] = nxt[descending]
          path_len[descending] += 1

      # a full expansion must fit in the C++ arena
      full = expand & (
        self.states_used + self.n_actions[lanes, current] > batch.states_cap)
      done |= full
      expand &= ~full
      if not expand.any():
        continue

      nodes[expand] += self.n_actions[lanes, current][expand]
      self._expand(batch, expand, current, path_len)
      self._backup(expand, path, path_len)

    # ---- root selection with temperature (over values, like v2) ----
    moves = np.zeros(B, np.int32)
    values = np.zeros(B, np.float32)
    root_vals = self._child_values(np.zeros(B, np.int32))
    for lane in np.flatnonzero(mask):
      vals = root_vals[lane, :self.n_actions[lane, 0]]
      if self.temperature == 0:
        selected = int(np.argmax(vals))
      else:
        logits = (vals - vals.max()) / self.temperature
        probabilities = np.exp(logits)
        probabilities /= probabilities.sum()
        selected = int(self.rng.choice(len(vals), p=probabilities))
      moves[lane] = selected
      values[lane] = vals[selected]

    return moves, {"value": values, "nodes": nodes}
