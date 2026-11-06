"""Budgeted alpha-beta with unchanged MVV-LVA ordering and quiescence.

V4 uses separate full-window root searches. V4.2 (root_prefetch=True)
first evaluates every root child, then iteratively deepens with shared alpha
and quiescence. Only completed iterations replace the decision and temperature
scores. Those scores may be bounds, not exact minimax values.
Quiescence searches all check evasions without stand pat. Checks can extend
past q_depth; evaluation/storage budgets still terminate search approximately.
Static evaluations and bounded, history-aware search results are cached separately.
Mate scores reserve magnitudes above mate_value/2 for distance normalization.
"""
from collections.abc import Callable
from collections import OrderedDict

import numpy as np

from engine.game import GameBatch, GameResult, MoveType, MAX_MOVES_CAPACITY

# (tree, state) -> value in [-1, 1] from that state's mover's perspective
Evaluate = Callable[[GameBatch, int], float]

ONGOING = int(GameResult.ONGOING)
CHECKMATE = int(GameResult.CHECKMATE)
EN_PASSANT = int(MoveType.EN_PASSANT)
INF = float("inf")
CACHE_MAX = 1_000_000  # eval cache entries before a full clear

# promotion bonus by move type (queen promos search first among promotions)
PROMO_BONUS = np.zeros(7, np.float32)
PROMO_BONUS[int(MoveType.PROMOTION_KNIGHT)] = 3.0
PROMO_BONUS[int(MoveType.PROMOTION_BISHOP)] = 3.0
PROMO_BONUS[int(MoveType.PROMOTION_ROOK)] = 5.0
PROMO_BONUS[int(MoveType.PROMOTION_QUEEN)] = 9.0


class AlphaBetaQuiescence:
  def __init__(
    self,
    evaluate: Evaluate,
    piece_values: np.ndarray,  # value per mailbox code 0..12, both colors > 0
    *,
    max_depth: int,
    q_depth: int,
    max_nodes: int,
    mate_value: float,
    tree_cap: int,
    temperature: float,
    rng: np.random.Generator,
    root_prefetch: bool = False,
    tt_capacity: int = 0,
  ):
    if max_depth < 1 or q_depth < 0 or max_nodes < 1 or temperature < 0:
      raise ValueError("invalid search depth, budget or temperature")
    if tt_capacity < 0:
      raise ValueError("tt_capacity must be nonnegative")
    self.tt_capacity = tt_capacity
    self.tt = OrderedDict()
    self._state_keys = {}
    self._tt_hits = self._tt_cutoffs = 0
    self.root_prefetch = root_prefetch
    self.evaluate = evaluate
    self.piece_values = np.asarray(piece_values, np.float32)
    self.max_depth = max_depth
    self.q_depth = q_depth
    self.max_nodes = max_nodes
    self.mate_value = mate_value
    self.temperature = temperature
    self.rng = rng
    self.tree = GameBatch(1, sim_cap=max(tree_cap, max_depth + q_depth + 2,
      MAX_MOVES_CAPACITY + 1 if root_prefetch else 1))
    self.evals: dict[int, float] = {}  # zobrist -> net eval

  def clear_cache(self) -> None:
    "Clear evaluation and search caches whenever the evaluator changes."
    self.evals.clear()
    self.tt.clear()

  # ---------------- per-node helpers ----------------
  def _eval(self, state: int) -> float:
    h = int(self.tree.hashes(states=np.array([state], np.int32))[0])
    value = self.evals.get(h)
    if value is None:
      self._nodes += 1
      value = float(self.evaluate(self.tree, state))
      if self.tt_capacity and (not np.isfinite(value) or abs(value) >= self.mate_value / 2):
        raise ValueError("static evaluation must be finite and below mate_value/2 in magnitude")
      if len(self.evals) >= CACHE_MAX:
        self.evals.clear()
      self.evals[h] = value
    return value

  def _ordered(self, state: int) -> tuple[np.ndarray, np.ndarray]:
    """MVV-LVA action order + per-action 'forcing' flag (capture/promo).
    Forcing moves score > 0, quiets 0, so forcing sorts first."""
    board, *_, mv, num = self.tree.encode(states=np.array([state], np.int32))
    rows = mv[0][:int(num[0])].astype(np.int32)
    to, mtype, attacker = rows[:, 1], rows[:, 2], rows[:, 3]
    victim = board[0][to].astype(np.int32)
    victim = np.where(mtype == EN_PASSANT, 7, victim)  # ep captures a pawn
    score = np.where(
      victim > 0,
      100.0 + 8.0 * self.piece_values[victim] - self.piece_values[attacker],
      0.0,
    ).astype(np.float32)
    score += PROMO_BONUS[mtype] * 10.0
    order = np.argsort(-score, kind="stable").astype(np.int32)
    return order, score > 0

  def _child(self, parent: int, action: int) -> tuple[int | None, int]:
    """Materialize one child; (None, 0) when the arena is full."""
    if self._used >= self.tree.states_cap:
      self._truncated += 1
      return None, 0
    self._used += 1
    s, _, r = self.tree.get_children(
      np.array([parent], np.int32), np.array([action], np.int32))
    return int(s[0]), int(r[0])

  def _cached_search(self, state, depth, ply, alpha, beta, mode, search):
    if not self.tt_capacity:
      return search()
    context = self._state_keys.get(state)
    if context is None:
      context = tuple(int(v) for v in self.tree.search_keys(
        states=np.array([state], np.int32))[0])
      self._state_keys[state] = context
    # Exact remaining depth and q_depth keep fixed-depth semantics. Cache
    # internal normal-search nodes only: leaf evaluation is already cached,
    # and probing/storing every quiescence node adds unnecessary overhead.
    key = (context, mode, depth, self.q_depth, self.mate_value)
    entry = self.tt.get(key)
    if entry is not None:
      self._tt_hits += 1
      self.tt.move_to_end(key)
      value, bound = entry
      if value > self.mate_value / 2:
        value -= ply
      elif value < -self.mate_value / 2:
        value += ply
      if bound == 'exact' or (bound == 'lower' and value >= beta) or (bound == 'upper' and value <= alpha):
        self._tt_cutoffs += 1
        return value
    before = self._truncated
    value = search()
    if self._truncated == before:
      bound = 'upper' if value <= alpha else 'lower' if value >= beta else 'exact'
      stored = value + ply if value > self.mate_value / 2 else value - ply if value < -self.mate_value / 2 else value
      self.tt[key] = (stored, bound)
      self.tt.move_to_end(key)
      if len(self.tt) > self.tt_capacity:
        self.tt.popitem(last=False)
    return value

  # ---------------- negamax ----------------
  def _search(self, state: int, result: int, depth: int, ply: int,
              alpha: float, beta: float) -> float:
    if result != ONGOING or depth <= 0:
      return self._search_uncached(state, result, depth, ply, alpha, beta)
    return self._cached_search(state, depth, ply, alpha, beta, 'normal',
      lambda: self._search_uncached(state, result, depth, ply, alpha, beta))

  def _search_uncached(self, state: int, result: int, depth: int, ply: int,
                       alpha: float, beta: float) -> float:
    if result == CHECKMATE:                # side to move here is mated
      return -(self.mate_value - ply)
    if result != ONGOING:
      return 0.0
    if self._nodes >= self.max_nodes:      # out of budget: score in place
      self._truncated += 1
      return self._eval(state)
    if depth <= 0:
      return self._qsearch(state, ply, self.q_depth, alpha, beta)

    order, _ = self._ordered(state)
    best = -INF
    for a in order:
      child, res = self._child(state, int(a))
      if child is None:                    # arena full mid-node
        break
      v = -self._search(child, res, depth - 1, ply + 1, -beta, -alpha)
      best = max(best, v)
      alpha = max(alpha, v)
      if alpha >= beta:
        break
    if best == -INF:                       # no child fit in the arena
      return self._eval(state)
    return best

  def _qsearch(self, state: int, ply: int, qdepth: int,
               alpha: float, beta: float) -> float:
    in_check = bool(self.tree.in_check(states=np.array([state], np.int32))[0])
    if self._nodes >= self.max_nodes:
      self._truncated += 1
      return self._eval(state)  # explicit budget cutoff, not a stand-pat bound
    if in_check:
      best = -INF  # passing is illegal; include quiet king moves and blocks
    else:
      stand = self._eval(state)
      if qdepth <= 0 or stand >= beta:
        return stand
      alpha = max(alpha, stand)
      best = stand

    order, forcing = self._ordered(state)
    for a in order:
      if not in_check and not forcing[a]:  # quiet evasions are mandatory
        break
      child, res = self._child(state, int(a))
      if child is None:
        break
      if res == CHECKMATE:
        v = self.mate_value - (ply + 1)
      elif res != ONGOING:
        v = 0.0
      else:
        v = -self._qsearch(child, ply + 1, qdepth - 1, -beta, -alpha)
      best = max(best, v)
      alpha = max(alpha, v)
      if alpha >= beta:
        break
    if best == -INF:  # arena exhausted before an evasion could be examined
      return self._eval(state)
    return best

  # ---------------- root ----------------
  def _select_one(self, batch: GameBatch, lane: int) -> tuple[int, float]:
    self.tree.copy_lane_from(batch, src_lane=lane)
    self.tree.reset_trees()
    self._state_keys.clear()
    self._tt_hits = self._tt_cutoffs = 0
    self._used = 1                         # root occupies state 0
    self._nodes = 0
    self._truncated = 0

    if self.root_prefetch:
      return self._select_prefetched()

    order, _ = self._ordered(0)
    actions, values = [], []
    for a in order:
      # never hand an UNSEARCHED root child to the argmax: out of budget or
      # arena, stop and pick among the (MVV-LVA-first) children we verified
      if actions and self._nodes >= self.max_nodes:
        self._truncated += 1
        break
      child, res = self._child(0, int(a))
      if child is None:
        break
      # full window for legacy sampling; budget truncation remains approximate
      v = -self._search(child, res, self.max_depth - 1, 1, -INF, INF)
      actions.append(int(a))
      values.append(v)

    return self._pick(actions, values)

  def _select_prefetched(self) -> tuple[int, float]:
    order, _ = self._ordered(0)
    if not len(order):
      raise ValueError("cannot select move on finished game")
    results, values = [], []
    # This mandatory coverage pass may exceed max_nodes when it is smaller
    # than the number of legal moves. Arena capacity reserves room for it.
    for action in order:
      child, result = self._child(0, int(action))
      assert child is not None
      results.append(result)
      values.append(self.mate_value - 1 if result == CHECKMATE else
                    0.0 if result != ONGOING else -self._eval(child))
    self._root_initial = np.asarray(values, np.float32).copy()
    self._root_actions = order.copy()
    self._root_values = self._root_initial.copy()
    self._root_deep = np.zeros(len(order), bool)
    self._completed_depth = 0  # zero denotes the static one-ply fallback
    self._attempted_depth = 0
    completed_index = int(np.argmax(values))
    terminal = [i for i, r in enumerate(results) if r != ONGOING]

    # A single cumulative evaluation budget covers all iterations. Reset only
    # the arena; cached static evaluations and the played-game history survive.
    if not any(r == CHECKMATE for r in results):
      for depth in range(1, self.max_depth + 1):
        if self._nodes >= self.max_nodes:
          self._truncated += 1
          break
        self._attempted_depth = depth
        self.tree.reset_trees()
        self._state_keys.clear()
        self._used = 1
        iteration_values = self._root_initial.copy()
        best_index = max(terminal, key=lambda i: values[i]) if terminal else None
        alpha = values[best_index] if best_index is not None else -INF
        complete = True
        for i, action in enumerate(order):
          if results[i] != ONGOING:
            continue
          if self._nodes >= self.max_nodes:
            self._truncated += 1
            complete = False
            break
          child, result = self._child(0, int(action))
          if child is None:
            complete = False
            break
          before = self._truncated
          value = -self._search(child, result, depth - 1, 1, -INF, -alpha)
          if self._truncated != before:
            complete = False
            break
          iteration_values[i] = value
          if value > alpha:
            alpha, best_index = value, i
        if not complete:
          break  # discard the entire interrupted iteration, including scores
        self._root_values = iteration_values
        self._root_deep = np.asarray(results) == ONGOING
        self._completed_depth = depth
        completed_index = best_index

    if self.temperature == 0:
      return int(order[completed_index]), float(self._root_values[completed_index])
    # All estimates come from one completed iteration. Alpha-beta bounds are
    # still deliberately accepted as approximate exploration scores.
    return self._pick(order, self._root_values)

  def _pick(self, actions, values) -> tuple[int, float]:
    values = np.array(values, np.float32)
    if self.temperature == 0:
      pick = int(np.argmax(values))
    else:
      logits = (values - values.max()) / self.temperature
      probabilities = np.exp(logits)
      probabilities /= probabilities.sum()
      pick = int(self.rng.choice(len(values), p=probabilities))
    return actions[pick], float(values[pick])

  def select_moves(self, batch: GameBatch, mask: np.ndarray):
    moves = np.zeros(len(batch), np.int32)
    values = np.zeros(len(batch), np.float32)
    nodes = np.zeros(len(batch), np.int64)
    truncated = np.zeros(len(batch), np.int64)
    root_evaluated = np.zeros(len(batch), np.int64)
    root_deep = np.zeros(len(batch), np.int64)
    completed_depth = np.zeros(len(batch), np.int64)
    attempted_depth = np.zeros(len(batch), np.int64)
    tt_hits = np.zeros(len(batch), np.int64)
    tt_cutoffs = np.zeros(len(batch), np.int64)
    for lane in np.flatnonzero(mask):
      action, value = self._select_one(batch, int(lane))
      moves[lane] = action
      values[lane] = value
      nodes[lane] = self._nodes
      truncated[lane] = self._truncated
      tt_hits[lane] = self._tt_hits
      tt_cutoffs[lane] = self._tt_cutoffs
      if self.root_prefetch:
        root_evaluated[lane] = len(self._root_actions)
        root_deep[lane] = self._root_deep.sum()
        completed_depth[lane] = self._completed_depth
        attempted_depth[lane] = self._attempted_depth
    return moves, {"value": values, "nodes": nodes, "truncated": truncated,
                   "root_evaluated": root_evaluated, "root_deep": root_deep,
                   "completed_depth": completed_depth, "attempted_depth": attempted_depth,
                   "tt_hits": tt_hits, "tt_cutoffs": tt_cutoffs}
