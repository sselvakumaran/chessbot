# V4 search changes

## V4.2 transposition table (current)

The v4.2 config enables a 50,000-entry LRU table (`tt_capacity`; zero disables it). It retains completed internal alpha-beta subtree results across iterative-deepening passes and played moves, while tree arenas can still be reset. Entries contain exact scores, lower bounds or upper bounds. A bound only skips work when it proves a cutoff for the current window. Results require matching remaining depth and quiescence configuration; shallow results do not stand in for deeper searches. Move ordering is unchanged.

The key combines the position Zobrist hash, halfmove clock and an order-independent fingerprint of position occurrence counts since the last pawn move/capture. The C++ engine maintains the fingerprint for both simulated and actual play and lane copies. Histories before a pawn move/capture can be dropped because those positions cannot recur. Otherwise, different repetition contexts do not share scores. Like other hashed tables, this relies on hash collision resistance, not a collision-proof representation. Static neural evaluations continue to use the separate board-only cache.

Mate distances are normalized for reuse after the root advances. Static values must remain strictly within half the mate-value magnitude; larger magnitudes are reserved for mate scores. Truncated subtrees are never stored. Completed subtrees from an interrupted overall iteration remain reusable. `refresh()`/`clear_cache()` clears both tables after changing weights. `tt_hits` and `tt_cutoffs` report found entries and reused exact scores/cutoff bounds.

To avoid duplicating the existing evaluation cache and adding leaf overhead, only internal normal-search nodes use the table. Their scores still include quiescence results. The table persists across moves; this does not preserve the actual tree structure. New-game contexts may safely reuse matching results for an unchanged evaluator, while differing draw contexts get different keys.

Validation: 24 API/search tests pass, covering context separation, actual/simulated key equality, irreversible-history reset, exact/bound window compatibility, depth/mode separation, mate-distance normalization, capacity, invalidation, interrupted-store rejection and table-on/off search parity. A four-position depth-three inference probe with existing v4.1 weights returned identical moves/values with the table off/on. Child visits fell from 13,411 to 12,397 across those positions. Repeating an already searched position fell from 2,045 visits to 80. Short timings are illustrative, not a general speed or strength claim; no training was performed.

## 2026-09-09: V4.2 iterative deepening (current behavior)

V4.2 now retains its mandatory static evaluation of every immediate move as an emergency fallback, then searches depth 1 with quiescence, depth 2 with quiescence, and so on up to max_depth. Every iteration uses shared root alpha and the unchanged MVV-LVA ordering. Only a fully completed root iteration replaces the selected move and score vector. An evaluation-budget or arena interruption discards that entire iteration.

The evaluation budget is cumulative across iterations. Search storage resets between iterations, while the evaluation cache and actual game history are preserved. Quiescence can still consume the budget in the first iteration; in that case the initial static fallback is used. Completed depth is nominal alpha-beta depth, excluding quiescence extensions.

At temperature zero, use the winner of the last completed iteration. At positive temperature, sample from that iteration's score vector, accepting alpha-beta bounds as approximate exploration scores. Scores from an interrupted deeper iteration are not mixed in. If no iteration completed, sample the initial depth-one static values. An immediate mate found in the initial pass bypasses further deepening.

Diagnostics: `completed_depth` is the last completed depth (0 means static fallback); `attempted_depth` includes an interrupted attempt; `root_deep` now counts nonterminal root candidates in the last completed iteration. The old root_prefetch configuration switch enables this behavior without changing model/checkpoint compatibility. Existing v4/v4.1 configs retain their legacy root search.

All 18 API/search tests pass, including interrupted-iteration score rollback for sampling and root-alpha reset per iteration. No new training or strength tournament was run for this change. Earlier match results below and in task reports concern the previous implementation.

## 2026-09-09: quiescence correction (v4 / v4.1)

The shared quiescence implementation now asks the C++ engine whether the mover is in check. In check it does not use stand pat and searches all legal evasions, including quiet king moves and blocks. Check evasions extend beyond the nominal q_depth until a non-check position is reached; evaluation/storage budgets still impose explicit approximate cutoffs. Terminal children retain mate/draw scoring. Static evaluations remain cached separately from path-dependent outcomes.

The default root algorithm for existing v4/v4.1 configs remains unchanged. Rebuild the native library with `make` after updating the engine binding. Historical tournament results used the previous implementation and should not be attributed to this corrected search.

## Previous V4.2 behavior (superseded): immediate-move coverage plus shared-alpha search

`configs/v4.2_quiescence.json` enables `root_prefetch: true` on the same bot/model/trainer. Its pre-existing training settings are preserved. Existing v4.1 weights are architecture-compatible, but no checkpoints were copied and no training run was started.

1. Materialize every legal root child in the existing MVV-LVA/promotion order. Score immediate mate/draw exactly; otherwise evaluate the child directly (depth one, no quiescence in this initial pass).
2. Use the remaining budget for ordinary deeper alpha-beta with a shared root alpha, retaining the same move ordering. Initial shallow scores do not establish the deeper pruning threshold; known terminal outcomes can. An interrupted subtree does not advance alpha.
3. At temperature zero, select the best completed deeper root candidate or known terminal outcome; if neither exists, use the depth-one fallback. Do not greedily rank incomplete estimates/upper bounds against fully searched moves.
4. At positive temperature, softmax over all root candidates. Replace initial scores with deeper returned estimates where available, including upper bounds or partially searched values. Leave unsearched candidates at their initial score. This deliberately approximate exploration distribution mixes depths and bounds. There is no arbitrary depth discount: multiplying a negative score by a discount could accidentally promote a bad move.

Returned `value` is the score associated with the selected move. Under positive temperature it is an exploration estimate, not a guaranteed minimax value. Existing training target logic is unchanged. `root_evaluated` and `root_deep` diagnostics expose initial coverage and the number of candidates whose deeper search was attempted; attempted does not mean fully completed. These diagnostics are zero for the legacy root mode.

The initial pass has priority over max_nodes, so tiny budgets may be exceeded to evaluate every immediate move. The arena reserves at least MAX_MOVES_CAPACITY + 1 states for this mode. Deeper coverage is still budget-dependent. Full root coverage guarantees detecting immediate terminal outcomes, not refutations multiple plies away. Sampling may still deliberately choose inferior moves.

## Validation

Engine rebuild succeeded. Regression tests cover check flags in both colors and child states, quiet evasions with q_depth zero, illegal stand-pat cutoff prevention, immediate mate coverage under a one-evaluation budget, mask handling, root alpha propagation, shallow/partial greedy fallback, finite exploratory sampling, and history-dependent repetition scoring. Existing API tests also pass. Both corrected v4.1 and v4.2 were smoke-tested with the existing v4.1 checkpoint. Strength improvement has not yet been measured.
