Problems with last version:
- Constantly hitting repetition draw (penalize or remove from possible move choices)
- Low signal per game (needs more training samples, repeat memory for reinforcement)
- Search is slow, must limit to depth 3, need better search algorithm
  - Not necessariliy alpha beta pruning is bad, but need to search deeper, avoid searching clearly poor states

NEW VERSION
- model architecture needs shift.
  - prioritize embedding parameters since makes up most of the work.
  - remove computing empty squares, not needed
  - maybe add castling, enpassant rights? would be richer / non hot-1 but can add in post.
- use best first minimax: evaluate best items, without needing policy grading.
- add replay buffer for better signal per game
---

UPDATES (after discussion)

corrections to problems above:
- repetition needs no fix: engine already scores repeats as draws in-tree. don't mask moves, don't feed rep count to model.
- real signal problem: v2 only trained on checkmates (early games are all draws -> no gradient). train on all outcomes.
- search root values were computed then thrown away every move -> use as training targets.
- alpha-beta can't batch leaf evals from python, that's the actual slowness. replace it, don't tune it.

v3 (mlp backend, nnue basis):
- model (models/v3_mlp_nnue): first layer = embedding table over sparse features. only pieces present (sq*12 + piece, ~2-32 active), + 4 castling rows + ep-file row in the same sum. two frames from the canonical encode (stm as-is; opp = flip rank, swap color), ONE shared table, head eats concat(stm, opp). tiny head: 2d -> 32 -> 32 -> WDL, clipped relu (clamp 0..1, quantization-ready later). feature extraction lives in its own function (index + mask lists) so future delta code reuses it.
  - NO incremental updates in v3: descent never evaluates (tree stats cache every node's value, each node hits the net exactly once at expansion). fresh gather is only ~1/3 of eval ops and mps is launch-bound anyway -> win by batching big instead. incremental moved back to later.
- search (search/v3_best_first_minimax): best-first minimax, not mcts. descend from root by argmax backed-up value until an unexpanded node (reachability for free: descent alternates perspective), expand = evaluate ALL its children in one batch, back up max (negamax) along the path. all lanes in lockstep waves: one get_children sweep + ONE forward per wave. terminal children never hit the net (mate/draw scored directly). root move via temperature over root values for self-play diversity. note: all-children expansion materializes ~35 states/expansion -> sim_cap ~4096, not 16.
- replay buffer (misc/replay_buffer): circular, store v_search and z as separate columns (blend at sample time -> lambda tunable without regenerating a night of data). no pi column, no policy yet.
- targets: lambda * (root search value) + (1-lambda) * z per result type: checkmate/stalemate -> trust z; repetition/50-move -> mostly bootstrap (z is noise between weak players); capped -> pure bootstrap. CONFIG OPTION: toggle off to fall back to v2-style checkmate-only targets, for a/b since unsure it helps.
- self-play on many parallel lanes (this is the gpu batching), minibatch grad steps sampled from buffer, decoupled from game generation.

later / test:
- engine perf (measured: ~half of move time is python/ffi skeleton, not the net; cpu beats mps at every batch size for this model). ordered by roi:
  - [engine] encode buffers sized by caller, not len(batch) (~5 lines game.py): kills the 31-chunk encode per wave. ~10-15%
  - [engine] expand_all in c++: materialize all children of the frontier in ONE call instead of ~30 get_children crossings per wave. ~10-15%
  - [engine] openmp over lanes in GameManager loops: c++ work is serial across lanes today. ~15-25% at 128 lanes
  - [engine] descend/expand/backup loop in c++, python keeps only the per-wave nn call: deletes most of the numpy skeleton. ~1.5-2x, but locks the search design in -- do after search proves itself
  - [engine] proper pin/check movegen (get_moves currently copy+make per pseudo-move): only if a profile says so, movegen is a small slice
  - first three = an afternoon, ~1.5-2x combined; loop-in-c++ later -> ~3-4x total. zobrist cache + async double-buffering (below) multiply on top
- eval arena vs fixed ladder (random, v1, past checkpoints) to measure elo, not vibes. want it soon but not part of v3 proper.
- [stockfish] incremental accumulator (the "EU"): child_acc = parent_acc - old rows + new rows at expansion. only saves the gather (~1/3 of ops), needs gpu acc arena + move->delta extraction (shares the features() logic). revisit with the cpu eval path + quantization, and measure.
- halfkp ablation: king-conditioned index (king_sq * 64*13 + sq*13 + piece, ~53k rows). one config knob. maybe redundant once attention exists. note: under halfkp every king move forces a full accumulator rebuild (all features king-indexed).
- [stockfish] zobrist eval cache: hash -> (p_logits, v). dedupes transpositions in-tree and across moves. needs hash exposed thru ffi (~15 lines). invalidate on weight update.
- [stockfish] quantization (int8/int16): only meaningful with the cpu eval path above, defer together.
- [pufferlib] async double-buffered rollouts: cpu descends group B while gpu evaluates group A. biggest future throughput lever.
- [pufferlib] static memory registry + graph capture (metal not cuda lol) to kill launch overhead.
- [pufferlib] prioritized replay: sample by |target - pred| instead of uniform.
- [pufferlib] muon for 2d trunk matrices; embeddings/heads stay adamw.
- frontier experiments (the point): fixed compute per move, sweep model width x n_sims, elo each config. redo in v4 to see frontier shift.

cut:
- quiescence search (alpha-beta only; mcts resolves tactics by deeper visits)
- classical transposition table (alpha-beta only; eval cache covers mcts)
- repetition masking / rep-count input (engine handles it)
- gpu move gen (keep engine cpu, gpu holds only the net — lc0/pufferlib shape)
- subtree stat reuse between moves (arena layout awkward; eval cache recovers most)

---

REWRITE (mvp, after measuring v3/v4 with games)

what the games said (16-game matches, both colors, T=0.02):
- v3 > v4.1 (12-3). same net under both searches: v3 search wins 13-1 -> the search, not the eval, decided it. best-first spends evals along the PV (effective depth 6-10 on critical lines); depth-4 alpha-beta + quiescence spends them uniformly. selective > uniform at this eval cost, opposite of folklore.
- v3@434 beat v3@25 12-0; v4.1@1000 vs v4.1@25 even. v4's pipeline never learned: 30 random plies then greedy = garbage openings + no decision diversity. loss went down anyway. loss never measures strength; only games do.
- both lost 0-16 to v1.1 (material+pst, depth 4). v3 beat material-only 10-3, v4.1 lost 5-10. learned evals ~ noisy material counters after a night.
- nets co-adapt to the search that trained them (each net better under its own search). compare whole bots.
- removing v4's quiet-move beam did NOT help (exhaustive v4 still lost 2-10). beam scrapped for simplicity, not as the culprit.

rewrite = one model, one trainer, two searches:
- shared trainer (v3/train.py, v4 subclasses): single process, replay buffer (board, castling, ep, z, distance), targets = discount**plies_to_end * onehot(z) + rest draw. draws + capped -> draw. NOTHING search-derived trained on. temperature over root values whole game (0.05 -- the setting that demonstrably learned; 0.2/30-ply scheme gone).
- v3 search unchanged. v4 search: beam + transposition table removed; MVV-LVA, quiescence, eval cache, node budget kept; root children full-window so root values are exact.
- scrapped from the plan on purpose: near-best sampling, in-loop arena, implied-piece-value diagnostic (all derivable from checkpoints later).
- new templates: configs/v3.1_nnue_inspired.json, configs/v4.2_quiescence.json. old runs' config.json stripped of obsolete keys (originals in config.orig.json) so their checkpoints still load.
- simplified further: no process pool, no numpy model port. one process, one bot sharing the trainer's model object; v4 evals are plain torch calls per leaf (slow but simple -- the eval cache carries it). perf work stays parked in later/test.

---

V5 PLAN (shallow transformer + AlphaZero-style MCTS; not yet built)

Scope: establish a correct policy/value self-play baseline and test strength gained per move-time and training-time budget. Use MCTS with PUCT as its selection rule, plus the explicitly chosen discounted-outcome tweak. Model implementation/width remain open; the following is the agreed input and search contract. Search/storage optimizations and KV caching belong in v6.

model and inputs:
- shallow transformer, 1-2 layers. HEAVY embedding means more ROWS / stored feature capacity, not necessarily wider transformer blocks. Lookups have cheap arithmetic; memory traffic, optimizer state, and updates still cost something. Benchmark width rather than assuming wider is faster.
- occupied-piece tokens only (at most 32), using actual board squares and a consistent mover-relative perspective. No empty-square tokens initially; absent squares are represented implicitly. Padding is masked and is not an empty-square feature.
- starting piece encoding: shared piece/side row + shared square row + joint piece/side/square row. This shares training signals while permitting square-specific exceptions; it is not extra expressive power over an unrestricted joint table. More rows are distinct from increasing trunk width. Exact dimensions/initialization remain open.
- separate castling-rights and en-passant state tokens. Preserve side to move through the canonical representation and cache key. No board-history, repetition-count, or halfmove-clock network inputs initially; the engine retains all of them for exact rules. A history-blind leaf value cannot anticipate every approaching draw; this is an explicit approximation.
- one token per engine-generated legal move. Encode source, destination, moving/resulting piece including promotion, captured piece and capture square, and special-move changes with distinguishable roles. Do not blindly reuse v1's additive from/to recipe; exact indexing stays open. No attack-map features initially.
- fixed board/state/value tokens attend only to fixed tokens. Each move token attends to fixed tokens and itself, never another move. Token-local normalization/FF operations; no move-order positional embeddings. This preserves independent move scores and makes chunking transparent.
- output current-position value and one policy logit per legal move; no auxiliary heads initially. The engine handles legality. A move logit ranks candidates and is not a calibrated action value. Value pooling/head implementation remains open.
- start with 48 move slots per dense execution chunk, subject to measurement. This is an execution size, never a cap on legal moves: process overflow in BOTH training and inference, concatenate every legal logit, then apply ONE global softmax/loss. Mask padding and count the position's value loss once. No context-extension fine-tuning is required. v5 repeats the fixed-token computation across chunks; KV reuse is v6.

search (v5 baseline):
- AlphaZero-style MCTS with PUCT replaces v3's greedy best-first minimax descent and max backup. Retain cross-game waves, not the old search semantics.
- each position node stores legal edges, expansion/terminal status, neural value or exact history-specific terminal result, and total visits. Each edge stores move, prior P, visit count N, accumulated value W, and child index/key; derive Q=W/N. Keep node and edge count conventions explicit and consistent.
- select using Q + c_puct * P * sqrt(N_parent)/(1 + N), from the selecting player's perspective. Define unvisited Q (baseline zero), and explicitly initialize the root so priors guide the first selection.
- each simulation records its traversed node/edge path while updating a simulated game state seeded with the real game's history. Stop at a terminal or an unexpanded frontier. Check exact terminal rules before consulting neural predictions.
- gather at most one frontier per active game per wave; batch cache misses across games. Evaluate each frontier's policy/value, initializing its legal move edges without evaluating all child boards. Hits and terminals need no forward pass. Chunk overflow may require additional network calls.
- back up along ONLY the recorded path, updating every visited node's count and selected edge's N/W with alternating perspective. Do not scan the search cache or update siblings. A terminal finishes that simulation, not the root search.
- use a fresh configurable simulation budget per actual move, or a move-time budget for timed comparisons. Retained visits do not consume the new simulation allowance. Simulations, unique expansions, and network evaluations are different counters: hits/terminal revisits do not guarantee another unique position.
- self-play root exploration: fresh Dirichlet noise on base legal priors, plus temperature over root VISIT COUNTS for an initial configurable number of plies; later select the most visited move. Evaluation uses no root noise. Constants remain explicit settings.
- retain the current tree storage approach where practical, adapted for MCTS statistics. No lc3-style hash node repository or shared-transposition search graph is required for v5.

search cache (v5; retained search work across actual moves):
- this name covers retained tree work plus reusable neural predictions. Keep their storage/validity distinct internally: neural entries contain ONLY network outputs (value and move logits), while path-specific tree nodes contain MCTS statistics and terminal results. This is separate from the short-lived KV cache below.
- neural prediction key covers ALL actual network inputs: pieces, mover perspective/side, castling and en passant for this baseline. Bind to fixed model weights. Store move logits with stable move identities and remap to the local legal-move order. Draw results and root noise must never overwrite a base neural prediction.
- different paths may reuse the same neural output, but do not merge their MCTS statistics or terminal flags by board hash: repetition/clock context belongs to the search path. Engine history remains complete even though the network omits it.
- after either side plays, promote that actual child to root and retain its descendants and P/N/W. New backups stop at the current root. If the child is unavailable, create a new root and reuse any matching neural prediction. Continue with a fresh search budget, allowing prior work to support deeper search.
- reclaim unplayed tree branches with bounded storage; retain their neural predictions until ordinary eviction. Positions in those branches can transpose back into future search, but measuring whether they justify their space is deferred to v6.
- use a simple bounded per-game prediction hash map with FIFO eviction to start. No branch-membership tracking or reference counting for prediction eviction. An evicted prediction must not invalidate retained tree statistics/priors; tree nodes own the data needed to keep searching.
- clear the entire search cache at each new game, and invalidate it immediately if model weights change. Freeze weights during a generation round, then train; never carry old statistics into a new model version.
- baseline policy targets use the root's total post-search edge visits, including retained visits. Record inherited versus fresh visits separately so reuse does not masquerade as new search work. Rebuild root noise from base priors instead of accumulating noise across searches.

training:
- fresh v5 replay buffer/schema; do not assume v3/v4 or old v1 samples are compatible. Preserve every legal move and its policy target across chunking.
- policy cross-entropy against the root visit distribution, with target-temperature convention explicit. Value target is discounted final outcome z, retained as the user-selected tweak; undiscounted outcomes are the AlphaZero reference for an ablation. Do not bootstrap value targets directly from search estimates.
- engine tracks repetition history and halfmove clock throughout real games and simulations. Document claimable-draw handling and the separate training game-length cap; capped games are not evidence of rule draws.
- synchronous batches across games with frozen weights during generation. Re-test CPU versus MPS with realistic batch packing once the architecture exists.

validation and what v5 tests:
- correctness before long training: engine legal moves including castling/en passant/promotions; padding and overflow coverage; reordered/chunked moves give matching logits, value and training gradients within tolerance; one global policy normalization and one value loss per position.
- small deterministic search cases verify alternating backup perspective, selected-path-only counts, terminal handling, prior-guided first selection, root promotion and the fresh simulation budget. Test the same neural input under different draw histories: prediction can be shared, terminal/search state cannot.
- verify search-cache hits avoid forward calls, base priors survive fresh root noise, FIFO eviction leaves retained nodes usable, and new-game/weight changes clear stale work.
- compare complete bots against fixed v3/v4 checkpoints and heuristic baselines under equal move-time budgets; use multiple games/seeds and report uncertainty. Track tactical blunders/sacrifices and midgame play, not just training loss. Version comparisons change both model and search, so they do not isolate a single cause.
- main experiment: can a small policy/value transformer plus MCTS and retained search work improve play per training hour? Measure games/positions per second, leading forward FLOPs including repeated/padded chunk work, actual CPU/MPS throughput, cache hit rate/memory, inherited/fresh visits, unique expansions, fresh neural evaluations, depth and playing strength.
- keep initial experiments bounded: establish one working model/search baseline, then compare a few shallow widths and search budgets. Compare reuse on/off at equal move time; outcome-discount ablation can follow. Avoid broad tuning before a reliable baseline.

later experiments (not v5 requirements):
- v6: search improvements / Gumbel comparisons under equal move-time budgets.
- v6 KV cache: ONLY reuse fixed-token per-layer K/V between move chunks of the SAME position. Run board/state tokens WITH the first move chunk, capturing each layer's fixed-token K/V only when overflow exists. Subsequent chunks use those K/V plus their own move K/V; reuse the first pass's board value. Single-chunk positions take the ordinary path without a retained KV cache. Discard K/V after that position's evaluation; never carry it across played moves.
  - preserve masks, attention biases, residuals, normalization and weights. Using only final board features at every move layer is a different architecture. Concatenate all logits before global softmax.
  - training reuse is possible with the shared computation graph attached: accumulate gradients from every move chunk, count value loss once, do not detach or reuse across optimizer updates. Match dropout behavior if introduced. Verify logits/value/gradients against dense execution before timing or strength comparisons.
  - revisit chunk size using total forward FLOPs and measured end-to-end throughput, memory and batch packing. Smaller chunks reduce padding after fixed work is shared, but launch/hardware overhead matters. Equivalent refactoring needs no fine-tuning.
- v6 search cache: measure occupancy and later hits for predictions from discarded branches. If they occupy substantial space without reuse, compare branch-aware eviction with simple FIFO. Optimize C++ node allocation/traversal/reclamation only after profiling; node pools/free lists or an lc3-style repository are options, not requirements. Path-keyed storage still represents a tree; shared-position graphs require additional history/statistics semantics.
- possible later history experiment: condition small heads on clock/repetition while caching a history-independent trunk. This is a separate architecture change, not exact KV refactoring or a v5 commitment.
- possible v7+: Lc0-style gathering of multiple leaves from ONE tree using temporary in-flight visits (virtual loss or unscored virtual visits), deduplicate identical evaluation requests, and measure collisions/stale selection against batching across independent games. Remove reservations when results return; perform each real backup once. This is inference scheduling, not prioritized replay.
- possible alternative learning experiment: next-move prediction from records or search-generated moves. Imitation differs from self-play policy improvement; no decision to replace AlphaZero or use it for pretraining.
- async rollouts, muon, pufferlib, NNUE incremental updates, quantization and prioritized replay remain deferred. No material prior, halfkp or two-model router.
- references: https://lczero.org/dev/lc0/search/alphazero/ and https://lczero.org/dev/lc0/search/lc3/overview/
- v6 inputs (added 2026-09-15, from Monroe & Chalmers "Mastering Chess with a Transformer Model", arXiv:2409.12272): feed the 50-move counter (halfmove clock / 100) and repetition information to the model. Chessformer uses a repetition flag for each of the current and past 7 positions; start with that (last-8 flags) and tune. The engine already exports rep and clock in encode(); the model currently ignores them by design. Consequence: the prediction cache must key on search_keys() (position + history fingerprint + clock) instead of hashes(), so expect a lower hit rate. Lifts the v5 "no history inputs" restriction; the history-blind-leaf approximation goes away.
- v6 auxiliary value head: predict the self-play search root value (root Q after that move's search, mover perspective) as a SEPARATE head alongside the pure WDL result head, instead of (or in addition to) mixing it into the main target as v5.1 does with bootstrap_lambda. Chessformer trains a "q" head on the self-play reward estimate plus a short-term head on an EMA of future values (expected depth 6), with the main result head kept pure. The v5.1 buffer already stores root_value per position.

---

V6 PLAN (ranked, 2026-09-15). Supersedes the ordering of the "later experiments" list above; items there that are not repeated here stay deferred.

what v5/v5.1 established: the model learns (value sign accuracy 92% on decisive positions, policy agreement with search 21% -> 46%); a capacity probe on the 1M-position buffer (2/4/6 layers, d 256/384, 600 supervised steps) moved held-out loss by 0.02 nats with train loss == held-out loss for every size, i.e. the run is DATA- and TARGET-limited, not capacity-limited. Self-play produced ~10k games; Chessformer's small model saw 53M. Won endgames end by repetition / 50-move rule because the search cannot execute a mate and the value head has no progress signal; v5.1 stops the value head from unlearning material in those games (bootstrapped targets, resignation with false-positive audit) but does not create the mating skill. Ranking follows from that: throughput first, then the model layout that removes chunking and enables the paper's tricks, then the search/heads that give a progress signal, then measurement.

tier 1 -- throughput (measured: GPU floor ~10-15k evals/s on M5 Pro, run averages ~3k; 3-5x realistic, 10x is not without cheaper moves)
1. lazy child materialization: create a child state only when its edge is first selected (one get_children call per wave instead of ~40, nodes/move ~n_sims instead of ~1400, node_cap ~n_sims+2 so the ~6 GB of pre-zeroed arenas at 128 lanes disappears). mate-in-1 among unvisited siblings is then found by a sim, not for free -- standard AZ.
2. vectorize _select/_backup across lanes in numpy, or move descend/expand/backup into C++ with python keeping only the per-wave NN call (v3 list). measured search+engine cost ~2.5 ms/sim-wave at 64 lanes; target <1 ms.
3. async double-buffering: two lane groups; group B selects on CPU while group A's forward runs on MPS. pufferlib-shaped. the single biggest lever after 1-2.
4. cheap forward wins: n_lanes 256+, bf16 autocast for inference at large batch (measured +20% on training), drop QKV biases and LayerNorm centering (+10% in the paper). no torch.compile on MPS (measured 2.5x slower), no int8 (nothing to gain on MPS).
5. tree reuse across played moves (root promotion, in the v5 plan, never built): retained visits reduce fresh evaluations per move and deepen search at equal cost. keep inherited vs fresh visit counts separate.

tier 2 -- model layout (v6 architecture; checkpoints will not be compatible)
6. 64 square tokens (piece-or-empty embedding + learned per-square offset) with a from x to BILINEAR POLICY HEAD: Q = X Wq, K = X Wk, logits[from,to] = q_from . k_to / sqrt(d), plus 4 promotion logits from a projection of the last-rank destination keys; mask to the legal list, one softmax. Gather logits in the engine's move order, so search/buffer/loss are unchanged and the engine is untouched. Removes move tokens, chunking, overflow, max_moves; fixed T=64 (fewer tokens than today's 34+48). Reverses the v5 piece-only-token decision on purpose.
7. relative position in attention. ALWAYS: scalar relative bias b_h[sq_i, sq_j] per head (64x64 table; +0.8 pts policy accuracy in the paper for ~zero cost). BEHIND A FLAG, ablate once: Shaw et al. -- learned per-pair vectors a^Q_ij, a^K_ij, a^V_ij added to q/k/v (e_ij = (x_i Wq + a^Q_ij).(x_j Wk + a^K_ij)/sqrt(d); z_i = sum_j alpha_ij (x_j Wv + a^V_ij)); +1.0 pts over the scalar bias, worth more than a model doubling in the paper, at ~3-4x attention compute which is a few % of the forward at T=64. Both need the fixed square layout (per-sample gathers otherwise). Numbers are from 53M games; effect at small data is unmeasured.
8. inputs: halfmove clock / 100 and repetition flags for the current and past 7 positions (engine already exports rep and clock). prediction cache keys on search_keys() instead of hashes(); accept the lower hit rate.
9. heads: keep the WDL result head PURE; add an auxiliary head predicting the self-play search root value (v5.1's bootstrap_lambda mix becomes this head; root_value is already stored); add a MOVES-LEFT head predicting plies-to-end (`distance` is already stored) and use it in search as a small bonus for shorter wins / longer losses when |Q| is large -- this is the progress signal the endgame lacks and Leela's actual fix for "cannot mate with a rook". optional: short-term value head (EMA of future root values, expected depth ~6) and a soft policy head (temperature 4).

tier 3 -- search and training signal
10. FPU (first-play urgency): unvisited Q = parent Q - c_fpu * sqrt(sum of visited priors) instead of 0. at 48-96 sims the first ~32 sims are a uniform sweep; FPU is what makes low-sim search depth-first. cheap; do before spending on more sims. note root value_sum is currently stored root-mover-perspective while other nodes are parent-perspective -- make consistent first.
11. temperature schedule: tau=1 over visit counts for the first ~30 plies, then small (v5 plan; v5 applied 0.2 all game). keep Dirichlet noise.
12. Gumbel AlphaZero comparison at equal move time (already listed).
13. LR: 3e-4 is untested. run the offline LR sweep on the buffer (materialize npz arrays first) and add warmup + step decay for long runs; the paper trains at 1e-3 then decays. add stochastic weight averaging of recent checkpoints for evaluation.

tier 4 -- measurement (do alongside tier 1; "loss does not measure strength")
14. automated arena at every checkpoint against a fixed ladder (v1.1_heuristic_pst, v3@434, v5.0@100, v5.1 checkpoints) at EQUAL MOVE TIME, both colours, >=32 games, Elo with error bars written to metrics. also log per step: draw fraction by type, mean game length, material lead at game end for drawn games (the diagnostic that found the endgame problem), policy entropy, value sign accuracy on decisive positions.

draws / endgame -- status
- cause (measured): won endgames (K+Q or K+R vs K, leads of 9-24 pawns, 3-8 pieces) end by repetition or the 50-move rule because 48-96 sims cannot find a ~10-move mate and the value head sees no difference between "queen on a3" and "queen on b4". those games were then labelled draws, teaching the value head that queen-vs-king is drawn (P(win) fell 0.107 -> 0.079 between v5.0 checkpoints 25 and 100).
- done in v5.1: bootstrapped value targets (untrusted endings take the search value, not "draw"), resignation with a false-positive audit (starts off, enables itself when the audit says the value head's "lost" is reliable). evaluate at v5.1 checkpoints 200/500 with the material-at-end diagnostic before adding anything else.
- next: items 8 and 9 (clock/repetition inputs, moves-left head) are the general fixes. more data is the rest.
- deliberately NOT doing: rule-based or tablebase adjudication of won endgames (Leela's approach, rejected as hardcoding), contempt / negative draw score (wrong sign under naive negamax backup; changes target semantics), dropping draws or capped games from the buffer (throws away policy targets and all endgame positions), masking repetition in search (would desynchronise search from the engine's adjudication).

deferred / moot
- KV cache between move chunks: moot once item 6 removes chunks.
- Lc0-style multi-leaf gathering with virtual loss, muon, prioritized replay, NNUE incremental updates, int8 quantization, CoreML/graph capture (revisit once shapes are static after item 6), imitation pretraining.

lessons carried: intact checkpoints do not prove strength; loss does not measure strength; nets co-adapt to their search, so compare whole bots. Earlier notes and cut/deferred lists above are historical; this plan supersedes them for v5. A search or chunking bug can silently poison a training run.
