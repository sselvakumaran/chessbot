# v6: throughput. Same model, same training targets as v5 (plain outcome
# targets, Dirichlet noise, grad clipping, lane refill). Added incrementally:
#
# 1. lazy child materialization (search/v6_mcts_lazy.py): a child state is created
#    when its edge is first selected. nodes/move ~n_sims+1; node_cap = n_sims+1.
# 2. bf16 autocast for INFERENCE only (bot._evaluate). training stays fp32
#    (measured 2x slower under autocast).
# 3. 256 lanes: inference batches into the flat part of the MPS latency curve.
# 4. vectorized select/backup: all lanes descend one level per numpy step.
# 5. hoist GameResult enum lookups out of the hot loop.
# 6. async double-buffering: two lane groups, CPU selects one while the GPU
#    evaluates the other.
#
# baseline (v5.1 config, 128 lanes x 96 sims, eager, fp32): 17.5 ms per
# sim-wave, ~109 s self-play + 12 s training per 64-wave step.
