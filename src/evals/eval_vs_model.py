"""Evaluate one v1 checkpoint against older ones via head-to-head matches.

Plays N_GAMES games per opponent (half with the challenger as White, half as
Black) using Gumbel search at N_SIMS / K for both sides, and reports the
challenger's W/D/L and score.

    python evals/eval_vs_model.py
    CHALLENGER=8500 OPPONENTS=7500,5000,2500 N_GAMES=64 python evals/eval_vs_model.py
"""
import os
import time
import numpy as np

from arena import load_evaluator, ModelPlayer, play_match, print_backend

N_GAMES = int(os.environ.get("N_GAMES", 64))      # ~32 per side
N_SIMS = int(os.environ.get("N_SIMS", 16))
K = int(os.environ.get("K", 8))
MAX_PLIES = int(os.environ.get("MAX_PLIES", 300))
SEED = int(os.environ.get("SEED", 0))
CHALLENGER = int(os.environ.get("CHALLENGER", 8500))
OPPONENTS = [int(x) for x in os.environ.get("OPPONENTS", "7500,5000,2500").split(",")]


def main():
    print_backend()
    print(f"challenger step {CHALLENGER} vs {OPPONENTS}  "
          f"({N_GAMES} games each, n_sims={N_SIMS}, k={K}, max_plies={MAX_PLIES})",
          flush=True)

    chal_eval = load_evaluator(CHALLENGER, N_GAMES)
    rows = []
    for opp in OPPONENTS:
        print(f"\n=== {CHALLENGER} vs {opp} ===", flush=True)
        opp_eval = load_evaluator(opp, N_GAMES)
        rng = np.random.default_rng(SEED)
        challenger = ModelPlayer(chal_eval, N_SIMS, K, rng, name=f"step{CHALLENGER}")
        opponent = ModelPlayer(opp_eval, N_SIMS, K, rng, name=f"step{opp}")

        t0 = time.time()
        wins, losses, draws = play_match(challenger, opponent, N_GAMES, MAX_PLIES)
        dt = time.time() - t0

        score = (wins + 0.5 * draws) / N_GAMES
        print(f"  step {CHALLENGER} vs step {opp}:  "
              f"W {wins}  D {draws}  L {losses}   "
              f"score {score:.3f}  ({dt:.0f}s)", flush=True)
        rows.append((opp, wins, draws, losses, score))

    print(f"\n===== summary: challenger step {CHALLENGER} (score from its POV) =====", flush=True)
    print(f"  {'opponent':>10}  {'W':>3} {'D':>3} {'L':>3}  {'score':>6}", flush=True)
    for opp, w, d, l, s in rows:
        print(f"  {('step'+str(opp)):>10}  {w:>3} {d:>3} {l:>3}  {s:>6.3f}", flush=True)


if __name__ == "__main__":
    main()
