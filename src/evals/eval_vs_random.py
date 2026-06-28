"""Evaluate a v1 checkpoint against a uniform-random-move bot.

Plays N_GAMES games (half with the model as White, half as Black). The model uses
Gumbel search at N_SIMS / K; the opponent picks a uniformly random legal move.
A sanity baseline: a trained model should win nearly all of these.

    python evals/eval_vs_random.py
    STEP=8500 N_GAMES=64 python evals/eval_vs_random.py
"""
import os
import time
import numpy as np

from arena import load_evaluator, ModelPlayer, RandomPlayer, play_match, print_backend

N_GAMES = int(os.environ.get("N_GAMES", 64))      # ~32 per side
N_SIMS = int(os.environ.get("N_SIMS", 16))
K = int(os.environ.get("K", 8))
MAX_PLIES = int(os.environ.get("MAX_PLIES", 300))
SEED = int(os.environ.get("SEED", 0))
STEP = int(os.environ.get("STEP", 8500))


def main():
    print_backend()
    print(f"step {STEP} vs random  "
          f"({N_GAMES} games, n_sims={N_SIMS}, k={K}, max_plies={MAX_PLIES})",
          flush=True)

    rng = np.random.default_rng(SEED)
    model = ModelPlayer(load_evaluator(STEP, N_GAMES), N_SIMS, K, rng, name=f"step{STEP}")
    opponent = RandomPlayer(rng)

    t0 = time.time()
    wins, losses, draws = play_match(model, opponent, N_GAMES, MAX_PLIES)
    dt = time.time() - t0

    score = (wins + 0.5 * draws) / N_GAMES
    print(f"\nstep {STEP} vs random:  W {wins}  D {draws}  L {losses}   "
          f"score {score:.3f}  ({dt:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
