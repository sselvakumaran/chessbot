"""Pit two chessbots against each other.

Colors are EXPLICIT on the Arena: you say who is white and who is black at
construction, and they stay that way for every game the instance plays.
Randomizing / alternating colors is the caller's job (build two Arenas, or
flip a coin before constructing one) — the class never hides who played
which side, so results are always attributable.

Usage:
    from arena import Arena
    from bot_base import load_bot

    arena = Arena(
      white=load_bot("v0_random"),
      black=load_bot("v1.0_heuristic"),
    )
    records = arena.play(n_games=32)  # one GameBatch with 32 lanes
"""
import time

import numpy as np

from engine.game import GameBatch, GameResult
from bot_base import ChessbotBase

ONGOING = int(GameResult.ONGOING)


class Arena:
  def __init__(self,
    white: ChessbotBase,
    black: ChessbotBase,
    max_plies: int = 512,   # games hitting this are recorded as "capped"
    sim_cap: int = 64,      # arena of OUR batch; bots that search deeper
                            # than this must bring their own (v1+ do)
    log_moves: bool = False,  # print every move + the bot's info dict
  ):
    self.white, self.black = white, black
    self.max_plies = max_plies
    self.sim_cap = sim_cap
    self.log_moves = log_moves

  def play_game(self) -> dict:
    """Play one game using the same batched path as play()."""
    return self._play_batch(1)[0]

  def _play_batch(self, n_games: int) -> list[dict]:
    """Play every game together in one n_games-lane GameBatch."""
    if n_games < 1:
      raise ValueError("n_games must be at least 1")

    batch = GameBatch(n_games, sim_cap=self.sim_cap)
    t0 = time.time()
    plies = np.zeros(n_games, dtype=np.int32)
    results = batch.results().copy()
    seconds = np.zeros(n_games, dtype=np.float64)

    while True:
      active = (results == ONGOING) & (plies < self.max_plies)
      if not np.any(active):
        break

      sides = batch.to_moves().copy()
      moves = np.zeros(n_games, dtype=np.int32)

      for side, bot in ((0, self.white), (1, self.black)):
        mask = active & (sides == side)
        if not np.any(mask):
          continue

        t_mv = time.time()
        selected, info = bot.select_moves(batch, mask=mask)
        selected = np.asarray(selected, dtype=np.int32)
        if selected.shape != (n_games,):
          raise ValueError(
            f"{bot.VERSION}.select_moves returned shape {selected.shape}; "
            f"expected {(n_games,)}"
          )
        moves[mask] = selected[mask]

        if self.log_moves:
          elapsed = time.time() - t_mv
          for lane in np.flatnonzero(mask):
            move = int(moves[lane])
            mv_str = str(batch[int(lane)].moves()[move])
            extras = "  ".join(
              f"{key}={value[lane] if isinstance(value, np.ndarray) else value}"
              for key, value in info.items()
            )
            print(
              f"game {lane + 1:>3} ply {plies[lane] + 1:>3} "
              f"{'W' if side == 0 else 'B'} {mv_str:<24} "
              f"({elapsed:.2f}s)  {extras}"
            )

      played = active.copy()
      turn_results = batch.play_batch(moves, mask=played)
      plies[played] += 1
      results[played] = turn_results[played]

      finished = played & (
        (results != ONGOING) | (plies >= self.max_plies)
      )
      seconds[finished] = time.time() - t0

    to_moves = batch.to_moves()
    records = []
    for lane in range(n_games):
      result = int(results[lane])
      winner = None
      if result == int(GameResult.CHECKMATE):
        # side to move after checkmate is the mated side
        winner = "black" if int(to_moves[lane]) == 0 else "white"
      records.append({
        "result": (
          GameResult(result).name if result != ONGOING else "CAPPED"
        ),
        "winner": winner,
        "plies": int(plies[lane]),
        "seconds": round(float(seconds[lane]), 2),
      })
    return records

  def play(self, n_games: int = 1, verbose: bool = True) -> list[dict]:
    records = self._play_batch(n_games)
    if verbose:
      for g, rec in enumerate(records):
        who = rec["winner"] or "-"
        print(f"game {g + 1:>3}: {rec['result']:<16} winner {who:<6} "
              f"{rec['plies']:>3} plies  ({rec['seconds']}s)")
    if verbose:
      w = sum(r["winner"] == "white" for r in records)
      b = sum(r["winner"] == "black" for r in records)
      d = len(records) - w - b
      print(f"white {self.white.VERSION} vs black {self.black.VERSION}: "
            f"+{w} ={d} -{b} (white's POV, {n_games} games)")
    return records
