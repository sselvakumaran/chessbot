from dataclasses import replace

import numpy as np
from engine.game import GameBatch, GameResult
from run import Run
from trainer_base import TrainerBase

from .bot import Chessbot, Config


class Trainer(TrainerBase[Config]):
  def __init__(self, config: Config):
    super().__init__(config)

    search_config = replace(
      config,
      pst_weight = 0.0,
      max_depth = config.train.max_depth,
      epsilon = config.train.epsilon
    )

    self.searcher = Chessbot(
      search_config,
      pst=None
    )

    self.learned_sum = np.zeros((13, 64), np.float32)
    self.learned_w = np.zeros((13, 64), np.float32)
    self.checkmates = 0


  def _train(self, run: Run) -> None:
    training = self.config.train
    game = GameBatch(1, sim_cap=16)
    sq_idxs = np.arange(64)

    while self.step < training.num_games:
      result = GameResult.ONGOING
      trajectory = []
      game.reset_where()

      # play a full game using non-pst bot
      while (
        result == GameResult.ONGOING
        and len(trajectory) < training.max_plies
      ):
        trajectory.append(game.get_encoding()[0][0].copy())
        moves, _ = self.searcher.select_moves(game)
        result = game.play_batch(moves)[0]

      # credit assignment only on checkmate
      winner = None
      if result == int(GameResult.CHECKMATE):
        winner = int(int(game.to_moves()[0]) == 0) # 0 white 1 black
        game_len = len(trajectory)

        for ply, board in enumerate(trajectory):
          w = pow(training.discount, game_len - 1 - ply)
          outcome = w if (ply % 2) == winner else -w

          occupied = board > 0
          np.add.at(
            self.learned_sum, 
            (board[occupied], sq_idxs[occupied]), 
            outcome
          )
          np.add.at(
            self.learned_w,
            (board[occupied], sq_idxs[occupied]),
            w
          )

        self.checkmates += 1
      
      self.step += 1
      # logging stuff 
      result_name = GameResult(result).name
      run.log(
        self.step,
        game_len = len(trajectory),
        result = result_name,
        checkmates = self.checkmates
      )
      winner_name = {0: "white", 1: "black"}.get(winner, "none")
      print(
        f"game {self.step:04d}/{training.num_games:04d} | "
        f"length {len(trajectory):03d}, result {result_name}, "
        f"winner {winner_name}",
        flush=True,
      )
      interval = max(
        1,
        training.checkpoint_interval
      )
      if self.step % interval == 0:
        self.save_checkpoint(run)

  def state_dict(self):
    return {
      "pst": _make_pst(
        self.learned_sum,
        self.learned_w
      ),
      "learned_sum": self.learned_sum,
      "learned_w": self.learned_w,
      "checkmates": self.checkmates,
      "rng_state": self.searcher.rng.bit_generator.state,
    }

  def load_state_dict(self, state: dict):
    self.learned_sum[...] = state["learned_sum"]
    self.learned_w[...] = state["learned_w"]
    self.checkmates = int(state["checkmates"])
    self.searcher.rng.bit_generator.state = state["rng_state"]

def _make_pst(
  learned_sum: np.ndarray,
  learned_w: np.ndarray
):
  pst = np.where(
    learned_w > 0,
    learned_sum / np.maximum(learned_w, 1e-9),
    0.0
  )

  def rank_mirror(x):
    return np.ascontiguousarray(x.reshape(-1, 8, 8)[:, ::-1, :].reshape(-1, 64))

  player_symmetric_pst = 0.5 * (pst[1:7] - rank_mirror(pst[7:13]))
  pst[1:7] = player_symmetric_pst
  pst[7:13] = -rank_mirror(player_symmetric_pst)
  pst = 0.5 * (pst + pst.reshape(13, 8, 8)[:, :, ::-1].reshape(13, 64))
  # center out
  pst[1:] -= pst[1:].mean(axis=1, keepdims=True)
  pst[0] = 0.0

  return pst.astype(np.float32)
