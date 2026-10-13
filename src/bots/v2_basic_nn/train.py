from dataclasses import replace

import numpy as np
import torch

from engine.game import GameBatch, GameResult
from models.v2_mlp_onehot import Model, get_optimizer
from run import Run
from trainer_base import TrainerBase

from .bot import Chessbot, Config

class Trainer(TrainerBase[Config]):
  def __init__(self, config: Config):
    super().__init__(config)

    torch.manual_seed(config.seed)
    search_config = replace(
      config,
      temperature=config.train.temperature
    )

    self.model = Model(config.model).to(config.device)
    self.optimizer = get_optimizer(
      self.model, 
      config.train.weight_decay,
      config.train.lr,
      config.train.betas,
      config.train.eps
    )
    self.searcher = Chessbot(search_config, self.model)

  def _train(self, run: Run) -> None:
    t_config = self.config.train
    game = GameBatch(1, sim_cap=16) # just to run through the game

    while self.step < t_config.max_steps:
      result = GameResult.ONGOING
      trajectory = []
      search_nodes = 0
      game.reset_where()

      # play game with searcher 
      while(
        result == GameResult.ONGOING
        and len(trajectory) < t_config.max_plies
      ):
        trajectory.append(game.get_encoding()[0][0].copy())
        moves, info = self.searcher.select_moves(game)
        search_nodes += int(info["nodes"][0])
        result = game.play_batch(moves)[0]
      
      game_len = len(trajectory)
      is_checkmate = result == int(GameResult.CHECKMATE)
      loss_item = None
      if is_checkmate:

        targets = []
        for ply in range(game_len):
          distance = game_len - 1 - ply
          g = pow(t_config.discount, distance)

          targets.append(
            [g, 1.0-g, 0.0]
            if distance % 2 == 0 else 
            [0, 1.0-g, g]
          )

        boards = torch.from_numpy(
          np.stack(trajectory)
        ).to(self.config.device)
        targets = torch.tensor(
          targets,
          dtype=torch.float32,
          device=self.config.device
        )

        self.model.train()
        logits, _ = self.model(boards)
        loss = -(targets * logits.log_softmax(-1)).sum(dim=-1).mean()

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        self.model.eval()
        loss_item = loss.item()

      self.step += 1
      result_name = "CAPPED" \
        if result == int(GameResult.ONGOING) \
        else GameResult(result).name
      run.log(
        self.step,
        loss = loss_item,
        game_len = game_len,
        result = result_name,
        search_nodes = search_nodes,
      )

      loss_text = "-" if loss_item is None else f"{loss_item:.4f}"
      nodes_per_move = search_nodes / game_len if game_len else 0.0
      print(
        f"step {self.step:05d}/{t_config.max_steps:05d} | "
        f"result {result_name:<15} | plies {game_len:03d} | "
        f"loss {loss_text:>7} | nodes/move {nodes_per_move:,.0f}",
        flush=True,
      )

      if self.step % t_config.checkpoint_interval == 0:
        self.save_checkpoint(run)

  def state_dict(self):
    return {
      "model": self.model.state_dict(),
      "optimizer": self.optimizer.state_dict(),
      "rng_state": self.searcher.rng.bit_generator.state
    }

  def load_state_dict(self, state: dict) -> None:
    self.model.load_state_dict(state["model"])
    self.optimizer.load_state_dict(state["optimizer"])
    self.searcher.rng.bit_generator.state = state["rng_state"]
