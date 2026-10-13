# v4 reuses v3's trainer wholesale; only the bot (search) differs, and
# self-play runs at train.max_depth instead of the inference depth.

from dataclasses import replace

from bots.v3_nnue_inspired.train import Trainer as SharedTrainer

from .bot import Chessbot, Config


class Trainer(SharedTrainer):
  Bot = Chessbot

  def _search_config(self, config: Config) -> Config:
    return replace(
      config,
      temperature=config.train.temperature,
      max_depth=config.train.max_depth,
    )
