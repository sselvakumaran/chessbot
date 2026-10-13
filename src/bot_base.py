from abc import ABC, abstractmethod
from importlib import import_module
from pathlib import Path
from engine.game import GameBatch
from run import Run
from typing import Any

import numpy as np

class ChessbotBase(ABC):
  "simple wrapper for a chessbot"
  VERSION = "unknown"

  def select_moves(
    self, 
    batch: GameBatch, 
    mask: np.ndarray | None = None
  ) -> tuple[np.ndarray, dict[str, Any]]:
    """
    select a move from a game, returns the move's index and generic info dict
    ie. value, policy, etc.
    """
    if mask is None:
      mask = batch.active
    return self._select_moves(
      batch,
      np.asarray(mask, dtype=bool)
    )

  @abstractmethod
  def _select_moves(
    self,
    batch: GameBatch,
    mask: np.ndarray
  ) -> tuple[np.ndarray, dict[str, Any]]:
    raise NotImplementedError()

def load_bot(
  config: str | Path,
  checkpoint: str | None = None,
  *,
  configs_dir: str | Path = "configs",
  runs_dir: str | Path = "runs",
  overrides: dict[str, Any] | None = None
) -> ChessbotBase:
  with Run(
    config,
    configs_dir=configs_dir,
    runs_dir=runs_dir
  ) as run:
    module = import_module(
      f"bots.{run.version}.bot"
    )

    bot_config = run.config(module.Config, overrides=overrides)
    state = run.load_checkpoint(checkpoint)

    return module.build(bot_config, state)
