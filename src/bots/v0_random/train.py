from run import Run
from trainer_base import TrainerBase

from .bot import Config

# really no training here!
class Trainer(TrainerBase[Config]):
  def _train(self, run: Run) -> None:
    pass

  def state_dict(self) -> dict:
    return {}

  def load_state_dict(self, state: dict) -> None:
    pass