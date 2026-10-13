from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

from run import Run, RunStatus


S = TypeVar("S")


class TrainerBase(ABC, Generic[S]):
  def __init__(self, config: S):
    self.config = config
    self.step = 0

  @abstractmethod
  def state_dict(self) -> dict[str, Any]:
    raise NotImplementedError()

  @abstractmethod
  def load_state_dict(self, state: dict[str, Any]) -> None:
    raise NotImplementedError()

  @abstractmethod
  def _train(self, run: Run) -> None:
    raise NotImplementedError()

  def train(self, run: Run) -> None:
    if run.status == RunStatus.COMPLETED:
      return

    if run.status == RunStatus.INCOMPLETE:
      state = run.load_checkpoint()
      self.step = int(state.pop("step"))
      self.load_state_dict(state)
    
    try:
      self._train(run)
    except BaseException:
      self.save_checkpoint(run)
      raise

    self.save_checkpoint(run)
    run.complete()

  def save_checkpoint(self, run: Run) -> None:
    state = {
      **self.state_dict(),
      "step": self.step
    }

    run.save_checkpoint(
      str(self.step),
      state
    )
