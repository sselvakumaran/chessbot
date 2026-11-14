# v12: v11's bot (C++ Gumbel search, double-buffered self-play, Muon) on the v12 network (models/v12_slim: 64 tokens,
# kernels/v12_metal). The search, the packed rows and the inference path are v11's; only the model and four new training
# settings differ.

from dataclasses import dataclass, field

import bots.v11_muon_metal.bot as v11
from models.v12_slim import Model
from models.v12_slim import Config as ModelConfig


@dataclass(frozen=True)
class TrainingConfig(v11.TrainingConfig):
  # new in v12; the defaults give v11's behaviour
  muon_lr_mult: float = 1.0         # Muon lr = lr * muon_lr_mult
  muon_momentum: float = 0.95
  adamw_weight_decay: float = 0.4   # embeddings + heads (the AdamW decay group); weight_decay is now the Muon matrices only
  gumbel_scale: float = 1.0         # scale of the self-play Gumbel noise (full searches, and fast ones in the opening)
  min_lr_ratio: float = 0.0         # the linear decay ends at this share of the peak lr, not at 0


@dataclass(frozen=True)
class Config(v11.Config):
  model: ModelConfig = field(default_factory=ModelConfig)
  train: TrainingConfig = field(default_factory=TrainingConfig)


class Chessbot(v11.Chessbot):
  VERSION = "v12_slim"


def build(config: Config, state: dict) -> Chessbot:
  model = Model(config.model)
  model.load_state_dict(state["model"])
  model.to(config.device)
  return Chessbot(config, model=model)
