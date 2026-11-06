from dataclasses import dataclass

from torch import Tensor, nn
from torch.optim import AdamW
import torch.nn.functional as F



@dataclass(frozen=True)
class Config:
  d_l1: int = 256
  d_l2: int = 128
  negative_slope: float = 0.0


class Model(nn.Module):
  def __init__(self, config: Config):
    super().__init__()
    self.config = config

    # input is (B, 64, 13) -> #(B, 832)

    self.l1 = nn.Linear(64*13, self.config.d_l1)
    self.l2 = nn.Linear(self.config.d_l1, self.config.d_l2)
    self.l3 = nn.Linear(self.config.d_l2, 3)

    init_weights(self)

  def forward(self,
    board: Tensor,
  ):
    x = F.one_hot(
      board.long(),
      num_classes=13,
    ).to(self.l1.weight.dtype)

    x = x.flatten(start_dim=1)

    x = F.leaky_relu(
      self.l1(x), negative_slope=self.config.negative_slope
    )
    x = F.leaky_relu(
      self.l2(x), negative_slope=self.config.negative_slope
    )
    logits = self.l3(x)
    probs = logits.softmax(-1)
    value = probs[:, 0] - probs[:, 2]
    return logits, value

def init_weights(model: Model):
  for layer in (model.l1, model.l2):
    nn.init.kaiming_normal_(
      layer.weight,
      a=model.config.negative_slope,
      mode="fan_in",
      nonlinearity="leaky_relu",
    )
    nn.init.zeros_(layer.bias)

  # The output has no ReLU, so keep its initial logits small.
  nn.init.normal_(model.l3.weight, mean=0.0, std=0.02)
  nn.init.zeros_(model.l3.bias)

def get_optimizer(
  model: Model, 
  weight_decay: float, lr: float,
  betas = (0.9, 0.95), eps=1e-8
):
  params = [p for p in model.parameters() if p.requires_grad]
  decay = [p for p in params if p.dim() >= 2]
  nodecay = [p for p in params if p.dim() < 2]
  return AdamW(
    [{"params": decay, "weight_decay": weight_decay},
     {"params": nodecay, "weight_decay": 0.0}],
     lr=lr, betas=betas, eps=eps
  )
