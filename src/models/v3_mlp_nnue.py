# TODO: review
# changes: rewrote v2_mlp_onehot into the nnue-shaped model from the changelog.
# feature engineering is its own function (features -> index + mask lists);
# model is embed (gather * mask -> sum per frame) + tiny clipped-relu head.
# forward keeps the v2 (logits, value) interface but now takes castling + ep.
# ideas: halfkp = change the index formula in features() only; a cpu/quantized
# path later reuses embed()'s output contract (head() doesn't care where the
# summed vector came from).
#
# NNUE-inspired improvements:
# 1. remove considering 'empty' values - only embed where pieces are
# 2. use an embedding as input: put more parameters here, is much faster than the MLP end
# 3. add en-passant and castling to features (not as necessary, but can help)

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.optim import AdamW

# feature table layout (shared by both frames)
N_PIECE = 64 * 12   # (square, piece) for occupied squares only
N_CASTLE = 4        # rights: mover K, mover Q, opponent K, opponent Q
N_EP = 8            # en-passant file
N_FEAT = N_PIECE + N_CASTLE + N_EP
N_SLOTS = 64 + N_CASTLE + 1  # feature slots per frame (board slots masked by occupancy)


@dataclass(frozen=True)
class Config:
  d_embed: int = 256
  d_head: int = 32


def features(board: Tensor, castling: Tensor, ep: Tensor):
  """Engine encode outputs -> active-feature lists for both frames.

  board (B, 64) mailbox codes 0..12, stm-canonical; castling (B,) 4 bits
  (0-1 mover, 2-3 opponent); ep (B,) canonical square or -1.
  Returns idx (B, 2, N_SLOTS) long, mask (B, 2, N_SLOTS) float.
  Frame 0 = side to move as encoded; frame 1 = opponent's view
  (rank-flipped squares, colors swapped).
  """
  B = board.shape[0]
  device = board.device
  board = board.long()
  castling, ep = castling.long(), ep.long()

  sq = torch.arange(64, device=device)
  occupied = (board > 0).float()                    # (B, 64)
  piece = (board - 1).clamp(min=0)                  # 0..11 where occupied

  idx_stm = sq * 12 + piece                         # (B, 64)
  idx_opp = (sq ^ 56) * 12 + (piece + 6) % 12

  # castling rights: same 4 rows, opponent frame swaps the bit pairs
  bit = torch.arange(N_CASTLE, device=device)
  rights = ((castling.unsqueeze(-1) >> bit) & 1).float()      # (B, 4)
  idx_castle = (N_PIECE + bit).expand(B, N_CASTLE)

  # en passant: one row per file; the rank flip keeps the file
  idx_ep = (N_PIECE + N_CASTLE + (ep.clamp(min=0) & 7)).unsqueeze(-1)
  mask_ep = (ep >= 0).float().unsqueeze(-1)

  idx = torch.stack([
    torch.cat([idx_stm, idx_castle, idx_ep], dim=1),
    torch.cat([idx_opp, idx_castle, idx_ep], dim=1),
  ], dim=1)
  mask = torch.stack([
    torch.cat([occupied, rights, mask_ep], dim=1),
    torch.cat([occupied, rights[:, [2, 3, 0, 1]], mask_ep], dim=1),
  ], dim=1)
  return idx, mask


class Model(nn.Module):
  def __init__(self, config: Config):
    super().__init__()
    self.config = config

    self.table = nn.Embedding(N_FEAT, config.d_embed)
    self.bias = nn.Parameter(torch.empty(config.d_embed))
    self.l1 = nn.Linear(2 * config.d_embed, config.d_head)
    self.l2 = nn.Linear(config.d_head, config.d_head)
    self.l3 = nn.Linear(config.d_head, 3)

    init_weights(self)

  def embed(self, idx: Tensor, mask: Tensor) -> Tensor:
    """Sum of active feature rows per frame -> (B, 2, d_embed)."""
    return (self.table(idx) * mask.unsqueeze(-1)).sum(dim=2) + self.bias

  def head(self, x: Tensor):
    # clipped relu throughout: bounded activations quantize trivially later
    x = x.flatten(1).clamp(0.0, 1.0)
    x = self.l1(x).clamp(0.0, 1.0)
    x = self.l2(x).clamp(0.0, 1.0)
    logits = self.l3(x)
    probs = logits.softmax(-1)
    value = probs[:, 0] - probs[:, 2]
    return logits, value

  def forward(self, board: Tensor, castling: Tensor, ep: Tensor):
    idx, mask = features(board, castling, ep)
    return self.head(self.embed(idx, mask))


def init_weights(model: Model):
  # sum of ~30 rows lands around std 0.25; bias 0.5 centers the clamp window
  nn.init.normal_(model.table.weight, mean=0.0, std=0.05)
  nn.init.constant_(model.bias, 0.5)
  for layer in (model.l1, model.l2):
    nn.init.kaiming_normal_(layer.weight, mode="fan_in", nonlinearity="relu")
    nn.init.zeros_(layer.bias)
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
