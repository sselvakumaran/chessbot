## transformer based version, attempt to keep throughput high.

from dataclasses import dataclass, field, asdict
from typing import NamedTuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass
class AuxConfig:
  moves_left: float = 0.1


@dataclass
class Config:
  d_hidden: int = 256
  n_heads: int = 4
  n_kv_heads: int = 2
  n_layers: int = 2
  mlp_fanout: float = 1.5 # ~halving, less important
  d_policy: int = 128
  d_square_value: int = 32 # per-square dimension of value representation
  d_full_value: int = 128
  bias_lr_mult: float = 10.0 # piece offset bias learning rate change
  aux: AuxConfig = field(default_factory=AuxConfig)


N_SQUARES = 64
N_TOKENS = 64 + 2
N_OFFSETS = 15*15 + 3 # 15*15 for relative positioning, 3 for ep + castling



class Embedding(nn.Module):
  def __init__(self, config: Config):
    super().__init__()
    d_hidden = config.d_hidden

    # embedding: use both piece, square, and piece x square (general + specific)
    self.embed_piece = nn.Embedding(13, d_hidden)
    self.embed_square = nn.Embedding(64, d_hidden) # no padding since dense
    self.embed_piece_square = nn.Embedding(13 * 64, d_hidden)
    # castle token: 4 states per player
    self.embed_castle = nn.Embedding(16, d_hidden)
    # en passant token: none + 8 files
    self.embed_ep = nn.Embedding(9, d_hidden)

    self.register_buffer("pos", torch.arange(64).view(1, 64), persistent=False)

    # need to normalize embeddings since adding
    self.out_norm = nn.LayerNorm(d_hidden)


  def forward(self,
    board: Tensor, # long (B, 64)
    castling: Tensor, # long (B,)
    ep_square: Tensor, # long (B,) 
    # NOT TAKING repetition / halfmove clock
  ):
    B = board.shape[0]

    squares = self.pos.expand(B, -1)
    # now no mask
    board_tokens = (
      self.embed_piece(board)
      + self.embed_square(squares)
      + self.embed_piece_square(64 * board + squares)
    )

    castling_token = self.embed_castle(castling).unsqueeze(1)
    ep_token = self.embed_ep(
      torch.where(ep_square < 0, 0, (ep_square % 8)+ 1)
    ).unsqueeze(1)

    out_embed = self.out_norm(
      torch.cat([board_tokens, castling_token, ep_token], dim=1)
    )

    return out_embed



class PieceOffsetBias(nn.Module):
  def __init__(self, n_heads: int):
    super().__init__()
    self.q_table = nn.Parameter(torch.zeros(n_heads, 14, N_OFFSETS))
    self.k_table = nn.Parameter(torch.zeros(n_heads, 14, N_OFFSETS))
    self._maps = None # reuse at inference


  def create_maps(self, offset_table: Tensor):
    T, H = offset_table.shape[0], self.q_table.shape[0]
    q_maps = (
      self.q_table[:, :, offset_table]
      .permute(2, 1, 0, 3).reshape(T, 14, H*T)
    )
    k_maps = (
      self.k_table[:, :, offset_table]
      .permute(3, 1, 0, 2).reshape(T, 14, H*T)
    )
    return q_maps, k_maps



  def forward(self, pieces: Tensor, offset_table: Tensor):
    B, T = pieces.shape
    H = self.q_table.shape[0]

    if self.training or torch.is_grad_enabled():
      q_maps, k_maps = self.create_maps(offset_table)
    else:
      key = (
        self.q_table._version, self.k_table._version, 
        self.q_table.dtype,
        self.q_table.device
      )
      if self._maps is None or self._maps[0] != key:
        self._maps = (key, *self.create_maps(offset_table))
      _, q_maps, k_maps = self._maps

    onehot = F.one_hot(pieces, 14).to(self.q_table.dtype).transpose(0, 1)
    # token_idx -> piece -> h, j

    # (B, H, i, j)
    bias_q = torch.bmm(onehot, q_maps).view(T, B, H, T).permute(1, 2, 0, 3)
    bias_k = torch.bmm(onehot, k_maps).view(T, B, H, T).permute(1, 2, 3, 0)
    return bias_q + bias_k



class Attention(nn.Module):
  def __init__(self, config: Config):
    super().__init__()
    self.n_heads, self.d_hidden = config.n_heads, config.d_hidden
    self.n_kv_heads = config.n_kv_heads or config.n_heads
    if self.d_hidden % self.n_heads or self.n_heads % self.n_kv_heads:
      raise ValueError("invalid n_heads")
    self.d_head = self.d_hidden // self.n_heads

    self.qkv_block = nn.Linear(
      self.d_hidden, 
      self.d_hidden + 2 * self.n_kv_heads * self.d_head
    )
    self.proj = nn.Linear(self.d_hidden, self.d_hidden)
    self.proj.__INIT_SCALAR__ = (2 * config.n_layers) ** -0.5
    # add new relative bias (RoPE-like) token x token, per head
    self.piece_offset = PieceOffsetBias(config.n_heads)


  def forward(self, x: Tensor, pieces: Tensor, offset_table: Tensor):
    B, T, _ = x.size()
    kv = self.n_kv_heads * self.d_head
    q, k, v = self.qkv_block(x).split([self.d_hidden, kv, kv], dim=-1)

    q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
    k = k.view(B, T, self.n_kv_heads, self.d_head).transpose(1, 2)
    v = v.view(B, T, self.n_kv_heads, self.d_head).transpose(1, 2)

    bias = self.piece_offset(pieces, offset_table).to(q.dtype)
    if torch.is_grad_enabled():
      rep = self.n_heads // self.n_kv_heads
      # enable_gqa slow during training 
      y = F.scaled_dot_product_attention(
        q.reshape(B, self.n_kv_heads, rep*T, self.d_head), k, v,
        attn_mask=bias.reshape(B, self.n_kv_heads, rep*T, T),
      ).view(B, self.n_heads, T, self.d_head)
    else:
      y = F.scaled_dot_product_attention(
        q, k, v, 
        attn_mask=bias,
        dropout_p=0.0,
        enable_gqa=self.n_kv_heads < self.n_heads
      )
    y_joined = y.transpose(1, 2).contiguous().view(B, T, self.d_hidden)
    return self.proj(y_joined)



# using SwiGLU now! gating seems important
class MultiLayerPerceptron(nn.Module):
  def __init__(self, config: Config):
    super().__init__()
    d_hidden = config.d_hidden
    d_fanout = int(config.d_hidden * config.mlp_fanout)

    self.gate_up = nn.Linear(d_hidden, 2 * d_fanout)
    self.proj = nn.Linear(d_fanout, d_hidden)
    self.proj.__INIT_SCALAR__ = (2 * config.n_layers) ** -0.5


  def forward(self, x: Tensor):
    gate, up = self.gate_up(x).chunk(2, dim=-1)
    return self.proj(F.silu(gate) * up)



class TransformerBlock(nn.Module):
  def __init__(self, config: Config):
    super().__init__()
    self.norm1 = nn.LayerNorm(config.d_hidden)
    self.attn = Attention(config)
    self.norm2 = nn.LayerNorm(config.d_hidden)
    self.mlp = MultiLayerPerceptron(config)


  def forward(self, x: Tensor, pieces: Tensor, offset_table: Tensor):
    x = x + self.attn(self.norm1(x), pieces, offset_table)
    x = x + self.mlp(self.norm2(x))
    return x



# auxilary heads
class MovesLeftHead(nn.Module):
  def __init__(self, config: Config):
    super().__init__()
    self.hidden = nn.Linear(config.d_hidden, 64)
    self.out = nn.Linear(64, 1)


  def forward(self, x: Tensor):
    return self.out(
      F.silu(
        self.hidden(x.mean(1))
      )
    ).squeeze(-1)



AUX_HEADS = {
  "moves_left": MovesLeftHead
}



class Model(nn.Module):
  def __init__(self, config: Config):
    super().__init__()
    self.config = config

    self.embed = Embedding(config)
    self.blocks = nn.ModuleList(
      [TransformerBlock(config) for _ in range(config.n_layers)]
    )
    self.norm_f = nn.LayerNorm(config.d_hidden)

    # policy head (hidden -> d_policy x 2)
    self.policy_from = nn.Linear(config.d_hidden, config.d_policy, bias=False)
    self.policy_to = nn.Linear(config.d_hidden, config.d_policy, bias=False)
    self.policy_promotion = nn.Linear(config.d_policy, 4) # to (d_policy) -> promote (N, B, R, Q)
    self.policy_scale = pow(config.d_policy, -0.5)

    # value head (2D, activation per square, )
    self.value_square_proj = nn.Linear(config.d_hidden, config.d_square_value) # square embed -> square value proj, include bias
    self.value_full_proj = nn.Linear(N_TOKENS * config.d_square_value, config.d_full_value) # all square values -> intermediate representation
    self.value_out = nn.Linear(config.d_full_value, 3)

    self.aux_heads = nn.ModuleDict({
      name: AUX_HEADS[name](config)
      for name, weight in asdict(config.aux).items() if weight > 0
    })

    # bias table
    sq = torch.arange(N_SQUARES)
    d_rank = (sq[None, :] // 8 - sq[:, None] // 8) + 7
    d_file = (sq[None, :] % 8 - sq[:, None] % 8) + 7
    offset_table = torch.full((N_TOKENS, N_TOKENS), 15*15, dtype=torch.long)
    offset_table[:N_SQUARES, :N_SQUARES] = d_rank * 15 + d_file
    offset_table[N_SQUARES:, :] = 15*15 + 1
    offset_table[:, N_SQUARES:] = 15*15 + 2
    self.register_buffer("offset_table", offset_table, persistent=False)

    init_weights(self)


  def forward(self,
    board: Tensor, # int (B, 64)
    castling: Tensor, # int (B,)
    en_passant_square: Tensor, # int (B,) -1 = none
    moves: Tensor, # int (B, M, 4)
    num_moves: Tensor, # int (B,)
    with_aux = False
  ):
    board = board.long()
    castling = castling.long()
    en_passant_square = en_passant_square.long()

    pieces = F.pad(board, (0, N_TOKENS - N_SQUARES), value=13)

    x = self.embed(
      board,
      castling,
      en_passant_square
    )

    for block in self.blocks:
      x = block(x, pieces, self.offset_table)

    x = self.norm_f(x)

    squares = x[:, :N_SQUARES]

    p_from = self.policy_from(squares) # (B, 64, d_policy)
    p_to = self.policy_to(squares) # (B, 64, d_policy)
    p_promo = self.policy_promotion(p_to)

    p_grid = (p_from @ p_to.transpose(1, 2)) * self.policy_scale # (B, 64, 64) - d_policy middle dimension

    moves = moves.long()
    m_from, m_to, m_type = moves[..., 0], moves[..., 1], moves[..., 2]

    p_logits = p_grid.flatten(1).gather(1, N_SQUARES * m_from + m_to)
    p_logits = p_logits + torch.where(
      m_type >= 3,
      p_promo.flatten(1).gather(1, 4 * m_to + (m_type - 3).clamp(min=0)),
      torch.zeros_like(p_logits)
    )
    move_mask = torch.arange(
      moves.shape[1], device=moves.device
    )[None] < num_moves.long()[:, None]

    p_logits = p_logits.masked_fill(~move_mask, float("-inf"))

    v = F.silu(self.value_square_proj(x)).flatten(1)
    wdl_logits = self.value_out(F.silu(self.value_full_proj(v)))
    wdl = wdl_logits.softmax(-1)

    aux = {name: head(x) for name, head in self.aux_heads.items()} if with_aux else {}
    return p_logits, (wdl[:, 0] - wdl[:, 2]).unsqueeze(-1), wdl_logits, aux



def _init_module(module: nn.Module, base_std: float):
  if isinstance(module, nn.Linear):
    std = base_std * getattr(module, "__INIT_SCALAR__", 1.0)
    nn.init.normal_(module.weight, mean=0.0, std=std)
    if module.bias is not None:
      nn.init.zeros_(module.bias)
  elif isinstance(module, nn.Embedding):
    nn.init.normal_(module.weight, mean=0.0, std=base_std)
    if module.padding_idx is not None:
      with torch.no_grad():
        module.weight[module.padding_idx].zero_()
  elif isinstance(module, nn.LayerNorm):
    nn.init.ones_(module.weight)
    nn.init.zeros_(module.bias)



def init_weights(model: Model, base_std: float = 0.02):
  model.apply(lambda m: _init_module(m, base_std))



def get_optimizer(model: Model, weight_decay: float, lr: float,
                  betas=(0.9, 0.95), eps=1e-8):

  decay, nodecay, tables = [], [], []

  for n, p in model.named_parameters():
    if not p.requires_grad:
      continue
    if n.endswith(("piece_offset.q_table", "piece_offset.k_table")):
      tables.append(p)
    elif p.dim() >= 2:
      decay.append(p)
    else:
      nodecay.append(p)

  return torch.optim.AdamW(
    [{"params": decay, "weight_decay": weight_decay},
     {"params": nodecay, "weight_decay": 0.0},
     {"params": tables, "weight_decay": 0.0, "lr": lr * model.config.bias_lr_mult}],
    lr=lr, betas=betas, eps=eps,
  )