## transformer based version, attempt to keep throughput high.

from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F



@dataclass
class Config:
  d_hidden: int = 256
  n_heads: int = 4
  n_layers: int = 2
  mlp_fanout: float = 8 / 3
  d_policy: int = 128
  d_square_value: int = 32 # per-square dimension of value representation
  d_full_value: int = 128


N_SQUARES = 64
N_TOKENS = 64 + 2



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



class Attention(nn.Module):
  def __init__(self, config: Config):
    super().__init__()
    self.n_heads, self.d_hidden = config.n_heads, config.d_hidden
    if self.d_hidden % self.n_heads != 0: raise ValueError("invalid n_heads")

    self.qkv_block = nn.Linear(self.d_hidden, 3 * self.d_hidden)
    self.proj = nn.Linear(self.d_hidden, self.d_hidden)
    self.proj.__INIT_SCALAR__ = (2 * config.n_layers) ** -0.5
    # add new relative bias (RoPE-like) token x token, per head
    self.rel_bias = nn.Parameter(torch.zeros(config.n_heads, N_TOKENS, N_TOKENS))


  def forward(self, x: Tensor, attn_bias: Tensor = None):
    B, T, _ = x.size()
    d_head = self.d_hidden // self.n_heads
    qkv: Tensor = self.qkv_block(x) # (B, T, 3*d)
    q_joined, k_joined, v_joined = qkv.split(self.d_hidden, dim=-1)

    q = q_joined.view(B, T, self.n_heads, d_head).transpose(1, 2)
    k = k_joined.view(B, T, self.n_heads, d_head).transpose(1, 2)
    v = v_joined.view(B, T, self.n_heads, d_head).transpose(1, 2)

    bias = self.rel_bias.unsqueeze(0).to(q.dtype)
    if attn_bias is not None:
      bias = bias + attn_bias
    y = F.scaled_dot_product_attention(
      q, k, v, 
      attn_mask=bias,
      dropout_p=0.0,
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


  def forward(self, x: Tensor, attn_bias: Tensor = None):
    x = x + self.attn(self.norm1(x), attn_bias)
    x = x + self.mlp(self.norm2(x))
    return x



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

    init_weights(self)


  def forward(self,
    board: Tensor, # int (B, 64)
    castling: Tensor, # int (B,)
    en_passant_square: Tensor, # int (B,) -1 = none
    moves: Tensor, # int (B, M, 4)
    num_moves: Tensor # int (B,)
  ):
    board = board.long()
    castling = castling.long()
    en_passant_square = en_passant_square.long()

    x = self.embed(
      board,
      castling,
      en_passant_square
    )

    for block in self.blocks:
      x = block(x)

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

    return p_logits, (wdl[:, 0] - wdl[:, 2]).unsqueeze(-1), wdl_logits



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

  fn_gets_decay = lambda n, p: (
    p.requires_grad
    and p.dim() >= 2
    and ("rel_bias" not in n)
  )  

  decay = [p for n, p in model.named_parameters() if fn_gets_decay(n, p)]
  nodecay = [p for n, p in model.named_parameters() if not fn_gets_decay(n, p)]

  return torch.optim.AdamW(
    [{"params": decay, "weight_decay": weight_decay},
     {"params": nodecay, "weight_decay": 0.0}],
    lr=lr, betas=betas, eps=eps,
  )