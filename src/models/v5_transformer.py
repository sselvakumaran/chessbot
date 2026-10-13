## transformer based version, attempt to keep throughput high.

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.optim import AdamW
import torch.nn.functional as F

N_FIXED = 32 + 2

@dataclass
class Config:
  max_moves: int = 48
  d_hidden: int = 256
  n_heads: int = 4
  n_layers: int = 2
  mlp_fanout: float = 8 / 3

class Embedding(nn.Module):
  def __init__(self, config: Config):
    super().__init__()
    d_hidden = config.d_hidden

    # embedding: use both piece, square, and piece x square (general + specific)
    self.embed_piece = nn.Embedding(12 + 1, d_hidden, padding_idx = 12)
    self.embed_square = nn.Embedding(64 + 1, d_hidden, padding_idx = 64)
    self.embed_piece_square = nn.Embedding(12 * 64 + 1, d_hidden, padding_idx = 12 * 64)
    # castle token: 4 states per player
    self.embed_castle = nn.Embedding(16, d_hidden)
    # en passant token: none + 8 files
    self.embed_ep = nn.Embedding(9, d_hidden)

    # seperate tables for moves
    self.embed_move_from = nn.Embedding(12 * 64 + 1, d_hidden, padding_idx = 12 * 64)
    self.embed_move_to = nn.Embedding(13 * 64 + 1, d_hidden, padding_idx = 13 * 64)
    self.embed_move_type = nn.Embedding(7 + 1, d_hidden, padding_idx = 7)
    self.embed_move_piece = nn.Embedding(6 + 1, d_hidden, padding_idx = 6)

    self.register_buffer("pos", torch.arange(64).view(1, 64), persistent=False)

    # need to normalize embeddings since adding
    self.out_norm = nn.LayerNorm(d_hidden)

  def forward(self,
    board: Tensor, # long (B, 64)
    castling: Tensor, # long (B,)
    ep_square: Tensor, # long (B,) 
    # NOT TAKING repetition / halfmove clock
    moves: Tensor, # long (B, M, 4)
    valid_move_mask: Tensor # bool (B, M)
  ):
    B = board.shape[0]
    active_board_mask = board != 0
    pos_stacked = self.pos.expand(B, -1)

    # 64 as padding token
    active_board_idxs = pos_stacked.masked_fill(
      ~active_board_mask, 64
    ).topk(32, dim=1, largest=False, sorted=False).values

    valid_piece_mask = active_board_idxs != 64
    active_board_pieces = board.gather(
      1, 
      active_board_idxs.masked_fill(~valid_piece_mask, 0)
    )

    board_tokens = (
      self.embed_piece(
        (active_board_pieces - 1).masked_fill(~valid_piece_mask, 12)
      )
      + self.embed_square(active_board_idxs)
      + self.embed_piece_square(
        (
          64 * (active_board_pieces - 1) + active_board_idxs
        ).masked_fill(~valid_piece_mask, 12 * 64)
      )
    )

    castling_token = self.embed_castle(castling).unsqueeze(1)
    ep_token = self.embed_ep(
      torch.where(ep_square < 0, 0, (ep_square % 8)+ 1)
    ).unsqueeze(1)

    # move tokens r
    move_from_idxs = moves[:, :, 0].masked_fill(~valid_move_mask, 0)
    move_to_idxs = moves[:, :, 1].masked_fill(~valid_move_mask, 0)
    move_from_pieces = board.gather(-1, move_from_idxs)
    move_to_pieces = board.gather(-1, move_to_idxs)

    move_tokens = (
      self.embed_move_from(
        (
          64 * (move_from_pieces - 1) + move_from_idxs
        ).masked_fill(~valid_move_mask, 12 * 64)
      ) + self.embed_move_to(
        (
          64 * move_to_pieces + move_to_idxs
        ).masked_fill(~valid_move_mask, 13 * 64)
      ) + self.embed_move_type(
        moves[:, :, 2].masked_fill(~valid_move_mask, 7)
      ) + self.embed_move_piece(
        (moves[:, :, 3] - 1).masked_fill(~valid_move_mask, 6)
      )
    )

    out_embed = self.out_norm(
      torch.cat([board_tokens, castling_token, ep_token, move_tokens], dim=1)
    )
    out_mask = torch.cat(
      [
        valid_piece_mask, 
        torch.ones((B, 2), dtype=torch.bool, device=board.device), 
        valid_move_mask
      ],
      dim=1
    )
    # just set padding to zero in case
    out_embed = out_embed.masked_fill(~out_mask.unsqueeze(-1), 0)

    return out_embed, out_mask


class Attention(nn.Module):
  def __init__(self, config: Config):
    super().__init__()
    self.n_heads, self.d_hidden = config.n_heads, config.d_hidden
    if self.d_hidden % self.n_heads != 0: raise ValueError("invalid n_heads")

    self.qkv_block = nn.Linear(self.d_hidden, 3 * self.d_hidden)
    self.proj = nn.Linear(self.d_hidden, self.d_hidden)
    self.proj.__INIT_SCALAR__ = (2 * config.n_layers) ** -0.5

  def forward(self, x: Tensor, attn_bias: Tensor):
    B, T, _ = x.size()
    d_head = self.d_hidden // self.n_heads
    qkv: Tensor = self.qkv_block(x) # (B, T, 3*d)
    q_joined, k_joined, v_joined = qkv.split(self.d_hidden, dim=-1)

    q = q_joined.view(B, T, self.n_heads, d_head).transpose(1, 2)
    k = k_joined.view(B, T, self.n_heads, d_head).transpose(1, 2)
    v = v_joined.view(B, T, self.n_heads, d_head).transpose(1, 2)

    y = F.scaled_dot_product_attention(
      q, k, v, 
      attn_mask=attn_bias.to(q.dtype),
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

  def forward(self, x: Tensor, attn_bias: Tensor):
    x = x + self.attn(self.norm1(x), attn_bias)
    x = x + self.mlp(self.norm2(x))
    return x

class Model(nn.Module):
  def __init__(self, config: Config):
    super().__init__()
    self.config = config
    self.max_moves = config.max_moves

    self.embed = Embedding(config)
    self.blocks = nn.ModuleList(
      [TransformerBlock(config) for _ in range(config.n_layers)]
    )
    self.norm_f = nn.LayerNorm(config.d_hidden)
    self.p_head = nn.Linear(config.d_hidden, 1)
    self.v_head = nn.Linear(config.d_hidden, 3)  # win / draw / loss logits

    # attention structure is input-independent: build it once.
    # moves cannot attend over one another; board cannot attend over moves
    # (this makes splitting moves between batches exact)
    T = N_FIXED + config.max_moves
    token_arange = torch.arange(T)
    key_fixed = (token_arange < N_FIXED).view(1, T)   # all queries see fixed keys
    is_move = (token_arange >= N_FIXED).view(T, 1)    # move queries also see themselves
    attn_allowed = key_fixed | (is_move & torch.eye(T, dtype=torch.bool))
    self.register_buffer(
      "struct_bias",
      torch.where(attn_allowed, 0.0, float("-inf")).view(1, 1, T, T),
      persistent=False,
    )
    self.register_buffer(
      "move_arange", torch.arange(config.max_moves).view(1, -1), persistent=False
    )

    init_weights(self)

  def forward(self,
    board: Tensor, # int (B, 64)
    castling: Tensor, # int (B,)
    en_passant_square: Tensor, # int (B,) -1 = none
    moves: Tensor, # int (B, M, 4)
    num_moves: Tensor # int (B,)
  ):
    B = board.shape[0]

    M = moves.shape[1]
    if M > self.max_moves:
      raise ValueError("too many moves")

    board, moves = board.long(), moves.long()
    castling = castling.long()
    en_passant_square = en_passant_square.long()

    valid_move_mask = (
      self.move_arange[:, :M]
      < num_moves.to(device=moves.device, dtype=torch.long).view(-1, 1)
    )

    x, seq_mask = self.embed(
      board,
      castling,
      en_passant_square,
      moves,
      valid_move_mask
    )

    T = x.shape[1]

    attn_bias = self.struct_bias[:, :, :T, :T].masked_fill(
      ~seq_mask[:, None, None, :],
      float("-inf")
    )

    for block in self.blocks:
      x = block(x, attn_bias)

    x = self.norm_f(x)

    move_x = x[:, N_FIXED:, :] # (B, M, d)
    p_logits = self.p_head(move_x).squeeze(-1).masked_fill(
      ~valid_move_mask, float("-inf")
    ) # (B, M)

    fixed_mask = seq_mask[:, :N_FIXED]
    fixed_count = fixed_mask.sum(dim=1, keepdim=True).to(x.dtype)

    fixed_x = x[:, :N_FIXED, :].masked_fill(~fixed_mask.unsqueeze(-1), 0)
    board_x = fixed_x.sum(dim=1) / fixed_count
    

    wdl_logits = self.v_head(board_x) # (B, 3)
    wdl = wdl_logits.softmax(-1)
    v_out = (wdl[:, 0] - wdl[:, 2]).unsqueeze(-1) # (B, 1)
    return p_logits, v_out, wdl_logits

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
  params = [p for p in model.parameters() if p.requires_grad]
  decay = [p for p in params if p.dim() >= 2]
  nodecay = [p for p in params if p.dim() < 2]
  return torch.optim.AdamW(
    [{"params": decay, "weight_decay": weight_decay},
     {"params": nodecay, "weight_decay": 0.0}],
    lr=lr, betas=betas, eps=eps,
  )