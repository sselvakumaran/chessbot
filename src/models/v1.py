from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F

@dataclass(frozen=True)
class Config:
  max_moves: int = 96   # per-forward move-token budget; total tokens = N_FIXED + max_moves
  d_hidden: int = 128
  n_heads: int = 4
  n_layers: int = 3
  mlp_fanout: float = 4.0

N_FIXED = 64 + 1 + 1

class Embedding(nn.Module):
  def __init__(self, config: Config):
    super().__init__()
    d_hidden = config.d_hidden

    # fused (square, piece) table for board tokens: index = square * 13 + code.
    # subsumes the old additive et_pos + et_piece (seed with their sum to
    # migrate old checkpoints) while allowing square-piece conjunctions
    self.et_board = nn.Embedding(64 * 13, d_hidden)
    self.et_piece = nn.Embedding(13, d_hidden)  # move tokens: mover piece
    # castle (plr, opp), en_passant, repetition_count, halfmove_clock, clock_base
    self.et_state = nn.Parameter(torch.empty(6, d_hidden))
    self.et_cls = nn.Parameter(torch.empty(d_hidden))

    self.et_move_from = nn.Embedding(64, d_hidden)
    self.et_move_to = nn.Embedding(64, d_hidden)
    # normal, castle, en passant, promotion (knight, bishop, rook, queen)
    self.et_move_type = nn.Embedding(7, d_hidden)
    # piece on the move's to-square: 0 = none, 7-12 = opp piece
    self.et_move_capture = nn.Embedding(13, d_hidden)
    self.out_norm = nn.LayerNorm(d_hidden)

    self.register_buffer("pos", torch.arange(64).view(1, 64), persistent=False)
    # castling-right embeddings live on the king's landing squares
    self.register_buffer("castle_square", torch.tensor([6, 2, 62, 58]), persistent=False)
    self.register_buffer("castle_bit", torch.arange(4), persistent=False)
    self.register_buffer("castle_side", torch.tensor([0, 0, 1, 1]), persistent=False)

  def forward(self,
    board: Tensor, # int (B, 64)
    castling: Tensor, # int (B,)
    en_passant_square: Tensor, # int (B,) -1 = none
    repetition_count: Tensor, # int (B,) [0, 2]
    halfmove_clock: Tensor, # int (B,) [0, 99]
    moves: Tensor, # int (B, M, 4) [from, to, mtype, piece]
  ):
    B = board.shape[0]
    board_embed = self.et_board(self.pos * 13 + board) # (B, 64, d)

    bits = ((castling.view(B, 1) >> self.castle_bit) & 1)                 # (B, 4)
    castle_add = bits.unsqueeze(-1) * self.et_state[self.castle_side]    # (B, 4, d)
    board_embed = board_embed.index_add(1, self.castle_square, castle_add)

    en_passant_mask = (self.pos == en_passant_square.view(B, 1)).unsqueeze(-1)
    board_embed = board_embed + en_passant_mask * self.et_state[2]

    clock_embed = (
      self.et_state[3] * (repetition_count.float() / 2).view(B, 1)
      + self.et_state[4] * (halfmove_clock.float() / 99).view(B, 1)
      + self.et_state[5]
    ).view(B, 1, -1) # (B, 1, d)

    cls_embed = self.et_cls.view(1, 1, -1).expand(B, 1, -1)

    # captured piece read straight off the canonical board input (0 for ep /
    # castle landing squares; ep captures are identified by move type)
    captured = board.gather(1, moves[:, :, 1])
    move_embed = (
      self.et_move_from(moves[:, :, 0])
      + self.et_move_to(moves[:, :, 1])
      + self.et_move_type(moves[:, :, 2])
      + self.et_piece(moves[:, :, 3])
      + self.et_move_capture(captured)
    )

    return self.out_norm(
      torch.cat([board_embed, clock_embed, cls_embed, move_embed], dim=1)
    )

class Attention(nn.Module):
  def __init__(self, config: Config):
    super().__init__()
    self.n_heads, self.d_hidden = config.n_heads, config.d_hidden
    if self.d_hidden % self.n_heads != 0: raise ValueError("invalid n_heads")

    self.qkv_block = nn.Linear(self.d_hidden, 3 * self.d_hidden)
    self.proj = nn.Linear(self.d_hidden, self.d_hidden)
    self.proj.__INIT_SCALAR__ = (2 * config.n_layers) ** -0.5

    # learned pairwise bias over fixed tokens (board geometry: files, ranks,
    # diagonals, knight relations). zero-init so the model starts identical to
    # the unbiased one; bare Parameter, so init_weights leaves it alone
    self.pair_bias = nn.Parameter(torch.zeros(config.n_heads, N_FIXED, N_FIXED))

  def forward(self, x: Tensor, attn_bias: Tensor):
    B, T, _ = x.size()
    d_head = self.d_hidden // self.n_heads
    qkv: Tensor = self.qkv_block(x) # (B, T, 3*d)
    q_joined, k_joined, v_joined = qkv.split(self.d_hidden, dim=-1)

    q = q_joined.view(B, T, self.n_heads, d_head).transpose(1, 2)
    k = k_joined.view(B, T, self.n_heads, d_head).transpose(1, 2)
    v = v_joined.view(B, T, self.n_heads, d_head).transpose(1, 2)

    bias = attn_bias + F.pad(
      self.pair_bias, (0, T - N_FIXED, 0, T - N_FIXED)
    ).unsqueeze(0)
    y = F.scaled_dot_product_attention(q, k, v, attn_mask=bias.to(q.dtype))
    y_joined = y.transpose(1, 2).contiguous().view(B, T, self.d_hidden)
    return self.proj(y_joined)

class MultiLayerPerceptron(nn.Module):
  def __init__(self, config: Config):
    super().__init__()
    d_hidden = config.d_hidden
    d_fanout = int(config.d_hidden * config.mlp_fanout)
    self.l1 = nn.Linear(d_hidden, d_fanout)
    self.proj = nn.Linear(d_fanout, d_hidden)
    self.proj.__INIT_SCALAR__ = (2 * config.n_layers) ** -0.5

  def forward(self, x: Tensor):
    return self.proj(self.l1(x).relu().square())

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
    self.v_head = nn.Linear(config.d_hidden, 1)

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
      torch.where(attn_allowed, 0.0, -1e9).view(1, 1, T, T),
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
    repetition_count: Tensor, # int (B,) [0, 2]
    halfmove_clock: Tensor, # int (B,) [0, 99]
    moves: Tensor, # int (B, M, 4)
    num_moves: Tensor # int (B,)
  ):
    B = board.shape[0]
    # accept raw engine dtypes (uint8/int8); embeddings want long
    board, moves = board.long(), moves.long()
    castling = castling.long()
    en_passant_square = en_passant_square.long()

    move_mask = self.move_arange < num_moves.long().view(B, 1)          # (B, M) bool
    seq_mask = torch.cat([move_mask.new_ones(B, N_FIXED), move_mask], dim=1)
    key_bias = torch.where(seq_mask, 0.0, -1e9).view(B, 1, 1, -1)       # pad keys out
    attn_bias = self.struct_bias + key_bias

    x = self.embed(
      board,
      castling,
      en_passant_square,
      repetition_count,
      halfmove_clock,
      moves,
    )
    x = x * seq_mask.unsqueeze(-1)

    for block in self.blocks:
      x = block(x, attn_bias)

    x = self.norm_f(x)
    move_x = x[:, N_FIXED:, :] # (B, M, d)
    p_logits = self.p_head(move_x).squeeze(-1).masked_fill(~move_mask, -1e9) # (B, M)
    v_out = self.v_head(x[:, N_FIXED - 1, :]).tanh()
    return (p_logits, v_out)

def _init_module(module: nn.Module, base_std: float):
  if isinstance(module, nn.Linear):
    std = base_std * getattr(module, "__INIT_SCALAR__", 1.0)
    nn.init.normal_(module.weight, mean=0.0, std=std)
    if module.bias is not None:
      nn.init.zeros_(module.bias)
  elif isinstance(module, nn.Embedding):
    nn.init.normal_(module.weight, mean=0.0, std=base_std)
  elif isinstance(module, nn.LayerNorm):
    nn.init.ones_(module.weight)
    nn.init.zeros_(module.bias)
  elif isinstance(module, Embedding):  # bare Parameters (et_state, et_cls)
    nn.init.normal_(module.et_state, mean=0.0, std=base_std)
    nn.init.normal_(module.et_cls, mean=0.0, std=base_std)

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