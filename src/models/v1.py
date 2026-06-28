from tinygrad import Tensor, nn
from tinygrad.nn.state import get_state_dict
from dataclasses import dataclass

# DISABLED Q HEAD!

@dataclass(frozen=True)
class Config:
  max_moves: int = 96   # per-forward move-token budget; total tokens = N_FIXED + max_moves
  d_hidden: int = 128
  n_heads: int = 4
  n_layers: int = 3
  mlp_fanout: float = 4.0

N_FIXED = 64 + 1 + 1

class Embedding:
  def __init__(self, config: Config):
    # block_size = N_FIXED + config.max_moves
    d_hidden = config.d_hidden

    self.et_pos = nn.Embedding(64, d_hidden)
    self.et_piece = nn.Embedding(13, d_hidden)
    # castle (plr, opp), en_passant, repetition_count, halfmove_clock, clock_base
    self.et_state = Tensor.glorot_uniform(6, d_hidden)
    self.et_cls = Tensor.glorot_uniform(d_hidden)

    self.et_move_from = nn.Embedding(64, d_hidden)
    self.et_move_to = nn.Embedding(64, d_hidden)
    # normal, castle, en passant, promotion (knight, bishop, rook, queen)
    self.et_move_type = nn.Embedding(7, d_hidden)
    self.out_norm = nn.LayerNorm(d_hidden)

  def __call__(self,
    board: Tensor, # int (B, 64)
    castling: Tensor, # int (B,)
    en_passant_square: Tensor, # int (B,) -1 = none
    repetition_count: Tensor, # int (B,) [0, 2]
    halfmove_clock: Tensor, # int (B,) [0, 99]
    moves: Tensor, # int (B, M, 4) [from, to, mtype, piece]
  ):
    B = board.shape[0]
    pos = Tensor.arange(64).reshape(1, 64)
    board_embed = self.et_pos(pos) + self.et_piece(board) # (B, 64, d)
    
    bits = [(castling >> i) & 1 for i in range(4)]
    for bit, s, side in zip(
      bits,
      [6, 2, 62, 58],
      [0, 0, 1, 1]
    ):
      mask = (bit.reshape(B, 1) * (pos == s)).unsqueeze(-1)
      board_embed = board_embed + mask * self.et_state[side]

    en_passant_mask = (pos == en_passant_square.reshape(B, 1)).unsqueeze(-1)

    board_embed = board_embed + en_passant_mask * self.et_state[2]

    clock_embed = (
      self.et_state[3] * (repetition_count / 2).reshape(B, 1)
      + self.et_state[4] * (halfmove_clock / 99).reshape(B, 1)
      + self.et_state[5]
    ).reshape(B, 1, -1) # (B, 1, d)

    # cls embedding
    cls_embed = self.et_cls.reshape(1, 1, -1).expand(B, 1, -1)
    
    move_embed = (
      self.et_move_from(moves[:,:,0])
      + self.et_move_to(moves[:,:,1])
      + self.et_move_type(moves[:,:,2])
      + self.et_piece(moves[:,:,3])
    )

    return self.out_norm(
      Tensor.cat(board_embed, clock_embed, cls_embed, move_embed, dim=1)
    )

# ATTENTION STUFF
class Attention:
  def __init__(self, config: Config):
    self.n_heads, self.d_hidden = config.n_heads, config.d_hidden
    self.d_head = self.d_hidden // self.n_heads
    if self.d_hidden % self.n_heads != 0: raise ValueError("invalid n_heads")

    self.qkv_block = nn.Linear(self.d_hidden, 3*self.d_hidden)
    self.proj = nn.Linear(self.d_hidden, self.d_hidden)

    self.attention_scalar = pow(self.d_head, -0.5)
  
  def __call__(self, x: Tensor, attn_bias: Tensor):
    B, T, _ = x.shape
    qkv = self.qkv_block(x)
    q_joined, k_joined, v_joined = qkv.split(self.d_hidden, dim=-1)
    # (B, n_heads, T, d_head)
    q = q_joined.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
    k = k_joined.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
    v = v_joined.view(B, T, self.n_heads, self.d_head).transpose(1, 2) 

    y = q.scaled_dot_product_attention(k, v, attn_mask=attn_bias)
    y = y.transpose(1, 2).reshape(B, T, self.d_hidden)
    return self.proj(y)

class MultiLayerPerceptron:
  def __init__(self, config: Config):
    d_hidden = config.d_hidden
    d_fanout = int(config.d_hidden * config.mlp_fanout)
    self.l1 = nn.Linear(d_hidden, d_fanout)
    self.proj = nn.Linear(d_fanout, d_hidden)
  
  def __call__(self, x: Tensor):
    return self.proj(
      self.l1(x)
      .relu().square()
    )

class TransformerBlock:
  def __init__(self, config: Config):
    self.norm1 = nn.LayerNorm(config.d_hidden)
    self.attn = Attention(config)
    self.norm2 = nn.LayerNorm(config.d_hidden)
    self.mlp = MultiLayerPerceptron(config)
  
  def __call__(self, x: Tensor, attn_bias: Tensor):
    x = x.add(self.attn(self.norm1(x), attn_bias))
    x = x.add(self.mlp(self.norm2(x)))
    return x

class Model:
  def __init__(self, config: Config):
    d_hidden = config.d_hidden
    max_moves = config.max_moves
    self.max_moves = max_moves

    self.embed = Embedding(config)
    self.blocks = [TransformerBlock(config) for _ in range(config.n_layers)]
    self.norm_f = nn.LayerNorm(d_hidden)
    self.p_head = nn.Linear(d_hidden, 1)
    self.v_head = nn.Linear(d_hidden, 1)
  
  def __call__(self,
    board: Tensor, # int (B, 64)
    castling: Tensor, # int (B,)
    en_passant_square: Tensor, # int (B,) -1 = none
    repetition_count: Tensor, # int (B,) [0, 2]
    halfmove_clock: Tensor, # int (B,) [0, 99]
    moves: Tensor, # int (B, M, 4)
    num_moves: Tensor # int (B, )
  ):
    M = self.max_moves
    B = board.shape[0]
    T = N_FIXED + M

    move_mask = (Tensor.arange(M).reshape(1, M) < num_moves.reshape(B, 1)).float()
    seq_mask = Tensor.ones(B, N_FIXED).cat(move_mask, dim=1)      # (B, T)

    # change attention mechanism: moves cannot attend over another;
    # board cannot attend over moves
    # (this makes splitting moves between batches less hacky)
    token_arange = Tensor.arange(T).float() # (T)
    # (batch, head, query, key)

    # all queries can attend to fixed columns
    key_fixed = (token_arange < N_FIXED).reshape(1, T)
    # move queries may attend to its own column
    is_move = (token_arange >= N_FIXED).reshape(T, 1)
    i_mtrx = (token_arange.reshape(T, 1) == token_arange.reshape(1, T)).float()
    attn_allowed = key_fixed + (is_move * i_mtrx)
    struct_bias = (1 - attn_allowed).reshape(1, 1, T, T) * -1e9
    # don't include padded
    key_bias = (1 - seq_mask).reshape(B, 1, 1, -1) * -1e9             # (B, 1, 1, T)
    attn_bias = struct_bias + key_bias

    x = self.embed(
      board, 
      castling, 
      en_passant_square, 
      repetition_count, 
      halfmove_clock, 
      moves
    )
    x = x * seq_mask.unsqueeze(-1)

    for block in self.blocks:
      x = block(x, attn_bias)
    
    x = self.norm_f(x)
    move_x = x[:, N_FIXED:, :] # (B, M, d)
    p_logits = self.p_head(move_x).squeeze(-1) + (1 - move_mask) * -1e9 # (B, M)
    v_out = self.v_head(x[:, N_FIXED - 1, :]).tanh()
    return (p_logits, v_out)  #(p_logits, q_out, v_out)

def init_weights(model: Model, config: Config, base_std=0.02):
  residual_scale = pow(2 * config.n_layers, -0.5)
  for name, t in get_state_dict(model).items():
    if "norm" in name: continue
    if name.endswith("proj.weight"):
      t.assign(Tensor.normal(*t.shape, mean=0.0, std=base_std * residual_scale))
    elif name.endswith(".bias"):
      t.assign(Tensor.zeros(*t.shape))
    elif t.ndim >= 1:
      t.assign(Tensor.normal(*t.shape, mean=0.0, std=base_std))