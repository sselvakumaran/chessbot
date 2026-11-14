# v12: v11 (v8_attn_bias network on Metal kernels, Muon + AdamW) made smaller.
# - 64 tokens: the castle and ep tokens are gone (79_fable_analysis: castle token unused, ep token read in 7 of 11,476
#   positions). The network no longer sees castling rights or the ep square; the legal moves still include castling / ep.
# - the kernels are kernels/v12_metal (v11's for 64 tokens). The piece-offset tables keep v11's shape (n_heads, 14, 228);
#   only rows 0..12 and the 15 * 15 board offsets are read, the rest stays 0.
# - get_optimizer: Muon gets its own lr multiplier and momentum; the AdamW decay group (embeddings + heads) its own
#   weight decay. Defaults = v11's behaviour.
# mlp_fanout / d_policy are config values (v12.0: 1.0 / 64). Not loadable from v8 / v11 weights (other shapes).

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from dataclasses import asdict

import kernels.v12_metal as metal
import models.v8_attn_bias as v8
from models.v8_attn_bias import Config, AuxConfig  # noqa: F401  (re-exported for the bot / trainer)
from models.v11_muon_metal import MultiOptimizer, UNorMuon

N_TOKENS = 64



class Attention(v8.Attention):
  # wrapper for attention kernel (v11's, on the v12 kernels)
  def forward(self, x: Tensor, pieces: Tensor, offset_table: Tensor):
    if not metal.enabled(x) or self.d_head != metal.D_HEAD or x.shape[1] > metal.MAX_T:
      return super().forward(x, pieces, offset_table)
    if getattr(self, "_offsets", None) is None or self._offsets.device != x.device:
      self._offsets = offset_table.to(torch.int32).contiguous()
    y = metal.attention(
      self.qkv_block(x), pieces.to(torch.int32), self._offsets,
      self.piece_offset.q_table, self.piece_offset.k_table, self.n_heads, self.n_kv_heads
    )
    return self.proj(y)



class MultiLayerPerceptron(v8.MultiLayerPerceptron):
  def forward(self, x: Tensor):
    if not metal.enabled(x):
      return super().forward(x)
    return self.proj(metal.swiglu(self.gate_up(x)))



class Embedding(nn.Module):
  def __init__(self, config: Config):
    super().__init__()
    d_hidden = config.d_hidden
    # embedding: piece, square, and piece x square (general + specific); no castle / ep tokens
    self.embed_piece = nn.Embedding(13, d_hidden)
    self.embed_square = nn.Embedding(64, d_hidden)
    self.embed_piece_square = nn.Embedding(13 * 64, d_hidden)
    self.register_buffer("pos", torch.arange(64).view(1, 64), persistent=False)
    self.out_norm = nn.LayerNorm(d_hidden)


  def forward(self, board: Tensor): # long (B, 64)
    weight = self.embed_piece.weight
    if metal.enabled(weight) and weight.shape[1] == metal.D_HIDDEN and abs(self.out_norm.eps - 1e-5) < 1e-12:
      return metal.embedding(board, self)
    squares = self.pos.expand(board.shape[0], -1)
    return self.out_norm(
      self.embed_piece(board)
      + self.embed_square(squares)
      + self.embed_piece_square(64 * board + squares)
    )



class Model(nn.Module):
  def __init__(self, config: Config):
    super().__init__()
    self.config = config

    self.embed = Embedding(config)
    self.blocks = nn.ModuleList(
      [v8.TransformerBlock(config) for _ in range(config.n_layers)]
    )
    for block in self.blocks:
      block.attn.__class__ = Attention
      block.mlp.__class__ = MultiLayerPerceptron
    self.norm_f = nn.LayerNorm(config.d_hidden)

    # policy head (hidden -> d_policy x 2)
    self.policy_from = nn.Linear(config.d_hidden, config.d_policy, bias=False)
    self.policy_to = nn.Linear(config.d_hidden, config.d_policy, bias=False)
    self.policy_promotion = nn.Linear(config.d_policy, 4) # to (d_policy) -> promote (N, B, R, Q)
    self.policy_scale = pow(config.d_policy, -0.5)

    # value head (activation per square, then all squares -> intermediate representation -> WDL)
    self.value_square_proj = nn.Linear(config.d_hidden, config.d_square_value)
    self.value_full_proj = nn.Linear(N_TOKENS * config.d_square_value, config.d_full_value)
    self.value_out = nn.Linear(config.d_full_value, 3)

    self.aux_heads = nn.ModuleDict({
      name: v8.AUX_HEADS[name](config)
      for name, weight in asdict(config.aux).items() if weight > 0
    })

    # bias table: (rank offset + 7) * 15 + (file offset + 7) from query square (row) to key square (column)
    sq = torch.arange(N_TOKENS)
    d_rank = (sq[None, :] // 8 - sq[:, None] // 8) + 7
    d_file = (sq[None, :] % 8 - sq[:, None] % 8) + 7
    self.register_buffer("offset_table", d_rank * 15 + d_file, persistent=False)

    v8.init_weights(self)


  def forward(self,
    board: Tensor, # int (B, 64)
    castling: Tensor, # int (B,)  NOT read (kept so the bot / trainer / replay code is v11's)
    en_passant_square: Tensor, # int (B,)  NOT read
    moves: Tensor, # int (B, M, 4)
    num_moves: Tensor, # int (B,)
    with_aux = False
  ):
    board = board.long()
    x = self.embed(board)
    for block in self.blocks:
      x = block(x, board, self.offset_table)
    x = self.norm_f(x)

    p_from = self.policy_from(x) # (B, 64, d_policy)
    p_to = self.policy_to(x) # (B, 64, d_policy)
    p_promo = self.policy_promotion(p_to)

    p_grid = (p_from @ p_to.transpose(1, 2)) * self.policy_scale # (B, 64, 64) - d_policy middle dimension

    moves = moves.long()
    m_from, m_to, m_type = moves[..., 0], moves[..., 1], moves[..., 2]

    p_logits = p_grid.flatten(1).gather(1, N_TOKENS * m_from + m_to)
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


  def _infer_tensors(self):
    # pull all tensors `infer()` reads and cache. rebuilt when the weights change
    pf, pt = self.policy_from.weight, self.policy_to.weight
    key = (pf._version, pt._version, pf.data_ptr(), pt.data_ptr())
    cached = getattr(self, "_infer_cache", None)
    if cached is None or cached[0] != key:
      emb, blocks = self.embed, self.blocks
      tables = (emb.embed_piece.weight, emb.embed_square.weight, emb.embed_piece_square.weight,
                emb.out_norm.weight, emb.out_norm.bias, blocks[0].norm1.weight, blocks[0].norm1.bias)
      layers = []
      for i, block in enumerate(blocks):
        attn, mlp = block.attn, block.mlp
        norm = blocks[i + 1].norm1 if i + 1 < len(blocks) else self.norm_f
        layers.append((attn.qkv_block.weight, attn.qkv_block.bias, attn.piece_offset.q_table,
                       attn.piece_offset.k_table, attn.proj.weight, attn.proj.bias, block.norm2.weight,
                       block.norm2.bias, mlp.gate_up.weight, mlp.gate_up.bias, mlp.proj.weight, mlp.proj.bias,
                       norm.weight, norm.bias))
      heads = (torch.cat([pf, pt]).contiguous(), self.policy_promotion.weight, self.policy_promotion.bias,
               self.value_square_proj.weight, self.value_square_proj.bias, self.value_full_proj.weight,
               self.value_full_proj.bias, self.value_out.weight, self.value_out.bias)
      cached = (key, tables, tuple(layers), heads, self.offset_table.to(torch.int32).contiguous())
      self._infer_cache = cached
    return cached[1:]


  def can_infer(self) -> bool:
    # only can use `infer()` when MPS + bf16 weights, autograd off, and correct size of tensors
    c, attn = self.config, self.blocks[0].attn
    return (metal.enabled(self.embed.embed_piece.weight) and c.d_hidden == metal.D_HIDDEN
            and attn.d_head == metal.D_HEAD and N_TOKENS * c.d_square_value <= metal.MAX_NJ
            and c.d_full_value <= metal.MAX_DF)


  @torch.inference_mode()
  def infer(self, packed: Tensor, M: int) -> Tensor:
    tables, layers, heads, offsets = self._infer_tensors()
    B, T, D = packed.shape[0], N_TOKENS, self.config.d_hidden
    n_heads, n_kv = self.blocks[0].attn.n_heads, self.blocks[0].attn.n_kv_heads
    x = torch.empty(B, T, D, device=packed.device, dtype=tables[0].dtype)
    h = torch.empty_like(x)
    pieces = torch.empty(B, T, device=packed.device, dtype=torch.int32)
    metal.embed_fused(packed, tables, x, h, pieces)
    f = layers[0][8].shape[0] // 2
    # tensor-op kernels (macOS 26+): whole tiles only (B % 16 == 0), 256 wide, 64-dim heads, MLP width a multiple of 128
    tensor = (B % 16 == 0 and D == metal.D_HIDDEN and n_heads * metal.D_HEAD == D and f % 128 == 0
              and metal.tensor_ops() is not None)
    if tensor:
      y = torch.empty(B, T, D, device=packed.device, dtype=x.dtype)
      u = torch.empty(B, T, f, device=packed.device, dtype=x.dtype)
    for qkv_w, qkv_b, q_table, k_table, proj_w, proj_b, n2_w, n2_b, gu_w, gu_b, mp_w, mp_b, nn_w, nn_b in layers:
      if tensor:
        # qkv stays on MPS (tensor ops only tie it); the other matmuls run with their epilogues fused
        metal.attention_tensor(F.linear(h, qkv_w, qkv_b), pieces, q_table, k_table, y, n_heads, n_kv)
        metal.linear_add_layernorm(y, proj_w, proj_b, x, n2_w, n2_b, h)
        metal.gate_up_swiglu(h, gu_w, gu_b, u)
        metal.linear_add_layernorm(u, mp_w, mp_b, x, nn_w, nn_b, h)
        continue
      # using F.linear for matmuls; can't compete with PyTorch GEMM!
      y = metal.attention(F.linear(h, qkv_w, qkv_b), pieces, offsets, q_table, k_table, n_heads, n_kv)
      metal.add_layernorm(x, F.linear(y, proj_w, proj_b), n2_w, n2_b, h)
      u = metal.swiglu(F.linear(h, gu_w, gu_b))
      metal.add_layernorm(x, F.linear(u, mp_w, mp_b), nn_w, nn_b, h)

    pft_w, promo_w, promo_b, sq_w, sq_b, full_w, full_b, out_w, out_b = heads
    out = torch.empty(B, M + 1, device=packed.device, dtype=torch.float32)
    metal.policy_head(F.linear(h, pft_w), packed, promo_w, promo_b, self.policy_scale, out, M)
    metal.value_head(F.linear(h, sq_w, sq_b), full_w, full_b, out_w, out_b, out, M)
    return out



def get_optimizer(
  model: Model,
  weight_decay: float,           # Muon matrices (the blocks' 2D weights)
  lr: float,
  betas=(0.9, 0.95),
  eps=1e-8,
  u_normuon=False,
  muon_lr_mult: float = 1.0,     # Muon lr = lr * muon_lr_mult (before Muon's own match_rms_adamw shape factor)
  muon_momentum: float = 0.95,
  adamw_weight_decay: float | None = None,  # AdamW decay group = embeddings + heads; None = weight_decay (v11)
) -> MultiOptimizer:
  muon_params = [
    p
    for n, p in model.named_parameters()
    if n.startswith("blocks.") and p.dim() == 2
  ]
  ids = {id(p) for p in muon_params}
  adamw_wd = weight_decay if adamw_weight_decay is None else adamw_weight_decay
  # v8's groups: [dim >= 2 -> decay, biases + LayerNorm -> 0, piece-offset tables -> 0 and lr * bias_lr_mult]
  groups = [
    {
      "params": [p for p in g["params"] if id(p) not in ids],
      **{k: v for k, v in g.items() if k != "params"}
    }
    for g in v8.get_optimizer(model, adamw_wd, lr, betas, eps).param_groups
  ]

  adamw = torch.optim.AdamW([g for g in groups if g["params"]], lr=lr, betas=betas, eps=eps)
  # adjust LR to be ~ adamW
  muon = (UNorMuon if u_normuon else torch.optim.Muon)(
    muon_params, lr=lr * muon_lr_mult, weight_decay=weight_decay, momentum=muon_momentum,
    adjust_lr_fn="match_rms_adamw"
  )
  return MultiOptimizer([adamw, muon])
