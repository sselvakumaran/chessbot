# v11: the v8_attn_bias network with Metal kernels on its inference path (kernels/v11_metal): fused embedding
# (+ LayerNorm), fused attention (piece-offset bias computed on the fly, no transpose) and fused SwiGLU.
# equivalent to v8, just BOOSTED on metal. can be loaded from v8 weights.
# Model.infer: the whole inference forward on kernels from packed, reduces kernel dispatching; on macOS 26+ the
# matmuls + epilogues run on Metal 4 tensor ops (kernels/v11_metal/tensor_ops.metal), ~1.5x faster
# get_optimizer: Muon on the transformer blocks' 2D weights otherwise AdamW

import torch
import torch.nn.functional as F
import torch.optim._muon as muon_impl
from torch import Tensor

import kernels.v11_metal as metal
import models.v8_attn_bias as v8
from models.v8_attn_bias import get_optimizer as get_adamw
from models.v8_attn_bias import Config, AuxConfig  # noqa: F401  (re-exported for the bot / trainer)



class Attention(v8.Attention):
  # wrapper for attention kernel
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



class Embedding(v8.Embedding):
  def forward(self, board: Tensor, castling: Tensor, ep_square: Tensor):
    weight = self.embed_piece.weight
    if not metal.enabled(weight) or weight.shape[1] != metal.D_HIDDEN or abs(self.out_norm.eps - 1e-5) > 1e-12:
      return super().forward(board, castling, ep_square)
    return metal.embedding(board, castling, ep_square, self)



class Model(v8.Model):
  def __init__(self, config: Config):
    super().__init__(config)
    # swap the embedding / attention / MLP classes in place
    self.embed.__class__ = Embedding
    for block in self.blocks:
      block.attn.__class__ = Attention
      block.mlp.__class__ = MultiLayerPerceptron


  def _infer_tensors(self):
    # pull all tensors `infer()` reads and cache. rebuilt when the weights change
    pf, pt = self.policy_from.weight, self.policy_to.weight
    key = (pf._version, pt._version, pf.data_ptr(), pt.data_ptr())
    cached = getattr(self, "_infer_cache", None)
    if cached is None or cached[0] != key:
      emb, blocks = self.embed, self.blocks
      tables = (emb.embed_piece.weight, emb.embed_square.weight, emb.embed_piece_square.weight,
                emb.embed_castle.weight, emb.embed_ep.weight, emb.out_norm.weight, emb.out_norm.bias,
                blocks[0].norm1.weight, blocks[0].norm1.bias)
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
            and attn.d_head == metal.D_HEAD and 66 * c.d_square_value <= metal.MAX_NJ
            and c.d_full_value <= metal.MAX_DF)


  @torch.inference_mode()
  def infer(self, packed: Tensor, M: int) -> Tensor:
    tables, layers, heads, offsets = self._infer_tensors()
    B, T, D = packed.shape[0], 66, self.config.d_hidden
    n_heads, n_kv = self.blocks[0].attn.n_heads, self.blocks[0].attn.n_kv_heads
    x = torch.empty(B, T, D, device=packed.device, dtype=tables[0].dtype)
    h = torch.empty_like(x)
    pieces = torch.empty(B, T, device=packed.device, dtype=torch.int32)
    metal.embed_fused(packed, tables, x, h, pieces)
    f = layers[0][8].shape[0] // 2
    # tensor-op kernels (macOS 26+): whole tiles only (B % 16 == 0), 256 wide, 64-dim heads
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
    # the two non-square tokens' policy rows are computed but never read
    metal.policy_head(F.linear(h, pft_w), packed, promo_w, promo_b, self.policy_scale, out, M)
    metal.value_head(F.linear(h, sq_w, sq_b), full_w, full_b, out_w, out_b, out, M)
    return out


# wrapper to manage multiple optimizers

class MultiOptimizer:
  def __init__(self, optimizers: list[torch.optim.Optimizer]):
    self.optimizers = optimizers


  @property
  def param_groups(self):
    return [g for o in self.optimizers for g in o.param_groups]


  def zero_grad(self, set_to_none=True):
    for o in self.optimizers:
      o.zero_grad(set_to_none=set_to_none)


  def step(self):
    for o in self.optimizers:
      o.step()


  def state_dict(self):
    return [o.state_dict() for o in self.optimizers]


  def load_state_dict(self, states):
    for o, s in zip(self.optimizers, states):
      o.load_state_dict(s)


# Muon x AdamW
class UNorMuon(torch.optim.Muon):
  """torch Muon + U-NorMuon (Aurora, arXiv 2606.27715): after Newton-Schulz, every row of a tall matrix's update gets the
  same RMS, then the update is scaled back to Muon's norm. Without it a row whose gradient is ~0 gets a ~0 update while
  weight decay keeps shrinking it, so SwiGLU units die (26% of v11.9's). Stateless; same state dict as torch Muon."""

  @torch.no_grad()
  def step(self, closure=None):
    for g in self.param_groups:
      lr, wd, momentum = g["lr"], g["weight_decay"], g["momentum"]
      for p in g["params"]:
        if p.grad is None:
          continue
        state = self.state[p]
        if "momentum_buffer" not in state:
          state["momentum_buffer"] = torch.zeros_like(p.grad)
        buf = state["momentum_buffer"]
        buf.lerp_(p.grad, 1 - momentum)
        update = p.grad.lerp(buf, momentum) if g["nesterov"] else buf
        update = muon_impl._zeropower_via_newtonschulz(update, g["ns_coefficients"], g["ns_steps"], g["eps"]).to(p.dtype)
        if p.shape[0] > p.shape[1]:   # tall only: rows of a wide matrix are already balanced by the polar factor
          rows = update / (update.float().pow(2).mean(1).sqrt()[:, None].to(update.dtype) + 1e-8)
          update = rows * (update.norm() / rows.norm().clamp(min=1e-12))
        p.mul_(1 - lr * wd)
        p.add_(update, alpha=-muon_impl._adjust_lr(lr, g["adjust_lr_fn"], p.shape))


def get_optimizer(
  model: v8.Model, 
  weight_decay: float, 
  lr: float, 
  betas=(0.9, 0.95),
  eps=1e-8,
  u_normuon=False
) -> MultiOptimizer:
  muon_params = [
    p 
    for n, p in model.named_parameters()
    if n.startswith("blocks.") and p.dim() == 2  
  ]
  ids = {id(p) for p in muon_params}
  groups = [
    {
      "params": [p for p in g["params"] if id(p) not in ids],
      **{k: v for k, v in g.items() if k != "params"}
    }
    for g in get_adamw(model, weight_decay, lr, betas, eps).param_groups
  ]

  adamw = torch.optim.AdamW([g for g in groups if g["params"]], lr=lr, betas=betas, eps=eps)
  # adjust LR to be ~ adamW
  muon = (UNorMuon if u_normuon else torch.optim.Muon)(
    muon_params, lr=lr, weight_decay=weight_decay, adjust_lr_fn="match_rms_adamw"
  )
  return MultiOptimizer([adamw, muon])
