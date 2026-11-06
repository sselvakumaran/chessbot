# Metal kernels for v11 compiled at runtime.
# can only be used when enabled() applies.

from pathlib import Path
import torch

MAX_T, D_HEAD, LANES, D_HIDDEN = 72, 64, 2, 256      # must match kernels.metal
MAX_NJ, MAX_DF = 66 * 32, 128                        # value_head limits (kernels.metal)
_lib = None
_tensor_lib = None                                   # tensor_ops.metal; False if it doesn't compile (needs macOS 26+)


def _kernels():
  global _lib
  if _lib is None:
    _lib = torch.mps.compile_shader((Path(__file__).parent / "kernels.metal").read_text())
  return _lib


def tensor_ops():
  """tensor_ops.metal (Metal 4 tensor ops) or None when this OS / GPU can't compile it."""
  global _tensor_lib
  if _tensor_lib is None:
    try:
      _tensor_lib = torch.mps.compile_shader((Path(__file__).parent / "tensor_ops.metal").read_text())
    except Exception:
      _tensor_lib = False
  return _tensor_lib or None


def enabled(x: torch.Tensor) -> bool:
  return x.device.type == "mps" and x.dtype == torch.bfloat16 and not torch.is_grad_enabled()


def attention(qkv, pieces, offsets, q_table, k_table, n_heads: int, n_kv_heads: int):
  """qkv (B, T, H*D + 2*KV*D) -> attention output (B, T, H*D), pieces (B, T), offsets (T, T) as int32"""
  B, T, _ = qkv.shape
  out = torch.empty(B, T, n_heads * D_HEAD, device=qkv.device, dtype=qkv.dtype)
  # one threadgroup per (position, kv head), serving the n_heads / n_kv_heads query heads that share it
  _kernels().attention(
    qkv.contiguous(), pieces, offsets, q_table, k_table, out,
    T, n_heads, n_kv_heads, q_table.shape[-1],
    threads=B * n_heads * T * LANES, group_size=(n_heads // n_kv_heads) * T * LANES
  )
  return out


def swiglu(gate_up):
  """(..., 2F) -> silu(first half) * second half, (..., F)."""
  f = gate_up.shape[-1] // 2
  out = torch.empty(
    *gate_up.shape[:-1], f, device=gate_up.device, dtype=gate_up.dtype
  )
  _kernels().swiglu(
    gate_up.contiguous(), out, f, threads=out.numel()
  )
  return out


def embedding(board, castling, ep_square, emb):
  """board (B, 64), castling (B,), ep_square (B,) as int64 + a v8 Embedding module -> LayerNormed tokens (B, 66, D_HIDDEN)."""
  B = board.shape[0]
  out = torch.empty(B, 66, D_HIDDEN, device=board.device, dtype=emb.embed_piece.weight.dtype)
  _kernels().embedding(
    board.contiguous(), castling.contiguous(), ep_square.contiguous(),
    emb.embed_piece.weight, emb.embed_square.weight, emb.embed_piece_square.weight,
    emb.embed_castle.weight, emb.embed_ep.weight, emb.out_norm.weight, emb.out_norm.bias, out, 66,
    threads=B * 66 * 32, group_size=256
  )
  return out



# ---- fused inference (Model.infer): the forward from the packed uint8 rows of bots.v11_muon_metal.policy.pack
# (board 0..63, castling 64, ep + 1 at 65, move count 66, moves from 67) to move probabilities + value

def embed_fused(packed, tables, x, h, pieces):
  """tables = (piece, square, piece_square, castle, ep embeddings, out_norm weight, bias, block 0 norm1 weight,
  bias): tokens + out_norm into x (B, 66, D_HIDDEN), norm1(x) into h, int32 piece ids (13 = castle / ep) into pieces."""
  B, T = x.shape[0], x.shape[1]
  _kernels().embed_fused(packed, *tables, x, h, pieces, T, packed.shape[1], threads=B * T * 32, group_size=256)


def add_layernorm(x, y, weight, bias, h):
  """x += y (in place), h = LayerNorm(x)."""
  _kernels().add_layernorm(x, y, weight, bias, h, threads=x.shape[0] * x.shape[1] * 32, group_size=256)


def policy_head(pft, packed, weight, bias, scale, out, M):
  """legal-move logits from pft = [p_from | p_to] (B, T, 2 * d_policy) and the promotion head (weight, bias),
  softmax -> out[:, :M]."""
  B, T = pft.shape[0], pft.shape[1]
  _kernels().policy_head(pft, packed, weight, bias, out, M, packed.shape[1], T, pft.shape[-1] // 2, float(scale),
                         threads=B * M, group_size=M)


def value_head(sq, full_w, full_b, out_w, out_b, out, M):
  """value MLP from sq = value_square_proj(x) (B, T, d_square_value), WDL softmax, p_win - p_loss -> out[:, M]."""
  B, T = sq.shape[0], sq.shape[1]
  _kernels().value_head(sq, full_w, full_b, out_w, out_b, out, M, T * sq.shape[-1], full_w.shape[0],
                        threads=B * 256, group_size=256)


# ---- tensor-op versions (tensor_ops.metal) for Model.infer; rows = B * 66 with B % 16 == 0

def attention_tensor(qkv, pieces, q_table, k_table, out, n_heads: int, n_kv_heads: int):
  """same as attention() (offsets computed in the kernel), into out (B, T, H*D)"""
  tensor_ops().attention_tensor(qkv, pieces, q_table, k_table, out, n_heads, n_kv_heads, q_table.shape[-1],
                                threads=qkv.shape[0] * n_heads * 384, group_size=384)


def linear_add_layernorm(a, weight, bias, x, norm_weight, norm_bias, h):
  """x += F.linear(a, weight, bias) (in place), h = LayerNorm(x); x, h (B, T, D_HIDDEN)"""
  rows = x.shape[0] * x.shape[1]
  tensor_ops().linear_add_layernorm(a, weight, bias, x, norm_weight, norm_bias, h, a.shape[-1],
                                    threads=(rows // 16) * 128, group_size=128)


def gate_up_swiglu(h, weight, bias, out):
  """out = swiglu(F.linear(h, weight, bias)); weight (2F, D), out (B, T, F)"""
  rows, f = h.shape[0] * h.shape[1], out.shape[-1]
  tensor_ops().gate_up_swiglu(h, weight, bias, out, h.shape[-1], f, threads=(rows // 32) * (f // 128) * 128, group_size=128)
