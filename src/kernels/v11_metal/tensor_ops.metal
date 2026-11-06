#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

// Model.infer kernels on Metal 4 tensor ops (macOS 26+). On M5, the GPU's matrix accelerators are only reachable
// through matmul2d; simdgroup_matrix runs on the normal ALUs (~6 TFLOP/s vs ~16 measured).
// The matmul result stays in registers (a cooperative tensor) and the epilogue (bias, SwiGLU, residual add,
// LayerNorm, softmax inputs) runs on it before anything is written to memory.
// Layout as torch: x (rows, K) bf16 row-major, w (N, K) = nn.Linear weight, y = x w^T + b.
// Rows per call must be a multiple of 32 (B % 16 == 0): no edge tiles.

constant constexpr uint T = 66;          // tokens per position
constant constexpr uint D = 64;          // head dim
constant constexpr uint D_HIDDEN = 256;  // model width
constant constexpr uint SP = 68;         // attention score row stride (floats); P (bf16) reuses the same memory

using tdev = tensor<device bfloat, dextents<int32_t, 2>, tensor_inline>;
using ttg = tensor<threadgroup bfloat, dextents<int32_t, 2>, tensor_inline>;

// rows [r0, r0 + R) of a row-major (?, K) bf16 matrix as a tensor (extent 0 = K, the contiguous one)
inline tdev rows(device bfloat *p, uint K, uint r0, uint R) {
  return tdev(p + r0 * K, dextents<int32_t, 2>(K, R), array<int32_t, 2>{1, int32_t(K)});
}

// same rule as Model.offset_table: castle/ep key column 227 wins over castle/ep query row 226
inline int offset_of(uint i, uint j) {
  if (j >= 64) return 227;
  if (i >= 64) return 226;
  return int((j >> 3) - (i >> 3) + 7) * 15 + int((j & 7) - (i & 7) + 7);
}

// gate_up + SwiGLU: w = gate_up weight (2F, K), rows [0, F) gate, [F, 2F) up -> y (rows, F) = silu(g) * u
// one threadgroup = 32 rows x 128 outputs, 4 simdgroups
kernel void gate_up_swiglu(
  device bfloat *x [[buffer(0)]],
  device bfloat *w [[buffer(1)]],
  device bfloat *bias [[buffer(2)]],
  device bfloat *y [[buffer(3)]],
  constant uint &K [[buffer(4)]],
  constant uint &F [[buffer(5)]],
  uint tg [[threadgroup_position_in_grid]]
) {
  constexpr uint BM = 32, BN = 128;
  uint nb = F / BN, m0 = (tg / nb) * BM, n0 = (tg % nb) * BN;
  tdev A = rows(x, K, m0, BM), Bg = rows(w, K, n0, BN), Bu = rows(w, K, F + n0, BN);
  constexpr auto desc = matmul2d_descriptor(BM, BN, static_cast<int>(dynamic_extent), false, true, false);
  matmul2d<desc, execution_simdgroups<4>> op;
  auto g = op.template get_destination_cooperative_tensor<tdev, tdev, float>();
  auto u = op.template get_destination_cooperative_tensor<tdev, tdev, float>();
  op.run(A, Bg, g);
  op.run(A, Bu, u);
#pragma unroll
  for (uint16_t i = 0; i < g.get_capacity(); ++i) {
    if (g.is_valid_element(i)) {
      auto idx = g.get_multidimensional_index(i);    // (col n, row m) within the tile
      uint n = n0 + idx[0], m = m0 + idx[1];
      float gv = float(bfloat(g[i] + float(bias[n])));  // rounded to bf16 like the unfused gate_up output
      float uv = float(bfloat(u[i] + float(bias[F + n])));
      y[m * F + n] = bfloat(gv / (1.0f + exp(-gv)) * uv);
    }
  }
}

// out projection / down projection + residual add + LayerNorm, whole 256-wide row per tile:
//   x (rows, 256) += a w^T + b (in place);  h = LayerNorm(x) * gamma + beta
// one threadgroup = 16 rows, 4 simdgroups; the sum goes through threadgroup memory, then one simdgroup per row norms it
kernel void linear_add_layernorm(
  device bfloat *a [[buffer(0)]],
  device bfloat *w [[buffer(1)]],
  device bfloat *bias [[buffer(2)]],
  device bfloat *x [[buffer(3)]],
  device bfloat *gamma [[buffer(4)]],
  device bfloat *beta [[buffer(5)]],
  device bfloat *h [[buffer(6)]],
  constant uint &K [[buffer(7)]],
  uint tg [[threadgroup_position_in_grid]],
  uint sg [[simdgroup_index_in_threadgroup]],
  uint lane [[thread_index_in_simdgroup]]
) {
  constexpr uint BM = 16, N = D_HIDDEN, NSG = 4;
  threadgroup bfloat sum[BM * N];
  uint m0 = tg * BM;
  tdev A = rows(a, K, m0, BM), Bw = rows(w, K, 0, N);
  constexpr auto desc = matmul2d_descriptor(BM, N, static_cast<int>(dynamic_extent), false, true, false);
  matmul2d<desc, execution_simdgroups<NSG>> op;
  auto c = op.template get_destination_cooperative_tensor<tdev, tdev, float>();
  op.run(A, Bw, c);
#pragma unroll
  for (uint16_t i = 0; i < c.get_capacity(); ++i) {
    if (c.is_valid_element(i)) {
      auto idx = c.get_multidimensional_index(i);
      uint n = idx[0], r = idx[1];
      float yv = float(bfloat(c[i] + float(bias[n])));  // rounded to bf16 like the unfused linear output
      sum[r * N + n] = bfloat(float(x[(m0 + r) * N + n]) + yv);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint r = sg; r < BM; r += NSG) {
    float v[8], s = 0.0f;
    for (uint k = 0; k < 8; k++) { v[k] = float(sum[r * N + lane * 8 + k]); s += v[k]; }
    float mean = simd_sum(s) / N;
    float q = 0.0f;
    for (uint k = 0; k < 8; k++) { float d = v[k] - mean; q += d * d; }
    float inv = rsqrt(simd_sum(q) / N + 1e-5f);
    device bfloat *xr = x + (m0 + r) * N + lane * 8, *hr = h + (m0 + r) * N + lane * 8;
    for (uint k = 0; k < 8; k++) {
      uint n = lane * 8 + k;
      xr[k] = bfloat(v[k]);
      hr[k] = bfloat((v[k] - mean) * inv * float(gamma[n]) + float(beta[n]));
    }
  }
}

// attention + piece-offset bias, one threadgroup per (position, query head), 12 simdgroups (384 threads)
//   S = Q K^T (tensor op, fp32) -> * 1/8 + q_table[piece_i, off] + k_table[piece_j, off] -> threadgroup memory
//   softmax, one simdgroup per row -> P (bf16, written over S)
//   O = P V (tensor op) -> out (bf16)
// 12 simdgroups: the per-row work (bias, softmax) needs the parallelism (4: 1.25x the old kernel, 12: 1.9x)
kernel void attention_tensor(
  device bfloat *qkv [[buffer(0)]],            // (B, T, (H + 2*KV)*D)
  device const int *pieces [[buffer(1)]],      // (B, T), 13 = castle / ep token
  device const bfloat *q_table [[buffer(2)]],  // (H, 14, N_OFF)
  device const bfloat *k_table [[buffer(3)]],  // (H, 14, N_OFF)
  device bfloat *out [[buffer(4)]],            // (B, T, H*D)
  constant uint &H [[buffer(5)]],
  constant uint &KV [[buffer(6)]],
  constant uint &N_OFF [[buffer(7)]],
  uint group [[threadgroup_position_in_grid]],
  uint tid [[thread_position_in_threadgroup]],
  uint sg [[simdgroup_index_in_threadgroup]],
  uint lane [[thread_index_in_simdgroup]]
) {
  constexpr uint NSG = 12, TM = 80, TN = 64;   // tile 80 >= T (M or N must be a multiple of 16)
  threadgroup float S[T * SP];
  threadgroup int P[T];
  threadgroup bfloat *Pm = (threadgroup bfloat *) S;
  constexpr uint PP = SP * 2;                  // P row stride (bf16): P row i overwrites S row i

  uint b = group / H, h = group % H, g = h / (H / KV), row = H * D + 2 * KV * D;
  device bfloat *base = qkv + b * T * row;
  for (uint t = tid; t < T; t += NSG * 32) P[t] = pieces[b * T + t];

  tdev Q(base + h * D, dextents<int32_t, 2>(D, T), array<int32_t, 2>{1, int32_t(row)});
  tdev K(base + H * D + g * D, dextents<int32_t, 2>(D, T), array<int32_t, 2>{1, int32_t(row)});
  tdev V(base + H * D + KV * D + g * D, dextents<int32_t, 2>(D, T), array<int32_t, 2>{1, int32_t(row)});

  // S = Q K^T
  constexpr auto d1 = matmul2d_descriptor(TM, TM, static_cast<int>(dynamic_extent), false, true, false);
  matmul2d<d1, execution_simdgroups<NSG>> op1;
  auto s = op1.template get_destination_cooperative_tensor<tdev, tdev, float>();
  op1.run(Q, K, s);
  threadgroup_barrier(mem_flags::mem_threadgroup);   // P[] visible
  device const bfloat *qt = q_table + h * 14 * N_OFF, *kt = k_table + h * 14 * N_OFF;
#pragma unroll
  for (uint16_t e = 0; e < s.get_capacity(); ++e) {
    if (s.is_valid_element(e)) {
      auto ix = s.get_multidimensional_index(e);       // (key j, query i)
      uint j = ix[0], i = ix[1];
      if (i < T && j < T) {
        int o = offset_of(i, j);
        S[i * SP + j] = s[e] * 0.125f + float(qt[P[i] * N_OFF + o]) + float(kt[P[j] * N_OFF + o]);
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // softmax: one simdgroup per row, lanes over keys j = lane, lane + 32, lane + 64
  for (uint i = sg; i < T; i += NSG) {
    threadgroup float *sr = S + i * SP;
    float v0 = sr[lane], v1 = sr[lane + 32], v2 = lane < T - 64 ? sr[lane + 64] : -INFINITY;
    float mx = simd_max(max(max(v0, v1), v2));
    v0 = exp(v0 - mx); v1 = exp(v1 - mx); v2 = exp(v2 - mx);
    float inv = 1.0f / simd_sum(v0 + v1 + v2);       // the reductions also order every lane's reads before the writes
    threadgroup bfloat *pr = Pm + i * PP;
    pr[lane] = bfloat(v0 * inv);
    pr[lane + 32] = bfloat(v1 * inv);
    if (lane < 8) pr[lane + 64] = bfloat(v2 * inv);   // keys 66..71 -> 0
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // O = P V
  ttg Pt(Pm, dextents<int32_t, 2>(T, T), array<int32_t, 2>{1, int32_t(PP)});
  constexpr auto d2 = matmul2d_descriptor(TM, TN, static_cast<int>(dynamic_extent), false, false, false);
  matmul2d<d2, execution_simdgroups<NSG>> op2;
  auto o = op2.template get_destination_cooperative_tensor<ttg, tdev, float>();
  op2.run(Pt, V, o);
  device bfloat *ob = out + b * T * (H * D) + h * D;
#pragma unroll
  for (uint16_t e = 0; e < o.get_capacity(); ++e) {
    if (o.is_valid_element(e)) {
      auto ix = o.get_multidimensional_index(e);       // (dim d, query i)
      uint d = ix[0], i = ix[1];
      if (i < T && d < D) ob[i * (H * D) + d] = bfloat(o[e]);
    }
  }
}
