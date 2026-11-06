#include <metal_stdlib>
using namespace metal;


constant constexpr uint MAX_T = 72; // max tokens per position
constant constexpr uint D = 64; // head dim
constant constexpr uint LANES = 2; // threads sharing one query row, each does D / LANES dims
constant constexpr uint D_HIDDEN = 256; // model width

// thread -> simdgroup (x32) -> threadgroup (<=1024 threads)

// fused attention + piece offset bias (incl. softmax)
// one threadgroup per (position, kv head), does (H/KV * T * LANES)
// = 2 * 66 * 2 = 264 threads
kernel void attention(
  device const bfloat *qkv [[buffer(0)]], // (B, T, (H + 2*KV)*D) fused q|k|v projection output
  device const int *pieces [[buffer(1)]], // (B, T)
  device const int *offsets [[buffer(2)]], // (T, T)
  device const bfloat *q_table [[buffer(3)]], // (H, 14, N_OFF)
  device const bfloat *k_table [[buffer(4)]], // (H, 14, N_OFF)
  device bfloat *out [[buffer(5)]],  // (B, T, H*D) before the out projection (MPS does that matmul)
  constant uint &T [[buffer(6)]],
  constant uint &H [[buffer(7)]],
  constant uint &KV [[buffer(8)]],
  constant uint &N_OFF [[buffer(9)]], // 15 * 15 board offsets + query castle/ep + key castle/ep + pad
  uint group_id [[threadgroup_position_in_grid]], // threadgroup id
  uint local_id [[thread_position_in_threadgroup]], // id within threadgroup
  uint n_threads [[threads_per_threadgroup]] // = 264
) {
  constexpr uint D_LANE = D / LANES; // dims per thread (32)
  constexpr uint D4 = D / 4; // chunks per head vector

  // memory shared by threadgroup (this kv head's k and v, piece values for offset)
  threadgroup bfloat k_tile[MAX_T * D];
  threadgroup bfloat v_tile[MAX_T * D];
  threadgroup int piece_at[MAX_T];

  // threadgroup = (position pos, kv head kv_head)
  uint pos = group_id / KV; // which item in batch (board)
  uint kv_head = group_id % KV; // which kv head
  uint row_len = H * D + 2 * KV * D; // elements per token in qkv
  device const bfloat *pos_qkv = qkv + pos * T * row_len; // qkv[pos]; (T, row_len)

  // load into k_tile, v_tile, piece_at
  // load in groups of 4 (bfloat4 = 4xbfloat vector)
  device const bfloat4 *pos_qkv4 = (device const bfloat4 *) pos_qkv;
  threadgroup bfloat4 *k_tile4 = (threadgroup bfloat4 *) k_tile;
  threadgroup bfloat4 *v_tile4 = (threadgroup bfloat4 *) v_tile;

  uint row_len4 = row_len / 4;
  uint k_start4 = (H * D + kv_head * D) / 4; // where the head's k/v starts
  uint v_start4 = (H * D + KV * D + kv_head * D) / 4;

  // write in chunks of 4
  for (uint chunk = local_id; chunk < T * D4; chunk += n_threads) {
    uint token = chunk / D4;
    uint c = chunk - token * D4;
    k_tile4[chunk] = pos_qkv4[token * row_len4 + k_start4 + c];
    v_tile4[chunk] = pos_qkv4[token * row_len4 + v_start4 + c];
  }
  for (uint token = local_id; token < T; token += n_threads)
    piece_at[token] = pieces[pos * T + token];

  // sync until tile writes are done
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // get query head, query token, dims for this local_id
  // CALCULATIONS SPLIT PER ROW (splits D)
  uint threads_per_head = T * LANES;
  if (local_id >= (H / KV) * threads_per_head)
    return;
  uint q_head = kv_head * (H / KV) + local_id / threads_per_head;
  uint slot = local_id % threads_per_head;
  uint query = slot / LANES;
  uint dim0 = (slot % LANES) * D_LANE;

  // load half of q and output into registers
  float scale = rsqrt(float(D));
  float q_part[D_LANE], out_acc[D_LANE];
  for (uint d = 0; d < D_LANE; d++) {
    q_part[d] = float(pos_qkv[query * row_len + q_head * D + dim0 + d]) * scale;
    out_acc[d] = 0.0f;
  }

  // piece offset bias
  device const bfloat *q_bias_row = q_table + (q_head * 14 + piece_at[query]) * N_OFF;
  device const bfloat *k_bias_head = k_table + q_head * 14 * N_OFF;
  device const int *offsets_row = offsets + query * T;

  // online softmax w/ bias, equiv:
  float running_max = -INFINITY, running_sum = 0.0f;
  for (uint j = 0; j < T; j++) {
    // scores = q @ k.T + bias[query, :]
    float score = 0.0f;
    for (uint d = 0; d < D_LANE; d++)
      score += q_part[d] * float(k_tile[j * D + dim0 + d]);
    // must add partner thread's values, simd_shuffle_xor gets compliment
    score += simd_shuffle_xor(score, 0b1); // mask
    // calculate bias
    int off = offsets_row[j];
    score += float(q_bias_row[off]) + float(k_bias_head[piece_at[j] * N_OFF + off]);

    // online softmax update
    float new_max = max(running_max, score);
    float rescale = exp(running_max - new_max);
    float weight = exp(score - new_max);
    running_sum = running_sum * rescale + weight;
    for (uint d = 0; d < D_LANE; d++)
      out_acc[d] = out_acc[d] * rescale + weight * float(v_tile[j * D + dim0 + d]);
    running_max = new_max;
  }

  // normalize
  device bfloat *out_row = out + (pos * T + query) * (H * D) + q_head * D + dim0;
  float inv_sum = 1.0f / running_sum;
  for (uint d = 0; d < D_LANE; d++)
    out_row[d] = bfloat(out_acc[d] * inv_sum);
}

// gate, up = gate_up.chunk(2, -1)
// out = F.silu(gate) * up
// does NOT do final projection
kernel void swiglu(
  device const bfloat *gate_up [[buffer(0)]], // (rows, 2F)
  device bfloat *out [[buffer(1)]], // (rows, F)
  constant uint &F [[buffer(2)]], // size of output
  uint idx [[thread_position_in_grid]]
) {
  uint row = idx / F;
  uint col = idx - row * F;
  float gate = float(gate_up[row * 2 * F + col]);
  float up = float(gate_up[row * 2 * F + F + col]);
  out[idx] = bfloat(gate / (1.0f + exp(-gate)) * up);
}


// embedding + LayerNorm (eager bf16 path; the fused path uses embed_fused below)
// squares = embed_piece[board] + embed_square[arange(64)] + embed_piece_square[64 * board + arange(64)]
// castle = embed_castle[castling];  ep = embed_ep[0 if ep < 0 else ep % 8 + 1]
// out = out_norm(cat([squares, castle, ep], 1))
// one simdgroup per output token (B * N_TOKENS of them), each lane does D_HIDDEN / 32 = 8 dims
kernel void embedding(
  device const long *board [[buffer(0)]], // (B, 64) piece on each square, 0 = empty, 1..12
  device const long *castling [[buffer(1)]], // (B,) castling rights 0..15
  device const long *ep [[buffer(2)]], // (B,) en-passant square, < 0 = none
  device const bfloat *embed_piece [[buffer(3)]], // (13, D_HIDDEN)
  device const bfloat *embed_square [[buffer(4)]], // (64, D_HIDDEN)
  device const bfloat *embed_piece_square [[buffer(5)]], // (13 * 64, D_HIDDEN)
  device const bfloat *embed_castle [[buffer(6)]], // (16, D_HIDDEN)
  device const bfloat *embed_ep [[buffer(7)]], // (9, D_HIDDEN): 0 = none, else file + 1
  device const bfloat *ln_weight [[buffer(8)]], // (D_HIDDEN,)
  device const bfloat *ln_bias [[buffer(9)]], // (D_HIDDEN,)
  device bfloat *out [[buffer(10)]], // (B, N_TOKENS, D_HIDDEN)
  constant uint &N_TOKENS [[buffer(11)]], // 66 = 64 squares + castle + ep
  uint global_id [[thread_position_in_grid]], // 0 .. B * N_TOKENS * 32 - 1
  uint lane [[thread_index_in_simdgroup]] // 0 .. 31
) {
  constexpr uint DIMS_PER_LANE = D_HIDDEN / 32;
  uint token_id = global_id / 32; // which output row (pos, token) this simdgroup builds
  uint pos = token_id / N_TOKENS; // which item in batch (board)
  uint token = token_id - pos * N_TOKENS; // token_id % N_TOKENS
  uint dim0 = lane * DIMS_PER_LANE; // this lane's dims: dim0 .. dim0 + 7

  float vec[DIMS_PER_LANE];
  if (token < 64) {
    // square: piece + square + (piece, square) embeddings
    long p = board[pos * 64 + token];
    device const bfloat *piece_row = embed_piece + p * D_HIDDEN + dim0;
    device const bfloat *square_row = embed_square + token * D_HIDDEN + dim0;
    device const bfloat *piece_square_row = embed_piece_square + (64 * p + token) * D_HIDDEN + dim0;
    for (uint k = 0; k < DIMS_PER_LANE; k++)
      vec[k] = float(piece_row[k]) + float(square_row[k]) + float(piece_square_row[k]);
  } else if (token == 64) {
    // castle token
    device const bfloat *castle_row = embed_castle + castling[pos] * D_HIDDEN + dim0;
    for (uint k = 0; k < DIMS_PER_LANE; k++)
      vec[k] = float(castle_row[k]);
  } else {
    // ep token: 0 = none, else the ep file + 1
    long e = ep[pos];
    device const bfloat *ep_row = embed_ep + (e < 0 ? 0 : e % 8 + 1) * D_HIDDEN + dim0;
    for (uint k = 0; k < DIMS_PER_LANE; k++)
      vec[k] = float(ep_row[k]);
  }

  // LayerNorm over the D_HIDDEN dims: each lane sums its 8, simd_sum adds across the 32 lanes (no memory needed)
  // mean = vec.mean();  var = ((vec - mean) ** 2).mean()
  // out = (vec - mean) / sqrt(var + 1e-5) * ln_weight + ln_bias
  float partial = 0.0f;
  for (uint k = 0; k < DIMS_PER_LANE; k++)
    partial += vec[k];
  float mean = simd_sum(partial) / D_HIDDEN;
  float partial_sq = 0.0f;
  for (uint k = 0; k < DIMS_PER_LANE; k++) {
    vec[k] -= mean;
    partial_sq += vec[k] * vec[k];
  }
  float inv_std = rsqrt(simd_sum(partial_sq) / D_HIDDEN + 1e-5f);
  device bfloat *out_row = out + token_id * D_HIDDEN + dim0;
  for (uint k = 0; k < DIMS_PER_LANE; k++)
    out_row[k] = bfloat(vec[k] * inv_std * float(ln_weight[dim0 + k]) + float(ln_bias[dim0 + k]));
}

// ---- fused inference (models/v11_muon_metal.Model.infer): the whole forward from forward_batch's packed rows to
// move probabilities + value, with the matmuls left to MPS ----
// per position: embed_fused -> per block [qkv matmul, attention, out matmul, add_layernorm, gate_up matmul, swiglu,
// down matmul, add_layernorm] -> policy matmul + policy_head, value matmul + value_head

constant constexpr uint MAX_NJ = 66 * 32; // value head: tokens x d_square_value
constant constexpr uint MAX_DF = 128;     // value head: d_full_value

// LayerNorm of one D_HIDDEN row held by a simdgroup (each lane D_HIDDEN / 32 values), same math as in `embedding`
// called by a whole simdgroup at once (simd_sum needs all 32 lanes); writes out_row[dim0 .. dim0 + 7]
inline void layernorm_row(thread float *vec, device const bfloat *ln_weight, device const bfloat *ln_bias, uint dim0,
                          device bfloat *out_row) {
  constexpr uint DIMS_PER_LANE = D_HIDDEN / 32;
  float partial = 0.0f;
  for (uint k = 0; k < DIMS_PER_LANE; k++)
    partial += vec[k];
  float mean = simd_sum(partial) / D_HIDDEN;
  float partial_sq = 0.0f;
  for (uint k = 0; k < DIMS_PER_LANE; k++)
    partial_sq += (vec[k] - mean) * (vec[k] - mean);
  float inv_std = rsqrt(simd_sum(partial_sq) / D_HIDDEN + 1e-5f);
  for (uint k = 0; k < DIMS_PER_LANE; k++)
    out_row[dim0 + k] = bfloat((vec[k] - mean) * inv_std * float(ln_weight[dim0 + k]) + float(ln_bias[dim0 + k]));
}

// embedding straight from the packed uint8 rows, then both norms block 0 needs:
// x = out_norm(tokens)        (the residual stream, same as `embedding`)
// h = blocks[0].norm1(x)      (what block 0's attention reads)
// pieces = piece id per token (13 for the castle / ep tokens), for the attention's offset bias
// packed row: board 0..63, castling 64, ep + 1 at 65 (0 = none), move count 66, moves from 67 (4 bytes each)
// one simdgroup per token, like `embedding`
kernel void embed_fused(
  device const uchar *packed [[buffer(0)]], // (B, S)
  device const bfloat *embed_piece [[buffer(1)]], // (13, D_HIDDEN)
  device const bfloat *embed_square [[buffer(2)]], // (64, D_HIDDEN)
  device const bfloat *embed_piece_square [[buffer(3)]], // (13 * 64, D_HIDDEN)
  device const bfloat *embed_castle [[buffer(4)]], // (16, D_HIDDEN)
  device const bfloat *embed_ep [[buffer(5)]], // (9, D_HIDDEN)
  device const bfloat *ln0_weight [[buffer(6)]], // out_norm (D_HIDDEN,)
  device const bfloat *ln0_bias [[buffer(7)]],
  device const bfloat *ln1_weight [[buffer(8)]], // blocks[0].norm1 (D_HIDDEN,)
  device const bfloat *ln1_bias [[buffer(9)]],
  device bfloat *x_out [[buffer(10)]], // (B, N_TOKENS, D_HIDDEN)
  device bfloat *h_out [[buffer(11)]], // (B, N_TOKENS, D_HIDDEN)
  device int *pieces_out [[buffer(12)]], // (B, N_TOKENS)
  constant uint &N_TOKENS [[buffer(13)]], // 66
  constant uint &S [[buffer(14)]], // bytes per packed row (67 + 4M)
  uint global_id [[thread_position_in_grid]], // 0 .. B * N_TOKENS * 32 - 1
  uint lane [[thread_index_in_simdgroup]] // 0 .. 31
) {
  constexpr uint DIMS_PER_LANE = D_HIDDEN / 32;
  uint token_id = global_id / 32;
  uint pos = token_id / N_TOKENS;
  uint token = token_id - pos * N_TOKENS;
  uint dim0 = lane * DIMS_PER_LANE;
  device const uchar *row = packed + pos * S; // packed[pos]

  float vec[DIMS_PER_LANE];
  if (token < 64) {
    uint p = row[token];
    device const bfloat *piece_row = embed_piece + p * D_HIDDEN + dim0;
    device const bfloat *square_row = embed_square + token * D_HIDDEN + dim0;
    device const bfloat *piece_square_row = embed_piece_square + (64 * p + token) * D_HIDDEN + dim0;
    for (uint k = 0; k < DIMS_PER_LANE; k++)
      vec[k] = float(piece_row[k]) + float(square_row[k]) + float(piece_square_row[k]);
  } else {
    // castle token reads embed_castle[castling]; ep token embed_ep[file + 1] (row[65] is ep square + 1, 0 = none)
    uint e = row[65];
    device const bfloat *castle_or_ep_row = token == 64 ? embed_castle + row[64] * D_HIDDEN + dim0
                                                        : embed_ep + (e == 0 ? 0 : (e - 1) % 8 + 1) * D_HIDDEN + dim0;
    for (uint k = 0; k < DIMS_PER_LANE; k++)
      vec[k] = float(castle_or_ep_row[k]);
  }
  if (lane == 0)
    pieces_out[token_id] = token < 64 ? int(row[token]) : 13;

  device bfloat *x_row = x_out + token_id * D_HIDDEN;
  layernorm_row(vec, ln0_weight, ln0_bias, dim0, x_row);
  for (uint k = 0; k < DIMS_PER_LANE; k++)
    vec[k] = float(x_row[dim0 + k]);          // block 0 reads the stored (bf16) residual stream, as in PyTorch
  layernorm_row(vec, ln1_weight, ln1_bias, dim0, h_out + token_id * D_HIDDEN);
}

// residual add + the next LayerNorm, used twice per block:
// x = x + y   (y = attention or MLP output after its matmul; bf16, in place)
// h = norm(x) (norm2 after attention; after the MLP, the next block's norm1, or the final norm after the last block)
// one simdgroup per token row
kernel void add_layernorm(
  device bfloat *x [[buffer(0)]], // (B * N_TOKENS, D_HIDDEN) residual stream, updated in place
  device const bfloat *y [[buffer(1)]], // (B * N_TOKENS, D_HIDDEN)
  device const bfloat *ln_weight [[buffer(2)]], // (D_HIDDEN,)
  device const bfloat *ln_bias [[buffer(3)]],
  device bfloat *h [[buffer(4)]], // (B * N_TOKENS, D_HIDDEN) input of the next layer
  uint global_id [[thread_position_in_grid]],
  uint lane [[thread_index_in_simdgroup]]
) {
  constexpr uint DIMS_PER_LANE = D_HIDDEN / 32;
  uint token_id = global_id / 32;
  uint dim0 = lane * DIMS_PER_LANE;
  device bfloat *x_row = x + token_id * D_HIDDEN;
  device const bfloat *y_row = y + token_id * D_HIDDEN;
  float vec[DIMS_PER_LANE];
  for (uint k = 0; k < DIMS_PER_LANE; k++) {
    // round the sum to bf16 before the norm, as PyTorch does (x is stored in bf16)
    bfloat sum = bfloat(float(x_row[dim0 + k]) + float(y_row[dim0 + k]));
    x_row[dim0 + k] = sum;
    vec[k] = float(sum);
  }
  layernorm_row(vec, ln_weight, ln_bias, dim0, h + token_id * D_HIDDEN);
}

// policy head on the legal moves only, then softmax over them:
// logit[k] = (p_from[from_k] . p_to[to_k]) * scale  (+ p_to[to_k] . wp[type_k - 3] + bp[type_k - 3] if a promotion)
// probs = softmax(logits[:count])  -> out[pos, :M], 0 past the move count
// one threadgroup per position, one thread per move slot k (M threads; M <= 1024)
kernel void policy_head(
  device const bfloat *pft [[buffer(0)]], // (B, N_TOKENS, 2 * DP) = [p_from | p_to] for every token
  device const uchar *packed [[buffer(1)]], // (B, S): the move list (from, to, type) per row
  device const bfloat *wp [[buffer(2)]], // (n promotion types, DP)
  device const bfloat *bp [[buffer(3)]], // (n promotion types,)
  device float *out [[buffer(4)]], // (B, M + 1): M probabilities, then the value (value_head fills column M)
  constant uint &M [[buffer(5)]], // move slots per row (max legal moves in this batch)
  constant uint &S [[buffer(6)]], // bytes per packed row
  constant uint &N_TOKENS [[buffer(7)]],
  constant uint &DP [[buffer(8)]], // d_policy
  constant float &scale [[buffer(9)]],
  uint pos [[threadgroup_position_in_grid]], // which item in batch (board)
  uint k [[thread_position_in_threadgroup]], // which move slot
  uint sg [[simdgroup_index_in_threadgroup]], // which simdgroup of this threadgroup
  uint lane [[thread_index_in_simdgroup]], // 0 .. 31
  uint nsg [[simdgroups_per_threadgroup]] // = ceil(M / 32)
) {
  threadgroup float part[32]; // one partial max / sum per simdgroup
  device const uchar *row = packed + pos * S;
  uint count = row[66]; // legal moves of this position
  float logit = -INFINITY; // empty slots: exp(-inf) = 0, so they drop out of the softmax
  if (k < count) {
    device const uchar *mv = row + 67 + 4 * k;
    uint from = mv[0], to = mv[1], type = mv[2];
    device const bfloat *pf = pft + (pos * N_TOKENS + from) * 2 * DP; // p_from of the from square
    device const bfloat *pt = pft + (pos * N_TOKENS + to) * 2 * DP + DP; // p_to of the to square
    float score = 0.0f;
    for (uint d = 0; d < DP; d++)
      score += float(pf[d]) * float(pt[d]);
    logit = score * scale;
    if (type >= 3) {
      // promotion: add a per-piece term
      device const bfloat *w = wp + (type - 3) * DP;
      float pr = float(bp[type - 3]);
      for (uint d = 0; d < DP; d++)
        pr += float(pt[d]) * float(w[d]);
      logit += pr;
    }
  }

  // softmax across the threadgroup (the M moves span several simdgroups): two reductions, max then sum
  // each reduction: simd_max / simd_sum inside each simdgroup, lane 0 writes it to part[sg], barrier, every thread
  // combines the nsg partials
  float running_max = simd_max(logit);
  if (lane == 0)
    part[sg] = running_max;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  running_max = part[0];
  for (uint i = 1; i < nsg; i++)
    running_max = max(running_max, part[i]);
  threadgroup_barrier(mem_flags::mem_threadgroup); // everyone has read part before it is reused for the sums
  float weight = k < count ? exp(logit - running_max) : 0.0f;
  float running_sum = simd_sum(weight);
  if (lane == 0)
    part[sg] = running_sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  running_sum = 0.0f;
  for (uint i = 0; i < nsg; i++)
    running_sum += part[i];
  out[pos * (M + 1) + k] = count > 0 ? weight / running_sum : 0.0f;
}

// value head MLP + WDL softmax:
// a = silu(sq.flatten())                      (N_TOKENS * DSQ = NJ inputs)
// hidden = silu(wf @ a + bf)                  (DF)
// z = wo @ hidden + bo                        (3: win, draw, loss logits)
// out[pos, M] = p_win - p_loss                (softmax(z)[0] - softmax(z)[2])
// one threadgroup per position (256 threads = 8 simdgroups)
kernel void value_head(
  device const bfloat *sq [[buffer(0)]], // (B, NJ) = value_square_proj(x), flattened per position
  device const bfloat *wf [[buffer(1)]], // value_full_proj weight (DF, NJ)
  device const bfloat *bf [[buffer(2)]], // (DF,)
  device const bfloat *wo [[buffer(3)]], // value_out weight (3, DF)
  device const bfloat *bo [[buffer(4)]], // (3,)
  device float *out [[buffer(5)]], // (B, M + 1): writes column M
  constant uint &M [[buffer(6)]],
  constant uint &NJ [[buffer(7)]], // N_TOKENS * d_square_value (<= MAX_NJ)
  constant uint &DF [[buffer(8)]], // d_full_value (<= MAX_DF)
  uint pos [[threadgroup_position_in_grid]], // which item in batch (board)
  uint local_id [[thread_position_in_threadgroup]],
  uint n_threads [[threads_per_threadgroup]], // = 256
  uint sg [[simdgroup_index_in_threadgroup]], // 0 .. 7
  uint lane [[thread_index_in_simdgroup]], // 0 .. 31
  uint nsg [[simdgroups_per_threadgroup]] // = 8
) {
  threadgroup float a[MAX_NJ]; // silu(sq[pos]) in fp32, shared by every row of wf
  threadgroup float hidden[MAX_DF];
  threadgroup float z[3];

  // a = silu(sq[pos]): all threads, grid-stride like attention's tile load
  for (uint j = local_id; j < NJ; j += n_threads) {
    float v = float(sq[pos * NJ + j]);
    a[j] = v / (1.0f + exp(-v));
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // hidden = silu(wf @ a + bf): simdgroup sg does rows sg, sg + 8, ...; its 32 lanes split the NJ-long dot product
  // (lane reads j = lane, lane + 32, ...: neighbouring lanes read neighbouring weights, one coalesced load)
  for (uint r = sg; r < DF; r += nsg) {
    device const bfloat *w = wf + r * NJ;
    float acc = 0.0f;
    for (uint j = lane; j < NJ; j += 32)
      acc += a[j] * float(w[j]);
    acc = simd_sum(acc) + float(bf[r]);
    if (lane == 0)
      hidden[r] = acc / (1.0f + exp(-acc));
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // z = wo @ hidden + bo: simdgroups 0, 1, 2 do one output each
  if (sg < 3) {
    float acc = 0.0f;
    for (uint r = lane; r < DF; r += 32)
      acc += hidden[r] * float(wo[sg * DF + r]);
    acc = simd_sum(acc) + float(bo[sg]);
    if (lane == 0)
      z[sg] = acc;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // softmax over the 3 logits, value = p_win - p_loss (one thread)
  if (local_id == 0) {
    float zm = max(z[0], max(z[1], z[2]));
    float e0 = exp(z[0] - zm), e1 = exp(z[1] - zm), e2 = exp(z[2] - zm);
    out[pos * (M + 1) + M] = (e0 - e2) / (e0 + e1 + e2);
  }
}
