// Standalone sparse paged batch-decode CUDA kernel with split-K parallelism.
//
// One thread block per (batch_idx, q_head, k_split). Each block reads a
// chunk of its sparse_len[b, qh] tokens listed in sparse_idx[b, qh, :S],
// gathers the corresponding K/V vectors from the paged KV cache directly
// (no intermediate gather tensor), and computes the partial online-softmax
// state (m, d, o) over its chunk.
//
// When K_SPLIT > 1 a small `sparse_decode_merge_kernel` reduces the per-split
// (m, d, o) partials per (b, qh) into the final output and LSE. When
// K_SPLIT == 1 the kernel writes the final output directly.
//
// FlashInfer-extracted helpers (vec_t, math::tanh, ptx_exp2) are reused for
// vectorized 16-byte loads + numerically stable log2-domain softmax math.

#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <algorithm>
#include <cstdint>

#include "flashinfer/math.cuh"
#include "flashinfer/vec_dtypes.cuh"

#ifndef CUDART_INF_F
#define CUDART_INF_F __int_as_float(0x7f800000)
#endif

namespace sparse_decode {

namespace fi = flashinfer;

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be CUDA")
#define CHECK_CONTIG(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")

// -----------------------------------------------------------------------------
// Main kernel. One CTA per (b, qh, ks).
//
// Token range processed: [ks * chunk_size, min((ks+1)*chunk_size, S)) where
// chunk_size = ceil(S / K_SPLIT). When K_SPLIT == 1 this collapses to [0, S).
// -----------------------------------------------------------------------------
template <typename DType, uint32_t HEAD_DIM, uint32_t VEC_SIZE, uint32_t BDX,
          uint32_t BDY, bool USE_SOFT_CAP>
__global__ __launch_bounds__(BDX * BDY, 4) void sparse_decode_kernel(
    const DType* __restrict__ q_ptr,           // [B, H_q, D]
    const DType* __restrict__ k_data,          // paged KV K base
    const DType* __restrict__ v_data,          // paged KV V base
    const int32_t* __restrict__ kv_indptr,     // [B+1]
    const int32_t* __restrict__ kv_indices,    // [total_pages]
    const int32_t* __restrict__ sparse_len_p,  // [B, H_q, 1]
    const int64_t* __restrict__ sparse_idx_p,  // [B, H_q, max_S]
    const float* __restrict__ sparse_w_p,      // [B, H_q, max_S]
    DType* __restrict__ o_ptr,                 // [B, H_q, D] (used iff K_SPLIT==1)
    float* __restrict__ lse_ptr,               // [B, H_q] or null (iff K_SPLIT==1)
    float* __restrict__ tmp_o_ptr,             // [B, H_q, K_SPLIT, D] (iff K_SPLIT>1)
    float* __restrict__ tmp_md_ptr,            // [B, H_q, K_SPLIT, 2] (iff K_SPLIT>1)
    uint32_t H_q, uint32_t H_kv, uint32_t page_size, uint32_t max_S,
    int64_t kv_stride_page, int64_t kv_stride_n, int64_t kv_stride_h,
    int K_SPLIT,
    float sm_scale_log2, float soft_cap, float soft_cap_pre_tanh_scale) {
  const uint32_t b = blockIdx.x;
  const uint32_t qh = blockIdx.y;
  const uint32_t ks = blockIdx.z;
  const uint32_t kv_head = qh / (H_q / H_kv);
  const uint32_t S_total = sparse_len_p[b * H_q + qh];

  const uint32_t tx = threadIdx.x;
  const uint32_t ty = threadIdx.y;
  const uint32_t lane_in_token = tx;
  const uint32_t feat_base = lane_in_token * VEC_SIZE;

  // Compute this split's token range.
  const uint32_t chunk_size = (S_total + K_SPLIT - 1) / K_SPLIT;
  const uint32_t s_lo = ks * chunk_size;
  const uint32_t s_hi = (s_lo + chunk_size > S_total) ? S_total : (s_lo + chunk_size);

  // Output destinations.
  DType* o_out = (K_SPLIT == 1) ? (o_ptr + (b * H_q + qh) * HEAD_DIM) : nullptr;
  float* tmp_o_out = (K_SPLIT > 1)
      ? (tmp_o_ptr + ((b * H_q + qh) * K_SPLIT + ks) * HEAD_DIM)
      : nullptr;
  float* tmp_md_out = (K_SPLIT > 1)
      ? (tmp_md_ptr + ((b * H_q + qh) * K_SPLIT + ks) * 2)
      : nullptr;

  // -------- Empty split fast-exit --------
  if (s_hi <= s_lo) {
    if (K_SPLIT == 1) {
      if (ty == 0) {
        fi::vec_t<DType, VEC_SIZE> zeros;
        zeros.fill(DType(0.f));
        zeros.store(o_out + feat_base);
      }
      if (lse_ptr != nullptr && tx == 0 && ty == 0) {
        lse_ptr[b * H_q + qh] = -CUDART_INF_F;
      }
    } else {
      // Write -INF / 0 partial so the merge kernel ignores this split.
      if (ty == 0 && tx == 0) {
        tmp_md_out[0] = -CUDART_INF_F;
        tmp_md_out[1] = 0.f;
      }
      if (ty == 0) {
        fi::vec_t<float, VEC_SIZE> zeros;
        zeros.fill(0.f);
        zeros.store(tmp_o_out + feat_base);
      }
    }
    return;
  }

  // -------- Load Q vector slice into registers --------
  fi::vec_t<float, VEC_SIZE> q_vec;
  fi::vec_t<DType, VEC_SIZE> q_load;
  q_load.load(q_ptr + (b * H_q + qh) * HEAD_DIM + feat_base);
#pragma unroll
  for (uint32_t i = 0; i < VEC_SIZE; ++i) {
    q_vec[i] = float(q_load[i]);
  }

  // -------- Per-(b, qh) sparse base pointers + page table --------
  const int64_t row_off = (int64_t)b * H_q * max_S + (int64_t)qh * max_S;
  const int64_t* idx_row = sparse_idx_p + row_off;
  const float* w_row = sparse_w_p + row_off;
  const int32_t indptr_b = kv_indptr[b];

  // -------- Online softmax state (per ty thread) --------
  float m = -CUDART_INF_F;
  float d = 0.f;
  fi::vec_t<float, VEC_SIZE> o_acc;
  o_acc.fill(0.f);

  // Process BDY tokens per outer iteration over [s_lo, s_hi).
  for (uint32_t base = s_lo; base < s_hi; base += BDY) {
    const uint32_t s = base + ty;
    const bool active = (s < s_hi);

    int64_t kv_off_tok = 0;
    float w_log = 0.f;
    if (active) {
      const int64_t t = idx_row[s];
      const float w = w_row[s];
      const float wpos = w > 0.f ? w : 1e-20f;
      w_log = log2f(wpos);
      const int32_t page_local = (int32_t)(t / page_size);
      const int32_t in_page = (int32_t)(t - (int64_t)page_local * page_size);
      const int32_t page_id = kv_indices[indptr_b + page_local];
      kv_off_tok = (int64_t)page_id * kv_stride_page +
                   (int64_t)kv_head * kv_stride_h +
                   (int64_t)in_page * kv_stride_n;
    }

    // Issue K + V loads up front (good memory-level parallelism via the
    // hardware load buffer; with split-K each CTA has few enough iterations
    // that an explicit cp.async double-buffer isn't worth the smem hop).
    fi::vec_t<DType, VEC_SIZE> k_vec_d;
    fi::vec_t<DType, VEC_SIZE> v_vec_d;
    if (active) {
      k_vec_d.load(k_data + kv_off_tok + (int64_t)feat_base);
      v_vec_d.load(v_data + kv_off_tok + (int64_t)feat_base);
    } else {
      k_vec_d.fill(DType(0.f));
    }

    float partial = 0.f;
#pragma unroll
    for (uint32_t i = 0; i < VEC_SIZE; ++i) {
      partial += q_vec[i] * float(k_vec_d[i]);
    }
#pragma unroll
    for (uint32_t step = BDX / 2; step > 0; step >>= 1) {
      partial += __shfl_xor_sync(0xFFFFFFFFu, partial, step, BDX);
    }

    float qk;
    if (active) {
      qk = partial;
      if constexpr (USE_SOFT_CAP) {
        qk = soft_cap * fi::math::tanh(qk * soft_cap_pre_tanh_scale);
      }
      qk = qk * sm_scale_log2 + w_log;
    } else {
      qk = -CUDART_INF_F;
    }

    if (active) {
      const float m_new = m > qk ? m : qk;
      const float scale_old = exp2f(m - m_new);
      const float p = exp2f(qk - m_new);
#pragma unroll
      for (uint32_t i = 0; i < VEC_SIZE; ++i) {
        o_acc[i] = o_acc[i] * scale_old + p * float(v_vec_d[i]);
      }
      d = d * scale_old + p;
      m = m_new;
    }
  }

  // -------- Within-CTA cross-ty merge (BDY rows -> single (m, d, o)) --------
  __shared__ float md_smem[BDY][2];
  __shared__ float o_smem[BDY][BDX * VEC_SIZE];
  __shared__ float global_md[2];

  if (tx == 0) {
    md_smem[ty][0] = m;
    md_smem[ty][1] = d;
  }
#pragma unroll
  for (uint32_t i = 0; i < VEC_SIZE; ++i) {
    o_smem[ty][lane_in_token * VEC_SIZE + i] = o_acc[i];
  }
  __syncthreads();

  if (ty == 0 && tx == 0) {
    float m_g = md_smem[0][0];
#pragma unroll
    for (uint32_t r = 1; r < BDY; ++r) {
      const float mr = md_smem[r][0];
      m_g = (mr > m_g) ? mr : m_g;
    }
    float d_g = 0.f;
#pragma unroll
    for (uint32_t r = 0; r < BDY; ++r) {
      d_g += md_smem[r][1] * exp2f(md_smem[r][0] - m_g);
    }
    global_md[0] = m_g;
    global_md[1] = d_g;
    if (K_SPLIT == 1 && lse_ptr != nullptr) {
      lse_ptr[b * H_q + qh] = (d_g > 0.f) ? (m_g + log2f(d_g)) : -CUDART_INF_F;
    }
    if (K_SPLIT > 1) {
      tmp_md_out[0] = m_g;
      tmp_md_out[1] = d_g;
    }
  }
  __syncthreads();

  const float m_g = global_md[0];
  const float d_g = global_md[1];

  // Combine BDY rows into final per-block output (this thread owns lanes
  // [tx*VEC_SIZE, (tx+1)*VEC_SIZE)).
  if (ty == 0) {
    fi::vec_t<float, VEC_SIZE> out_vec;
    out_vec.fill(0.f);
#pragma unroll
    for (uint32_t r = 0; r < BDY; ++r) {
      const float scale = exp2f(md_smem[r][0] - m_g) / d_g;
#pragma unroll
      for (uint32_t i = 0; i < VEC_SIZE; ++i) {
        out_vec[i] += o_smem[r][lane_in_token * VEC_SIZE + i] * scale;
      }
    }
    if (K_SPLIT == 1) {
      fi::vec_t<DType, VEC_SIZE> out_d;
#pragma unroll
      for (uint32_t i = 0; i < VEC_SIZE; ++i) {
        out_d[i] = DType(out_vec[i]);
      }
      out_d.store(o_out + feat_base);
    } else {
      // Write per-split partial to tmp_o (in float). Note: tmp_o is in
      // pre-normalised space (i.e. o_acc * exp2(m_r - m_g) summed) -- the
      // merge kernel rescales by global m again across splits.
      out_vec.store(tmp_o_out + feat_base);
    }
  }
}

// -----------------------------------------------------------------------------
// Merge kernel. One CTA per (b, qh). Combines K_SPLIT per-split (m, d, o)
// partials (each already normalised by its own d_block) into the final
// (m_g, d_g, o) using the standard online-softmax merge formula.
//
// For each split ks:
//   o_split[ks] = sum_t in split p_t / d_block_ks * V_t  (already rescaled by
//                 d_block in main kernel's final write).
//   d_block_ks  = sum_t in split p_t  (in the per-block-m frame).
//   m_block_ks  = max_t in split logit
//
// Combined:
//   m_g = max_ks m_block_ks
//   d_g = sum_ks d_block_ks * exp2(m_block_ks - m_g)
//   out = sum_ks o_split[ks] * d_block_ks * exp2(m_block_ks - m_g) / d_g
// -----------------------------------------------------------------------------
template <typename DType, uint32_t HEAD_DIM, uint32_t VEC_SIZE, uint32_t BDX>
__global__ void sparse_decode_merge_kernel(
    const float* __restrict__ tmp_o_ptr,    // [B, H_q, K_SPLIT, D]
    const float* __restrict__ tmp_md_ptr,   // [B, H_q, K_SPLIT, 2]
    DType* __restrict__ o_ptr,              // [B, H_q, D]
    float* __restrict__ lse_ptr,            // [B, H_q] or null
    uint32_t H_q, int K_SPLIT) {
  const uint32_t b = blockIdx.x;
  const uint32_t qh = blockIdx.y;
  const uint32_t tx = threadIdx.x;
  const uint32_t feat_base = tx * VEC_SIZE;

  const float* md_base = tmp_md_ptr + ((b * H_q + qh) * (uint32_t)K_SPLIT) * 2;
  const float* o_base = tmp_o_ptr + ((b * H_q + qh) * (uint32_t)K_SPLIT) * HEAD_DIM;

  // First pass: global max.
  float m_g = -CUDART_INF_F;
  for (int ks = 0; ks < K_SPLIT; ++ks) {
    const float mk = md_base[ks * 2 + 0];
    if (mk > m_g) m_g = mk;
  }
  // Second pass: global denom.
  float d_g = 0.f;
  for (int ks = 0; ks < K_SPLIT; ++ks) {
    const float mk = md_base[ks * 2 + 0];
    const float dk = md_base[ks * 2 + 1];
    d_g += dk * exp2f(mk - m_g);
  }

  if (lse_ptr != nullptr && tx == 0) {
    lse_ptr[b * H_q + qh] = (d_g > 0.f) ? (m_g + log2f(d_g)) : -CUDART_INF_F;
  }

  // Combine per-split o (already locally normalised).
  fi::vec_t<float, VEC_SIZE> out_vec;
  out_vec.fill(0.f);
  for (int ks = 0; ks < K_SPLIT; ++ks) {
    const float mk = md_base[ks * 2 + 0];
    const float dk = md_base[ks * 2 + 1];
    if (dk == 0.f) continue;
    const float scale = dk * exp2f(mk - m_g) / d_g;
    fi::vec_t<float, VEC_SIZE> o_part;
    o_part.load(o_base + (uint32_t)ks * HEAD_DIM + feat_base);
#pragma unroll
    for (uint32_t i = 0; i < VEC_SIZE; ++i) {
      out_vec[i] += o_part[i] * scale;
    }
  }
  fi::vec_t<DType, VEC_SIZE> out_d;
#pragma unroll
  for (uint32_t i = 0; i < VEC_SIZE; ++i) {
    out_d[i] = DType(out_vec[i]);
  }
  out_d.store(o_ptr + (b * H_q + qh) * HEAD_DIM + feat_base);
}

// -----------------------------------------------------------------------------
// Launcher: pick template constants based on dtype and head_dim.
// -----------------------------------------------------------------------------
template <typename DType, uint32_t HEAD_DIM, bool USE_SOFT_CAP>
static void launch_sparse_decode(
    const DType* q_ptr, const DType* k_data, const DType* v_data,
    const int32_t* indptr, const int32_t* indices,
    const int32_t* sparse_len_ptr, const int64_t* sparse_idx_ptr,
    const float* sparse_w_ptr, DType* o_ptr, float* lse_ptr,
    float* tmp_o_ptr, float* tmp_md_ptr,
    int B, int H_q, int H_kv, int page_size, int max_S, int K_SPLIT,
    int64_t kv_stride_page, int64_t kv_stride_n, int64_t kv_stride_h,
    float sm_scale, float soft_cap, cudaStream_t stream) {
  constexpr uint32_t VEC_SIZE = std::max<uint32_t>(16U / sizeof(DType), HEAD_DIM / 32U);
  constexpr uint32_t BDX = HEAD_DIM / VEC_SIZE;
  static_assert(BDX <= 32, "BDX must be <= 32");
  constexpr uint32_t BDY = (BDX <= 16) ? (128 / BDX) : 4;
  static_assert(BDX * BDY <= 1024, "block too large");

  float sm_scale_log2;
  float pre_tanh_scale = 0.f;
  if (USE_SOFT_CAP) {
    sm_scale_log2 = fi::math::log2e;
    pre_tanh_scale = sm_scale / soft_cap;
  } else {
    sm_scale_log2 = sm_scale * fi::math::log2e;
  }

  dim3 grid(B, H_q, K_SPLIT);
  dim3 block(BDX, BDY);
  auto kernel = sparse_decode_kernel<DType, HEAD_DIM, VEC_SIZE, BDX, BDY, USE_SOFT_CAP>;
  kernel<<<grid, block, 0, stream>>>(
      q_ptr, k_data, v_data, indptr, indices, sparse_len_ptr, sparse_idx_ptr,
      sparse_w_ptr, o_ptr, lse_ptr, tmp_o_ptr, tmp_md_ptr,
      (uint32_t)H_q, (uint32_t)H_kv, (uint32_t)page_size, (uint32_t)max_S,
      kv_stride_page, kv_stride_n, kv_stride_h, K_SPLIT,
      sm_scale_log2, soft_cap, pre_tanh_scale);

  if (K_SPLIT > 1) {
    auto merge = sparse_decode_merge_kernel<DType, HEAD_DIM, VEC_SIZE, BDX>;
    dim3 mgrid(B, H_q);
    dim3 mblock(BDX);
    merge<<<mgrid, mblock, 0, stream>>>(
        tmp_o_ptr, tmp_md_ptr, o_ptr, lse_ptr, (uint32_t)H_q, K_SPLIT);
  }
}

#define DISPATCH_HD(HD_VAR, HD, ...)         \
  switch ((HD_VAR)) {                        \
    case 64: { constexpr uint32_t HD = 64; __VA_ARGS__; break; }  \
    case 128: { constexpr uint32_t HD = 128; __VA_ARGS__; break; } \
    case 256: { constexpr uint32_t HD = 256; __VA_ARGS__; break; } \
    default: TORCH_CHECK(false, "unsupported head_dim ", (HD_VAR));  \
  }

#define DISPATCH_BOOL(BVAR, NAME, ...)         \
  do {                                         \
    if ((BVAR)) {                              \
      constexpr bool NAME = true; __VA_ARGS__; \
    } else {                                   \
      constexpr bool NAME = false; __VA_ARGS__; \
    }                                          \
  } while (0)

// -----------------------------------------------------------------------------
// Public entry point.
// -----------------------------------------------------------------------------
std::vector<torch::Tensor> sparse_decode_run(
    torch::Tensor q,
    torch::Tensor k_cache,
    torch::Tensor v_cache,
    int64_t kv_stride_page,
    int64_t kv_stride_n,
    int64_t kv_stride_h,
    int64_t kv_v_offset_elem,
    torch::Tensor indptr,
    torch::Tensor indices,
    torch::Tensor sparse_len,
    torch::Tensor sparse_idx,
    torch::Tensor sparse_weights,
    int64_t H_kv_arg,
    int64_t page_size_arg,
    double sm_scale,
    double soft_cap,
    int64_t k_split_arg,
    bool return_lse) {
  CHECK_CUDA(q);
  CHECK_CONTIG(q);
  CHECK_CUDA(k_cache);
  CHECK_CUDA(v_cache);
  CHECK_CUDA(indptr);
  CHECK_CONTIG(indptr);
  CHECK_CUDA(indices);
  CHECK_CONTIG(indices);
  CHECK_CUDA(sparse_len);
  CHECK_CONTIG(sparse_len);
  CHECK_CUDA(sparse_idx);
  CHECK_CONTIG(sparse_idx);
  CHECK_CUDA(sparse_weights);
  CHECK_CONTIG(sparse_weights);

  TORCH_CHECK(q.dim() == 3, "q must be [B, H_q, D]");
  TORCH_CHECK(indptr.dtype() == torch::kInt32, "indptr must be int32");
  TORCH_CHECK(indices.dtype() == torch::kInt32, "indices must be int32");
  TORCH_CHECK(sparse_len.dtype() == torch::kInt32, "sparse_len must be int32");
  TORCH_CHECK(sparse_idx.dtype() == torch::kInt64, "sparse_idx must be int64");
  TORCH_CHECK(sparse_weights.dtype() == torch::kFloat32,
              "sparse_weights must be float32");

  const int B = (int)q.size(0);
  const int H_q = (int)q.size(1);
  const int HEAD_DIM = (int)q.size(2);
  const int H_kv = (int)H_kv_arg;
  const int page_size = (int)page_size_arg;
  const int max_S = (int)sparse_idx.size(2);
  const int K_SPLIT = std::max<int>(1, (int)k_split_arg);
  TORCH_CHECK(H_q % H_kv == 0, "H_q must be a multiple of H_kv");
  TORCH_CHECK(sparse_len.size(0) == B && sparse_len.size(1) == H_q,
              "sparse_len shape mismatch");
  TORCH_CHECK(sparse_idx.size(0) == B && sparse_idx.size(1) == H_q,
              "sparse_idx shape mismatch");
  TORCH_CHECK(sparse_weights.sizes() == sparse_idx.sizes(),
              "sparse_weights shape mismatch");

  auto out = torch::empty_like(q);
  torch::Tensor lse;
  if (return_lse) {
    lse = torch::empty({B, H_q}, q.options().dtype(torch::kFloat32));
  } else {
    lse = torch::empty({0}, q.options().dtype(torch::kFloat32));
  }

  // Tmp buffers (only needed when K_SPLIT > 1). Allocated as float for
  // numerical precision in the merge.
  torch::Tensor tmp_o, tmp_md;
  if (K_SPLIT > 1) {
    tmp_o = torch::empty({B, H_q, K_SPLIT, HEAD_DIM},
                         q.options().dtype(torch::kFloat32));
    tmp_md = torch::empty({B, H_q, K_SPLIT, 2},
                          q.options().dtype(torch::kFloat32));
  }
  float* tmp_o_ptr = (K_SPLIT > 1) ? tmp_o.data_ptr<float>() : nullptr;
  float* tmp_md_ptr = (K_SPLIT > 1) ? tmp_md.data_ptr<float>() : nullptr;

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const bool use_soft_cap = soft_cap > 0.f;

  if (q.scalar_type() == torch::kHalf) {
    auto* qp = reinterpret_cast<const __half*>(q.data_ptr<at::Half>());
    auto* kp = reinterpret_cast<const __half*>(k_cache.data_ptr<at::Half>());
    auto* vp = reinterpret_cast<const __half*>(v_cache.data_ptr<at::Half>()) +
               kv_v_offset_elem;
    auto* outp = reinterpret_cast<__half*>(out.data_ptr<at::Half>());
    float* lsep = return_lse ? lse.data_ptr<float>() : nullptr;
    DISPATCH_HD(HEAD_DIM, HD, {
      DISPATCH_BOOL(use_soft_cap, USE_CAP, {
        launch_sparse_decode<__half, HD, USE_CAP>(
            qp, kp, vp, indptr.data_ptr<int32_t>(), indices.data_ptr<int32_t>(),
            sparse_len.data_ptr<int32_t>(), sparse_idx.data_ptr<int64_t>(),
            sparse_weights.data_ptr<float>(), outp, lsep,
            tmp_o_ptr, tmp_md_ptr,
            B, H_q, H_kv, page_size, max_S, K_SPLIT,
            kv_stride_page, kv_stride_n, kv_stride_h,
            (float)sm_scale, (float)soft_cap, stream);
      });
    });
  } else if (q.scalar_type() == torch::kBFloat16) {
    auto* qp = reinterpret_cast<const __nv_bfloat16*>(q.data_ptr<at::BFloat16>());
    auto* kp = reinterpret_cast<const __nv_bfloat16*>(k_cache.data_ptr<at::BFloat16>());
    auto* vp = reinterpret_cast<const __nv_bfloat16*>(v_cache.data_ptr<at::BFloat16>()) +
               kv_v_offset_elem;
    auto* outp = reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>());
    float* lsep = return_lse ? lse.data_ptr<float>() : nullptr;
    DISPATCH_HD(HEAD_DIM, HD, {
      DISPATCH_BOOL(use_soft_cap, USE_CAP, {
        launch_sparse_decode<__nv_bfloat16, HD, USE_CAP>(
            qp, kp, vp, indptr.data_ptr<int32_t>(), indices.data_ptr<int32_t>(),
            sparse_len.data_ptr<int32_t>(), sparse_idx.data_ptr<int64_t>(),
            sparse_weights.data_ptr<float>(), outp, lsep,
            tmp_o_ptr, tmp_md_ptr,
            B, H_q, H_kv, page_size, max_S, K_SPLIT,
            kv_stride_page, kv_stride_n, kv_stride_h,
            (float)sm_scale, (float)soft_cap, stream);
      });
    });
  } else {
    TORCH_CHECK(false, "Only fp16/bf16 supported for q");
  }

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {out, lse};
}

// -----------------------------------------------------------------------------
// Plan helper. Picks a K_SPLIT value at plan time given (B, H_q, max_S).
// Aim for ~2 CTAs per SM after split. Bound by max_S so we don't split into
// chunks smaller than ~16 tokens (degenerate splits hurt the merge).
// -----------------------------------------------------------------------------
int64_t pick_k_split(int64_t B, int64_t H_q, int64_t max_S, int64_t target_blocks_per_sm) {
  int sm_count = 132;  // H100 default
  cudaDeviceProp prop;
  int dev = 0;
  if (cudaGetDevice(&dev) == cudaSuccess && cudaGetDeviceProperties(&prop, dev) == cudaSuccess) {
    sm_count = prop.multiProcessorCount;
  }
  const int64_t want = (int64_t)sm_count * std::max<int64_t>(1, target_blocks_per_sm);
  const int64_t bH = std::max<int64_t>(1, B * H_q);
  int64_t k = (want + bH - 1) / bH;
  // No point splitting into smaller than ~16-token chunks.
  const int64_t min_chunk = 16;
  if (max_S > 0) {
    int64_t k_max_by_chunk = (max_S + min_chunk - 1) / min_chunk;
    if (k_max_by_chunk < 1) k_max_by_chunk = 1;
    if (k > k_max_by_chunk) k = k_max_by_chunk;
  }
  if (k < 1) k = 1;
  if (k > 32) k = 32;
  return k;
}

}  // namespace sparse_decode

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("sparse_decode_run", &sparse_decode::sparse_decode_run,
        "Standalone paged batch decode with per-(b, qh) sparse selection.");
  m.def("pick_k_split", &sparse_decode::pick_k_split,
        "Choose K_SPLIT given (B, H_q, max_S, target_blocks_per_sm).");
}
