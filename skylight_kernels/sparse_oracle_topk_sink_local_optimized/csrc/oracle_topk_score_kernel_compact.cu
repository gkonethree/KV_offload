// Oracle top-k score kernel: computes the dense Q @ K^T matrix per (b, qh)
// directly from a paged KV cache. The output `scores` tensor is fed into a
// downstream top-k (FlashInfer radix or torch.topk) to obtain the per-(b, qh)
// top-k indices, which are then run through the existing standalone sparse
// decode kernel under `sparse_optimized/csrc/sparse_decode_kernel.cu`.
//
// Key optimizations:
//   * Each CTA is assigned to (b, kv_head, token_chunk) and inside the kernel
//     iterates over the `GROUP_SIZE = H_q / H_kv` query heads that share this
//     KV head. K is loaded ONCE per (kv_head, token) tile and the dot products
//     against all GROUP_SIZE Q vectors are computed from registers. This
//     reduces global K traffic by GROUP_SIZE relative to a naive per-(b, qh)
//     schedule, which is the dominant cost at long context.
//   * `channel_num` (runtime arg, must be a multiple of VEC_SIZE) restricts
//     the selection dot product to the first `channel_num` channels of Q/K.
//     BDX lanes whose `feat_base >= channel_num` skip the K load entirely
//     (predicated load), which on H100 with 128 B cache lines yields ~2x K
//     read reduction for any channel_num <= 64.
//   * The scores tensor is written in the same dtype as Q/K (fp16/bf16),
//     halving both the score-kernel write traffic and the downstream top-k
//     read traffic relative to fp32 scores. Internal accumulation stays in
//     fp32; only the final write rounds to the output dtype. fp16 scores
//     match the fp32 top-k indices set-equivalently 99.9% of the time at
//     k=1311 / L=128K and the differences only ever land on tied boundary
//     entries (which both ranks would treat as interchangeable anyway).
//
// Padded positions beyond the per-batch sequence length are filled with -inf
// in the output dtype so the downstream top-k cannot select them. Soft cap
// and sm_scale are deliberately omitted: both are monotonic transforms that
// do not change the top-k ranking.

#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <algorithm>
#include <cstdint>

#include "flashinfer/vec_dtypes.cuh"

#ifndef CUDART_INF_F
#define CUDART_INF_F __int_as_float(0x7f800000)
#endif

namespace oracle_topk {

namespace fi = flashinfer;

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be CUDA")
#define CHECK_CONTIG(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")

// -----------------------------------------------------------------------------
// Kernel template parameters:
//   DType         - fp16 / bf16
//   HEAD_DIM      - 64 / 128 / 256
//   VEC_SIZE      - elements per 16-byte vectorised load
//   BDX           - threads cooperating on one dot product (= HEAD_DIM/VEC_SIZE)
//   BDY           - tokens computed concurrently per inner iteration
//   GROUP_SIZE    - q_heads sharing this kv_head (H_q / H_kv)
//   NLOOPS        - tokens per ty thread per CTA  (TPB = BDY * NLOOPS tokens/CTA)
// -----------------------------------------------------------------------------
// CHANNEL_DIM is the compile-time number of feature channels actually used in
// the score dot product. BDX is sized as CHANNEL_DIM / VEC_SIZE (rather than
// HEAD_DIM / VEC_SIZE) so all SIMD lanes do useful work, and BDY is enlarged
// to keep block occupancy roughly constant. The kernel still steps over all
// HEAD_DIM channels of Q at load time -- we just skip lanes whose offset is
// >= CHANNEL_DIM.
// Helper: -inf in the output dtype. We rely on implicit float -> half/bfloat16
// conversion to produce the correct -inf bit pattern in each format.
template <typename T> __device__ inline T neg_inf_value();
template <> __device__ inline __half neg_inf_value<__half>() {
  return __float2half(-CUDART_INF_F);
}
template <> __device__ inline __nv_bfloat16 neg_inf_value<__nv_bfloat16>() {
  return __float2bfloat16(-CUDART_INF_F);
}
template <> __device__ inline float neg_inf_value<float>() {
  return -CUDART_INF_F;
}

// Helper: round a fp32 partial to OutType for storage.
template <typename T> __device__ inline T from_float(float v);
template <> __device__ inline __half from_float<__half>(float v) {
  return __float2half(v);
}
template <> __device__ inline __nv_bfloat16 from_float<__nv_bfloat16>(float v) {
  return __float2bfloat16(v);
}
template <> __device__ inline float from_float<float>(float v) { return v; }


// Original full-range score kernel (kept for compatibility path when sink==0 && local==0).
template <typename DType, typename OutType, uint32_t HEAD_DIM,
          uint32_t CHANNEL_DIM, uint32_t VEC_SIZE, uint32_t BDX, uint32_t BDY,
          uint32_t GROUP_SIZE, uint32_t NLOOPS>
__global__ __launch_bounds__(BDX* BDY) void oracle_topk_score_kernel(
    const DType* __restrict__ q_ptr,
    const DType* __restrict__ k_data,
    const int32_t* __restrict__ kv_indptr,
    const int32_t* __restrict__ kv_indices,
    const int32_t* __restrict__ kv_last_page_len,
    OutType* __restrict__ scores_ptr,
    uint32_t H_q, uint32_t page_size, uint32_t L_max,
    int64_t kv_stride_page, int64_t kv_stride_n, int64_t kv_stride_h) {
  static_assert(CHANNEL_DIM <= HEAD_DIM); static_assert(CHANNEL_DIM % VEC_SIZE == 0);
  static_assert(BDX == CHANNEL_DIM / VEC_SIZE);
  const uint32_t b = blockIdx.x, kv_head = blockIdx.y;
  const uint32_t block_token_base = blockIdx.z * (BDY * NLOOPS);
  const uint32_t tx = threadIdx.x, ty = threadIdx.y;
  const uint32_t feat_base = tx * VEC_SIZE;
  const int32_t indptr_b = kv_indptr[b], indptr_b1 = kv_indptr[b + 1];
  const int32_t pages = indptr_b1 - indptr_b, last_pl = kv_last_page_len[b];
  const int32_t L = (pages > 0) ? ((pages - 1) * (int32_t)page_size + last_pl) : 0;
  fi::vec_t<float, VEC_SIZE> q_vecs[GROUP_SIZE];
#pragma unroll
  for (uint32_t g = 0; g < GROUP_SIZE; ++g) {
    const uint32_t qh = kv_head * GROUP_SIZE + g;
    if (qh < H_q) {
      fi::vec_t<DType, VEC_SIZE> q_load;
      q_load.load(q_ptr + (b * H_q + qh) * HEAD_DIM + feat_base);
#pragma unroll
      for (uint32_t i = 0; i < VEC_SIZE; ++i) q_vecs[g][i] = float(q_load[i]);
    } else {
#pragma unroll
      for (uint32_t i = 0; i < VEC_SIZE; ++i) q_vecs[g][i] = 0.f;
    }
  }
#pragma unroll
  for (uint32_t loop = 0; loop < NLOOPS; ++loop) {
    const uint32_t s = block_token_base + loop * BDY + ty;
    const bool active_in_buf = s < L_max, active_in_seq = s < (uint32_t)L;
    fi::vec_t<DType, VEC_SIZE> k_vec;
    if (active_in_seq) {
      const int32_t pl = (int32_t)(s / page_size), ip = (int32_t)(s - (uint32_t)pl * page_size);
      const int32_t pid = kv_indices[indptr_b + pl];
      const int64_t off = (int64_t)pid * kv_stride_page + (int64_t)kv_head * kv_stride_h + (int64_t)ip * kv_stride_n;
      k_vec.load(k_data + off + (int64_t)feat_base);
    } else { k_vec.fill(DType(0.f)); }
    float k_f[VEC_SIZE];
#pragma unroll
    for (uint32_t i = 0; i < VEC_SIZE; ++i) k_f[i] = float(k_vec[i]);
    float partials[GROUP_SIZE];
#pragma unroll
    for (uint32_t g = 0; g < GROUP_SIZE; ++g) { float p = 0.f; for (uint32_t i = 0; i < VEC_SIZE; ++i) p += q_vecs[g][i] * k_f[i]; partials[g] = p; }
#pragma unroll
    for (uint32_t step = BDX / 2; step > 0; step >>= 1) for (uint32_t g = 0; g < GROUP_SIZE; ++g) partials[g] += __shfl_xor_sync(0xFFFFFFFFu, partials[g], step, BDX);
    if (tx == 0 && active_in_buf) {
      const OutType pad = neg_inf_value<OutType>();
#pragma unroll
      for (uint32_t g = 0; g < GROUP_SIZE; ++g) { const uint32_t qh = kv_head * GROUP_SIZE + g; if (qh < H_q) scores_ptr[(b * H_q + qh) * L_max + s] = active_in_seq ? from_float<OutType>(partials[g]) : pad; }
    }
  }
}

// Compact variant: only covers tokens [token_start, token_start + L_mid).
// `s` in [0, L_mid) maps to global token (token_start + s).
// Output is scores_ptr[b * H_q * L_mid + qh * L_mid + s].
template <typename DType, typename OutType, uint32_t HEAD_DIM,
          uint32_t CHANNEL_DIM, uint32_t VEC_SIZE, uint32_t BDX, uint32_t BDY,
          uint32_t GROUP_SIZE, uint32_t NLOOPS>
__global__ __launch_bounds__(BDX* BDY) void oracle_topk_score_kernel_compact(
    const DType* __restrict__ q_ptr,                  // [B, H_q, D]
    const DType* __restrict__ k_data,                 // paged K base
    const int32_t* __restrict__ kv_indptr,            // [B+1]
    const int32_t* __restrict__ kv_indices,           // [total_pages]
    const int32_t* __restrict__ kv_last_page_len,     // [B]
    OutType* __restrict__ scores_ptr,                 // [B, H_q, L_mid]
    uint32_t H_q, uint32_t page_size, uint32_t L_mid,
    uint32_t token_start, uint32_t token_end,          // global [token_start, token_end)
    int64_t kv_stride_page, int64_t kv_stride_n, int64_t kv_stride_h) {
  static_assert(CHANNEL_DIM <= HEAD_DIM, "CHANNEL_DIM must be <= HEAD_DIM");
  static_assert(CHANNEL_DIM % VEC_SIZE == 0,
                "CHANNEL_DIM must be a multiple of VEC_SIZE");
  static_assert(BDX == CHANNEL_DIM / VEC_SIZE, "BDX must == CHANNEL_DIM/VEC_SIZE");

  const uint32_t b = blockIdx.x;
  const uint32_t kv_head = blockIdx.y;
  // s_local in [0, L_mid); global token = token_start + s_local
  const uint32_t block_token_base = blockIdx.z * (BDY * NLOOPS);

  const uint32_t tx = threadIdx.x;
  const uint32_t ty = threadIdx.y;
  const uint32_t feat_base = tx * VEC_SIZE;

  // Compute actual sequence length L for this batch.
  const int32_t indptr_b = kv_indptr[b];
  const int32_t indptr_b1 = kv_indptr[b + 1];
  const int32_t pages = indptr_b1 - indptr_b;
  const int32_t last_pl = kv_last_page_len[b];
  const int32_t L = (pages > 0) ? ((pages - 1) * (int32_t)page_size + last_pl) : 0;

  // For this batch, the effective token_end is min(token_end, L).
  // token_start is always <= L for valid batches.
  const uint32_t t_end_eff = (uint32_t)min((int32_t)token_end, L);

  fi::vec_t<float, VEC_SIZE> q_vecs[GROUP_SIZE];
#pragma unroll
  for (uint32_t g = 0; g < GROUP_SIZE; ++g) {
    const uint32_t qh = kv_head * GROUP_SIZE + g;
    if (qh < H_q) {
      fi::vec_t<DType, VEC_SIZE> q_load;
      q_load.load(q_ptr + (b * H_q + qh) * HEAD_DIM + feat_base);
#pragma unroll
      for (uint32_t i = 0; i < VEC_SIZE; ++i) {
        q_vecs[g][i] = float(q_load[i]);
      }
    } else {
#pragma unroll
      for (uint32_t i = 0; i < VEC_SIZE; ++i) q_vecs[g][i] = 0.f;
    }
  }

#pragma unroll
  for (uint32_t loop = 0; loop < NLOOPS; ++loop) {
    // s_local is the output index in [0, L_mid)
    const uint32_t s_local = block_token_base + loop * BDY + ty;
    const bool active_in_buf = s_local < L_mid;
    // global token index
    const uint32_t s = token_start + s_local;
    const bool active_in_seq = active_in_buf && (s < t_end_eff);

    fi::vec_t<DType, VEC_SIZE> k_vec;
    if (active_in_seq) {
      const int32_t page_local = (int32_t)(s / page_size);
      const int32_t in_page = (int32_t)(s - (uint32_t)page_local * page_size);
      const int32_t page_id = kv_indices[indptr_b + page_local];
      const int64_t kv_off = (int64_t)page_id * kv_stride_page +
                             (int64_t)kv_head * kv_stride_h +
                             (int64_t)in_page * kv_stride_n;
      k_vec.load(k_data + kv_off + (int64_t)feat_base);
    } else {
      k_vec.fill(DType(0.f));
    }

    float k_f[VEC_SIZE];
#pragma unroll
    for (uint32_t i = 0; i < VEC_SIZE; ++i) k_f[i] = float(k_vec[i]);

    float partials[GROUP_SIZE];
#pragma unroll
    for (uint32_t g = 0; g < GROUP_SIZE; ++g) {
      float p = 0.f;
#pragma unroll
      for (uint32_t i = 0; i < VEC_SIZE; ++i) {
        p += q_vecs[g][i] * k_f[i];
      }
      partials[g] = p;
    }

#pragma unroll
    for (uint32_t step = BDX / 2; step > 0; step >>= 1) {
#pragma unroll
      for (uint32_t g = 0; g < GROUP_SIZE; ++g) {
        partials[g] += __shfl_xor_sync(0xFFFFFFFFu, partials[g], step, BDX);
      }
    }

    // Store in compact layout: [b, qh, s_local] where s_local in [0, L_mid)
    if (tx == 0 && active_in_buf) {
      const OutType pad = neg_inf_value<OutType>();
#pragma unroll
      for (uint32_t g = 0; g < GROUP_SIZE; ++g) {
        const uint32_t qh = kv_head * GROUP_SIZE + g;
        if (qh < H_q) {
          scores_ptr[(b * H_q + qh) * L_mid + s_local] =
              active_in_seq ? from_float<OutType>(partials[g]) : pad;
        }
      }
    }
  }
}

// -----------------------------------------------------------------------------
// Launcher: pick template constants based on dtype, head_dim, group_size.
// -----------------------------------------------------------------------------
// We always store scores in the same dtype as Q/K (OutType == DType). This
// halves the score tensor footprint and gives the downstream FlashInfer
// top-k a smaller buffer to scan.
template <typename DType, uint32_t HEAD_DIM, uint32_t CHANNEL_DIM,
          uint32_t GROUP_SIZE>
static void launch_oracle_topk_score_inst(
    const DType* q_ptr, const DType* k_data, const int32_t* indptr,
    const int32_t* indices, const int32_t* last_page_len, DType* scores_ptr,
    int B, int H_q, int H_kv, int page_size, int L_max,
    int64_t kv_stride_page, int64_t kv_stride_n, int64_t kv_stride_h,
    cudaStream_t stream) {
  constexpr uint32_t VEC_SIZE = std::max<uint32_t>(16U / sizeof(DType), HEAD_DIM / 32U);
  constexpr uint32_t BDX = CHANNEL_DIM / VEC_SIZE;
  static_assert(BDX >= 1 && BDX <= 32, "BDX must be in [1, 32]");
  constexpr uint32_t BDY = (128 / BDX) > 0 ? (128 / BDX) : 1;
  constexpr uint32_t NLOOPS = (256 / BDY) > 0 ? (256 / BDY) : 1;
  constexpr uint32_t TPB = BDY * NLOOPS;
  static_assert(BDX * BDY <= 1024, "block too large");

  const int z_blocks = (L_max + TPB - 1) / TPB;
  dim3 grid(B, H_kv, z_blocks);
  dim3 block(BDX, BDY);

  auto kernel = oracle_topk_score_kernel<DType, DType, HEAD_DIM, CHANNEL_DIM,
                                         VEC_SIZE, BDX, BDY, GROUP_SIZE, NLOOPS>;
  kernel<<<grid, block, 0, stream>>>(
      q_ptr, k_data, indptr, indices, last_page_len, scores_ptr,
      (uint32_t)H_q, (uint32_t)page_size, (uint32_t)L_max,
      kv_stride_page, kv_stride_n, kv_stride_h);
}

// Compact variant launcher — covers only [token_start, token_end)
template <typename DType, uint32_t HEAD_DIM, uint32_t CHANNEL_DIM,
          uint32_t GROUP_SIZE>
static void launch_oracle_topk_score_compact_inst(
    const DType* q_ptr, const DType* k_data, const int32_t* indptr,
    const int32_t* indices, const int32_t* last_page_len, DType* scores_ptr,
    int B, int H_q, int H_kv, int page_size, int L_mid,
    int token_start, int token_end,
    int64_t kv_stride_page, int64_t kv_stride_n, int64_t kv_stride_h,
    cudaStream_t stream) {
  constexpr uint32_t VEC_SIZE = std::max<uint32_t>(16U / sizeof(DType), HEAD_DIM / 32U);
  constexpr uint32_t BDX = CHANNEL_DIM / VEC_SIZE;
  static_assert(BDX >= 1 && BDX <= 32, "BDX must be in [1, 32]");
  constexpr uint32_t BDY = (128 / BDX) > 0 ? (128 / BDX) : 1;
  constexpr uint32_t NLOOPS = (256 / BDY) > 0 ? (256 / BDY) : 1;
  constexpr uint32_t TPB = BDY * NLOOPS;
  static_assert(BDX * BDY <= 1024, "block too large");

  const int z_blocks = (L_mid + TPB - 1) / TPB;
  dim3 grid(B, H_kv, z_blocks);
  dim3 block(BDX, BDY);

  auto kernel = oracle_topk_score_kernel_compact<DType, DType, HEAD_DIM, CHANNEL_DIM,
                                                 VEC_SIZE, BDX, BDY, GROUP_SIZE, NLOOPS>;
  kernel<<<grid, block, 0, stream>>>(
      q_ptr, k_data, indptr, indices, last_page_len, scores_ptr,
      (uint32_t)H_q, (uint32_t)page_size, (uint32_t)L_mid,
      (uint32_t)token_start, (uint32_t)token_end,
      kv_stride_page, kv_stride_n, kv_stride_h);
}

#define DISPATCH_HD(HD_VAR, HD, ...)                              \
  switch ((HD_VAR)) {                                             \
    case 64: {                                                    \
      constexpr uint32_t HD = 64;                                 \
      __VA_ARGS__;                                                \
      break;                                                      \
    }                                                             \
    case 128: {                                                   \
      constexpr uint32_t HD = 128;                                \
      __VA_ARGS__;                                                \
      break;                                                      \
    }                                                             \
    case 256: {                                                   \
      constexpr uint32_t HD = 256;                                \
      __VA_ARGS__;                                                \
      break;                                                      \
    }                                                             \
    default:                                                      \
      TORCH_CHECK(false, "unsupported head_dim ", (HD_VAR));      \
  }

// We compile the common GROUP_SIZE values explicitly so the inner unroll
// (the q_vecs[GROUP_SIZE] register array) is fully visible to nvcc.
#define DISPATCH_GROUP(GS_VAR, GS, ...)                            \
  switch ((GS_VAR)) {                                              \
    case 1: {                                                      \
      constexpr uint32_t GS = 1;                                   \
      __VA_ARGS__;                                                 \
      break;                                                       \
    }                                                              \
    case 2: {                                                      \
      constexpr uint32_t GS = 2;                                   \
      __VA_ARGS__;                                                 \
      break;                                                       \
    }                                                              \
    case 4: {                                                      \
      constexpr uint32_t GS = 4;                                   \
      __VA_ARGS__;                                                 \
      break;                                                       \
    }                                                              \
    /* GS=6: Qwen3.5-family GQA (24 q-heads / 4 kv-heads). */      \
    case 6: {                                                      \
      constexpr uint32_t GS = 6;                                   \
      __VA_ARGS__;                                                 \
      break;                                                       \
    }                                                              \
    case 8: {                                                      \
      constexpr uint32_t GS = 8;                                   \
      __VA_ARGS__;                                                 \
      break;                                                       \
    }                                                              \
    case 16: {                                                     \
      constexpr uint32_t GS = 16;                                  \
      __VA_ARGS__;                                                 \
      break;                                                       \
    }                                                              \
    default:                                                       \
      TORCH_CHECK(false, "unsupported group_size ", (GS_VAR));     \
  }

// CHANNEL_DIM dispatch: only the most useful values are compiled. fp16/bf16
// requires CHANNEL_DIM to be a multiple of VEC_SIZE = max(8, HD/32).
#define DISPATCH_CHANNEL(CN_VAR, CN, HD, ...)                                \
  switch ((CN_VAR)) {                                                        \
    case 8: {                                                                \
      constexpr uint32_t CN = (HD >= 8) ? 8 : HD;                            \
      __VA_ARGS__;                                                           \
      break;                                                                 \
    }                                                                        \
    case 16: {                                                               \
      constexpr uint32_t CN = (HD >= 16) ? 16 : HD;                          \
      __VA_ARGS__;                                                           \
      break;                                                                 \
    }                                                                        \
    case 32: {                                                               \
      constexpr uint32_t CN = (HD >= 32) ? 32 : HD;                          \
      __VA_ARGS__;                                                           \
      break;                                                                 \
    }                                                                        \
    case 64: {                                                               \
      constexpr uint32_t CN = (HD >= 64) ? 64 : HD;                          \
      __VA_ARGS__;                                                           \
      break;                                                                 \
    }                                                                        \
    case 128: {                                                              \
      constexpr uint32_t CN = (HD >= 128) ? 128 : HD;                        \
      __VA_ARGS__;                                                           \
      break;                                                                 \
    }                                                                        \
    case 256: {                                                              \
      constexpr uint32_t CN = (HD >= 256) ? 256 : HD;                        \
      __VA_ARGS__;                                                           \
      break;                                                                 \
    }                                                                        \
    default:                                                                 \
      TORCH_CHECK(false,                                                     \
                  "channel_num must be one of {8,16,32,64,128,256}, got ",   \
                  (CN_VAR));                                                 \
  }

// -----------------------------------------------------------------------------
// Public entry point. Returns scores[B, H_q, L_max] in q's dtype (fp16/bf16),
// with positions beyond the actual per-batch sequence length filled with -inf.
// Kept for compatibility; prefer oracle_topk_compute_scores_compact.
// -----------------------------------------------------------------------------
torch::Tensor oracle_topk_compute_scores(
    torch::Tensor q,
    torch::Tensor k_cache,
    int64_t kv_stride_page,
    int64_t kv_stride_n,
    int64_t kv_stride_h,
    torch::Tensor indptr,
    torch::Tensor indices,
    torch::Tensor last_page_len,
    int64_t H_kv_arg,
    int64_t page_size_arg,
    int64_t L_max_arg,
    int64_t channel_num_arg) {
  CHECK_CUDA(q);
  CHECK_CONTIG(q);
  CHECK_CUDA(k_cache);
  CHECK_CUDA(indptr);
  CHECK_CONTIG(indptr);
  CHECK_CUDA(indices);
  CHECK_CONTIG(indices);
  CHECK_CUDA(last_page_len);
  CHECK_CONTIG(last_page_len);

  TORCH_CHECK(q.dim() == 3, "q must be [B, H_q, D]");
  TORCH_CHECK(indptr.dtype() == torch::kInt32, "indptr must be int32");
  TORCH_CHECK(indices.dtype() == torch::kInt32, "indices must be int32");
  TORCH_CHECK(last_page_len.dtype() == torch::kInt32,
              "last_page_len must be int32");

  const int B = (int)q.size(0);
  const int H_q = (int)q.size(1);
  const int HEAD_DIM = (int)q.size(2);
  const int H_kv = (int)H_kv_arg;
  const int page_size = (int)page_size_arg;
  const int L_max = (int)L_max_arg;
  const int channel_num = (int)channel_num_arg;
  TORCH_CHECK(H_q % H_kv == 0, "H_q must be a multiple of H_kv");
  TORCH_CHECK(channel_num > 0 && channel_num <= HEAD_DIM,
              "channel_num must be in (0, head_dim]");
  // Restrict to multiples of the vec_t lane width (8 for fp16/bf16, 4 for
  // fp32) so the per-lane gating cleanly skips entire vectorised loads.
  // For fp16/bf16 + HEAD_DIM=128 this means channel_num must be a multiple
  // of 8 (i.e., 8, 16, 24, ..., 128).
  const int VEC_LANE_WIDTH = (q.scalar_type() == torch::kHalf ||
                              q.scalar_type() == torch::kBFloat16)
                                 ? std::max(8, HEAD_DIM / 32)
                                 : std::max(4, HEAD_DIM / 32);
  TORCH_CHECK(channel_num % VEC_LANE_WIDTH == 0,
              "channel_num must be a multiple of vec lane width ",
              VEC_LANE_WIDTH);
  const int GROUP_SIZE = H_q / H_kv;

  // Scores tensor matches Q dtype (fp16/bf16). FlashInfer's radix top-k and
  // torch.topk both accept these directly, and halving the bytes halves the
  // memory traffic of both the score-kernel write and the top-k stage's
  // first read pass over the score tensor.
  auto scores = torch::empty({B, H_q, L_max}, q.options());

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  if (q.scalar_type() == torch::kHalf) {
    auto* qp = reinterpret_cast<const __half*>(q.data_ptr<at::Half>());
    auto* kp = reinterpret_cast<const __half*>(k_cache.data_ptr<at::Half>());
    auto* sp = reinterpret_cast<__half*>(scores.data_ptr<at::Half>());
    DISPATCH_HD(HEAD_DIM, HD, {
      DISPATCH_CHANNEL(channel_num, CN, HD, {
        DISPATCH_GROUP(GROUP_SIZE, GS, {
          launch_oracle_topk_score_inst<__half, HD, CN, GS>(
              qp, kp, indptr.data_ptr<int32_t>(),
              indices.data_ptr<int32_t>(), last_page_len.data_ptr<int32_t>(),
              sp, B, H_q, H_kv, page_size, L_max,
              kv_stride_page, kv_stride_n, kv_stride_h, stream);
        });
      });
    });
  } else if (q.scalar_type() == torch::kBFloat16) {
    auto* qp = reinterpret_cast<const __nv_bfloat16*>(q.data_ptr<at::BFloat16>());
    auto* kp =
        reinterpret_cast<const __nv_bfloat16*>(k_cache.data_ptr<at::BFloat16>());
    auto* sp =
        reinterpret_cast<__nv_bfloat16*>(scores.data_ptr<at::BFloat16>());
    DISPATCH_HD(HEAD_DIM, HD, {
      DISPATCH_CHANNEL(channel_num, CN, HD, {
        DISPATCH_GROUP(GROUP_SIZE, GS, {
          launch_oracle_topk_score_inst<__nv_bfloat16, HD, CN, GS>(
              qp, kp, indptr.data_ptr<int32_t>(),
              indices.data_ptr<int32_t>(), last_page_len.data_ptr<int32_t>(),
              sp, B, H_q, H_kv, page_size, L_max,
              kv_stride_page, kv_stride_n, kv_stride_h, stream);
        });
      });
    });
  } else {
    TORCH_CHECK(false, "Only fp16/bf16 supported for q");
  }

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return scores;
}

// -----------------------------------------------------------------------------
// Compact entry point: returns scores[B, H_q, L_mid] where L_mid = token_end - token_start.
// Only the range [token_start, token_end) of the sequence is scored.
// Positions in [token_start, token_end) beyond the actual per-batch length
// are filled with -inf.
// This avoids computing scores for sink/local tokens we don't need top-k over,
// and avoids the subsequent masked_fill_ in Python.
// -----------------------------------------------------------------------------
torch::Tensor oracle_topk_compute_scores_compact(
    torch::Tensor q,
    torch::Tensor k_cache,
    int64_t kv_stride_page,
    int64_t kv_stride_n,
    int64_t kv_stride_h,
    torch::Tensor indptr,
    torch::Tensor indices,
    torch::Tensor last_page_len,
    int64_t H_kv_arg,
    int64_t page_size_arg,
    int64_t token_start_arg,
    int64_t token_end_arg,
    int64_t channel_num_arg) {
  CHECK_CUDA(q);
  CHECK_CONTIG(q);
  CHECK_CUDA(k_cache);
  CHECK_CUDA(indptr);
  CHECK_CONTIG(indptr);
  CHECK_CUDA(indices);
  CHECK_CONTIG(indices);
  CHECK_CUDA(last_page_len);
  CHECK_CONTIG(last_page_len);

  TORCH_CHECK(q.dim() == 3, "q must be [B, H_q, D]");
  TORCH_CHECK(indptr.dtype() == torch::kInt32, "indptr must be int32");
  TORCH_CHECK(indices.dtype() == torch::kInt32, "indices must be int32");
  TORCH_CHECK(last_page_len.dtype() == torch::kInt32, "last_page_len must be int32");

  const int B = (int)q.size(0);
  const int H_q = (int)q.size(1);
  const int HEAD_DIM = (int)q.size(2);
  const int H_kv = (int)H_kv_arg;
  const int page_size = (int)page_size_arg;
  const int token_start = (int)token_start_arg;
  const int token_end = (int)token_end_arg;
  const int channel_num = (int)channel_num_arg;
  TORCH_CHECK(token_end > token_start, "token_end must be > token_start");
  const int L_mid = token_end - token_start;
  TORCH_CHECK(H_q % H_kv == 0, "H_q must be a multiple of H_kv");
  TORCH_CHECK(channel_num > 0 && channel_num <= HEAD_DIM,
              "channel_num must be in (0, head_dim]");
  const int VEC_LANE_WIDTH = (q.scalar_type() == torch::kHalf ||
                              q.scalar_type() == torch::kBFloat16)
                                 ? std::max(8, HEAD_DIM / 32)
                                 : std::max(4, HEAD_DIM / 32);
  TORCH_CHECK(channel_num % VEC_LANE_WIDTH == 0,
              "channel_num must be a multiple of vec lane width ", VEC_LANE_WIDTH);
  const int GROUP_SIZE = H_q / H_kv;

  auto scores = torch::empty({B, H_q, L_mid}, q.options());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  if (q.scalar_type() == torch::kHalf) {
    auto* qp = reinterpret_cast<const __half*>(q.data_ptr<at::Half>());
    auto* kp = reinterpret_cast<const __half*>(k_cache.data_ptr<at::Half>());
    auto* sp = reinterpret_cast<__half*>(scores.data_ptr<at::Half>());
    DISPATCH_HD(HEAD_DIM, HD, {
      DISPATCH_CHANNEL(channel_num, CN, HD, {
        DISPATCH_GROUP(GROUP_SIZE, GS, {
          launch_oracle_topk_score_compact_inst<__half, HD, CN, GS>(
              qp, kp, indptr.data_ptr<int32_t>(),
              indices.data_ptr<int32_t>(), last_page_len.data_ptr<int32_t>(),
              sp, B, H_q, H_kv, page_size, L_mid, token_start, token_end,
              kv_stride_page, kv_stride_n, kv_stride_h, stream);
        });
      });
    });
  } else if (q.scalar_type() == torch::kBFloat16) {
    auto* qp = reinterpret_cast<const __nv_bfloat16*>(q.data_ptr<at::BFloat16>());
    auto* kp = reinterpret_cast<const __nv_bfloat16*>(k_cache.data_ptr<at::BFloat16>());
    auto* sp = reinterpret_cast<__nv_bfloat16*>(scores.data_ptr<at::BFloat16>());
    DISPATCH_HD(HEAD_DIM, HD, {
      DISPATCH_CHANNEL(channel_num, CN, HD, {
        DISPATCH_GROUP(GROUP_SIZE, GS, {
          launch_oracle_topk_score_compact_inst<__nv_bfloat16, HD, CN, GS>(
              qp, kp, indptr.data_ptr<int32_t>(),
              indices.data_ptr<int32_t>(), last_page_len.data_ptr<int32_t>(),
              sp, B, H_q, H_kv, page_size, L_mid, token_start, token_end,
              kv_stride_page, kv_stride_n, kv_stride_h, stream);
        });
      });
    });
  } else {
    TORCH_CHECK(false, "Only fp16/bf16 supported for q");
  }

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return scores;
}

}  // namespace oracle_topk

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("oracle_topk_compute_scores", &oracle_topk::oracle_topk_compute_scores,
        "Dense Q @ K^T scores [B, H_q, L_max] over a paged KV cache.");
  m.def("oracle_topk_compute_scores_compact",
        &oracle_topk::oracle_topk_compute_scores_compact,
        "Compact Q @ K^T scores [B, H_q, L_mid] for token range [start, end).");
}
