// Standalone Torch C++ extension that calls FlashInfer's
// `BatchDecodeWithPagedKVCacheKernel` directly.
//
// The actual CUDA kernel + the host-side dispatch / merge live in the copied
// `flashinfer/` headers under csrc/flashinfer (extracted verbatim from the
// flashinfer pip package). This file is a thin Torch binding that:
//
//   * `query_max_grid_size`: wraps cudaOccupancyMaxActiveBlocksPerMultiprocessor
//     for the BatchDecodeWithPagedKVCacheKernel instantiation that matches the
//     given (dtype, head_dim, group_size). Used by Python plan() to decide
//     split-K parameters once.
//   * `paged_decode_run`: builds Params + paged_kv_t and launches the kernel.
//     Receives all bookkeeping (request_indices/kv_tile_indices/o_indptr/
//     kv_chunk_size_ptr/block_valid_mask/tmp_v/tmp_s) pre-built on device by
//     Python plan().
//
// We intentionally keep the surface area small (fp16/bf16, NHD/HND, soft cap,
// sliding window, no RoPE / no FP8 KV / no MLA), matching what our previous
// hand-rolled kernel supported.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <algorithm>
#include <cstdint>
#include <vector>

#include "flashinfer/attention/decode.cuh"
#include "flashinfer/attention/default_decode_params.cuh"
#include "flashinfer/attention/variants.cuh"
#include "flashinfer/layout.cuh"
#include "flashinfer/page.cuh"
#include "flashinfer/pos_enc.cuh"
#include "flashinfer/utils.cuh"

namespace paged_decode {

using flashinfer::BatchDecodeParams;
using flashinfer::DefaultAttention;
using flashinfer::PosEncodingMode;
using flashinfer::QKVLayout;
using flashinfer::paged_kv_t;

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be CUDA")
#define CHECK_CONTIG(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIG(x)

// DISPATCH_BOOL: take a runtime bool and define a CONST_NAME constexpr inside
// the body. Variadic so nested template commas don't break macro parsing.
#define DISPATCH_BOOL(BVAR, CONST_NAME, ...) \
  do {                                       \
    if (BVAR) {                              \
      constexpr bool CONST_NAME = true;      \
      __VA_ARGS__;                           \
    } else {                                 \
      constexpr bool CONST_NAME = false;     \
      __VA_ARGS__;                           \
    }                                        \
  } while (0)

#define DISPATCH_DTYPE(SCALAR_TYPE, DTYPE_NAME, ...)               \
  do {                                                             \
    if ((SCALAR_TYPE) == torch::kHalf) {                           \
      using DTYPE_NAME = __half;                                   \
      __VA_ARGS__;                                                 \
    } else if ((SCALAR_TYPE) == torch::kBFloat16) {                \
      using DTYPE_NAME = __nv_bfloat16;                            \
      __VA_ARGS__;                                                 \
    } else {                                                       \
      TORCH_CHECK(false, "Only fp16/bf16 supported, got ",         \
                  c10::toString(SCALAR_TYPE));                     \
    }                                                              \
  } while (0)

// -----------------------------------------------------------------------------
// query_max_grid_size: report num_blocks_per_sm * num_sm for the
// BatchDecodeWithPagedKVCacheKernel instantiation that matches (dtype, D, G).
// -----------------------------------------------------------------------------
int64_t query_max_grid_size(int64_t scalar_dtype_int, int64_t head_dim,
                            int64_t group_size) {
  c10::ScalarType dt = static_cast<c10::ScalarType>(scalar_dtype_int);
  int max_grid = 1;
  DISPATCH_DTYPE(dt, DType, {
    DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, {
      DISPATCH_GQA_GROUP_SIZE(group_size, GROUP_SIZE, {
        constexpr uint32_t vec_size = std::max(16UL / sizeof(DType), HEAD_DIM / 32UL);
        constexpr uint32_t bdx = HEAD_DIM / vec_size;
        constexpr uint32_t bdy = GROUP_SIZE;
        constexpr uint32_t num_threads = std::max(128U, bdx * bdy);
        constexpr uint32_t bdz = num_threads / (bdx * bdy);
        constexpr uint32_t tile_size_per_bdx =
            GROUP_SIZE == 1 ? (sizeof(DType) == 1 ? 2U : 4U) : 1U;
        constexpr PosEncodingMode POS_MODE = PosEncodingMode::kNone;
        auto compute_capacity = flashinfer::GetCudaComputeCapability();
        DISPATCH_COMPUTE_CAP_DECODE_NUM_STAGES_SMEM(compute_capacity, NUM_STAGES_SMEM, {
          using Variant = DefaultAttention</*use_custom_mask=*/false,
                                            /*use_sliding_window=*/false,
                                            /*use_logits_soft_cap=*/false,
                                            /*use_alibi=*/false>;
          using Params = BatchDecodeParams<DType, DType, DType, int32_t>;
          const uint32_t smem_size =
              2 * NUM_STAGES_SMEM * tile_size_per_bdx * bdy * bdz * HEAD_DIM * sizeof(DType) +
              std::max(tile_size_per_bdx * num_threads * sizeof(DType*),
                       2 * bdy * bdz * sizeof(float));
          auto kernel =
              flashinfer::BatchDecodeWithPagedKVCacheKernel<POS_MODE, NUM_STAGES_SMEM,
                                                            tile_size_per_bdx, vec_size, bdx, bdy,
                                                            bdz, Variant, Params>;
          int num_blocks_per_sm = 0;
          int num_sm = 0;
          int dev_id = 0;
          cudaGetDevice(&dev_id);
          cudaDeviceGetAttribute(&num_sm, cudaDevAttrMultiProcessorCount, dev_id);
          cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks_per_sm, kernel, num_threads,
                                                        smem_size);
          max_grid = num_blocks_per_sm * num_sm;
        });
      });
    });
  });
  return max_grid;
}

// -----------------------------------------------------------------------------
// paged_decode_run: launch BatchDecodeWithPagedKVCacheDispatched with the
// pre-built bookkeeping tensors from plan().
// -----------------------------------------------------------------------------
template <typename DType>
static void run_decode_dispatched(
    DType* q_ptr, DType* k_ptr, DType* v_ptr, DType* o_ptr, float* lse_ptr,
    int32_t* indptr_ptr, int32_t* indices_ptr, int32_t* last_page_len_ptr,
    int batch_size, int num_qo_heads, int num_kv_heads, int head_dim, int page_size,
    int64_t kv_stride_page, int64_t kv_stride_n, int64_t kv_stride_h,
    QKVLayout layout, float sm_scale, float soft_cap, int32_t window_left,
    // pre-built bookkeeping
    int32_t* request_indices, int32_t* kv_tile_indices, int32_t* o_indptr,
    int32_t* kv_chunk_size_ptr, bool* block_valid_mask, int32_t padded_batch_size,
    DType* tmp_v_ptr, float* tmp_s_ptr, bool split_kv,
    cudaStream_t stream) {
  int64_t kv_strides[3] = {kv_stride_page, kv_stride_n, kv_stride_h};
  paged_kv_t<DType, int32_t> paged_kv(
      /*num_heads=*/(uint32_t)num_kv_heads, /*page_size=*/(uint32_t)page_size,
      /*head_dim=*/(uint32_t)head_dim, /*batch_size=*/(uint32_t)batch_size,
      /*layout=*/layout, /*k_data=*/k_ptr, /*v_data=*/v_ptr,
      /*kv_strides=*/kv_strides, /*indices=*/indices_ptr, /*indptr=*/indptr_ptr,
      /*last_page_len=*/last_page_len_ptr, /*rope_pos_offset=*/nullptr);

  BatchDecodeParams<DType, DType, DType, int32_t> params(
      /*q=*/q_ptr, /*q_rope_offset=*/nullptr, /*paged_kv=*/paged_kv,
      /*o=*/o_ptr, /*lse=*/lse_ptr, /*maybe_alibi_slopes=*/nullptr,
      /*num_qo_heads=*/(uint32_t)num_qo_heads,
      /*q_stride_n=*/(int32_t)(num_qo_heads * head_dim),
      /*q_stride_h=*/(int32_t)head_dim,
      /*window_left=*/window_left,
      /*logits_soft_cap=*/soft_cap, /*sm_scale=*/sm_scale,
      /*rope_scale=*/1.0f, /*rope_theta=*/10000.0f);

  params.padded_batch_size = (uint32_t)padded_batch_size;
  params.request_indices = request_indices;
  params.kv_tile_indices = kv_tile_indices;
  params.o_indptr = o_indptr;
  params.kv_chunk_size_ptr = kv_chunk_size_ptr;
  params.block_valid_mask = block_valid_mask;

  const bool use_soft_cap = soft_cap > 0.f;
  const bool use_window = window_left >= 0;
  const uint32_t group_size = num_qo_heads / num_kv_heads;

  DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, {
    DISPATCH_GQA_GROUP_SIZE(group_size, GROUP_SIZE, {
      constexpr PosEncodingMode POS_MODE = PosEncodingMode::kNone;
      DISPATCH_BOOL(use_soft_cap, USE_SOFT_CAP, {
        DISPATCH_BOOL(use_window, USE_WINDOW, {
          using Variant = DefaultAttention</*use_custom_mask=*/false,
                                            /*use_sliding_window=*/USE_WINDOW,
                                            /*use_logits_soft_cap=*/USE_SOFT_CAP,
                                            /*use_alibi=*/false>;
          using Params = BatchDecodeParams<DType, DType, DType, int32_t>;
          auto err = flashinfer::BatchDecodeWithPagedKVCacheDispatched<HEAD_DIM, POS_MODE, Variant,
                                                                       Params>(
              params, split_kv ? tmp_v_ptr : nullptr, split_kv ? tmp_s_ptr : nullptr,
              /*enable_pdl=*/false, stream);
          TORCH_CHECK(err == cudaSuccess,
                      "BatchDecodeWithPagedKVCacheDispatched failed: ",
                      cudaGetErrorString(err));
        });
      });
    });
  });
}

std::vector<torch::Tensor> paged_decode_run(
    torch::Tensor q,
    torch::Tensor k_cache,
    torch::Tensor v_cache,
    int64_t kv_stride_page,
    int64_t kv_stride_n,
    int64_t kv_stride_h,
    int64_t kv_v_offset_elem,
    torch::Tensor indptr,
    torch::Tensor indices,
    torch::Tensor last_page_len,
    int64_t page_size_arg,
    int64_t H_kv_arg,
    double sm_scale,
    double soft_cap,
    int64_t window_left,
    bool kv_layout_nhd,
    bool return_lse,
    // pre-built bookkeeping from plan()
    torch::Tensor request_indices,
    torch::Tensor kv_tile_indices,
    torch::Tensor o_indptr,
    torch::Tensor kv_chunk_size_ptr,
    c10::optional<torch::Tensor> block_valid_mask,
    c10::optional<torch::Tensor> tmp_v,
    c10::optional<torch::Tensor> tmp_s,
    int64_t padded_batch_size,
    bool split_kv) {
  CHECK_INPUT(q);
  CHECK_CUDA(k_cache);
  CHECK_CUDA(v_cache);
  TORCH_CHECK(q.dim() == 3, "q must be [B, H_q, D]");

  const int B = (int)q.size(0);
  const int H_q = (int)q.size(1);
  const int HEAD_DIM = (int)q.size(2);
  const int H_kv = (int)H_kv_arg;
  const int page_size = (int)page_size_arg;

  auto out = torch::empty_like(q);
  torch::Tensor lse;
  if (return_lse) {
    lse = torch::empty({B, H_q}, q.options().dtype(torch::kFloat32));
  } else {
    lse = torch::empty({0}, q.options().dtype(torch::kFloat32));
  }

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const QKVLayout layout = kv_layout_nhd ? QKVLayout::kNHD : QKVLayout::kHND;

  bool* mask_ptr = block_valid_mask.has_value()
                       ? block_valid_mask.value().data_ptr<bool>()
                       : nullptr;

  if (q.scalar_type() == torch::kHalf) {
    auto* qp = reinterpret_cast<__half*>(q.data_ptr<at::Half>());
    auto* kp = reinterpret_cast<__half*>(k_cache.data_ptr<at::Half>());
    auto* vp = reinterpret_cast<__half*>(v_cache.data_ptr<at::Half>()) + kv_v_offset_elem;
    auto* outp = reinterpret_cast<__half*>(out.data_ptr<at::Half>());
    float* lsep = return_lse ? lse.data_ptr<float>() : nullptr;
    __half* tmp_vp = tmp_v.has_value()
                         ? reinterpret_cast<__half*>(tmp_v.value().data_ptr())
                         : nullptr;
    float* tmp_sp = tmp_s.has_value() ? tmp_s.value().data_ptr<float>() : nullptr;
    run_decode_dispatched<__half>(
        qp, kp, vp, outp, lsep, indptr.data_ptr<int32_t>(),
        indices.data_ptr<int32_t>(), last_page_len.data_ptr<int32_t>(), B, H_q,
        H_kv, HEAD_DIM, page_size, kv_stride_page, kv_stride_n, kv_stride_h,
        layout, (float)sm_scale, (float)soft_cap, (int32_t)window_left,
        request_indices.data_ptr<int32_t>(),
        kv_tile_indices.data_ptr<int32_t>(), o_indptr.data_ptr<int32_t>(),
        kv_chunk_size_ptr.data_ptr<int32_t>(), mask_ptr,
        (int32_t)padded_batch_size, tmp_vp, tmp_sp, split_kv, stream);
  } else if (q.scalar_type() == torch::kBFloat16) {
    auto* qp = reinterpret_cast<__nv_bfloat16*>(q.data_ptr<at::BFloat16>());
    auto* kp = reinterpret_cast<__nv_bfloat16*>(k_cache.data_ptr<at::BFloat16>());
    auto* vp = reinterpret_cast<__nv_bfloat16*>(v_cache.data_ptr<at::BFloat16>()) +
               kv_v_offset_elem;
    auto* outp = reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>());
    float* lsep = return_lse ? lse.data_ptr<float>() : nullptr;
    __nv_bfloat16* tmp_vp =
        tmp_v.has_value() ? reinterpret_cast<__nv_bfloat16*>(tmp_v.value().data_ptr())
                          : nullptr;
    float* tmp_sp = tmp_s.has_value() ? tmp_s.value().data_ptr<float>() : nullptr;
    run_decode_dispatched<__nv_bfloat16>(
        qp, kp, vp, outp, lsep, indptr.data_ptr<int32_t>(),
        indices.data_ptr<int32_t>(), last_page_len.data_ptr<int32_t>(), B, H_q,
        H_kv, HEAD_DIM, page_size, kv_stride_page, kv_stride_n, kv_stride_h,
        layout, (float)sm_scale, (float)soft_cap, (int32_t)window_left,
        request_indices.data_ptr<int32_t>(),
        kv_tile_indices.data_ptr<int32_t>(), o_indptr.data_ptr<int32_t>(),
        kv_chunk_size_ptr.data_ptr<int32_t>(), mask_ptr,
        (int32_t)padded_batch_size, tmp_vp, tmp_sp, split_kv, stream);
  } else {
    TORCH_CHECK(false, "Only fp16/bf16 q dtype supported");
  }

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {out, lse};
}

}  // namespace paged_decode

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("query_max_grid_size", &paged_decode::query_max_grid_size,
        "Return cudaOccupancy max-active-blocks * SMs for decode kernel "
        "instantiation matching (dtype, head_dim, group_size).");
  m.def("paged_decode_run", &paged_decode::paged_decode_run,
        "Launch FlashInfer's BatchDecodeWithPagedKVCacheDispatched.");
}
