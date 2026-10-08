// Fused per-block K/V norm scorer (CUDA).
//
// One kernel launch computes mean-per-token L2 norms for BOTH K and V halves
// of every selected page across ALL layers:
//   k_out[l, b, h] = mean_t ||K_t||,  v_out[l, b, h] = mean_t ||V_t||
// This replaces the per-layer index_select + norm + stack loop in
// KVOffloadManager.score_blocks (and the Triton equivalent) with a single
// native kernel launch.
//
// Layout contract (this fork's combined per-layer cache):
//   cache[l]: [P, H, N, 2C] fp16/bf16, last dim contiguous (stride == 1)
//   K half = [..., :C], V half = [..., C:2C]
//
// Parallelisation: one thread block per (l, b, h); 4 warps; warps stride the
// token axis; within a token, lanes stride the head_dim axis; warp-shuffle
// reductions accumulate sum-of-squares, lane 0 accumulates sqrt of it.
//
// Parity contract: value-equal (within fp32 accumulation tolerance) to
// compute_block_kv_norms' per-head mean-over-token norms (mean of per-token
// norms, not norm of means).

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <torch/extension.h>

#include <cstdint>
#include <vector>

#define FULL_WARP_MASK 0xffffffffu

namespace {

template <typename scalar_t>
struct Cvt;
template <>
struct Cvt<__half> {
  static __device__ __forceinline__ float f(const __half v) { return __half2float(v); }
};
template <>
struct Cvt<__nv_bfloat16> {
  static __device__ __forceinline__ float f(const __nv_bfloat16 v) { return __bfloat162float(v); }
};

constexpr int kBlockThreads = 128;   // 4 warps

template <typename scalar_t>
__global__ void kv_block_norm_kernel(
    const int64_t* __restrict__ layer_ptrs,  // [L] data_ptr() per layer
    const int32_t* __restrict__ block_ids,   // [B] page ids to score
    float* __restrict__ k_out,               // [L, B, H]
    float* __restrict__ v_out,               // [L, B, H]
    const int32_t num_blocks,                // B
    const int32_t num_heads,                 // H
    const int32_t page_tokens,               // N
    const int32_t head_dim,                  // C
    const int64_t stride_page,               // elements per page
    const int64_t stride_head,               // elements per head
    const int64_t stride_tok)                // elements per token
{
  const int gid = blockIdx.x;
  const int h = gid % num_heads;
  const int rest = gid / num_heads;
  const int b = rest % num_blocks;
  const int l = rest / num_blocks;

  const scalar_t* base =
      reinterpret_cast<const scalar_t*>(layer_ptrs[l]) +
      static_cast<int64_t>(block_ids[b]) * stride_page +
      static_cast<int64_t>(h) * stride_head;

  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  constexpr int kWarps = kBlockThreads / 32;

  float acc_k = 0.f, acc_v = 0.f;
  for (int t = warp; t < page_tokens; t += kWarps) {
    const scalar_t* row = base + static_cast<int64_t>(t) * stride_tok;
    float sk = 0.f, sv = 0.f;
    for (int c = lane; c < head_dim; c += 32) {
      const float k = Cvt<scalar_t>::f(row[c]);
      const float v = Cvt<scalar_t>::f(row[c + head_dim]);
      sk += k * k;
      sv += v * v;
    }
    // warp reduce
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
      sk += __shfl_down_sync(FULL_WARP_MASK, sk, off);
      sv += __shfl_down_sync(FULL_WARP_MASK, sv, off);
    }
    if (lane == 0) {
      acc_k += sqrtf(sk);
      acc_v += sqrtf(sv);
    }
  }

  __shared__ float share_k[kWarps];
  __shared__ float share_v[kWarps];
  if (lane == 0) {
    share_k[warp] = acc_k;
    share_v[warp] = acc_v;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    float tot_k = 0.f, tot_v = 0.f;
    #pragma unroll
    for (int w = 0; w < kWarps; ++w) {
      tot_k += share_k[w];
      tot_v += share_v[w];
    }
    const int64_t out_idx =
        (static_cast<int64_t>(l) * num_blocks + b) * num_heads + h;
    k_out[out_idx] = tot_k / page_tokens;
    v_out[out_idx] = tot_v / page_tokens;
  }
}

}  // namespace

std::vector<torch::Tensor> kv_norm_fused(
    std::vector<torch::Tensor> caches,  // each [P, H, N, 2C]
    torch::Tensor block_ids)            // int32 CUDA [B]
{
  TORCH_CHECK(!caches.empty(), "kv_norm_fused: no layer caches");
  TORCH_CHECK(block_ids.is_cuda(), "kv_norm_fused: block_ids must be CUDA");
  TORCH_CHECK(block_ids.scalar_type() == torch::kInt32,
              "kv_norm_fused: block_ids must be int32");
  TORCH_CHECK(block_ids.is_contiguous(), "kv_norm_fused: block_ids must be contiguous");

  const auto& first = caches[0];
  TORCH_CHECK(first.dim() == 4, "kv_norm_fused: cache must be 4D [P,H,N,2C]");
  TORCH_CHECK(first.is_cuda(), "kv_norm_fused: cache must be CUDA");
  TORCH_CHECK(first.stride(3) == 1, "kv_norm_fused: last dim must be contiguous");
  const int64_t C2 = first.size(3);
  TORCH_CHECK(C2 % 2 == 0, "kv_norm_fused: last dim must be even (K||V)");

  const int64_t L = static_cast<int64_t>(caches.size());
  const int64_t B = block_ids.numel();
  const int64_t H = first.size(1);
  const int64_t page_tokens = first.size(2);
  const int64_t head_dim = C2 / 2;

  auto opts_f32 = torch::TensorOptions()
                      .dtype(torch::kFloat32)
                      .device(first.device());
  torch::Tensor k_out = torch::empty({L, B, H}, opts_f32);
  torch::Tensor v_out = torch::empty({L, B, H}, opts_f32);
  if (B == 0) return {k_out, v_out};

  // Stash layer base pointers in an int64 CUDA tensor for the kernel.
  std::vector<int64_t> ptrs_host(L);
  for (int64_t l = 0; l < L; ++l) {
    TORCH_CHECK(caches[l].is_cuda() && caches[l].dim() == 4 &&
                    caches[l].sizes() == first.sizes() &&
                    caches[l].scalar_type() == first.scalar_type() &&
                    caches[l].stride(0) == first.stride(0) &&
                    caches[l].stride(1) == first.stride(1) &&
                    caches[l].stride(2) == first.stride(2) &&
                    caches[l].stride(3) == 1,
                "kv_norm_fused: layer ", l, " layout mismatch");
    ptrs_host[l] = reinterpret_cast<int64_t>(caches[l].data_ptr());
  }
  torch::Tensor ptrs = torch::tensor(ptrs_host,
      torch::TensorOptions().dtype(torch::kInt64)).to(first.device());

  const int64_t grid = L * B * H;
  auto stream = at::cuda::getCurrentCUDAStream();

  const auto stype = first.scalar_type();
  if (stype == torch::kHalf) {
    kv_block_norm_kernel<__half><<<grid, kBlockThreads, 0, stream>>>(
        ptrs.data_ptr<int64_t>(),
        block_ids.data_ptr<int32_t>(),
        k_out.data_ptr<float>(), v_out.data_ptr<float>(),
        static_cast<int32_t>(B), static_cast<int32_t>(H),
        static_cast<int32_t>(page_tokens), static_cast<int32_t>(head_dim),
        first.stride(0), first.stride(1), first.stride(2));
  } else if (stype == torch::kBFloat16) {
    kv_block_norm_kernel<__nv_bfloat16><<<grid, kBlockThreads, 0, stream>>>(
        ptrs.data_ptr<int64_t>(),
        block_ids.data_ptr<int32_t>(),
        k_out.data_ptr<float>(), v_out.data_ptr<float>(),
        static_cast<int32_t>(B), static_cast<int32_t>(H),
        static_cast<int32_t>(page_tokens), static_cast<int32_t>(head_dim),
        first.stride(0), first.stride(1), first.stride(2));
  } else {
    TORCH_CHECK(false, "kv_norm_fused: unsupported dtype ", stype);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {k_out, v_out};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("kv_norm_fused", &kv_norm_fused,
        "Fused K/V block norm scorer over combined per-layer caches");
}
