// Token-offload staging kernels (TokenStagingStore, token-engine mode).
//
// The torch path spends a decode step's per-layer fetch in ~60 tiny ops and
// 6-7 host syncs (boolean-mask nonzero, torch.unique, .cpu(), .item(), the
// slot-id round trip, a CPU gather and a pinned-slab copy). These kernels do
// the same work in 3 launches and one counter readback per layer:
//
//   tok_discover  classify selected positions (resident / cold), stamp cold
//                 hits in the LRU, emit each missing key once (stamp-array
//                 dedup instead of a sort-based unique) + counters.
//   tok_fill      for victims chosen by the caller: update owner / tok_slot /
//                 last_use and copy each token's [H, 2C] row ZERO-COPY from the
//                 pinned host token pool straight into the layer's staging tail.
//   tok_phys      physical address per selected position (resident through the
//                 row's compacted page list, staged cold from tok_slot).
//
// Per-(layer, head) cell mode (HeadTokenOffloadEngine): keys carry the KV
// head, ``key = (row * H + kvh) * mbp + pos``; ``tok_discover`` looks the
// cell up in ``cell [S, max_len, H]`` (kvh from the entry's q head), and
// tok_fill_cells / tok_phys_cells move and address single 2C rows of the
// flat cell array ``kv_rows [cells, C2]`` (see token_mode_heads.py). Staging
// is one pool for all layers: slot v is cell ``stage_units[v / HP] * HP +
// v % HP``, and ``owner[v]`` indexes the all-layer ``tok_slot`` (layer_off +
// key).
//
// Layouts (per layer, all contiguous):
//   phys_of_logical, cpu_slot : int32 [S, max_len]
//   tok_slot                  : int32 [S * mbp]          (-1 = not staged)
//   owner, last_use           : int64 [C]                (C staging token slots)
//   seen                      : int64 [S * mbp]          (tick a key was requested)
//   cpu_pool                  : [N, H, C2] pinned host   (rows of one layer)
//   k_cache                   : [pages, H, P, C2]        (any 16B-aligned dtype)
//   counters                  : int32 [4] = {n_need, n_hit_new, err, n_cold}
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>

namespace {

using ull = unsigned long long;

__global__ void tok_discover_kernel(
    const int64_t* __restrict__ idx, int64_t n_per_row, int64_t total,
    const int64_t* __restrict__ rows,
    const int32_t* __restrict__ pol, const int32_t* __restrict__ cpu, int64_t max_len,
    const int32_t* __restrict__ tok_slot,
    int64_t* __restrict__ last_use, int64_t* __restrict__ seen,
    int64_t tick, int64_t mbp, int64_t H, int64_t ksel, int64_t group,
    int32_t* __restrict__ phys_pos, bool* __restrict__ cold_out,
    int64_t* __restrict__ key_out, int64_t* __restrict__ need,
    int32_t* __restrict__ counters) {
  int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
  const bool live = i < total;
  bool cold = false;
  int32_t p = 0;
  int64_t key = 0;
  int64_t mi = 0;
  if (live) {
    const int64_t b = i / n_per_row;
    const int64_t row = rows[b];
    const int64_t pos = idx[i];
    const int64_t kvh = H == 1 ? 0 : ((i - b * n_per_row) / ksel) / group;
    mi = (row * max_len + pos) * H + kvh;
    p = pol[mi];
    key = (row * H + kvh) * mbp + pos;
    cold = p < 0;
  }
  // Selected-but-offloaded entries (diagnostics): one atomic per warp.
  const unsigned ballot = __ballot_sync(0xffffffffu, cold);
  if ((threadIdx.x & 31) == 0 && ballot) atomicAdd(counters + 3, __popc(ballot));
  if (!live) return;
  phys_pos[i] = p;
  cold_out[i] = cold;
  key_out[i] = key;
  if (!cold) return;
  const int32_t s = tok_slot[key];
  if (s >= 0) {
    // Count each slot once per tick: the host needs (#slots stamped this
    // tick) for the capacity check that the torch path gets from topk/.item().
    if ((int64_t)atomicExch(reinterpret_cast<ull*>(last_use + s), (ull)tick) != tick)
      atomicAdd(counters + 1, 1);
  } else if ((int64_t)atomicExch(reinterpret_cast<ull*>(seen + key), (ull)tick) != tick) {
    // A miss must have a CPU copy; report it in THIS layer's readback,
    // before anything is fetched.
    if (cpu[mi] < 0) atomicExch(counters + 2, 1);
    need[atomicAdd(counters, 1)] = key;
  }
}

// One block per fetched token. Thread 0 updates the maps; every thread copies
// 16-byte vectors of the token's H rows (host -> device, zero-copy over PCIe).
__global__ void tok_fill_kernel(
    const int64_t* __restrict__ keys, const int64_t* __restrict__ victims,
    int64_t* __restrict__ owner, int32_t* __restrict__ tok_slot,
    int64_t* __restrict__ last_use, int64_t tick,
    const int32_t* __restrict__ cpu_slot, int64_t max_len, int64_t mbp,
    const int4* __restrict__ pool, int64_t row_vecs, int64_t head_vecs,
    int4* __restrict__ kc, int64_t s_page, int64_t s_head, int64_t s_off,
    int64_t base, int64_t P, int32_t* __restrict__ counters) {
  const int64_t i = blockIdx.x;
  const int64_t key = keys[i];
  const int64_t v = victims[i];
  if (threadIdx.x == 0) {
    const int64_t old = owner[v];
    if (old >= 0) tok_slot[old] = -1;   // old keys are staged, new ones are not
    owner[v] = key;
    tok_slot[key] = (int32_t)v;
    last_use[v] = tick;
  }
  const int64_t row = key / mbp;
  const int64_t cs = cpu_slot[row * max_len + (key - row * mbp)];
  if (cs < 0) {
    if (threadIdx.x == 0) atomicExch(counters + 2, 1);
    return;
  }
  const int4* src = pool + cs * row_vecs;
  const int64_t page = base + v / P;
  int4* dst = kc + page * s_page + (v % P) * s_off;
  for (int64_t e = threadIdx.x; e < row_vecs; e += blockDim.x) {
    const int64_t h = e / head_vecs;
    dst[h * s_head + (e - h * head_vecs)] = src[e];
  }
}

template <typename IT>
__global__ void tok_phys_kernel(
    const int32_t* __restrict__ phys_pos, const int64_t* __restrict__ key,
    const bool* __restrict__ cold, int64_t n_per_row, int64_t total,
    const IT* __restrict__ indptr, const IT* __restrict__ indices,
    const int32_t* __restrict__ tok_slot, int64_t base, int64_t P,
    int64_t* __restrict__ out) {
  int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
  if (i >= total) return;
  if (cold[i]) {
    const int32_t s = tok_slot[key[i]];
    out[i] = s >= 0 ? (int64_t)s + base * P : 0;
  } else {
    const int64_t b = i / n_per_row;
    const int64_t p = phys_pos[i];
    const int64_t blk = p / P;
    out[i] = (int64_t)indices[(int64_t)indptr[b] + blk] * P + (p - blk * P);
  }
}

// Cell mode: one warp per fetched key copies ONE 2C row (16-byte vectors)
// host -> device zero-copy; lane 0 updates the maps.
__global__ void tok_fill_cells_kernel(
    const int64_t* __restrict__ keys, const int64_t* __restrict__ victims, int64_t n,
    int64_t* __restrict__ owner, int32_t* __restrict__ tok_slot_all, int64_t layer_off,
    int64_t* __restrict__ last_use, int64_t tick,
    const int32_t* __restrict__ cpu_row, int64_t max_len, int64_t H, int64_t mbp,
    const int4* __restrict__ pool, int64_t row_vecs, int4* __restrict__ kv_rows,
    const int64_t* __restrict__ stage_units, int64_t HP,
    int32_t* __restrict__ counters) {
  const int64_t i = blockIdx.x * (int64_t)(blockDim.x / 32) + threadIdx.x / 32;
  const int lane = threadIdx.x & 31;
  if (i >= n) return;
  const int64_t key = keys[i];
  const int64_t v = victims[i];
  if (lane == 0) {
    const int64_t old = owner[v];
    if (old >= 0) tok_slot_all[old] = -1;   // may belong to another layer
    owner[v] = layer_off + key;
    tok_slot_all[layer_off + key] = (int32_t)v;
    last_use[v] = tick;
  }
  const int64_t rh = key / mbp;              // row * H + kvh
  const int64_t pos = key - rh * mbp;
  const int64_t row = rh / H;
  const int64_t cs = cpu_row[(row * max_len + pos) * H + (rh - row * H)];
  if (cs < 0) {
    if (lane == 0) atomicExch(counters + 2, 1);
    return;
  }
  const int64_t cell = stage_units[v / HP] * HP + v % HP;
  const int4* src = pool + cs * row_vecs;
  int4* dst = kv_rows + cell * row_vecs;
  for (int64_t e = lane; e < row_vecs; e += 32) dst[e] = src[e];
}

__global__ void tok_phys_cells_kernel(
    const int32_t* __restrict__ cell, const int64_t* __restrict__ key,
    const bool* __restrict__ cold, int64_t total, const int32_t* __restrict__ tok_slot,
    const int64_t* __restrict__ stage_units, int64_t HP, int64_t* __restrict__ out) {
  int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
  if (i >= total) return;
  if (cold[i]) {
    const int64_t s = tok_slot[key[i]];
    out[i] = s >= 0 ? stage_units[s / HP] * HP + s % HP : 0;
  } else {
    out[i] = cell[i];
  }
}

constexpr int kThreads = 256;

void check_cuda(const torch::Tensor& t, const char* name, at::ScalarType dt) {
  TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(t.scalar_type() == dt, name, " has dtype ", t.scalar_type());
}

}  // namespace

// Returns nothing; writes phys_pos / cold / key / need / counters[0:2].
void tok_discover(torch::Tensor idx, torch::Tensor rows, torch::Tensor pol,
                  torch::Tensor cpu, torch::Tensor tok_slot, torch::Tensor last_use, torch::Tensor seen,
                  int64_t tick, int64_t mbp, int64_t H, int64_t ksel, int64_t group,
                  torch::Tensor phys_pos, torch::Tensor cold,
                  torch::Tensor key, torch::Tensor need, torch::Tensor counters) {
  check_cuda(idx, "idx", at::kLong);
  check_cuda(rows, "rows", at::kLong);
  check_cuda(pol, "phys_of_logical", at::kInt);
  check_cuda(cpu, "cpu map", at::kInt);
  TORCH_CHECK(cpu.sizes() == pol.sizes(), "cpu map must match the residency map");
  check_cuda(tok_slot, "tok_slot", at::kInt);
  check_cuda(last_use, "last_use", at::kLong);
  check_cuda(seen, "seen", at::kLong);
  check_cuda(counters, "counters", at::kInt);
  TORCH_CHECK(idx.dim() == 2 && (H == 1 ? pol.dim() == 2 : pol.dim() == 3 && pol.size(2) == H),
              "idx [B, N], map [S, max_len] (or [S, max_len, H] per head)");
  TORCH_CHECK(H == 1 || (ksel > 0 && group > 0 && idx.size(1) % ksel == 0 &&
                         idx.size(1) / ksel == H * group),
              "per-head discover: N must be H * group * ksel");
  TORCH_CHECK(need.numel() >= idx.numel(), "need buffer too small");
  const at::cuda::CUDAGuard guard(idx.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  C10_CUDA_CHECK(cudaMemsetAsync(counters.data_ptr<int32_t>(), 0, 4 * sizeof(int32_t), stream));
  const int64_t total = idx.numel();
  if (total == 0) return;
  const int blocks = (int)((total + kThreads - 1) / kThreads);
  tok_discover_kernel<<<blocks, kThreads, 0, stream>>>(
      idx.data_ptr<int64_t>(), idx.size(1), total, rows.data_ptr<int64_t>(),
      pol.data_ptr<int32_t>(), cpu.data_ptr<int32_t>(), pol.size(1), tok_slot.data_ptr<int32_t>(),
      last_use.data_ptr<int64_t>(), seen.data_ptr<int64_t>(), tick, mbp, H, ksel, group,
      phys_pos.data_ptr<int32_t>(), cold.data_ptr<bool>(), key.data_ptr<int64_t>(),
      need.data_ptr<int64_t>(), counters.data_ptr<int32_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Device-side alias of a pinned host tensor, or 0 when it is not device-mapped.
int64_t device_alias(torch::Tensor host) {
  TORCH_CHECK(host.device().is_cpu(), "expected a host tensor");
  cudaPointerAttributes attr;
  if (cudaPointerGetAttributes(&attr, host.data_ptr()) != cudaSuccess) {
    cudaGetLastError();
    return 0;
  }
  if (attr.type != cudaMemoryTypeHost || attr.devicePointer == nullptr) return 0;
  return reinterpret_cast<int64_t>(attr.devicePointer);
}

void tok_fill(torch::Tensor keys, torch::Tensor victims, torch::Tensor owner,
              torch::Tensor tok_slot, torch::Tensor last_use, int64_t tick,
              torch::Tensor cpu_slot, int64_t mbp, torch::Tensor pool,
              int64_t pool_dev_ptr, torch::Tensor k_cache, int64_t base,
              int64_t P, torch::Tensor counters) {
  check_cuda(keys, "keys", at::kLong);
  check_cuda(victims, "victims", at::kLong);
  check_cuda(owner, "owner", at::kLong);
  check_cuda(tok_slot, "tok_slot", at::kInt);
  check_cuda(last_use, "last_use", at::kLong);
  check_cuda(cpu_slot, "cpu_slot", at::kInt);
  check_cuda(counters, "counters", at::kInt);
  TORCH_CHECK(k_cache.is_cuda() && k_cache.dim() == 4 && k_cache.stride(3) == 1,
              "k_cache [pages, H, P, C2] with unit last stride");
  TORCH_CHECK(pool.dim() == 3 && pool.is_contiguous(), "pool [N, H, C2] contiguous");
  TORCH_CHECK(pool.scalar_type() == k_cache.scalar_type(), "pool / k_cache dtype mismatch");
  TORCH_CHECK(pool.size(1) == k_cache.size(1) && pool.size(2) == k_cache.size(3),
              "pool / k_cache row shape mismatch");
  const int64_t es = k_cache.element_size();
  const int64_t head_bytes = k_cache.size(3) * es;
  TORCH_CHECK(head_bytes % 16 == 0 && (k_cache.stride(0) * es) % 16 == 0 &&
                  (k_cache.stride(1) * es) % 16 == 0 && (k_cache.stride(2) * es) % 16 == 0 &&
                  reinterpret_cast<uintptr_t>(k_cache.data_ptr()) % 16 == 0 &&
                  pool_dev_ptr % 16 == 0,
              "tok_fill needs 16-byte aligned rows");
  TORCH_CHECK(pool_dev_ptr != 0, "pool is not device-mapped");
  const int64_t n = keys.numel();
  if (n == 0) return;
  const at::cuda::CUDAGuard guard(keys.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  const int64_t head_vecs = head_bytes / 16;
  const int64_t row_vecs = head_vecs * k_cache.size(1);
  tok_fill_kernel<<<(unsigned)n, 128, 0, stream>>>(
      keys.data_ptr<int64_t>(), victims.data_ptr<int64_t>(), owner.data_ptr<int64_t>(),
      tok_slot.data_ptr<int32_t>(), last_use.data_ptr<int64_t>(), tick,
      cpu_slot.data_ptr<int32_t>(), cpu_slot.size(1), mbp,
      reinterpret_cast<const int4*>(pool_dev_ptr), row_vecs, head_vecs,
      reinterpret_cast<int4*>(k_cache.data_ptr()), k_cache.stride(0) * es / 16,
      k_cache.stride(1) * es / 16, k_cache.stride(2) * es / 16, base, P,
      counters.data_ptr<int32_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

torch::Tensor tok_phys(torch::Tensor phys_pos, torch::Tensor key, torch::Tensor cold,
                       torch::Tensor indptr, torch::Tensor indices, torch::Tensor tok_slot,
                       int64_t base, int64_t P) {
  check_cuda(phys_pos, "phys_pos", at::kInt);
  check_cuda(key, "key", at::kLong);
  check_cuda(cold, "cold", at::kBool);
  check_cuda(tok_slot, "tok_slot", at::kInt);
  TORCH_CHECK(indptr.scalar_type() == indices.scalar_type() &&
                  (indptr.scalar_type() == at::kInt || indptr.scalar_type() == at::kLong),
              "indptr / indices must both be int32 or int64");
  TORCH_CHECK(indptr.is_contiguous() && indices.is_contiguous(), "indptr / indices contiguous");
  auto out = torch::empty(phys_pos.sizes(), phys_pos.options().dtype(at::kLong));
  const int64_t total = phys_pos.numel();
  if (total == 0) return out;
  const at::cuda::CUDAGuard guard(phys_pos.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  const int blocks = (int)((total + kThreads - 1) / kThreads);
  const int64_t npr = phys_pos.size(-1);
  if (indptr.scalar_type() == at::kInt) {
    tok_phys_kernel<int32_t><<<blocks, kThreads, 0, stream>>>(
        phys_pos.data_ptr<int32_t>(), key.data_ptr<int64_t>(), cold.data_ptr<bool>(), npr,
        total, indptr.data_ptr<int32_t>(), indices.data_ptr<int32_t>(),
        tok_slot.data_ptr<int32_t>(), base, P, out.data_ptr<int64_t>());
  } else {
    tok_phys_kernel<int64_t><<<blocks, kThreads, 0, stream>>>(
        phys_pos.data_ptr<int32_t>(), key.data_ptr<int64_t>(), cold.data_ptr<bool>(), npr,
        total, indptr.data_ptr<int64_t>(), indices.data_ptr<int64_t>(),
        tok_slot.data_ptr<int32_t>(), base, P, out.data_ptr<int64_t>());
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

void tok_fill_cells(torch::Tensor keys, torch::Tensor victims, torch::Tensor owner,
                    torch::Tensor tok_slot_all, int64_t layer_off, torch::Tensor last_use,
                    int64_t tick, torch::Tensor cpu_row, int64_t mbp, torch::Tensor pool,
                    int64_t pool_dev_ptr, torch::Tensor kv_rows, torch::Tensor stage_units,
                    int64_t HP, torch::Tensor counters) {
  check_cuda(keys, "keys", at::kLong);
  check_cuda(victims, "victims", at::kLong);
  check_cuda(owner, "owner", at::kLong);
  check_cuda(tok_slot_all, "tok_slot_all", at::kInt);
  check_cuda(last_use, "last_use", at::kLong);
  check_cuda(cpu_row, "cpu_row", at::kInt);
  check_cuda(stage_units, "stage_units", at::kLong);
  check_cuda(counters, "counters", at::kInt);
  TORCH_CHECK(cpu_row.dim() == 3, "cpu_row [S, max_len, H]");
  TORCH_CHECK(kv_rows.is_cuda() && kv_rows.dim() == 2 && kv_rows.stride(1) == 1 &&
                  kv_rows.stride(0) == kv_rows.size(1),
              "kv_rows [cells, C2] dense rows");
  TORCH_CHECK(pool.dim() == 2 && pool.is_contiguous() && pool.size(1) == kv_rows.size(1) &&
                  pool.scalar_type() == kv_rows.scalar_type(),
              "pool [N, C2] contiguous, same row as kv_rows");
  const int64_t row_bytes = kv_rows.size(1) * kv_rows.element_size();
  TORCH_CHECK(row_bytes % 16 == 0 && reinterpret_cast<uintptr_t>(kv_rows.data_ptr()) % 16 == 0 &&
                  pool_dev_ptr % 16 == 0 && pool_dev_ptr != 0,
              "tok_fill_cells needs 16-byte aligned, device-mapped rows");
  const int64_t n = keys.numel();
  if (n == 0) return;
  const at::cuda::CUDAGuard guard(keys.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  constexpr int kWarps = 4;
  tok_fill_cells_kernel<<<(unsigned)((n + kWarps - 1) / kWarps), 32 * kWarps, 0, stream>>>(
      keys.data_ptr<int64_t>(), victims.data_ptr<int64_t>(), n, owner.data_ptr<int64_t>(),
      tok_slot_all.data_ptr<int32_t>(), layer_off, last_use.data_ptr<int64_t>(), tick,
      cpu_row.data_ptr<int32_t>(), cpu_row.size(1), cpu_row.size(2), mbp,
      reinterpret_cast<const int4*>(pool_dev_ptr), row_bytes / 16,
      reinterpret_cast<int4*>(kv_rows.data_ptr()), stage_units.data_ptr<int64_t>(), HP,
      counters.data_ptr<int32_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

torch::Tensor tok_phys_cells(torch::Tensor cell, torch::Tensor key, torch::Tensor cold,
                             torch::Tensor tok_slot, torch::Tensor stage_units, int64_t HP) {
  check_cuda(cell, "cell", at::kInt);
  check_cuda(key, "key", at::kLong);
  check_cuda(cold, "cold", at::kBool);
  check_cuda(tok_slot, "tok_slot", at::kInt);
  check_cuda(stage_units, "stage_units", at::kLong);
  auto out = torch::empty(cell.sizes(), cell.options().dtype(at::kLong));
  const int64_t total = cell.numel();
  if (total == 0) return out;
  const at::cuda::CUDAGuard guard(cell.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  tok_phys_cells_kernel<<<(int)((total + kThreads - 1) / kThreads), kThreads, 0, stream>>>(
      cell.data_ptr<int32_t>(), key.data_ptr<int64_t>(), cold.data_ptr<bool>(), total,
      tok_slot.data_ptr<int32_t>(), stage_units.data_ptr<int64_t>(), HP, out.data_ptr<int64_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("tok_discover", &tok_discover, "classify + LRU-stamp + dedup missing keys");
  m.def("tok_fill", &tok_fill, "map update + zero-copy host->staging copy");
  m.def("tok_phys", &tok_phys, "physical address per selected position");
  m.def("tok_fill_cells", &tok_fill_cells, "cell mode: map update + zero-copy row fill");
  m.def("tok_phys_cells", &tok_phys_cells, "cell mode: global cell per selected position");
  m.def("device_alias", &device_alias, "device pointer of a pinned host tensor (0 if none)");
}
