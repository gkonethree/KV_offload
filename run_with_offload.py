#!/usr/bin/env python
"""
Example: Running vLLM with KV Cache Offloading and Sparse Attention (KV Connector)

This script demonstrates how to:
1. Enable KV cache offloading via KVTransferConfig
2. Register the sparse attention kernel (skylight_kernels sparse_oracle_topk_optimized)
3. Run inference with offloading and sparse attention
4. Enable profiling/timing

Prerequisites:
- vLLM 0.29.0 installed with OffloadPackageConnector
- offload-package installed
- skylight_kernels sparse_oracle_topk_optimized compiled

The sparse_oracle_topk_optimized kernel handles its own oracle top-k selection
internally (Phase 1: score + top-k, Phase 2: sparse decode). The offload package
manages KV block offloading using PagedEvictionScorer via KV Connector.
"""

import os
import sys
import json
import torch

# =============================================================================
# STEP 0: Disable multiprocessing for offline inference (required for direct model_runner access)
# =============================================================================
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

# =============================================================================
# STEP 1: Configure Offloading via KVTransferConfig
# =============================================================================

from vllm.config import KVTransferConfig

kv_transfer_config = KVTransferConfig(
    kv_connector="OffloadPackageConnector",
    kv_connector_module_path="vllm.distributed.kv_transfer.kv_connector.v1.offload_package_connector",
    kv_connector_extra_config={
        "enabled": True,
        "offload_fraction": 0.80,
        "num_cpu_slots": 4096,
        "num_staging_slots": 64,
        "interval_tokens": 32,
        "scorer_decay": 1.0,
        "scorer_head_reduction": "mean",
    },
    kv_role="kv_both",
)

print("✓ KV Transfer config created for OffloadPackageConnector")

# =============================================================================
# STEP 2: Import and Setup
# =============================================================================

from vllm import LLM, SamplingParams

# For sparse attention - try to import skylight_kernels sparse_oracle_topk_optimized
# Note: Requires the CUDA extensions to be built
SPARSE_KERNEL_AVAILABLE = False
try:
    import sys
    sys.path.insert(0, '/home/gaurav_kumar/KV_offload/skylight_kernels')
    from sparse_oracle_topk_optimized import BatchDecodeWithPagedKVCacheWrapper
    SPARSE_KERNEL_AVAILABLE = True
    print("✓ skylight_kernels sparse_oracle_topk_optimized available")
except ImportError as e:
    print(f"⚠ skylight_kernels sparse_oracle_topk_optimized not available: {e}")
    print("  Will use dense attention fallback")
except Exception as e:
    print(f"⚠ skylight_kernels import failed: {e}")
    print("  Will use dense attention fallback")

# =============================================================================
# STEP 3: Run Inference
# =============================================================================

def run_inference(
    model_name="meta-llama/Llama-2-7b-hf",
    prompts=None,
    max_tokens=128,
    temperature=0.7,
    top_p=0.9,
    topk=0.1,          # Sparse attention: keep 10% of tokens
    channel_num=64,    # Channels for oracle top-k scoring (sparse_oracle_topk_optimized)
    max_seq_len=4096,  # Max sequence length
    enable_profiling=False,
    use_sparse=True,
    gpu_memory_utilization=0.35,
    staging_slots=64,
    enforce_eager=False,
    tensor_parallel_size=1,
):
    """Run vLLM inference with KV cache offloading and sparse attention.
    
    The sparse_oracle_topk_optimized kernel handles its own oracle top-k:
    Phase 1: Computes dense Q@K^T on first `channel_num` channels → top-k
    Phase 2: Runs sparse decode on selected tokens
    
    The offload package manages KV block offloading via PagedEvictionScorer
    (scores blocks by ||V||/||K|| norm ratio, evicts lowest scoring).
    
    Args:
        model_name: HF model name or path
        prompts: List of input prompts
        max_tokens: Max tokens to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling top-p
        topk: Sparse attention fraction (0.0 - 1.0)
        channel_num: Channels for oracle top-k scoring
        max_seq_len: Max sequence length
        enable_profiling: Enable timing and memory profiling
        gpu_memory_utilization: GPU memory utilization fraction
        staging_slots: GPU staging slots for sparse attention
        enforce_eager: Disable torch.compile and CUDAGraphs
        tensor_parallel_size: Number of GPUs for tensor parallelism
    """
    
    # Update staging slots env var
    os.environ["OFFLOAD_PACKAGE_STAGING_SLOTS"] = str(staging_slots)
    
    if prompts is None:
        prompts = [
            "The future of AI is",
            "Machine learning models can",
            "In a world where technology",
        ]
    
    # Create sampling params
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )
    
    # Create LLM with offloading enabled
    print(f"[DEBUG] Creating LLM: model={model_name}, max_seq_len={max_seq_len}, gpu_util={gpu_memory_utilization}, enforce_eager={enforce_eager}, tp={tensor_parallel_size}")
    import time
    t0 = time.time()
    llm = LLM(
        model=model_name,
        dtype="float16",
        max_model_len=max_seq_len,
        gpu_memory_utilization=gpu_memory_utilization,
        enforce_eager=enforce_eager,
        tensor_parallel_size=tensor_parallel_size,
        kv_transfer_config=kv_transfer_config,
    )
    print(f"[DEBUG] LLM created in {time.time() - t0:.2f}s")
    
    # Register sparse kernel (sparse_oracle_topk_optimized handles its own top-k)
    if use_sparse and SPARSE_KERNEL_AVAILABLE:
        try:
            # Create sparse kernel (handles both Phase 1 + 2)
            print(f"[DEBUG] Creating sparse kernel...")
            import time
            t0 = time.time()
            workspace = torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
            sparse_kernel = BatchDecodeWithPagedKVCacheWrapper(
                float_workspace_buffer=workspace,
                kv_layout="NHD",
                topk=topk,
                channel_num=channel_num,
                max_seq_len=max_seq_len,
            )
            print(f"[DEBUG] Sparse kernel created in {time.time() - t0:.2f}s")
            
            # Register sparse kernel on all workers via collective_rpc
            # Use a callable that gets executed on each worker
            def _set_sparse_kernel_on_worker(worker, kernel):
                """Set sparse kernel on worker's model_runner."""
                if hasattr(worker, 'model_runner') and worker.model_runner is not None:
                    worker.model_runner.set_sparse_kernel(kernel)
                else:
                    raise RuntimeError("Worker does not have model_runner initialized")
            
            # Use collective_rpc to register on all workers
            # For UniProcExecutor (multiprocessing disabled), this runs directly on driver worker
            print(f"[DEBUG] Registering sparse kernel via collective_rpc...")
            t0 = time.time()
            llm.llm_engine.engine_core.collective_rpc(
                _set_sparse_kernel_on_worker,
                args=(sparse_kernel,),
            )
            print(f"[DEBUG] Sparse kernel registered in {time.time() - t0:.2f}s")
            
            print(f"✓ Sparse oracle top-k registered (topk={topk}, channel_num={channel_num})")
        except Exception as e:
            print(f"⚠ Failed to register sparse attention: {e}")
            import traceback
            traceback.print_exc()
            print("  Falling back to dense attention")
    else:
        print("Using dense attention (no sparse kernel registered)")
    
    # Store sparse kernel reference for debugging
    sparse_kernel_ref = sparse_kernel if use_sparse and SPARSE_KERNEL_AVAILABLE else None
    
    def debug_sparse_topk(worker):
        """Debug function to print top-k indices and their residency."""
        if hasattr(worker, 'model_runner') and worker.model_runner is not None:
            runner = worker.model_runner
            
            # Get sparse kernel's last top-k indices
            sparse_kernel = getattr(runner, '_sparse_kernel', None)
            if sparse_kernel is not None:
                # The Skylight backend wraps the sparse kernel in _NoFastPlanDecodeWrapper
                inner = sparse_kernel._inner if hasattr(sparse_kernel, '_inner') else sparse_kernel
                if hasattr(inner, 'get_last_topk_indices'):
                    topk_indices = inner.get_last_topk_indices()
                    if topk_indices is not None:
                        # Get orchestrator via KV connector on the model_runner
                        if hasattr(runner, 'kv_connector') and runner.kv_connector is not None:
                            active_connector = runner.kv_connector
                            if hasattr(active_connector, 'kv_connector'):
                                kv_connector = active_connector.kv_connector
                                if hasattr(kv_connector, 'worker_handler') and kv_connector.worker_handler:
                                    orch = kv_connector.worker_handler._orchestrator
                                    if orch is not None:
                                        batch_size, num_heads, k = topk_indices.shape
                                        result = []
                                        # Get actual request IDs from the offload state
                                        req_ids = list(orch.residency._blocks.keys())
                                        if req_ids:
                                            # Group by request_id
                                            req_to_blocks = {}
                                            for bref in orch.residency._blocks:
                                                req_to_blocks.setdefault(bref.request_id, []).append(bref)
                                            
                                            for b in range(min(batch_size, len(req_ids))):
                                                req_id = req_ids[b] if b < len(req_ids) else f"req_{b}"
                                                for h in range(min(num_heads, 4)):  # Limit to first 4 heads
                                                    tokens = topk_indices[b, h].tolist() if b < topk_indices.shape[0] else []
                                                    # Get residency for each token's block
                                                    block_info = []
                                                    for token_idx in tokens:
                                                        block_idx = token_idx // 16  # block_size=16
                                                        from offload_package.staging.residency import BlockRef
                                                        bref = BlockRef(req_id, block_idx)
                                                        loc = orch.residency.get(bref)
                                                        if loc:
                                                            block_info.append(f"token={token_idx}, block={block_idx}, residency={loc.residency.name}, gpu_block={loc.gpu_block_id}, cpu_slot={loc.cpu_slot}")
                                                        else:
                                                            block_info.append(f"token={token_idx}, block={block_idx}, residency=UNKNOWN")
                                                    result.append(f"batch={b}, head={h}, req_id={req_id}, tokens={tokens}, blocks={block_info}")
                                        return result
        return []
    
    # Run generation
    if enable_profiling:
        outputs = run_with_profiling(llm, prompts, sampling_params, debug_sparse_topk)
    else:
        print(f"[DEBUG] Starting generation: {len(prompts)} prompts, max_tokens={max_tokens}")
        import time
        t0 = time.time()
        outputs = llm.generate(prompts, sampling_params)
        print(f"[DEBUG] Generation completed in {time.time() - t0:.2f}s")
    
    return outputs, llm

# =============================================================================
# STEP 4: Profiling/Timing Utilities
# =============================================================================

def run_with_profiling(llm, prompts, sampling_params, debug_sparse_topk=None):
    """Run inference with detailed profiling."""
    import time
    import torch.cuda.profiler as profiler
    
    print("\n" + "="*60)
    print("PROFILING RUN")
    print("="*60)
    
    # Warmup
    print("Warming up...")
    _ = llm.generate(prompts[:1], sampling_params)
    torch.cuda.synchronize()
    
    # Start CUDA profiler
    profiler.start()
    
    start_time = time.perf_counter()
    torch.cuda.synchronize()
    
    # Run generation
    outputs = llm.generate(prompts, sampling_params)
    
    torch.cuda.synchronize()
    end_time = time.perf_counter()
    
    # Stop profiler
    profiler.stop()
    
    # Print top-k debug info if available
    if debug_sparse_topk is not None:
        print("\n" + "="*60)
        print("SPARSE TOP-K DEBUG (at end of generation)")
        print("="*60)
        results = llm.llm_engine.engine_core.collective_rpc(debug_sparse_topk)
        for i, result in enumerate(results):
            if result:
                for line in result:
                    print(line)
    
    # Print timing
    total_time = end_time - start_time
    total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    
    print(f"\nTiming Results:")
    print(f"  Total time: {total_time:.3f}s")
    print(f"  Total tokens: {total_tokens}")
    print(f"  Throughput: {total_tokens/total_time:.1f} tokens/s")
    print(f"  Latency/token: {total_time/total_tokens*1000:.2f} ms")
    print(f"  Req 0: prompt={len(outputs[0].prompt_token_ids)}, generated={len(outputs[0].outputs[0].token_ids)}")
    
    # Memory stats
    print(f"\nGPU Memory:")
    print(f"  Allocated: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
    print(f"  Reserved: {torch.cuda.memory_reserved()/1024**3:.2f} GB")
    print(f"  Max allocated: {torch.cuda.max_memory_allocated()/1024**3:.2f} GB")
    
    return outputs

# =============================================================================
# STEP 5: Offload Statistics
# =============================================================================

def print_offload_stats(llm):
    """Print offloading statistics from the model runner."""
    try:
        # Get offload stats from worker via collective_rpc
        def _get_offload_stats(worker):
            # Debug: print worker attributes
            print(f"[DEBUG] Worker type: {type(worker)}", flush=True)
            print(f"[DEBUG] Worker has model_runner: {hasattr(worker, 'model_runner')}", flush=True)
            if hasattr(worker, 'model_runner'):
                print(f"[DEBUG] model_runner: {worker.model_runner}", flush=True)
                if worker.model_runner is not None:
                    print(f"[DEBUG] runner type: {type(worker.model_runner)}", flush=True)
                    print(f"[DEBUG] runner has kv_connector: {hasattr(worker.model_runner, 'kv_connector')}", flush=True)
                    if hasattr(worker.model_runner, 'kv_connector'):
                        print(f"[DEBUG] kv_connector: {worker.model_runner.kv_connector}", flush=True)
            
            # Access through model_runner -> ActiveKVConnector -> actual connector
            if hasattr(worker, 'model_runner') and worker.model_runner is not None:
                runner = worker.model_runner
                if hasattr(runner, 'kv_connector') and runner.kv_connector is not None:
                    # runner.kv_connector is ActiveKVConnector, which wraps the actual connector
                    active_connector = runner.kv_connector
                    print(f"[DEBUG] active_connector type: {type(active_connector)}", flush=True)
                    print(f"[DEBUG] active_connector has kv_connector: {hasattr(active_connector, 'kv_connector')}", flush=True)
                    if hasattr(active_connector, 'kv_connector'):
                        print(f"[DEBUG] active_connector.kv_connector: {active_connector.kv_connector}", flush=True)
                        kv_connector = active_connector.kv_connector
                        print(f"[DEBUG] kv_connector type: {type(kv_connector)}", flush=True)
                        print(f"[DEBUG] kv_connector has worker_handler: {hasattr(kv_connector, 'worker_handler')}", flush=True)
                        if hasattr(kv_connector, 'worker_handler'):
                            print(f"[DEBUG] kv_connector.worker_handler: {kv_connector.worker_handler}", flush=True)
                        if hasattr(kv_connector, 'worker_handler') and kv_connector.worker_handler:
                            orch = kv_connector.worker_handler._orchestrator
                            if orch is None:
                                return {"enabled": False}
                            
                            residency = orch.residency
                            scored_blocks = len(orch._scored)
                            staging = orch.staging_pool
                            # Calculate used slots: total - free
                            staging_used = staging.num_slots - len(staging._free)
                            staging_total = staging.num_slots
                            cpu_pool = orch.offload_manager._cpu_pool
                            cpu_free = cpu_pool.num_free
                            
                            return {
                                "enabled": True,
                                "residency": str(residency),
                                "scored_blocks": scored_blocks,
                                "staging_used": staging_used,
                                "staging_total": staging_total,
                                "cpu_free": cpu_free,
                            }
            return {"enabled": False, "error": "No model_runner or connector"}
        
        results = llm.llm_engine.engine_core.collective_rpc(
            _get_offload_stats,
            args=(),
        )
        
        # Results is a list (one per worker), take first for single worker
        if results and isinstance(results, list):
            stats = results[0]
        else:
            stats = results
            
        if not stats.get("enabled", False):
            print("Offloading not enabled")
            if "error" in stats:
                print(f"  Reason: {stats['error']}")
            return
        
        print("\n" + "="*60)
        print("OFFLOAD STATISTICS")
        print("="*60)
        
        print(f"Residency: {stats['residency']}")
        print(f"Scored blocks: {stats['scored_blocks']}")
        print(f"Staging pool: {stats['staging_used']}/{stats['staging_total']} slots used")
        print(f"CPU pool: {stats['cpu_free']} slots free")
        
    except Exception as e:
        print(f"Could not get offload stats: {e}")
        import traceback
        traceback.print_exc()


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    print("hi")
    import argparse
    
    parser = argparse.ArgumentParser(description="vLLM with KV Offloading + Sparse Oracle Top-K")
    parser.add_argument("--model", default="meta-llama/Llama-2-7b-hf", help="Model name")
    parser.add_argument("--prompts", nargs="+", default=None, help="Input prompts")
    parser.add_argument("--max-tokens", type=int, default=128, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p sampling")
    parser.add_argument("--topk", type=float, default=0.1, help="Sparse attention fraction (0.0-1.0)")
    parser.add_argument("--channel-num", type=int, default=64, help="Channels for oracle top-k scoring")
    parser.add_argument("--max-seq-len", type=int, default=4096, help="Max sequence length")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.35, help="GPU memory utilization")
    parser.add_argument("--staging-slots", type=int, default=64, help="GPU staging slots for sparse attention")
    parser.add_argument("--enforce-eager", action="store_true", help="Disable torch.compile and CUDAGraphs")
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="Number of GPUs for tensor parallelism")
    parser.add_argument("--profile", action="store_true", help="Enable profiling")
    parser.add_argument("--no-sparse", action="store_true", help="Disable sparse attention")
    
    args = parser.parse_args()
    
    # Run inference
    outputs, llm = run_inference(
        model_name=args.model,
        prompts=args.prompts,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        topk=args.topk,
        channel_num=args.channel_num,
        max_seq_len=args.max_seq_len,
        enable_profiling=args.profile,
        use_sparse=not args.no_sparse,
        gpu_memory_utilization=args.gpu_memory_utilization,
        staging_slots=args.staging_slots,
        enforce_eager=args.enforce_eager,
        tensor_parallel_size=args.tensor_parallel_size,
    )
    
    # Print results
    print("\n" + "="*60)
    print("GENERATION RESULTS")
    print("="*60)
    for i, output in enumerate(outputs):
        print(f"\nPrompt {i}: {output.prompt}")
        print(f"Generated: {output.outputs[0].text}")
    
    # Print offload stats
    print_offload_stats(llm)