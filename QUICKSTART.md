# Quick Start Guide

## Enable Offloading

Set environment variables before running vLLM:

```bash
export OFFLOAD_PACKAGE_ENABLED=1
export OFFLOAD_PACKAGE_OFFLOAD_FRACTION=0.20
export OFFLOAD_PACKAGE_CPU_SLOTS=4096
export OFFLOAD_PACKAGE_STAGING_SLOTS=64
export OFFLOAD_PACKAGE_INTERVAL=32

# Run vLLM as usual
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-2-7b-hf \
    --gpu-memory-utilization 0.9
```

## Configuration Parameters

- `OFFLOAD_PACKAGE_ENABLED=1`: Enable (0 to disable)
- `OFFLOAD_PACKAGE_OFFLOAD_FRACTION=0.20`: Offload 20% of GPU blocks after prefill (0-1)
- `OFFLOAD_PACKAGE_CPU_SLOTS=4096`: Pre-allocate 4096 CPU slots
- `OFFLOAD_PACKAGE_STAGING_SLOTS=64`: Reserve 64 GPU pages for staging
- `OFFLOAD_PACKAGE_INTERVAL=32`: Trigger offload every 32 decode tokens
- `OFFLOAD_PACKAGE_SCORER_DECAY=1.0`: Use only latest scores (1=yes, 0=never update)
- `OFFLOAD_PACKAGE_SCORER_HEAD_REDUCTION=mean`: Score reduction method (mean or min)

## Tuning Guide

### For Small Models (< 7B)
```bash
export OFFLOAD_PACKAGE_OFFLOAD_FRACTION=0.10
export OFFLOAD_PACKAGE_CPU_SLOTS=2048
export OFFLOAD_PACKAGE_STAGING_SLOTS=32
export OFFLOAD_PACKAGE_INTERVAL=16
```

### For Large Models (> 30B)
```bash
export OFFLOAD_PACKAGE_OFFLOAD_FRACTION=0.30
export OFFLOAD_PACKAGE_CPU_SLOTS=8192
export OFFLOAD_PACKAGE_STAGING_SLOTS=128
export OFFLOAD_PACKAGE_INTERVAL=64
```

### For High-Throughput (many concurrent requests)
```bash
export OFFLOAD_PACKAGE_OFFLOAD_FRACTION=0.25
export OFFLOAD_PACKAGE_CPU_SLOTS=6144
export OFFLOAD_PACKAGE_STAGING_SLOTS=96
export OFFLOAD_PACKAGE_INTERVAL=32
```

### For Low-Latency (minimize attention delay)
```bash
export OFFLOAD_PACKAGE_OFFLOAD_FRACTION=0.15
export OFFLOAD_PACKAGE_CPU_SLOTS=3072
export OFFLOAD_PACKAGE_STAGING_SLOTS=48
export OFFLOAD_PACKAGE_INTERVAL=64
```

## Monitoring

### Check Offload Status
```python
# In your vLLM worker/runner code
if hasattr(runner, "_offload_orchestrator"):
    orch = runner._offload_orchestrator
    residency = orch.residency
    
    # Total blocks
    print(f"Total blocks: {len(residency)}")
    
    # GPU vs CPU split
    gpu_blocks = residency.all_gpu_blocks()
    cpu_blocks = [
        ref for ref in residency._blocks
        if residency.get(ref).residency == Residency.CPU
    ]
    print(f"GPU: {len(gpu_blocks)}, CPU: {len(cpu_blocks)}")
    
    # CPU pool status
    manager = orch.offload_manager
    used = manager._cpu_pool.num_free
    total = manager._cpu_pool.num_free + len(manager._store_events)
    print(f"CPU pool: {used}/{total} used")
else:
    print("Offloading not enabled")
```

### Check for Errors
Look for these warning messages in logs:
- `"CPU offload pool exhausted"`: Too many blocks offloaded
- `"Cannot evict staging block"`: Staging area full
- `"Failed to track"`: Block tracking error (non-fatal)
- `"Failed to patch block tables"`: Attention patching error (degrade)

## Troubleshooting

### Model Accuracy Issues
1. Disable offloading: `OFFLOAD_PACKAGE_ENABLED=0`
2. Compare outputs: If disabled output is correct, offloading has a bug
3. Check logs for residency corruption errors
4. Verify block tracking is working (enable debug logging)

### High Latency
1. Check CPU memory availability: `free -h`
2. Reduce `OFFLOAD_PACKAGE_OFFLOAD_FRACTION` (offload less)
3. Increase `OFFLOAD_PACKAGE_INTERVAL` (trigger less often)
4. Profile attention: Check if `prepare_attn()` is slow

### Out of Memory
1. Reduce `OFFLOAD_PACKAGE_CPU_SLOTS` (free CPU memory)
2. Reduce `OFFLOAD_PACKAGE_STAGING_SLOTS` (reduce temporary GPU usage)
3. Reduce batch size

### CPU Pool Exhausted
1. Reduce `OFFLOAD_PACKAGE_OFFLOAD_FRACTION` (offload less aggressively)
2. Increase `OFFLOAD_PACKAGE_INTERVAL` (trigger less frequently)
3. Allocate more CPU memory and increase `OFFLOAD_PACKAGE_CPU_SLOTS`

## Performance Expectations

### Memory Savings
- Typical: 20-30% GPU memory saved
- Range: 10-50% depending on model and configuration

### Latency Impact
- Prefill: ~0% (batched, overlaps)
- Decode: 5-15% per-token latency increase
  - Depends on: offload fraction, staging fill rate, CPU bandwidth

### Throughput
- Typically maintains or improves throughput (more concurrent requests fit in GPU)
- Some latency increase per-request, but more requests can run

## Running Tests

### Unit Tests
```bash
cd /home/gaurav_kumar/KV_offload
pytest offload_package/tests/test_scoring.py -v
pytest offload_package/tests/test_residency.py -v
pytest offload_package/tests/test_offload_manager.py -v
```

### Integration Tests
```bash
pytest offload_package/tests/test_integration.py -v
pytest offload_package/tests/test_core.py -v
```

### All Tests
```bash
pytest offload_package/tests/ -v --tb=short
```

## Architecture References

- **Architecture Overview**: See `ARCHITECTURE.md`
- **Implementation Details**: See `IMPLEMENTATION.md`
- **Test Documentation**: See `offload_package/tests/README.md`

## Getting Help

1. **Check logs**: Enable debug logging with `logging.basicConfig(level=logging.DEBUG)`
2. **Review documentation**: Read ARCHITECTURE.md and IMPLEMENTATION.md
3. **Check test examples**: Look at tests in offload_package/tests/
4. **Monitor state**: Use monitoring code above to inspect orchestrator state

## What's Next?

After successfully deploying offloading:

1. **Monitor**: Track GPU memory savings and latency impact
2. **Tune**: Adjust hyperparameters for your hardware/workload
3. **Optimize**: If needed, explore future enhancements:
   - Predictive prefetching
   - Direct CPU attention
   - Hierarchical storage (SSD/NVMe)
4. **Contribute**: Report issues, suggest improvements
