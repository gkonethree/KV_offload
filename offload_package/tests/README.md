# Offload Package Tests

This directory contains comprehensive tests for the KV cache offloading system.

## Test Files

### `conftest.py`
Shared pytest fixtures:
- `torch_device`: CUDA device if available, else CPU
- `sample_kv_cache`: Sample K/V cache tensors [num_pages, num_heads, head_dim]
- `sample_cpu_cache`: Sample CPU cache [num_layers, num_slots, 2, page_size, num_heads, head_dim]

### `test_scoring.py` (10 tests)
Tests for the scoring module:
- `TestPagedEvictionScorer`:
  - Initialization with valid/invalid parameters
  - Score storage and decay semantics
  - Block selection for eviction
  - Forget and reset functionality
- `TestComputeBlockKVNorms`:
  - Basic norm computation
  - Custom logical block IDs
  - Empty input handling
  - Different KV layouts
  - Input validation

### `test_residency.py` (15 tests)
Tests for residency tracking state machine:
- `TestBlockLocation`: Creating various residency locations
- `TestResidencyTable`:
  - GPU → CPU → STAGING → IN_FLIGHT transitions
  - Marking blocks complete
  - Querying by residency type
  - Per-request block queries
  - State consistency checks

### `test_offload_manager.py` (11 tests)
Tests for the offload manager:
- `TestCPUBlockPool`:
  - Pool initialization and exhaustion
  - Slot allocation and freeing
- `TestKVOffloadManager`:
  - Block scoring
  - GPU budget enforcement
  - Offload fraction calculation
  - CPU pool management
  - Event synchronization

### `test_core.py` (3 tests)
Existing core integration tests:
- Decay semantics for scoring
- Per-request decode interval timing
- Paged index patching with staging

### `test_integration.py` (12 tests)
Full pipeline integration tests:
- `TestOffloadOrchestrator`:
  - Initialization and component setup
  - Request block tracking
  - Prefill completion and offloading
  - Per-request W-token intervals
  - Staging preparation
  - Request cleanup
  - Multiple concurrent requests
  - Residency consistency
  - CPU memory budget enforcement
  - Scoring triggers

## Running Tests

### Run all tests:
```bash
pytest offload_package/tests/ -v
```

### Run specific test file:
```bash
pytest offload_package/tests/test_scoring.py -v
```

### Run specific test:
```bash
pytest offload_package/tests/test_scoring.py::TestPagedEvictionScorer::test_decay -v
```

### Run with coverage:
```bash
pytest offload_package/tests/ --cov=offload_package --cov-report=html
```

## Test Coverage

The test suite covers:
- **Scoring**: V/K norm computation, decay-weighted averaging, block selection
- **Residency**: Full state machine with GPU ↔ CPU ↔ STAGING ↔ IN_FLIGHT transitions
- **Offload Manager**: Block scoring, GPU→CPU copies, CPU pool management
- **Controller**: Per-request W-token tracking and decode interval triggers
- **Orchestrator**: Request lifecycle, concurrent requests, resource cleanup
- **Integration**: Full end-to-end offloading pipeline

## Design Principles

1. **Isolation**: Each test focuses on one component
2. **Reusability**: Shared fixtures in conftest.py
3. **Real Objects**: Minimal mocking; tests use real objects where possible
4. **Edge Cases**: Tests cover normal paths, boundaries, and error conditions
5. **Consistency**: Tests verify state consistency across operations

## Notes

- Tests use CUDA if available, fall back to CPU
- CPU cache fixtures pre-allocate pinned memory
- Block IDs use the BlockRef(request_id, block_idx) format
- Residency transitions follow a strict state machine
- CPU slot allocation is LIFO (last in, first out) by design
