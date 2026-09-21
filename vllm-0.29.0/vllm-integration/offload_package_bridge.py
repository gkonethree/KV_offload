"""Small bridge intended to be imported by the vLLM fork.

This file deliberately contains only vLLM-facing glue.  The scoring,
residency, CPU offload, and staging implementation remains in offload_package.
"""
from __future__ import annotations

from collections import defaultdict

import torch

from offload_package.integration.vllm_bridge import initialize_for_runner


def install_runner_state(runner) -> None:
    """Initialize per-runner state. Safe to call more than once."""
    if not hasattr(runner, "_offload_req_state"):
        runner._offload_req_state = {}
    if not hasattr(runner, "_offload_orchestrator"):
        runner._offload_orchestrator = None
    if not hasattr(runner, "_offload_current_req_ids"):
        runner._offload_current_req_ids = []


def on_initialize_kv_cache(runner, kv_cache_config) -> None:
    install_runner_state(runner)
    # vLLM BlockPool still owns kv_cache_config.num_blocks.  Tail pages are
    # added to the underlying tensors by the allocator hook described in the
    # integration README, so this is the first physical staging page.
    # Use the orchestrator already created by model_runner
    if hasattr(runner, "_offload_orchestrator_instance") and runner._offload_orchestrator_instance is not None:
        runner._offload_orchestrator = runner._offload_orchestrator_instance
    else:
        runner._offload_orchestrator = initialize_for_runner(
            runner,
            normal_gpu_pages=kv_cache_config.num_blocks,
            block_size=kv_cache_config.kv_cache_groups[0].kv_cache_spec.block_size,
        )


def on_new_request(runner, req) -> None:
    """Register a request using scheduler-provided block IDs."""
    if runner._offload_orchestrator is None:
        return
    req_id = str(req.req_id)
    prompt_len = len(req.prompt_token_ids or ())
    block_ids = list(req.block_ids or ())
    runner._offload_req_state[req_id] = {
        "prompt_len": prompt_len,
        "total_tokens": 0,
        "prefill_done": prompt_len == 0,
        "block_ids": block_ids,
    }
    runner._offload_orchestrator.sync_request_blocks(req_id, block_ids, 0)


def on_cached_request_update(runner, req_id, new_block_ids, num_scheduled_tokens) -> None:
    """Track newly allocated blocks and cumulative tokens for a request."""
    if runner._offload_orchestrator is None:
        return
    req_id = str(req_id)
    state = runner._offload_req_state.get(req_id)
    if state is None:
        return
    if new_block_ids:
        state["block_ids"].extend(int(x) for x in new_block_ids)
    state["total_tokens"] += int(num_scheduled_tokens)
    if state["total_tokens"] >= state["prompt_len"]:
        state["prefill_done"] = True
    runner._offload_orchestrator.sync_request_blocks(
        req_id,
        state["block_ids"],
        state["total_tokens"],
    )


def on_prefill_new_request(runner, req, num_scheduled_tokens: int) -> None:
    if runner._offload_orchestrator is None:
        return
    state = runner._offload_req_state.get(str(req.req_id))
    if state is None:
        on_new_request(runner, req)
        state = runner._offload_req_state[str(req.req_id)]
    state["total_tokens"] += int(num_scheduled_tokens)
    state["prefill_done"] = state["total_tokens"] >= state["prompt_len"]
    runner._offload_orchestrator.sync_request_blocks(
        str(req.req_id),
        state["block_ids"],
        state["total_tokens"],
    )
    if state["prefill_done"]:
        runner._offload_orchestrator.on_prefill_complete(str(req.req_id))


def on_decode_batch(runner, request_ids, num_scheduled_tokens_by_request) -> None:
    if runner._offload_orchestrator is None:
        return
    active = []
    for req_id in request_ids:
        state = runner._offload_req_state.get(str(req_id))
        if state is None or not state["prefill_done"]:
            continue
        state["total_tokens"] += int(num_scheduled_tokens_by_request.get(req_id, 1))
        runner._offload_orchestrator.sync_request_blocks(
            str(req_id), state["block_ids"], state["total_tokens"]
        )
        active.append(str(req_id))
    if active:
        runner._offload_current_req_ids = active
        
        # Debug: Print block residency for each active request (skip warmup)
        orch = runner._offload_orchestrator
        for req_id in active:
            if req_id.startswith("_warmup_"):
                continue
            state = runner._offload_req_state.get(req_id)
            if state:
                block_ids = state["block_ids"]
                total_tokens = state["total_tokens"]
                print(f"[BRIDGE DECODE] req={req_id}, total_tokens={total_tokens}, blocks={block_ids}")
                # Print ALL residency entries for this request
                for ref, loc in orch.residency._blocks.items():
                    if ref.request_id == req_id:
                        print(f"  block_idx={ref.block_idx}, gpu_block_id={loc.gpu_block_id}, residency={loc.residency.name}, complete={loc.complete}")
        
        runner._offload_orchestrator.on_decode_step(active, 1)


def on_request_done(runner, req_id) -> None:
    if runner._offload_orchestrator is not None:
        runner._offload_orchestrator.on_request_done(str(req_id))
    runner._offload_req_state.pop(str(req_id), None)
