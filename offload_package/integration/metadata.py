# SPDX-License-Identifier: Apache-2.0
"""Metadata for OffloadPackageConnector."""

from dataclasses import dataclass, field
from typing import Optional

import torch

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorMetadata,
    KVConnectorWorkerMetadata,
)

INVALID_JOB_ID = -1


@dataclass
class OffloadPackageMetadata(KVConnectorMetadata):
    """Scheduler → Worker metadata for offload/staging operations."""

    # Store (offload GPU → CPU)
    store_event: int = INVALID_JOB_ID
    store_gpu_blocks: list[int] = field(default_factory=list)
    store_cpu_blocks: list[int] = field(default_factory=list)

    # Load (fetch CPU → GPU staging for sparse attention)
    load_event: int = INVALID_JOB_ID
    load_gpu_blocks: list[int] = field(default_factory=list)
    load_cpu_blocks: list[int] = field(default_factory=list)

    # Sparse attention context
    sparse_idx: Optional[torch.Tensor] = None
    sparse_len: Optional[torch.Tensor] = None
    request_ids: list[str] = field(default_factory=list)

    # Reverse mapping for completion tracking
    load_event_to_reqs: dict[int, list[str]] = field(default_factory=dict)
    store_event_to_reqs: dict[int, list[str]] = field(default_factory=dict)


@dataclass
class OffloadPackageWorkerMetadata(KVConnectorWorkerMetadata):
    """Worker → Scheduler metadata for completed transfers."""

    completed_store_events: dict[int, int]
    completed_load_events: dict[int, int] = field(default_factory=dict)

    def aggregate(self, other: "KVConnectorWorkerMetadata") -> "KVConnectorWorkerMetadata":
        assert isinstance(other, OffloadPackageWorkerMetadata)
        merged_store = dict(self.completed_store_events)
        merged_load = dict(self.completed_load_events)
        for k, v in other.completed_store_events.items():
            merged_store[k] = merged_store.get(k, 0) + v
        for k, v in other.completed_load_events.items():
            merged_load[k] = merged_load.get(k, 0) + v
        return OffloadPackageWorkerMetadata(
            completed_store_events=merged_store,
            completed_load_events=merged_load,
        )