# SPDX-License-Identifier: Apache-2.0
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorRole,
    SupportsHMA,
    supports_hma,
)
from vllm.distributed.kv_transfer.kv_connector.v1.decode_bench_connector import (  # noqa: E501
    DecodeBenchConnector,
)

__all__ = [
    "KVConnectorRole",
    "KVConnectorBase_V1",
    "supports_hma",
    "SupportsHMA",
    "DecodeBenchConnector",
    "OffloadPackageConnector",
]


def __getattr__(name: str):
    # Lazy: importing offload_package_connector eagerly at package init time
    # creates a cycle (it imports offload_package.integration.connector, which
    # itself imports vllm.distributed.kv_transfer...). The connector factory
    # loads it via kv_connector_module_path anyway, so nothing is lost.
    if name == "OffloadPackageConnector":
        from vllm.distributed.kv_transfer.kv_connector.v1.offload_package_connector import (
            OffloadPackageConnector,
        )
        return OffloadPackageConnector
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")