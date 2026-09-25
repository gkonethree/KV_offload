# SPDX-License-Identifier: Apache-2.0
"""OffloadPackageConnector registration for vLLM."""

from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorBase_V1

# Import to register the connector
from offload_package.integration.connector import OffloadPackageConnector

__all__ = ["OffloadPackageConnector"]
