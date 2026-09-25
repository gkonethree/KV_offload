# SPDX-License-Identifier: Apache-2.0
"""Registration of OffloadPackageConnector with KVConnectorFactory."""

def register_offload_package_connector():
    """Register OffloadPackageConnector with KVConnectorFactory.
    
    This function should be called after all imports are resolved to avoid
    circular import issues.
    """
    try:
        from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
        KVConnectorFactory.register_connector(
            "OffloadPackageConnector",
            "offload_package.integration.connector",
            "OffloadPackageConnector",
        )
        return True
    except ImportError:
        return False
    except ValueError as e:
        if "already registered" in str(e):
            return True
        raise

# Auto-register when this module is imported
register_offload_package_connector()
