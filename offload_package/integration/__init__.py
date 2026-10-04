"""offload_package.integration — lazy exports to avoid import cycles with vLLM.

Eagerly importing the connector/worker modules from here creates a package-level
cycle: vllm.distributed.kv_transfer.kv_connector.v1 imports the registered
offload_package_connector module, which imports back into this package. With
lazy exports, importing ``offload_package.integration`` never touches vLLM.
"""

from .manager import OffloadOrchestrator  # lightweight: no vllm imports
from .vllm_bridge import initialize_for_runner, prepare_sparse_pages

__all__ = [
    "OffloadOrchestrator",
    "initialize_for_runner",
    "prepare_sparse_pages",
    "OffloadPackageConnector",
    "OffloadPackageScheduler",
    "OffloadPackageWorker",
    "OffloadPackageMetadata",
    "OffloadPackageWorkerMetadata",
]


def __getattr__(name: str):
    if name == "OffloadPackageConnector":
        from .connector import OffloadPackageConnector
        return OffloadPackageConnector
    if name == "OffloadPackageScheduler":
        from .scheduler_connector import OffloadPackageScheduler
        return OffloadPackageScheduler
    if name == "OffloadPackageWorker":
        from .worker_connector import OffloadPackageWorker
        return OffloadPackageWorker
    if name in ("OffloadPackageMetadata", "OffloadPackageWorkerMetadata"):
        from .metadata import OffloadPackageMetadata, OffloadPackageWorkerMetadata
        return {"OffloadPackageMetadata": OffloadPackageMetadata,
                "OffloadPackageWorkerMetadata": OffloadPackageWorkerMetadata}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")