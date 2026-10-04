# SPDX-License-Identifier: Apache-2.0
"""Tiny debug-print gate for the offload package (and the skylight backend /
vllm fork call sites that opt into it).

Set OFFLOAD_PACKAGE_DEBUG=1 to enable the per-step [DEBUG] channels. Default
is silent. Fires once per process at import; not for structured logging.
"""

from __future__ import annotations

import os

ENABLED = os.environ.get("OFFLOAD_PACKAGE_DEBUG", "0") == "1"


def dbg(msg: str, *args, **kwargs) -> None:
    if ENABLED:
        kwargs.setdefault("flush", True)
        print(msg, *args, **kwargs)
