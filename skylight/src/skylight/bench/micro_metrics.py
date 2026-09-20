"""Lightweight jsonl micro-metrics logger (SAH-compatible shape).

Used by the sparse vLLM server to record per-layer sparsity fractions.
Controlled via env:

  SKYLIGHT_METRICS_LOG_DIR  — directory; writes ``micro_metrics.jsonl`` here
  SKYLIGHT_METRICS_SAMPLING — float in [0,1]; default 0.01 for agentic runs
"""
from __future__ import annotations

import inspect
import json
import os
import random
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


class MicroMetricLogger:
    """Thread-safe append-only jsonl logger with probabilistic sampling."""

    _lock = threading.Lock()
    _instance: Optional["MicroMetricLogger"] = None

    def __init__(
        self,
        log_dir: Optional[Path] = None,
        sampling: float = 0.01,
    ) -> None:
        self._log_dir = log_dir
        self._sampling = max(0.0, min(1.0, sampling))
        self._path: Optional[Path] = None
        self._fh = None
        if log_dir is not None:
            log_dir.mkdir(parents=True, exist_ok=True)
            self._path = log_dir / "micro_metrics.jsonl"
            self._fh = self._path.open("a", encoding="utf-8")

    @classmethod
    def from_env(cls) -> "MicroMetricLogger":
        """Singleton keyed off ``SKYLIGHT_METRICS_LOG_DIR`` (disabled if unset)."""
        log_dir_raw = os.environ.get("SKYLIGHT_METRICS_LOG_DIR")
        if not log_dir_raw:
            if cls._instance is None:
                cls._instance = cls(log_dir=None)
            return cls._instance
        log_dir = Path(log_dir_raw)
        sampling_raw = os.environ.get("SKYLIGHT_METRICS_SAMPLING", "0.01")
        try:
            sampling = float(sampling_raw)
        except ValueError:
            sampling = 0.01
        with cls._lock:
            if (
                cls._instance is None
                or cls._instance._log_dir != log_dir
                or cls._instance._sampling != sampling
            ):
                if cls._instance is not None:
                    cls._instance.flush()
                cls._instance = cls(log_dir=log_dir, sampling=sampling)
        return cls._instance

    @property
    def enabled(self) -> bool:
        return self._fh is not None

    def log(
        self,
        metric: str,
        value: Any,
        metadata: Optional[dict[str, Any]] = None,
        location: str = "",
    ) -> None:
        if self._fh is None:
            return
        if random.random() > self._sampling:
            return
        if not location:
            frame = inspect.currentframe()
            if frame and frame.f_back:
                co = frame.f_back.f_code
                location = f"{co.co_name}"
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metric": metric,
            "value": value,
            "metadata": metadata or {},
            "location": location,
        }
        line = json.dumps(event, default=str) + "\n"
        with self._lock:
            self._fh.write(line)
            self._fh.flush()

    def flush(self) -> None:
        with self._lock:
            if self._fh is not None:
                self._fh.flush()

    def close(self) -> None:
        with self._lock:
            if self._fh is not None:
                self._fh.flush()
                self._fh.close()
                self._fh = None


def get_logger() -> MicroMetricLogger:
    return MicroMetricLogger.from_env()
