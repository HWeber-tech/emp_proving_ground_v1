"""Collect resource usage metrics for execution health checks."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping, MutableMapping

UTC = timezone.utc


@dataclass(frozen=True)
class ResourceSample:
    """Represents a sampled view of process resources."""

    timestamp: datetime
    cpu_percent: float | None
    memory_mb: float | None
    memory_percent: float | None


class ResourceUsageMonitor:
    """Sample CPU and memory usage for the current process."""

    def __init__(self, *, process: Any | None = None) -> None:
        self._psutil = self._import_psutil()
        self._process = self._resolve_process(process)
        self._last_snapshot: Mapping[str, object | None] = self._build_snapshot(None)

    def sample(self) -> Mapping[str, object | None]:
        """Capture a new resource usage sample and return the snapshot."""

        timestamp = datetime.now(tz=UTC)
        sample = self._collect_sample(timestamp)
        self._last_snapshot = self._build_snapshot(sample)
        return self._last_snapshot

    def snapshot(self) -> Mapping[str, object | None]:
        """Return the most recent snapshot."""

        return dict(self._last_snapshot)

    def _collect_sample(self, timestamp: datetime) -> ResourceSample | None:
        if self._process is None:
            return None

        cpu_percent = self._safe_cpu_percent()
        memory_mb, memory_percent = self._safe_memory_usage()
        return ResourceSample(
            timestamp=timestamp,
            cpu_percent=cpu_percent,
            memory_mb=memory_mb,
            memory_percent=memory_percent,
        )

    def _safe_cpu_percent(self) -> float | None:
        if self._process is None:
            return None
        try:
            return float(self._process.cpu_percent(interval=None))
        except Exception:
            return None

    def _safe_memory_usage(self) -> tuple[float | None, float | None]:
        if self._process is None:
            return (None, None)
        try:
            memory_info = self._process.memory_info()
            memory_mb = float(memory_info.rss) / 1024.0 / 1024.0
        except Exception:
            memory_mb = None
        try:
            memory_percent = float(self._process.memory_percent())
        except Exception:
            memory_percent = None
        return (memory_mb, memory_percent)

    def _build_snapshot(self, sample: ResourceSample | None) -> Mapping[str, object | None]:
        if sample is None:
            return {
                "timestamp": None,
                "cpu_percent": None,
                "memory_mb": None,
                "memory_percent": None,
            }
        payload: MutableMapping[str, object | None] = {
            "timestamp": sample.timestamp.astimezone(UTC).isoformat(),
            "cpu_percent": sample.cpu_percent,
            "memory_mb": sample.memory_mb,
            "memory_percent": sample.memory_percent,
        }
        return payload

    def _import_psutil(self) -> Any | None:
        try:
            import psutil  # type: ignore
        except Exception:
            return None
        return psutil

    def _resolve_process(self, process: Any | None) -> Any | None:
        if self._psutil is None:
            return None
        if process is not None:
            return process
        try:
            return self._psutil.Process()
        except Exception:
            return None
