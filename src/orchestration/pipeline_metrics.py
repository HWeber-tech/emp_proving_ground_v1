"""Pipeline heartbeat and latency monitoring helpers."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import UTC, datetime
import math
import time
from typing import Mapping, Sequence

__all__ = ["PipelineLatencySnapshot", "PipelineLatencyMonitor"]


def _percentile(sorted_samples: Sequence[float], percentile: float) -> float | None:
    """Estimate ``percentile`` from ``sorted_samples`` using linear interpolation."""

    if not sorted_samples:
        return None
    if percentile <= 0:
        return float(sorted_samples[0])
    if percentile >= 100:
        return float(sorted_samples[-1])

    rank = (len(sorted_samples) - 1) * (percentile / 100.0)
    lower_index = math.floor(rank)
    upper_index = math.ceil(rank)
    lower_value = float(sorted_samples[lower_index])
    upper_value = float(sorted_samples[upper_index])
    if lower_index == upper_index:
        return lower_value
    weight = rank - lower_index
    return lower_value + (upper_value - lower_value) * weight


@dataclass(frozen=True)
class PipelineLatencySnapshot:
    """Serialised snapshot describing pipeline heartbeat and latency percentiles."""

    heartbeat: Mapping[str, object]
    latency: Mapping[str, Mapping[str, float | int | None]]


class PipelineLatencyMonitor:
    """Accumulates pipeline heartbeat ticks and stage latency percentiles."""

    _DEFAULT_STAGES = ("ingest", "signal", "order", "ack", "total")

    def __init__(self, *, history_size: int = 512) -> None:
        if history_size <= 0:
            raise ValueError("history_size must be positive")
        self._history_size = int(history_size)
        self._histories: dict[str, deque[float]] = {
            stage: deque(maxlen=self._history_size) for stage in self._DEFAULT_STAGES
        }
        self._tick_count: int = 0
        self._orders_attempted: int = 0
        self._last_tick_at: datetime | None = None
        self._last_tick_monotonic: float | None = None

    # ------------------------------------------------------------------
    def observe_tick(
        self,
        latencies: Mapping[str, float | None] | None,
        *,
        timestamp: datetime | None = None,
        order_attempted: bool = False,
    ) -> None:
        """Record a pipeline tick with optional latency measurements."""

        self._tick_count += 1
        if order_attempted:
            self._orders_attempted += 1

        if latencies:
            for stage, value in latencies.items():
                if value is None:
                    continue
                try:
                    numeric = float(value)
                except (TypeError, ValueError):
                    continue
                if not math.isfinite(numeric) or numeric < 0.0:
                    continue
                history = self._histories.get(stage)
                if history is None:
                    history = deque(maxlen=self._history_size)
                    self._histories[stage] = history
                history.append(numeric)

        now = timestamp or datetime.now(tz=UTC)
        if self._last_tick_at is None or now >= self._last_tick_at:
            self._last_tick_at = now
            self._last_tick_monotonic = time.perf_counter()

    # ------------------------------------------------------------------
    def snapshot(self) -> PipelineLatencySnapshot:
        """Return a serialisable snapshot of heartbeat and latency metrics."""

        heartbeat: dict[str, object] = {
            "ticks": self._tick_count,
            "orders_attempted": self._orders_attempted,
            "last_tick_at": self._last_tick_at.isoformat() if self._last_tick_at else None,
        }
        if self._last_tick_monotonic is not None:
            heartbeat["seconds_since_last_tick"] = max(
                time.perf_counter() - self._last_tick_monotonic,
                0.0,
            )
        else:
            heartbeat["seconds_since_last_tick"] = None

        latency_payload: dict[str, dict[str, float | int | None]] = {}
        for stage, history in sorted(self._histories.items()):
            latency_payload[stage] = self._summarise_history(history)

        return PipelineLatencySnapshot(heartbeat=heartbeat, latency=latency_payload)

    # ------------------------------------------------------------------
    def _summarise_history(self, samples: deque[float]) -> dict[str, float | int | None]:
        values = list(samples)
        if not values:
            return {
                "samples": 0,
                "p50": None,
                "p99": None,
                "p99_9": None,
                "latest": None,
            }
        sorted_values = sorted(values)
        return {
            "samples": len(values),
            "p50": _percentile(sorted_values, 50.0),
            "p99": _percentile(sorted_values, 99.0),
            "p99_9": _percentile(sorted_values, 99.9),
            "latest": float(values[-1]),
        }
