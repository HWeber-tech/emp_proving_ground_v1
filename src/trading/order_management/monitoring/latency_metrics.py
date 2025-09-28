"""Latency monitoring for FIX order lifecycles."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from src.core.interfaces.metrics import HistogramLike
from src.operational.metrics_registry import MetricsRegistry, get_registry

from ..order_state_machine import (
    OrderExecutionEvent,
    OrderLifecycleSnapshot,
    OrderState,
)

__all__ = ["OrderLatencyMonitor", "LatencyMetrics"]


@dataclass(slots=True)
class LatencyMetrics:
    """Calculated latency artefacts for a lifecycle transition."""

    ack_latency: Optional[float] = None
    first_fill_latency: Optional[float] = None
    final_fill_latency: Optional[float] = None
    cancel_latency: Optional[float] = None
    reject_latency: Optional[float] = None


class OrderLatencyMonitor:
    """Publish latency metrics for order lifecycles to the metrics registry."""

    def __init__(
        self,
        *,
        registry: MetricsRegistry | None = None,
        buckets: Optional[list[float]] = None,
    ) -> None:
        self._registry = registry or get_registry()
        histogram_buckets = buckets or [0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0]
        self._ack_hist = self._registry.get_histogram(
            "order_ack_latency_seconds",
            "Latency from order creation to acknowledgement",
            histogram_buckets,
            ["venue"],
        )
        self._first_fill_hist = self._registry.get_histogram(
            "order_first_fill_latency_seconds",
            "Latency from order creation to first fill",
            histogram_buckets,
            ["venue"],
        )
        self._final_fill_hist = self._registry.get_histogram(
            "order_final_fill_latency_seconds",
            "Latency from order creation to final fill",
            histogram_buckets,
            ["venue"],
        )
        self._cancel_hist = self._registry.get_histogram(
            "order_cancel_latency_seconds",
            "Latency from order creation to cancel acknowledgement",
            histogram_buckets,
            ["venue"],
        )
        self._reject_hist = self._registry.get_histogram(
            "order_reject_latency_seconds",
            "Latency from order creation to reject",
            histogram_buckets,
            ["venue"],
        )

    # ------------------------------------------------------------------
    def record_transition(
        self,
        state: OrderState,
        event: OrderExecutionEvent,
        snapshot: OrderLifecycleSnapshot,
    ) -> LatencyMetrics:
        """Calculate and publish latency metrics for the supplied transition."""

        created_at = state.created_at
        venue = state.metadata.venue or "unknown"
        metrics = LatencyMetrics()

        def _observe(hist: HistogramLike, start: Optional[datetime], end: Optional[datetime]) -> Optional[float]:
            if start is None or end is None:
                return None
            latency = (end - start).total_seconds()
            if latency < 0:
                return None
            hist.labels(venue=venue).observe(latency)
            return latency

        if event.event_type == "acknowledged" and state.acknowledged_at == event.timestamp:
            metrics.ack_latency = _observe(self._ack_hist, created_at, state.acknowledged_at)

        if event.event_type in {"partial_fill", "filled"} and state.first_fill_at == event.timestamp:
            metrics.first_fill_latency = _observe(
                self._first_fill_hist, created_at, state.first_fill_at
            )

        if event.event_type == "filled" and state.final_fill_at == event.timestamp:
            metrics.final_fill_latency = _observe(
                self._final_fill_hist, created_at, state.final_fill_at
            )

        if event.event_type == "cancelled" and state.cancelled_at == event.timestamp:
            metrics.cancel_latency = _observe(self._cancel_hist, created_at, state.cancelled_at)

        if event.event_type == "rejected" and state.rejected_at == event.timestamp:
            metrics.reject_latency = _observe(self._reject_hist, created_at, state.rejected_at)

        return metrics
