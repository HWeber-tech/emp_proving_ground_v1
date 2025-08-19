"""
Core Telemetry Port (Protocol)
==============================

Defines a minimal metrics interface decoupled from concrete exporters.
Domain layers depend on this port only.

Concrete implementations live in higher layers (e.g., src/operational/metrics.py)
and can register themselves via set_metrics_sink().
"""

from __future__ import annotations

from typing import Dict, List, Optional, Protocol, runtime_checkable


@runtime_checkable
class MetricsSink(Protocol):
    """
    Abstract sink for metrics/telemetry.

    Methods are best-effort; implementations must not raise.
    """

    def set_gauge(self, name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Set a gauge to a value."""
        ...

    def inc_counter(self, name: str, amount: float = 1.0, labels: Optional[Dict[str, str]] = None) -> None:
        """Increment a counter."""
        ...

    def observe_histogram(
        self,
        name: str,
        value: float,
        buckets: Optional[List[float]] = None,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """Observe a value for a histogram."""
        ...


class _NoOpMetricsSink:
    """Default no-op implementation used until an adapter is registered."""

    def set_gauge(self, name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        return

    def inc_counter(self, name: str, amount: float = 1.0, labels: Optional[Dict[str, str]] = None) -> None:
        return

    def observe_histogram(
        self,
        name: str,
        value: float,
        buckets: Optional[List[float]] = None,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        return


_SINK: Optional[MetricsSink] = None
_NOOP = _NoOpMetricsSink()


def set_metrics_sink(sink: MetricsSink) -> None:
    """Register a process-wide metrics sink implementation."""
    global _SINK
    _SINK = sink


def get_metrics_sink() -> MetricsSink:
    """Return the currently-registered metrics sink, or a no-op sink."""
    return _SINK if _SINK is not None else _NOOP


def has_metrics_sink() -> bool:
    """True if a non-default sink is registered."""
    return _SINK is not None