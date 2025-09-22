"""Typed stubs for src.core.telemetry.

Provides Protocol for MetricsSink and function set_metrics_sink used by operational metrics module to register a sink implementation while keeping runtime dependency optional.
"""

from typing import Dict, List, Optional, Protocol

class MetricsSink(Protocol):
    def set_gauge(
        self, name: str, value: float, labels: Optional[Dict[str, str]] = ...
    ) -> None: ...
    def inc_counter(
        self, name: str, amount: float = ..., labels: Optional[Dict[str, str]] = ...
    ) -> None: ...
    def observe_histogram(
        self,
        name: str,
        value: float,
        buckets: Optional[List[float]] = ...,
        labels: Optional[Dict[str, str]] = ...,
    ) -> None: ...

def set_metrics_sink(sink: MetricsSink) -> None: ...

__all__ = ["MetricsSink", "set_metrics_sink"]
