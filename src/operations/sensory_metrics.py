"""Build and publish sensory cortex metrics for runtime dashboards."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Iterable, Mapping, MutableMapping, Sequence

from src.core.event_bus import Event, EventBus, TopicBus
from src.operations.event_bus_failover import publish_event_with_failover
from src.operations.sensory_summary import SensorySummary, build_sensory_summary

__all__ = [
    "DimensionMetric",
    "SensoryMetrics",
    "build_sensory_metrics",
    "build_sensory_metrics_from_status",
    "publish_sensory_metrics",
]


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DimensionMetric:
    """Structured telemetry for a single sensory dimension."""

    name: str
    signal: float | None
    confidence: float | None
    state: str | None
    threshold_state: str | None

    def as_dict(self) -> Mapping[str, object]:
        payload: MutableMapping[str, object] = {"name": self.name}
        if self.signal is not None:
            payload["signal"] = self.signal
        if self.confidence is not None:
            payload["confidence"] = self.confidence
        if self.state is not None:
            payload["state"] = self.state
        if self.threshold_state is not None:
            payload["threshold_state"] = self.threshold_state
        return payload


@dataclass(frozen=True)
class SensoryMetrics:
    """Aggregated metrics derived from a sensory cortex snapshot."""

    symbol: str | None
    generated_at: datetime | None
    samples: int
    integrated_strength: float | None
    integrated_confidence: float | None
    integrated_direction: float | None
    dimensions: tuple[DimensionMetric, ...]
    drift_alerts: tuple[str, ...]

    def as_dict(self) -> Mapping[str, object]:
        return {
            "symbol": self.symbol,
            "generated_at": self.generated_at.isoformat() if self.generated_at else None,
            "samples": self.samples,
            "integrated_strength": self.integrated_strength,
            "integrated_confidence": self.integrated_confidence,
            "integrated_direction": self.integrated_direction,
            "dimensions": [dimension.as_dict() for dimension in self.dimensions],
            "drift_alerts": list(self.drift_alerts),
        }

    def filter_dimensions(self, names: Sequence[str]) -> tuple[DimensionMetric, ...]:
        lookup = {name.upper(): name for name in names}
        selected: list[DimensionMetric] = []
        for dimension in self.dimensions:
            if dimension.name.upper() in lookup:
                selected.append(dimension)
        return tuple(selected)


def build_sensory_metrics(summary: SensorySummary) -> SensoryMetrics:
    """Convert a sensory summary into dimension metrics."""

    dimensions = tuple(
        DimensionMetric(
            name=dimension.name,
            signal=dimension.signal,
            confidence=dimension.confidence,
            state=dimension.state,
            threshold_state=dimension.threshold_state,
        )
        for dimension in summary.dimensions
    )

    drift_alerts = _extract_drift_alerts(summary)

    return SensoryMetrics(
        symbol=summary.symbol,
        generated_at=summary.generated_at,
        samples=summary.samples,
        integrated_strength=summary.integrated_strength,
        integrated_confidence=summary.integrated_confidence,
        integrated_direction=summary.integrated_direction,
        dimensions=dimensions,
        drift_alerts=drift_alerts,
    )


def build_sensory_metrics_from_status(status: Mapping[str, object] | None) -> SensoryMetrics:
    """Build metrics from a sensory organ status mapping."""

    summary = build_sensory_summary(status)
    return build_sensory_metrics(summary)


def publish_sensory_metrics(
    metrics: SensoryMetrics,
    *,
    event_bus: EventBus,
    event_type: str = "telemetry.sensory.metrics",
    global_bus_factory: Callable[[], TopicBus] | None = None,
) -> None:
    """Publish sensory metrics using the hardened event-bus failover helper."""

    event = Event(
        type=event_type,
        payload=metrics.as_dict(),
        source="operations.sensory_metrics",
    )

    publish_event_with_failover(
        event_bus,
        event,
        logger=logger,
        runtime_fallback_message="Runtime bus rejected sensory metrics; falling back to global bus",
        runtime_unexpected_message="Unexpected error publishing sensory metrics via runtime bus",
        runtime_none_message="Runtime bus returned no result while publishing sensory metrics",
        global_not_running_message="Global event bus not running while publishing sensory metrics",
        global_unexpected_message="Unexpected error publishing sensory metrics via global bus",
        global_bus_factory=global_bus_factory,  # type: ignore[arg-type]
    )


def _extract_drift_alerts(summary: SensorySummary) -> tuple[str, ...]:
    payload = summary.drift_summary or {}
    exceeded = payload.get("exceeded") if isinstance(payload, Mapping) else None
    if not isinstance(exceeded, Iterable):
        return ()

    alerts: list[str] = []
    for entry in exceeded:
        if not isinstance(entry, Mapping):
            continue
        sensor = entry.get("sensor")
        if isinstance(sensor, str):
            alerts.append(sensor)
    return tuple(alerts)
