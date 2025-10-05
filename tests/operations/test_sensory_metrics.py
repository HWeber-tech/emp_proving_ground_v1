from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from src.operations.sensory_metrics import (
    DimensionMetric,
    build_sensory_metrics_from_status,
    publish_sensory_metrics,
)


def _sample_status() -> dict[str, Any]:
    generated_at = datetime.now(timezone.utc) - timedelta(minutes=5)
    return {
        "samples": 8,
        "latest": {
            "symbol": "EURUSD",
            "generated_at": generated_at.isoformat(),
            "integrated_signal": {
                "strength": 0.35,
                "confidence": 0.62,
                "direction": 1.0,
                "contributing": ["WHY", "HOW", "ANOMALY"],
            },
            "dimensions": {
                "WHY": {
                    "signal": 0.38,
                    "confidence": 0.70,
                    "metadata": {"state": "bullish", "threshold_assessment": {"state": "warn"}},
                },
                "HOW": {
                    "signal": 0.22,
                    "confidence": 0.55,
                    "metadata": {"state": "nominal", "threshold_assessment": {"state": "nominal"}},
                },
                "ANOMALY": {
                    "signal": -0.6,
                    "confidence": 0.65,
                    "metadata": {"state": "alert", "threshold_assessment": {"state": "alert"}},
                },
            },
        },
        "drift_summary": {
            "exceeded": [
                {"sensor": "ANOMALY", "z_score": 3.1},
            ]
        },
    }


class _StubEventBus:
    def __init__(self, *, running: bool = True, raise_runtime: bool = False) -> None:
        self.running = running
        self.raise_runtime = raise_runtime
        self.events: list[Any] = []

    def is_running(self) -> bool:
        return self.running

    def publish_from_sync(self, event: Any) -> int:
        if self.raise_runtime:
            raise RuntimeError("runtime bus failure")
        self.events.append(event)
        return 1


class _StubTopicBus:
    def __init__(self) -> None:
        self.events: list[tuple[str, Any, str]] = []

    def publish_sync(self, event_type: str, payload: Any, source: str) -> None:
        self.events.append((event_type, payload, source))


def test_build_sensory_metrics_from_status_focuses_on_dimensions() -> None:
    metrics = build_sensory_metrics_from_status(_sample_status())

    assert metrics.symbol == "EURUSD"
    assert metrics.integrated_strength == pytest.approx(0.35)
    assert metrics.integrated_confidence == pytest.approx(0.62)

    dimensions = metrics.dimensions
    assert len(dimensions) == 3
    assert isinstance(dimensions[0], DimensionMetric)

    how_metric = {metric.name: metric for metric in metrics.dimensions}["HOW"]
    assert how_metric.state == "nominal"
    assert how_metric.threshold_state == "nominal"

    assert metrics.drift_alerts == ("ANOMALY",)


def test_publish_sensory_metrics_uses_event_bus() -> None:
    metrics = build_sensory_metrics_from_status(_sample_status())
    bus = _StubEventBus()

    publish_sensory_metrics(metrics, event_bus=bus)

    assert len(bus.events) == 1
    event = bus.events[0]
    assert event.type == "telemetry.sensory.metrics"
    assert event.payload["symbol"] == "EURUSD"


def test_publish_sensory_metrics_falls_back_to_global_bus() -> None:
    metrics = build_sensory_metrics_from_status(_sample_status())
    bus = _StubEventBus(raise_runtime=True)
    topic_bus = _StubTopicBus()

    publish_sensory_metrics(metrics, event_bus=bus, global_bus_factory=lambda: topic_bus)

    assert len(bus.events) == 0
    assert len(topic_bus.events) == 1
    event_type, payload, source = topic_bus.events[0]
    assert event_type == "telemetry.sensory.metrics"
    assert payload["symbol"] == "EURUSD"
    assert source == "operations.sensory_metrics"
