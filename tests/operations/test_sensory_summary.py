from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from src.operations.sensory_summary import (
    build_sensory_summary,
    publish_sensory_summary,
)


def _sample_status() -> dict[str, Any]:
    generated_at = datetime.now(timezone.utc) - timedelta(minutes=1)
    return {
        "samples": 12,
        "latest": {
            "symbol": "EURUSD",
            "generated_at": generated_at.isoformat(),
            "integrated_signal": {
                "strength": 0.42,
                "confidence": 0.68,
                "direction": 1.0,
                "contributing": ["WHY", "HOW", "ANOMALY"],
            },
            "dimensions": {
                "WHY": {
                    "signal": 0.45,
                    "confidence": 0.70,
                    "metadata": {
                        "state": "bullish",
                        "threshold_assessment": {"state": "warn"},
                    },
                },
                "HOW": {
                    "signal": 0.15,
                    "confidence": 0.55,
                    "metadata": {
                        "state": "nominal",
                        "threshold_assessment": {"state": "nominal"},
                    },
                },
                "ANOMALY": {
                    "signal": -0.6,
                    "confidence": 0.65,
                    "metadata": {
                        "state": "alert",
                        "threshold_assessment": {"state": "alert"},
                    },
                },
            },
        },
        "sensor_audit": [
            {
                "generated_at": generated_at.isoformat(),
                "dimensions": {
                    "WHY": {"signal": 0.45, "confidence": 0.70},
                    "ANOMALY": {"signal": -0.6, "confidence": 0.65},
                },
            }
        ],
        "drift_summary": {
            "parameters": {
                "baseline_window": 5,
                "evaluation_window": 3,
                "min_observations": 2,
                "z_threshold": 2.5,
            },
            "results": [
                {
                    "sensor": "WHY",
                    "baseline_mean": 0.3,
                    "baseline_std": 0.1,
                    "baseline_count": 5,
                    "evaluation_mean": 0.45,
                    "evaluation_std": 0.05,
                    "evaluation_count": 3,
                    "z_score": 3.2,
                    "drift_ratio": 0.5,
                    "exceeded": True,
                }
            ],
            "exceeded": [
                {
                    "sensor": "WHY",
                    "z_score": 3.2,
                }
            ],
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


def test_build_sensory_summary_extracts_metrics() -> None:
    summary = build_sensory_summary(_sample_status())

    assert summary.symbol == "EURUSD"
    assert summary.integrated_strength == pytest.approx(0.42)
    assert summary.integrated_confidence == pytest.approx(0.68)
    assert summary.samples == 12
    assert len(summary.dimensions) == 3

    names = [dimension.name for dimension in summary.dimensions]
    assert set(names) == {"WHY", "HOW", "ANOMALY"}

    top_dimension = summary.top_dimensions(1)[0]
    assert top_dimension.name in {"WHY", "ANOMALY"}
    assert top_dimension.threshold_state in {"warn", "alert"}

    markdown = summary.to_markdown()
    assert "Dimension" in markdown
    assert "Drift alerts" in markdown


def test_build_sensory_summary_handles_unparseable_timestamp(caplog: pytest.LogCaptureFixture) -> None:
    status = _sample_status()
    status["latest"]["generated_at"] = {"bad": "payload"}

    with caplog.at_level("DEBUG"):
        summary = build_sensory_summary(status)

    assert summary.generated_at is None
    assert any("Failed to parse sensory timestamp" in record.getMessage() for record in caplog.records)


def test_build_sensory_summary_converts_naive_timestamp_to_utc() -> None:
    status = _sample_status()
    status["latest"]["generated_at"] = "2024-01-01T12:30:45"

    summary = build_sensory_summary(status)

    assert summary.generated_at is not None
    assert summary.generated_at.tzinfo == timezone.utc


def test_publish_sensory_summary_uses_runtime_bus() -> None:
    summary = build_sensory_summary(_sample_status())
    bus = _StubEventBus()

    publish_sensory_summary(summary, event_bus=bus)

    assert len(bus.events) == 1
    event = bus.events[0]
    assert event.type == "telemetry.sensory.summary"
    assert "markdown" in event.payload


def test_publish_sensory_summary_falls_back_to_global_bus() -> None:
    summary = build_sensory_summary(_sample_status())
    bus = _StubEventBus(raise_runtime=True)
    topic_bus = _StubTopicBus()

    publish_sensory_summary(summary, event_bus=bus, global_bus_factory=lambda: topic_bus)

    assert len(bus.events) == 0
    assert len(topic_bus.events) == 1
    event_type, payload, source = topic_bus.events[0]
    assert event_type == "telemetry.sensory.summary"
    assert payload["symbol"] == "EURUSD"
    assert source == "operations.sensory_summary"
