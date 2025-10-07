from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest

from src.operational import metrics as operational_metrics
from src.operations.observability_diary import ThrottleStateSnapshot
from src.understanding.diagnostics import UnderstandingDiagnosticsBuilder
from src.understanding.metrics import (
    export_throttle_metrics,
    export_understanding_throttle_metrics,
)

try:  # Python 3.10 compatibility
    from datetime import UTC
except ImportError:  # pragma: no cover - fallback path for older runtimes
    from datetime import timezone

    UTC = timezone.utc


class _RecordingGauge:
    def __init__(self, name: str) -> None:
        self.name = name
        self.labels_calls: List[Dict[str, str]] = []
        self.set_calls: List[float] = []

    def labels(self, **labels: str) -> "_RecordingGauge":
        self.labels_calls.append(dict(labels))
        return self

    def set(self, value: float) -> None:
        self.set_calls.append(float(value))


class _RecordingRegistry:
    def __init__(self) -> None:
        self.gauge_requests: List[Tuple[str, Tuple[str, ...] | None]] = []
        self.gauges: List[_RecordingGauge] = []

    def get_gauge(
        self, name: str, description: str, labelnames: List[str] | None = None
    ) -> _RecordingGauge:
        entry = (name, tuple(labelnames) if labelnames else None)
        self.gauge_requests.append(entry)
        gauge = _RecordingGauge(name)
        self.gauges.append(gauge)
        return gauge


@pytest.fixture
def _registry(monkeypatch: pytest.MonkeyPatch) -> _RecordingRegistry:
    registry = _RecordingRegistry()
    monkeypatch.setattr(operational_metrics, "get_registry", lambda: registry)
    return registry


def test_export_throttle_metrics_uses_replay_fixture(
    _registry: _RecordingRegistry,
) -> None:
    fixture_path = Path(__file__).with_name("fixtures") / "throttle_prometheus_replay.json"
    payload = json.loads(fixture_path.read_text())
    regime = payload.get("regime")
    decision = payload.get("decision")
    states = tuple(
        ThrottleStateSnapshot(
            name=entry["name"],
            state=entry["state"],
            active=bool(entry["active"]),
            multiplier=float(entry["multiplier"]),
            reason=entry.get("reason"),
            metadata={},
        )
        for entry in payload["throttle_states"]
    )

    export_throttle_metrics(states, regime=regime, decision_id=decision)

    assert _registry.gauge_requests == [
        ("understanding_throttle_active", ("throttle", "state", "regime", "decision")),
        ("understanding_throttle_multiplier", ("throttle", "state", "regime", "decision")),
        ("understanding_throttle_active", ("throttle", "state", "regime", "decision")),
        ("understanding_throttle_multiplier", ("throttle", "state", "regime", "decision")),
    ]

    first_active, first_multiplier, second_active, second_multiplier = _registry.gauges

    assert first_active.labels_calls[0]["throttle"] == "volatility"
    assert first_active.set_calls == [1.0]
    assert first_multiplier.set_calls == [pytest.approx(0.7)]

    assert second_active.labels_calls[0]["throttle"] == "drawdown"
    assert second_active.set_calls == [0.0]
    assert second_multiplier.set_calls == [pytest.approx(1.0)]


def test_export_understanding_throttle_metrics_from_snapshot(
    _registry: _RecordingRegistry,
) -> None:
    builder = UnderstandingDiagnosticsBuilder(
        now=lambda: datetime(2024, 6, 1, tzinfo=UTC)
    )
    snapshot = builder.build().to_snapshot()

    export_understanding_throttle_metrics(snapshot)

    assert _registry.gauge_requests[0][0] == "understanding_throttle_active"
    active_gauge = _registry.gauges[0]
    multiplier_gauge = _registry.gauges[1]

    assert active_gauge.labels_calls == [
        {
            "throttle": "drift_sentry",
            "state": "observing",
            "regime": snapshot.regime_state.regime,
            "decision": snapshot.decision.tactic_id,
        }
    ]
    assert active_gauge.set_calls == [0.0]

    assert multiplier_gauge.set_calls == [pytest.approx(1.0)]
