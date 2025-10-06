from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.operational import metrics as operational_metrics
from src.operations.slo import (
    SLOStatus,
    UnderstandingLoopSLOInputs,
    evaluate_understanding_loop_slos,
)


pytestmark = pytest.mark.guardrail


def _capture_metric(name: str, calls: list[tuple[str, object]]) -> callable:
    def _recorder(value: object) -> None:
        calls.append((name, value))

    return _recorder


def test_evaluate_understanding_loop_slos_success(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, object]] = []
    monkeypatch.setattr(operational_metrics, "set_understanding_loop_latency", _capture_metric("latency", calls))
    monkeypatch.setattr(
        operational_metrics,
        "set_understanding_loop_latency_status",
        _capture_metric("latency_status", calls),
    )
    monkeypatch.setattr(
        operational_metrics,
        "set_understanding_loop_drift_freshness",
        _capture_metric("drift", calls),
    )
    monkeypatch.setattr(
        operational_metrics,
        "set_understanding_loop_drift_status",
        _capture_metric("drift_status", calls),
    )
    monkeypatch.setattr(
        operational_metrics,
        "set_understanding_loop_replay_determinism",
        _capture_metric("replay", calls),
    )
    monkeypatch.setattr(
        operational_metrics,
        "set_understanding_loop_replay_status",
        _capture_metric("replay_status", calls),
    )

    inputs = UnderstandingLoopSLOInputs(
        latency_seconds=0.52,
        latency_samples=128,
        drift_alert_age_seconds=180.0,
        drift_alert_count=3,
        replay_determinism_ratio=0.998,
        replay_trials=42,
        generated_at=datetime(2024, 5, 1, 12, 0, tzinfo=timezone.utc),
        metadata={"loop": "alpha"},
        alert_routes={"understanding_loop": "pagerduty:loop"},
    )

    snapshot = evaluate_understanding_loop_slos(inputs)

    assert snapshot.status is SLOStatus.met
    assert snapshot.service == "understanding_loop"
    assert snapshot.metadata["latency_status"] == SLOStatus.met.value
    assert snapshot.metadata["loop"] == "alpha"

    summary, latency_slo, drift_slo, replay_slo = snapshot.slos
    assert summary.name == "understanding_loop"
    assert summary.status is SLOStatus.met
    assert summary.alert_route == "pagerduty:loop"
    assert latency_slo.status is SLOStatus.met
    assert latency_slo.observed["latency_seconds"] == pytest.approx(0.52)
    assert latency_slo.target["p95_seconds"] == pytest.approx(inputs.latency_target_seconds)
    assert drift_slo.status is SLOStatus.met
    assert drift_slo.observed["age_seconds"] == pytest.approx(180.0)
    assert replay_slo.status is SLOStatus.met
    assert replay_slo.observed["determinism_ratio"] == pytest.approx(0.998)

    assert calls == [
        ("latency", 0.52),
        ("latency_status", SLOStatus.met.value),
        ("drift", 180.0),
        ("drift_status", SLOStatus.met.value),
        ("replay", 0.998),
        ("replay_status", SLOStatus.met.value),
    ]


def test_evaluate_understanding_loop_slos_handles_degradation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, object]] = []
    monkeypatch.setattr(operational_metrics, "set_understanding_loop_latency", _capture_metric("latency", calls))
    monkeypatch.setattr(
        operational_metrics,
        "set_understanding_loop_latency_status",
        _capture_metric("latency_status", calls),
    )
    monkeypatch.setattr(
        operational_metrics,
        "set_understanding_loop_drift_freshness",
        _capture_metric("drift", calls),
    )
    monkeypatch.setattr(
        operational_metrics,
        "set_understanding_loop_drift_status",
        _capture_metric("drift_status", calls),
    )
    monkeypatch.setattr(
        operational_metrics,
        "set_understanding_loop_replay_determinism",
        _capture_metric("replay", calls),
    )
    monkeypatch.setattr(
        operational_metrics,
        "set_understanding_loop_replay_status",
        _capture_metric("replay_status", calls),
    )

    inputs = UnderstandingLoopSLOInputs(
        latency_seconds=1.8,
        latency_target_seconds=0.9,
        drift_alert_age_seconds=1500.0,
        drift_freshness_target_seconds=400.0,
        replay_determinism_ratio=0.91,
        replay_target_ratio=0.98,
        metadata={"environment": "paper"},
    )

    snapshot = evaluate_understanding_loop_slos(inputs)

    assert snapshot.status is SLOStatus.breached
    summary, latency_slo, drift_slo, replay_slo = snapshot.slos
    assert summary.status is SLOStatus.breached
    assert "breached" in summary.message
    assert latency_slo.status is SLOStatus.breached
    assert drift_slo.status is SLOStatus.breached
    assert replay_slo.status is SLOStatus.at_risk

    assert snapshot.metadata["environment"] == "paper"
    assert snapshot.metadata["drift_status"] == SLOStatus.breached.value

    assert calls == [
        ("latency", 1.8),
        ("latency_status", SLOStatus.breached.value),
        ("drift", 1500.0),
        ("drift_status", SLOStatus.breached.value),
        ("replay", 0.91),
        ("replay_status", SLOStatus.at_risk.value),
    ]
