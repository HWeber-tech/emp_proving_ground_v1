from __future__ import annotations

import pytest

from src.orchestration.decision_latency_guard import (
    DecisionLatencyBaseline,
    evaluate_decision_latency,
)


def test_evaluate_decision_latency_pass() -> None:
    metrics = {"p50": 0.08, "p99": 0.15, "samples": 25}
    result = evaluate_decision_latency(metrics)
    assert result["status"] == "pass"
    assert result["current"]["p50_s"] == pytest.approx(0.08)
    assert result["thresholds"]["p50_s"] >= result["baseline"]["p50_s"]


def test_evaluate_decision_latency_warn_on_insufficient_samples() -> None:
    metrics = {"p50": 0.08, "p99": 0.15, "samples": 5}
    result = evaluate_decision_latency(metrics, min_samples=10)
    assert result["status"] == "warn"
    assert result["reason"] == "insufficient_samples"


def test_evaluate_decision_latency_fail_when_threshold_exceeded() -> None:
    baseline = DecisionLatencyBaseline(p50_s=0.05, p99_s=0.1, samples=20)
    metrics = {"p50": 0.06, "p99": 0.12, "samples": 20}
    result = evaluate_decision_latency(metrics, baseline=baseline, tolerance_pct=0.05)
    assert result["status"] == "fail"
    assert result["reason"] == "threshold_exceeded"
    assert set(result["failures"]) == {"p50", "p99"}


def test_evaluate_decision_latency_no_data() -> None:
    result = evaluate_decision_latency(None)
    assert result["status"] == "no_data"
    assert result["reason"] == "missing_metrics"
