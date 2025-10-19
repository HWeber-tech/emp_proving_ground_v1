from __future__ import annotations

import json
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterator

import pytest

from src.runtime.paper_run_guardian import (
    MemoryTracker,
    PaperRunConfig,
    PaperRunMonitor,
    PaperRunStatus,
    persist_error_events,
    persist_summary,
)
from src.runtime.paper_simulation import (
    PaperTradingSimulationProgress,
    PaperTradingSimulationReport,
)


def _iterating_sampler(values: list[float]) -> MemoryTracker:
    iterator: Iterator[float] = iter(values)

    def _sample() -> float:
        try:
            return next(iterator)
        except StopIteration:
            return values[-1]

    return MemoryTracker(sampler=_sample)


def _progress(
    runtime_seconds: float,
    orders: int,
    *,
    p99_latency: float | None = None,
    avg_latency: float | None = None,
    last_error: dict[str, Any] | None = None,
) -> PaperTradingSimulationProgress:
    metrics: dict[str, Any] = {"latency_samples": orders}
    if p99_latency is not None:
        metrics["p99_latency_s"] = p99_latency
    if avg_latency is not None:
        metrics["avg_latency_s"] = avg_latency
    metrics.setdefault("last_latency_s", avg_latency)
    return PaperTradingSimulationProgress(
        timestamp=datetime.now(timezone.utc),
        runtime_seconds=runtime_seconds,
        orders_observed=orders,
        errors_observed=0,
        decisions_observed=orders,
        paper_metrics=metrics,
        last_error=last_error,
    )


def test_guardian_marks_latency_threshold_breach() -> None:
    config = PaperRunConfig(latency_p99_threshold=0.05)
    tracker = _iterating_sampler([100.0, 100.5])
    monitor = PaperRunMonitor(config, memory_tracker=tracker)

    monitor.record_progress(
        _progress(runtime_seconds=10.0, orders=5, p99_latency=0.08, avg_latency=0.06)
    )

    assert monitor.status is PaperRunStatus.DEGRADED
    assert any("Latency p99 exceeded" in alert for alert in monitor.alerts)
    assert monitor.should_stop is False


def test_guardian_detects_invariant_violation() -> None:
    config = PaperRunConfig()
    tracker = _iterating_sampler([100.0, 100.1])
    monitor = PaperRunMonitor(config, memory_tracker=tracker)

    invariant_error = {
        "stage": "risk_validation",
        "message": "Invariant breach: exposure",
    }

    monitor.record_progress(
        _progress(
            runtime_seconds=5.0,
            orders=1,
            p99_latency=0.01,
            avg_latency=0.01,
            last_error=invariant_error,
        )
    )

    assert monitor.status is PaperRunStatus.FAILED
    assert monitor.should_stop is True
    assert "risk-invariant-breach" in monitor.stop_reasons
    assert monitor.invariant_breaches


def test_guardian_summary_includes_metrics(tmp_path: Path) -> None:
    config = PaperRunConfig(memory_growth_threshold_mb=5.0)
    tracker = _iterating_sampler([100.0, 108.0, 108.0])
    monitor = PaperRunMonitor(config, memory_tracker=tracker)

    monitor.record_progress(
        _progress(runtime_seconds=1.0, orders=1, p99_latency=0.01, avg_latency=0.01)
    )
    monitor.record_progress(
        _progress(runtime_seconds=2.0, orders=2, p99_latency=0.02, avg_latency=0.02)
    )

    report = PaperTradingSimulationReport(
        orders=[{"order_id": "A"}],
        errors=[],
        decisions=3,
        diary_entries=2,
        runtime_seconds=120.0,
        paper_metrics={"p99_latency_s": 0.02},
    )

    summary = monitor.finalise(report)
    assert summary.status is PaperRunStatus.FAILED
    metrics = summary.metrics
    assert metrics["orders"] == 1
    assert metrics["memory_growth_mb"] == pytest.approx(8.0, rel=1e-6)

    destination_dir = tmp_path / "guardian"
    destination_dir.mkdir()
    destination = destination_dir / "summary.json"
    persist_summary(summary, destination)
    assert destination.exists()
    payload = destination.read_text(encoding="utf-8")
    assert "memory_growth" in payload


def test_guardian_enforces_minimum_runtime() -> None:
    config = PaperRunConfig(minimum_runtime_seconds=3600.0)
    tracker = _iterating_sampler([100.0, 100.0])
    monitor = PaperRunMonitor(config, memory_tracker=tracker)

    monitor.record_progress(
        _progress(runtime_seconds=10.0, orders=1, p99_latency=0.01, avg_latency=0.01)
    )

    report = PaperTradingSimulationReport(
        orders=[{"order_id": "A"}],
        errors=[],
        decisions=1,
        diary_entries=0,
        runtime_seconds=900.0,
    )

    summary = monitor.finalise(report)

    assert summary.status is PaperRunStatus.DEGRADED
    assert any("minimum duration" in alert.lower() for alert in summary.alerts)
    metrics = summary.metrics
    assert metrics["minimum_runtime_seconds"] == pytest.approx(3600.0)
    assert metrics["meets_minimum_runtime"] is False
    assert metrics["runtime_shortfall_seconds"] == pytest.approx(2700.0)


def test_guardian_requests_stop_on_memory_growth_threshold() -> None:
    config = PaperRunConfig(memory_growth_threshold_mb=4.0)
    tracker = _iterating_sampler([100.0, 105.5])
    monitor = PaperRunMonitor(config, memory_tracker=tracker)

    monitor.record_progress(
        _progress(runtime_seconds=10.0, orders=1, p99_latency=0.02, avg_latency=0.01)
    )
    monitor.record_progress(
        _progress(runtime_seconds=20.0, orders=2, p99_latency=0.02, avg_latency=0.01)
    )

    assert monitor.status is PaperRunStatus.FAILED
    assert monitor.should_stop is True
    assert "memory-growth-threshold-exceeded" in monitor.stop_reasons
    assert any("Memory growth exceeded threshold" in alert for alert in monitor.alerts)


def test_guardian_objective_compliance_flags() -> None:
    config = PaperRunConfig(
        minimum_runtime_seconds=10.0,
        latency_p99_threshold=0.05,
        memory_growth_threshold_mb=10.0,
    )
    tracker = _iterating_sampler([100.0, 103.0, 104.0])
    monitor = PaperRunMonitor(config, memory_tracker=tracker)

    monitor.record_progress(
        _progress(runtime_seconds=5.0, orders=5, p99_latency=0.03, avg_latency=0.02)
    )
    monitor.record_progress(
        _progress(runtime_seconds=15.0, orders=10, p99_latency=0.04, avg_latency=0.03)
    )

    report = PaperTradingSimulationReport(
        orders=[{"order_id": "A"}, {"order_id": "B"}],
        errors=[],
        decisions=20,
        diary_entries=4,
        runtime_seconds=600.0,
    )

    summary = monitor.finalise(report)
    metrics = summary.metrics

    assert metrics["objectives_met"] is True
    objectives = metrics["objective_compliance"]
    assert objectives["minimum_runtime_met"] is True
    assert objectives["latency_p99_within_threshold"] is True
    assert objectives["memory_growth_within_threshold"] is True
    assert objectives["no_invariant_breaches"] is True
    assert metrics["invariant_breach_count"] == 0
    assert metrics["memory_growth_within_threshold"] is True


def test_persist_error_events_writes_json(tmp_path: Path) -> None:
    config = PaperRunConfig()
    tracker = _iterating_sampler([100.0])
    monitor = PaperRunMonitor(config, memory_tracker=tracker)

    error_payload = {
        "stage": "risk_validation",
        "message": "Invariant breach detected",
        "exception": "InvariantError",
    }

    monitor.record_progress(
        _progress(
            runtime_seconds=5.0,
            orders=1,
            p99_latency=0.02,
            avg_latency=0.01,
            last_error=error_payload,
        )
    )

    report = PaperTradingSimulationReport(
        orders=[],
        errors=[error_payload],
        decisions=1,
        diary_entries=0,
        runtime_seconds=10.0,
    )

    summary = monitor.finalise(report)

    destination = tmp_path / "errors.json"
    persist_error_events(summary, destination)

    content = json.loads(destination.read_text(encoding="utf-8"))
    assert isinstance(content, list)
    assert content and content[0]["stage"] == "risk_validation"
