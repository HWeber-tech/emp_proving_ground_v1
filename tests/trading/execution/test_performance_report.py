"""Tests for the execution performance report renderer."""

from __future__ import annotations

from datetime import datetime, timezone

from src.trading.execution.performance_report import (
    build_execution_performance_report,
    build_performance_health_report,
)


def test_build_execution_performance_report_includes_throttle_and_throughput() -> None:
    stats = {
        "orders_submitted": 5,
        "orders_executed": 3,
        "orders_failed": 1,
        "avg_latency_ms": 14.5,
        "latency_p99_ms": 22.0,
        "max_latency_ms": 25.0,
        "trade_throttle": {
            "name": "rate_limit",
            "state": "rate_limited",
            "reason": "max_1_trades_per_60s",
            "message": "Throttled: too many trades in short time (limit 1 trade per 1 minute)",
            "scope_key": ["str:'alpha'"],
            "metadata": {
                "retry_at": datetime(2024, 1, 1, 12, 1, tzinfo=timezone.utc).isoformat(),
                "context": {"symbol": "EURUSD", "strategy_id": "alpha"},
                "scope": {"strategy_id": "alpha"},
                "max_trades": 1,
                "remaining_trades": 0,
                "window_utilisation": 1.0,
                "retry_in_seconds": 45.0,
                "window_reset_in_seconds": 55.0,
                "window_reset_at": datetime(2024, 1, 1, 12, 1, 5, tzinfo=timezone.utc).isoformat(),
            },
        },
        "throughput": {
            "samples": 8,
            "avg_processing_ms": 12.5,
            "p95_processing_ms": 18.0,
            "max_processing_ms": 19.0,
            "avg_lag_ms": 2.0,
            "max_lag_ms": 4.0,
            "throughput_per_min": 96.0,
        },
        "backlog": {
            "samples": 5,
            "threshold_ms": 150.0,
            "max_lag_ms": 80.0,
            "avg_lag_ms": 25.0,
            "latest_lag_ms": 30.0,
            "p95_lag_ms": 70.0,
            "breaches": 1,
            "breach_rate": 0.2,
            "max_breach_streak": 2,
            "worst_breach_ms": 120.0,
            "last_breach_at": datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc).isoformat(),
            "healthy": True,
        },
        "resource_usage": {
            "timestamp": datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc).isoformat(),
            "cpu_percent": 35.0,
            "memory_mb": 512.0,
            "memory_percent": 42.0,
            "io_read_mb": 1.5,
            "io_write_mb": 0.75,
            "io_read_count": 12,
            "io_write_count": 3,
        },
    }

    report = build_execution_performance_report(stats)

    assert "Execution performance summary" in report
    assert "Submitted: 5" in report
    assert "Control: `rate_limit`" in report
    assert "max_1_trades_per_60s" in report
    assert "Throttled: too many trades in short time" in report
    assert "retry_at" not in report.lower()  # Should keep original casing
    assert "Context: strategy_id=alpha, symbol=EURUSD" in report
    assert "Capacity: 0 / 1 trades remaining" in report
    assert "100.0% utilised" in report
    assert "Retry in: 45s" in report
    assert "Window resets in: 55s" in report
    assert "- Scope:" in report
    assert "Scope key:" in report
    assert "Throughput (per min)" in report
    assert "96.00" in report
    assert "P99 order latency (ms)" in report
    assert "Event backlog" in report
    assert "Threshold (ms): 150.00" in report
    assert "Latest lag (ms): 30.00" in report
    assert "P95 lag (ms): 70.00" in report
    assert "Breach rate: 20.0%" in report
    assert "Max breach streak: 2" in report
    assert "Worst breach (ms): 120.00" in report
    assert "Resource usage snapshot" in report
    assert "CPU (%): 35.00" in report
    assert "Memory (MB): 512.00" in report
    assert "IO read (MB): 1.50" in report
    assert "IO write (MB): 0.75" in report
    assert "IO read ops: 12" in report
    assert "IO write ops: 3" in report


def test_build_execution_performance_report_handles_missing_sections() -> None:
    report = build_execution_performance_report({})

    assert "Submitted: —" in report
    assert "Throughput window" not in report
    assert "Event backlog" not in report
    assert "Resource usage snapshot" not in report


def test_build_performance_health_report_includes_sections() -> None:
    assessment = {
        "healthy": True,
        "throughput": {
            "healthy": True,
            "max_processing_ms": 120.0,
            "max_lag_ms": 40.0,
            "samples": 8,
        },
        "backlog": {
            "evaluated": True,
            "healthy": True,
            "threshold_ms": 180.0,
            "max_lag_ms": 75.0,
            "p95_lag_ms": 65.0,
            "samples": 6,
            "breach_rate": 0.1,
            "max_breach_streak": 1,
        },
        "resource": {
            "healthy": True,
            "status": "ok",
            "limits": {"max_cpu_percent": 75.0},
            "sample": {
                "timestamp": "2024-07-01T12:00:00+00:00",
                "cpu_percent": 35.0,
            },
            "violations": {},
        },
        "throttle": {
            "active": False,
            "reason": None,
            "message": None,
            "context": {"strategy_id": "alpha"},
            "scope": {"strategy_id": "alpha"},
            "scope_key": ["str:'alpha'"],
            "remaining_trades": 2,
            "max_trades": 5,
            "window_utilisation": 0.6,
            "retry_in_seconds": 12.5,
            "window_reset_in_seconds": 40.0,
            "window_reset_at": "2024-01-01T12:00:40+00:00",
        },
    }

    report = build_performance_health_report(assessment)

    assert "# Performance health assessment" in report
    assert "Overall status: Healthy" in report
    assert "## Throughput" in report
    assert "## Backlog" in report
    assert "P95 lag (ms): 65.00" in report
    assert "Breach rate: 10.0%" in report
    assert "Max breach streak: 1" in report
    assert "## Resource utilisation" in report
    assert "## Trade throttle" in report
    assert "Remaining trades: 2 / 5" in report
    assert "Window utilisation: 60.0%" in report
    assert "Retry in: 12.50s" in report
    assert "Scope key:" in report


def test_build_performance_health_report_handles_missing_data() -> None:
    report = build_performance_health_report({})

    assert "No throughput data" in report
    assert "No backlog data" in report
    assert "No resource data" in report
