"""Tests for the execution performance report renderer."""

from __future__ import annotations

from datetime import datetime, timezone

from src.trading.execution.performance_report import build_execution_performance_report


def test_build_execution_performance_report_includes_throttle_and_throughput() -> None:
    stats = {
        "orders_submitted": 5,
        "orders_executed": 3,
        "orders_failed": 1,
        "trade_throttle": {
            "name": "rate_limit",
            "state": "rate_limited",
            "reason": "max_1_trades_per_60s",
            "message": "Throttled: too many trades in short time (limit 1 trade per 1 minute)",
            "metadata": {
                "retry_at": datetime(2024, 1, 1, 12, 1, tzinfo=timezone.utc).isoformat(),
                "context": {"symbol": "EURUSD", "strategy_id": "alpha"},
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
    }

    report = build_execution_performance_report(stats)

    assert "Execution performance summary" in report
    assert "Submitted: 5" in report
    assert "Control: `rate_limit`" in report
    assert "max_1_trades_per_60s" in report
    assert "Throttled: too many trades in short time" in report
    assert "retry_at" not in report.lower()  # Should keep original casing
    assert "Context: strategy_id=alpha, symbol=EURUSD" in report
    assert "Throughput (per min)" in report
    assert "96.00" in report


def test_build_execution_performance_report_handles_missing_sections() -> None:
    report = build_execution_performance_report({})

    assert "Submitted: —" in report
    assert "Throughput window" not in report
