from __future__ import annotations

from datetime import datetime

try:  # Python < 3.11 compatibility
    from datetime import UTC
except ImportError:  # pragma: no cover - fallback
    from datetime import timezone

    UTC = timezone.utc  # type: ignore[misc]

import pytest

from src.operations.gate_dashboard import (
    GateDashboardStatus,
    GateMetricDirection,
    GateMetricVisual,
    build_gate_dashboard,
    build_gate_dashboard_from_mapping,
)


def test_gate_metric_status_at_least() -> None:
    metric = GateMetricVisual(
        name="success_rate",
        value=0.92,
        warn_threshold=0.95,
        fail_threshold=0.9,
        direction=GateMetricDirection.AT_LEAST,
        multiplier=100.0,
        unit="%",
    )

    assert metric.status() is GateDashboardStatus.WARN
    assert metric.value_text == "92.0%"

    failing_metric = GateMetricVisual(
        name="success_rate",
        value=0.82,
        warn_threshold=0.95,
        fail_threshold=0.9,
        direction=GateMetricDirection.AT_LEAST,
        multiplier=100.0,
        unit="%",
    )

    assert failing_metric.status() is GateDashboardStatus.FAIL


def test_gate_metric_status_at_most_and_gauge() -> None:
    metric = GateMetricVisual(
        name="mttr",
        value=140.0,
        warn_threshold=120.0,
        fail_threshold=180.0,
        direction=GateMetricDirection.AT_MOST,
        unit="m",
    )

    assert metric.status() is GateDashboardStatus.WARN
    gauge = metric.gauge(width=14)
    assert gauge.startswith("[") and gauge.endswith("]")
    assert len(gauge) == 14

    healthy = GateMetricVisual(
        name="mttr",
        value=90.0,
        warn_threshold=120.0,
        fail_threshold=180.0,
        direction=GateMetricDirection.AT_MOST,
        unit="m",
    )
    assert healthy.status() is GateDashboardStatus.OK


def test_build_gate_dashboard_and_markdown() -> None:
    moment = datetime(2025, 1, 9, 12, 0, tzinfo=UTC)
    dashboard = build_gate_dashboard(
        [
            {
                "name": "success_rate",
                "value": 0.97,
                "warn": 0.95,
                "fail": 0.9,
                "direction": "at_least",
                "unit": "%",
                "multiplier": 100.0,
            },
            {
                "name": "mtta_minutes",
                "value": 48.0,
                "warn": 60.0,
                "fail": 90.0,
                "direction": "at_most",
                "unit": "m",
            },
        ],
        generated_at=moment,
        metadata={"source": "unit-test"},
    )

    assert dashboard.status() is GateDashboardStatus.OK
    payload = dashboard.as_dict()
    assert payload["status"] == "ok"
    assert payload["generated_at"].startswith("2025-01-09T12:00:00")
    assert payload["metadata"]["source"] == "unit-test"
    assert payload["metrics"][0]["name"] == "success_rate"
    markdown = dashboard.to_markdown()
    assert "| Metric |" in markdown
    assert "success_rate" in markdown
    assert "mtta_minutes" in markdown
    details = dashboard.panel_details()
    assert len(details) == 2
    assert "WARN" not in details[0]


def test_build_gate_dashboard_from_mapping_validates_metrics() -> None:
    payload = {
        "generated_at": "2025-01-10T05:30:00+00:00",
        "metrics": [
            {
                "name": "latency",
                "value": 95.0,
                "warn_threshold": 110.0,
                "fail_threshold": 130.0,
                "direction": "at_most",
                "unit": "ms",
            }
        ],
    }

    dashboard = build_gate_dashboard_from_mapping(payload)
    assert dashboard.status() is GateDashboardStatus.OK
    assert dashboard.metrics[0].status() is GateDashboardStatus.OK
    assert dashboard.metrics[0].fail_text.startswith("<=")


def test_build_gate_dashboard_rejects_missing_name() -> None:
    with pytest.raises(ValueError):
        build_gate_dashboard([{"value": 1.0}])
