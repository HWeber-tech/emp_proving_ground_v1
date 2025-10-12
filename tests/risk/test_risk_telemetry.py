from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pytest

from src.risk import telemetry as telemetry_module
from src.risk.telemetry import (
    RiskLimitStatus,
    RiskLimitCheck,
    RiskTelemetrySnapshot,
    RiskThresholdType,
    evaluate_risk_posture,
    format_risk_markdown,
    publish_risk_snapshot,
)


class StubBus:
    def __init__(self) -> None:
        self.events: list[Any] = []

    async def publish(self, event: Any) -> None:
        self.events.append(event)


def _sample_state() -> dict[str, Any]:
    return {
        "open_positions_count": 4,
        "current_daily_drawdown": 0.11,
        "equity": 125_000.0,
        "peak_equity": 150_000.0,
        "open_positions": {
            "EURUSD": {"quantity": 1.5, "current_value": 18_500.0},
            "USDJPY": {"quantity": -2.0, "current_value": -15_000.0},
        },
    }


def _sample_limits() -> dict[str, Any]:
    return {
        "limits": {"max_open_positions": 5, "max_daily_drawdown": 0.1},
        "telemetry": {"total_checks": 10, "approved": 7, "rejected": 3},
    }


def _sample_decision() -> dict[str, Any]:
    return {
        "status": "rejected",
        "reason": "insufficient_liquidity",
        "symbol": "EURUSD",
        "checks": [
            {"name": "liquidity_confidence", "value": 0.2, "threshold": 0.3},
            {"name": "position_sizer", "recommended": 1.1, "requested": 2.0},
        ],
    }


def test_evaluate_risk_posture_grades_limits() -> None:
    snapshot = evaluate_risk_posture(
        _sample_state(),
        _sample_limits(),
        last_decision=_sample_decision(),
    )

    assert snapshot.status is RiskLimitStatus.alert
    assert pytest.approx(snapshot.approval_rate or 0.0, rel=1e-6) == 0.7
    exposures = snapshot.exposures
    assert exposures["long"] == pytest.approx(18_500.0)
    assert exposures["short"] == pytest.approx(15_000.0)
    assert snapshot.generated_at.tzinfo is timezone.utc

    check_map = {check.name: check for check in snapshot.checks}
    assert check_map["open_positions"].status is RiskLimitStatus.warn
    assert check_map["daily_drawdown"].status is RiskLimitStatus.alert
    assert check_map["decision.liquidity_confidence"].status is RiskLimitStatus.alert

    assert snapshot.last_decision is not None
    assert snapshot.last_decision["reason"] == "insufficient_liquidity"


def test_risk_limit_check_as_dict_includes_optionals() -> None:
    check = RiskLimitCheck(
        name="test",
        value=1.23,
        threshold=4.56,
        threshold_type=RiskThresholdType.maximum,
        status=RiskLimitStatus.warn,
        ratio=0.27,
        metadata={"notes": "sample"},
    )

    payload = check.as_dict()

    assert payload["ratio"] == pytest.approx(0.27)
    assert payload["metadata"] == {"notes": "sample"}


def test_risk_telemetry_snapshot_as_dict_skips_nulls() -> None:
    snapshot = RiskTelemetrySnapshot(
        status=RiskLimitStatus.ok,
        generated_at=datetime.now(timezone.utc),
        checks=(
            RiskLimitCheck(
                name="dummy",
                value=None,
                threshold=None,
                threshold_type=RiskThresholdType.maximum,
                status=RiskLimitStatus.ok,
            ),
        ),
        exposures={"long": 0.0},
        limits={},
        telemetry={},
        approval_rate=None,
        portfolio_equity=None,
        peak_equity=None,
        last_decision=None,
    )

    payload = snapshot.as_dict()

    assert "approval_rate" not in payload
    assert payload["checks"][0]["name"] == "dummy"


def test_threshold_helpers_cover_minimum_and_maximum_paths() -> None:
    assert telemetry_module._infer_threshold_type("confidence") is RiskThresholdType.minimum
    assert telemetry_module._infer_threshold_type("max_positions") is RiskThresholdType.maximum

    status, ratio = telemetry_module._grade_limit(5, 10, threshold_type=RiskThresholdType.maximum)
    assert status is RiskLimitStatus.ok and ratio == pytest.approx(0.5)

    status, ratio = telemetry_module._grade_limit(10, 10, threshold_type=RiskThresholdType.maximum)
    assert status is RiskLimitStatus.alert and ratio == pytest.approx(1.0)

    status, ratio = telemetry_module._grade_limit(None, 10, threshold_type=RiskThresholdType.minimum)
    assert status is RiskLimitStatus.warn and ratio is None

    status, ratio = telemetry_module._grade_limit(4, 5, threshold_type=RiskThresholdType.minimum)
    assert status is RiskLimitStatus.warn and ratio == pytest.approx(0.8)

    status, ratio = telemetry_module._grade_limit(0.0, 0.0, threshold_type=RiskThresholdType.maximum)
    assert status is RiskLimitStatus.ok and ratio is None


def test_format_risk_markdown_includes_status() -> None:
    snapshot = evaluate_risk_posture(
        _sample_state(),
        _sample_limits(),
        last_decision=_sample_decision(),
    )
    markdown = format_risk_markdown(snapshot)
    assert "STATUS" in markdown.upper()
    assert "OPEN_POSITIONS" in markdown.upper()
    assert "LAST DECISION" in markdown.upper()


def test_format_risk_markdown_handles_sparse_snapshot() -> None:
    snapshot = RiskTelemetrySnapshot(
        status=RiskLimitStatus.ok,
        generated_at=datetime.now(timezone.utc),
        checks=(
            RiskLimitCheck(
                name="latency",
                value=None,
                threshold=None,
                threshold_type=RiskThresholdType.maximum,
                status=RiskLimitStatus.ok,
            ),
        ),
        exposures={},
        limits={},
        telemetry={},
        approval_rate=None,
        portfolio_equity=None,
        peak_equity=None,
    )

    markdown = format_risk_markdown(snapshot)

    assert "latency" not in markdown.lower()
    assert "Exposure" not in markdown


def test_helper_normalisers_and_exposure_calculations() -> None:
    state = {"open_positions": {"EURUSD": {"quantity": 2, "avg_price": 1.2}}}
    exposures = telemetry_module._compute_exposures(state["open_positions"])
    assert exposures["long"] > 0

    state = {"open_positions": {"USDJPY": {"quantity": -1, "current_value": 5000}}}
    exposures = telemetry_module._compute_exposures(state["open_positions"])
    assert exposures["short"] == pytest.approx(5000)

    assert telemetry_module._resolve_open_positions({"open_positions_count": "3"}) == 3
    assert telemetry_module._resolve_open_positions({"open_positions": {"A": {}}}) == 1
    assert telemetry_module._resolve_open_positions({}) == 0

    assert telemetry_module._approval_rate({"total_checks": 0}) is None
    assert telemetry_module._approval_rate({"total_checks": 10, "approved": 5}) == pytest.approx(0.5)

    assert telemetry_module._normalise_limits(None) == {}
    assert telemetry_module._normalise_limits({"max": "10"}) == {"max": 10.0}

    assert telemetry_module._normalise_telemetry(None) == {}
    assert telemetry_module._normalise_telemetry({"latency": "5"}) == {"latency": 5.0}


def test_decision_sanitisation_and_extraction_cover_duplicate_skips() -> None:
    raw_decision = {
        "status": "rejected",
        "reason": "latency",
        "symbol": "EURUSD",
        "strategy_id": "alpha",
        "checks": [
            {"name": "open_positions", "threshold": 5, "value": 3},
            {"name": "latency", "threshold": 1.0, "value": 0.5, "recommended": 1.5},
            {"name": "latency", "threshold": 1.0, "value": 0.4},
            {"name": "quality_floor", "threshold": 0.6},
        ],
    }

    sanitised = telemetry_module._sanitize_decision(raw_decision)
    assert sanitised["status"] == "rejected"
    assert len(sanitised["checks"]) == 4

    extracted = list(telemetry_module._extract_decision_checks(raw_decision, seen=set()))
    names = [item.name for item in extracted]
    assert "decision.latency" in names
    assert "decision.quality_floor" in names
    assert all(item.name != "decision.open_positions" for item in extracted)


@pytest.mark.asyncio()
async def test_publish_risk_snapshot_emits_event() -> None:
    snapshot = evaluate_risk_posture(
        _sample_state(),
        _sample_limits(),
        last_decision=_sample_decision(),
    )
    bus = StubBus()
    await publish_risk_snapshot(bus, snapshot, source="test")
    assert bus.events, "expected a telemetry event to be published"
    event = bus.events[-1]
    assert event.type == "telemetry.risk.posture"
    assert event.payload["status"] == RiskLimitStatus.alert.value
    assert "markdown" in event.payload
