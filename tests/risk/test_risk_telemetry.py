from __future__ import annotations

from datetime import timezone
from typing import Any

import pytest

from src.risk.telemetry import (
    RiskLimitStatus,
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
