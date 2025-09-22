from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.operations.roi import (
    RoiCostModel,
    RoiStatus,
    evaluate_roi_posture,
    format_roi_markdown,
    publish_roi_snapshot,
)


class StubBus:
    def __init__(self) -> None:
        self.events: list[object] = []

    async def publish(self, event: object) -> None:
        self.events.append(event)


def test_evaluate_roi_posture_computes_costs() -> None:
    state = {
        "equity": 112_000.0,
        "total_pnl": 12_000.0,
    }
    model = RoiCostModel(
        initial_capital=100_000.0,
        target_annual_roi=0.25,
        infrastructure_daily_cost=40.0,
        broker_fee_flat=3.0,
        broker_fee_bps=0.2,
    )
    start = datetime.now(timezone.utc) - timedelta(days=10)
    snapshot = evaluate_roi_posture(
        state,
        model,
        executed_trades=24,
        total_notional=480_000.0,
        period_start=start,
        as_of=start + timedelta(days=10),
    )

    assert snapshot.status is RoiStatus.ahead
    assert snapshot.executed_trades == 24
    assert snapshot.total_notional == pytest.approx(480_000.0)
    assert snapshot.infrastructure_cost == pytest.approx(400.0)
    assert snapshot.fees == pytest.approx(81.6, rel=1e-5)
    assert snapshot.net_pnl == pytest.approx(11_518.4, rel=1e-5)
    assert snapshot.annualised_roi > model.target_annual_roi


def test_format_roi_markdown_includes_summary() -> None:
    model = RoiCostModel.bootstrap_defaults(50_000.0)
    start = datetime.now(timezone.utc) - timedelta(days=5)
    snapshot = evaluate_roi_posture(
        {"equity": 52_500.0, "total_pnl": 2_500.0},
        model,
        executed_trades=10,
        total_notional=120_000.0,
        period_start=start,
    )

    markdown = format_roi_markdown(snapshot)
    assert "STATUS" in markdown.upper()
    assert "ROI" in markdown.upper()


@pytest.mark.asyncio()
async def test_publish_roi_snapshot_emits_event() -> None:
    model = RoiCostModel.bootstrap_defaults(25_000.0)
    snapshot = evaluate_roi_posture(
        {"equity": 26_000.0, "total_pnl": 1_000.0},
        model,
        executed_trades=5,
        total_notional=30_000.0,
        period_start=datetime.now(timezone.utc) - timedelta(days=2),
    )

    bus = StubBus()
    await publish_roi_snapshot(bus, snapshot, source="test")
    assert bus.events, "expected a telemetry event"
    event = bus.events[-1]
    assert getattr(event, "type", "") == "telemetry.operational.roi"
    payload = getattr(event, "payload", {})
    assert payload.get("status") in {status.value for status in RoiStatus}
    assert "markdown" in payload
