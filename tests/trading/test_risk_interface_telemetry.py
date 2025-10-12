"""Regression coverage for the trading risk interface telemetry helpers."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

import pytest

from src.config.risk.risk_config import RiskConfig
from src.core.event_bus import Event
from src.trading.risk.risk_api import RiskApiError, TradingRiskInterface
from src.trading.risk.risk_interface_telemetry import (
    RiskInterfaceErrorAlert,
    RiskInterfaceSnapshot,
    build_risk_interface_error,
    build_risk_interface_snapshot,
    format_risk_interface_error_markdown,
    format_risk_interface_markdown,
    publish_risk_interface_error,
    publish_risk_interface_snapshot,
)


class _RecordingEventBus:
    """Minimal event bus stub capturing published events for assertions."""

    def __init__(self) -> None:
        self.events: list[Event] = []

    async def publish(self, event: Event) -> None:  # pragma: no cover - exercised in tests
        self.events.append(event)


def _make_interface() -> TradingRiskInterface:
    config = RiskConfig(
        max_risk_per_trade_pct=Decimal("0.03"),
        max_total_exposure_pct=Decimal("0.40"),
        sector_exposure_limits={"FX": Decimal("0.4")},
        instrument_sector_map={"EURUSD": "fx"},
        research_mode=True,
    )
    status = {
        "policy_limits": {"daily_loss_pct": 5},
        "policy_research_mode": True,
        "snapshot": {"breaches": 0, "liquidity": "nominal"},
    }
    return TradingRiskInterface(config=config, status=status)


def test_snapshot_as_dict_includes_status_copy() -> None:
    timestamp = datetime(2024, 12, 31, 23, 59, tzinfo=timezone.utc)
    interface = _make_interface()
    snapshot = build_risk_interface_snapshot(interface, generated_at=timestamp)

    payload = snapshot.as_dict()

    assert payload["summary"]["policy_limits"] == {"daily_loss_pct": 5}
    assert payload["config"]["mandatory_stop_loss"] is True
    assert payload["status"] == interface.status
    assert payload["generated_at"] == timestamp.isoformat()


def test_snapshot_as_dict_omits_status_when_missing() -> None:
    config = RiskConfig()
    interface = TradingRiskInterface(config=config, status=None)
    snapshot = build_risk_interface_snapshot(interface)

    payload = snapshot.as_dict()

    assert "status" not in payload
    assert payload["summary"]["max_risk_per_trade_pct"] == pytest.approx(0.02)


def test_format_risk_interface_markdown_renders_limits() -> None:
    interface = _make_interface()
    snapshot = build_risk_interface_snapshot(interface)

    markdown = format_risk_interface_markdown(snapshot)

    assert "**Trading risk interface summary**" in markdown
    assert "max_risk_per_trade=" in markdown
    assert "Policy limits: daily_loss_pct=5" in markdown
    assert "Latest telemetry: breaches=0" in markdown
    assert "research_mode=True" in markdown


def test_build_risk_interface_error_copies_metadata() -> None:
    error = RiskApiError(
        "failed", details={"field": "max_risk"}, runbook="https://example.com/runbook"
    )
    timestamp = datetime(2025, 1, 1, tzinfo=timezone.utc)

    alert = build_risk_interface_error(error, generated_at=timestamp)

    assert isinstance(alert, RiskInterfaceErrorAlert)
    assert alert.message == "failed"
    assert alert.details == {"field": "max_risk"}
    assert alert.runbook == "https://example.com/runbook"
    assert alert.generated_at is timestamp


def test_format_risk_interface_error_markdown_includes_details() -> None:
    alert = RiskInterfaceErrorAlert(
        message="bad payload",
        runbook="https://example.com/runbook",
        details={"reason": "missing status"},
    )

    markdown = format_risk_interface_error_markdown(alert)

    assert markdown.splitlines()[0].startswith("⚠️ **Trading risk interface error**")
    assert "**Message:** bad payload" in markdown
    assert "Details: reason=missing status" in markdown


@pytest.mark.asyncio
async def test_publish_risk_interface_snapshot_wraps_event_payload() -> None:
    interface = _make_interface()
    snapshot = build_risk_interface_snapshot(interface)
    bus = _RecordingEventBus()

    await publish_risk_interface_snapshot(bus, snapshot, source="audit")

    assert len(bus.events) == 1
    event = bus.events[0]
    assert event.type == "telemetry.risk.interface"
    assert event.source == "audit"
    assert event.payload["summary"]["policy_limits"] == {"daily_loss_pct": 5}
    assert "markdown" in event.payload
    assert "Trading risk interface summary" in event.payload["markdown"]


@pytest.mark.asyncio
async def test_publish_risk_interface_error_wraps_event_payload() -> None:
    error = RiskApiError("boom", details={"field": "exposure"})
    alert = build_risk_interface_error(error)
    bus = _RecordingEventBus()

    await publish_risk_interface_error(bus, alert, source="audit")

    assert len(bus.events) == 1
    event = bus.events[0]
    assert event.type == "telemetry.risk.interface_error"
    assert event.source == "audit"
    assert event.payload["details"] == {"field": "exposure"}
    assert event.payload["markdown"].startswith("⚠️ **Trading risk interface error**")


def test_error_alert_as_dict_drops_empty_details() -> None:
    alert = RiskInterfaceErrorAlert(message="issue", runbook="https://runbook", details={})

    payload = alert.as_dict()

    assert payload == {
        "message": "issue",
        "runbook": "https://runbook",
        "generated_at": alert.generated_at.isoformat(),
    }


def test_snapshot_round_trip_preserves_summary_and_status() -> None:
    interface = _make_interface()
    snapshot = build_risk_interface_snapshot(interface)
    recovered = RiskInterfaceSnapshot(**snapshot.as_dict())

    assert recovered.summary == snapshot.summary
    assert recovered.config == snapshot.config
    assert recovered.status == snapshot.status
