from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

import pytest

from src.config.risk.risk_config import RiskConfig
from src.trading.risk.risk_api import RiskApiError, TradingRiskInterface
from src.trading.risk.risk_interface_telemetry import (
    build_risk_interface_error,
    build_risk_interface_snapshot,
    format_risk_interface_error_markdown,
    format_risk_interface_markdown,
    publish_risk_interface_error,
    publish_risk_interface_snapshot,
)


class _DummyEventBus:
    def __init__(self) -> None:
        self.events: list[object] = []

    async def publish(self, event: object) -> None:
        self.events.append(event)


@pytest.mark.asyncio
async def test_publish_risk_interface_snapshot_emits_structured_payload() -> None:
    config = RiskConfig(
        max_total_exposure_pct=Decimal("0.4"),
        sector_exposure_limits={"FX": Decimal("0.3")},
        instrument_sector_map={"EURUSD": "FX"},
    )

    interface = TradingRiskInterface(
        config=config,
        status={
            "policy_limits": {"daily_orders": 25},
            "policy_research_mode": True,
            "snapshot": {"state": "healthy", "updated_at": "2024-01-01T00:00:00Z"},
        },
    )

    snapshot_time = datetime(2024, 1, 1, tzinfo=timezone.utc)
    snapshot = build_risk_interface_snapshot(interface, generated_at=snapshot_time)

    markdown = format_risk_interface_markdown(snapshot)
    assert "max_total_exposure=0.4" in markdown
    assert "Policy limits: daily_orders=25" in markdown
    assert "Latest telemetry: state=healthy" in markdown

    bus = _DummyEventBus()
    await publish_risk_interface_snapshot(bus, snapshot, source="tests")

    assert len(bus.events) == 1
    event = bus.events[0]
    assert getattr(event, "type") == "telemetry.risk.interface"
    assert getattr(event, "source") == "tests"

    payload = getattr(event, "payload")
    assert payload["summary"]["policy_limits"] == {"daily_orders": 25}
    assert payload["generated_at"] == snapshot_time.isoformat()
    assert "Policy limits: daily_orders=25" in payload["markdown"]


@pytest.mark.asyncio
async def test_publish_risk_interface_error_includes_runbook_and_details() -> None:
    bus = _DummyEventBus()
    error_time = datetime(2024, 2, 2, tzinfo=timezone.utc)

    error = RiskApiError(
        "Risk policy violation detected",
        details={"limit": "exposure", "value": 1.2},
        runbook="docs/runbooks/risk_violation.md",
    )

    alert = build_risk_interface_error(error, generated_at=error_time)

    markdown = format_risk_interface_error_markdown(alert)
    assert "Risk policy violation detected" in markdown
    assert "**Runbook:** docs/runbooks/risk_violation.md" in markdown
    assert "limit=exposure" in markdown

    await publish_risk_interface_error(bus, alert, source="tests")

    assert len(bus.events) == 1
    event = bus.events[0]
    assert getattr(event, "type") == "telemetry.risk.interface_error"
    assert getattr(event, "source") == "tests"

    payload = getattr(event, "payload")
    assert payload["message"] == "Risk policy violation detected"
    assert payload["runbook"] == "docs/runbooks/risk_violation.md"
    assert payload["generated_at"] == error_time.isoformat()
    assert payload["details"] == {"limit": "exposure", "value": 1.2}
    assert "⚠️ **Trading risk interface error**" in payload["markdown"]
