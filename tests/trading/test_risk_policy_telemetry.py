from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Mapping

import pytest

from src.config.risk.risk_config import RiskConfig
from src.trading.risk.policy_telemetry import (
    PolicyCheckStatus,
    RiskPolicyViolationAlert,
    build_policy_snapshot,
    build_policy_violation_alert,
    format_policy_markdown,
    format_policy_violation_markdown,
    publish_policy_snapshot,
    publish_policy_violation,
)
from src.trading.risk.risk_policy import RiskPolicy, RiskPolicyDecision


class StubBus:
    def __init__(self) -> None:
        self.events: list[Any] = []

    async def publish(self, event: Any) -> None:
        self.events.append(event)


def _portfolio_state(
    open_positions: Mapping[str, Mapping[str, float]] | None = None,
) -> Mapping[str, object]:
    return {
        "equity": 100_000.0,
        "open_positions": open_positions or {},
        "current_daily_drawdown": 0.02,
    }


def test_build_policy_snapshot_records_metadata() -> None:
    policy = RiskPolicy.from_config(
        RiskConfig(
            max_risk_per_trade_pct=Decimal("0.02"),
            max_total_exposure_pct=Decimal("0.2"),
            max_leverage=Decimal("2"),
            max_drawdown_pct=Decimal("0.1"),
            min_position_size=1,
        )
    )

    decision = policy.evaluate(
        symbol="EURUSD",
        quantity=1_000.0,
        price=1.05,
        stop_loss_pct=0.02,
        portfolio_state=_portfolio_state(),
    )

    snapshot = build_policy_snapshot(decision, policy)

    assert snapshot.symbol == "EURUSD"
    assert snapshot.approved is True
    assert snapshot.policy_limits["max_total_exposure_pct"] == pytest.approx(0.2)
    assert snapshot.metadata["equity"] == pytest.approx(100_000.0)
    assert snapshot.research_mode is policy.research_mode
    assert snapshot.checks


def test_format_policy_markdown_includes_status() -> None:
    policy = RiskPolicy.from_config(
        RiskConfig(
            max_risk_per_trade_pct=Decimal("0.01"),
            max_total_exposure_pct=Decimal("0.05"),
            max_leverage=Decimal("1.5"),
            max_drawdown_pct=Decimal("0.05"),
            min_position_size=1,
        )
    )
    state = _portfolio_state({"EURUSD": {"quantity": 5000.0, "last_price": 1.2}})

    decision = policy.evaluate(
        symbol="EURUSD",
        quantity=10_000.0,
        price=1.2,
        stop_loss_pct=0.05,
        portfolio_state=state,
    )

    snapshot = build_policy_snapshot(decision, policy)
    markdown = format_policy_markdown(snapshot)

    assert "POLICY STATUS" in markdown.upper()
    assert "VIOLATIONS" in markdown.upper()
    assert "LIMITS" in markdown.upper()


def test_build_policy_snapshot_without_policy_uses_metadata() -> None:
    decision = RiskPolicyDecision(
        approved=False,
        reason="exposure_limit",
        checks=(
            {
                "name": "policy.max_total_exposure_pct",
                "status": "violation",
                "value": "100000",
                "threshold": "50000",
                "extra": {
                    "ratio": "2.0",
                    "projected_total_exposure": "100000",
                },
            },
        ),
        metadata={
            "symbol": "GBPUSD",
            "equity": "250000",
            "projected_total_exposure": "100000",
            "research_mode": True,
        },
        violations=("policy.max_total_exposure_pct",),
    )

    generated_at = datetime(2024, 2, 1, 15, 30, tzinfo=timezone.utc)
    snapshot = build_policy_snapshot(decision, None, generated_at=generated_at)

    assert snapshot.generated_at == generated_at
    assert snapshot.symbol == "GBPUSD"
    assert snapshot.policy_limits == {}
    assert snapshot.research_mode is True
    assert snapshot.approved is False
    assert snapshot.violations == ("policy.max_total_exposure_pct",)

    (check,) = snapshot.checks
    assert check.status is PolicyCheckStatus.violation
    assert check.value == pytest.approx(100000.0)
    assert check.threshold == pytest.approx(50000.0)
    assert check.metadata["ratio"] == "2.0"
    assert check.metadata["projected_total_exposure"] == "100000"

    markdown = format_policy_markdown(snapshot)
    assert "RESEARCH MODE" in markdown.upper()
    assert "**Exposure:** equity=250,000.00 projected=100,000.00" in markdown


@pytest.mark.asyncio()
async def test_publish_policy_snapshot_emits_event() -> None:
    policy = RiskPolicy.from_config(
        RiskConfig(
            max_risk_per_trade_pct=Decimal("0.02"),
            max_total_exposure_pct=Decimal("0.2"),
            max_leverage=Decimal("2"),
            max_drawdown_pct=Decimal("0.1"),
            min_position_size=1,
        )
    )
    decision = policy.evaluate(
        symbol="EURUSD",
        quantity=2_000.0,
        price=1.1,
        stop_loss_pct=0.02,
        portfolio_state=_portfolio_state(),
    )

    snapshot = build_policy_snapshot(decision, policy)
    bus = StubBus()

    await publish_policy_snapshot(bus, snapshot, source="tests")

    assert bus.events, "expected telemetry event"
    event = bus.events[-1]
    assert event.type == "telemetry.risk.policy"
    assert event.payload["approved"] is True
    assert "markdown" in event.payload


def test_build_policy_violation_alert_wraps_snapshot() -> None:
    policy = RiskPolicy.from_config(
        RiskConfig(
            max_risk_per_trade_pct=Decimal("0.02"),
            max_total_exposure_pct=Decimal("0.2"),
            max_leverage=Decimal("2"),
            max_drawdown_pct=Decimal("0.1"),
            min_position_size=10,
        )
    )
    decision = policy.evaluate(
        symbol="EURUSD",
        quantity=1.0,
        price=1.1,
        stop_loss_pct=0.0,
        portfolio_state=_portfolio_state(),
    )
    snapshot = build_policy_snapshot(decision, policy)
    alert = build_policy_violation_alert(
        snapshot,
        severity="critical",
        runbook="docs/operations/runbooks/risk_policy_violation.md",
    )

    assert isinstance(alert, RiskPolicyViolationAlert)
    assert alert.snapshot is snapshot
    assert alert.severity == "critical"
    assert alert.runbook.endswith("risk_policy_violation.md")
    markdown = format_policy_violation_markdown(alert)
    assert "POLICY VIOLATION" in markdown.upper()
    assert "RUNBOOK" in markdown.upper()


@pytest.mark.asyncio()
async def test_publish_policy_violation_emits_event() -> None:
    policy = RiskPolicy.from_config(
        RiskConfig(
            max_risk_per_trade_pct=Decimal("0.02"),
            max_total_exposure_pct=Decimal("0.2"),
            max_leverage=Decimal("2"),
            max_drawdown_pct=Decimal("0.1"),
            min_position_size=100,
        )
    )
    decision = policy.evaluate(
        symbol="EURUSD",
        quantity=1.0,
        price=1.1,
        stop_loss_pct=0.0,
        portfolio_state=_portfolio_state(),
    )
    snapshot = build_policy_snapshot(decision, policy)
    alert = build_policy_violation_alert(snapshot, severity="warning")
    bus = StubBus()

    await publish_policy_violation(bus, alert, source="tests")

    assert bus.events, "expected violation telemetry"
    event = bus.events[-1]
    assert event.type == "telemetry.risk.policy_violation"
    assert event.payload["severity"] == "warning"
    assert event.payload["snapshot"]["approved"] is snapshot.approved
    assert "markdown" in event.payload
