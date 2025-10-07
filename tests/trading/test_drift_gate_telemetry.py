from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pytest

from src.operations.sensory_drift import DriftSeverity
from src.trading.gating.drift_sentry_gate import DriftSentryDecision
from src.trading.gating.telemetry import (
    DriftGateEvent,
    format_drift_gate_markdown,
    publish_drift_gate_event,
)


class StubBus:
    def __init__(self) -> None:
        self.events: list[Any] = []

    async def publish(self, event: Any) -> None:
        self.events.append(event)


def _sample_decision(**overrides: Any) -> DriftSentryDecision:
    base = DriftSentryDecision(
        allowed=True,
        severity=DriftSeverity.warn,
        evaluated_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        reason="drift_detected",
        requirements={"release_stage": "limited_live"},
        blocked_dimensions=("HOW", "WHY"),
        snapshot_metadata={"source": "test"},
        force_paper=True,
    )
    if not overrides:
        return base
    return DriftSentryDecision(
        allowed=overrides.get("allowed", base.allowed),
        severity=overrides.get("severity", base.severity),
        evaluated_at=overrides.get("evaluated_at", base.evaluated_at),
        reason=overrides.get("reason", base.reason),
        requirements=overrides.get("requirements", base.requirements),
        blocked_dimensions=overrides.get("blocked_dimensions", base.blocked_dimensions),
        snapshot_metadata=overrides.get("snapshot_metadata", base.snapshot_metadata),
        force_paper=overrides.get("force_paper", base.force_paper),
    )


def test_drift_gate_event_as_dict_serialises_decision() -> None:
    decision = _sample_decision()
    event = DriftGateEvent(
        event_id="evt-123",
        strategy_id="alpha",
        symbol="EURUSD",
        status="executed",
        decision=decision,
        confidence=0.82,
        notional=12_500.0,
        release={"route": "paper", "forced_reason": "drift_gate_severity_warn"},
    )

    payload = event.as_dict()
    assert payload["event_id"] == "evt-123"
    assert payload["status"] == "executed"
    assert payload["decision"]["severity"] == DriftSeverity.warn.value
    assert payload["decision"]["force_paper"] is True
    assert payload["release"]["route"] == "paper"
    assert payload["confidence"] == pytest.approx(0.82)
    assert payload["notional"] == pytest.approx(12_500.0)

    markdown = format_drift_gate_markdown(event)
    assert "Severity" in markdown
    assert "Force paper" in markdown
    assert "Execution route" in markdown


@pytest.mark.asyncio()
async def test_publish_drift_gate_event_emits_event() -> None:
    bus = StubBus()
    decision = _sample_decision()
    event = DriftGateEvent(
        event_id="evt-abc",
        strategy_id="alpha",
        symbol="EURUSD",
        status="gated",
        decision=decision,
        confidence=0.4,
        notional=5_000.0,
    )

    await publish_drift_gate_event(bus, event, source="unit-test")

    assert bus.events, "expected drift gate telemetry event"
    emitted = bus.events[-1]
    assert emitted.type == "telemetry.trading.drift_gate"
    assert emitted.source == "unit-test"
    assert emitted.payload["status"] == "gated"
    assert emitted.payload["decision"]["force_paper"] is True
    assert "markdown" in emitted.payload
