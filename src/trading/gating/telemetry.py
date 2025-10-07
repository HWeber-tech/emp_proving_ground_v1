"""Telemetry helpers for DriftSentry gating decisions."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping

from src.core.event_bus import Event, EventBus

from .drift_sentry_gate import DriftSentryDecision

__all__ = [
    "DriftGateEvent",
    "format_drift_gate_markdown",
    "publish_drift_gate_event",
]


@dataclass(frozen=True, slots=True)
class DriftGateEvent:
    """Structured payload describing a DriftSentry gating decision."""

    event_id: str
    strategy_id: str | None
    symbol: str | None
    status: str
    decision: DriftSentryDecision
    confidence: float | None = None
    notional: float | None = None
    release: Mapping[str, Any] | None = None
    execution: Mapping[str, Any] | None = None
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "event_id": self.event_id,
            "status": self.status,
            "decision": self.decision.as_dict(),
            "generated_at": self.generated_at.isoformat(),
        }
        if self.strategy_id:
            payload["strategy_id"] = self.strategy_id
        if self.symbol:
            payload["symbol"] = self.symbol
        if self.confidence is not None:
            payload["confidence"] = float(self.confidence)
        if self.notional is not None:
            payload["notional"] = float(self.notional)
        if self.release:
            payload["release"] = dict(self.release)
        if self.execution:
            payload["execution"] = dict(self.execution)
        return payload


def format_drift_gate_markdown(event: DriftGateEvent) -> str:
    """Render a concise markdown summary for drift-gate telemetry."""

    decision = event.decision
    lines = [
        f"**Status:** {event.status}",
        f"**Severity:** {decision.severity.value}",
        f"**Allowed:** {'yes' if decision.allowed else 'no'}",
        f"**Force paper:** {'yes' if decision.force_paper else 'no'}",
    ]
    if decision.reason:
        lines.append(f"**Reason:** {decision.reason}")
    if event.symbol:
        lines.append(f"**Symbol:** {event.symbol}")
    if event.strategy_id:
        lines.append(f"**Strategy:** {event.strategy_id}")
    if event.confidence is not None:
        lines.append(f"**Confidence:** {event.confidence:.4f}")
    if event.notional is not None:
        lines.append(f"**Notional:** {event.notional:,.2f}")
    release_stage = decision.requirements.get("release_stage")
    if release_stage:
        lines.append(f"**Release stage:** {release_stage}")
    release_gate = decision.requirements.get("release_stage_gate")
    if release_gate:
        lines.append(f"**Stage gate:** {release_gate}")
    if event.release:
        forced_reason = event.release.get("forced_reason")
        route = event.release.get("route")
        if route:
            lines.append(f"**Execution route:** {route}")
        if forced_reason:
            lines.append(f"**Forced reason:** {forced_reason}")
    blocked = decision.blocked_dimensions
    if blocked:
        lines.append(f"**Blocked dimensions:** {', '.join(blocked)}")
    return "\n".join(lines)


async def publish_drift_gate_event(
    event_bus: EventBus,
    event: DriftGateEvent,
    *,
    source: str = "trading_manager",
) -> None:
    """Publish the drift-gate telemetry event to the runtime event bus."""

    payload = event.as_dict()
    payload["markdown"] = format_drift_gate_markdown(event)
    await event_bus.publish(
        Event(type="telemetry.trading.drift_gate", payload=payload, source=source)
    )
