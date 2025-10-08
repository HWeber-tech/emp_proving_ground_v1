"""Telemetry helpers for DriftSentry and release routing decisions."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping

from src.core.event_bus import Event, EventBus

from .drift_sentry_gate import DriftSentryDecision

__all__ = [
    "DriftGateEvent",
    "ReleaseRouteEvent",
    "format_drift_gate_markdown",
    "format_release_route_markdown",
    "publish_drift_gate_event",
    "publish_release_route_event",
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


@dataclass(frozen=True, slots=True)
class ReleaseRouteEvent:
    """Structured payload describing release-aware execution routing."""

    event_id: str
    status: str
    strategy_id: str | None = None
    stage: str | None = None
    route: str | None = None
    forced: bool = False
    forced_reason: str | None = None
    forced_reasons: tuple[str, ...] = ()
    overridden: bool | None = None
    audit: Mapping[str, Any] | None = None
    drift_severity: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "event_id": self.event_id,
            "status": self.status,
            "generated_at": self.generated_at.isoformat(),
            "forced": bool(self.forced),
        }
        if self.strategy_id:
            payload["strategy_id"] = self.strategy_id
        if self.stage:
            payload["stage"] = self.stage
        if self.route:
            payload["route"] = self.route
        if self.forced_reason:
            payload["forced_reason"] = self.forced_reason
        if self.forced_reasons:
            payload["forced_reasons"] = list(self.forced_reasons)
        if self.overridden is not None:
            payload["overridden"] = self.overridden
        if self.audit:
            payload["audit"] = dict(self.audit)
        if self.drift_severity:
            payload["drift_severity"] = self.drift_severity
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
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


def format_release_route_markdown(event: ReleaseRouteEvent) -> str:
    """Render a concise markdown summary for release-route telemetry."""

    lines = [
        f"**Status:** {event.status}",
        f"**Stage:** {event.stage or 'unknown'}",
        f"**Route:** {event.route or 'unknown'}",
        f"**Forced:** {'yes' if event.forced else 'no'}",
    ]
    if event.forced_reason:
        lines.append(f"**Forced reason:** {event.forced_reason}")
    if event.forced_reasons and (event.forced_reason is None or len(event.forced_reasons) > 1):
        lines.append(f"**Forced reasons:** {', '.join(event.forced_reasons)}")
    if event.overridden:
        lines.append("**Route overridden:** yes")
    if event.drift_severity:
        lines.append(f"**Drift severity:** {event.drift_severity}")
    if event.audit:
        enforced = event.audit.get("enforced")
        if enforced:
            lines.append("**Audit enforced:** yes")
        gaps = event.audit.get("gaps")
        if gaps:
            lines.append(f"**Audit gaps:** {', '.join(str(gap) for gap in gaps)}")
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


async def publish_release_route_event(
    event_bus: EventBus,
    event: ReleaseRouteEvent,
    *,
    source: str = "trading_manager",
) -> None:
    """Publish the release-route telemetry event to the runtime event bus."""

    payload = event.as_dict()
    payload["markdown"] = format_release_route_markdown(event)
    await event_bus.publish(
        Event(type="telemetry.trading.release_route", payload=payload, source=source)
    )
