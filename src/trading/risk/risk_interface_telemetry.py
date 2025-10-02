"""Telemetry helpers for publishing trading risk interface posture."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping

from src.core.event_bus import Event, EventBus

from .risk_api import RiskApiError, TradingRiskInterface


@dataclass(frozen=True)
class RiskInterfaceSnapshot:
    """Structured telemetry payload for the trading risk interface."""

    summary: Mapping[str, object]
    config: Mapping[str, Any]
    status: Mapping[str, object] | None
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def as_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "summary": dict(self.summary),
            "config": dict(self.config),
            "generated_at": self.generated_at.isoformat(),
        }
        if self.status is not None:
            payload["status"] = dict(self.status)
        return payload


def build_risk_interface_snapshot(
    interface: TradingRiskInterface,
    *,
    generated_at: datetime | None = None,
) -> RiskInterfaceSnapshot:
    """Create a telemetry snapshot from ``interface``."""

    timestamp = generated_at or datetime.now(timezone.utc)
    summary = interface.summary()
    status: Mapping[str, object] | None = None
    if interface.status is not None:
        status = dict(interface.status)
    return RiskInterfaceSnapshot(
        summary=summary,
        config=interface.config.dict(),
        status=status,
        generated_at=timestamp,
    )


def format_risk_interface_markdown(snapshot: RiskInterfaceSnapshot) -> str:
    """Render ``snapshot`` as Markdown for logging/alerts."""

    summary = snapshot.summary
    lines = ["**Trading risk interface summary**"]
    limits = [
        f"max_risk_per_trade={summary.get('max_risk_per_trade_pct', 'n/a')}",
        f"max_total_exposure={summary.get('max_total_exposure_pct', 'n/a')}",
        f"max_drawdown={summary.get('max_drawdown_pct', 'n/a')}",
    ]
    if "mandatory_stop_loss" in summary:
        state = "ENABLED" if summary["mandatory_stop_loss"] else "DISABLED"
        limits.append(f"stop_loss={state}")
    if "research_mode" in summary:
        limits.append(f"research_mode={summary['research_mode']}")
    lines.append(" • ".join(limits))
    policy_limits = summary.get("policy_limits")
    if isinstance(policy_limits, Mapping) and policy_limits:
        fragments = [f"{key}={value}" for key, value in policy_limits.items()]
        lines.append("Policy limits: " + ", ".join(fragments))
    snapshot_status = summary.get("latest_snapshot")
    if isinstance(snapshot_status, Mapping) and snapshot_status:
        parts = [f"{key}={value}" for key, value in snapshot_status.items()]
        lines.append("Latest telemetry: " + ", ".join(parts))
    return "\n".join(lines)


@dataclass(frozen=True)
class RiskInterfaceErrorAlert:
    """Telemetry alert capturing a Risk API contract violation."""

    message: str
    runbook: str
    details: Mapping[str, object]
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def as_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "message": self.message,
            "runbook": self.runbook,
            "generated_at": self.generated_at.isoformat(),
        }
        if self.details:
            payload["details"] = dict(self.details)
        return payload


def build_risk_interface_error(
    error: RiskApiError,
    *,
    generated_at: datetime | None = None,
) -> RiskInterfaceErrorAlert:
    """Convert ``error`` into an alert payload."""

    timestamp = generated_at or datetime.now(timezone.utc)
    return RiskInterfaceErrorAlert(
        message=str(error),
        runbook=error.runbook,
        details=error.to_metadata().get("details", {}),
        generated_at=timestamp,
    )


def format_risk_interface_error_markdown(alert: RiskInterfaceErrorAlert) -> str:
    """Render ``alert`` into Markdown."""

    lines = [
        "⚠️ **Trading risk interface error**",
        f"**Message:** {alert.message}",
        f"**Runbook:** {alert.runbook}",
    ]
    if alert.details:
        fragments = [f"{key}={value}" for key, value in alert.details.items()]
        lines.append("Details: " + ", ".join(fragments))
    return "\n".join(lines)


async def publish_risk_interface_snapshot(
    event_bus: EventBus,
    snapshot: RiskInterfaceSnapshot,
    *,
    source: str = "trading_manager",
) -> None:
    """Publish ``snapshot`` on the event bus."""

    payload = snapshot.as_dict()
    payload["markdown"] = format_risk_interface_markdown(snapshot)
    event = Event(type="telemetry.risk.interface", payload=payload, source=source)
    await event_bus.publish(event)


async def publish_risk_interface_error(
    event_bus: EventBus,
    alert: RiskInterfaceErrorAlert,
    *,
    source: str = "trading_manager",
) -> None:
    """Publish ``alert`` on the event bus."""

    payload = alert.as_dict()
    payload["markdown"] = format_risk_interface_error_markdown(alert)
    event = Event(type="telemetry.risk.interface_error", payload=payload, source=source)
    await event_bus.publish(event)


__all__ = [
    "RiskInterfaceSnapshot",
    "RiskInterfaceErrorAlert",
    "build_risk_interface_snapshot",
    "build_risk_interface_error",
    "format_risk_interface_markdown",
    "format_risk_interface_error_markdown",
    "publish_risk_interface_snapshot",
    "publish_risk_interface_error",
]
