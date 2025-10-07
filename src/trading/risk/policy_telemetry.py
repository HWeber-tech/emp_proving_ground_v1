"""Telemetry helpers for :mod:`src.trading.risk.risk_policy`."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from numbers import Real
from typing import Any, Mapping, MutableMapping, Sequence, cast

from src.core.event_bus import Event, EventBus


RISK_POLICY_VIOLATION_RUNBOOK = "docs/operations/runbooks/risk_policy_violation.md"

from .risk_policy import RiskPolicy, RiskPolicyDecision


class PolicyCheckStatus(StrEnum):
    """Status emitted for an individual policy guardrail."""

    ok = "ok"
    warn = "warn"
    violation = "violation"


@dataclass(frozen=True)
class RiskPolicyCheckSnapshot:
    """Serializable view of a single policy check evaluation."""

    name: str
    status: PolicyCheckStatus
    value: float | None = None
    threshold: float | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "name": self.name,
            "status": self.status.value,
        }
        if self.value is not None:
            payload["value"] = self.value
        if self.threshold is not None:
            payload["threshold"] = self.threshold
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload


@dataclass(frozen=True)
class RiskPolicyEvaluationSnapshot:
    """Structured telemetry payload for a policy decision."""

    symbol: str
    approved: bool
    reason: str | None
    generated_at: datetime
    checks: tuple[RiskPolicyCheckSnapshot, ...]
    policy_limits: Mapping[str, float]
    metadata: Mapping[str, object]
    violations: tuple[str, ...]
    research_mode: bool

    def as_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "symbol": self.symbol,
            "approved": self.approved,
            "generated_at": self.generated_at.isoformat(),
            "checks": [check.as_dict() for check in self.checks],
            "policy_limits": dict(self.policy_limits),
            "violations": list(self.violations),
            "research_mode": self.research_mode,
        }
        if self.reason is not None:
            payload["reason"] = self.reason
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload


def _normalise_status(value: object) -> PolicyCheckStatus:
    if isinstance(value, PolicyCheckStatus):
        return value
    try:
        return PolicyCheckStatus(str(value))
    except Exception:
        return PolicyCheckStatus.ok


def _coerce_float(value: object | None) -> float | None:
    """Best-effort conversion of ``value`` into ``float``."""

    if value is None:
        return None
    if isinstance(value, Real):
        return float(value)
    if isinstance(value, str):
        candidate = value.strip()
        if not candidate:
            return None
        try:
            return float(candidate)
        except ValueError:
            return None
    try:
        return float(cast(Any, value))
    except (TypeError, ValueError):
        return None


def _coerce_mapping(value: object) -> Mapping[str, object]:
    if isinstance(value, Mapping):
        return cast(Mapping[str, object], value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return {str(index): element for index, element in enumerate(value)}
    if value is None:
        return {}
    return {"value": value}


def build_policy_snapshot(
    decision: RiskPolicyDecision,
    policy: RiskPolicy | None = None,
    *,
    generated_at: datetime | None = None,
) -> RiskPolicyEvaluationSnapshot:
    """Create a telemetry snapshot for ``decision``."""

    timestamp = generated_at or datetime.now(timezone.utc)
    checks: list[RiskPolicyCheckSnapshot] = []

    for entry in decision.checks:
        if not isinstance(entry, Mapping):
            continue
        entry_mapping = cast(Mapping[str, object], entry)
        metadata: MutableMapping[str, object] = {}
        extra = entry_mapping.get("extra")
        if isinstance(extra, Mapping):
            metadata.update(cast(Mapping[str, object], extra))
        for key in ("ratio", "resolved_price", "projected_total_exposure"):
            value = entry_mapping.get(key)
            if value is not None and key not in metadata:
                metadata[key] = value
        checks.append(
            RiskPolicyCheckSnapshot(
                name=str(entry_mapping.get("name", "unknown")),
                status=_normalise_status(entry_mapping.get("status")),
                value=_coerce_float(entry_mapping.get("value")),
                threshold=_coerce_float(entry_mapping.get("threshold")),
                metadata=dict(metadata) if metadata else {},
            )
        )

    limits: Mapping[str, float]
    if policy is not None:
        limits = policy.limit_snapshot()
    else:
        limits = cast(Mapping[str, float], {})
    decision_metadata: dict[str, object] = dict(decision.metadata)
    symbol = str(decision_metadata.get("symbol", "UNKNOWN"))
    research_mode = bool(
        decision_metadata.get("research_mode", getattr(policy, "research_mode", False))
    )

    return RiskPolicyEvaluationSnapshot(
        symbol=symbol,
        approved=decision.approved,
        reason=decision.reason,
        generated_at=timestamp,
        checks=tuple(checks),
        policy_limits=limits,
        metadata=decision_metadata,
        violations=tuple(decision.violations),
        research_mode=research_mode,
    )


def format_policy_markdown(snapshot: RiskPolicyEvaluationSnapshot) -> str:
    """Render ``snapshot`` as a compact Markdown summary."""

    lines = [
        f"**Policy status:** {'APPROVED' if snapshot.approved else 'REJECTED'}",
        f"**Symbol:** {snapshot.symbol}",
    ]
    if snapshot.reason:
        lines.append(f"**Reason:** {snapshot.reason}")
    if snapshot.violations:
        lines.append("**Violations:** " + ", ".join(snapshot.violations))
    if snapshot.policy_limits:
        limit_parts = [f"{key}={value}" for key, value in snapshot.policy_limits.items()]
        lines.append("**Limits:** " + ", ".join(limit_parts))
    if snapshot.metadata:
        equity = _coerce_float(snapshot.metadata.get("equity"))
        projected = _coerce_float(snapshot.metadata.get("projected_total_exposure"))
        if equity is not None or projected is not None:
            lines.append(
                "**Exposure:** equity={:,.2f} projected={:,.2f}".format(
                    equity if equity is not None else 0.0,
                    projected if projected is not None else 0.0,
                )
            )
    if snapshot.research_mode:
        lines.append("**Research mode:** enabled")
    for check in snapshot.checks:
        threshold = f" / {check.threshold}" if check.threshold is not None else ""
        value = check.value if check.value is not None else "n/a"
        lines.append(f"- `{check.name}` → {check.status.value.upper()}: {value}{threshold}")
    return "\n".join(lines)


async def publish_policy_snapshot(
    event_bus: EventBus,
    snapshot: RiskPolicyEvaluationSnapshot,
    *,
    source: str = "risk_gateway",
) -> None:
    """Publish ``snapshot`` on the event bus."""

    payload = snapshot.as_dict()
    payload["markdown"] = format_policy_markdown(snapshot)
    event = Event(type="telemetry.risk.policy", payload=payload, source=source)
    await event_bus.publish(event)


@dataclass(frozen=True)
class RiskPolicyViolationAlert:
    """Telemetry surface for policy violations requiring escalation."""

    snapshot: RiskPolicyEvaluationSnapshot
    severity: str
    runbook: str | None
    generated_at: datetime

    def as_dict(self) -> dict[str, object]:
        payload = {
            "severity": self.severity,
            "runbook": self.runbook,
            "generated_at": self.generated_at.isoformat(),
            "snapshot": self.snapshot.as_dict(),
        }
        if self.runbook is None:
            payload.pop("runbook")
        return payload


def _first_violation(snapshot: RiskPolicyEvaluationSnapshot) -> str | None:
    if snapshot.violations:
        return snapshot.violations[0]
    for check in snapshot.checks:
        if check.status is PolicyCheckStatus.violation:
            return check.name
    return None


def build_policy_violation_alert(
    snapshot: RiskPolicyEvaluationSnapshot,
    *,
    severity: str = "critical",
    runbook: str | None = RISK_POLICY_VIOLATION_RUNBOOK,
    generated_at: datetime | None = None,
) -> RiskPolicyViolationAlert:
    """Create a structured alert for policy violations."""

    timestamp = generated_at or datetime.now(timezone.utc)
    return RiskPolicyViolationAlert(
        snapshot=snapshot,
        severity=severity,
        runbook=runbook,
        generated_at=timestamp,
    )


def format_policy_violation_markdown(alert: RiskPolicyViolationAlert) -> str:
    snapshot = alert.snapshot
    lines = [
        "🚨 **Policy violation detected**",
        f"**Symbol:** {snapshot.symbol}",
        f"**Approved:** {'YES' if snapshot.approved else 'NO'}",
    ]
    primary = _first_violation(snapshot)
    if primary:
        lines.append(f"**Primary violation:** {primary}")
    if snapshot.reason:
        lines.append(f"**Reason:** {snapshot.reason}")
    if snapshot.policy_limits:
        limit_parts = [f"{key}={value}" for key, value in snapshot.policy_limits.items()]
        lines.append("**Limits:** " + ", ".join(limit_parts))
    if snapshot.metadata:
        equity = _coerce_float(snapshot.metadata.get("equity"))
        projected = _coerce_float(snapshot.metadata.get("projected_total_exposure"))
        if equity is not None or projected is not None:
            lines.append(
                "**Exposure:** equity={:,.2f} projected={:,.2f}".format(
                    equity if equity is not None else 0.0,
                    projected if projected is not None else 0.0,
                )
            )
    if alert.runbook:
        lines.append(f"**Runbook:** {alert.runbook}")
    return "\n".join(lines)


async def publish_policy_violation(
    event_bus: EventBus,
    alert: RiskPolicyViolationAlert,
    *,
    source: str = "risk_gateway",
) -> None:
    """Publish a policy violation alert on the event bus."""

    payload = alert.as_dict()
    payload["markdown"] = format_policy_violation_markdown(alert)
    event = Event(type="telemetry.risk.policy_violation", payload=payload, source=source)
    await event_bus.publish(event)


__all__ = [
    "PolicyCheckStatus",
    "RiskPolicyCheckSnapshot",
    "RiskPolicyEvaluationSnapshot",
    "RiskPolicyViolationAlert",
    "build_policy_snapshot",
    "build_policy_violation_alert",
    "format_policy_markdown",
    "format_policy_violation_markdown",
    "publish_policy_snapshot",
    "publish_policy_violation",
    "RISK_POLICY_VIOLATION_RUNBOOK",
]
