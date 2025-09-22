"""Telemetry helpers for :mod:`src.trading.risk.risk_policy`."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from typing import Mapping, MutableMapping, Sequence

from src.core.event_bus import Event, EventBus

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
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_mapping(value: object) -> Mapping[str, object]:
    if isinstance(value, Mapping):
        return value
    if isinstance(value, Sequence):
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
        metadata: MutableMapping[str, object] = {}
        extra = entry.get("extra")
        if isinstance(extra, Mapping):
            metadata.update(extra)
        for key in ("ratio", "resolved_price", "projected_total_exposure"):
            if key in entry and key not in metadata:
                metadata[key] = entry[key]  # type: ignore[index]
        checks.append(
            RiskPolicyCheckSnapshot(
                name=str(entry.get("name", "unknown")),
                status=_normalise_status(entry.get("status")),
                value=_coerce_float(entry.get("value")),
                threshold=_coerce_float(entry.get("threshold")),
                metadata=dict(metadata) if metadata else {},
            )
        )

    limits = policy.limit_snapshot() if policy is not None else {}
    metadata = dict(decision.metadata)
    symbol = str(metadata.get("symbol", "UNKNOWN"))
    research_mode = bool(metadata.get("research_mode", getattr(policy, "research_mode", False)))

    return RiskPolicyEvaluationSnapshot(
        symbol=symbol,
        approved=decision.approved,
        reason=decision.reason,
        generated_at=timestamp,
        checks=tuple(checks),
        policy_limits=limits,
        metadata=metadata,
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
        equity = snapshot.metadata.get("equity")
        projected = snapshot.metadata.get("projected_total_exposure")
        if equity is not None or projected is not None:
            lines.append(
                "**Exposure:** equity={:,.2f} projected={:,.2f}".format(
                    float(equity or 0.0), float(projected or 0.0)
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


__all__ = [
    "PolicyCheckStatus",
    "RiskPolicyCheckSnapshot",
    "RiskPolicyEvaluationSnapshot",
    "build_policy_snapshot",
    "format_policy_markdown",
    "publish_policy_snapshot",
]
