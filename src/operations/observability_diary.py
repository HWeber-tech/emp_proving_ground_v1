from __future__ import annotations

"""Decision narration capsules aligned with the observability diary schema."""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Iterable, Mapping, MutableMapping, Sequence

from src.core.event_bus import Event, EventBus, TopicBus
from src.operations.event_bus_failover import publish_event_with_failover

__all__ = [
    "PolicyLedgerDiff",
    "SigmaStabilitySnapshot",
    "ThrottleStateSnapshot",
    "DecisionNarrationCapsule",
    "build_decision_narration_capsule",
    "publish_decision_narration_capsule",
]


logger = logging.getLogger(__name__)
UTC = timezone.utc


def _normalise_float(value: object | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        candidate = value.strip()
        if not candidate:
            return None
        try:
            return float(candidate)
        except ValueError:
            return None
    return None


def _normalise_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"false", "0", "no", "off"}:
            return False
        if lowered in {"true", "1", "yes", "on"}:
            return True
    return bool(value)


def _normalise_datetime(value: object | None) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.astimezone(UTC) if value.tzinfo else value.replace(tzinfo=UTC)
    if isinstance(value, str):
        candidate = value.strip()
        if not candidate:
            return None
        try:
            parsed = datetime.fromisoformat(candidate)
        except ValueError:
            return None
        return parsed.astimezone(UTC) if parsed.tzinfo else parsed.replace(tzinfo=UTC)
    return None


def _normalise_mapping(value: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    return {str(key): inner for key, inner in value.items()}


def _normalise_notes(notes: Iterable[object] | None) -> tuple[str, ...]:
    if notes is None:
        return ()
    cleaned: list[str] = []
    for note in notes:
        if note is None:
            continue
        text = str(note).strip()
        if text:
            cleaned.append(text)
    return tuple(cleaned)


@dataclass(frozen=True)
class PolicyLedgerDiff:
    """Diff entry capturing a policy ledger transition."""

    policy_id: str
    change_type: str
    before: Mapping[str, Any] | None = None
    after: Mapping[str, Any] | None = None
    approvals: tuple[str, ...] = field(default_factory=tuple)
    notes: tuple[str, ...] = field(default_factory=tuple)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Mapping[str, Any]:
        payload: MutableMapping[str, Any] = {
            "policy_id": self.policy_id,
            "change_type": self.change_type,
            "approvals": list(self.approvals),
            "notes": list(self.notes),
            "metadata": dict(self.metadata),
        }
        if self.before is not None:
            payload["before"] = dict(self.before)
        if self.after is not None:
            payload["after"] = dict(self.after)
        return payload


def _build_policy_ledger_diff(entry: Mapping[str, Any] | PolicyLedgerDiff) -> PolicyLedgerDiff:
    if isinstance(entry, PolicyLedgerDiff):
        return entry
    policy_id = str(entry.get("policy_id", "unknown"))
    change_type = str(entry.get("change_type", "updated"))
    before = entry.get("before")
    after = entry.get("after")
    approvals: Sequence[object] = entry.get("approvals", ())  # type: ignore[assignment]
    notes: Sequence[object] = entry.get("notes", ())  # type: ignore[assignment]
    metadata = _normalise_mapping(entry.get("metadata"))

    return PolicyLedgerDiff(
        policy_id=policy_id,
        change_type=change_type,
        before=dict(before) if isinstance(before, Mapping) else None,
        after=dict(after) if isinstance(after, Mapping) else None,
        approvals=tuple(str(approval).strip() for approval in approvals if str(approval).strip()),
        notes=_normalise_notes(notes),
        metadata=metadata,
    )


@dataclass(frozen=True)
class SigmaStabilitySnapshot:
    """Telemetry describing sigma stability during a change window."""

    symbol: str
    sigma_before: float | None
    sigma_after: float | None
    sigma_target: float | None
    stability_index: float | None
    delta: float | None
    metadata: Mapping[str, Any]

    def as_dict(self) -> Mapping[str, Any]:
        return {
            "symbol": self.symbol,
            "sigma_before": self.sigma_before,
            "sigma_after": self.sigma_after,
            "sigma_target": self.sigma_target,
            "stability_index": self.stability_index,
            "delta": self.delta,
            "metadata": dict(self.metadata),
        }


def _build_sigma_stability_snapshot(payload: Mapping[str, Any]) -> SigmaStabilitySnapshot:
    mapping = _normalise_mapping(payload)
    symbol = str(mapping.get("symbol", "UNKNOWN"))
    sigma_before = _normalise_float(mapping.get("sigma_before"))
    sigma_after = _normalise_float(mapping.get("sigma_after"))
    sigma_target = _normalise_float(mapping.get("sigma_target"))
    stability_index = _normalise_float(mapping.get("stability_index"))
    delta: float | None = None
    if sigma_before is not None and sigma_after is not None:
        delta = sigma_after - sigma_before
    metadata = {key: value for key, value in mapping.items() if key not in {
        "symbol",
        "sigma_before",
        "sigma_after",
        "sigma_target",
        "stability_index",
    }}
    return SigmaStabilitySnapshot(
        symbol=symbol,
        sigma_before=sigma_before,
        sigma_after=sigma_after,
        sigma_target=sigma_target,
        stability_index=stability_index,
        delta=delta,
        metadata=metadata,
    )


@dataclass(frozen=True)
class ThrottleStateSnapshot:
    """Capture of a throttle control during a change window."""

    name: str
    state: str
    active: bool
    multiplier: float | None
    reason: str | None
    metadata: Mapping[str, Any]

    def as_dict(self) -> Mapping[str, Any]:
        payload: MutableMapping[str, Any] = {
            "name": self.name,
            "state": self.state,
            "active": self.active,
            "metadata": dict(self.metadata),
        }
        if self.multiplier is not None:
            payload["multiplier"] = self.multiplier
        if self.reason:
            payload["reason"] = self.reason
        return payload


def _build_throttle_state(entry: Mapping[str, Any] | ThrottleStateSnapshot) -> ThrottleStateSnapshot:
    if isinstance(entry, ThrottleStateSnapshot):
        return entry
    name = str(entry.get("name", "unknown"))
    state = str(entry.get("state", "unknown"))
    multiplier = _normalise_float(entry.get("multiplier"))
    reason = entry.get("reason")
    reason_text = str(reason).strip() if isinstance(reason, str) else None
    metadata = _normalise_mapping(entry.get("metadata"))
    active = _normalise_bool(entry.get("active", False))

    return ThrottleStateSnapshot(
        name=name,
        state=state,
        active=active,
        multiplier=multiplier,
        reason=reason_text if reason_text else None,
        metadata=metadata,
    )


@dataclass(frozen=True)
class DecisionNarrationCapsule:
    """Aggregated narration for an AlphaTrade change window."""

    capsule_id: str
    generated_at: datetime
    window_start: datetime | None
    window_end: datetime | None
    policy_diffs: tuple[PolicyLedgerDiff, ...]
    sigma_stability: SigmaStabilitySnapshot
    throttle_states: tuple[ThrottleStateSnapshot, ...]
    notes: tuple[str, ...]
    metadata: Mapping[str, Any]

    def as_dict(self) -> Mapping[str, Any]:
        return {
            "capsule_id": self.capsule_id,
            "generated_at": self.generated_at.astimezone(UTC).isoformat(),
            "window_start": self.window_start.isoformat() if self.window_start else None,
            "window_end": self.window_end.isoformat() if self.window_end else None,
            "policy_diffs": [diff.as_dict() for diff in self.policy_diffs],
            "sigma_stability": dict(self.sigma_stability.as_dict()),
            "throttle_states": [state.as_dict() for state in self.throttle_states],
            "notes": list(self.notes),
            "metadata": dict(self.metadata),
        }

    def to_markdown(self) -> str:
        lines = [
            f"# Decision narration capsule — {self.capsule_id}",
            f"- Generated at: {self.generated_at.astimezone(UTC).isoformat()}",
        ]
        if self.window_start or self.window_end:
            start = self.window_start.astimezone(UTC).isoformat() if self.window_start else "?"
            end = self.window_end.astimezone(UTC).isoformat() if self.window_end else "?"
            lines.append(f"- Change window: {start} → {end}")
        if self.notes:
            lines.append("- Notes: " + "; ".join(self.notes))
        lines.append("")

        lines.append("## Policy ledger diffs")
        if self.policy_diffs:
            lines.append("| Policy | Change | Approvals | Notes |")
            lines.append("| --- | --- | --- | --- |")
            for diff in self.policy_diffs:
                approvals = ", ".join(diff.approvals) if diff.approvals else "—"
                notes = "; ".join(diff.notes) if diff.notes else "—"
                lines.append(
                    f"| {diff.policy_id} | {diff.change_type} | {approvals or '—'} | {notes or '—'} |"
                )
        else:
            lines.append("No policy ledger changes recorded.")
        lines.append("")

        stability = self.sigma_stability
        lines.append("## Sigma stability")
        parts: list[str] = [f"symbol={stability.symbol}"]
        if stability.sigma_before is not None:
            parts.append(f"before={stability.sigma_before:.4f}")
        if stability.sigma_after is not None:
            parts.append(f"after={stability.sigma_after:.4f}")
        if stability.delta is not None:
            parts.append(f"delta={stability.delta:+.4f}")
        if stability.sigma_target is not None:
            parts.append(f"target={stability.sigma_target:.4f}")
        if stability.stability_index is not None:
            parts.append(f"stability_index={stability.stability_index:.4f}")
        lines.append("- " + ", ".join(parts))
        if stability.metadata:
            lines.append("- Metadata: " + ", ".join(f"{k}={v}" for k, v in stability.metadata.items()))
        lines.append("")

        lines.append("## Throttle states")
        if self.throttle_states:
            lines.append("| Throttle | State | Active | Multiplier | Reason |")
            lines.append("| --- | --- | --- | --- | --- |")
            for state in self.throttle_states:
                multiplier = f"{state.multiplier:.2f}" if state.multiplier is not None else "—"
                reason = state.reason or "—"
                lines.append(
                    f"| {state.name} | {state.state} | {'yes' if state.active else 'no'} | {multiplier} | {reason} |"
                )
        else:
            lines.append("No throttle updates recorded.")

        if self.metadata:
            lines.append("")
            lines.append("## Metadata")
            for key, value in sorted(self.metadata.items()):
                lines.append(f"- **{key}**: {value}")

        return "\n".join(lines)


def build_decision_narration_capsule(
    *,
    capsule_id: str,
    window_start: datetime | str | None,
    window_end: datetime | str | None,
    policy_diffs: Iterable[Mapping[str, Any] | PolicyLedgerDiff],
    sigma_metrics: Mapping[str, Any],
    throttle_states: Iterable[Mapping[str, Any] | ThrottleStateSnapshot],
    notes: Iterable[object] | None = None,
    metadata: Mapping[str, Any] | None = None,
    generated_at: datetime | None = None,
) -> DecisionNarrationCapsule:
    generated = generated_at or datetime.now(tz=UTC)
    window_start_dt = _normalise_datetime(window_start)
    window_end_dt = _normalise_datetime(window_end)

    ledger_diffs = tuple(_build_policy_ledger_diff(diff) for diff in policy_diffs)
    sigma_snapshot = _build_sigma_stability_snapshot(sigma_metrics)
    throttle_snapshot = tuple(_build_throttle_state(state) for state in throttle_states)
    note_entries = _normalise_notes(notes)
    metadata_dict = _normalise_mapping(metadata)

    return DecisionNarrationCapsule(
        capsule_id=str(capsule_id),
        generated_at=generated,
        window_start=window_start_dt,
        window_end=window_end_dt,
        policy_diffs=ledger_diffs,
        sigma_stability=sigma_snapshot,
        throttle_states=throttle_snapshot,
        notes=note_entries,
        metadata=metadata_dict,
    )


def publish_decision_narration_capsule(
    event_bus: EventBus,
    capsule: DecisionNarrationCapsule,
    *,
    source: str = "observability.diary",
    event_type: str = "observability.decision_narration",
    global_bus_factory: Callable[[], TopicBus] | None = None,
) -> None:
    payload = {
        "capsule": capsule.as_dict(),
        "markdown": capsule.to_markdown(),
        "version": 1,
    }
    event = Event(type=event_type, payload=payload, source=source)

    publish_event_with_failover(
        event_bus,
        event,
        logger=logger,
        runtime_fallback_message="Runtime event bus unavailable, falling back to global bus",
        runtime_unexpected_message="Unexpected runtime bus error while publishing decision narration",
        runtime_none_message="Runtime bus returned no subscribers for decision narration capsule",
        global_not_running_message="Global bus unavailable for decision narration capsule",
        global_unexpected_message="Unexpected global bus error while publishing decision narration",
        global_bus_factory=global_bus_factory,
    )
