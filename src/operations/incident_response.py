"""Incident response readiness evaluation and telemetry helpers."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Mapping, MutableMapping, Sequence

from src.core.event_bus import Event, EventBus, get_global_bus

logger = logging.getLogger(__name__)


class IncidentResponseStatus(StrEnum):
    """Severity grading exposed by the incident response snapshot."""

    ok = "ok"
    warn = "warn"
    fail = "fail"


_STATUS_ORDER: Mapping[IncidentResponseStatus, int] = {
    IncidentResponseStatus.ok: 0,
    IncidentResponseStatus.warn: 1,
    IncidentResponseStatus.fail: 2,
}


def _escalate(
    current: IncidentResponseStatus, candidate: IncidentResponseStatus
) -> IncidentResponseStatus:
    if _STATUS_ORDER[candidate] > _STATUS_ORDER[current]:
        return candidate
    return current


def _coerce_tuple(value: object | None) -> tuple[str, ...]:
    if value is None:
        return tuple()
    if isinstance(value, str):
        parts = [part.strip() for part in value.replace(";", ",").split(",")]
        return tuple(part for part in parts if part)
    if isinstance(value, Sequence):
        items: list[str] = []
        for entry in value:
            if entry is None:
                continue
            text = str(entry).strip()
            if text:
                items.append(text)
        return tuple(items)
    return tuple()


def _coerce_int(value: object | None, default: int) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if value is None:
        return default
    try:
        return int(float(str(value).strip()))
    except (ValueError, TypeError):
        return default


def _coerce_float(value: object | None, default: float | None = None) -> float | None:
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).strip())
    except (ValueError, TypeError):
        return default


def _coerce_bool(value: object | None, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return default


@dataclass(frozen=True)
class IncidentResponsePolicy:
    """Declarative policy describing the desired incident response posture."""

    required_runbooks: tuple[str, ...] = tuple()
    training_interval_days: int = 90
    drill_interval_days: int = 60
    minimum_primary_responders: int = 1
    minimum_secondary_responders: int = 1
    postmortem_sla_hours: float = 24.0
    maximum_open_incidents: int = 1
    require_chatops: bool = True

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, object] | None) -> "IncidentResponsePolicy":
        mapping = mapping or {}
        required_runbooks = _coerce_tuple(mapping.get("INCIDENT_REQUIRED_RUNBOOKS"))
        training_interval = _coerce_int(mapping.get("INCIDENT_TRAINING_INTERVAL_DAYS"), 90)
        drill_interval = _coerce_int(mapping.get("INCIDENT_DRILL_INTERVAL_DAYS"), 60)
        minimum_primary = max(_coerce_int(mapping.get("INCIDENT_MIN_PRIMARY_RESPONDERS"), 1), 0)
        minimum_secondary = max(_coerce_int(mapping.get("INCIDENT_MIN_SECONDARY_RESPONDERS"), 1), 0)
        postmortem_sla = _coerce_float(mapping.get("INCIDENT_POSTMORTEM_SLA_HOURS"), 24.0)
        max_open = max(_coerce_int(mapping.get("INCIDENT_MAX_OPEN_INCIDENTS"), 1), 0)
        require_chatops = _coerce_bool(mapping.get("INCIDENT_REQUIRE_CHATOPS"), True)

        return cls(
            required_runbooks=required_runbooks,
            training_interval_days=max(training_interval, 0),
            drill_interval_days=max(drill_interval, 0),
            minimum_primary_responders=minimum_primary,
            minimum_secondary_responders=minimum_secondary,
            postmortem_sla_hours=float(postmortem_sla or 0.0),
            maximum_open_incidents=max_open,
            require_chatops=require_chatops,
        )


@dataclass(frozen=True)
class IncidentResponseState:
    """Observed state used to evaluate incident readiness."""

    available_runbooks: tuple[str, ...] = tuple()
    training_age_days: float | None = None
    drill_age_days: float | None = None
    primary_oncall: tuple[str, ...] = tuple()
    secondary_oncall: tuple[str, ...] = tuple()
    open_incidents: tuple[str, ...] = tuple()
    postmortem_backlog_hours: float | None = None
    chatops_ready: bool = False
    last_major_incident_age_days: float | None = None

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, object] | None) -> "IncidentResponseState":
        mapping = mapping or {}
        return cls(
            available_runbooks=_coerce_tuple(mapping.get("INCIDENT_AVAILABLE_RUNBOOKS")),
            training_age_days=_coerce_float(mapping.get("INCIDENT_TRAINING_AGE_DAYS")),
            drill_age_days=_coerce_float(mapping.get("INCIDENT_DRILL_AGE_DAYS")),
            primary_oncall=_coerce_tuple(mapping.get("INCIDENT_PRIMARY_RESPONDERS")),
            secondary_oncall=_coerce_tuple(mapping.get("INCIDENT_SECONDARY_RESPONDERS")),
            open_incidents=_coerce_tuple(mapping.get("INCIDENT_OPEN_INCIDENTS")),
            postmortem_backlog_hours=_coerce_float(
                mapping.get("INCIDENT_POSTMORTEM_BACKLOG_HOURS")
            ),
            chatops_ready=_coerce_bool(mapping.get("INCIDENT_CHATOPS_READY"), False),
            last_major_incident_age_days=_coerce_float(mapping.get("INCIDENT_LAST_MAJOR_AGE_DAYS")),
        )


@dataclass(frozen=True)
class IncidentResponseSnapshot:
    """Aggregated incident response readiness snapshot."""

    service: str
    generated_at: datetime
    status: IncidentResponseStatus
    missing_runbooks: tuple[str, ...]
    training_age_days: float | None
    drill_age_days: float | None
    primary_oncall: tuple[str, ...]
    secondary_oncall: tuple[str, ...]
    open_incidents: tuple[str, ...]
    issues: tuple[str, ...] = field(default_factory=tuple)
    metadata: Mapping[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "service": self.service,
            "generated_at": self.generated_at.isoformat(),
            "status": self.status.value,
            "missing_runbooks": list(self.missing_runbooks),
            "training_age_days": self.training_age_days,
            "drill_age_days": self.drill_age_days,
            "primary_oncall": list(self.primary_oncall),
            "secondary_oncall": list(self.secondary_oncall),
            "open_incidents": list(self.open_incidents),
            "issues": list(self.issues),
            "metadata": dict(self.metadata),
        }
        return payload

    def to_markdown(self) -> str:
        lines = [
            f"**Incident response – {self.service}**",
            f"- Status: {self.status.value.upper()}",
            f"- Generated: {self.generated_at.isoformat()}",
            f"- Primary responders: {len(self.primary_oncall)}",
            f"- Secondary responders: {len(self.secondary_oncall)}",
            f"- Open incidents: {len(self.open_incidents)}",
        ]
        if self.training_age_days is not None:
            lines.append(f"- Training age (days): {self.training_age_days:.1f}")
        if self.drill_age_days is not None:
            lines.append(f"- Drill age (days): {self.drill_age_days:.1f}")
        if self.missing_runbooks:
            lines.append("- Missing runbooks: " + ", ".join(sorted(self.missing_runbooks)))
        else:
            lines.append("- Missing runbooks: none")
        backlog = self.metadata.get("postmortem_backlog_hours")
        if isinstance(backlog, (int, float)):
            lines.append(f"- Postmortem backlog (hours): {backlog:.1f}")
        context = self.metadata.get("context")
        if isinstance(context, Mapping):
            lines.append("")
            lines.append("**Context:**")
            for key, value in sorted(context.items()):
                lines.append(f"- {key}: {value}")
        if self.issues:
            lines.append("")
            lines.append("**Issues:**")
            for issue in self.issues:
                lines.append(f"- {issue}")
        return "\n".join(lines)


def format_incident_response_markdown(snapshot: IncidentResponseSnapshot) -> str:
    """Return a Markdown summary of the incident response snapshot."""

    return snapshot.to_markdown()


def evaluate_incident_response(
    policy: IncidentResponsePolicy,
    state: IncidentResponseState,
    *,
    service: str = "emp_incidents",
    now: datetime | None = None,
    metadata: Mapping[str, object] | None = None,
) -> IncidentResponseSnapshot:
    """Grade the current incident response posture against the policy."""

    moment = now or datetime.now(tz=UTC)
    status = IncidentResponseStatus.ok
    issues: list[str] = []

    available = {runbook.lower(): runbook for runbook in state.available_runbooks}
    missing: list[str] = []
    for runbook in policy.required_runbooks:
        if runbook.lower() not in available:
            missing.append(runbook)
    if missing:
        status = _escalate(status, IncidentResponseStatus.fail)
        issues.append("Missing required runbooks: " + ", ".join(sorted(missing)))

    training_age = state.training_age_days
    if training_age is None:
        status = _escalate(status, IncidentResponseStatus.warn)
        issues.append("No incident response training recorded")
    else:
        if training_age > policy.training_interval_days * 2:
            status = _escalate(status, IncidentResponseStatus.fail)
            issues.append("Incident response training overdue by more than 2x the interval")
        elif training_age > policy.training_interval_days:
            status = _escalate(status, IncidentResponseStatus.warn)
            issues.append("Incident response training overdue")

    drill_age = state.drill_age_days
    if drill_age is None:
        status = _escalate(status, IncidentResponseStatus.warn)
        issues.append("No incident response drill recorded")
    else:
        if drill_age > policy.drill_interval_days * 2:
            status = _escalate(status, IncidentResponseStatus.fail)
            issues.append("Incident drill cadence has lapsed significantly")
        elif drill_age > policy.drill_interval_days:
            status = _escalate(status, IncidentResponseStatus.warn)
            issues.append("Incident drill overdue")

    primary_count = len(state.primary_oncall)
    if primary_count < policy.minimum_primary_responders:
        status = _escalate(status, IncidentResponseStatus.fail)
        issues.append("Primary on-call roster below minimum required responders")

    secondary_count = len(state.secondary_oncall)
    if secondary_count < policy.minimum_secondary_responders:
        status = _escalate(status, IncidentResponseStatus.warn)
        issues.append("Secondary on-call roster below recommended responders")

    open_count = len(state.open_incidents)
    if open_count > policy.maximum_open_incidents:
        status = _escalate(status, IncidentResponseStatus.fail)
        issues.append(
            f"{open_count} open incidents exceeds threshold of {policy.maximum_open_incidents}"
        )
    elif open_count and open_count == policy.maximum_open_incidents:
        status = _escalate(status, IncidentResponseStatus.warn)
        issues.append("Incident volume approaching the configured threshold")

    backlog = state.postmortem_backlog_hours
    if backlog is not None and backlog > policy.postmortem_sla_hours:
        severity = (
            IncidentResponseStatus.fail
            if backlog > policy.postmortem_sla_hours * 2
            else IncidentResponseStatus.warn
        )
        status = _escalate(status, severity)
        issues.append("Postmortem backlog exceeds the configured SLA")

    if policy.require_chatops and not state.chatops_ready:
        status = _escalate(status, IncidentResponseStatus.warn)
        issues.append("ChatOps automations disabled or unavailable")

    combined_metadata: MutableMapping[str, object] = {
        "policy": {
            "required_runbooks": list(policy.required_runbooks),
            "training_interval_days": policy.training_interval_days,
            "drill_interval_days": policy.drill_interval_days,
            "minimum_primary_responders": policy.minimum_primary_responders,
            "minimum_secondary_responders": policy.minimum_secondary_responders,
            "postmortem_sla_hours": policy.postmortem_sla_hours,
            "maximum_open_incidents": policy.maximum_open_incidents,
            "require_chatops": policy.require_chatops,
        },
        "postmortem_backlog_hours": backlog if backlog is not None else 0.0,
    }
    if state.last_major_incident_age_days is not None:
        combined_metadata["last_major_incident_age_days"] = state.last_major_incident_age_days
    if metadata:
        combined_metadata["context"] = dict(metadata)

    return IncidentResponseSnapshot(
        service=service,
        generated_at=moment,
        status=status,
        missing_runbooks=tuple(sorted(missing)),
        training_age_days=training_age,
        drill_age_days=drill_age,
        primary_oncall=state.primary_oncall,
        secondary_oncall=state.secondary_oncall,
        open_incidents=state.open_incidents,
        issues=tuple(issues),
        metadata=combined_metadata,
    )


def publish_incident_response_snapshot(
    event_bus: EventBus, snapshot: IncidentResponseSnapshot
) -> None:
    """Publish the incident response snapshot onto the runtime event bus."""

    event = Event(
        type="telemetry.operational.incident_response",
        payload=snapshot.as_dict(),
        source="operations.incident_response",
    )

    publish_from_sync = getattr(event_bus, "publish_from_sync", None)
    if callable(publish_from_sync) and event_bus.is_running():
        try:
            publish_from_sync(event)
            return
        except Exception:  # pragma: no cover - defensive publish fallback
            logger.debug(
                "Failed to publish incident response snapshot via runtime bus",
                exc_info=True,
            )

    try:
        topic_bus = get_global_bus()
        topic_bus.publish_sync(event.type, event.payload, source=event.source)
    except Exception:  # pragma: no cover - diagnostics only
        logger.debug("Incident response telemetry publish skipped", exc_info=True)


__all__ = [
    "IncidentResponsePolicy",
    "IncidentResponseSnapshot",
    "IncidentResponseState",
    "IncidentResponseStatus",
    "evaluate_incident_response",
    "format_incident_response_markdown",
    "publish_incident_response_snapshot",
]
