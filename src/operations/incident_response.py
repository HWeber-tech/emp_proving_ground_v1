"""Incident response readiness evaluation and telemetry helpers."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Mapping, MutableMapping, Sequence

from src.core.event_bus import Event, EventBus
from src.operations.alerts import (
    AlertDispatchResult,
    AlertEvent,
    AlertManager,
    AlertSeverity,
)
from src.operations.event_bus_failover import publish_event_with_failover

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


_SEVERITY_MAP: Mapping[IncidentResponseStatus, AlertSeverity] = {
    IncidentResponseStatus.ok: AlertSeverity.info,
    IncidentResponseStatus.warn: AlertSeverity.warning,
    IncidentResponseStatus.fail: AlertSeverity.critical,
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
    mtta_warn_minutes: float | None = 30.0
    mtta_fail_minutes: float | None = 60.0
    mttr_warn_minutes: float | None = 180.0
    mttr_fail_minutes: float | None = 360.0
    metrics_stale_warn_hours: float | None = 24.0
    metrics_stale_fail_hours: float | None = 48.0

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
        mtta_warn = _coerce_float(mapping.get("INCIDENT_MTTA_WARN_MINUTES"), 30.0)
        mtta_fail = _coerce_float(mapping.get("INCIDENT_MTTA_FAIL_MINUTES"), 60.0)
        mttr_warn = _coerce_float(mapping.get("INCIDENT_MTTR_WARN_MINUTES"), 180.0)
        mttr_fail = _coerce_float(mapping.get("INCIDENT_MTTR_FAIL_MINUTES"), 360.0)
        stale_warn = _coerce_float(mapping.get("INCIDENT_METRICS_STALE_WARN_HOURS"), 24.0)
        stale_fail = _coerce_float(mapping.get("INCIDENT_METRICS_STALE_FAIL_HOURS"), 48.0)

        return cls(
            required_runbooks=required_runbooks,
            training_interval_days=max(training_interval, 0),
            drill_interval_days=max(drill_interval, 0),
            minimum_primary_responders=minimum_primary,
            minimum_secondary_responders=minimum_secondary,
            postmortem_sla_hours=float(postmortem_sla or 0.0),
            maximum_open_incidents=max_open,
            require_chatops=require_chatops,
            mtta_warn_minutes=(mtta_warn if mtta_warn and mtta_warn > 0 else None),
            mtta_fail_minutes=(mtta_fail if mtta_fail and mtta_fail > 0 else None),
            mttr_warn_minutes=(mttr_warn if mttr_warn and mttr_warn > 0 else None),
            mttr_fail_minutes=(mttr_fail if mttr_fail and mttr_fail > 0 else None),
            metrics_stale_warn_hours=(
                stale_warn if stale_warn and stale_warn > 0 else None
            ),
            metrics_stale_fail_hours=(
                stale_fail if stale_fail and stale_fail > 0 else None
            ),
        )


@dataclass(frozen=True)
class IncidentResponseMetrics:
    """Observed MTTA/MTTR metrics captured from alert timelines."""

    mtta_minutes: float | None = None
    mttr_minutes: float | None = None
    acknowledged_incidents: int | None = None
    resolved_incidents: int | None = None
    sample_window_days: float | None = None
    data_age_hours: float | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "mtta_minutes": self.mtta_minutes,
            "mttr_minutes": self.mttr_minutes,
            "acknowledged_incidents": self.acknowledged_incidents,
            "resolved_incidents": self.resolved_incidents,
            "sample_window_days": self.sample_window_days,
            "data_age_hours": self.data_age_hours,
        }

    def has_signal(self) -> bool:
        return any(
            value not in (None, 0, 0.0)
            for value in (
                self.mtta_minutes,
                self.mttr_minutes,
                self.acknowledged_incidents,
                self.resolved_incidents,
            )
        )

    @classmethod
    def from_mapping(
        cls, mapping: Mapping[str, object] | None
    ) -> "IncidentResponseMetrics | None":
        if not mapping:
            return None

        mtta = _coerce_float(mapping.get("INCIDENT_METRICS_MTTA_MINUTES"))
        mttr = _coerce_float(mapping.get("INCIDENT_METRICS_MTTR_MINUTES"))
        acknowledged = _coerce_int(mapping.get("INCIDENT_METRICS_ACKNOWLEDGED"), 0)
        resolved = _coerce_int(mapping.get("INCIDENT_METRICS_RESOLVED"), 0)
        window_days = _coerce_float(mapping.get("INCIDENT_METRICS_SAMPLE_WINDOW_DAYS"))
        age_hours = _coerce_float(mapping.get("INCIDENT_METRICS_DATA_AGE_HOURS"))

        metrics = cls(
            mtta_minutes=mtta,
            mttr_minutes=mttr,
            acknowledged_incidents=acknowledged if acknowledged else None,
            resolved_incidents=resolved if resolved else None,
            sample_window_days=window_days,
            data_age_hours=age_hours,
        )

        if metrics.has_signal():
            return metrics
        return None


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
    metrics: IncidentResponseMetrics | None = None

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
            metrics=IncidentResponseMetrics.from_mapping(mapping),
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
    metrics: IncidentResponseMetrics | None = None

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
        if self.metrics is not None:
            payload["metrics"] = self.metrics.as_dict()
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
        metrics = self.metrics
        if metrics is None:
            metrics_payload = self.metadata.get("reliability_metrics")
            if isinstance(metrics_payload, Mapping):
                metrics = IncidentResponseMetrics.from_mapping(
                    {
                        "INCIDENT_METRICS_MTTA_MINUTES": metrics_payload.get("mtta_minutes"),
                        "INCIDENT_METRICS_MTTR_MINUTES": metrics_payload.get("mttr_minutes"),
                        "INCIDENT_METRICS_ACKNOWLEDGED": metrics_payload.get("acknowledged_incidents"),
                        "INCIDENT_METRICS_RESOLVED": metrics_payload.get("resolved_incidents"),
                        "INCIDENT_METRICS_SAMPLE_WINDOW_DAYS": metrics_payload.get("sample_window_days"),
                        "INCIDENT_METRICS_DATA_AGE_HOURS": metrics_payload.get("data_age_hours"),
                    }
                )
        if metrics is not None and metrics.has_signal():
            lines.append("")
            lines.append("**Reliability metrics:**")
            if metrics.mtta_minutes is not None:
                lines.append(f"- MTTA (minutes): {metrics.mtta_minutes:.1f}")
            if metrics.mttr_minutes is not None:
                lines.append(f"- MTTR (minutes): {metrics.mttr_minutes:.1f}")
            if metrics.acknowledged_incidents is not None:
                lines.append(
                    f"- Incidents acknowledged: {metrics.acknowledged_incidents}"
                )
            if metrics.resolved_incidents is not None:
                lines.append(f"- Incidents resolved: {metrics.resolved_incidents}")
            if metrics.sample_window_days is not None:
                lines.append(f"- Metrics window (days): {metrics.sample_window_days:.1f}")
            if metrics.data_age_hours is not None:
                lines.append(f"- Metrics age (hours): {metrics.data_age_hours:.1f}")
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
    issue_details: list[dict[str, object]] = []
    issue_catalog: dict[str, list[dict[str, object]]] = {}
    issue_counts: dict[str, int] = {}

    def _record_issue(
        category: str,
        severity: IncidentResponseStatus,
        message: str,
        *,
        detail: Mapping[str, object] | None = None,
    ) -> None:
        """Track an issue, escalating status and metadata consistently."""

        nonlocal status
        status = _escalate(status, severity)
        issues.append(message)
        entry = {
            "category": category,
            "severity": severity.value,
            "message": message,
        }
        if detail:
            entry["detail"] = dict(detail)
        issue_details.append(entry)
        issue_catalog.setdefault(category, []).append(entry)
        issue_counts[severity.value] = issue_counts.get(severity.value, 0) + 1

    available = {runbook.lower(): runbook for runbook in state.available_runbooks}
    missing: list[str] = []
    for runbook in policy.required_runbooks:
        if runbook.lower() not in available:
            missing.append(runbook)
    if missing:
        _record_issue(
            "missing_runbooks",
            IncidentResponseStatus.fail,
            "Missing required runbooks: " + ", ".join(sorted(missing)),
        )

    training_age = state.training_age_days
    if training_age is None:
        _record_issue(
            "training",
            IncidentResponseStatus.warn,
            "No incident response training recorded",
        )
    else:
        if training_age > policy.training_interval_days * 2:
            _record_issue(
                "training",
                IncidentResponseStatus.fail,
                "Incident response training overdue by more than 2x the interval",
            )
        elif training_age > policy.training_interval_days:
            _record_issue(
                "training",
                IncidentResponseStatus.warn,
                "Incident response training overdue",
            )

    drill_age = state.drill_age_days
    if drill_age is None:
        _record_issue(
            "drill",
            IncidentResponseStatus.warn,
            "No incident response drill recorded",
        )
    else:
        if drill_age > policy.drill_interval_days * 2:
            _record_issue(
                "drill",
                IncidentResponseStatus.fail,
                "Incident drill cadence has lapsed significantly",
            )
        elif drill_age > policy.drill_interval_days:
            _record_issue(
                "drill",
                IncidentResponseStatus.warn,
                "Incident drill overdue",
            )

    primary_count = len(state.primary_oncall)
    if primary_count < policy.minimum_primary_responders:
        _record_issue(
            "primary_roster",
            IncidentResponseStatus.fail,
            "Primary on-call roster below minimum required responders",
        )

    secondary_count = len(state.secondary_oncall)
    if secondary_count < policy.minimum_secondary_responders:
        _record_issue(
            "secondary_roster",
            IncidentResponseStatus.warn,
            "Secondary on-call roster below recommended responders",
        )

    open_count = len(state.open_incidents)
    if open_count > policy.maximum_open_incidents:
        _record_issue(
            "open_incidents",
            IncidentResponseStatus.fail,
            f"{open_count} open incidents exceeds threshold of {policy.maximum_open_incidents}",
        )
    elif open_count and open_count == policy.maximum_open_incidents:
        _record_issue(
            "open_incidents",
            IncidentResponseStatus.warn,
            "Incident volume approaching the configured threshold",
        )

    backlog = state.postmortem_backlog_hours
    if backlog is not None and backlog > policy.postmortem_sla_hours:
        severity = (
            IncidentResponseStatus.fail
            if backlog > policy.postmortem_sla_hours * 2
            else IncidentResponseStatus.warn
        )
        _record_issue(
            "postmortem_backlog",
            severity,
            "Postmortem backlog exceeds the configured SLA",
        )

    if policy.require_chatops and not state.chatops_ready:
        _record_issue(
            "chatops",
            IncidentResponseStatus.warn,
            "ChatOps automations disabled or unavailable",
        )

    metrics = state.metrics

    if metrics is not None:

        def _metric_issue(
            *,
            value_minutes: float | None,
            warn_minutes: float | None,
            fail_minutes: float | None,
            category: str,
            label: str,
        ) -> None:
            if value_minutes is None:
                return
            threshold = None
            severity: IncidentResponseStatus | None = None
            if fail_minutes is not None and value_minutes >= fail_minutes:
                threshold = fail_minutes
                severity = IncidentResponseStatus.fail
            elif warn_minutes is not None and value_minutes >= warn_minutes:
                threshold = warn_minutes
                severity = IncidentResponseStatus.warn
            if severity is None or threshold is None:
                return
            comparison = "exceeds" if severity is IncidentResponseStatus.fail else "approaches"
            _record_issue(
                category,
                severity,
                f"{label} {value_minutes:.1f} minutes {comparison} threshold {threshold:.1f} minutes",
                detail={
                    "value_minutes": value_minutes,
                    "warn_minutes": warn_minutes,
                    "fail_minutes": fail_minutes,
                },
            )

        _metric_issue(
            value_minutes=metrics.mtta_minutes,
            warn_minutes=policy.mtta_warn_minutes,
            fail_minutes=policy.mtta_fail_minutes,
            category="metrics_mtta",
            label="MTTA",
        )
        _metric_issue(
            value_minutes=metrics.mttr_minutes,
            warn_minutes=policy.mttr_warn_minutes,
            fail_minutes=policy.mttr_fail_minutes,
            category="metrics_mttr",
            label="MTTR",
        )

        if metrics.data_age_hours is not None:
            stale_threshold = policy.metrics_stale_fail_hours
            warn_threshold = policy.metrics_stale_warn_hours
            staleness_threshold = None
            staleness_severity: IncidentResponseStatus | None = None
            if stale_threshold is not None and metrics.data_age_hours >= stale_threshold:
                staleness_threshold = stale_threshold
                staleness_severity = IncidentResponseStatus.fail
            elif warn_threshold is not None and metrics.data_age_hours >= warn_threshold:
                staleness_threshold = warn_threshold
                staleness_severity = IncidentResponseStatus.warn
            if staleness_threshold is not None and staleness_severity is not None:
                comparison = (
                    "exceeds"
                    if staleness_severity is IncidentResponseStatus.fail
                    else "approaches"
                )
                _record_issue(
                    "metrics_staleness",
                    staleness_severity,
                    (
                        f"Incident metrics age {metrics.data_age_hours:.1f} hours {comparison} "
                        f"threshold {staleness_threshold:.1f} hours"
                    ),
                    detail={
                        "age_hours": metrics.data_age_hours,
                        "warn_hours": warn_threshold,
                        "fail_hours": stale_threshold,
                    },
                )

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
            "mtta_warn_minutes": policy.mtta_warn_minutes,
            "mtta_fail_minutes": policy.mtta_fail_minutes,
            "mttr_warn_minutes": policy.mttr_warn_minutes,
            "mttr_fail_minutes": policy.mttr_fail_minutes,
            "metrics_stale_warn_hours": policy.metrics_stale_warn_hours,
            "metrics_stale_fail_hours": policy.metrics_stale_fail_hours,
        },
        "postmortem_backlog_hours": backlog if backlog is not None else 0.0,
        "primary_responders": list(state.primary_oncall),
        "secondary_responders": list(state.secondary_oncall),
        "open_incident_count": len(state.open_incidents),
        "open_incident_ids": list(state.open_incidents),
        "chatops_ready": state.chatops_ready,
    }
    if state.last_major_incident_age_days is not None:
        combined_metadata["last_major_incident_age_days"] = state.last_major_incident_age_days
    if metrics is not None and metrics.has_signal():
        combined_metadata["reliability_metrics"] = metrics.as_dict()
    if issue_details:
        combined_metadata["issue_details"] = tuple(issue_details)
        combined_metadata["issue_counts"] = dict(issue_counts)
        combined_metadata["issue_catalog"] = {
            category: tuple(entries) for category, entries in issue_catalog.items()
        }
        category_severity: dict[str, str] = {}
        for category, entries in issue_catalog.items():
            severity_rank = IncidentResponseStatus.ok
            for entry in entries:
                try:
                    entry_status = IncidentResponseStatus(entry["severity"])
                except ValueError:
                    # Skip unknown severities gracefully.
                    continue
                if _STATUS_ORDER[entry_status] > _STATUS_ORDER[severity_rank]:
                    severity_rank = entry_status
            category_severity[category] = severity_rank.value
        if category_severity:
            combined_metadata["issue_category_severity"] = category_severity
        try:
            highest_issue = max(
                (IncidentResponseStatus(severity) for severity in issue_counts),
                key=_STATUS_ORDER.__getitem__,
            )
            combined_metadata["highest_issue_severity"] = highest_issue.value
        except ValueError:
            pass
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
        metrics=metrics if metrics and metrics.has_signal() else None,
    )


def _meets_threshold(
    status: IncidentResponseStatus, threshold: IncidentResponseStatus
) -> bool:
    return _STATUS_ORDER[status] >= _STATUS_ORDER[threshold]


def _detail_severity(
    preferred: IncidentResponseStatus,
    *,
    minimum: IncidentResponseStatus = IncidentResponseStatus.warn,
) -> IncidentResponseStatus:
    if _STATUS_ORDER[preferred] >= _STATUS_ORDER[minimum]:
        return preferred
    return minimum


def derive_incident_response_alerts(
    snapshot: IncidentResponseSnapshot,
    *,
    threshold: IncidentResponseStatus = IncidentResponseStatus.warn,
    include_status_event: bool = True,
    include_detail_events: bool = True,
    base_tags: Sequence[str] = ("incident-response",),
) -> list[AlertEvent]:
    """Translate an incident response snapshot into alert events."""

    tags = tuple(base_tags)
    snapshot_payload = snapshot.as_dict()
    events: list[AlertEvent] = []

    if include_status_event and _meets_threshold(snapshot.status, threshold):
        events.append(
            AlertEvent(
                category="incident_response.status",
                severity=_SEVERITY_MAP[snapshot.status],
                message=f"Incident response status {snapshot.status.value}",
                tags=tags,
                context={"snapshot": snapshot_payload},
            )
        )

    if not include_detail_events:
        return events

    metadata = snapshot.metadata if isinstance(snapshot.metadata, Mapping) else {}
    policy = metadata.get("policy") if isinstance(metadata, Mapping) else None
    policy_mapping: Mapping[str, object] = policy if isinstance(policy, Mapping) else {}

    issue_detail_map: dict[str, Mapping[str, object]] = {}
    raw_issue_details = metadata.get("issue_details") if isinstance(metadata, Mapping) else None
    if isinstance(raw_issue_details, Sequence):
        for entry in raw_issue_details:
            if isinstance(entry, Mapping):
                message = str(entry.get("message", ""))
                if message:
                    issue_detail_map[message] = entry

    def _policy_int(key: str, default: int | None = None) -> int | None:
        if key not in policy_mapping:
            return default
        return _coerce_int(policy_mapping.get(key), default)

    def _policy_float(key: str, default: float | None = None) -> float | None:
        if key not in policy_mapping:
            return default
        return _coerce_float(policy_mapping.get(key), default)

    metrics_payload: Mapping[str, object] | None = None
    if snapshot.metrics is not None:
        metrics_payload = snapshot.metrics.as_dict()
    else:
        reliability_metrics = metadata.get("reliability_metrics")
        if isinstance(reliability_metrics, Mapping):
            metrics_payload = reliability_metrics

    def _metric_status(
        value_minutes: float | None,
        warn_minutes: float | None,
        fail_minutes: float | None,
    ) -> IncidentResponseStatus | None:
        if value_minutes is None:
            return None
        if fail_minutes is not None and value_minutes >= fail_minutes:
            return IncidentResponseStatus.fail
        if warn_minutes is not None and value_minutes >= warn_minutes:
            return IncidentResponseStatus.warn
        return None

    if metrics_payload is not None:
        mtta_value = _coerce_float(metrics_payload.get("mtta_minutes"))
        mttr_value = _coerce_float(metrics_payload.get("mttr_minutes"))
        acknowledged = _coerce_int(metrics_payload.get("acknowledged_incidents"), None)
        resolved = _coerce_int(metrics_payload.get("resolved_incidents"), None)

        def _emit_metric_alert(
            *,
            metric_value: float | None,
            warn_limit: float | None,
            fail_limit: float | None,
            category: str,
            label: str,
        ) -> None:
            metric_status = _metric_status(metric_value, warn_limit, fail_limit)
            if metric_status is None or not _meets_threshold(metric_status, threshold):
                return
            limit = fail_limit if metric_status is IncidentResponseStatus.fail else warn_limit
            if metric_value is None or limit is None:
                return
            comparison = "exceeds" if metric_status is IncidentResponseStatus.fail else "approaches"
            events.append(
                AlertEvent(
                    category=category,
                    severity=_SEVERITY_MAP[metric_status],
                    message=f"{label} {metric_value:.1f} minutes {comparison} {limit:.1f} minute threshold",
                    tags=tags + ("reliability", label.lower()),
                    context={
                        "snapshot": snapshot_payload,
                        "metrics": dict(metrics_payload),
                        "policy": dict(policy_mapping),
                        "acknowledged_incidents": acknowledged,
                        "resolved_incidents": resolved,
                    },
                )
            )

        _emit_metric_alert(
            metric_value=mtta_value,
            warn_limit=_policy_float("mtta_warn_minutes"),
            fail_limit=_policy_float("mtta_fail_minutes"),
            category="incident_response.mtta",
            label="MTTA",
        )
        _emit_metric_alert(
            metric_value=mttr_value,
            warn_limit=_policy_float("mttr_warn_minutes"),
            fail_limit=_policy_float("mttr_fail_minutes"),
            category="incident_response.mttr",
            label="MTTR",
        )

        age_hours = _coerce_float(metrics_payload.get("data_age_hours"))
        if age_hours is not None:
            warn_hours = _policy_float("metrics_stale_warn_hours")
            fail_hours = _policy_float("metrics_stale_fail_hours")
            staleness_status: IncidentResponseStatus | None = None
            limit_hours: float | None = None
            if fail_hours is not None and age_hours >= fail_hours:
                staleness_status = IncidentResponseStatus.fail
                limit_hours = fail_hours
            elif warn_hours is not None and age_hours >= warn_hours:
                staleness_status = IncidentResponseStatus.warn
                limit_hours = warn_hours
            if (
                staleness_status is not None
                and limit_hours is not None
                and _meets_threshold(staleness_status, threshold)
            ):
                comparison = (
                    "exceeds"
                    if staleness_status is IncidentResponseStatus.fail
                    else "approaches"
                )
                events.append(
                    AlertEvent(
                        category="incident_response.metrics_staleness",
                        severity=_SEVERITY_MAP[staleness_status],
                        message=(
                            f"Incident metrics age {age_hours:.1f} hours {comparison} "
                            f"{limit_hours:.1f} hour threshold"
                        ),
                        tags=tags + ("reliability", "staleness"),
                        context={
                            "snapshot": snapshot_payload,
                            "metrics": dict(metrics_payload),
                            "policy": dict(policy_mapping),
                            "acknowledged_incidents": acknowledged,
                            "resolved_incidents": resolved,
                        },
                    )
                )

    # Missing runbook escalation
    if snapshot.missing_runbooks and _meets_threshold(
        IncidentResponseStatus.fail, threshold
    ):
        events.append(
            AlertEvent(
                category="incident_response.missing_runbooks",
                severity=_SEVERITY_MAP[IncidentResponseStatus.fail],
                message="Missing required incident runbooks: "
                + ", ".join(sorted(snapshot.missing_runbooks)),
                tags=tags + ("runbook",),
                context={
                    "snapshot": snapshot_payload,
                    "missing_runbooks": list(snapshot.missing_runbooks),
                },
            )
        )

    # Training cadence checks
    training_interval = _policy_int("training_interval_days")
    training_age = snapshot.training_age_days
    if training_age is None:
        severity_status = IncidentResponseStatus.warn
        message = "Incident response training not recorded"
    elif training_interval:
        if training_age > training_interval * 2:
            severity_status = IncidentResponseStatus.fail
            message = (
                "Incident response training overdue more than twice the interval"
            )
        elif training_age > training_interval:
            severity_status = IncidentResponseStatus.warn
            message = "Incident response training overdue"
        else:
            severity_status = IncidentResponseStatus.ok
            message = ""
    else:
        severity_status = IncidentResponseStatus.ok
        message = ""
    if message and _meets_threshold(severity_status, threshold):
        events.append(
            AlertEvent(
                category="incident_response.training",
                severity=_SEVERITY_MAP[severity_status],
                message=message,
                tags=tags + ("training",),
                context={
                    "snapshot": snapshot_payload,
                    "training_age_days": training_age,
                    "training_interval_days": training_interval,
                },
            )
        )

    # Drill cadence checks
    drill_interval = _policy_int("drill_interval_days")
    drill_age = snapshot.drill_age_days
    if drill_age is None:
        drill_status = IncidentResponseStatus.warn
        drill_message = "Incident response drill not recorded"
    elif drill_interval:
        if drill_age > drill_interval * 2:
            drill_status = IncidentResponseStatus.fail
            drill_message = "Incident response drill cadence has lapsed"
        elif drill_age > drill_interval:
            drill_status = IncidentResponseStatus.warn
            drill_message = "Incident response drill overdue"
        else:
            drill_status = IncidentResponseStatus.ok
            drill_message = ""
    else:
        drill_status = IncidentResponseStatus.ok
        drill_message = ""
    if drill_message and _meets_threshold(drill_status, threshold):
        events.append(
            AlertEvent(
                category="incident_response.drill",
                severity=_SEVERITY_MAP[drill_status],
                message=drill_message,
                tags=tags + ("drill",),
                context={
                    "snapshot": snapshot_payload,
                    "drill_age_days": drill_age,
                    "drill_interval_days": drill_interval,
                },
            )
        )

    # On-call coverage
    minimum_primary = _policy_int("minimum_primary_responders") or 0
    if minimum_primary and len(snapshot.primary_oncall) < minimum_primary and _meets_threshold(
        IncidentResponseStatus.fail, threshold
    ):
        events.append(
            AlertEvent(
                category="incident_response.roster.primary",
                severity=_SEVERITY_MAP[IncidentResponseStatus.fail],
                message=
                f"Primary responder coverage {len(snapshot.primary_oncall)}/{minimum_primary} below policy",
                tags=tags + ("roster",),
                context={
                    "snapshot": snapshot_payload,
                    "minimum_primary": minimum_primary,
                    "primary_oncall": list(snapshot.primary_oncall),
                },
            )
        )

    minimum_secondary = _policy_int("minimum_secondary_responders") or 0
    if minimum_secondary and len(snapshot.secondary_oncall) < minimum_secondary and _meets_threshold(
        IncidentResponseStatus.warn, threshold
    ):
        events.append(
            AlertEvent(
                category="incident_response.roster.secondary",
                severity=_SEVERITY_MAP[IncidentResponseStatus.warn],
                message=
                f"Secondary responder coverage {len(snapshot.secondary_oncall)}/{minimum_secondary} below target",
                tags=tags + ("roster",),
                context={
                    "snapshot": snapshot_payload,
                    "minimum_secondary": minimum_secondary,
                    "secondary_oncall": list(snapshot.secondary_oncall),
                },
            )
        )

    # Open incident volume
    open_incidents = list(snapshot.open_incidents)
    if open_incidents:
        max_open = _policy_int("maximum_open_incidents")
        if max_open is not None and max_open >= 0:
            if len(open_incidents) > max_open:
                open_status = IncidentResponseStatus.fail
                message = (
                    f"{len(open_incidents)} open incidents exceeds policy maximum {max_open}"
                )
            elif len(open_incidents) == max_open:
                open_status = IncidentResponseStatus.warn
                message = (
                    f"Open incidents reached policy maximum {max_open}" if max_open > 0 else ""
                )
            else:
                open_status = IncidentResponseStatus.warn
                message = f"{len(open_incidents)} open incidents require triage"
        else:
            open_status = IncidentResponseStatus.warn
            message = f"{len(open_incidents)} open incidents require triage"

        if message and _meets_threshold(open_status, threshold):
            events.append(
                AlertEvent(
                    category="incident_response.open_incidents",
                    severity=_SEVERITY_MAP[open_status],
                    message=message,
                    tags=tags + ("incident",),
                    context={
                        "snapshot": snapshot_payload,
                        "open_incidents": open_incidents,
                        "policy_max_open_incidents": max_open,
                    },
                )
            )

    # Postmortem backlog escalation
    backlog_hours = metadata.get("postmortem_backlog_hours")
    if isinstance(backlog_hours, (int, float)) and backlog_hours > 0:
        sla = _policy_float("postmortem_sla_hours")
        backlog_status = IncidentResponseStatus.warn
        if sla is not None and sla > 0:
            if backlog_hours > sla * 2:
                backlog_status = IncidentResponseStatus.fail
            elif backlog_hours > sla:
                backlog_status = IncidentResponseStatus.warn
        elif backlog_hours > 24.0:
            backlog_status = IncidentResponseStatus.fail
        if _meets_threshold(backlog_status, threshold):
            events.append(
                AlertEvent(
                    category="incident_response.postmortem_backlog",
                    severity=_SEVERITY_MAP[backlog_status],
                    message=f"Postmortem backlog at {backlog_hours:.1f} hours",
                    tags=tags + ("postmortem",),
                    context={
                        "snapshot": snapshot_payload,
                        "postmortem_backlog_hours": backlog_hours,
                        "postmortem_sla_hours": sla,
                    },
                )
            )

    # ChatOps readiness
    require_chatops = bool(policy_mapping.get("require_chatops", False))
    chatops_ready = bool(metadata.get("chatops_ready", False))
    if require_chatops and not chatops_ready and _meets_threshold(
        IncidentResponseStatus.warn, threshold
    ):
        events.append(
            AlertEvent(
                category="incident_response.chatops",
                severity=_SEVERITY_MAP[IncidentResponseStatus.warn],
                message="ChatOps automations disabled or unavailable",
                tags=tags + ("chatops",),
                context={"snapshot": snapshot_payload, "chatops_ready": chatops_ready},
            )
        )

    # Explicit issues captured in the snapshot
    for issue in snapshot.issues:
        issue_status = _detail_severity(snapshot.status)
        if not _meets_threshold(issue_status, threshold):
            continue
        issue_tags = tags + ("issue",)
        detail_entry = issue_detail_map.get(issue)
        if detail_entry is not None:
            category = detail_entry.get("category")
            if category:
                issue_tags = issue_tags + (str(category),)
        context: dict[str, object] = {"snapshot": snapshot_payload, "issue": issue}
        if detail_entry is not None:
            context["detail"] = dict(detail_entry)
        events.append(
            AlertEvent(
                category="incident_response.issue",
                severity=_SEVERITY_MAP[issue_status],
                message=issue,
                tags=issue_tags,
                context=context,
            )
        )

    return events


def route_incident_response_alerts(
    manager: AlertManager,
    snapshot: IncidentResponseSnapshot,
    *,
    threshold: IncidentResponseStatus = IncidentResponseStatus.warn,
    include_status_event: bool = True,
    include_detail_events: bool = True,
    base_tags: Sequence[str] = ("incident-response",),
) -> list[AlertDispatchResult]:
    """Dispatch incident response alerts via an alert manager."""

    events = derive_incident_response_alerts(
        snapshot,
        threshold=threshold,
        include_status_event=include_status_event,
        include_detail_events=include_detail_events,
        base_tags=base_tags,
    )
    results: list[AlertDispatchResult] = []
    for event in events:
        results.append(manager.dispatch(event))
    return results


def publish_incident_response_snapshot(
    event_bus: EventBus, snapshot: IncidentResponseSnapshot
) -> None:
    """Publish the incident response snapshot onto the runtime event bus."""

    event = Event(
        type="telemetry.operational.incident_response",
        payload=snapshot.as_dict(),
        source="operations.incident_response",
    )

    publish_event_with_failover(
        event_bus,
        event,
        logger=logger,
        runtime_fallback_message=
            "Primary event bus publish_from_sync failed; falling back to global bus",
        runtime_unexpected_message=
            "Unexpected error publishing incident response via runtime event bus",
        runtime_none_message=
            "Primary event bus publish_from_sync returned None; falling back to global bus",
        global_not_running_message=
            "Global event bus not running while publishing incident response snapshot",
        global_unexpected_message=
            "Unexpected error publishing incident response snapshot via global bus",
    )


__all__ = [
    "IncidentResponseMetrics",
    "IncidentResponsePolicy",
    "IncidentResponseSnapshot",
    "IncidentResponseState",
    "IncidentResponseStatus",
    "derive_incident_response_alerts",
    "evaluate_incident_response",
    "format_incident_response_markdown",
    "publish_incident_response_snapshot",
    "route_incident_response_alerts",
]
