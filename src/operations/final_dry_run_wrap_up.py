"""Wrap-up reporting helpers for the AlphaTrade final dry run."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any, Iterable, Mapping, MutableMapping, Sequence

from src.operations.dry_run_audit import DryRunStatus, humanise_timedelta

__all__ = [
    "WrapUpIncident",
    "WrapUpBacklogItem",
    "FinalDryRunWrapUp",
    "build_wrap_up_report",
]


_STATUS_PRIORITY: Mapping[DryRunStatus, int] = {
    DryRunStatus.fail: 3,
    DryRunStatus.warn: 2,
    DryRunStatus.pass_: 1,
}


@dataclass(frozen=True)
class WrapUpIncident:
    """Incident captured during the dry run harness orchestration."""

    severity: DryRunStatus
    occurred_at: datetime | None
    message: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Mapping[str, Any]:
        payload: MutableMapping[str, Any] = {
            "severity": self.severity.value,
            "message": self.message,
            "metadata": {str(key): value for key, value in self.metadata.items()},
        }
        if self.occurred_at is not None:
            payload["occurred_at"] = self.occurred_at.astimezone(UTC).isoformat()
        return payload


@dataclass(frozen=True)
class WrapUpBacklogItem:
    """Follow-up work captured from dry run evidence."""

    severity: DryRunStatus
    category: str
    description: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Mapping[str, Any]:
        return {
            "severity": self.severity.value,
            "category": self.category,
            "description": self.description,
            "metadata": {str(key): value for key, value in self.metadata.items()},
        }


@dataclass(frozen=True)
class FinalDryRunWrapUp:
    """Structured report used to run the final dry run wrap-up meeting."""

    generated_at: datetime
    status: DryRunStatus
    summary_status: DryRunStatus | None
    sign_off_status: DryRunStatus | None
    review_status: DryRunStatus | None
    duration_seconds: float | None
    duration_target_seconds: float | None
    duration_met: bool | None
    backlog_items: tuple[WrapUpBacklogItem, ...]
    incidents: tuple[WrapUpIncident, ...]
    highlights: Mapping[str, Any] = field(default_factory=dict)
    notes: tuple[str, ...] = tuple()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Mapping[str, Any]:
        payload: MutableMapping[str, Any] = {
            "generated_at": self.generated_at.astimezone(UTC).isoformat(),
            "status": self.status.value,
            "summary_status": self.summary_status.value if self.summary_status else None,
            "sign_off_status": self.sign_off_status.value if self.sign_off_status else None,
            "review_status": self.review_status.value if self.review_status else None,
            "duration_seconds": self.duration_seconds,
            "duration_target_seconds": self.duration_target_seconds,
            "duration_met": self.duration_met,
            "backlog_items": [item.as_dict() for item in self.backlog_items],
            "incidents": [incident.as_dict() for incident in self.incidents],
            "highlights": {key: value for key, value in self.highlights.items()},
            "notes": list(self.notes),
            "metadata": {str(key): value for key, value in self.metadata.items()},
        }
        return payload

    def to_markdown(self) -> str:
        """Render the wrap-up report as Markdown minutes for the review meeting."""

        lines = [
            f"# Final Dry Run Wrap-up — {self.status.value.upper()}",
            "",
            f"Generated: {self.generated_at.astimezone(UTC).isoformat()}",
            "",
        ]

        duration_lines = []
        if self.duration_target_seconds is not None:
            duration_lines.append(
                f"Target duration: {humanise_timedelta(timedelta(seconds=self.duration_target_seconds))}"
            )
        if self.duration_seconds is not None:
            duration_lines.append(
                f"Observed duration: {humanise_timedelta(timedelta(seconds=self.duration_seconds))}"
            )
        if self.duration_met is not None:
            duration_lines.append("Duration met: " + ("Yes" if self.duration_met else "No"))
        if duration_lines:
            lines.append("## Duration")
            lines.extend(f"- {entry}" for entry in duration_lines)
            lines.append("")

        lines.append("## Status Summary")
        lines.append(
            f"- Evidence summary: {self.summary_status.value.upper()}"
            if self.summary_status
            else "- Evidence summary: N/A"
        )
        lines.append(
            f"- Sign-off assessment: {self.sign_off_status.value.upper()}"
            if self.sign_off_status
            else "- Sign-off assessment: N/A"
        )
        lines.append(
            f"- Review brief: {self.review_status.value.upper()}"
            if self.review_status
            else "- Review brief: N/A"
        )
        lines.append("")

        if self.highlights:
            lines.append("## Highlights")
            for key, value in sorted(self.highlights.items()):
                lines.append(f"- {key}: {value}")
            lines.append("")

        lines.append("## Backlog & Follow-ups")
        if self.backlog_items:
            for item in self.backlog_items:
                lines.append(
                    f"- {item.severity.value.upper()} — {item.category}: {item.description}"
                )
                if item.metadata:
                    for key, value in sorted(item.metadata.items()):
                        lines.append(f"  - {key}: {value}")
        else:
            lines.append("- None")
        lines.append("")

        lines.append("## Harness Incidents")
        if self.incidents:
            for incident in self.incidents:
                occurred = (
                    incident.occurred_at.astimezone(UTC).isoformat()
                    if incident.occurred_at is not None
                    else "Unknown"
                )
                lines.append(
                    f"- {incident.severity.value.upper()} — {occurred}: {incident.message}"
                )
                if incident.metadata:
                    for key, value in sorted(incident.metadata.items()):
                        lines.append(f"  - {key}: {value}")
        else:
            lines.append("- None")
        lines.append("")

        if self.notes:
            lines.append("## Notes")
            for note in self.notes:
                lines.append(f"- {note}")
            lines.append("")

        if self.metadata:
            lines.append("## Metadata")
            for key, value in sorted(self.metadata.items()):
                lines.append(f"- {key}: {value}")
            lines.append("")

        return "\n".join(lines).rstrip() + "\n"


def build_wrap_up_report(
    bundle: Mapping[str, Any],
    *,
    required_duration: timedelta | None = None,
    duration_tolerance: timedelta = timedelta(minutes=5),
    treat_warn_as_failure: bool = False,
) -> FinalDryRunWrapUp:
    """Derive a wrap-up report from the harness summary bundle."""

    summary_payload = _expect_mapping(bundle.get("summary"), "summary")
    summary_status = _parse_status(summary_payload.get("status"))
    summary_metadata = dict(_coerce_mapping(summary_payload.get("metadata")))

    log_payload = _coerce_mapping(summary_payload.get("logs"))
    log_duration_seconds = _coerce_float(log_payload.get("duration_seconds"))
    log_highlights = _build_log_highlights(log_payload)

    diary_payload = _coerce_mapping(summary_payload.get("diary"))
    performance_payload = _coerce_mapping(summary_payload.get("performance"))

    sign_off_payload = _coerce_mapping(bundle.get("sign_off"))
    sign_off_status = _parse_status(sign_off_payload.get("status")) if sign_off_payload else None

    review_payload = _coerce_mapping(bundle.get("review"))
    review_status = _parse_status(review_payload.get("status")) if review_payload else None
    review_notes: Sequence[str] = ()
    if review_payload:
        raw_notes = review_payload.get("notes")
        if isinstance(raw_notes, Sequence) and not isinstance(raw_notes, (str, bytes)):
            review_notes = tuple(
                str(note).strip()
                for note in raw_notes
                if str(note).strip()
            )

    incidents = _normalise_incidents(bundle.get("incidents"))

    backlog: list[_WrapBacklogEntry] = []
    backlog.extend(_derive_log_backlog(log_payload))
    backlog.extend(_derive_diary_backlog(diary_payload))
    backlog.extend(_derive_performance_backlog(performance_payload))
    backlog.extend(_derive_sign_off_backlog(sign_off_payload))
    backlog.extend(_derive_review_backlog(review_payload))
    backlog.extend(_derive_incident_backlog(incidents))

    duration_met: bool | None = None
    duration_status: DryRunStatus | None = None
    if required_duration is not None and log_duration_seconds is not None:
        target_seconds = required_duration.total_seconds()
        tolerance_seconds = max(duration_tolerance.total_seconds(), 0.0)
        duration_met = log_duration_seconds + tolerance_seconds >= target_seconds
        duration_status = DryRunStatus.pass_ if duration_met else DryRunStatus.fail
        if not duration_met:
            backlog.append(
                _WrapBacklogEntry(
                    severity=duration_status,
                    category="duration",
                    description=(
                        "Observed uptime fell short of the target duration "
                        f"({humanise_timedelta(timedelta(seconds=log_duration_seconds or 0))} < "
                        f"{humanise_timedelta(required_duration)})"
                    ),
                    metadata={
                        "target_seconds": target_seconds,
                        "observed_seconds": log_duration_seconds,
                    },
                )
            )
    else:
        duration_status = None

    backlog_items = tuple(
        WrapUpBacklogItem(
            severity=entry.severity,
            category=entry.category,
            description=entry.description,
            metadata=entry.metadata,
        )
        for entry in backlog
    )

    status_candidates = [
        _parse_status(bundle.get("status")),
        summary_status,
        sign_off_status,
        review_status,
        duration_status,
        *(item.severity for item in backlog_items),
        *(incident.severity for incident in incidents),
    ]
    overall_status = _max_status(status_candidates, treat_warn_as_failure=treat_warn_as_failure)

    generated_at = _parse_datetime(bundle.get("generated_at"))
    if generated_at is None:
        generated_at = _parse_datetime(summary_payload.get("generated_at"))
    if generated_at is None:
        generated_at = datetime.now(tz=UTC)

    highlights: dict[str, Any] = {**log_highlights}
    if review_payload:
        highlights.update(dict(_coerce_mapping(review_payload.get("highlights"))))
    highlights.update(summary_metadata)

    return FinalDryRunWrapUp(
        generated_at=generated_at,
        status=overall_status,
        summary_status=summary_status,
        sign_off_status=sign_off_status,
        review_status=review_status,
        duration_seconds=log_duration_seconds,
        duration_target_seconds=(required_duration.total_seconds() if required_duration else None),
        duration_met=duration_met,
        backlog_items=backlog_items,
        incidents=incidents,
        highlights=highlights,
        notes=tuple(review_notes),
        metadata=dict(_coerce_mapping(bundle.get("metadata"))),
    )


@dataclass(frozen=True)
class _WrapBacklogEntry:
    severity: DryRunStatus
    category: str
    description: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


def _derive_log_backlog(payload: Mapping[str, Any]) -> list[_WrapBacklogEntry]:
    entries: list[_WrapBacklogEntry] = []
    if not payload:
        return entries

    for field, category in (
        ("errors", "logs"),
        ("warnings", "logs"),
        ("gap_incidents", "logs"),
        ("content_incidents", "logs"),
    ):
        for incident in _iter_incident_dicts(payload.get(field)):
            severity = _parse_status(incident.get("severity")) or DryRunStatus.warn
            description = str(incident.get("summary") or incident.get("message") or "log incident")
            metadata = {key: value for key, value in incident.items() if key not in {"severity", "summary", "message"}}
            entries.append(
                _WrapBacklogEntry(
                    severity=severity,
                    category=category,
                    description=description,
                    metadata=metadata,
                )
            )

    level_counts = payload.get("level_counts")
    if isinstance(level_counts, Mapping):
        if any(level_counts.get(level, 0) for level in ("error", "critical", "exception", "fatal")):
            entries.append(
                _WrapBacklogEntry(
                    severity=DryRunStatus.fail,
                    category="logs",
                    description="Error-level log lines detected during the run.",
                    metadata=dict(level_counts),
                )
            )
    return entries


def _derive_diary_backlog(payload: Mapping[str, Any]) -> list[_WrapBacklogEntry]:
    entries: list[_WrapBacklogEntry] = []
    if not payload:
        return entries

    for issue in _iter_incident_dicts(payload.get("issues")):
        severity = _parse_status(issue.get("severity")) or DryRunStatus.warn
        description = str(issue.get("reason") or "Diary issue")
        metadata = {
            key: value
            for key, value in issue.items()
            if key not in {"severity", "reason"}
        }
        entries.append(
            _WrapBacklogEntry(
                severity=severity,
                category="diary",
                description=description,
                metadata=metadata,
            )
        )
    return entries


def _derive_performance_backlog(payload: Mapping[str, Any]) -> list[_WrapBacklogEntry]:
    entries: list[_WrapBacklogEntry] = []
    if not payload:
        return entries

    status = _parse_status(payload.get("status"))
    if status is not None and status is not DryRunStatus.pass_:
        entries.append(
            _WrapBacklogEntry(
                severity=status,
                category="performance",
                description="Performance telemetry flagged non-pass status.",
                metadata=dict(payload),
            )
        )
    return entries


def _derive_sign_off_backlog(payload: Mapping[str, Any]) -> list[_WrapBacklogEntry]:
    entries: list[_WrapBacklogEntry] = []
    if not payload:
        return entries

    for finding in _iter_incident_dicts(payload.get("findings")):
        severity = _parse_status(finding.get("severity")) or DryRunStatus.warn
        description = str(finding.get("message") or "Sign-off finding")
        metadata = {
            key: value
            for key, value in finding.items()
            if key not in {"severity", "message"}
        }
        entries.append(
            _WrapBacklogEntry(
                severity=severity,
                category="sign_off",
                description=description,
                metadata=metadata,
            )
        )
    return entries


def _derive_review_backlog(payload: Mapping[str, Any]) -> list[_WrapBacklogEntry]:
    entries: list[_WrapBacklogEntry] = []
    if not payload:
        return entries

    for item in _iter_incident_dicts(payload.get("action_items")):
        severity = _parse_status(item.get("severity")) or DryRunStatus.warn
        description = str(item.get("description") or "Review action item")
        metadata = {
            key: value
            for key, value in item.items()
            if key not in {"severity", "description"}
        }
        category = str(item.get("category") or "review")
        entries.append(
            _WrapBacklogEntry(
                severity=severity,
                category=category,
                description=description,
                metadata=metadata,
            )
        )

    for objective in _iter_incident_dicts(payload.get("objectives")):
        severity = _parse_status(objective.get("status"))
        if severity is None or severity is DryRunStatus.pass_:
            continue
        description = str(objective.get("note") or "Objective requires follow-up")
        metadata = {
            key: value
            for key, value in objective.items()
            if key not in {"status", "note"}
        }
        entries.append(
            _WrapBacklogEntry(
                severity=severity,
                category="objective",
                description=description,
                metadata=metadata,
            )
        )
    return entries


def _derive_incident_backlog(incidents: Sequence[WrapUpIncident]) -> list[_WrapBacklogEntry]:
    entries: list[_WrapBacklogEntry] = []
    for incident in incidents:
        entries.append(
            _WrapBacklogEntry(
                severity=incident.severity,
                category="harness",
                description=incident.message,
                metadata=incident.metadata,
            )
        )
    return entries


def _normalise_incidents(data: Any) -> tuple[WrapUpIncident, ...]:
    incidents: list[WrapUpIncident] = []
    for incident in _iter_incident_dicts(data):
        severity = _parse_status(incident.get("severity")) or DryRunStatus.warn
        occurred = _parse_datetime(incident.get("occurred_at"))
        message = str(incident.get("message") or incident.get("summary") or "incident")
        metadata = {
            key: value
            for key, value in incident.items()
            if key not in {"severity", "occurred_at", "message", "summary"}
        }
        incidents.append(
            WrapUpIncident(
                severity=severity,
                occurred_at=occurred,
                message=message,
                metadata=metadata,
            )
        )
    return tuple(incidents)


def _build_log_highlights(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    if not payload:
        return {}
    highlights: MutableMapping[str, Any] = {}
    if payload.get("started_at"):
        highlights["Log start"] = payload["started_at"]
    if payload.get("ended_at"):
        highlights["Log end"] = payload["ended_at"]
    if payload.get("duration_seconds") is not None:
        highlights["Observed duration"] = humanise_timedelta(
            timedelta(seconds=_coerce_float(payload.get("duration_seconds")) or 0.0)
        )
    if payload.get("uptime_ratio") is not None:
        highlights["Uptime"] = f"{float(payload['uptime_ratio']) * 100:.2f}%"
    level_counts = payload.get("level_counts")
    if isinstance(level_counts, Mapping) and level_counts:
        highlights["Log levels"] = ", ".join(
            f"{level}={count}"
            for level, count in sorted(level_counts.items())
        )
    return highlights


def _max_status(
    statuses: Iterable[DryRunStatus | None],
    *,
    treat_warn_as_failure: bool = False,
) -> DryRunStatus:
    highest = DryRunStatus.pass_
    for status in statuses:
        if status is None:
            continue
        candidate = status
        if treat_warn_as_failure and candidate is DryRunStatus.warn:
            candidate = DryRunStatus.fail
        if _STATUS_PRIORITY[candidate] > _STATUS_PRIORITY[highest]:
            highest = candidate
    return highest


def _parse_status(value: Any) -> DryRunStatus | None:
    if value is None:
        return None
    if isinstance(value, DryRunStatus):
        return value
    token = str(value).strip().lower()
    if not token:
        return None
    if token in {"pass", "pass_", "ok", "ready", "success"}:
        return DryRunStatus.pass_
    if token in {"warn", "warning", "at_risk", "yellow"}:
        return DryRunStatus.warn
    if token in {"fail", "failed", "block", "blocked", "no", "red", "error"}:
        return DryRunStatus.fail
    try:
        return DryRunStatus(token)
    except ValueError:
        return None


def _parse_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.astimezone(UTC)
    text = str(value).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _iter_incident_dicts(data: Any) -> Iterable[Mapping[str, Any]]:
    if isinstance(data, Mapping):
        yield data
    elif isinstance(data, Sequence) and not isinstance(data, (str, bytes)):
        for item in data:
            if isinstance(item, Mapping):
                yield item


def _expect_mapping(value: Any, label: str) -> Mapping[str, Any]:
    mapping = _coerce_mapping(value)
    if mapping:
        return mapping
    raise ValueError(f"Expected {label} payload to be a mapping")


def _coerce_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    return {}


def _coerce_float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return None
    return None
