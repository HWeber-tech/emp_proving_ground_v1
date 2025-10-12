"""Utilities for auditing final AlphaTrade dry runs.

This module assembles structured log evidence, decision diary
summaries, and performance telemetry into an auditable snapshot that can
be attached to the final roadmap sign-off packet.  The helpers are pure
and deterministic so they can be exercised via unit tests and reused by
CLI tooling or notebooks.
"""

from __future__ import annotations

import bz2
import gzip
import lzma
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import StrEnum
import json
from itertools import pairwise
from pathlib import Path
from typing import Any, Iterable, Mapping, MutableMapping, Sequence

from src.understanding.decision_diary import DecisionDiaryEntry, DecisionDiaryStore


DEFAULT_WARN_GAP = timedelta(hours=1)
DEFAULT_FAIL_GAP = timedelta(hours=6)
DEFAULT_DIARY_COVERAGE_TOLERANCE = timedelta(minutes=15)

_COMPRESSED_OPENERS = {
    ".gz": gzip.open,
    ".gzip": gzip.open,
    ".bz2": bz2.open,
    ".xz": lzma.open,
    ".lzma": lzma.open,
}


class DryRunStatus(StrEnum):
    """Severity level for a dry run component."""

    pass_ = "pass"
    warn = "warn"
    fail = "fail"

    def __str__(self) -> str:  # pragma: no cover - convenience for formatting
        return self.value


@dataclass(frozen=True)
class StructuredLogRecord:
    """Normalised representation of a structured log line."""

    timestamp: datetime
    level: str
    event: str | None
    message: str | None
    payload: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Mapping[str, Any]:
        return {
            "timestamp": self.timestamp.astimezone(UTC).isoformat(),
            "level": self.level,
            "event": self.event,
            "message": self.message,
            "payload": _normalise_mapping(self.payload),
        }


@dataclass(frozen=True)
class LogParseResult:
    """Outcome of parsing log lines from JSONL files."""

    records: tuple[StructuredLogRecord, ...]
    ignored_lines: int = 0


@dataclass(frozen=True)
class DryRunIncident:
    """Incident derived from either logs or diary entries."""

    severity: DryRunStatus
    occurred_at: datetime
    summary: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Mapping[str, Any]:
        return {
            "severity": self.severity.value,
            "occurred_at": self.occurred_at.astimezone(UTC).isoformat(),
            "summary": self.summary,
            "metadata": _normalise_mapping(self.metadata),
        }


@dataclass(frozen=True)
class DryRunLogSummary:
    """Aggregated view of log evidence captured during a dry run."""

    records: tuple[StructuredLogRecord, ...]
    ignored_lines: int
    level_counts: Mapping[str, int]
    event_counts: Mapping[str, int]
    gap_incidents: tuple[DryRunIncident, ...] = field(default_factory=tuple)
    content_incidents: tuple[DryRunIncident, ...] = field(default_factory=tuple)
    uptime_ratio: float | None = None

    @property
    def started_at(self) -> datetime | None:
        if not self.records:
            return None
        return min(record.timestamp for record in self.records)

    @property
    def ended_at(self) -> datetime | None:
        if not self.records:
            return None
        return max(record.timestamp for record in self.records)

    @property
    def duration(self) -> timedelta | None:
        if not self.records:
            return None
        return self.ended_at - self.started_at  # type: ignore[return-value]

    @property
    def warnings(self) -> tuple[DryRunIncident, ...]:
        incidents: list[DryRunIncident] = []
        for record in self.records:
            if record.level not in {"warning", "warn"}:
                continue
            incidents.append(
                DryRunIncident(
                    severity=DryRunStatus.warn,
                    occurred_at=record.timestamp,
                    summary=record.message or record.event or "warning",
                    metadata={"event": record.event, **_normalise_mapping(record.payload)},
                )
            )
        incidents.sort(key=lambda incident: incident.occurred_at)
        return tuple(incidents)

    @property
    def errors(self) -> tuple[DryRunIncident, ...]:
        incidents: list[DryRunIncident] = []
        for record in self.records:
            if record.level not in {"error", "exception", "critical", "fatal"}:
                continue
            incidents.append(
                DryRunIncident(
                    severity=DryRunStatus.fail,
                    occurred_at=record.timestamp,
                    summary=record.message or record.event or "error",
                    metadata={"event": record.event, **_normalise_mapping(record.payload)},
                )
            )
        incidents.sort(key=lambda incident: incident.occurred_at)
        return tuple(incidents)

    @property
    def status(self) -> DryRunStatus:
        if self.errors or any(
            incident.severity is DryRunStatus.fail
            for incident in (*self.gap_incidents, *self.content_incidents)
        ):
            return DryRunStatus.fail
        if self.warnings or any(
            incident.severity is DryRunStatus.warn
            for incident in (*self.gap_incidents, *self.content_incidents)
        ):
            return DryRunStatus.warn
        return DryRunStatus.pass_

    def as_dict(self) -> Mapping[str, Any]:
        payload: MutableMapping[str, Any] = {
            "ignored_lines": self.ignored_lines,
            "level_counts": dict(self.level_counts),
            "event_counts": dict(self.event_counts),
            "gap_incidents": [incident.as_dict() for incident in self.gap_incidents],
            "content_incidents": [
                incident.as_dict() for incident in self.content_incidents
            ],
            "warnings": [incident.as_dict() for incident in self.warnings],
            "errors": [incident.as_dict() for incident in self.errors],
            "status": self.status.value,
        }
        if self.started_at is not None:
            payload["started_at"] = self.started_at.astimezone(UTC).isoformat()
        if self.ended_at is not None:
            payload["ended_at"] = self.ended_at.astimezone(UTC).isoformat()
        if self.duration is not None:
            payload["duration_seconds"] = self.duration.total_seconds()
        if self.uptime_ratio is not None:
            payload["uptime_ratio"] = self.uptime_ratio
        return payload

    def to_markdown(self) -> str:
        rows = [
            "| Metric | Value |",
            "| --- | --- |",
            f"| Records | {len(self.records)} |",
            f"| Ignored lines | {self.ignored_lines} |",
        ]
        if self.started_at is not None:
            rows.append(
                f"| Started at | {self.started_at.astimezone(UTC).isoformat()} |"
            )
        if self.ended_at is not None:
            rows.append(
                f"| Ended at | {self.ended_at.astimezone(UTC).isoformat()} |"
            )
        if self.duration is not None:
            rows.append(
                f"| Duration | {humanise_timedelta(self.duration)} |"
            )
        if self.level_counts:
            level_summary = ", ".join(
                f"{level}={count}" for level, count in sorted(self.level_counts.items())
            )
            rows.append(f"| Levels | {level_summary} |")
        if self.event_counts:
            most_common = ", ".join(
                f"{event}={count}" for event, count in self._top_event_counts(5)
            )
            rows.append(f"| Top events | {most_common} |")
        if self.uptime_ratio is not None:
            rows.append(f"| Uptime | {self.uptime_ratio * 100:.2f}% |")
        if self.gap_incidents:
            rows.append(f"| Gap incidents | {len(self.gap_incidents)} |")
        if self.content_incidents:
            rows.append(f"| Content incidents | {len(self.content_incidents)} |")
        if self.errors:
            rows.append(f"| Errors | {len(self.errors)} |")
        if self.warnings:
            rows.append(f"| Warnings | {len(self.warnings)} |")
        return "\n".join(rows)

    def _top_event_counts(self, limit: int) -> tuple[tuple[str, int], ...]:
        sorted_events = sorted(
            ((event or "", count) for event, count in self.event_counts.items()),
            key=lambda item: (-item[1], item[0]),
        )
        return tuple(sorted_events[:limit])


@dataclass(frozen=True)
class DryRunDiaryIssue:
    """Diary entry flagged as noteworthy for dry run sign-off."""

    entry_id: str
    policy_id: str
    recorded_at: datetime
    severity: DryRunStatus
    reason: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Mapping[str, Any]:
        return {
            "entry_id": self.entry_id,
            "policy_id": self.policy_id,
            "recorded_at": self.recorded_at.astimezone(UTC).isoformat(),
            "severity": self.severity.value,
            "reason": self.reason,
            "metadata": _normalise_mapping(self.metadata),
        }


@dataclass(frozen=True)
class DryRunDiarySummary:
    """Aggregated statistics for decision diary evidence."""

    entries: tuple[DecisionDiaryEntry, ...]
    issues: tuple[DryRunDiaryIssue, ...]
    policy_counts: Mapping[str, int]

    @property
    def total_entries(self) -> int:
        return len(self.entries)

    @property
    def first_recorded_at(self) -> datetime | None:
        if not self.entries:
            return None
        return min(entry.recorded_at for entry in self.entries)

    @property
    def last_recorded_at(self) -> datetime | None:
        if not self.entries:
            return None
        return max(entry.recorded_at for entry in self.entries)

    @property
    def status(self) -> DryRunStatus:
        if any(issue.severity is DryRunStatus.fail for issue in self.issues):
            return DryRunStatus.fail
        if any(issue.severity is DryRunStatus.warn for issue in self.issues):
            return DryRunStatus.warn
        return DryRunStatus.pass_

    def as_dict(self) -> Mapping[str, Any]:
        payload: MutableMapping[str, Any] = {
            "total_entries": self.total_entries,
            "policy_counts": dict(self.policy_counts),
            "issues": [issue.as_dict() for issue in self.issues],
            "status": self.status.value,
        }
        if self.first_recorded_at is not None:
            payload["first_recorded_at"] = self.first_recorded_at.astimezone(UTC).isoformat()
        if self.last_recorded_at is not None:
            payload["last_recorded_at"] = self.last_recorded_at.astimezone(UTC).isoformat()
        return payload

    def to_markdown(self) -> str:
        rows = [
            "| Metric | Value |",
            "| --- | --- |",
            f"| Entries | {self.total_entries} |",
            f"| Policies | {len(self.policy_counts)} |",
        ]
        if self.first_recorded_at is not None:
            rows.append(
                f"| First entry | {self.first_recorded_at.astimezone(UTC).isoformat()} |"
            )
        if self.last_recorded_at is not None:
            rows.append(
                f"| Last entry | {self.last_recorded_at.astimezone(UTC).isoformat()} |"
            )
        if self.issues:
            rows.append(f"| Flagged entries | {len(self.issues)} |")
        return "\n".join(rows)


@dataclass(frozen=True)
class DryRunPerformanceSummary:
    """Lightweight summary of the performance tracker output."""

    generated_at: datetime
    period_start: datetime
    total_trades: int
    roi: float | None
    win_rate: float | None
    sharpe_ratio: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def status(self) -> DryRunStatus:
        if self.roi is not None and self.roi < 0:
            return DryRunStatus.warn
        return DryRunStatus.pass_

    @property
    def window_duration(self) -> timedelta:
        """Duration covered by the performance telemetry."""

        if self.generated_at < self.period_start:
            return timedelta(0)
        return self.generated_at - self.period_start

    def as_dict(self) -> Mapping[str, Any]:
        payload: MutableMapping[str, Any] = {
            "generated_at": self.generated_at.astimezone(UTC).isoformat(),
            "period_start": self.period_start.astimezone(UTC).isoformat(),
            "total_trades": self.total_trades,
            "metadata": _normalise_mapping(self.metadata),
            "status": self.status.value,
        }
        if self.roi is not None:
            payload["roi"] = self.roi
        if self.win_rate is not None:
            payload["win_rate"] = self.win_rate
        if self.sharpe_ratio is not None:
            payload["sharpe_ratio"] = self.sharpe_ratio
        if self.window_duration:
            payload["window_duration_seconds"] = self.window_duration.total_seconds()
        return payload

    def to_markdown(self) -> str:
        rows = [
            "| Metric | Value |",
            "| --- | --- |",
            f"| Generated at | {self.generated_at.astimezone(UTC).isoformat()} |",
            f"| Period start | {self.period_start.astimezone(UTC).isoformat()} |",
            f"| Trades | {self.total_trades} |",
        ]
        if self.roi is not None:
            rows.append(f"| ROI | {self.roi:.4f} |")
        if self.win_rate is not None:
            rows.append(f"| Win rate | {self.win_rate:.4f} |")
        if self.sharpe_ratio is not None:
            rows.append(f"| Sharpe ratio | {self.sharpe_ratio:.4f} |")
        rows.append(
            f"| Window | {humanise_timedelta(self.window_duration)} |"
            if self.window_duration
            else "| Window | 0s |"
        )
        return "\n".join(rows)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "DryRunPerformanceSummary":
        generated_at = _parse_timestamp(payload.get("generated_at"))
        period_start = _parse_timestamp(payload.get("period_start"))
        if generated_at is None or period_start is None:
            raise ValueError("Performance report must include generated_at and period_start timestamps")
        aggregates = payload.get("aggregates", {})
        if isinstance(aggregates, Mapping):
            total_trades = int(aggregates.get("trades", payload.get("trades", 0)))
            roi_raw = aggregates.get("roi")
            win_rate_raw = aggregates.get("win_rate")
            sharpe_raw = aggregates.get("sharpe_ratio") or aggregates.get("sharpe")
        else:
            total_trades = int(payload.get("trades", 0))
            roi_raw = payload.get("roi")
            win_rate_raw = payload.get("win_rate")
            sharpe_raw = payload.get("sharpe_ratio") or payload.get("sharpe")
        roi = float(roi_raw) if isinstance(roi_raw, (int, float)) else None
        win_rate = float(win_rate_raw) if isinstance(win_rate_raw, (int, float)) else None
        metadata_payload = payload.get("metadata", {})
        if sharpe_raw is None and isinstance(metadata_payload, Mapping):
            sharpe_raw = metadata_payload.get("sharpe_ratio") or metadata_payload.get("sharpe")
        sharpe_ratio = float(sharpe_raw) if isinstance(sharpe_raw, (int, float)) else None
        metadata = dict(metadata_payload) if isinstance(metadata_payload, Mapping) else {}
        return cls(
            generated_at=generated_at,
            period_start=period_start,
            total_trades=total_trades,
            roi=roi,
            win_rate=win_rate,
            sharpe_ratio=sharpe_ratio,
            metadata=metadata,
        )


@dataclass(frozen=True)
class DryRunSummary:
    """Top-level aggregation for the final dry run evidence bundle."""

    generated_at: datetime
    log_summary: DryRunLogSummary | None = None
    diary_summary: DryRunDiarySummary | None = None
    performance_summary: DryRunPerformanceSummary | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def status(self) -> DryRunStatus:
        statuses: list[DryRunStatus] = []
        if self.log_summary is not None:
            statuses.append(self.log_summary.status)
        if self.diary_summary is not None:
            statuses.append(self.diary_summary.status)
        if self.performance_summary is not None:
            statuses.append(self.performance_summary.status)
        if not statuses:
            return DryRunStatus.pass_
        if any(status is DryRunStatus.fail for status in statuses):
            return DryRunStatus.fail
        if any(status is DryRunStatus.warn for status in statuses):
            return DryRunStatus.warn
        return DryRunStatus.pass_

    def as_dict(self) -> Mapping[str, Any]:
        payload: MutableMapping[str, Any] = {
            "generated_at": self.generated_at.astimezone(UTC).isoformat(),
            "status": self.status.value,
            "metadata": _normalise_mapping(self.metadata),
        }
        if self.log_summary is not None:
            payload["logs"] = self.log_summary.as_dict()
        if self.diary_summary is not None:
            payload["diary"] = self.diary_summary.as_dict()
        if self.performance_summary is not None:
            payload["performance"] = self.performance_summary.as_dict()
        return payload

    def to_markdown(self) -> str:
        parts = [
            f"# Final dry run summary — {self.status.value.upper()}",
            "",
            f"Generated at: {self.generated_at.astimezone(UTC).isoformat()}",
            "",
        ]
        if self.log_summary is not None:
            parts.extend(["## Logs", self.log_summary.to_markdown(), ""])
            if self.log_summary.errors:
                parts.append("### Log incidents")
                for incident in self.log_summary.errors:
                    parts.append(
                        f"- {incident.occurred_at.astimezone(UTC).isoformat()}: {incident.summary}"
                    )
                parts.append("")
            if self.log_summary.warnings:
                parts.append("### Log warnings")
                for incident in self.log_summary.warnings:
                    parts.append(
                        f"- {incident.occurred_at.astimezone(UTC).isoformat()}: {incident.summary}"
                    )
                parts.append("")
            if self.log_summary.gap_incidents:
                parts.append("### Log gaps")
                for incident in self.log_summary.gap_incidents:
                    parts.append(
                        f"- {incident.occurred_at.astimezone(UTC).isoformat()}: {incident.summary}"
                    )
                parts.append("")
            if self.log_summary.content_incidents:
                parts.append("### Log anomalies")
                for incident in self.log_summary.content_incidents:
                    parts.append(
                        f"- {incident.occurred_at.astimezone(UTC).isoformat()}: {incident.summary}"
                    )
                parts.append("")
        if self.diary_summary is not None:
            parts.extend(["## Decision diary", self.diary_summary.to_markdown(), ""])
            if self.diary_summary.issues:
                parts.append("### Diary issues")
                for issue in self.diary_summary.issues:
                    parts.append(
                        f"- {issue.recorded_at.astimezone(UTC).isoformat()} — {issue.policy_id}: {issue.reason}"
                    )
                parts.append("")
        if self.performance_summary is not None:
            parts.extend(["## Performance", self.performance_summary.to_markdown(), ""])
        if self.metadata:
            parts.append("## Metadata")
            for key, value in sorted(self.metadata.items()):
                parts.append(f"- {key}: {value}")
            parts.append("")
        return "\n".join(parts).rstrip() + "\n"


@dataclass(frozen=True)
class DryRunSignOffFinding:
    """Individual finding surfaced during sign-off evaluation."""

    severity: DryRunStatus
    message: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Mapping[str, Any]:
        return {
            "severity": self.severity.value,
            "message": self.message,
            "metadata": _normalise_mapping(self.metadata),
        }


@dataclass(frozen=True)
class DryRunSignOffReport:
    """Result of assessing whether a dry run is ready for final sign-off."""

    evaluated_at: datetime
    findings: tuple[DryRunSignOffFinding, ...] = field(default_factory=tuple)
    criteria: Mapping[str, Any] = field(default_factory=dict)

    @property
    def status(self) -> DryRunStatus:
        if any(finding.severity is DryRunStatus.fail for finding in self.findings):
            return DryRunStatus.fail
        if any(finding.severity is DryRunStatus.warn for finding in self.findings):
            return DryRunStatus.warn
        return DryRunStatus.pass_

    def as_dict(self) -> Mapping[str, Any]:
        return {
            "evaluated_at": self.evaluated_at.astimezone(UTC).isoformat(),
            "status": self.status.value,
            "criteria": _normalise_mapping(self.criteria),
            "findings": [finding.as_dict() for finding in self.findings],
        }

    def to_markdown(self) -> str:
        lines = [
            f"# Dry run sign-off — {self.status.value.upper()}",
            "",
        ]
        if self.criteria:
            lines.append("## Criteria")
            for key, value in sorted(self.criteria.items()):
                lines.append(f"- **{key}**: {value}")
            lines.append("")
        if self.findings:
            lines.append("## Findings")
            for finding in self.findings:
                lines.append(f"- **{finding.severity.value.upper()}** — {finding.message}")
                if finding.metadata:
                    for meta_key, meta_value in sorted(finding.metadata.items()):
                        lines.append(f"  - {meta_key}: {meta_value}")
            lines.append("")
        else:
            lines.append("All sign-off criteria satisfied.")
            lines.append("")
        return "\n".join(lines)


def parse_structured_log_line(line: str) -> StructuredLogRecord | None:
    """Parse a JSON-formatted log line into a :class:`StructuredLogRecord`."""

    line = line.strip()
    if not line:
        return None
    try:
        payload = json.loads(line)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, Mapping):
        return None
    timestamp = _parse_timestamp(
        payload.get("timestamp")
        or payload.get("ts")
        or payload.get("@timestamp")
        or payload.get("time")
    )
    if timestamp is None:
        return None
    level_raw = payload.get("level") or payload.get("lvl") or payload.get("severity")
    if isinstance(level_raw, str):
        level = level_raw.strip().lower()
    elif isinstance(level_raw, (int, float)):
        level = str(level_raw)
    else:
        level = "info"
    event = payload.get("event")
    if event is not None:
        event = str(event)
    message_raw = payload.get("message") or payload.get("msg") or payload.get("event")
    message = str(message_raw) if message_raw is not None else None
    preserved = {"timestamp", "ts", "@timestamp", "time", "level", "lvl", "severity", "event", "message", "msg"}
    metadata = {
        str(key): value
        for key, value in payload.items()
        if key not in preserved
    }
    return StructuredLogRecord(
        timestamp=timestamp,
        level=level,
        event=event,
        message=message,
        payload=metadata,
    )


def _open_log_stream(path: Path):
    suffixes = [suffix.lower() for suffix in path.suffixes]
    opener = _COMPRESSED_OPENERS.get(suffixes[-1]) if suffixes else None
    if opener is None:
        return open(path, "rt", encoding="utf-8", errors="replace")
    return opener(path, "rt", encoding="utf-8", errors="replace")


def _iter_log_lines(path: Path) -> Iterable[str]:
    if not path.exists():
        return
    with _open_log_stream(path) as handle:
        for line in handle:
            yield line.rstrip("\r\n")


def load_structured_logs(paths: Iterable[Path]) -> LogParseResult:
    """Load structured logs from the provided JSONL files."""

    records: list[StructuredLogRecord] = []
    ignored = 0
    for path in paths:
        for line in _iter_log_lines(Path(path)):
            record = parse_structured_log_line(line)
            if record is None:
                ignored += 1
                continue
            records.append(record)
    records.sort(key=lambda record: record.timestamp)
    return LogParseResult(records=tuple(records), ignored_lines=ignored)


def analyse_structured_logs(
    result: LogParseResult,
    *,
    warn_gap: timedelta = DEFAULT_WARN_GAP,
    fail_gap: timedelta = DEFAULT_FAIL_GAP,
    minimum_duration: timedelta | None = None,
    minimum_uptime_ratio: float | None = None,
) -> DryRunLogSummary:
    """Derive roll-up statistics from parsed structured logs."""

    warn_gap = max(warn_gap, timedelta(0))
    fail_gap = max(fail_gap, warn_gap)
    level_counts = Counter(record.level for record in result.records)
    event_counts = Counter(record.event for record in result.records if record.event)

    if minimum_duration is not None:
        minimum_duration = max(minimum_duration, timedelta(0))
    if minimum_uptime_ratio is not None and not 0.0 <= minimum_uptime_ratio <= 1.0:
        raise ValueError("minimum_uptime_ratio must be between 0.0 and 1.0")

    gap_incidents, uptime_ratio = _analyse_log_gaps(
        result.records, warn_gap=warn_gap, fail_gap=fail_gap
    )
    incidents = list(gap_incidents)
    content_incidents = list(_detect_log_content_incidents(result.records))

    if minimum_duration is not None:
        duration = None
        if result.records:
            duration = result.records[-1].timestamp - result.records[0].timestamp
        if duration is None:
            incidents.append(
                DryRunIncident(
                    severity=DryRunStatus.fail,
                    occurred_at=datetime.now(tz=UTC),
                    summary="No logs captured during dry run",
                    metadata={
                        "required_min_duration_seconds": minimum_duration.total_seconds(),
                        "actual_duration_seconds": None,
                    },
                )
            )
        elif duration < minimum_duration:
            incidents.append(
                DryRunIncident(
                    severity=DryRunStatus.fail,
                    occurred_at=result.records[-1].timestamp,
                    summary=(
                        "Dry run duration "
                        f"{humanise_timedelta(duration)} below minimum "
                        f"{humanise_timedelta(minimum_duration)}"
                    ),
                    metadata={
                        "required_min_duration_seconds": minimum_duration.total_seconds(),
                        "actual_duration_seconds": duration.total_seconds(),
                    },
                )
            )

    if minimum_uptime_ratio is not None:
        if uptime_ratio is None or uptime_ratio < minimum_uptime_ratio:
            incidents.append(
                DryRunIncident(
                    severity=DryRunStatus.fail,
                    occurred_at=(
                        result.records[-1].timestamp if result.records else datetime.now(tz=UTC)
                    ),
                    summary=(
                        "Dry run uptime ratio "
                        f"{uptime_ratio if uptime_ratio is not None else 'unknown'} "
                        f"below minimum {minimum_uptime_ratio:.2f}"
                    ),
                    metadata={
                        "required_minimum_uptime_ratio": minimum_uptime_ratio,
                        "actual_uptime_ratio": uptime_ratio,
                    },
                )
            )

    return DryRunLogSummary(
        records=result.records,
        ignored_lines=result.ignored_lines,
        level_counts=dict(level_counts),
        event_counts=dict(event_counts),
        gap_incidents=tuple(incidents),
        content_incidents=tuple(content_incidents),
        uptime_ratio=uptime_ratio,
    )


def summarise_diary_entries(
    entries: Sequence[DecisionDiaryEntry],
    *,
    expected_window: tuple[datetime, datetime] | None = None,
    coverage_tolerance: timedelta = DEFAULT_DIARY_COVERAGE_TOLERANCE,
    minimum_entries_per_day: int | None = None,
    daily_coverage_severity: DryRunStatus = DryRunStatus.warn,
) -> DryRunDiarySummary:
    """Summarise the supplied diary entries for dry run review."""

    ordered = tuple(sorted(entries, key=lambda entry: entry.recorded_at))
    policy_counts = Counter(entry.policy_id for entry in ordered)
    issues: list[DryRunDiaryIssue] = []
    for entry in ordered:
        severity, reason = _classify_diary_entry(entry)
        if severity is None:
            continue
        issues.append(
            DryRunDiaryIssue(
                entry_id=entry.entry_id,
                policy_id=entry.policy_id,
                recorded_at=entry.recorded_at,
                severity=severity,
                reason=reason,
                metadata=dict(entry.metadata),
            )
        )
    if expected_window is not None:
        start_expected, end_expected = expected_window
        tolerance = max(coverage_tolerance, timedelta(0))
        if minimum_entries_per_day is not None and minimum_entries_per_day < 0:
            raise ValueError("minimum_entries_per_day must be non-negative")
        if not ordered:
            issues.append(
                DryRunDiaryIssue(
                    entry_id="coverage/missing",
                    policy_id="*",
                    recorded_at=start_expected,
                    severity=DryRunStatus.fail,
                    reason="No decision diary entries recorded during expected dry run window.",
                    metadata={
                        "expected_window_start": start_expected.astimezone(UTC).isoformat(),
                        "expected_window_end": end_expected.astimezone(UTC).isoformat(),
                        "coverage_tolerance_seconds": tolerance.total_seconds(),
                    },
                )
            )
        else:
            first_entry = ordered[0].recorded_at
            last_entry = ordered[-1].recorded_at
            start_gap = first_entry - start_expected
            if start_gap > tolerance:
                issues.append(
                    DryRunDiaryIssue(
                        entry_id="coverage/start_gap",
                        policy_id="*",
                        recorded_at=first_entry,
                        severity=DryRunStatus.warn,
                        reason="Decision diary missing coverage near dry run start.",
                        metadata={
                            "expected_window_start": start_expected.astimezone(UTC).isoformat(),
                            "first_entry": first_entry.astimezone(UTC).isoformat(),
                            "gap_seconds": start_gap.total_seconds(),
                            "coverage_tolerance_seconds": tolerance.total_seconds(),
                        },
                    )
                )
            end_gap = end_expected - last_entry
            if end_gap > tolerance:
                issues.append(
                    DryRunDiaryIssue(
                        entry_id="coverage/end_gap",
                        policy_id="*",
                        recorded_at=last_entry,
                        severity=DryRunStatus.warn,
                        reason="Decision diary missing coverage near dry run end.",
                        metadata={
                            "expected_window_end": end_expected.astimezone(UTC).isoformat(),
                            "last_entry": last_entry.astimezone(UTC).isoformat(),
                            "gap_seconds": end_gap.total_seconds(),
                            "coverage_tolerance_seconds": tolerance.total_seconds(),
                        },
                    )
                )

        min_entries = minimum_entries_per_day or 0
        if min_entries > 0:
            # Normalise dates in UTC to avoid timezone drift when comparing coverage.
            start_day = start_expected.astimezone(UTC).date()
            end_day = end_expected.astimezone(UTC).date()
            per_day_counts = Counter(
                entry.recorded_at.astimezone(UTC).date() for entry in ordered
            )
            day = start_day
            while day <= end_day:
                count = per_day_counts.get(day, 0)
                if count < min_entries:
                    recorded_at = datetime.combine(
                        day,
                        datetime.min.time(),
                        tzinfo=UTC,
                    )
                    issues.append(
                        DryRunDiaryIssue(
                            entry_id=f"coverage/daily/{day.isoformat()}",
                            policy_id="*",
                            recorded_at=recorded_at,
                            severity=(
                                DryRunStatus.fail
                                if daily_coverage_severity is DryRunStatus.fail
                                else DryRunStatus.warn
                            ),
                            reason=(
                                "Decision diary missing minimum entries for day"
                                f" {day.isoformat()}."
                            ),
                            metadata={
                                "day": day.isoformat(),
                                "required_entries": min_entries,
                                "actual_entries": count,
                            },
                        )
                    )
                day = day + timedelta(days=1)

    return DryRunDiarySummary(
        entries=ordered,
        issues=tuple(sorted(issues, key=lambda issue: issue.recorded_at)),
        policy_counts=dict(policy_counts),
    )


def load_decision_diary_entries(path: Path) -> tuple[DecisionDiaryEntry, ...]:
    """Load diary entries from the JSON file backing :class:`DecisionDiaryStore`."""

    if not path.exists():
        return tuple()
    store = DecisionDiaryStore(path, publish_on_record=False)
    return store.entries()


def load_performance_summary(path: Path) -> DryRunPerformanceSummary | None:
    """Load a :class:`DryRunPerformanceSummary` from a JSON payload."""

    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, Mapping):
        return DryRunPerformanceSummary.from_mapping(payload)
    raise ValueError("Performance report payload must be a mapping")


def evaluate_dry_run(
    *,
    log_paths: Sequence[Path],
    diary_path: Path | None = None,
    performance_path: Path | None = None,
    metadata: Mapping[str, Any] | None = None,
    log_gap_warn: timedelta | None = None,
    log_gap_fail: timedelta | None = None,
    minimum_run_duration: timedelta | None = None,
    minimum_uptime_ratio: float | None = None,
    diary_min_entries_per_day: int | None = None,
    diary_daily_issue_severity: DryRunStatus = DryRunStatus.warn,
) -> DryRunSummary:
    """Evaluate dry run evidence from logs, diaries, and performance telemetry."""

    parse_result = load_structured_logs(log_paths)
    log_summary = analyse_structured_logs(
        parse_result,
        warn_gap=log_gap_warn or DEFAULT_WARN_GAP,
        fail_gap=log_gap_fail or DEFAULT_FAIL_GAP,
        minimum_duration=minimum_run_duration,
        minimum_uptime_ratio=minimum_uptime_ratio,
    )

    diary_summary: DryRunDiarySummary | None = None
    if diary_path is not None:
        diary_entries = load_decision_diary_entries(diary_path)
        expected_window: tuple[datetime, datetime] | None = None
        if (
            log_summary.started_at is not None
            and log_summary.ended_at is not None
        ):
            expected_window = (log_summary.started_at, log_summary.ended_at)
        diary_summary = summarise_diary_entries(
            diary_entries,
            expected_window=expected_window,
            minimum_entries_per_day=diary_min_entries_per_day,
            daily_coverage_severity=diary_daily_issue_severity,
        )

    performance_summary: DryRunPerformanceSummary | None = None
    if performance_path is not None:
        performance_summary = load_performance_summary(performance_path)

    aggregate_metadata = {"log_paths": [str(path) for path in log_paths]}
    if diary_path is not None:
        aggregate_metadata["diary_path"] = str(diary_path)
    if performance_path is not None:
        aggregate_metadata["performance_path"] = str(performance_path)
    if metadata:
        aggregate_metadata.update({str(key): value for key, value in metadata.items()})

    return DryRunSummary(
        generated_at=datetime.now(tz=UTC),
        log_summary=log_summary,
        diary_summary=diary_summary,
        performance_summary=performance_summary,
        metadata=aggregate_metadata,
    )


def assess_sign_off_readiness(
    summary: DryRunSummary,
    *,
    minimum_duration: timedelta | None = None,
    minimum_uptime_ratio: float | None = None,
    require_diary: bool = False,
    require_performance: bool = False,
    allow_warnings: bool = False,
    minimum_sharpe_ratio: float | None = None,
) -> DryRunSignOffReport:
    """Evaluate whether a dry run summary satisfies sign-off criteria."""

    criteria: dict[str, Any] = {
        "minimum_duration_seconds": minimum_duration.total_seconds()
        if isinstance(minimum_duration, timedelta)
        else None,
        "minimum_uptime_ratio": minimum_uptime_ratio,
        "require_diary": require_diary,
        "require_performance": require_performance,
        "allow_warnings": allow_warnings,
        "minimum_sharpe_ratio": minimum_sharpe_ratio,
    }

    if minimum_sharpe_ratio is not None and minimum_sharpe_ratio < 0:
        raise ValueError("minimum_sharpe_ratio must be non-negative")

    findings: list[DryRunSignOffFinding] = []

    log_summary = summary.log_summary
    if log_summary is None:
        findings.append(
            DryRunSignOffFinding(
                severity=DryRunStatus.fail,
                message="Dry run summary is missing structured log evidence.",
            )
        )
    else:
        if log_summary.status is DryRunStatus.fail:
            findings.append(
                DryRunSignOffFinding(
                    severity=DryRunStatus.fail,
                    message="Structured log summary contains failures.",
                )
            )
        elif log_summary.status is DryRunStatus.warn:
            findings.append(
                DryRunSignOffFinding(
                    severity=(DryRunStatus.warn if allow_warnings else DryRunStatus.fail),
                    message="Structured log summary contains warnings.",
                )
            )
        if minimum_duration is not None:
            duration = log_summary.duration
            if duration is None:
                findings.append(
                    DryRunSignOffFinding(
                        severity=DryRunStatus.fail,
                        message="Dry run duration is unknown.",
                        metadata={"required_seconds": minimum_duration.total_seconds()},
                    )
                )
            elif duration < minimum_duration:
                findings.append(
                    DryRunSignOffFinding(
                        severity=DryRunStatus.fail,
                        message="Dry run duration below required minimum.",
                        metadata={
                            "required_seconds": minimum_duration.total_seconds(),
                            "actual_seconds": duration.total_seconds(),
                        },
                    )
                )
        if minimum_uptime_ratio is not None:
            uptime = log_summary.uptime_ratio
            if uptime is None:
                findings.append(
                    DryRunSignOffFinding(
                        severity=DryRunStatus.fail,
                        message="Dry run uptime ratio is unknown.",
                        metadata={"required_ratio": minimum_uptime_ratio},
                    )
                )
            elif uptime < minimum_uptime_ratio:
                findings.append(
                    DryRunSignOffFinding(
                        severity=DryRunStatus.fail,
                        message="Dry run uptime ratio below required minimum.",
                        metadata={
                            "required_ratio": minimum_uptime_ratio,
                            "actual_ratio": uptime,
                        },
                    )
                )

    diary_summary = summary.diary_summary
    if require_diary:
        if diary_summary is None:
            findings.append(
                DryRunSignOffFinding(
                    severity=DryRunStatus.fail,
                    message="Decision diary evidence is required for sign-off.",
                )
            )
        elif diary_summary.status is DryRunStatus.fail:
            findings.append(
                DryRunSignOffFinding(
                    severity=DryRunStatus.fail,
                    message="Decision diary contains failing issues.",
                )
            )
        elif diary_summary.status is DryRunStatus.warn:
            findings.append(
                DryRunSignOffFinding(
                    severity=(DryRunStatus.warn if allow_warnings else DryRunStatus.fail),
                    message="Decision diary contains warnings.",
                )
            )

    performance_summary = summary.performance_summary
    performance_required = require_performance or minimum_sharpe_ratio is not None
    if performance_required:
        if performance_summary is None:
            findings.append(
                DryRunSignOffFinding(
                    severity=DryRunStatus.fail,
                    message="Performance telemetry is required for sign-off.",
                )
            )
        elif performance_summary.status is DryRunStatus.fail:
            findings.append(
                DryRunSignOffFinding(
                    severity=DryRunStatus.fail,
                    message="Performance telemetry indicates failure.",
                )
            )
        elif performance_summary.status is DryRunStatus.warn:
            findings.append(
                DryRunSignOffFinding(
                    severity=(DryRunStatus.warn if allow_warnings else DryRunStatus.fail),
                    message="Performance telemetry indicates warnings.",
                )
            )
        if (
            performance_summary is not None
            and minimum_sharpe_ratio is not None
            and performance_summary.sharpe_ratio is not None
            and performance_summary.sharpe_ratio < minimum_sharpe_ratio
        ):
            findings.append(
                DryRunSignOffFinding(
                    severity=DryRunStatus.fail,
                    message="Sharpe ratio below required minimum.",
                    metadata={
                        "required_sharpe_ratio": minimum_sharpe_ratio,
                        "actual_sharpe_ratio": performance_summary.sharpe_ratio,
                    },
                )
            )
        if (
            performance_summary is not None
            and minimum_sharpe_ratio is not None
            and performance_summary.sharpe_ratio is None
        ):
            findings.append(
                DryRunSignOffFinding(
                    severity=DryRunStatus.fail,
                    message="Sharpe ratio is unavailable in performance telemetry.",
                    metadata={"required_sharpe_ratio": minimum_sharpe_ratio},
                )
            )

    return DryRunSignOffReport(
        evaluated_at=datetime.now(tz=UTC),
        findings=tuple(findings),
        criteria=criteria,
    )


# Helper utilities ---------------------------------------------------------

_FAIL_STATUSES = {"error", "failed", "halted", "fatal", "critical"}
_WARN_STATUSES = {"warn", "warning", "degraded", "throttled", "skipped"}

_TRACEBACK_TOKENS = (
    "traceback (most recent call last)",
    "uncaught exception",
    "unhandled exception",
    "fatal error",
    "panic:",
    "panic ",
)
_STACK_PAYLOAD_KEYS = {"exception", "traceback", "stacktrace", "stack_trace"}


def _detect_log_content_incidents(
    records: Sequence[StructuredLogRecord],
) -> tuple[DryRunIncident, ...]:
    """Flag suspicious log lines that indicate uncaught exceptions or traces."""

    incidents: list[DryRunIncident] = []
    for record in records:
        message = (record.message or "").lower()
        payload_strings: list[str] = []
        for key, value in record.payload.items():
            if key in _STACK_PAYLOAD_KEYS and isinstance(value, str):
                payload_strings.append(value.lower())
        combined_payload = "\n".join(payload_strings)

        has_traceback = any(token in message for token in _TRACEBACK_TOKENS)
        has_traceback = has_traceback or any(
            token in combined_payload for token in _TRACEBACK_TOKENS
        )
        has_traceback = has_traceback or "traceback" in message or "traceback" in combined_payload
        mentions_exception = (
            "exception" in message
            or "exception" in combined_payload
            or any(key.endswith("exception") for key in record.payload)
        )

        if not (has_traceback or mentions_exception and "handled" not in message):
            continue

        incidents.append(
            DryRunIncident(
                severity=DryRunStatus.fail,
                occurred_at=record.timestamp,
                summary=(
                    record.message
                    or record.event
                    or "Exception detected in structured logs"
                ),
                metadata={
                    "event": record.event,
                    "level": record.level,
                    "payload_keys": sorted(str(key) for key in record.payload.keys()),
                },
            )
        )

    incidents.sort(key=lambda incident: incident.occurred_at)
    return tuple(incidents)


def _classify_diary_entry(entry: DecisionDiaryEntry) -> tuple[DryRunStatus | None, str]:
    """Determine whether a diary entry should be flagged as an issue."""

    candidates: list[tuple[str, str]] = []
    for source, payload in (
        ("metadata", entry.metadata),
        ("decision", entry.decision),
        ("outcomes", entry.outcomes),
    ):
        if not isinstance(payload, Mapping):
            continue
        status = payload.get("status")
        if isinstance(status, str) and status.strip():
            candidates.append((source, status.strip().lower()))
    for note in entry.notes:
        if not note:
            continue
        lowered = note.lower()
        if any(token in lowered for token in ("incident", "halt", "error")):
            return DryRunStatus.fail, f"note: {note}"
        if "warn" in lowered or "degraded" in lowered:
            return DryRunStatus.warn, f"note: {note}"
    for source, status in candidates:
        if status in _FAIL_STATUSES:
            return DryRunStatus.fail, f"{source}.status={status}"
    for source, status in candidates:
        if status in _WARN_STATUSES:
            return DryRunStatus.warn, f"{source}.status={status}"
    return None, ""


def _parse_timestamp(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.astimezone(UTC)
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(float(value), tz=UTC)
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return None
        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(raw)
        except ValueError:
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        return parsed.astimezone(UTC)
    return None


def _normalise_mapping(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    normalised: dict[str, Any] = {}
    for key, value in payload.items():
        normalised[str(key)] = _normalise_value(value)
    return normalised


def _normalise_value(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.astimezone(UTC).isoformat()
    if isinstance(value, Mapping):
        return _normalise_mapping(value)
    if isinstance(value, (list, tuple, set)):
        return [_normalise_value(item) for item in value]
    if isinstance(value, (int, float, str)) or value is None or isinstance(value, bool):
        return value
    return str(value)


def humanise_timedelta(delta: timedelta) -> str:
    """Return a compact human-readable representation of a timedelta."""

    total_seconds = int(delta.total_seconds())
    days, remainder = divmod(total_seconds, 24 * 3600)
    hours, remainder = divmod(remainder, 3600)
    minutes, seconds = divmod(remainder, 60)
    parts: list[str] = []
    if days:
        parts.append(f"{days}d")
    if hours or (days and (minutes or seconds)):
        parts.append(f"{hours}h")
    if minutes or (hours and seconds):
        parts.append(f"{minutes}m")
    elif not parts:
        parts.append("0m")
    parts.append(f"{seconds}s")
    return " ".join(parts)


__all__ = [
    "DEFAULT_WARN_GAP",
    "DEFAULT_FAIL_GAP",
    "DryRunStatus",
    "StructuredLogRecord",
    "LogParseResult",
    "DryRunIncident",
    "DryRunLogSummary",
    "DryRunDiaryIssue",
    "DryRunDiarySummary",
    "DryRunPerformanceSummary",
    "DryRunSummary",
    "parse_structured_log_line",
    "load_structured_logs",
    "analyse_structured_logs",
    "summarise_diary_entries",
    "load_decision_diary_entries",
    "load_performance_summary",
    "evaluate_dry_run",
    "humanise_timedelta",
    "_detect_log_content_incidents",
]


def _analyse_log_gaps(
    records: Sequence[StructuredLogRecord],
    *,
    warn_gap: timedelta,
    fail_gap: timedelta,
) -> tuple[tuple[DryRunIncident, ...], float | None]:
    if len(records) < 2:
        return tuple(), 1.0 if records else None

    warn_seconds = warn_gap.total_seconds()
    total_span = (records[-1].timestamp - records[0].timestamp).total_seconds()
    total_span = max(total_span, 0.0)
    total_excess_gap = 0.0
    incidents: list[DryRunIncident] = []

    for previous, current in pairwise(records):
        gap = current.timestamp - previous.timestamp
        if gap <= warn_gap:
            continue
        severity = DryRunStatus.fail if gap >= fail_gap else DryRunStatus.warn
        summary = (
            f"{humanise_timedelta(gap)} gap between "
            f"{previous.event or previous.message or 'previous log'}"
            f" and {current.event or current.message or 'next log'}"
        )
        incidents.append(
            DryRunIncident(
                severity=severity,
                occurred_at=current.timestamp,
                summary=summary,
                metadata={
                    "gap_seconds": gap.total_seconds(),
                    "previous_event": previous.event,
                    "next_event": current.event,
                },
            )
        )
        total_excess_gap += max(0.0, gap.total_seconds() - warn_seconds)

    if not incidents:
        return tuple(), 1.0 if total_span > 0 else None

    if total_span == 0:
        uptime_ratio = 0.0 if incidents else 1.0
    else:
        uptime_ratio = max(0.0, 1.0 - (total_excess_gap / total_span))

    return tuple(incidents), uptime_ratio
