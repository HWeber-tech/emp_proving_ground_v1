"""Helpers for parsing and presenting final dry run progress snapshots."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, MutableMapping, Sequence

from src.operations.dry_run_audit import DryRunStatus, humanise_timedelta

UTC = timezone.utc

__all__ = [
    "DryRunProgressIncident",
    "DryRunProgressSnapshot",
    "format_progress_snapshot",
    "load_progress_snapshot",
    "parse_progress_snapshot",
]


def _parse_datetime(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        timestamp = datetime.fromisoformat(text)
    except ValueError:
        return None
    if timestamp.tzinfo is None:
        return timestamp.replace(tzinfo=UTC)
    return timestamp.astimezone(UTC)


def _parse_float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _parse_int_mapping(mapping: Any) -> Mapping[str, int]:
    if not isinstance(mapping, Mapping):
        return {}
    result: MutableMapping[str, int] = {}
    for key, value in mapping.items():
        if not isinstance(key, str):
            continue
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            result[key] = int(value)
        elif isinstance(value, str):
            try:
                result[key] = int(float(value))
            except ValueError:
                continue
    return dict(sorted(result.items()))


def _parse_bool(value: Any, *, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y"}:
            return True
        if lowered in {"0", "false", "no", "n"}:
            return False
    return default


@dataclass(frozen=True)
class DryRunProgressIncident:
    """Incident captured in the progress snapshot."""

    severity: DryRunStatus | None
    occurred_at: datetime | None
    message: str
    metadata: Mapping[str, Any]

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "DryRunProgressIncident":
        severity_value: DryRunStatus | None
        try:
            severity_value = DryRunStatus(str(data.get("severity", "")).strip())
        except ValueError:
            severity_value = None
        occurred_at = _parse_datetime(data.get("occurred_at"))
        message = str(data.get("message") or data.get("summary") or "").strip()
        metadata: Mapping[str, Any]
        metadata_value = data.get("metadata")
        if isinstance(metadata_value, Mapping):
            metadata = dict(metadata_value)
        else:
            metadata = {}
        return cls(
            severity=severity_value,
            occurred_at=occurred_at,
            message=message,
            metadata=metadata,
        )

    def headline(self) -> str:
        label = (self.severity.value.upper() if self.severity is not None else "INFO")
        timestamp = (
            self.occurred_at.astimezone(UTC).isoformat()
            if self.occurred_at is not None
            else "unknown"
        )
        body = self.message or "(no description provided)"
        return f"[{label}] {timestamp} — {body}"


@dataclass(frozen=True)
class DryRunProgressSnapshot:
    """Normalised representation of a progress snapshot JSON payload."""

    status: str
    status_severity: DryRunStatus | None
    phase: str | None
    now: datetime | None
    started_at: datetime | None
    elapsed: timedelta | None
    target_duration: timedelta | None
    required_duration: timedelta | None
    total_lines: int | None
    line_counts: Mapping[str, int]
    level_counts: Mapping[str, int]
    first_line_at: datetime | None
    last_line_at: datetime | None
    minimum_uptime_ratio: float | None
    require_diary_evidence: bool
    require_performance_evidence: bool
    config_metadata: Mapping[str, Any]
    incidents: tuple[DryRunProgressIncident, ...]
    summary_status: DryRunStatus | None
    sign_off_status: DryRunStatus | None
    raw: Mapping[str, Any]

    @property
    def status_label(self) -> str:
        if self.status_severity is not None:
            return self.status_severity.value.upper()
        if self.status:
            return self.status.upper()
        return "UNKNOWN"

    @property
    def elapsed_ratio(self) -> float | None:
        if self.elapsed is None or self.target_duration is None:
            return None
        target_seconds = self.target_duration.total_seconds()
        if target_seconds <= 0:
            return None
        elapsed_seconds = max(self.elapsed.total_seconds(), 0.0)
        ratio = min(elapsed_seconds / target_seconds, 1.0)
        return max(ratio, 0.0)

    @property
    def is_terminal(self) -> bool:
        return self.status_severity in {
            DryRunStatus.pass_,
            DryRunStatus.warn,
            DryRunStatus.fail,
        }

    @property
    def countdown(self) -> timedelta | None:
        if self.elapsed is None or self.target_duration is None:
            return None
        return max(self.target_duration - self.elapsed, timedelta())


def parse_progress_snapshot(data: Mapping[str, Any]) -> DryRunProgressSnapshot:
    status_text = str(data.get("status") or "").strip()
    try:
        status_severity = DryRunStatus(status_text)
    except ValueError:
        status_severity = None

    phase_value = data.get("phase")
    phase = str(phase_value).strip() if isinstance(phase_value, str) else None

    now = _parse_datetime(data.get("now"))
    started_at = _parse_datetime(data.get("started_at"))

    elapsed_seconds = _parse_float(data.get("elapsed_seconds"))
    elapsed = (
        timedelta(seconds=max(elapsed_seconds, 0.0))
        if elapsed_seconds is not None
        else None
    )

    target_seconds = _parse_float(data.get("target_duration_seconds"))
    target_duration = (
        timedelta(seconds=max(target_seconds, 0.0))
        if target_seconds is not None
        else None
    )

    required_seconds = _parse_float(data.get("required_duration_seconds"))
    required_duration = (
        timedelta(seconds=max(required_seconds, 0.0))
        if required_seconds is not None
        else None
    )

    total_lines_value = data.get("total_lines")
    if isinstance(total_lines_value, (int, float)):
        total_lines = int(total_lines_value)
    else:
        total_lines = None

    line_counts = _parse_int_mapping(data.get("line_counts"))
    level_counts = _parse_int_mapping(data.get("level_counts"))

    first_line_at = _parse_datetime(data.get("first_line_at"))
    last_line_at = _parse_datetime(data.get("last_line_at"))

    minimum_uptime_ratio = _parse_float(data.get("minimum_uptime_ratio"))
    require_diary = _parse_bool(
        data.get("require_diary_evidence"), default=False
    )
    require_performance = _parse_bool(
        data.get("require_performance_evidence"), default=False
    )

    config_metadata_raw = data.get("config_metadata")
    config_metadata: Mapping[str, Any]
    if isinstance(config_metadata_raw, Mapping):
        config_metadata = dict(config_metadata_raw)
    else:
        config_metadata = {}

    incident_values: Sequence[Mapping[str, Any]]
    raw_incidents = data.get("incidents")
    if isinstance(raw_incidents, Sequence):
        incident_values = [item for item in raw_incidents if isinstance(item, Mapping)]
    else:
        incident_values = []
    incidents = tuple(
        DryRunProgressIncident.from_mapping(item) for item in incident_values
    )

    summary_status: DryRunStatus | None = None
    summary_value = data.get("summary")
    if isinstance(summary_value, Mapping):
        try:
            summary_status = DryRunStatus(
                str(summary_value.get("status") or "").strip()
            )
        except ValueError:
            summary_status = None

    sign_off_status: DryRunStatus | None = None
    sign_off_value = data.get("sign_off")
    if isinstance(sign_off_value, Mapping):
        try:
            sign_off_status = DryRunStatus(
                str(sign_off_value.get("status") or "").strip()
            )
        except ValueError:
            sign_off_status = None

    return DryRunProgressSnapshot(
        status=status_text,
        status_severity=status_severity,
        phase=phase,
        now=now,
        started_at=started_at,
        elapsed=elapsed,
        target_duration=target_duration,
        required_duration=required_duration,
        total_lines=total_lines,
        line_counts=line_counts,
        level_counts=level_counts,
        first_line_at=first_line_at,
        last_line_at=last_line_at,
        minimum_uptime_ratio=minimum_uptime_ratio,
        require_diary_evidence=require_diary,
        require_performance_evidence=require_performance,
        config_metadata=config_metadata,
        incidents=incidents,
        summary_status=summary_status,
        sign_off_status=sign_off_status,
        raw=dict(data),
    )


def load_progress_snapshot(path: Path) -> DryRunProgressSnapshot:
    text = path.read_text(encoding="utf-8")
    payload = json.loads(text)
    if not isinstance(payload, Mapping):  # pragma: no cover - defensive guard
        raise ValueError("Progress file did not contain a JSON object")
    return parse_progress_snapshot(payload)


def _format_counts(label: str, mapping: Mapping[str, int]) -> str | None:
    if not mapping:
        return None
    parts = [f"{key}={value}" for key, value in sorted(mapping.items())]
    return f"{label}: {', '.join(parts)}"


def format_progress_snapshot(
    snapshot: DryRunProgressSnapshot,
    *,
    include_incidents: bool = True,
    max_incidents: int = 5,
    include_summary: bool = True,
    include_sign_off: bool = True,
) -> str:
    lines: list[str] = []

    timestamp = (
        snapshot.now.astimezone(UTC).isoformat()
        if snapshot.now is not None
        else "unknown"
    )
    phase = snapshot.phase or "n/a"
    lines.append(f"Status: {snapshot.status_label} (phase: {phase}, observed at {timestamp})")

    if snapshot.elapsed is not None:
        elapsed_text = humanise_timedelta(snapshot.elapsed)
        if snapshot.target_duration is not None:
            target_text = humanise_timedelta(snapshot.target_duration)
            ratio = snapshot.elapsed_ratio
            percent_text = f"{ratio * 100:.1f}%" if ratio is not None else "n/a"
            lines.append(f"Elapsed: {elapsed_text} / {target_text} ({percent_text})")
        else:
            lines.append(f"Elapsed: {elapsed_text}")

    if snapshot.countdown is not None:
        lines.append(f"Time remaining: {humanise_timedelta(snapshot.countdown)}")

    if snapshot.required_duration is not None:
        lines.append(
            "Required minimum: " + humanise_timedelta(snapshot.required_duration)
        )

    if snapshot.minimum_uptime_ratio is not None:
        lines.append(
            f"Uptime target: {snapshot.minimum_uptime_ratio:.2%}"
        )

    if snapshot.total_lines is not None:
        lines.append(f"Total log lines: {snapshot.total_lines}")

    streams_line = _format_counts("Streams", snapshot.line_counts)
    levels_line = _format_counts("Levels", snapshot.level_counts)
    if streams_line:
        lines.append(streams_line)
    if levels_line:
        lines.append(levels_line)

    if snapshot.first_line_at is not None:
        lines.append(
            f"First log: {snapshot.first_line_at.astimezone(UTC).isoformat()}"
        )
    if snapshot.last_line_at is not None:
        lines.append(
            f"Last log: {snapshot.last_line_at.astimezone(UTC).isoformat()}"
        )

    if snapshot.require_diary_evidence or snapshot.require_performance_evidence:
        required: list[str] = []
        if snapshot.require_diary_evidence:
            required.append("diary")
        if snapshot.require_performance_evidence:
            required.append("performance")
        lines.append("Required evidence: " + ", ".join(required))

    if snapshot.config_metadata:
        pairs = [f"{key}={value}" for key, value in sorted(snapshot.config_metadata.items())]
        lines.append("Config metadata: " + ", ".join(pairs))

    if include_summary and snapshot.summary_status is not None:
        lines.append(f"Summary status: {snapshot.summary_status.value.upper()}")

    if include_sign_off and snapshot.sign_off_status is not None:
        lines.append(f"Sign-off status: {snapshot.sign_off_status.value.upper()}")

    if include_incidents and snapshot.incidents:
        lines.append("Incidents:")
        for incident in list(snapshot.incidents)[-max(1, max_incidents):]:
            lines.append(f"  - {incident.headline()}")

    return "\n".join(lines)
