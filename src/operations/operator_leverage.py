"""Operator leverage telemetry helpers.

This module analyses recent experimentation events and produces a snapshot that
tracks velocity (experiments per operator per week) and quality posture. The
resulting payload supports roadmap milestones that require higher experiment
throughput *without* quality regressions by surfacing underperforming
operators, missing telemetry, and top failure reasons in a deterministic data
structure.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import StrEnum
from typing import Any, Mapping, MutableMapping, Sequence

try:  # Python 3.10 compatibility
    from datetime import UTC
except ImportError:  # pragma: no cover - fallback for older runtimes
    UTC = timezone.utc


_POSITIVE_STATUSES = {
    "executed",
    "completed",
    "success",
    "passed",
    "ok",
    "accepted",
    "promoted",
    "deployed",
}

_NEGATIVE_STATUSES = {
    "rejected",
    "failed",
    "error",
    "timeout",
    "aborted",
    "forced_paper",
    "blocked",
    "rolled_back",
    "cancelled",
    "canceled",
}


class OperatorLeverageStatus(StrEnum):
    """Overall status grading for operator leverage telemetry."""

    ok = "ok"
    warn = "warn"
    fail = "fail"


@dataclass(frozen=True)
class OperatorExperimentStats:
    """Per-operator experimentation metrics."""

    operator: str
    experiments: int
    experiments_per_week: float
    quality_passes: int
    quality_failures: int
    quality_rate: float | None
    last_experiment_at: datetime | None
    recent_statuses: tuple[str, ...] = field(default_factory=tuple)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "operator": self.operator,
            "experiments": self.experiments,
            "experiments_per_week": self.experiments_per_week,
            "quality_passes": self.quality_passes,
            "quality_failures": self.quality_failures,
        }
        if self.quality_rate is not None:
            payload["quality_rate"] = self.quality_rate
        if self.last_experiment_at is not None:
            payload["last_experiment_at"] = (
                self.last_experiment_at.astimezone(UTC).isoformat()
            )
        if self.recent_statuses:
            payload["recent_statuses"] = list(self.recent_statuses)
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload


@dataclass(frozen=True)
class OperatorLeverageSnapshot:
    """Aggregated operator leverage telemetry."""

    generated_at: datetime
    status: OperatorLeverageStatus
    experiments_total: int
    operator_count: int
    weeks: float
    experiments_per_week: float | None
    experiments_per_week_total: float | None
    quality_pass_rate: float | None
    operators: tuple[OperatorExperimentStats, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "generated_at": self.generated_at.astimezone(UTC).isoformat(),
            "status": self.status.value,
            "experiments_total": self.experiments_total,
            "operator_count": self.operator_count,
            "weeks": self.weeks,
            "experiments_per_week": self.experiments_per_week,
            "experiments_per_week_total": self.experiments_per_week_total,
            "quality_pass_rate": self.quality_pass_rate,
            "operators": [operator.as_dict() for operator in self.operators],
        }
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload

    def to_markdown(self) -> str:
        """Render the snapshot as a Markdown table for dashboards."""

        lines = [
            "| Operator | Experiments | /week | Quality | Last experiment |",
            "| --- | ---:| ---:| ---:| --- |",
        ]
        for stats in self.operators:
            per_week = f"{stats.experiments_per_week:.2f}"
            quality = "n/a"
            if stats.quality_rate is not None:
                quality = f"{stats.quality_rate:.1%}"
            last_seen = (
                stats.last_experiment_at.astimezone(UTC).isoformat()
                if stats.last_experiment_at is not None
                else "n/a"
            )
            lines.append(
                "| {operator} | {experiments} | {per_week} | {quality} | {last} |".format(
                    operator=stats.operator,
                    experiments=stats.experiments,
                    per_week=per_week,
                    quality=quality,
                    last=last_seen,
                )
            )

        avg_per_week = (self.experiments_per_week or 0.0)
        quality_summary = (
            f"{self.quality_pass_rate:.1%}"
            if self.quality_pass_rate is not None
            else "n/a"
        )
        summary = (
            f"**Total** experiments {self.experiments_total} across {self.operator_count} "
            f"operators ({avg_per_week:.2f}/week avg, quality {quality_summary})"
        )
        lines.append(summary)
        return "\n".join(lines)


def _coerce_str(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        return text or None
    return str(value).strip() or None


def _coerce_float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip().rstrip("%")
        if not text:
            return None
        try:
            numeric = float(text)
        except ValueError:
            return None
        if value.endswith("%"):
            numeric /= 100.0
        return numeric
    return None


def _parse_timestamp(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value.astimezone(UTC) if value.tzinfo else value.replace(tzinfo=UTC)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError:
            return None
        return parsed.astimezone(UTC) if parsed.tzinfo else parsed.replace(tzinfo=UTC)
    return None


def _normalise_operator(entry: Mapping[str, Any]) -> str:
    for key in ("operator", "owner", "assigned_to", "analyst", "author", "user"):
        value = entry.get(key)
        name = _coerce_str(value)
        if name:
            return name
    metadata = entry.get("metadata")
    if isinstance(metadata, Mapping):
        for key in ("operator", "owner", "assigned_to", "analyst", "author", "user"):
            value = metadata.get(key)
            name = _coerce_str(value)
            if name:
                return name
    return "unknown"


def _normalise_quality(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"pass", "passed", "success", "ok", "true"}:
            return True
        if lowered in {"fail", "failed", "error", "false", "bad"}:
            return False
        numeric = _coerce_float(value)
        if numeric is not None:
            return numeric >= 0.5
        return None
    if isinstance(value, Mapping):
        mapping = {str(key): inner for key, inner in value.items()}
        for key in ("passed", "pass", "success", "ok"):
            if key in mapping:
                normalised = _normalise_quality(mapping[key])
                if normalised is not None:
                    return normalised
        status_value = mapping.get("status")
        if status_value is not None:
            status_normalised = _normalise_quality(status_value)
            if status_normalised is not None:
                return status_normalised
        score = _coerce_float(mapping.get("score"))
        if score is not None:
            threshold = _coerce_float(mapping.get("threshold"))
            if threshold is None:
                threshold = _coerce_float(mapping.get("target"))
            if threshold is None:
                threshold = 0.5
            return score >= threshold
    return None


def _extract_quality(entry: Mapping[str, Any]) -> tuple[bool | None, str]:
    for key in ("quality", "quality_pass", "quality_status", "passed", "pass"):
        if key in entry:
            quality = _normalise_quality(entry[key])
            if quality is not None:
                return quality, "explicit"
    metadata = entry.get("metadata")
    if isinstance(metadata, Mapping):
        for key in ("quality", "quality_pass", "quality_status", "passed", "pass"):
            if key in metadata:
                quality = _normalise_quality(metadata[key])
                if quality is not None:
                    return quality, "explicit"
    status = entry.get("status")
    if status is not None:
        status_str = _coerce_str(status)
        if status_str:
            lowered = status_str.lower()
            if lowered in _POSITIVE_STATUSES:
                return True, "status"
            if lowered in _NEGATIVE_STATUSES:
                return False, "status"
    decision = entry.get("decision")
    if isinstance(decision, Mapping):
        decision_status = _coerce_str(decision.get("status"))
        if decision_status:
            lowered = decision_status.lower()
            if lowered in _POSITIVE_STATUSES:
                return True, "status"
            if lowered in _NEGATIVE_STATUSES:
                return False, "status"
    return None, "missing"


def _extract_failure_reason(entry: Mapping[str, Any]) -> str | None:
    for key in ("reason", "failure_reason", "error", "message"):
        value = entry.get(key)
        reason = _coerce_str(value)
        if reason:
            return reason
    metadata = entry.get("metadata")
    if isinstance(metadata, Mapping):
        for key in ("reason", "failure_reason", "error", "message"):
            value = metadata.get(key)
            reason = _coerce_str(value)
            if reason:
                return reason
    decision = entry.get("decision")
    if isinstance(decision, Mapping):
        for key in ("reason", "error", "message"):
            value = decision.get(key)
            reason = _coerce_str(value)
            if reason:
                return reason
    return None


def _escalate_status(
    current: OperatorLeverageStatus, candidate: OperatorLeverageStatus
) -> OperatorLeverageStatus:
    order = {
        OperatorLeverageStatus.ok: 0,
        OperatorLeverageStatus.warn: 1,
        OperatorLeverageStatus.fail: 2,
    }
    if order[candidate] > order[current]:
        return candidate
    return current


def _weeks_from_days(days: float) -> float:
    return max(days, 1.0) / 7.0


def evaluate_operator_leverage(
    events: Sequence[Mapping[str, Any]] | None,
    *,
    lookback_days: float = 28.0,
    target_experiments_per_week: float = 3.0,
    warn_experiments_per_week: float = 2.0,
    target_quality_rate: float = 0.8,
    warn_quality_rate: float = 0.65,
    generated_at: datetime | None = None,
) -> OperatorLeverageSnapshot:
    """Aggregate experimentation events into an operator leverage snapshot."""

    now = generated_at.astimezone(UTC) if generated_at else datetime.now(tz=UTC)
    lookback_days = max(float(lookback_days), 1.0)
    window_start = now - timedelta(days=lookback_days)
    weeks = _weeks_from_days(lookback_days)

    operator_data: MutableMapping[str, MutableMapping[str, Any]] = {}
    failure_reasons: Counter[str] = Counter()
    quality_sources: Counter[str] = Counter()
    quality_passes = 0
    quality_failures = 0
    experiments_total = 0
    skipped_due_to_timestamp = 0

    if events is None:
        events = ()

    for entry in events:
        if not isinstance(entry, Mapping):
            continue
        timestamp = _parse_timestamp(entry.get("timestamp") or entry.get("created_at"))
        if timestamp is None:
            skipped_due_to_timestamp += 1
            continue
        if timestamp < window_start:
            continue

        operator = _normalise_operator(entry)
        data = operator_data.setdefault(
            operator,
            {
                "experiments": 0,
                "quality_passes": 0,
                "quality_failures": 0,
                "quality_considered": 0,
                "statuses": deque(maxlen=5),
                "last_at": None,
                "quality_sources": Counter(),
            },
        )

        data["experiments"] += 1
        experiments_total += 1

        status_value = _coerce_str(entry.get("status"))
        if status_value:
            data["statuses"].append(status_value)

        quality, quality_source = _extract_quality(entry)
        quality_sources[quality_source] += 1
        data["quality_sources"][quality_source] += 1
        if quality is not None:
            data["quality_considered"] += 1
            if quality:
                data["quality_passes"] += 1
                quality_passes += 1
            else:
                data["quality_failures"] += 1
                quality_failures += 1
                reason = _extract_failure_reason(entry)
                if reason:
                    failure_reasons[reason] += 1

        last_at = data["last_at"]
        if last_at is None or timestamp > last_at:
            data["last_at"] = timestamp

    operator_stats: list[OperatorExperimentStats] = []
    status = OperatorLeverageStatus.ok
    low_velocity_warn: list[str] = []
    low_velocity_fail: list[str] = []
    quality_warn: list[str] = []
    quality_fail: list[str] = []
    missing_quality: list[str] = []

    warn_threshold = min(target_experiments_per_week, warn_experiments_per_week)
    warn_threshold = max(warn_threshold, 0.0)
    target = max(target_experiments_per_week, warn_threshold)

    quality_warn_threshold = min(target_quality_rate, warn_quality_rate)
    quality_warn_threshold = max(min(quality_warn_threshold, 1.0), 0.0)
    quality_target = max(target_quality_rate, quality_warn_threshold)
    quality_target = min(max(quality_target, 0.0), 1.0)

    total_per_week = None
    avg_per_week = None

    if operator_data:
        total_per_week = experiments_total / weeks
        per_week_accumulator = 0.0
        for operator, data in sorted(operator_data.items()):
            experiments = int(data["experiments"])
            per_week = experiments / weeks
            quality_considered = int(data["quality_considered"])
            passes = int(data["quality_passes"])
            failures = int(data["quality_failures"])
            quality_rate = None
            if quality_considered > 0:
                quality_rate = passes / quality_considered
            else:
                missing_quality.append(operator)

            if per_week < warn_threshold:
                low_velocity_fail.append(operator)
                status = _escalate_status(status, OperatorLeverageStatus.fail)
            elif per_week < target:
                low_velocity_warn.append(operator)
                status = _escalate_status(status, OperatorLeverageStatus.warn)

            if quality_rate is None:
                status = _escalate_status(status, OperatorLeverageStatus.warn)
            elif quality_rate < quality_warn_threshold:
                quality_fail.append(operator)
                status = _escalate_status(status, OperatorLeverageStatus.fail)
            elif quality_rate < quality_target:
                quality_warn.append(operator)
                status = _escalate_status(status, OperatorLeverageStatus.warn)

            last_at = data["last_at"]
            recent_statuses = tuple(data["statuses"])
            metadata: MutableMapping[str, Any] = {
                "quality_events": quality_considered,
                "quality_sources": dict(data["quality_sources"]),
            }
            if last_at is not None:
                metadata["days_since_last"] = (
                    now - last_at
                ).total_seconds() / 86400.0

            operator_stats.append(
                OperatorExperimentStats(
                    operator=operator,
                    experiments=experiments,
                    experiments_per_week=per_week,
                    quality_passes=passes,
                    quality_failures=failures,
                    quality_rate=quality_rate,
                    last_experiment_at=last_at,
                    recent_statuses=recent_statuses,
                    metadata=metadata,
                )
            )
            per_week_accumulator += per_week

        avg_per_week = per_week_accumulator / len(operator_stats)

        if avg_per_week < warn_threshold:
            status = _escalate_status(status, OperatorLeverageStatus.fail)
        elif avg_per_week < target:
            status = _escalate_status(status, OperatorLeverageStatus.warn)
    else:
        status = OperatorLeverageStatus.fail

    quality_pass_rate = None
    quality_total = quality_passes + quality_failures
    if quality_total > 0:
        quality_pass_rate = quality_passes / quality_total
        if quality_pass_rate < quality_warn_threshold:
            status = _escalate_status(status, OperatorLeverageStatus.fail)
        elif quality_pass_rate < quality_target:
            status = _escalate_status(status, OperatorLeverageStatus.warn)
    else:
        status = _escalate_status(status, OperatorLeverageStatus.warn)

    operator_stats.sort(key=lambda stat: stat.experiments_per_week, reverse=True)

    snapshot_metadata: dict[str, Any] = {
        "lookback_days": lookback_days,
        "window_start": window_start.astimezone(UTC).isoformat(),
        "thresholds": {
            "target_experiments_per_week": target,
            "warn_experiments_per_week": warn_threshold,
            "target_quality_rate": quality_target,
            "warn_quality_rate": quality_warn_threshold,
        },
        "quality_source_counts": dict(quality_sources),
        "skipped_no_timestamp": skipped_due_to_timestamp,
        "low_velocity_warn": tuple(low_velocity_warn),
        "low_velocity_fail": tuple(low_velocity_fail),
        "quality_warn": tuple(quality_warn),
        "quality_fail": tuple(quality_fail),
        "quality_missing": tuple(missing_quality),
    }

    if failure_reasons:
        snapshot_metadata["top_failure_reasons"] = dict(failure_reasons.most_common(5))

    return OperatorLeverageSnapshot(
        generated_at=now,
        status=status,
        experiments_total=experiments_total,
        operator_count=len(operator_stats),
        weeks=weeks,
        experiments_per_week=avg_per_week,
        experiments_per_week_total=total_per_week,
        quality_pass_rate=quality_pass_rate,
        operators=tuple(operator_stats),
        metadata=snapshot_metadata,
    )


def format_operator_leverage_markdown(
    snapshot: OperatorLeverageSnapshot,
) -> str:
    """Return a Markdown rendering suitable for runbooks and dashboards."""

    header = (
        f"# Operator leverage ({snapshot.status.value.upper()})\n"
        f"- Generated at: {snapshot.generated_at.astimezone(UTC).isoformat()}\n"
        f"- Experiments: {snapshot.experiments_total} across {snapshot.operator_count} operators\n"
    )
    if snapshot.experiments_per_week is not None:
        header += f"- Average per operator: {snapshot.experiments_per_week:.2f}/week\n"
    if snapshot.quality_pass_rate is not None:
        header += f"- Quality pass rate: {snapshot.quality_pass_rate:.1%}\n"
    else:
        header += "- Quality pass rate: n/a\n"
    header += "\n"
    return header + snapshot.to_markdown()


__all__ = [
    "OperatorExperimentStats",
    "OperatorLeverageSnapshot",
    "OperatorLeverageStatus",
    "evaluate_operator_leverage",
    "format_operator_leverage_markdown",
]
