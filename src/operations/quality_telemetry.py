from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Mapping


class QualityStatus(StrEnum):
    """Severity levels describing CI quality posture."""

    ok = "ok"
    warn = "warn"
    fail = "fail"


_STATUS_ORDER = {
    QualityStatus.ok: 0,
    QualityStatus.warn: 1,
    QualityStatus.fail: 2,
}


def _escalate(current: QualityStatus, candidate: QualityStatus) -> QualityStatus:
    if _STATUS_ORDER[candidate] > _STATUS_ORDER[current]:
        return candidate
    return current


def _parse_timestamp(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


@dataclass(frozen=True)
class QualityTelemetrySnapshot:
    """Aggregated telemetry describing CI quality guardrails."""

    generated_at: datetime
    status: QualityStatus
    coverage_percent: float | None
    coverage_target: float
    staleness_hours: float | None
    max_staleness_hours: float
    notes: tuple[str, ...] = field(default_factory=tuple)
    remediation_items: tuple[str, ...] = field(default_factory=tuple)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at.astimezone(UTC).isoformat(),
            "status": self.status.value,
            "coverage_percent": self.coverage_percent,
            "coverage_target": self.coverage_target,
            "staleness_hours": self.staleness_hours,
            "max_staleness_hours": self.max_staleness_hours,
            "notes": list(self.notes),
            "remediation_items": list(self.remediation_items),
            "metadata": dict(self.metadata),
        }


def build_quality_telemetry_snapshot(
    metrics: Mapping[str, Any],
    *,
    coverage_target: float = 80.0,
    max_staleness_hours: float = 24.0,
    generated_at: datetime | None = None,
) -> QualityTelemetrySnapshot:
    now = generated_at or datetime.now(tz=UTC)
    notes: list[str] = []
    remediation_items: list[str] = []
    metadata: dict[str, Any] = {}

    coverage_trend = list(metrics.get("coverage_trend", []))
    latest_coverage = coverage_trend[-1] if coverage_trend else None

    coverage_percent: float | None = None
    staleness_hours: float | None = None

    status = QualityStatus.ok

    if latest_coverage:
        metadata["coverage_entry"] = dict(latest_coverage)
        coverage_raw = latest_coverage.get("coverage_percent")
        try:
            coverage_percent = float(coverage_raw)
        except (TypeError, ValueError):
            coverage_percent = None

        timestamp = (
            _parse_timestamp(latest_coverage.get("generated_at"))
            or _parse_timestamp(latest_coverage.get("label"))
        )
        if timestamp is not None:
            staleness_hours = (now.astimezone(UTC) - timestamp.astimezone(UTC)).total_seconds() / 3600.0
        else:
            notes.append("Coverage telemetry missing timestamp metadata")
            status = QualityStatus.warn

    if coverage_percent is not None:
        notes.append(f"Coverage {coverage_percent:.2f}% (target {coverage_target:.2f}%)")
        if coverage_percent < max(coverage_target - 5.0, 0.0):
            status = QualityStatus.fail
        elif coverage_percent < coverage_target:
            status = _escalate(status, QualityStatus.warn)
    else:
        notes.append("Coverage percentage unavailable")
        status = _escalate(status, QualityStatus.warn)

    if staleness_hours is not None:
        notes.append(f"Coverage telemetry age {staleness_hours:.1f}h (max {max_staleness_hours:.1f}h)")
        if staleness_hours > max_staleness_hours * 2:
            status = QualityStatus.fail
        elif staleness_hours > max_staleness_hours:
            status = _escalate(status, QualityStatus.warn)
    else:
        notes.append("Coverage telemetry staleness unknown")

    coverage_domain_trend = list(metrics.get("coverage_domain_trend", []))
    if coverage_domain_trend:
        latest_domain = coverage_domain_trend[-1]
        metadata["coverage_domain_entry"] = dict(latest_domain)
        lagging = latest_domain.get("lagging_domains") or []
        if lagging:
            lagging_list = [str(item) for item in lagging if item]
            notes.append("Lagging domains: " + ", ".join(sorted(lagging_list)))
            status = _escalate(status, QualityStatus.warn)

    remediation_trend = list(metrics.get("remediation_trend", []))
    if remediation_trend:
        latest_remediation = remediation_trend[-1]
        metadata["remediation_entry"] = dict(latest_remediation)
        note = latest_remediation.get("note")
        if isinstance(note, str) and note.strip():
            remediation_items.append(note.strip())
        statuses = latest_remediation.get("statuses")
        if isinstance(statuses, Mapping):
            lowered = {str(key): str(value).lower() for key, value in statuses.items()}
            if any(value == "fail" for value in lowered.values()):
                status = QualityStatus.fail
            elif any(value == "warn" for value in lowered.values()):
                status = _escalate(status, QualityStatus.warn)

    metadata.setdefault("trend_counts", {
        "coverage_trend": len(coverage_trend),
        "coverage_domain_trend": len(coverage_domain_trend),
        "remediation_trend": len(remediation_trend),
    })

    return QualityTelemetrySnapshot(
        generated_at=now,
        status=status,
        coverage_percent=coverage_percent,
        coverage_target=float(coverage_target),
        staleness_hours=staleness_hours,
        max_staleness_hours=float(max_staleness_hours),
        notes=tuple(notes),
        remediation_items=tuple(remediation_items),
        metadata=metadata,
    )
