"""CLI utilities for grading observability dashboard freshness and coverage."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence

from .ci_metrics import summarise_dashboard_payload


class DashboardGuardStatus(StrEnum):
    """Severity levels reported by the dashboard guard."""

    ok = "ok"
    warn = "warn"
    fail = "fail"


_STATUS_ORDER: Mapping[DashboardGuardStatus, int] = {
    DashboardGuardStatus.ok: 0,
    DashboardGuardStatus.warn: 1,
    DashboardGuardStatus.fail: 2,
}


def _escalate(
    current: DashboardGuardStatus, candidate: DashboardGuardStatus
) -> DashboardGuardStatus:
    if _STATUS_ORDER[candidate] > _STATUS_ORDER[current]:
        return candidate
    return current


@dataclass(frozen=True)
class DashboardGuardReport:
    """Evaluation summary for an observability dashboard snapshot."""

    status: DashboardGuardStatus
    generated_at: datetime | None
    age_seconds: float | None
    panel_counts: Mapping[str, int]
    failing_panels: Sequence[str]
    warning_panels: Sequence[str]
    missing_panels: Sequence[str]
    issues: Sequence[str]

    def as_dict(self) -> dict[str, Any]:
        payload: MutableMapping[str, Any] = {
            "status": self.status.value,
            "panel_counts": dict(self.panel_counts),
            "failing_panels": tuple(self.failing_panels),
            "warning_panels": tuple(self.warning_panels),
            "missing_panels": tuple(self.missing_panels),
            "issues": tuple(self.issues),
        }
        if self.generated_at is not None:
            payload["generated_at"] = self.generated_at.astimezone(UTC).isoformat()
        else:
            payload["generated_at"] = None
        payload["age_seconds"] = self.age_seconds
        return dict(payload)


def _parse_generated_at(timestamp: Any) -> datetime | None:
    if not isinstance(timestamp, str) or not timestamp:
        return None
    try:
        parsed = datetime.fromisoformat(timestamp)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed


def _normalise_summary(
    summary: Mapping[str, Any] | "ObservabilityDashboard",
) -> Mapping[str, Any]:
    if isinstance(summary, Mapping) and {
        "panel_counts",
        "failing_panels",
        "warning_panels",
        "healthy_panels",
    }.issubset(summary.keys()):
        return {
            "generated_at": summary.get("generated_at"),
            "overall_status": summary.get("overall_status"),
            "panel_counts": summary.get("panel_counts", {}),
            "failing_panels": summary.get("failing_panels", ()),
            "warning_panels": summary.get("warning_panels", ()),
            "healthy_panels": summary.get("healthy_panels", ()),
        }
    return summarise_dashboard_payload(summary)


def evaluate_dashboard_health(
    summary: Mapping[str, Any] | "ObservabilityDashboard",
    *,
    max_age: timedelta = timedelta(minutes=30),
    required_panels: Sequence[str] | None = None,
    current_time: datetime | None = None,
) -> DashboardGuardReport:
    """Evaluate dashboard freshness and panel coverage."""

    normalised = _normalise_summary(summary)
    generated_at = _parse_generated_at(normalised.get("generated_at"))
    now = current_time or datetime.now(tz=UTC)
    issues: list[str] = []
    status = DashboardGuardStatus.ok

    age_seconds: float | None = None
    if generated_at is None:
        issues.append("Snapshot missing valid generated_at timestamp")
        status = DashboardGuardStatus.fail
    else:
        age = now - generated_at
        age_seconds = age.total_seconds()
        if age_seconds < 0:
            issues.append("Snapshot timestamp is in the future")
            status = _escalate(status, DashboardGuardStatus.warn)
        elif age > max_age:
            issues.append(
                "Snapshot is stale: {age:.0f}s old (limit {limit:.0f}s)".format(
                    age=age.total_seconds(), limit=max_age.total_seconds()
                )
            )
            status = DashboardGuardStatus.fail

    failing_panels = tuple(str(name) for name in normalised.get("failing_panels", ()))
    if failing_panels:
        issues.append(
            "Failing panels detected: " + ", ".join(sorted(failing_panels))
        )
        status = DashboardGuardStatus.fail

    warning_panels = tuple(str(name) for name in normalised.get("warning_panels", ()))
    if warning_panels and status is not DashboardGuardStatus.fail:
        issues.append(
            "Warning panels present: " + ", ".join(sorted(warning_panels))
        )
        status = _escalate(status, DashboardGuardStatus.warn)

    raw_counts = normalised.get("panel_counts", {})
    panel_counts: dict[str, int] = {}
    if isinstance(raw_counts, Mapping):
        for key, value in raw_counts.items():
            try:
                panel_counts[str(key)] = int(value)
            except (TypeError, ValueError):  # pragma: no cover - defensive guard
                continue

    present_panels = set(failing_panels) | set(warning_panels)
    present_panels |= set(str(name) for name in normalised.get("healthy_panels", ()))

    required = tuple(required_panels or ())
    missing = tuple(sorted(name for name in required if name not in present_panels))
    if missing:
        issues.append(
            "Required panels missing: " + ", ".join(missing)
        )
        status = DashboardGuardStatus.fail

    return DashboardGuardReport(
        status=status,
        generated_at=generated_at,
        age_seconds=age_seconds,
        panel_counts=panel_counts,
        failing_panels=failing_panels,
        warning_panels=warning_panels,
        missing_panels=missing,
        issues=tuple(issues),
    )


def _format_human(report: DashboardGuardReport) -> str:
    lines = [f"Status: {report.status.value.upper()}"]
    if report.generated_at is not None:
        lines.append(
            "Generated at: {timestamp}".format(
                timestamp=report.generated_at.astimezone(UTC).isoformat()
            )
        )
        if report.age_seconds is not None:
            lines.append(f"Age: {int(report.age_seconds)} seconds")
    else:
        lines.append("Generated at: unknown")

    lines.append("Panel counts: " + json.dumps(dict(report.panel_counts), sort_keys=True))
    if report.issues:
        lines.append("Issues:")
        lines.extend(f"- {issue}" for issue in report.issues)
    else:
        lines.append("Issues: none")
    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate an observability dashboard snapshot for freshness and panel coverage.",
    )
    parser.add_argument(
        "dashboard_json",
        type=Path,
        help="Path to the dashboard JSON payload",
    )
    parser.add_argument(
        "--max-age-minutes",
        type=float,
        default=30.0,
        help="Maximum allowed age (in minutes) before the snapshot is considered stale",
    )
    parser.add_argument(
        "--require-panel",
        action="append",
        default=[],
        help="Panel name that must be present (repeat for multiple panels)",
    )
    parser.add_argument(
        "--format",
        choices=("human", "json"),
        default="human",
        help="Output format",
    )
    return parser


_EXIT_CODES: Mapping[DashboardGuardStatus, int] = {
    DashboardGuardStatus.ok: 0,
    DashboardGuardStatus.warn: 1,
    DashboardGuardStatus.fail: 2,
}


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if not args.dashboard_json.exists():
        parser.error(f"Dashboard JSON not found: {args.dashboard_json}")

    try:
        payload = json.loads(args.dashboard_json.read_text())
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive guard
        parser.error(f"Failed to parse dashboard JSON: {exc}")

    report = evaluate_dashboard_health(
        payload,
        max_age=timedelta(minutes=float(args.max_age_minutes)),
        required_panels=tuple(args.require_panel or ()),
    )

    if args.format == "json":
        print(json.dumps(report.as_dict(), indent=2, sort_keys=True))
    else:
        print(_format_human(report))

    return _EXIT_CODES.get(report.status, 1)


__all__ = [
    "DashboardGuardStatus",
    "DashboardGuardReport",
    "evaluate_dashboard_health",
    "main",
]
