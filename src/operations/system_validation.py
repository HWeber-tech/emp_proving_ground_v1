from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from src.core.event_bus import Event, EventBus
from src.core.coercion import coerce_float, coerce_int
from src.operations.alerts import (
    AlertDispatchResult,
    AlertEvent,
    AlertManager,
    AlertSeverity,
)
from src.operations.event_bus_failover import publish_event_with_failover

logger = logging.getLogger(__name__)


class SystemValidationStatus(StrEnum):
    """Severity levels exposed by system validation telemetry."""

    passed = "pass"
    warn = "warn"
    fail = "fail"


_STATUS_ORDER: Mapping[SystemValidationStatus, int] = {
    SystemValidationStatus.passed: 0,
    SystemValidationStatus.warn: 1,
    SystemValidationStatus.fail: 2,
}


_SEVERITY_MAP: Mapping[SystemValidationStatus, AlertSeverity] = {
    SystemValidationStatus.passed: AlertSeverity.info,
    SystemValidationStatus.warn: AlertSeverity.warning,
    SystemValidationStatus.fail: AlertSeverity.critical,
}


def _escalate(
    current: SystemValidationStatus, candidate: SystemValidationStatus
) -> SystemValidationStatus:
    if _STATUS_ORDER[candidate] > _STATUS_ORDER[current]:
        return candidate
    return current


def _meets_threshold(
    status: SystemValidationStatus, threshold: SystemValidationStatus
) -> bool:
    return _STATUS_ORDER[status] >= _STATUS_ORDER[threshold]


def _parse_timestamp(value: object | None) -> datetime:
    if isinstance(value, datetime):
        stamp = value
    else:
        text = str(value or "").strip()
        if text.endswith("Z"):
            text = f"{text[:-1]}+00:00"
        try:
            stamp = datetime.fromisoformat(text)
        except ValueError:
            stamp = datetime.now(timezone.utc)
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=timezone.utc)
    return stamp


def _coerce_success_rate(passed_checks: int, total_checks: int, explicit: object | None) -> float:
    rate = coerce_float(explicit, default=None)
    if rate is not None:
        if rate > 1.0:
            rate = rate / 100.0
        if rate >= 0:
            return min(rate, 1.0)
    if total_checks <= 0:
        return 0.0
    return max(0.0, min(1.0, passed_checks / max(1, total_checks)))


def _coerce_status(value: object | None, default: SystemValidationStatus) -> SystemValidationStatus:
    if value is None:
        return default
    text = str(value).strip().lower()
    if not text:
        return default
    if text in {"pass", "passed", "ok", "green", "healthy"}:
        return SystemValidationStatus.passed
    if text in {"warn", "warning", "partial", "degraded"}:
        return SystemValidationStatus.warn
    if text in {"fail", "failed", "error", "critical"}:
        return SystemValidationStatus.fail
    return default


@dataclass(frozen=True)
class SystemValidationCheck:
    """Represents a single validation check outcome."""

    name: str
    passed: bool
    message: str | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {"name": self.name, "passed": self.passed}
        if self.message:
            payload["message"] = self.message
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload


@dataclass(frozen=True)
class SystemValidationSnapshot:
    """Aggregated system validation telemetry."""

    status: SystemValidationStatus
    generated_at: datetime
    total_checks: int
    passed_checks: int
    failed_checks: int
    success_rate: float
    checks: tuple[SystemValidationCheck, ...] = field(default_factory=tuple)
    metadata: Mapping[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "status": self.status.value,
            "generated_at": self.generated_at.isoformat(),
            "total_checks": self.total_checks,
            "passed_checks": self.passed_checks,
            "failed_checks": self.failed_checks,
            "success_rate": self.success_rate,
            "checks": [check.as_dict() for check in self.checks],
            "metadata": dict(self.metadata),
        }

    def to_markdown(self) -> str:
        lines = [
            f"**System validation – status: {self.status.value}**",
            f"- Generated at: {self.generated_at.isoformat()}",
            f"- Checks passed: {self.passed_checks}/{self.total_checks}",
            f"- Success rate: {self.success_rate:.2%}",
        ]
        failing_checks = [check for check in self.checks if not check.passed]
        if failing_checks:
            summary_names = [
                check.name if check.name else "unknown" for check in failing_checks
            ]
            lines.append("- Failing checks: " + ", ".join(summary_names))
        if isinstance(self.metadata, Mapping):
            validator = self.metadata.get("validator")
            if validator:
                lines.append(f"- Validator: {validator}")
            summary_message = self.metadata.get("summary_message")
            if summary_message:
                lines.append(f"- Summary: {summary_message}")
            reliability = self.metadata.get("reliability")
            if isinstance(reliability, Mapping):
                rel_status = reliability.get("status")
                if rel_status:
                    lines.append(f"- Reliability status: {rel_status}")
                stale_hours = reliability.get("stale_hours")
                if isinstance(stale_hours, (int, float)):
                    lines.append(f"- Snapshot staleness (hours): {stale_hours:.1f}")
                avg_rate = reliability.get("average_success_rate")
                if isinstance(avg_rate, (int, float)):
                    lines.append(f"- Reliability average success rate: {avg_rate:.2%}")
                fail_streak = reliability.get("fail_streak")
                if isinstance(fail_streak, int) and fail_streak > 0:
                    lines.append(f"- Consecutive FAIL streak: {fail_streak}")
                warn_streak = reliability.get("warn_streak")
                if isinstance(warn_streak, int) and warn_streak > 0:
                    lines.append(f"- Consecutive WARN streak: {warn_streak}")
                reliability_issues = reliability.get("issues")
                if isinstance(reliability_issues, Sequence) and reliability_issues:
                    lines.append("")
                    lines.append("**Reliability issues:**")
                    for issue in reliability_issues:
                        lines.append(f"- {issue}")
        if self.checks:
            lines.append("")
            lines.append("**Checks:**")
            for check in self.checks:
                icon = "✅" if check.passed else "❌"
                message_part = f": {check.message}" if check.message else ""
                lines.append(f"- {icon} {check.name}{message_part}")
        return "\n".join(lines)


@dataclass(frozen=True)
class SystemValidationHistoryEntry:
    """Historical record used for reliability calculations."""

    generated_at: datetime
    status: SystemValidationStatus
    success_rate: float

    def as_dict(self) -> dict[str, object]:
        return {
            "generated_at": self.generated_at.isoformat(),
            "status": self.status.value,
            "success_rate": self.success_rate,
        }

    @classmethod
    def from_snapshot(cls, snapshot: SystemValidationSnapshot) -> "SystemValidationHistoryEntry":
        return cls(
            generated_at=snapshot.generated_at,
            status=snapshot.status,
            success_rate=snapshot.success_rate,
        )

    @classmethod
    def from_mapping(
        cls, payload: Mapping[str, object] | None
    ) -> "SystemValidationHistoryEntry | None":
        if not isinstance(payload, Mapping):
            return None
        try:
            generated_at = _parse_timestamp(payload.get("generated_at"))
            status = _coerce_status(payload.get("status"), SystemValidationStatus.warn)
            success_rate_value = coerce_float(payload.get("success_rate"), default=None)
        except (TypeError, ValueError) as exc:  # pragma: no cover - defensive guard
            logger.debug("Invalid system validation history payload: %s", payload, exc_info=exc)
            return None
        if success_rate_value is None:
            success_rate_value = 0.0
        return cls(
            generated_at=generated_at,
            status=status,
            success_rate=max(0.0, min(1.0, float(success_rate_value))),
        )


@dataclass(frozen=True)
class SystemValidationReliabilitySummary:
    """Summary describing reliability of recent system validation runs."""

    status: SystemValidationStatus
    evaluated_at: datetime
    window_hours: float
    samples_considered: int
    fail_streak: int
    warn_streak: int
    average_success_rate: float | None
    latest_timestamp: datetime | None
    stale_hours: float | None
    issues: tuple[str, ...]
    metadata: Mapping[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "status": self.status.value,
            "evaluated_at": self.evaluated_at.isoformat(),
            "window_hours": self.window_hours,
            "samples_considered": self.samples_considered,
            "fail_streak": self.fail_streak,
            "warn_streak": self.warn_streak,
            "average_success_rate": self.average_success_rate,
            "stale_hours": self.stale_hours,
            "issues": list(self.issues),
        }
        if self.latest_timestamp is not None:
            payload["latest_timestamp"] = self.latest_timestamp.isoformat()
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload


def format_system_validation_markdown(snapshot: SystemValidationSnapshot) -> str:
    """Convenience wrapper mirroring other operational formatters."""

    return snapshot.to_markdown()


def _extract_checks(results: object | None) -> Iterable[SystemValidationCheck]:
    if isinstance(results, Mapping):
        for name, value in results.items():
            message = None
            metadata_dict: dict[str, object] = {}
            passed = bool(value)
            if isinstance(value, Mapping):
                passed = bool(value.get("passed", value.get("status", value)))
                if "message" in value:
                    message = str(value["message"])
                metadata_dict = {
                    key: val
                    for key, val in value.items()
                    if key not in {"passed", "status", "message"}
                }
            yield SystemValidationCheck(
                name=str(name),
                passed=passed,
                message=message,
                metadata=metadata_dict,
            )
    elif isinstance(results, Sequence):
        for entry in results:
            if not isinstance(entry, Mapping):
                continue
            name = str(entry.get("name") or entry.get("check") or "unknown")
            passed = bool(entry.get("passed", entry.get("status")))
            message = entry.get("message")
            metadata_dict = {
                key: value
                for key, value in entry.items()
                if key not in {"name", "check", "passed", "status", "message"}
            }
            yield SystemValidationCheck(
                name=name,
                passed=passed,
                message=str(message) if message is not None else None,
                metadata=metadata_dict,
            )


def _normalise_success_threshold(value: float | None) -> float | None:
    if value is None:
        return None
    if value > 1.0:
        return min(1.0, value / 100.0)
    if value < 0.0:
        return 0.0
    return value


def _summarise_system_validation_reliability(
    entries: Sequence[SystemValidationHistoryEntry],
    *,
    now: datetime,
    window_hours: float,
    stale_warn_hours: float | None,
    stale_fail_hours: float | None,
    warn_success_rate: float,
    fail_success_rate: float,
    warn_fail_streak: int,
    fail_fail_streak: int,
) -> SystemValidationReliabilitySummary:
    evaluated_at = now
    if not entries:
        issues = ("No system validation history available",)
        return SystemValidationReliabilitySummary(
            status=SystemValidationStatus.fail,
            evaluated_at=evaluated_at,
            window_hours=window_hours,
            samples_considered=0,
            fail_streak=0,
            warn_streak=0,
            average_success_rate=None,
            latest_timestamp=None,
            stale_hours=None,
            issues=issues,
            metadata={
                "reason": "no_history",
                "warn_success_rate": warn_success_rate,
                "fail_success_rate": fail_success_rate,
            },
        )

    ordered = sorted(entries, key=lambda entry: entry.generated_at)
    latest_entry = ordered[-1]
    stale_delta = now - latest_entry.generated_at
    stale_hours = max(stale_delta.total_seconds() / 3600.0, 0.0)

    status = SystemValidationStatus.passed
    issues: list[str] = []

    if stale_fail_hours is not None and stale_hours > stale_fail_hours:
        status = SystemValidationStatus.fail
        issues.append(
            f"Latest system validation snapshot stale {stale_hours:.1f}h (fail>={stale_fail_hours:.1f}h)"
        )
    elif stale_warn_hours is not None and stale_hours > stale_warn_hours:
        status = _escalate(status, SystemValidationStatus.warn)
        issues.append(
            f"Latest system validation snapshot stale {stale_hours:.1f}h (warn>={stale_warn_hours:.1f}h)"
        )

    fail_streak = 0
    for entry in reversed(ordered):
        if entry.status is SystemValidationStatus.fail:
            fail_streak += 1
        else:
            break

    warn_streak = 0
    for entry in reversed(ordered):
        if entry.status is SystemValidationStatus.warn:
            warn_streak += 1
        else:
            break

    if fail_fail_streak > 0 and fail_streak >= fail_fail_streak:
        status = SystemValidationStatus.fail
        issues.append(f"{fail_streak} consecutive FAIL snapshots")
    elif warn_fail_streak > 0 and fail_streak >= warn_fail_streak:
        status = _escalate(status, SystemValidationStatus.warn)
        issues.append(f"{fail_streak} consecutive FAIL snapshots")

    if warn_streak >= 2 and fail_streak == 0:
        status = _escalate(status, SystemValidationStatus.warn)
        issues.append(f"{warn_streak} consecutive WARN snapshots")

    if window_hours <= 0:
        window_cutoff = None
    else:
        window_cutoff = now - timedelta(hours=window_hours)

    if window_cutoff is None:
        window_entries = list(ordered)
    else:
        window_entries = [entry for entry in ordered if entry.generated_at >= window_cutoff]

    samples_considered = len(window_entries)
    average_success_rate: float | None
    if window_entries:
        average_success_rate = (
            sum(entry.success_rate for entry in window_entries) / samples_considered
        )
        fail_threshold = _normalise_success_threshold(fail_success_rate) or 0.0
        warn_threshold = _normalise_success_threshold(warn_success_rate) or 0.0
        if average_success_rate < fail_threshold:
            status = SystemValidationStatus.fail
            issues.append(
                f"Average success rate {average_success_rate:.0%} below fail threshold {fail_threshold:.0%}"
            )
        elif average_success_rate < warn_threshold:
            status = _escalate(status, SystemValidationStatus.warn)
            issues.append(
                f"Average success rate {average_success_rate:.0%} below warn threshold {warn_threshold:.0%}"
            )
    else:
        average_success_rate = None
        issues.append("No system validation runs within reliability window")
        status = _escalate(status, SystemValidationStatus.warn)

    metadata = {
        "warn_success_rate": warn_success_rate,
        "fail_success_rate": fail_success_rate,
        "warn_fail_streak": warn_fail_streak,
        "fail_fail_streak": fail_fail_streak,
        "stale_warn_hours": stale_warn_hours,
        "stale_fail_hours": stale_fail_hours,
    }

    return SystemValidationReliabilitySummary(
        status=status,
        evaluated_at=evaluated_at,
        window_hours=window_hours,
        samples_considered=samples_considered,
        fail_streak=fail_streak,
        warn_streak=warn_streak,
        average_success_rate=average_success_rate,
        latest_timestamp=latest_entry.generated_at,
        stale_hours=stale_hours,
        issues=tuple(issues),
        metadata=metadata,
    )


def evaluate_system_validation(
    report: Mapping[str, object],
    *,
    metadata: Mapping[str, object] | None = None,
    history: Sequence[SystemValidationSnapshot | Mapping[str, object]] | None = None,
    reliability_window_hours: float = 72.0,
    reliability_stale_warn_hours: float | None = 24.0,
    reliability_stale_fail_hours: float | None = 48.0,
    reliability_warn_success_rate: float = 0.95,
    reliability_fail_success_rate: float = 0.85,
    reliability_warn_fail_streak: int = 1,
    reliability_fail_fail_streak: int = 2,
    now: datetime | None = None,
) -> SystemValidationSnapshot:
    generated_at = _parse_timestamp(report.get("timestamp"))
    checks = tuple(_extract_checks(report.get("results") or report.get("checks")))
    total_checks_value = coerce_int(report.get("total_checks"), default=None)
    total_checks = total_checks_value if total_checks_value is not None else len(checks)
    if total_checks < len(checks):
        total_checks = len(checks)
    if not checks and total_checks > 0:
        # Create placeholder checks from boolean mapping if present
        raw_results = report.get("results")
        if isinstance(raw_results, Mapping):
            checks = tuple(
                SystemValidationCheck(name=str(name), passed=bool(value))
                for name, value in raw_results.items()
            )
            total_checks = len(checks)
    passed_checks_value = coerce_int(report.get("passed_checks"), default=None)
    passed_checks = (
        passed_checks_value
        if passed_checks_value is not None
        else sum(1 for check in checks if check.passed)
    )
    failed_checks = max(0, total_checks - passed_checks)
    success_rate = _coerce_success_rate(passed_checks, total_checks, report.get("success_rate"))

    computed_status = SystemValidationStatus.fail
    if total_checks == 0:
        computed_status = SystemValidationStatus.warn
    elif passed_checks == total_checks:
        computed_status = SystemValidationStatus.passed
    elif passed_checks > 0:
        computed_status = SystemValidationStatus.warn

    summary = report.get("summary")
    summary_status = None
    summary_message = None
    if isinstance(summary, Mapping):
        summary_status = summary.get("status")
        summary_message = summary.get("message")

    reported_status = report.get("status")
    status = computed_status
    status = _escalate(status, _coerce_status(summary_status, status))
    status = _escalate(status, _coerce_status(reported_status, status))

    snapshot_metadata: dict[str, object] = {}
    if isinstance(metadata, Mapping):
        snapshot_metadata.update(dict(metadata))
    validator = report.get("validator")
    if validator is not None:
        snapshot_metadata.setdefault("validator", str(validator))
    version = report.get("version")
    if version is not None:
        snapshot_metadata.setdefault("version", str(version))
    if summary_message:
        snapshot_metadata.setdefault("summary_message", str(summary_message))

    failing_checks = tuple(check for check in checks if not check.passed)
    if failing_checks:
        snapshot_metadata.setdefault(
            "failing_checks",
            tuple(
                {
                    "name": check.name,
                    "message": check.message,
                    "metadata": dict(check.metadata),
                }
                for check in failing_checks
            ),
        )
        snapshot_metadata.setdefault(
            "failing_check_names",
            tuple(check.name for check in failing_checks),
        )

    if total_checks:
        snapshot_metadata.setdefault(
            "check_status_breakdown",
            {
                "passed": passed_checks,
                "failed": failed_checks,
            },
        )

    current_entry = SystemValidationHistoryEntry(
        generated_at=generated_at,
        status=status,
        success_rate=success_rate,
    )

    history_entries: list[SystemValidationHistoryEntry] = [current_entry]
    if history:
        for item in history:
            if isinstance(item, SystemValidationSnapshot):
                history_entries.append(SystemValidationHistoryEntry.from_snapshot(item))
            elif isinstance(item, Mapping):
                entry = SystemValidationHistoryEntry.from_mapping(item)
                if entry is not None:
                    history_entries.append(entry)

    now_dt = now or datetime.now(timezone.utc)
    reliability_summary = _summarise_system_validation_reliability(
        history_entries,
        now=now_dt,
        window_hours=reliability_window_hours,
        stale_warn_hours=reliability_stale_warn_hours,
        stale_fail_hours=reliability_stale_fail_hours,
        warn_success_rate=reliability_warn_success_rate,
        fail_success_rate=reliability_fail_success_rate,
        warn_fail_streak=reliability_warn_fail_streak,
        fail_fail_streak=reliability_fail_fail_streak,
    )

    status = _escalate(status, reliability_summary.status)
    snapshot_metadata["reliability"] = reliability_summary.as_dict()
    if reliability_summary.issues:
        snapshot_metadata.setdefault("reliability_issues", reliability_summary.issues)
        reliability_entries = tuple(
            {
                "category": "reliability",
                "severity": reliability_summary.status.value,
                "message": issue,
            }
            for issue in reliability_summary.issues
        )
        existing_details = list(snapshot_metadata.get("issue_details", ()))
        existing_details.extend(reliability_entries)
        snapshot_metadata["issue_details"] = tuple(existing_details)
        catalog = dict(snapshot_metadata.get("issue_catalog", {}))
        catalog["reliability"] = reliability_entries
        snapshot_metadata["issue_catalog"] = catalog
        counts = dict(snapshot_metadata.get("issue_counts", {}))
        counts[reliability_summary.status.value] = (
            counts.get(reliability_summary.status.value, 0) + len(reliability_summary.issues)
        )
        snapshot_metadata["issue_counts"] = counts
        try:
            highest = max(
                counts,
                key=lambda value: _STATUS_ORDER[SystemValidationStatus(value)],
            )
            snapshot_metadata["highest_issue_severity"] = highest
        except ValueError:
            pass

    return SystemValidationSnapshot(
        status=status,
        generated_at=generated_at,
        total_checks=total_checks,
        passed_checks=passed_checks,
        failed_checks=failed_checks,
        success_rate=success_rate,
        checks=checks,
        metadata=snapshot_metadata,
    )


def derive_system_validation_alerts(
    snapshot: SystemValidationSnapshot,
    *,
    threshold: SystemValidationStatus = SystemValidationStatus.warn,
    include_status_event: bool = True,
    include_failing_checks: bool = True,
    include_reliability_event: bool = True,
    include_gate_event: bool = False,
    gate_result: SystemValidationGateResult | None = None,
    base_tags: Sequence[str] = ("system-validation",),
) -> list[AlertEvent]:
    """Translate a system validation snapshot into alert events."""

    events: list[AlertEvent] = []
    tags = tuple(base_tags)
    payload = snapshot.as_dict()

    if include_status_event and _meets_threshold(snapshot.status, threshold):
        events.append(
            AlertEvent(
                category="system_validation.status",
                severity=_SEVERITY_MAP[snapshot.status],
                message=f"System validation status {snapshot.status.value}",
                tags=tags,
                context={"snapshot": payload},
            )
        )

    if include_failing_checks:
        failing_checks = [check for check in snapshot.checks if not check.passed]
        if failing_checks:
            detail_status = (
                snapshot.status
                if snapshot.status is SystemValidationStatus.fail
                else SystemValidationStatus.warn
            )
            if _meets_threshold(detail_status, threshold):
                severity = _SEVERITY_MAP[detail_status]
                for check in failing_checks:
                    message = f"Validation check failed: {check.name}"
                    if check.message:
                        message += f" – {check.message}"
                    events.append(
                        AlertEvent(
                            category="system_validation.check",
                            severity=severity,
                            message=message,
                            tags=tags + ("check",),
                            context={
                                "snapshot": payload,
                                "check": check.as_dict(),
                            },
                        )
                    )

    if include_reliability_event:
        metadata = snapshot.metadata if isinstance(snapshot.metadata, Mapping) else {}
        reliability = metadata.get("reliability") if isinstance(metadata, Mapping) else None
        if isinstance(reliability, Mapping):
            raw_status = reliability.get("status", snapshot.status.value)
            try:
                reliability_status = SystemValidationStatus(str(raw_status))
            except ValueError:
                reliability_status = None
            if reliability_status is not None and _meets_threshold(reliability_status, threshold):
                issues = reliability.get("issues")
                if isinstance(issues, Sequence) and issues:
                    message = str(issues[0])
                else:
                    message = f"System validation reliability {reliability_status.value}"
                events.append(
                    AlertEvent(
                        category="system_validation.reliability",
                        severity=_SEVERITY_MAP[reliability_status],
                        message=message,
                        tags=tags + ("reliability",),
                        context={
                            "snapshot": payload,
                            "reliability": dict(reliability),
                        },
                    )
                )

    if include_gate_event:
        evaluated_gate = gate_result
        if evaluated_gate is None:
            evaluated_gate = evaluate_system_validation_gate(snapshot)
        gate_status = (
            SystemValidationStatus.fail if evaluated_gate.should_block else evaluated_gate.status
        )
        if evaluated_gate.should_block or _meets_threshold(gate_status, threshold):
            if evaluated_gate.reasons:
                headline = evaluated_gate.reasons[0]
            else:
                headline = f"System validation gate {gate_status.value}"
            events.append(
                AlertEvent(
                    category="system_validation.gate",
                    severity=_SEVERITY_MAP[gate_status],
                    message=headline,
                    tags=tags + ("gate",),
                    context={
                        "snapshot": payload,
                        "gate": evaluated_gate.as_dict(),
                    },
                )
            )

    return events


def route_system_validation_alerts(
    manager: AlertManager,
    snapshot: SystemValidationSnapshot,
    *,
    threshold: SystemValidationStatus = SystemValidationStatus.warn,
    include_status_event: bool = True,
    include_failing_checks: bool = True,
    include_reliability_event: bool = True,
    include_gate_event: bool = False,
    gate_result: SystemValidationGateResult | None = None,
    base_tags: Sequence[str] = ("system-validation",),
) -> list[AlertDispatchResult]:
    """Dispatch system validation alerts via an alert manager."""

    evaluated_gate = gate_result
    if include_gate_event and evaluated_gate is None:
        evaluated_gate = evaluate_system_validation_gate(snapshot)

    events = derive_system_validation_alerts(
        snapshot,
        threshold=threshold,
        include_status_event=include_status_event,
        include_failing_checks=include_failing_checks,
        include_reliability_event=include_reliability_event,
        include_gate_event=include_gate_event,
        gate_result=evaluated_gate,
        base_tags=base_tags,
    )
    results: list[AlertDispatchResult] = []
    for event in events:
        results.append(manager.dispatch(event))
    return results


def publish_system_validation_snapshot(
    event_bus: EventBus,
    snapshot: SystemValidationSnapshot,
    *,
    source: str = "operations.system_validation",
) -> None:
    """Publish the system validation snapshot onto the runtime bus."""

    event = Event(
        type="telemetry.operational.system_validation",
        payload=snapshot.as_dict(),
        source=source,
    )

    publish_event_with_failover(
        event_bus,
        event,
        logger=logger,
        runtime_fallback_message=
        "Runtime event bus rejected system validation snapshot; falling back to global bus",
        runtime_unexpected_message=
        "Unexpected error publishing system validation snapshot via runtime bus",
        runtime_none_message=
        "Runtime event bus returned None while publishing system validation snapshot; using global bus",
        global_not_running_message=
        "Global event bus not running while publishing system validation snapshot",
        global_unexpected_message=
        "Unexpected error publishing system validation snapshot via global bus",
    )


def load_system_validation_snapshot(
    path: str | Path,
    *,
    metadata: Mapping[str, object] | None = None,
) -> SystemValidationSnapshot | None:
    candidate = Path(path)
    try:
        text = candidate.read_text(encoding="utf-8")
    except FileNotFoundError:
        logger.debug("System validation report missing at %s", candidate)
        return None
    except OSError:  # pragma: no cover - defensive logging
        logger.debug("System validation report unreadable at %s", candidate, exc_info=True)
        return None
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        logger.warning("System validation report at %s is not valid JSON", candidate)
        return None
    if not isinstance(payload, Mapping):
        logger.debug("System validation report payload is not a mapping at %s", candidate)
        return None
    merged_metadata: dict[str, object] = {"report_path": str(candidate)}
    if isinstance(metadata, Mapping):
        merged_metadata.update(dict(metadata))
    return evaluate_system_validation(payload, metadata=merged_metadata)


@dataclass(frozen=True)
class SystemValidationGateResult:
    """Gate evaluation result for system validation telemetry."""

    status: SystemValidationStatus
    should_block: bool
    reasons: tuple[str, ...]
    reason_codes: tuple[str, ...] = field(default_factory=tuple)
    metadata: Mapping[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "status": self.status.value,
            "should_block": self.should_block,
            "reasons": list(self.reasons),
            "reason_codes": list(self.reason_codes),
            "metadata": dict(self.metadata),
        }


def _reason_code(*parts: object) -> str:
    tokens: list[str] = []
    for part in parts:
        text = re.sub(r"[^a-z0-9]+", "_", str(part).lower()).strip("_")
        if not text:
            text = "unknown"
        tokens.append(text)
    return "_".join(tokens)


def evaluate_system_validation_gate(
    snapshot: SystemValidationSnapshot,
    *,
    min_success_rate: float = 0.95,
    block_on_warn: bool = False,
    required_checks: Sequence[str] = (),
    consider_reliability: bool = False,
    reliability_block_on_warn: bool = False,
    reliability_max_stale_hours: float | None = None,
) -> SystemValidationGateResult:
    """Determine whether a snapshot meets deployment expectations."""

    reasons: list[str] = []
    reason_codes: list[str] = []
    gate_status = snapshot.status

    if snapshot.status is SystemValidationStatus.fail:
        reasons.append("System validation status is FAIL")
        reason_codes.append(_reason_code("status", "fail"))
        gate_status = _escalate(gate_status, SystemValidationStatus.fail)
    elif block_on_warn and snapshot.status is SystemValidationStatus.warn:
        reasons.append("System validation status is WARN and block_on_warn is enabled")
        reason_codes.append(_reason_code("status", "warn", "blocked"))
        gate_status = _escalate(gate_status, SystemValidationStatus.warn)

    if snapshot.success_rate < min_success_rate:
        reasons.append(
            "Success rate "
            f"{snapshot.success_rate:.1%} below minimum {min_success_rate:.1%}"
        )
        reason_codes.append(_reason_code("success_rate", "below_min"))
        gate_status = _escalate(gate_status, SystemValidationStatus.fail)

    if required_checks:
        checks_by_name = {check.name.lower(): check for check in snapshot.checks}
        for required in required_checks:
            key = required.lower()
            check = checks_by_name.get(key)
            if check is None:
                reasons.append(f"Required check missing from snapshot: {required}")
                reason_codes.append(_reason_code("required_check", "missing", required))
                gate_status = _escalate(gate_status, SystemValidationStatus.fail)
                continue
            if not check.passed:
                reasons.append(f"Required check failed: {check.name}")
                reason_codes.append(_reason_code("required_check", "failed", check.name))
                gate_status = _escalate(gate_status, SystemValidationStatus.fail)

    metadata = snapshot.metadata if isinstance(snapshot.metadata, Mapping) else {}
    reliability_metadata: dict[str, object] | None = None
    reliability_warnings: list[str] = []

    if consider_reliability:
        raw_reliability = metadata.get("reliability") if isinstance(metadata, Mapping) else None
        if isinstance(raw_reliability, Mapping):
            reliability_metadata = dict(raw_reliability)
            raw_status = raw_reliability.get("status")
            reliability_status: SystemValidationStatus | None
            try:
                reliability_status = SystemValidationStatus(str(raw_status))
            except ValueError:
                reliability_status = None

            if reliability_status is SystemValidationStatus.fail:
                reasons.append("System validation reliability status is FAIL")
                reason_codes.append(_reason_code("reliability", "status", "fail"))
                gate_status = _escalate(gate_status, SystemValidationStatus.fail)
            elif reliability_status is SystemValidationStatus.warn:
                message = "System validation reliability status is WARN"
                gate_status = _escalate(gate_status, SystemValidationStatus.warn)
                if reliability_block_on_warn:
                    reasons.append(message)
                    reason_codes.append(_reason_code("reliability", "status", "warn", "blocked"))
                else:
                    reliability_warnings.append(message)

            stale_hours = raw_reliability.get("stale_hours")
            if isinstance(stale_hours, (int, float)) and reliability_max_stale_hours is not None:
                if stale_hours > reliability_max_stale_hours:
                    reasons.append(
                        "Reliability snapshot stale "
                        f"{stale_hours:.1f} hours exceeds limit {reliability_max_stale_hours:.1f} hours"
                    )
                    reason_codes.append(_reason_code("reliability", "snapshot", "stale"))
                    gate_status = _escalate(gate_status, SystemValidationStatus.fail)
        else:
            reliability_warnings.append("System validation reliability metadata unavailable")

    gate_metadata: dict[str, object] = {
        "min_success_rate": min_success_rate,
        "block_on_warn": block_on_warn,
        "required_checks": tuple(required_checks),
        "success_rate": snapshot.success_rate,
        "snapshot_metadata": dict(snapshot.metadata),
    }
    if consider_reliability:
        gate_metadata["consider_reliability"] = True
        gate_metadata["reliability_block_on_warn"] = reliability_block_on_warn
        if reliability_max_stale_hours is not None:
            gate_metadata["reliability_max_stale_hours"] = reliability_max_stale_hours
        if reliability_metadata is not None:
            gate_metadata["reliability_snapshot"] = reliability_metadata
        if reliability_warnings:
            gate_metadata["reliability_warnings"] = tuple(reliability_warnings)

    return SystemValidationGateResult(
        status=gate_status,
        should_block=bool(reasons),
        reasons=tuple(reasons),
        reason_codes=tuple(reason_codes),
        metadata=gate_metadata,
    )


__all__ = [
    "SystemValidationStatus",
    "SystemValidationCheck",
    "SystemValidationSnapshot",
    "SystemValidationHistoryEntry",
    "SystemValidationReliabilitySummary",
    "SystemValidationGateResult",
    "derive_system_validation_alerts",
    "evaluate_system_validation",
    "evaluate_system_validation_gate",
    "format_system_validation_markdown",
    "route_system_validation_alerts",
    "publish_system_validation_snapshot",
    "load_system_validation_snapshot",
]
