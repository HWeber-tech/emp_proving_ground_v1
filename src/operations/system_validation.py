from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
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
        if self.checks:
            lines.append("")
            lines.append("**Checks:**")
            for check in self.checks:
                icon = "✅" if check.passed else "❌"
                message_part = f": {check.message}" if check.message else ""
                lines.append(f"- {icon} {check.name}{message_part}")
        return "\n".join(lines)


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


def evaluate_system_validation(
    report: Mapping[str, object],
    *,
    metadata: Mapping[str, object] | None = None,
) -> SystemValidationSnapshot:
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

    return SystemValidationSnapshot(
        status=status,
        generated_at=_parse_timestamp(report.get("timestamp")),
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

    return events


def route_system_validation_alerts(
    manager: AlertManager,
    snapshot: SystemValidationSnapshot,
    *,
    threshold: SystemValidationStatus = SystemValidationStatus.warn,
    include_status_event: bool = True,
    include_failing_checks: bool = True,
    base_tags: Sequence[str] = ("system-validation",),
) -> list[AlertDispatchResult]:
    """Dispatch system validation alerts via an alert manager."""

    events = derive_system_validation_alerts(
        snapshot,
        threshold=threshold,
        include_status_event=include_status_event,
        include_failing_checks=include_failing_checks,
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
    metadata: Mapping[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "status": self.status.value,
            "should_block": self.should_block,
            "reasons": list(self.reasons),
            "metadata": dict(self.metadata),
        }


def evaluate_system_validation_gate(
    snapshot: SystemValidationSnapshot,
    *,
    min_success_rate: float = 0.95,
    block_on_warn: bool = False,
    required_checks: Sequence[str] = (),
) -> SystemValidationGateResult:
    """Determine whether a snapshot meets deployment expectations."""

    reasons: list[str] = []

    if snapshot.status is SystemValidationStatus.fail:
        reasons.append("System validation status is FAIL")
    elif block_on_warn and snapshot.status is SystemValidationStatus.warn:
        reasons.append("System validation status is WARN and block_on_warn is enabled")

    if snapshot.success_rate < min_success_rate:
        reasons.append(
            "Success rate "
            f"{snapshot.success_rate:.1%} below minimum {min_success_rate:.1%}"
        )

    if required_checks:
        checks_by_name = {check.name.lower(): check for check in snapshot.checks}
        for required in required_checks:
            key = required.lower()
            check = checks_by_name.get(key)
            if check is None:
                reasons.append(f"Required check missing from snapshot: {required}")
                continue
            if not check.passed:
                reasons.append(f"Required check failed: {check.name}")

    gate_metadata: dict[str, object] = {
        "min_success_rate": min_success_rate,
        "block_on_warn": block_on_warn,
        "required_checks": tuple(required_checks),
        "success_rate": snapshot.success_rate,
        "snapshot_metadata": dict(snapshot.metadata),
    }

    return SystemValidationGateResult(
        status=snapshot.status,
        should_block=bool(reasons),
        reasons=tuple(reasons),
        metadata=gate_metadata,
    )


__all__ = [
    "SystemValidationStatus",
    "SystemValidationCheck",
    "SystemValidationSnapshot",
    "SystemValidationGateResult",
    "derive_system_validation_alerts",
    "evaluate_system_validation",
    "evaluate_system_validation_gate",
    "format_system_validation_markdown",
    "route_system_validation_alerts",
    "publish_system_validation_snapshot",
    "load_system_validation_snapshot",
]
