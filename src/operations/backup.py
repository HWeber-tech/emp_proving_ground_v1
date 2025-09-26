"""Backup readiness evaluation and telemetry helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Iterable, Mapping


class BackupStatus(StrEnum):
    """Severity levels for backup readiness."""

    ok = "ok"
    warn = "warn"
    fail = "fail"


_STATUS_ORDER: Mapping[BackupStatus, int] = {
    BackupStatus.ok: 0,
    BackupStatus.warn: 1,
    BackupStatus.fail: 2,
}


def _escalate(current: BackupStatus, candidate: BackupStatus) -> BackupStatus:
    if _STATUS_ORDER[candidate] > _STATUS_ORDER[current]:
        return candidate
    return current


@dataclass(frozen=True)
class BackupPolicy:
    """Declarative policy describing backup expectations."""

    enabled: bool = True
    expected_frequency_seconds: float = 86_400.0
    retention_days: int = 7
    minimum_retention_days: int = 7
    restore_test_interval_days: int = 30
    warn_after_seconds: float | None = None
    fail_after_seconds: float | None = None
    providers: tuple[str, ...] = ()
    storage_location: str | None = None


@dataclass(frozen=True)
class BackupState:
    """Observed backup execution state."""

    last_backup_at: datetime | None = None
    last_backup_status: str | None = None
    last_restore_test_at: datetime | None = None
    last_restore_status: str | None = None
    recorded_failures: tuple[str, ...] = ()


@dataclass(frozen=True)
class BackupReadinessSnapshot:
    """Aggregated backup readiness telemetry."""

    service: str
    generated_at: datetime
    status: BackupStatus
    latest_backup_at: datetime | None
    next_backup_due_at: datetime | None
    retention_days: int
    issues: tuple[str, ...] = field(default_factory=tuple)
    metadata: dict[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "service": self.service,
            "generated_at": self.generated_at.isoformat(),
            "status": self.status.value,
            "retention_days": self.retention_days,
            "issues": list(self.issues),
            "metadata": dict(self.metadata),
        }
        if self.latest_backup_at is not None:
            payload["latest_backup_at"] = self.latest_backup_at.isoformat()
        if self.next_backup_due_at is not None:
            payload["next_backup_due_at"] = self.next_backup_due_at.isoformat()
        return payload

    def to_markdown(self) -> str:
        lines = [
            f"**Backup readiness – {self.service}**",
            f"- Status: {self.status.value}",
            f"- Generated: {self.generated_at.isoformat()}",
            f"- Retention days: {self.retention_days}",
        ]
        if self.latest_backup_at is not None:
            lines.append(f"- Latest backup: {self.latest_backup_at.isoformat()}")
        else:
            lines.append("- Latest backup: not recorded")
        if self.next_backup_due_at is not None:
            lines.append(f"- Next backup due: {self.next_backup_due_at.isoformat()}")
        if self.metadata:
            policy_metadata = self.metadata.get("policy")
            if isinstance(policy_metadata, Mapping):
                providers = policy_metadata.get("providers")
                if isinstance(providers, Iterable):
                    formatted = [str(provider) for provider in providers]
                    if formatted:
                        lines.append(f"- Providers: {', '.join(formatted)}")
                storage = policy_metadata.get("storage_location")
                if storage:
                    lines.append(f"- Storage: {storage}")
        if self.issues:
            lines.append("")
            lines.append("**Issues:**")
            for issue in self.issues:
                lines.append(f"- {issue}")
        return "\n".join(lines)


def format_backup_markdown(snapshot: BackupReadinessSnapshot) -> str:
    """Convenience wrapper mirroring other operational formatters."""

    return snapshot.to_markdown()


def _normalise_status(value: str | None) -> str:
    if value is None:
        return ""
    return str(value).strip().lower()


def _compute_due_at(
    last_backup_at: datetime | None, expected_frequency_seconds: float
) -> datetime | None:
    if last_backup_at is None:
        return None
    return last_backup_at + timedelta(seconds=max(expected_frequency_seconds, 0.0))


def evaluate_backup_readiness(
    policy: BackupPolicy,
    state: BackupState,
    *,
    service: str = "timescale_backups",
    now: datetime | None = None,
    metadata: Mapping[str, object] | None = None,
) -> BackupReadinessSnapshot:
    """Assess backup posture against the supplied policy and state."""

    moment = now or datetime.now(tz=UTC)
    issues: list[str] = []
    status = BackupStatus.ok

    expected_frequency = max(policy.expected_frequency_seconds, 1.0)
    warn_after = policy.warn_after_seconds or expected_frequency * 1.5
    fail_after = policy.fail_after_seconds or expected_frequency * 2.5
    latest_backup = state.last_backup_at
    next_due = _compute_due_at(latest_backup, expected_frequency)

    if not policy.enabled:
        status = BackupStatus.fail
        issues.append("Backups are disabled in configuration")

    if policy.retention_days < policy.minimum_retention_days:
        status = _escalate(status, BackupStatus.warn)
        issues.append(
            "Retention window below minimum target"
            f" ({policy.retention_days}d < {policy.minimum_retention_days}d)"
        )

    backup_state = _normalise_status(state.last_backup_status)
    if backup_state in {"fail", "failed", "error"}:
        status = BackupStatus.fail
        issues.append("Last backup reported failure")
    elif backup_state in {"partial", "warn", "warning"}:
        status = _escalate(status, BackupStatus.warn)
        issues.append("Last backup completed with warnings")

    if latest_backup is None:
        status = BackupStatus.fail
        issues.append("No successful backup recorded")
    else:
        age_seconds = (moment - latest_backup).total_seconds()
        if age_seconds > fail_after:
            status = BackupStatus.fail
            issues.append(
                f"Last backup exceeds failure threshold ({int(age_seconds)}s > {int(fail_after)}s)"
            )
        elif age_seconds > warn_after:
            status = _escalate(status, BackupStatus.warn)
            issues.append(
                f"Last backup exceeds warning threshold ({int(age_seconds)}s > {int(warn_after)}s)"
            )

    if state.recorded_failures:
        status = _escalate(status, BackupStatus.warn)
        unique_failures = ", ".join(dict.fromkeys(state.recorded_failures))
        issues.append(f"Recent backup failures: {unique_failures}")

    restore_interval_days = max(policy.restore_test_interval_days, 0)
    if restore_interval_days:
        allowed_seconds = restore_interval_days * 86_400
        restore_state = _normalise_status(state.last_restore_status)
        if state.last_restore_test_at is None:
            status = _escalate(status, BackupStatus.warn)
            issues.append("No restore drill recorded within policy window")
        else:
            restore_age = (moment - state.last_restore_test_at).total_seconds()
            if restore_age > allowed_seconds * 2:
                status = BackupStatus.fail
                issues.append(
                    f"Restore drill overdue ({int(restore_age)}s > {int(allowed_seconds * 2)}s)"
                )
            elif restore_age > allowed_seconds:
                status = _escalate(status, BackupStatus.warn)
                issues.append(
                    "Restore drill past target window"
                    f" ({int(restore_age)}s > {int(allowed_seconds)}s)"
                )
        if restore_state in {"fail", "failed", "error"}:
            status = BackupStatus.fail
            issues.append("Last restore drill failed")
        elif restore_state in {"partial", "warn", "warning"}:
            status = _escalate(status, BackupStatus.warn)
            issues.append("Last restore drill completed with warnings")

    policy_payload: dict[str, object] = {
        "enabled": policy.enabled,
        "expected_frequency_seconds": expected_frequency,
        "warn_after_seconds": warn_after,
        "fail_after_seconds": fail_after,
        "retention_days": policy.retention_days,
        "minimum_retention_days": policy.minimum_retention_days,
        "restore_test_interval_days": policy.restore_test_interval_days,
        "providers": list(policy.providers),
        "storage_location": policy.storage_location,
    }
    state_payload: dict[str, object] = {
        "last_backup_status": state.last_backup_status,
        "last_restore_status": state.last_restore_status,
        "recorded_failures": list(state.recorded_failures),
    }
    if latest_backup is not None:
        state_payload["last_backup_at"] = latest_backup.isoformat()
    if state.last_restore_test_at is not None:
        state_payload["last_restore_test_at"] = state.last_restore_test_at.isoformat()

    metadata_payload: dict[str, object] = {
        "policy": policy_payload,
        "state": state_payload,
    }
    if metadata:
        metadata_payload["context"] = dict(metadata)

    unique_issues = tuple(dict.fromkeys(issues))

    return BackupReadinessSnapshot(
        service=service,
        generated_at=moment,
        status=status,
        latest_backup_at=latest_backup,
        next_backup_due_at=next_due,
        retention_days=policy.retention_days,
        issues=unique_issues,
        metadata=metadata_payload,
    )


__all__ = [
    "BackupPolicy",
    "BackupReadinessSnapshot",
    "BackupState",
    "BackupStatus",
    "evaluate_backup_readiness",
    "format_backup_markdown",
]
