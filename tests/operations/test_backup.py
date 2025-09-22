from datetime import UTC, datetime, timedelta

import pytest

from src.operations.backup import (
    BackupPolicy,
    BackupReadinessSnapshot,
    BackupState,
    BackupStatus,
    evaluate_backup_readiness,
    format_backup_markdown,
)


def test_backup_readiness_ok() -> None:
    now = datetime(2024, 1, 3, 12, 0, tzinfo=UTC)
    policy = BackupPolicy(expected_frequency_seconds=86_400.0)
    state = BackupState(
        last_backup_at=now - timedelta(hours=2),
        last_restore_test_at=now - timedelta(hours=1),
    )

    snapshot = evaluate_backup_readiness(policy, state, now=now, service="timescale")

    assert snapshot.status is BackupStatus.ok
    assert snapshot.latest_backup_at == state.last_backup_at
    assert snapshot.next_backup_due_at is not None
    markdown = format_backup_markdown(snapshot)
    assert "Backup readiness" in markdown


@pytest.mark.parametrize(
    "age_hours,expected_status",
    [
        (2, BackupStatus.warn),
        (10, BackupStatus.fail),
    ],
)
def test_backup_readiness_escalates_with_age(age_hours: int, expected_status: BackupStatus) -> None:
    now = datetime(2024, 1, 4, 0, 0, tzinfo=UTC)
    policy = BackupPolicy(expected_frequency_seconds=3600.0)
    state = BackupState(
        last_backup_at=now - timedelta(hours=age_hours),
        last_restore_test_at=now,
    )

    snapshot = evaluate_backup_readiness(policy, state, now=now)

    assert snapshot.status is expected_status
    assert snapshot.issues


def test_backup_readiness_handles_restore_tests() -> None:
    now = datetime(2024, 2, 1, tzinfo=UTC)
    policy = BackupPolicy(expected_frequency_seconds=7200.0, restore_test_interval_days=7)
    state = BackupState(
        last_backup_at=now - timedelta(hours=1),
        last_restore_test_at=now - timedelta(days=10),
        last_restore_status="warn",
    )

    snapshot = evaluate_backup_readiness(policy, state, now=now)

    assert snapshot.status is BackupStatus.warn
    assert any("restore" in issue for issue in snapshot.issues)


def test_backup_readiness_disabled_policy_fails() -> None:
    now = datetime(2024, 1, 5, tzinfo=UTC)
    policy = BackupPolicy(enabled=False)
    state = BackupState()

    snapshot = evaluate_backup_readiness(policy, state, now=now)

    assert snapshot.status is BackupStatus.fail
    assert "disabled" in " ".join(snapshot.issues)


def test_backup_snapshot_serialisation() -> None:
    now = datetime.now(tz=UTC)
    snapshot = BackupReadinessSnapshot(
        service="timescale",
        generated_at=now,
        status=BackupStatus.ok,
        latest_backup_at=now,
        next_backup_due_at=now + timedelta(hours=12),
        retention_days=14,
        issues=("test",),
        metadata={"policy": {"providers": ["s3"]}},
    )

    payload = snapshot.as_dict()

    assert payload["status"] == "ok"
    assert payload["issues"] == ["test"]
    assert payload["metadata"]["policy"]["providers"] == ["s3"]
