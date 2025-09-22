from __future__ import annotations

from datetime import UTC, datetime

from src.data_foundation.persist.timescale import (
    TimescaleExecutionJournal,
    TimescaleMigrator,
    TimescaleConnectionSettings,
)
from src.operations.execution import (
    ExecutionIssue,
    ExecutionPolicy,
    ExecutionReadinessSnapshot,
    ExecutionState,
    ExecutionStatus,
)


def _build_snapshot(
    *,
    service: str,
    status: ExecutionStatus = ExecutionStatus.passed,
    orders_submitted: int = 4,
    orders_executed: int = 3,
    metadata: dict[str, object] | None = None,
) -> ExecutionReadinessSnapshot:
    policy = ExecutionPolicy()
    state = ExecutionState(
        orders_submitted=orders_submitted,
        orders_executed=orders_executed,
        orders_failed=max(0, orders_submitted - orders_executed),
        pending_orders=max(0, orders_submitted - orders_executed),
        avg_latency_ms=125.0,
        max_latency_ms=480.0,
        drop_copy_lag_seconds=2.5,
        drop_copy_active=True,
        connection_healthy=True,
        sessions_active=("trade",),
        metadata={"window": "smoke"},
    )
    issue = ExecutionIssue(
        code="drop_copy",
        message="Drop-copy lag elevated",
        severity=ExecutionStatus.warn,
        metadata={"lag": 2.5},
    )
    return ExecutionReadinessSnapshot(
        service=service,
        generated_at=datetime.now(tz=UTC),
        status=status,
        policy=policy,
        state=state,
        issues=(issue,) if status is not ExecutionStatus.passed else tuple(),
        metadata=metadata or {"source": "test"},
    )


def _create_sqlite_engine(tmp_path) -> TimescaleExecutionJournal:
    db_path = tmp_path / "execution_journal.db"
    settings = TimescaleConnectionSettings(url=f"sqlite:///{db_path}")
    engine = settings.create_engine()
    TimescaleMigrator(engine).ensure_execution_tables()
    return TimescaleExecutionJournal(engine)


def test_execution_journal_round_trip(tmp_path) -> None:
    journal = _create_sqlite_engine(tmp_path)
    snapshot = _build_snapshot(service="paper-stack", status=ExecutionStatus.warn)

    stored = journal.record_snapshot(snapshot, strategy_id="alpha")

    assert stored["service"] == "paper-stack"
    assert stored["status"] == ExecutionStatus.warn.value
    assert stored["orders_executed"] == 3
    assert stored["strategy_id"] == "alpha"

    recent = journal.fetch_recent(limit=5, strategy_id="alpha")
    assert len(recent) == 1
    assert recent[0].service == "paper-stack"

    latest = journal.fetch_latest(service="paper-stack", strategy_id="alpha")
    assert latest is not None
    assert latest.snapshot_id == recent[0].snapshot_id


def test_execution_journal_service_filter(tmp_path) -> None:
    journal = _create_sqlite_engine(tmp_path)
    primary = _build_snapshot(service="primary")
    secondary = _build_snapshot(service="secondary", orders_submitted=6, orders_executed=6)

    journal.record_snapshot(primary, strategy_id="tier-1")
    journal.record_snapshot(secondary, strategy_id="tier-1")

    latest_primary = journal.fetch_latest(service="primary", strategy_id="tier-1")
    assert latest_primary is not None
    assert latest_primary.service == "primary"

    recent_secondary = journal.fetch_recent(service="secondary", strategy_id="tier-1")
    assert len(recent_secondary) == 1
    assert recent_secondary[0].service == "secondary"
    assert recent_secondary[0].orders_submitted == 6


def test_execution_journal_summarise(tmp_path) -> None:
    journal = _create_sqlite_engine(tmp_path)
    journal.record_snapshot(
        _build_snapshot(service="primary", status=ExecutionStatus.passed),
        strategy_id="alpha",
    )
    journal.record_snapshot(
        _build_snapshot(service="secondary", status=ExecutionStatus.fail),
        strategy_id="beta",
    )

    summary_all = journal.summarise()
    assert summary_all["total_snapshots"] == 2
    assert summary_all["status_counts"][ExecutionStatus.passed.value] == 1
    assert summary_all["status_counts"][ExecutionStatus.fail.value] == 1
    assert summary_all["service_counts"]["primary"] == 1

    summary_primary = journal.summarise(service="primary")
    assert summary_primary["total_snapshots"] == 1
    assert summary_primary["status_counts"][ExecutionStatus.passed.value] == 1
