from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from src.compliance.trade_compliance import (
    TradeComplianceMonitor,
    TradeCompliancePolicy,
)
from src.core.event_bus import Event
from src.data_foundation.persist.timescale import (
    TimescaleComplianceJournal,
    TimescaleConnectionSettings,
    TimescaleMigrator,
)


class _StubEventBus:
    def __init__(self) -> None:
        self.subscriptions: dict[str, list] = {}
        self.published: list[Event] = []

    def subscribe(self, event_type: str, handler):  # pragma: no cover - simple stub
        self.subscriptions.setdefault(event_type, []).append(handler)
        return SimpleNamespace(
            id=len(self.subscriptions[event_type]), event_type=event_type, handler=handler
        )

    def unsubscribe(self, handle):  # pragma: no cover - simple stub
        handlers = self.subscriptions.get(handle.event_type, [])
        if handle.handler in handlers:
            handlers.remove(handle.handler)

    def publish_from_sync(self, event: Event):
        self.published.append(event)
        return 1


class _StubAuditLogger:
    def __init__(self) -> None:
        self.records: list[dict[str, object]] = []

    def log_compliance_check(self, **payload):  # pragma: no cover - simple stub
        self.records.append(payload)


@pytest.mark.asyncio()
async def test_trade_compliance_monitor_flags_notional_violation() -> None:
    bus = _StubEventBus()
    audit = _StubAuditLogger()
    policy = TradeCompliancePolicy(
        policy_name="unit",
        max_single_trade_notional=10_000.0,
        max_daily_symbol_notional=50_000.0,
        max_trades_per_symbol_per_day=10,
    )
    monitor = TradeComplianceMonitor(
        event_bus=bus, policy=policy, audit_logger=audit, strategy_id="test"
    )

    report = SimpleNamespace(
        event_id="evt-1",
        symbol="EURUSD",
        side="buy",
        quantity=12_000,
        price=1.25,
        timestamp=datetime(2025, 1, 2, 14, 0, tzinfo=timezone.utc),
        status="FILLED",
    )

    await monitor.on_execution_report(report)

    snapshot = monitor.last_snapshot
    assert snapshot is not None
    assert snapshot.status == "fail"
    assert any(
        not check.passed and check.rule_id == "single_trade_notional" for check in snapshot.checks
    )
    assert bus.published and bus.published[0].type == policy.report_channel
    assert audit.records and audit.records[0]["passed"] is False

    monitor.close()


@pytest.mark.asyncio()
async def test_trade_compliance_monitor_tracks_daily_limits() -> None:
    bus = _StubEventBus()
    audit = _StubAuditLogger()
    policy = TradeCompliancePolicy(
        policy_name="daily",
        max_single_trade_notional=100_000.0,
        max_daily_symbol_notional=150_000.0,
        max_trades_per_symbol_per_day=3,
    )
    monitor = TradeComplianceMonitor(event_bus=bus, policy=policy, audit_logger=audit)

    base_time = datetime(2025, 2, 1, 9, 30, tzinfo=timezone.utc)

    first = SimpleNamespace(
        event_id="evt-1",
        symbol="GBPUSD",
        side="SELL",
        quantity=20_000,
        price=2.0,
        timestamp=base_time,
    )
    second = SimpleNamespace(
        event_id="evt-2",
        symbol="GBPUSD",
        side="SELL",
        quantity=25_000,
        price=2.0,
        timestamp=base_time + timedelta(minutes=5),
    )
    third = SimpleNamespace(
        event_id="evt-3",
        symbol="GBPUSD",
        side="SELL",
        quantity=40_000,
        price=2.0,
        timestamp=base_time + timedelta(minutes=10),
    )

    for report in (first, second, third):
        await monitor.on_execution_report(report)

    snapshot = monitor.last_snapshot
    assert snapshot is not None
    assert snapshot.status == "fail"
    assert any(
        not check.passed and check.rule_id == "daily_symbol_notional" for check in snapshot.checks
    )
    summary = monitor.summary()
    assert summary["daily_totals"]["GBPUSD"]["trades"] == 3
    assert summary["daily_totals"]["GBPUSD"]["notional"] > policy.max_daily_symbol_notional

    monitor.close()


@pytest.mark.asyncio()
async def test_trade_compliance_monitor_persists_journal_entries(tmp_path) -> None:
    bus = _StubEventBus()
    audit = _StubAuditLogger()
    db_path = tmp_path / "compliance_journal.db"
    settings = TimescaleConnectionSettings(url=f"sqlite:///{db_path}")
    engine = settings.create_engine()
    TimescaleMigrator(engine).ensure_compliance_tables()
    journal = TimescaleComplianceJournal(engine)

    policy = TradeCompliancePolicy(policy_name="journal")
    monitor = TradeComplianceMonitor(
        event_bus=bus,
        policy=policy,
        audit_logger=audit,
        strategy_id="strategy-123",
        snapshot_journal=journal,
    )

    report = SimpleNamespace(
        event_id="evt-journal",
        symbol="AAPL",
        side="BUY",
        quantity=1_000,
        price=2.5,
        timestamp=datetime(2025, 6, 1, 10, 30, tzinfo=timezone.utc),
        status="FILLED",
    )

    await monitor.on_execution_report(report)

    entries = journal.fetch_recent()
    assert entries
    entry = entries[0]
    assert entry["trade_id"] == "evt-journal"
    assert entry["strategy_id"] == "strategy-123"
    assert entry["passed"] is True

    summary = monitor.summary()
    journal_block = summary.get("journal")
    assert journal_block is not None
    last_entry = journal_block.get("last_entry")
    assert last_entry is not None
    assert last_entry["trade_id"] == "evt-journal"

    monitor.close()


def test_trade_compliance_policy_from_mapping_parses_thresholds() -> None:
    mapping = {
        "COMPLIANCE_POLICY_NAME": "tier1",
        "COMPLIANCE_MAX_SINGLE_NOTIONAL": "250000",
        "COMPLIANCE_MAX_DAILY_NOTIONAL": "750000",
        "COMPLIANCE_MAX_TRADES_PER_SYMBOL": "5",
        "COMPLIANCE_RESTRICTED_SYMBOLS": "BTCUSD, ETHUSD",
        "COMPLIANCE_ALLOWED_SIDES": "buy,sell",
        "COMPLIANCE_REPORT_CHANNEL": "telemetry.compliance.custom",
    }

    policy = TradeCompliancePolicy.from_mapping(mapping)

    assert policy.policy_name == "tier1"
    assert policy.max_single_trade_notional == 250_000.0
    assert "BTCUSD" in policy.restricted_symbols
    assert policy.allowed_sides == frozenset({"BUY", "SELL"})
    assert policy.report_channel == "telemetry.compliance.custom"
