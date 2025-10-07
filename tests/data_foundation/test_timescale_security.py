from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import create_engine

from src.data_foundation.persist.timescale import (
    TimescaleComplianceJournal,
    TimescaleKycJournal,
    TimescaleMigrator,
)


def _build_engine():
    engine = create_engine("sqlite:///:memory:")
    TimescaleMigrator(engine).ensure_compliance_tables()
    return engine


def test_compliance_journal_handles_untrusted_strategy_ids() -> None:
    engine = _build_engine()
    journal = TimescaleComplianceJournal(engine)

    snapshot = {
        "trade_id": "trade-1",
        "intent_id": "intent-1",
        "symbol": "AAPL",
        "side": "buy",
        "status": "approved",
        "policy_name": "risk_policy",
        "quantity": 10,
        "price": 101.25,
        "notional": 1012.5,
        "timestamp": datetime.now(tz=UTC).isoformat(),
        "checks": [
            {"name": "risk_limit", "passed": True, "severity": "info", "message": "ok"}
        ],
        "totals": {"checks": 1},
    }
    malicious_id = "strategy'); DROP TABLE telemetry.compliance_audit; --"

    journal.record_snapshot(snapshot, strategy_id=malicious_id)

    fetched = journal.fetch_recent(strategy_id=malicious_id)
    assert len(fetched) == 1
    assert fetched[0]["strategy_id"] == malicious_id

    summary = journal.summarise(strategy_id=malicious_id)
    assert summary["total_records"] == 1
    assert summary["passed_records"] == 1


def test_kyc_journal_handles_untrusted_filters() -> None:
    engine = _build_engine()
    journal = TimescaleKycJournal(engine)

    snapshot = {
        "case_id": "case-1",
        "entity_id": "entity'); DROP TABLE telemetry.compliance_kyc; --",
        "entity_type": "individual",
        "status": "open",
        "risk_rating": "medium",
        "risk_score": 42,
        "checklist": [{"name": "sanction_screen", "value": "clear"}],
        "alerts": ["address_mismatch"],
        "metadata": {"source": "kyc"},
        "evaluated_at": datetime.now(tz=UTC).isoformat(),
    }
    malicious_strategy = "kyc'); DROP TABLE telemetry.compliance_kyc; --"

    journal.record_case(snapshot, strategy_id=malicious_strategy)

    fetched = journal.fetch_recent(
        strategy_id=malicious_strategy,
        entity_id=snapshot["entity_id"],
    )
    assert len(fetched) == 1
    assert fetched[0]["strategy_id"] == malicious_strategy
    assert fetched[0]["entity_id"] == snapshot["entity_id"]

    summary = journal.summarise(
        strategy_id=malicious_strategy,
        entity_id=snapshot["entity_id"],
    )
    assert summary["total_cases"] == 1
