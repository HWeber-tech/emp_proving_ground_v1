from datetime import UTC, datetime

from src.data_foundation.persist.timescale import (
    TimescaleComplianceJournal,
    TimescaleConnectionSettings,
    TimescaleKycJournal,
    TimescaleMigrator,
)


def test_timescale_compliance_journal_round_trip(tmp_path) -> None:
    db_path = tmp_path / "compliance.db"
    settings = TimescaleConnectionSettings(url=f"sqlite:///{db_path}")
    engine = settings.create_engine()
    TimescaleMigrator(engine).ensure_compliance_tables()
    journal = TimescaleComplianceJournal(engine)

    snapshot = {
        "trade_id": "trade-1",
        "intent_id": None,
        "symbol": "EURUSD",
        "side": "BUY",
        "quantity": 1_000.0,
        "price": 1.25,
        "notional": 1_250.0,
        "timestamp": datetime(2025, 5, 10, 12, 0, tzinfo=UTC).isoformat(),
        "status": "OK",
        "checks": [
            {
                "rule_id": "recorded",
                "name": "Execution recorded",
                "passed": True,
                "severity": "info",
                "message": "Execution processed",
                "metadata": {},
            }
        ],
        "totals": {"daily_notional": 1_250.0},
        "policy_name": "institutional",
    }

    recorded = journal.record_snapshot(snapshot, strategy_id="alpha")
    assert recorded["trade_id"] == "trade-1"
    assert recorded["strategy_id"] == "alpha"
    assert recorded["violations"] == []

    entries = journal.fetch_recent(strategy_id="alpha")
    assert entries
    entry = entries[0]
    assert entry["trade_id"] == "trade-1"
    assert entry["totals"]["daily_notional"] == 1_250.0
    assert entry["passed"] is True

    journal.close()


def test_timescale_compliance_journal_summarise(tmp_path) -> None:
    db_path = tmp_path / "compliance_summary.db"
    settings = TimescaleConnectionSettings(url=f"sqlite:///{db_path}")
    engine = settings.create_engine()
    TimescaleMigrator(engine).ensure_compliance_tables()
    journal = TimescaleComplianceJournal(engine)

    base = {
        "trade_id": "trade-1",
        "symbol": "EURUSD",
        "side": "BUY",
        "quantity": 5_000.0,
        "price": 1.2,
        "notional": 6_000.0,
        "policy_name": "institutional",
        "timestamp": datetime(2025, 6, 1, 9, 0, tzinfo=UTC).isoformat(),
        "checks": [
            {
                "rule_id": "size",
                "name": "Size within bounds",
                "passed": True,
                "severity": "info",
                "message": "Trade within limits",
            }
        ],
        "totals": {"daily_notional": 6_000.0},
    }

    journal.record_snapshot(base, strategy_id="alpha")

    failing = {
        **base,
        "trade_id": "trade-2",
        "status": "REJECTED",
        "checks": [
            {
                "rule_id": "limit",
                "name": "Limit exceeded",
                "passed": False,
                "severity": "error",
                "message": "Exceeded policy limit",
            }
        ],
    }
    journal.record_snapshot(failing, strategy_id="beta")

    summary_all = journal.summarise()
    assert summary_all["total_records"] == 2
    assert summary_all["passed_records"] == 1
    assert summary_all["failed_records"] == 1
    assert summary_all["severity_counts"]["error"] == 1
    assert summary_all["status_counts"]["REJECTED"] == 1
    assert summary_all["recent_records"] == 2
    assert summary_all["recent_window_seconds"] == 24 * 3600

    summary_alpha = journal.summarise(strategy_id="alpha")
    assert summary_alpha["total_records"] == 1
    assert summary_alpha["passed_records"] == 1
    assert summary_alpha["failed_records"] == 0
    assert summary_alpha["recent_records"] == 1

    journal.close()


def test_timescale_kyc_journal_round_trip(tmp_path) -> None:
    db_path = tmp_path / "kyc.db"
    settings = TimescaleConnectionSettings(url=f"sqlite:///{db_path}")
    engine = settings.create_engine()
    TimescaleMigrator(engine).ensure_compliance_tables()
    journal = TimescaleKycJournal(engine)

    snapshot = {
        "case_id": "case-1",
        "entity_id": "client-42",
        "entity_type": "institution",
        "status": "ESCALATED",
        "risk_rating": "CRITICAL",
        "risk_score": 91.2,
        "watchlist_hits": ["OFAC"],
        "outstanding_items": ["Enhanced due diligence"],
        "checklist": [
            {
                "item_id": "edd",
                "name": "Enhanced due diligence",
                "status": "PENDING",
                "severity": "critical",
            }
        ],
        "alerts": ["Watchlist hit"],
        "metadata": {"jurisdiction": "EU"},
        "assigned_to": "analyst-7",
        "last_reviewed_at": datetime(2025, 1, 10, 10, 0, tzinfo=UTC).isoformat(),
        "next_review_due": datetime(2025, 4, 10, 10, 0, tzinfo=UTC).isoformat(),
        "evaluated_at": datetime(2025, 2, 1, 8, 0, tzinfo=UTC).isoformat(),
    }

    recorded = journal.record_case(snapshot, strategy_id="alpha")
    assert recorded["case_id"] == "case-1"
    assert recorded["strategy_id"] == "alpha"
    assert recorded["watchlist_hits"] == ["OFAC"]

    entries = journal.fetch_recent(entity_id="client-42")
    assert entries
    entry = entries[0]
    assert entry["case_id"] == "case-1"
    assert entry["risk_rating"] == "CRITICAL"
    assert entry["alerts"] == ["Watchlist hit"]

    journal.close()


def test_timescale_kyc_journal_summarise(tmp_path) -> None:
    db_path = tmp_path / "kyc_summary.db"
    settings = TimescaleConnectionSettings(url=f"sqlite:///{db_path}")
    engine = settings.create_engine()
    TimescaleMigrator(engine).ensure_compliance_tables()
    journal = TimescaleKycJournal(engine)

    base = {
        "case_id": "case-1",
        "entity_id": "client-42",
        "entity_type": "institution",
        "status": "APPROVED",
        "risk_rating": "LOW",
        "risk_score": 10.0,
        "checklist": [],
        "alerts": [],
        "metadata": {},
    }

    journal.record_case(base, strategy_id="alpha")
    journal.record_case(
        {
            **base,
            "case_id": "case-2",
            "status": "ESCALATED",
            "risk_rating": "CRITICAL",
            "risk_score": 95.0,
        },
        strategy_id="beta",
    )

    summary_all = journal.summarise()
    assert summary_all["total_cases"] == 2
    assert summary_all["status_counts"]["APPROVED"] == 1
    assert summary_all["status_counts"]["ESCALATED"] == 1
    assert summary_all["risk_rating_counts"]["CRITICAL"] == 1
    assert summary_all["recent_cases"] == 2
    assert summary_all["recent_window_seconds"] == 24 * 3600

    summary_alpha = journal.summarise(strategy_id="alpha")
    assert summary_alpha["total_cases"] == 1
    assert summary_alpha["status_counts"]["APPROVED"] == 1
    assert summary_alpha["recent_cases"] == 1

    journal.close()
