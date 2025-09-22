from datetime import UTC, datetime

from src.compliance.kyc import KycAmlMonitor


class _RecordingBus:
    def __init__(self) -> None:
        self.events: list[object] = []

    def publish_from_sync(self, event) -> int:
        self.events.append(event)
        return 1


class _RecordingJournal:
    def __init__(self) -> None:
        self.entries: list[dict[str, object]] = []
        self.closed = False

    def record_case(self, snapshot, *, strategy_id: str):
        entry = dict(snapshot)
        entry["strategy_id"] = strategy_id
        self.entries.append(entry)
        return entry

    def fetch_recent(
        self, *, limit: int = 5, strategy_id: str | None = None, entity_id: str | None = None
    ):
        results = list(self.entries)
        if strategy_id is not None:
            results = [entry for entry in results if entry.get("strategy_id") == strategy_id]
        if entity_id is not None:
            results = [entry for entry in results if entry.get("entity_id") == entity_id]
        return results[-limit:]

    def close(self) -> None:
        self.closed = True


def test_kyc_monitor_emits_events_and_journals() -> None:
    bus = _RecordingBus()
    journal = _RecordingJournal()
    monitor = KycAmlMonitor(
        event_bus=bus,
        strategy_id="alpha",
        snapshot_journal=journal,
        history_limit=3,
    )

    snapshot = monitor.evaluate_case(
        {
            "case_id": "case-123",
            "entity_id": "client-99",
            "entity_type": "institution",
            "risk_score": 85.0,
            "checklist": [
                {"item_id": "passport", "name": "Passport", "status": "COMPLETE"},
                {
                    "item_id": "edd",
                    "name": "Enhanced due diligence",
                    "status": "PENDING",
                    "severity": "critical",
                },
            ],
            "watchlist_hits": ["OFAC"],
            "alerts": ["EDD missing"],
            "metadata": {"jurisdiction": "EU"},
            "assigned_to": "analyst-1",
            "last_reviewed_at": datetime(2025, 3, 1, 9, 0, tzinfo=UTC),
            "review_frequency_days": 30,
        }
    )

    assert snapshot.status == "ESCALATED"
    assert snapshot.risk_rating == "CRITICAL"

    assert bus.events, "expected telemetry event to be published"
    event = bus.events[-1]
    assert event.type == "telemetry.compliance.kyc"
    assert event.payload["case_id"] == "case-123"
    assert "markdown" in event.payload

    summary = monitor.summary()
    assert summary["last_snapshot"]["case_id"] == "case-123"
    assert summary["escalations"] == 1
    assert summary["open_cases"] == 1
    assert summary["journal"]["last_entry"]["case_id"] == "case-123"

    monitor.close()
    assert journal.closed is True
