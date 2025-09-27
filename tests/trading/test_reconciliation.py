from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.trading.order_management import PositionTracker
from src.trading.order_management.reconciliation import (
    load_broker_positions,
    load_order_journal_records,
    replay_order_events,
    report_to_dict,
)

def test_replay_order_events_updates_tracker() -> None:
    tracker = PositionTracker()
    records = [
        {
            "order_id": "ORD-1",
            "symbol": "EURUSD",
            "side": "BUY",
            "event_type": "acknowledged",
            "event_timestamp": "2025-01-01T00:00:00+00:00",
        },
        {
            "order_id": "ORD-1",
            "symbol": "EURUSD",
            "side": "BUY",
            "event_type": "partial_fill",
            "last_quantity": 50000,
            "last_price": 1.1,
            "event_timestamp": "2025-01-01T00:00:01+00:00",
        },
        {
            "order_id": "ORD-1",
            "symbol": "EURUSD",
            "side": "BUY",
            "event_type": "filled",
            "last_quantity": 50000,
            "last_price": 1.2,
            "event_timestamp": "2025-01-01T00:00:02+00:00",
        },
    ]

    replay_order_events(records, tracker)

    snapshot = tracker.get_position_snapshot("EURUSD")
    assert snapshot.net_quantity == pytest.approx(100000)
    assert snapshot.average_long_price == pytest.approx(1.15, abs=1e-9)


def test_report_to_dict_round_trips(tmp_path: Path) -> None:
    tracker = PositionTracker()
    tracker.record_fill("AAPL", 10, 100.0)
    broker = {"AAPL": 12}
    report = tracker.generate_reconciliation_report(broker)
    payload = report_to_dict(report)
    assert payload["account"] == "PRIMARY"
    assert payload["differences"][0]["symbol"] == "AAPL"
    path = tmp_path / "report.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["differences"][0]["symbol"] == "AAPL"


def test_load_broker_positions_json(tmp_path: Path) -> None:
    path = tmp_path / "broker.json"
    path.write_text(json.dumps({"AAPL": 5, "TSLA": "-3"}), encoding="utf-8")
    positions = load_broker_positions(path)
    assert positions == {"AAPL": 5.0, "TSLA": -3.0}


def test_load_broker_positions_csv(tmp_path: Path) -> None:
    path = tmp_path / "broker.csv"
    path.write_text("symbol,quantity\nAAPL,10\nTSLA,-2\n", encoding="utf-8")
    positions = load_broker_positions(path)
    assert positions == {"AAPL": 10.0, "TSLA": -2.0}


def test_load_order_journal_records_json(tmp_path: Path) -> None:
    path = tmp_path / "order_events.parquet.jsonl"
    records = [
        {
            "order_id": "1",
            "symbol": "EURUSD",
            "side": "BUY",
            "event_type": "filled",
            "last_quantity": 1000,
            "last_price": 1.1,
            "event_timestamp": "2025-01-01T00:00:00+00:00",
        },
        {
            "order_id": "1",
            "symbol": "EURUSD",
            "side": "BUY",
            "event_type": "acknowledged",
            "event_timestamp": "2024-12-31T23:59:59+00:00",
        },
    ]
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record) + "\n")

    loaded = load_order_journal_records(path)
    assert loaded[0]["event_type"] == "acknowledged"
    assert loaded[1]["event_type"] == "filled"
