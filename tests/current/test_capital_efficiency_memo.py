from __future__ import annotations

from datetime import datetime, timezone

from src.config.risk.risk_config import RiskConfig
from src.trading.order_management import PositionTracker
from src.trading.order_management.journal_loader import (
    load_order_journal,
    replay_journal_into_tracker,
)
from scripts.generate_capital_efficiency_memo import compute_capital_efficiency


def _record(
    *,
    order_id: str,
    symbol: str,
    side: str,
    quantity: float,
    price: float,
    timestamp: str,
) -> dict[str, object]:
    return {
        "event_type": "filled",
        "order_id": order_id,
        "symbol": symbol,
        "side": side,
        "filled_quantity": quantity,
        "last_price": price,
        "event_timestamp": timestamp,
    }


def test_replay_journal_tracks_positions(tmp_path) -> None:
    path = tmp_path / "journal.jsonl"
    entries = [
        _record(
            order_id="1",
            symbol="AAPL",
            side="BUY",
            quantity=100.0,
            price=10.0,
            timestamp="2025-01-01T10:00:00+00:00",
        ),
        _record(
            order_id="2",
            symbol="AAPL",
            side="SELL",
            quantity=50.0,
            price=10.5,
            timestamp="2025-01-01T12:00:00+00:00",
        ),
    ]
    with path.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(f"{entry}\n".replace("'", '"'))

    records = load_order_journal(path)
    tracker = PositionTracker()
    fills = replay_journal_into_tracker(records, tracker)

    assert len(fills) == 2
    snapshot = tracker.get_position_snapshot("AAPL")
    assert snapshot.net_quantity == 50.0
    assert snapshot.realized_pnl == 25.0


def test_compute_capital_efficiency_generates_summary() -> None:
    tracker = PositionTracker()
    records = [
        {
            "event_type": "filled",
            "order_id": "1",
            "symbol": "AAPL",
            "side": "BUY",
            "filled_quantity": 100.0,
            "last_price": 10.0,
            "event_timestamp": datetime(2025, 1, 1, 10, tzinfo=timezone.utc).isoformat(),
        },
        {
            "event_type": "filled",
            "order_id": "2",
            "symbol": "AAPL",
            "side": "SELL",
            "filled_quantity": 100.0,
            "last_price": 11.0,
            "event_timestamp": datetime(2025, 1, 2, 11, tzinfo=timezone.utc).isoformat(),
        },
    ]

    fills = replay_journal_into_tracker(records, tracker)
    memo = compute_capital_efficiency(
        fills,
        risk_config=RiskConfig(),
        account_balance=100_000.0,
    )

    assert len(memo.days) == 2
    first_day, second_day = memo.days
    assert first_day.total_exposure == 1_000.0
    assert first_day.trades == 1
    assert second_day.realised_pnl == 100.0
    assert memo.total_notional_traded == 1_000.0 + 1_100.0
    assert memo.max_utilisation > 0.0

