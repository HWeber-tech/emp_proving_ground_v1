from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.order_lifecycle_dry_run import run_dry_run


def _write_events(path: Path, events: list[dict[str, object]]) -> Path:
    path.write_text("\n".join(json.dumps(evt) for evt in events) + "\n", encoding="utf-8")
    return path


def test_run_dry_run_success(tmp_path: Path) -> None:
    log_path = _write_events(
        tmp_path / "events.jsonl",
        [
            {
                "order_id": "ORD-1",
                "exec_type": "0",
                "symbol": "ESM4",
                "side": "BUY",
                "order_qty": 1,
                "timestamp": "2024-01-01T14:30:00Z",
            },
            {
                "order_id": "ORD-1",
                "exec_type": "1",
                "last_qty": 0.4,
                "last_price": 4500.0,
                "cum_qty": 0.4,
                "symbol": "ESM4",
                "side": "BUY",
                "order_qty": 1,
                "timestamp": "2024-01-01T14:30:01Z",
            },
            {
                "order_id": "ORD-1",
                "exec_type": "2",
                "last_qty": 0.6,
                "last_price": 4501.0,
                "cum_qty": 1.0,
                "symbol": "ESM4",
                "side": "BUY",
                "order_qty": 1,
                "timestamp": "2024-01-01T14:30:02Z",
            },
        ],
    )

    result = run_dry_run(log_path, verbose=False)

    assert result.events_processed == 3
    assert result.errors == []
    assert len(result.order_summaries) == 1
    summary = result.order_summaries[0]
    assert summary.order_id == "ORD-1"
    assert summary.status == "FILLED"
    assert summary.filled_quantity == pytest.approx(1.0)
    assert summary.remaining_quantity == pytest.approx(0.0)

    assert len(result.position_snapshots) == 1
    position = result.position_snapshots[0]
    assert position["symbol"] == "ESM4"
    assert position["net_quantity"] == pytest.approx(1.0)
    assert position["long_quantity"] == pytest.approx(1.0)
    assert position["short_quantity"] == pytest.approx(0.0)
    assert position["realized_pnl"] == pytest.approx(0.0)


def test_run_dry_run_records_errors(tmp_path: Path) -> None:
    log_path = _write_events(
        tmp_path / "invalid.jsonl",
        [
            {
                "order_id": "ORD-2",
                "exec_type": "0",
                "symbol": "NQH4",
                "side": "SELL",
                "order_qty": 1,
                "timestamp": "2024-02-01T10:00:00Z",
            },
            {
                "order_id": "ORD-2",
                "exec_type": "2",
                "last_qty": 1.0,
                "cum_qty": 2.0,
                "last_price": 16000.0,
                "symbol": "NQH4",
                "side": "SELL",
                "order_qty": 1,
                "timestamp": "2024-02-01T10:00:01Z",
            },
        ],
    )

    result = run_dry_run(log_path, verbose=False)

    assert result.events_processed == 1  # acknowledgement succeeds
    assert len(result.errors) == 1
    assert "overfilled" in result.errors[0]

    assert len(result.order_summaries) == 1
    summary = result.order_summaries[0]
    assert summary.status == "ACKNOWLEDGED"
    assert summary.filled_quantity == pytest.approx(0.0)
    assert summary.remaining_quantity == pytest.approx(1.0)
