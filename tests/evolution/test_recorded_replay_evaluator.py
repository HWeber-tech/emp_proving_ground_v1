from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.core.genome import NoOpGenomeProvider
from src.evolution.evaluation import (
    RecordedSensoryEvaluator,
    RecordedSensorySnapshot,
    RecordedTrade,
)
from src.sensory.signals import IntegratedSignal


def _make_snapshot(ts: datetime, price: float, strength: float, confidence: float) -> RecordedSensorySnapshot:
    payload = {
        "generated_at": ts,
        "integrated_signal": IntegratedSignal(
            direction=1.0 if strength >= 0 else -1.0,
            strength=strength,
            confidence=confidence,
            contributing=["WHY", "WHAT", "WHEN", "HOW", "ANOMALY"],
        ),
        "dimensions": {
            "WHAT": {
                "signal": strength,
                "confidence": confidence,
                "value": {"last_close": price},
                "metadata": {"last_close": price},
            }
        },
    }
    return RecordedSensorySnapshot.from_snapshot(payload)


def test_evaluator_generates_positive_return_for_trending_signals() -> None:
    start = datetime.now(timezone.utc) - timedelta(minutes=30)
    price = 100.0
    snapshots: list[RecordedSensorySnapshot] = []
    for idx in range(20):
        ts = start + timedelta(minutes=idx)
        price += 0.6
        strength = 0.65 if idx > 2 else 0.1
        confidence = 0.8
        snapshots.append(_make_snapshot(ts, price, strength, confidence))

    genome = NoOpGenomeProvider().new_genome(
        "genome-1",
        {
            "entry_threshold": 0.4,
            "exit_threshold": 0.15,
            "risk_fraction": 0.5,
            "min_confidence": 0.6,
        },
    )
    evaluator = RecordedSensoryEvaluator(reversed(snapshots))  # ensure sorting works
    result = evaluator.evaluate(genome)

    assert result.total_return > 0
    assert result.trades >= 1
    assert result.win_rate >= 0.5
    assert len(result.equity_curve) == len(snapshots)
    assert all(isinstance(trade, RecordedTrade) for trade in result.trade_log)
    assert len(result.trade_log) == result.trades
    ledger = result.as_dict()["trade_log"]
    assert isinstance(ledger, list)
    assert ledger
    first_trade = ledger[0]
    assert first_trade["opened_at"] < first_trade["closed_at"]
    assert result.max_consecutive_losses >= 0
    assert result.average_trade_duration_minutes > 0


def test_evaluator_handles_low_confidence_with_no_trades() -> None:
    start = datetime.now(timezone.utc)
    price = 50.0
    snapshots: list[RecordedSensorySnapshot] = []
    for idx in range(10):
        ts = start + timedelta(minutes=idx)
        price += 0.1
        snapshots.append(_make_snapshot(ts, price, strength=0.05, confidence=0.2))

    genome = NoOpGenomeProvider().new_genome(
        "genome-flat",
        {
            "entry_threshold": 0.3,
            "exit_threshold": 0.1,
            "risk_fraction": 0.4,
            "min_confidence": 0.7,
        },
    )
    evaluator = RecordedSensoryEvaluator(snapshots)
    result = evaluator.evaluate(genome)

    assert result.trades == 0
    assert result.total_return == pytest.approx(0.0, abs=1e-6)
    assert result.max_drawdown == pytest.approx(0.0, abs=1e-6)
    assert result.trade_log == ()
    assert result.max_consecutive_losses == 0
    assert result.average_trade_duration_minutes == pytest.approx(0.0, abs=1e-6)


def test_snapshot_from_serialised_payload_extracts_fields() -> None:
    ts = datetime.now(timezone.utc)
    payload = {
        "generated_at": ts.isoformat(),
        "integrated_signal": {"strength": 0.42, "confidence": 0.73},
        "dimensions": {
            "WHAT": {
                "value": {"last_close": 123.45},
            }
        },
    }

    snapshot = RecordedSensorySnapshot.from_snapshot(payload)

    assert snapshot.timestamp.tzinfo is not None
    assert snapshot.price == pytest.approx(123.45)
    assert snapshot.strength == pytest.approx(0.42)
    assert snapshot.confidence == pytest.approx(0.73)
