from datetime import datetime, timedelta, timezone

import pytest

from src.core.genome import NoOpGenomeProvider
from src.evolution import (
    RecordedEvaluationResult,
    RecordedSensoryEvaluator,
    RecordedSensorySnapshot,
    RecordedTrade,
    summarise_recorded_replay,
)
from src.sensory.signals import IntegratedSignal


def _snapshot(ts: datetime, price: float, strength: float, confidence: float) -> RecordedSensorySnapshot:
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


def test_recorded_replay_summary_builds_lineage_and_markdown() -> None:
    start = datetime.now(timezone.utc) - timedelta(minutes=40)
    price = 100.0
    snapshots: list[RecordedSensorySnapshot] = []
    for idx in range(30):
        ts = start + timedelta(minutes=idx)
        price += 0.45
        strength = 0.6 if idx > 3 else 0.2
        confidence = 0.78
        snapshots.append(_snapshot(ts, price, strength, confidence))

    genome = NoOpGenomeProvider().new_genome(
        "genome-telemetry",
        {
            "entry_threshold": 0.4,
            "exit_threshold": 0.2,
            "risk_fraction": 0.4,
            "min_confidence": 0.7,
            "cooldown_steps": 1,
        },
    )

    evaluator = RecordedSensoryEvaluator(snapshots)
    result = evaluator.evaluate(genome)

    summary = summarise_recorded_replay(
        result,
        genome_id=getattr(genome, "id", "genome-telemetry"),
        dataset_id="timescale-session-42",
        evaluation_id="replay-20240101",
        parameters=getattr(genome, "parameters", {}),
        metadata={"run": "integration"},
    )

    data = summary.as_dict()
    assert data["status"] in {"normal", "warn", "alert"}
    assert data["lineage"]["metadata"]["genome_id"] == getattr(genome, "id", "genome-telemetry")
    assert data["lineage"]["inputs"]["dataset_id"] == "timescale-session-42"
    assert pytest.approx(result.total_return, rel=1e-6) == data["metrics"]["total_return"]
    assert "Total return" in summary.to_markdown()
    assert data["trade_summary"]["profit_factor"] >= 0
    assert "best_trade" in data["trade_summary"]


def test_recorded_replay_summary_flags_alert_on_large_drawdown() -> None:
    now = datetime.now(timezone.utc)
    trade = RecordedTrade(
        opened_at=now - timedelta(minutes=10),
        closed_at=now,
        direction=-1,
        entry_price=100.0,
        exit_price=80.0,
        return_pct=-0.2,
        confidence_open=0.9,
        confidence_close=0.2,
        strength_open=0.5,
        strength_close=-0.3,
    )
    result = RecordedEvaluationResult(
        equity_curve=(1.0, 0.8),
        total_return=-0.2,
        max_drawdown=0.4,
        sharpe_ratio=-1.1,
        volatility=0.3,
        win_rate=0.0,
        trades=1,
        wins=0,
        losses=1,
        trade_log=(trade,),
        max_consecutive_losses=1,
        average_trade_duration_minutes=10.0,
    )

    summary = summarise_recorded_replay(
        result,
        genome_id="stress-genome",
        warn_drawdown=0.1,
        alert_drawdown=0.3,
    )

    assert summary.status == "alert"
    assert summary.trade_summary["worst_trade"]["return_pct"] == pytest.approx(-0.2)
    assert summary.trade_summary["exposure_minutes"] == pytest.approx(10.0)
