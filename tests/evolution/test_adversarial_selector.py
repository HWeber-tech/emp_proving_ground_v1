from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.core.genome import NoOpGenomeProvider
from src.evolution.evaluation import RecordedSensorySnapshot
from src.evolution.selection.adversarial_selector import AdversarialSelector
from src.sensory.lineage import SensorLineageRecorder
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
                "value": {"last_close": price},
                "metadata": {"last_close": price},
            }
        },
    }
    return RecordedSensorySnapshot.from_snapshot(payload)


def _build_snapshots() -> list[RecordedSensorySnapshot]:
    start = datetime.now(timezone.utc) - timedelta(minutes=40)
    price = 100.0
    snapshots: list[RecordedSensorySnapshot] = []
    for idx in range(30):
        ts = start + timedelta(minutes=idx)
        price += 0.6 if idx >= 3 else 0.1
        strength = 0.65 if idx % 3 else 0.45
        confidence = 0.78 if idx >= 5 else 0.55
        snapshots.append(_snapshot(ts, price, strength, confidence))
    return snapshots


def test_adversarial_selector_prioritises_profitable_genome() -> None:
    snapshots = _build_snapshots()
    recorder = SensorLineageRecorder(max_records=8)
    selector = AdversarialSelector(
        snapshots=snapshots,
        dataset_id="replay-eurusd",
        evaluation_id="eval-001",
        lineage_recorder=recorder,
    )

    provider = NoOpGenomeProvider()
    aggressive = provider.new_genome(
        "aggressive",
        {
            "entry_threshold": 0.35,
            "exit_threshold": 0.15,
            "risk_fraction": 0.45,
            "min_confidence": 0.5,
        },
    )
    conservative = provider.new_genome(
        "conservative",
        {
            "entry_threshold": 0.7,
            "exit_threshold": 0.25,
            "risk_fraction": 0.2,
            "min_confidence": 0.8,
        },
    )

    neutral = provider.new_genome(
        "neutral",
        {
            "entry_threshold": 0.5,
            "exit_threshold": 0.2,
            "risk_fraction": 0.3,
            "min_confidence": 0.6,
        },
    )

    selection = selector.select([conservative, aggressive, neutral], k=2)

    assert selection[0] is aggressive
    assert aggressive in selection
    assert selector.score_for(aggressive) is not None
    assert selector.score_for(aggressive) > selector.score_for(conservative)

    telemetry = selector.telemetry_for(aggressive, serialise=True)
    assert isinstance(telemetry, dict)
    assert telemetry["genome_id"] == "aggressive"
    assert telemetry["dataset_id"] == "replay-eurusd"
    assert recorder.latest() is not None


def test_selector_falls_back_without_evaluator() -> None:
    provider = NoOpGenomeProvider()
    genomes = [provider.new_genome(f"g-{idx}", {"entry_threshold": 0.4}) for idx in range(3)]

    selector = AdversarialSelector()
    selection = selector.select(genomes, k=2)

    assert selection == genomes[:2]
