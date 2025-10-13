from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pytest

from src.sensory.real_sensory_organ import RealSensoryOrgan
from src.sensory.signals import SensorSignal


class _BareSensor:
    """Minimal sensor stub that omits quality and lineage metadata."""

    def __init__(self, dimension: str, *, strength: float, confidence: float) -> None:
        self._dimension = dimension
        self._strength = strength
        self._confidence = confidence

    def process(self, *_args, **_kwargs) -> list[SensorSignal]:  # pragma: no cover - simple stub
        return [
            SensorSignal(
                signal_type=self._dimension,
                value={"strength": self._strength},
                confidence=self._confidence,
                timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
                metadata=None,
            )
        ]


def test_real_sensory_organ_backfills_quality_and_lineage() -> None:
    sensors = {
        "WHY": _BareSensor("WHY", strength=0.18, confidence=0.51),
        "WHAT": _BareSensor("WHAT", strength=0.27, confidence=0.63),
        "WHEN": _BareSensor("WHEN", strength=-0.12, confidence=0.4),
        "HOW": _BareSensor("HOW", strength=0.33, confidence=0.72),
        "ANOMALY": _BareSensor("ANOMALY", strength=0.44, confidence=0.81),
    }

    organ = RealSensoryOrgan(
        why_sensor=sensors["WHY"],
        what_sensor=sensors["WHAT"],
        when_sensor=sensors["WHEN"],
        how_sensor=sensors["HOW"],
        anomaly_sensor=sensors["ANOMALY"],
    )

    snapshot = organ.observe(pd.DataFrame())
    dimensions = snapshot["dimensions"]

    assert set(dimensions.keys()) == {"WHY", "WHAT", "WHEN", "HOW", "ANOMALY"}

    for dimension, payload in dimensions.items():
        metadata = payload["metadata"]
        quality = payload["quality"]
        lineage = payload["lineage"]

        assert isinstance(metadata, dict)
        assert metadata["quality"] == quality
        assert quality["source"] == f"sensory.{dimension.lower()}"
        assert isinstance(quality["timestamp"], str)
        assert pytest.approx(quality["confidence"]) == pytest.approx(payload["confidence"])
        assert pytest.approx(quality["strength"]) == pytest.approx(payload["signal"])

        assert isinstance(lineage, dict)
        assert lineage["source"] == "sensory.real_organ"
        assert lineage["metadata"]["mode"] == "generated"
        assert lineage["metadata"]["reason"] == "missing_lineage"
        assert pytest.approx(lineage["outputs"]["strength"]) == pytest.approx(payload["signal"])
        assert metadata["lineage"] == lineage
