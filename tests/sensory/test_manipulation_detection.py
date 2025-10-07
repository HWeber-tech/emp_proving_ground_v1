from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd
import pytest

from src.sensory.enhanced.anomaly.manipulation_detection import ManipulationDetectionSystem
from src.sensory.lineage import SensorLineageRecorder


def _build_test_frame() -> pd.DataFrame:
    start = datetime(2024, 1, 1, 12, 0, 0)
    price = 100.0
    rows: list[dict[str, float | datetime]] = []

    for idx in range(60):
        timestamp = start + timedelta(minutes=idx)
        if 10 <= idx < 18:
            price += 2.0  # pump
        elif 18 <= idx < 24:
            price -= 2.5  # dump
        else:
            price += 0.1 if idx % 3 else -0.1

        volume = 1_000.0 + (idx % 5) * 120.0
        if idx in (12, 19):
            volume *= 6.0  # volume spike for spoofing heuristic

        rows.append(
            {
                "timestamp": timestamp,
                "open": price - 0.3,
                "high": price + 0.5,
                "low": price - 0.6,
                "close": price,
                "volume": volume,
            }
        )

    return pd.DataFrame(rows)


@pytest.mark.asyncio
async def test_manipulation_detection_emits_lineage_and_patterns() -> None:
    df = _build_test_frame()
    recorder = SensorLineageRecorder(max_records=4)
    system = ManipulationDetectionSystem(lineage_recorder=recorder)

    result = await system.detect_manipulation({"price_data": df})

    assert result.patterns, "expected manipulation patterns to be detected"
    assert result.overall_risk_score > 0.0
    assert result.confidence >= 0.2
    assert result.status in {"warn", "alert"}

    lineage_payload = result.lineage.as_dict()
    assert lineage_payload["dimension"] == "ANOMALY"
    assert lineage_payload["outputs"]["pattern_count"] == len(result.patterns)
    assert recorder.latest() is not None


@pytest.mark.asyncio
async def test_manipulation_detection_handles_empty_payload() -> None:
    system = ManipulationDetectionSystem()
    result = await system.detect_manipulation({})

    assert result.patterns == tuple()
    assert result.status == "nominal"
    assert result.overall_risk_score == pytest.approx(0.0)
