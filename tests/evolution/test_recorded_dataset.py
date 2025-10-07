from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

import json
import pandas as pd

from src.evolution.evaluation import RecordedSensoryEvaluator
from src.evolution.evaluation.datasets import (
    dump_recorded_snapshots,
    load_recorded_snapshots,
)
from src.sensory.real_sensory_organ import RealSensoryOrgan


def _build_market_frame(base_price: float, *, offset_minutes: int = 0) -> pd.DataFrame:
    start = datetime.now(timezone.utc) - timedelta(minutes=40)
    rows: list[dict[str, object]] = []
    price = base_price
    for idx in range(24):
        ts = start + timedelta(minutes=idx + offset_minutes)
        price += 0.0002 + 0.00005 * (idx % 3)
        rows.append(
            {
                "timestamp": ts,
                "symbol": "EURUSD",
                "open": price - 0.0003,
                "high": price + 0.0004,
                "low": price - 0.00035,
                "close": price,
                "volume": 1200 + idx * 35,
                "volatility": 0.0005 + 0.00002 * idx,
                "spread": 0.00005,
                "depth": 5000 + idx * 80,
                "order_imbalance": 0.1 + 0.01 * idx,
                "data_quality": 0.9,
            }
        )
    return pd.DataFrame(rows)


def _observe_snapshots(organ: RealSensoryOrgan, frames: Iterable[pd.DataFrame]) -> list[dict[str, object]]:
    snapshots: list[dict[str, object]] = []
    for frame in frames:
        snapshots.append(organ.observe(frame))
    return snapshots


def test_dump_and_load_recorded_snapshots_roundtrip(tmp_path: Path) -> None:
    organ = RealSensoryOrgan()
    frames = [
        _build_market_frame(1.10 + idx * 0.001, offset_minutes=idx * 2)
        for idx in range(5)
    ]
    snapshots = _observe_snapshots(organ, frames)

    destination = tmp_path / "snapshots.jsonl"
    written = dump_recorded_snapshots(snapshots, destination)
    assert written == len(snapshots)

    loaded = load_recorded_snapshots(destination)
    assert len(loaded) == len(snapshots)
    assert all(s.timestamp.tzinfo is not None for s in loaded)
    assert loaded[0].timestamp <= loaded[-1].timestamp
    assert loaded[0].price > 0
    assert isinstance(loaded[0].strength, float)

    evaluator = RecordedSensoryEvaluator(loaded)
    result = evaluator.evaluate(
        {
            "entry_threshold": 0.05,
            "exit_threshold": 0.02,
            "risk_fraction": 0.2,
            "min_confidence": 0.0,
        }
    )
    assert result.trades >= 0
    assert result.equity_curve


def test_dump_recorded_snapshots_preserves_lineage(tmp_path: Path) -> None:
    organ = RealSensoryOrgan()
    frame = _build_market_frame(1.11)
    snapshot = organ.observe(frame)

    path = tmp_path / "snapshot.jsonl"
    dump_recorded_snapshots([snapshot], path)

    with path.open("r", encoding="utf-8") as fh:
        payload = json.loads(fh.readline())

    assert payload["lineage"]["dimension"] == "SENSORY_FUSION"
    assert payload["dimensions"]["HOW"]["metadata"]["lineage"]["dimension"] == "HOW"


def test_load_recorded_snapshots_skips_invalid_payload(tmp_path: Path) -> None:
    path = tmp_path / "mixed.jsonl"
    path.write_text("not-json\n", encoding="utf-8")

    organ = RealSensoryOrgan()
    frame = _build_market_frame(1.12)
    snapshot = organ.observe(frame)
    dump_recorded_snapshots([snapshot], path, append=True)

    loaded = load_recorded_snapshots(path)
    assert len(loaded) == 1
