from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from src.sensory.monitoring.live_diagnostics import (
    build_live_sensory_diagnostics,
    build_live_sensory_diagnostics_from_manager,
)
from src.sensory.real_sensory_organ import RealSensoryOrgan, SensoryDriftConfig


def _build_market_frame(total_points: int = 45, spike_start: int = 35) -> pd.DataFrame:
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    price = 1.10
    rows: list[dict[str, object]] = []

    for idx in range(total_points):
        timestamp = base + timedelta(minutes=idx)
        if idx >= spike_start:
            price += 0.02
            volatility = 0.035 + 0.002 * (idx - spike_start)
            spread = 0.0015 + 0.0001 * (idx - spike_start)
            volume = 4200 + 175 * idx
            macro_bias = -0.35
            data_quality = 0.90
            order_imbalance = -1.2
            depth = 2800.0
            yield_two_year = 0.028 + 0.0003 * (idx - spike_start)
            yield_ten_year = 0.024 + 0.0002 * (idx - spike_start)
        else:
            price += 0.0003
            volatility = 0.0006
            spread = 0.00002
            volume = 1200 + 25 * idx
            macro_bias = 0.12
            data_quality = 0.97
            order_imbalance = 0.1
            depth = 6200.0
            yield_two_year = 0.021 + 0.00002 * idx
            yield_ten_year = 0.027 + 0.000015 * idx

        rows.append(
            {
                "timestamp": timestamp,
                "symbol": "EURUSD",
                "open": price - 0.0005,
                "high": price + (0.0005 if idx < spike_start else 0.015),
                "low": price - (0.0005 if idx < spike_start else 0.015),
                "close": price,
                "volume": volume,
                "volatility": volatility,
                "spread": spread,
                "depth": depth,
                "order_imbalance": order_imbalance,
                "macro_bias": macro_bias,
                "yield_curve": {"2Y": yield_two_year, "10Y": yield_ten_year},
                "yield_2y": yield_two_year,
                "yield_10y": yield_ten_year,
                "data_quality": data_quality,
            }
        )

    return pd.DataFrame(rows)


def test_build_live_sensory_diagnostics_produces_summary() -> None:
    frame = _build_market_frame()
    drift_config = SensoryDriftConfig(
        baseline_window=20,
        evaluation_window=10,
        min_observations=5,
        z_threshold=0.9,
        sensors=("ANOMALY", "WHY"),
    )
    organ = RealSensoryOrgan(drift_config=drift_config)

    diagnostics = build_live_sensory_diagnostics(frame, organ=organ)

    assert diagnostics.symbol == "EURUSD"
    assert diagnostics.samples >= frame.shape[0]
    assert diagnostics.integrated_signal["contributing"]

    anomaly_metadata = diagnostics.anomaly.get("metadata", {})
    assert isinstance(anomaly_metadata, dict)
    assert anomaly_metadata.get("state") in {"nominal", "warn", "warning", "alert", "critical"}

    assert diagnostics.drift_summary is not None
    drift_results = diagnostics.drift_summary.get("results", [])
    assert isinstance(drift_results, list) and drift_results

    why_quality = diagnostics.why_quality
    assert why_quality.get("source") == "sensory.why"
    assert "WHY strength" in diagnostics.why_explanation


class _StubManager:
    def __init__(self, frame: pd.DataFrame) -> None:
        self._frame = frame

    def fetch_data(
        self,
        symbol: str,
        period: str | None = None,
        interval: str | None = None,
        start: str | datetime | None = None,
        end: str | datetime | None = None,
    ) -> pd.DataFrame:
        frame = self._frame.copy()
        frame["symbol"] = symbol
        return frame


def test_build_live_sensory_diagnostics_from_manager_overrides_symbol() -> None:
    frame = _build_market_frame()
    manager = _StubManager(frame)

    organ = RealSensoryOrgan(
        drift_config=SensoryDriftConfig(
            baseline_window=18,
            evaluation_window=8,
            min_observations=5,
            z_threshold=0.8,
            sensors=("ANOMALY",),
        )
    )

    diagnostics = build_live_sensory_diagnostics_from_manager(
        manager,
        symbol="USDJPY",
        organ=organ,
    )

    assert diagnostics.symbol == "USDJPY"
    assert diagnostics.samples >= frame.shape[0]
    assert diagnostics.why_quality


def test_build_live_sensory_diagnostics_rejects_empty_frame() -> None:
    with pytest.raises(ValueError):
        build_live_sensory_diagnostics(pd.DataFrame())
