from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from src.core.event_bus import Event
from src.understanding.belief_regime_calibrator import (
    BeliefRegimeCalibration,
    build_calibrated_belief_components,
    calibrate_belief_and_regime,
)


@dataclass
class _RecordingBus:
    events: list[Event]

    def __init__(self) -> None:
        self.events = []

    def is_running(self) -> bool:  # pragma: no cover - simple test bus
        return True

    def publish_from_sync(self, event: Event) -> int:
        self.events.append(event)
        return 1


def _load_prices(limit: int) -> Sequence[float]:
    csv_path = Path(__file__).resolve().parents[1] / "data" / "eurusd_daily_slice.csv"
    frame = pd.read_csv(csv_path)
    series = frame["close"].astype(float)
    if limit:
        series = series.iloc[:limit]
    return series.to_list()


def _build_snapshots(
    prices: Iterable[float],
    *,
    start: datetime,
    volatility_boost: float,
) -> list[dict[str, object]]:
    closes = pd.Series(list(prices), dtype=float)
    returns = closes.pct_change().fillna(0.0)
    strength = returns.rolling(window=3, min_periods=1).mean().fillna(0.0)
    volatility = returns.abs().rolling(window=5, min_periods=1).std(ddof=0).fillna(0.0)
    scaled_volatility = (volatility * volatility_boost).clip(lower=0.0)
    timestamps = [start + timedelta(days=idx) for idx in range(len(closes))]

    snapshots: list[dict[str, object]] = []
    for idx in range(len(closes)):
        confidence = float(np.clip(0.9 - scaled_volatility.iloc[idx] * 0.6, 0.35, 0.98))
        anomaly_level = float(scaled_volatility.iloc[idx] * 0.75)
        is_anomaly = anomaly_level > 0.35
        snapshots.append(
            {
                "symbol": "EURUSD",
                "generated_at": timestamps[idx].replace(tzinfo=UTC),
                "integrated_signal": {
                    "strength": float(strength.iloc[idx]),
                    "confidence": confidence,
                },
                "dimensions": {
                    "WHAT": {"signal": float(strength.iloc[idx]), "confidence": confidence},
                    "WHEN": {"signal": float(returns.iloc[idx]), "confidence": confidence * 0.9},
                    "HOW": {
                        "signal": float(scaled_volatility.iloc[idx]),
                        "confidence": confidence * 0.85,
                        "value": {"volatility": float(scaled_volatility.iloc[idx])},
                    },
                    "ANOMALY": {
                        "signal": anomaly_level,
                        "confidence": confidence * 0.8,
                        "value": {"is_anomaly": is_anomaly, "z_score": anomaly_level * 4.0},
                        "metadata": {"audit": {"z_score": anomaly_level * 4.0}},
                    },
                },
                "lineage": {"id": f"snapshot-{idx}", "timestamp": timestamps[idx].isoformat()},
            }
        )
    return snapshots


def test_calibration_computes_reasonable_parameters() -> None:
    prices = _load_prices(120)
    calibration = calibrate_belief_and_regime(prices)

    assert isinstance(calibration, BeliefRegimeCalibration)
    assert 0.0 < calibration.learning_rate <= 0.35
    assert 0.0 < calibration.decay <= 0.25
    assert calibration.min_variance > 0.0
    assert calibration.max_variance >= calibration.min_variance
    assert 0.0 < calibration.calm_threshold < calibration.storm_threshold
    assert 8 <= calibration.volatility_window <= 96
    assert calibration.volatility_feature in calibration.volatility_features
    assert calibration.diagnostics["returns_std"] >= 0.0


def test_calibration_drives_calm_and_storm_transitions() -> None:
    prices = _load_prices(200)
    calibration = calibrate_belief_and_regime(prices)
    bus = _RecordingBus()
    buffer, emitter, fsm = build_calibrated_belief_components(
        calibration,
        belief_id="calibrated-belief",
        regime_signal_id="calibrated-regime",
        event_bus=bus,
    )

    calm_snapshots = _build_snapshots(
        [1.0] * 24,
        start=datetime(2025, 3, 1, tzinfo=UTC),
        volatility_boost=0.2,
    )
    for snapshot in calm_snapshots:
        state = emitter.emit(snapshot)
        eigenvalues = np.linalg.eigvalsh(np.array(state.posterior.covariance))
        assert np.all(eigenvalues >= -1e-9)
        assert np.all(eigenvalues <= calibration.max_variance + 1e-6)
        regime_signal = fsm.publish(state)
        assert regime_signal.regime_state.volatility_state in {"calm", "normal"}

    erratic_snapshots = _build_snapshots(
        [1.0 + ((-1) ** idx) * 0.01 * (idx % 5 + 1) for idx in range(24)],
        start=datetime(2025, 4, 1, tzinfo=UTC),
        volatility_boost=600.0,
    )
    signals = [fsm.publish(emitter.emit(snapshot)) for snapshot in erratic_snapshots]
    assert any(signal.regime_state.volatility_state == "storm" for signal in signals)
    assert bus.events, "calibrated components should emit telemetry"

