from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import pytest

from src.core.event_bus import Event
from src.understanding.belief_regime_calibrator import (
    build_calibrated_belief_components,
    calibrate_belief_and_regime,
)
from src.understanding.live_belief_manager import LiveBeliefManager


@dataclass
class _RecordingBus:
    events: list[Event]

    def __init__(self) -> None:
        self.events = []

    def is_running(self) -> bool:  # pragma: no cover - simple stub
        return True

    def publish_from_sync(self, event: Event) -> int:
        self.events.append(event)
        return 1


class _StubOrgan:
    def observe(self, *_args, **_kwargs):  # pragma: no cover - defensive stub
        raise RuntimeError("observe should not be called in this test")


def _frame_from_prices(prices: Sequence[float]) -> pd.DataFrame:
    base = datetime(2025, 4, 1, tzinfo=UTC)
    rows = []
    for offset, price in enumerate(prices):
        timestamp = base + timedelta(minutes=offset)
        rows.append(
            {
                "timestamp": timestamp,
                "symbol": "EURUSD",
                "open": float(price),
                "high": float(price) + 0.0005,
                "low": float(price) - 0.0005,
                "close": float(price),
                "adj_close": float(price),
                "volume": 1_500 + offset * 10,
            }
        )
    return pd.DataFrame(rows)


def _load_market_series(limit: int) -> pd.Series:
    csv_path = Path(__file__).resolve().parent.parent / "data" / "eurusd_daily_slice.csv"
    frame = pd.read_csv(csv_path)
    series = frame["close"].astype(float)
    if limit:
        series = series.iloc[:limit]
    return series.reset_index(drop=True)


def _build_snapshots(
    prices: Iterable[float],
    *,
    start: datetime,
    volatility_boost: float,
) -> list[Mapping[str, object]]:
    closes = pd.Series(list(prices), dtype=float)
    returns = closes.pct_change().fillna(0.0)
    strength = returns.rolling(window=3, min_periods=1).mean().fillna(0.0)
    volatility = returns.abs().rolling(window=5, min_periods=1).std(ddof=0).fillna(0.0)
    volatility_signal = (volatility * volatility_boost).clip(lower=0.0)
    timestamps = [start + timedelta(days=idx) for idx in range(len(closes))]

    snapshots: list[Mapping[str, object]] = []
    for idx in range(len(closes)):
        symbol = "EURUSD"
        signal_strength = float(strength.iloc[idx])
        confidence = float(np.clip(0.85 - volatility_signal.iloc[idx] * 0.6, 0.35, 0.95))
        when_signal = float(np.clip(returns.iloc[idx] * 0.5, -1.0, 1.0))
        how_signal = float(volatility_signal.iloc[idx])
        anomaly_magnitude = float(volatility_signal.iloc[idx] * 0.8)
        is_anomaly = anomaly_magnitude > 0.3

        dimensions = {
            "WHY": {"signal": signal_strength * 0.5, "confidence": confidence * 0.9},
            "WHAT": {"signal": signal_strength, "confidence": confidence * 0.95},
            "WHEN": {"signal": when_signal, "confidence": confidence * 0.85},
            "HOW": {
                "signal": how_signal,
                "confidence": confidence * 0.8,
                "value": {"volatility": how_signal},
            },
            "ANOMALY": {
                "signal": anomaly_magnitude,
                "confidence": confidence * 0.75,
                "value": {
                    "is_anomaly": is_anomaly,
                    "z_score": anomaly_magnitude * 4.0,
                },
                "metadata": {
                    "audit": {"z_score": anomaly_magnitude * 4.0},
                    "is_anomaly": is_anomaly,
                },
            },
        }

        snapshots.append(
            {
                "symbol": symbol,
                "generated_at": timestamps[idx].replace(tzinfo=UTC),
                "integrated_signal": {"strength": signal_strength, "confidence": confidence},
                "dimensions": dimensions,
                "lineage": {
                    "source": "tests.synthetic",
                    "sequence": idx,
                },
            }
        )
    return snapshots


@pytest.mark.guardrail
def test_live_belief_manager_bootstrap_from_market_data() -> None:
    calm_prices = [1.10 for _ in range(32)]
    frame = _frame_from_prices(calm_prices)
    bus = _RecordingBus()

    manager, snapshot, belief_state, regime_signal = LiveBeliefManager.from_market_data(
        market_data=frame,
        symbol="EURUSD",
        belief_id="calm-belief",
        event_bus=bus,
    )

    assert snapshot["symbol"] == "EURUSD"
    assert manager.calibration is not None

    covariance = np.array(belief_state.posterior.covariance)
    eigenvalues = np.linalg.eigvalsh(covariance)
    assert np.all(eigenvalues >= -1e-9)

    assert any(event.type == "telemetry.understanding.belief" for event in bus.events)
    assert any(event.type == "telemetry.understanding.regime" for event in bus.events)
    assert regime_signal.signal_id.endswith("-regime")


@pytest.mark.guardrail
def test_live_belief_manager_regime_transitions_with_snapshots() -> None:
    series = _load_market_series(40)
    calm_snapshots = _build_snapshots(
        series.iloc[:20],
        start=datetime(2025, 5, 1, tzinfo=UTC),
        volatility_boost=1.0,
    )
    storm_snapshots = _build_snapshots(
        series.iloc[20:40],
        start=datetime(2025, 5, 21, tzinfo=UTC),
        volatility_boost=400.0,
    )

    calibration = calibrate_belief_and_regime(series.to_list())
    bus = _RecordingBus()
    buffer, emitter, regime_fsm = build_calibrated_belief_components(
        calibration,
        belief_id="live-belief",
        regime_signal_id="live-belief-regime",
        event_bus=bus,
    )

    manager = LiveBeliefManager(
        belief_id="live-belief",
        symbol="EURUSD",
        organ=_StubOrgan(),
        emitter=emitter,
        regime_fsm=regime_fsm,
        calibration=calibration,
    )

    last_state = None
    for snapshot in calm_snapshots:
        _, belief_state, regime_signal = manager.process_snapshot(snapshot, apply_threshold_scaling=False)
        last_state = regime_signal.regime_state

    assert last_state is not None
    assert last_state.volatility_state in {"calm", "normal"}

    for snapshot in storm_snapshots:
        _, belief_state, regime_signal = manager.process_snapshot(snapshot, apply_threshold_scaling=False)
        last_state = regime_signal.regime_state

    assert last_state.volatility_state == "storm"

    covariance = np.array(belief_state.posterior.covariance)
    eigenvalues = np.linalg.eigvalsh(covariance)
    assert np.all(eigenvalues >= -1e-9)

    regime_events = [event for event in bus.events if event.type == "telemetry.understanding.regime"]
    assert regime_events, "expected regime telemetry to be published"
