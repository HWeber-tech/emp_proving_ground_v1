from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd
import pytest

from src.core.event_bus import Event
from src.sensory.lineage import build_lineage_record
from src.understanding.belief import BeliefBuffer, BeliefEmitter, RegimeFSM


@dataclass
class _RecordingBus:
    events: list[Event]

    def __init__(self) -> None:
        self.events = []

    def is_running(self) -> bool:  # pragma: no cover - simple container
        return True

    def publish_from_sync(self, event: Event) -> int:
        self.events.append(event)
        return 1


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

        lineage = build_lineage_record(
            "SENSORY_FUSION",
            "tests.real_data.synthetic",
            inputs={"sequence": idx, "symbol": symbol},
            outputs={
                "strength": signal_strength,
                "confidence": confidence,
                "volatility": how_signal,
            },
        )

        snapshots.append(
            {
                "symbol": symbol,
                "generated_at": timestamps[idx].replace(tzinfo=UTC),
                "integrated_signal": {"strength": signal_strength, "confidence": confidence},
                "dimensions": dimensions,
                "lineage": lineage,
            }
        )
    return snapshots


@pytest.mark.guardrail
def test_belief_buffer_covenant_with_real_market_returns() -> None:
    prices = _load_market_series(60)
    snapshots = _build_snapshots(prices, start=datetime(2025, 1, 1, tzinfo=UTC), volatility_boost=1.0)

    buffer = BeliefBuffer(
        belief_id="real-belief",
        max_variance=0.25,
        min_variance=1e-6,
        volatility_window=32,
        volatility_features=("HOW_signal", "ANOMALY_signal"),
    )
    bus = _RecordingBus()
    emitter = BeliefEmitter(buffer=buffer, event_bus=bus)

    for snapshot in snapshots:
        state = emitter.emit(snapshot)
        eigenvalues = np.linalg.eigvalsh(np.array(state.posterior.covariance))
        assert np.all(eigenvalues >= 0.0)
        assert np.all(eigenvalues <= 0.25 + 1e-9)
        assert state.metadata["covariance_condition"] >= 1.0
        assert state.metadata["covariance_max_eigenvalue"] <= 0.25 + 1e-9
        assert state.metadata["covariance_min_eigenvalue"] >= 0.0
        assert state.metadata["covariance_trace"] >= 0.0

    assert buffer.latest() is not None
    assert bus.events, "real-data emission should publish telemetry"


@pytest.mark.guardrail
def test_regime_fsm_detects_calm_and_storm_volatility() -> None:
    series = _load_market_series(40)
    calm_snapshots = _build_snapshots(series.iloc[:20], start=datetime(2025, 2, 1, tzinfo=UTC), volatility_boost=1.0)
    storm_snapshots = _build_snapshots(series.iloc[20:40], start=datetime(2025, 2, 21, tzinfo=UTC), volatility_boost=400.0)

    buffer = BeliefBuffer(
        belief_id="regime-real",
        max_variance=0.5,
        min_variance=1e-6,
        volatility_window=24,
        volatility_features=("HOW_signal",),
    )
    bus = _RecordingBus()
    emitter = BeliefEmitter(buffer=buffer, event_bus=bus)
    fsm = RegimeFSM(
        event_bus=bus,
        signal_id="regime",
        calm_threshold=0.04,
        storm_threshold=0.2,
        volatility_window=12,
    )

    calm_signals = [fsm.publish(emitter.emit(snapshot)) for snapshot in calm_snapshots]
    calm_states = [signal.regime_state for signal in calm_signals]
    assert calm_states, "should produce calm regime states"
    assert {state.volatility_state for state in calm_states} <= {"calm", "normal"}
    assert calm_states[-1].volatility_state == "calm"

    storm_signals = [fsm.publish(emitter.emit(snapshot)) for snapshot in storm_snapshots]
    storm_states = [signal.regime_state for signal in storm_signals]
    assert storm_states[-1].volatility_state == "storm"
    assert storm_states[-1].volatility >= 0.2
    assert any(state.volatility_state == "storm" for state in storm_states)

    published = [event for event in bus.events if event.type == "telemetry.understanding.regime"]
    assert published, "regime FSM should publish signals"
    assert published[-1].payload["metadata"]["volatility_state"] == "storm"
