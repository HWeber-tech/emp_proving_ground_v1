from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Sequence

import numpy as np
import pandas as pd
import pytest

from src.core.event_bus import Event
from src.data_integration.real_data_slice import build_belief_from_market_data
from src.understanding.belief_regime_calibrator import BeliefRegimeCalibration


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


def _frame_from_prices(prices: Sequence[float]) -> pd.DataFrame:
    base = datetime(2025, 1, 1, tzinfo=timezone.utc)
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
                "volume": 1_000 + offset * 10,
            }
        )
    return pd.DataFrame(rows)


@pytest.mark.guardrail
def test_build_belief_from_market_data_calibrates_parameters() -> None:
    calm_prices = [1.10 for _ in range(32)]
    frame = _frame_from_prices(calm_prices)
    bus = _RecordingBus()

    snapshot, belief_state, regime_signal, calibration = build_belief_from_market_data(
        market_data=frame,
        symbol="EURUSD",
        belief_id="calm-belief",
        event_bus=bus,
    )

    assert isinstance(calibration, BeliefRegimeCalibration)
    assert belief_state.metadata["learning_rate"] == pytest.approx(calibration.learning_rate)
    assert belief_state.metadata["decay"] == pytest.approx(calibration.decay)
    assert regime_signal.regime_state.volatility_state in {"calm", "normal"}
    assert bus.events, "calibrated build should emit telemetry"

    covariance = np.array(belief_state.posterior.covariance)
    eigenvalues = np.linalg.eigvalsh(covariance)
    assert np.all(eigenvalues >= -1e-9)
    assert np.all(eigenvalues <= calibration.max_variance + 1e-6)
    assert belief_state.metadata["covariance_condition"] >= 1.0
    assert belief_state.metadata["covariance_max_eigenvalue"] <= calibration.max_variance + 1e-6
    assert belief_state.metadata["covariance_min_eigenvalue"] >= 0.0
    assert belief_state.metadata["covariance_trace"] >= 0.0
    assert snapshot["symbol"] == "EURUSD"


@pytest.mark.guardrail
def test_build_belief_from_market_data_erratic_triggers_storm() -> None:
    erratic = [1.1 + ((-1) ** idx) * 0.02 * (idx % 4 + 1) for idx in range(48)]
    frame = _frame_from_prices(erratic)
    bus = _RecordingBus()

    _, _, regime_signal, calibration = build_belief_from_market_data(
        market_data=frame,
        symbol="EURUSD",
        belief_id="storm-belief",
        event_bus=bus,
    )

    assert isinstance(calibration, BeliefRegimeCalibration)
    assert regime_signal.regime_state.volatility_state == "storm"
    assert regime_signal.metadata["volatility"] >= calibration.calm_threshold
    assert bus.events, "erratic build should emit telemetry"
