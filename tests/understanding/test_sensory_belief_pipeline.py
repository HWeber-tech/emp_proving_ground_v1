from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Mapping

import pandas as pd
import pytest

from src.core.event_bus import Event
from src.sensory.enhanced.how_dimension import InstitutionalUnderstandingEngine
from src.sensory.how.how_sensor import HowSensor
from src.sensory.real_sensory_organ import RealSensoryOrgan, SensoryDriftConfig
from src.understanding.belief import BeliefBuffer, BeliefEmitter


class _RecordingBus:
    def __init__(self) -> None:
        self.events: list[Event] = []

    def is_running(self) -> bool:
        return True

    def publish_from_sync(self, event: Event) -> int:
        self.events.append(event)
        return 1


def _build_market_frame(total_points: int = 30, spike_start: int = 24) -> pd.DataFrame:
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    price = 1.10
    rows: list[dict[str, object]] = []
    for idx in range(total_points):
        timestamp = base + timedelta(minutes=idx)
        in_spike = idx >= spike_start

        if in_spike:
            price += 0.05
            volatility = 0.03
            spread = 0.004
            order_imbalance = -2.0
            depth = 1200.0
            volume = 400.0 + idx * 5.0
            macro_bias = -0.35
            yield_two_year = 0.031 + idx * 0.0002
            yield_ten_year = 0.027 + idx * 0.00015
        else:
            price += 0.0005
            volatility = 0.0005
            spread = 0.00002
            order_imbalance = 0.1
            depth = 7500.0
            volume = 1200.0 + idx * 10.0
            macro_bias = 0.12
            yield_two_year = 0.021 + idx * 0.00005
            yield_ten_year = 0.029 + idx * 0.00004

        open_price = price - 0.0004
        high_price = price + (0.0007 if not in_spike else 0.02)
        low_price = price - (0.0006 if not in_spike else 0.02)

        rows.append(
            {
                "timestamp": timestamp,
                "symbol": "EURUSD",
                "open": open_price,
                "high": high_price,
                "low": low_price,
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
                "data_quality": 0.95,
            }
        )
    return pd.DataFrame(rows)


@pytest.mark.guardrail
def test_sensory_pipeline_emits_belief_and_detects_drift() -> None:
    frame = _build_market_frame()

    how_engine = InstitutionalUnderstandingEngine(random_source=lambda: 0.0)
    drift_config = SensoryDriftConfig(
        baseline_window=10,
        evaluation_window=5,
        min_observations=5,
        z_threshold=0.8,
        sensors=("HOW",),
    )
    organ = RealSensoryOrgan(
        how_sensor=HowSensor(engine=how_engine),
        drift_config=drift_config,
    )

    bus = _RecordingBus()
    buffer = BeliefBuffer(
        belief_id="acceptance-sensory",
        max_variance=0.5,
        min_variance=1e-6,
        volatility_features=("HOW_signal", "ANOMALY_z_score"),
        volatility_window=12,
    )
    emitter = BeliefEmitter(buffer=buffer, event_bus=bus)

    anomaly_hits = 0
    states = []

    for idx in range(frame.shape[0]):
        window = frame.iloc[: idx + 1]
        snapshot = organ.observe(window)
        dimensions = snapshot["dimensions"]
        assert set(dimensions.keys()) == {"WHY", "WHAT", "WHEN", "HOW", "ANOMALY"}

        anomaly_payload = dimensions["ANOMALY"].get("value")
        if isinstance(anomaly_payload, Mapping) and anomaly_payload.get("is_anomaly"):
            anomaly_hits += 1

        state = emitter.emit(snapshot)
        states.append(state)

    assert anomaly_hits >= 1, "expected anomaly flag during volatility spike"

    final_state = states[-1]
    feature_means = dict(zip(final_state.features, final_state.posterior.mean))
    assert feature_means.get("ANOMALY_flag", 0.0) > 0.0
    assert feature_means.get("ANOMALY_z_score", 0.0) > 0.0
    assert feature_means.get("WHAT_last_close", 0.0) > 0.0
    assert "HOW_liquidity" in feature_means
    assert feature_means.get("HOW_volatility", 0.0) > 0.0
    assert feature_means.get("HOW_quality_confidence", 0.0) > 0.0
    assert feature_means.get("WHAT_data_quality", 0.0) > 0.0
    assert feature_means.get("WHEN_data_quality", 0.0) > 0.0
    assert feature_means.get("WHY_quality_strength", 0.0) != 0.0
    assert feature_means.get("ANOMALY_data_quality", 0.0) > 0.0
    assert "WHEN_session" in feature_means
    assert "WHEN_news" in feature_means
    assert "WHEN_gamma" in feature_means

    drift_summary = organ.status()["drift_summary"]
    assert drift_summary is not None
    exceeded = drift_summary.get("exceeded") or []
    assert any(entry.get("sensor") == "HOW" for entry in exceeded)

    assert bus.events, "belief emitter should publish telemetry events"
