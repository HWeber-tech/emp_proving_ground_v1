from __future__ import annotations

from copy import deepcopy
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


@pytest.mark.guardrail
def test_live_belief_manager_rescales_thresholds_for_large_spike() -> None:
    base_prices = [1.105 + idx * 0.00005 for idx in range(64)]
    frame = _frame_from_prices(base_prices)
    bus = _RecordingBus()

    manager, snapshot, _belief_state, _signal = LiveBeliefManager.from_market_data(
        market_data=frame,
        symbol="EURUSD",
        belief_id="spike-belief",
        event_bus=bus,
    )

    initial_health = manager.regime_fsm.healthcheck()
    base_calm = float(initial_health["calm_threshold"])
    base_storm = float(initial_health["storm_threshold"])

    # Craft a volatility spike by cloning the last processed snapshot and amplifying HOW/ANOMALY signals.
    spike_snapshot = deepcopy(snapshot)
    spike_snapshot["generated_at"] = snapshot["generated_at"] + timedelta(minutes=1)

    calibration = manager.calibration
    saturation_cap = None
    if calibration is not None:
        saturation_cap = float(calibration.calm_threshold * 1e4)

    how_payload = spike_snapshot["dimensions"]["HOW"]
    spike_magnitude = base_calm * 1.2 or 1.0
    if saturation_cap is not None:
        spike_magnitude = min(spike_magnitude, saturation_cap * 0.99)
    spike_magnitude = float(max(spike_magnitude, base_calm * 1.05, 1e-4))
    how_payload["signal"] = float(spike_magnitude)
    how_payload.setdefault("confidence", 0.7)
    value_payload = how_payload.setdefault("value", {})
    value_payload["volatility"] = float(spike_magnitude)
    value_payload["strength"] = float(spike_magnitude)
    metadata_payload = how_payload.setdefault("metadata", {})
    metadata_payload["state"] = "alert"

    anomaly_payload = spike_snapshot["dimensions"]["ANOMALY"]
    anomaly_signal = spike_magnitude * 0.9
    anomaly_payload["signal"] = float(anomaly_signal)
    anomaly_payload.setdefault("confidence", 0.6)
    anomaly_value = anomaly_payload.setdefault("value", {})
    anomaly_value["is_anomaly"] = True
    anomaly_value["z_score"] = float(anomaly_signal * 5.0)
    anomaly_meta = anomaly_payload.setdefault("metadata", {})
    anomaly_meta["is_anomaly"] = True
    audit_payload = anomaly_meta.setdefault("audit", {})
    audit_payload["z_score"] = float(anomaly_signal * 5.0)

    integrated = spike_snapshot["integrated_signal"]
    integrated.strength = float(min(max(integrated.strength + 0.15, -1.0), 1.0))
    integrated.confidence = float(min(max(integrated.confidence, 0.35), 0.95))

    _, _belief_state, first_spike_signal = manager.process_snapshot(
        spike_snapshot,
        apply_threshold_scaling=True,
    )

    first_spike_health = manager.regime_fsm.healthcheck()
    assert first_spike_health["calm_threshold"] > base_calm
    assert first_spike_health["storm_threshold"] > base_storm
    assert first_spike_signal.regime_state.volatility_state in {"calm", "normal", "storm"}

    mega_snapshot = deepcopy(spike_snapshot)
    mega_snapshot["generated_at"] = spike_snapshot["generated_at"] + timedelta(minutes=1)
    mega_magnitude = base_calm * 1.4
    if saturation_cap is not None:
        mega_magnitude = min(mega_magnitude, saturation_cap * 0.99)
    mega_magnitude = float(max(mega_magnitude, spike_magnitude * 1.1))
    mega_snapshot["dimensions"]["HOW"]["signal"] = float(mega_magnitude)
    mega_snapshot["dimensions"]["HOW"]["value"]["volatility"] = float(mega_magnitude)
    mega_snapshot["dimensions"]["HOW"]["value"]["strength"] = float(mega_magnitude)
    mega_snapshot["dimensions"]["ANOMALY"]["signal"] = float(mega_magnitude * 0.9)
    mega_snapshot["dimensions"]["ANOMALY"]["value"]["z_score"] = float(mega_magnitude * 4.5)
    mega_snapshot["dimensions"]["ANOMALY"]["metadata"]["audit"]["z_score"] = float(mega_magnitude * 4.5)
    _, _belief_state, mega_signal = manager.process_snapshot(
        mega_snapshot,
        apply_threshold_scaling=True,
    )

    mega_health = manager.regime_fsm.healthcheck()
    assert mega_health["calm_threshold"] >= first_spike_health["calm_threshold"]
    assert mega_health["storm_threshold"] >= first_spike_health["storm_threshold"]
    assert mega_health["calm_threshold"] < mega_health["storm_threshold"]
    assert mega_signal.regime_state.volatility_state in {"calm", "normal", "storm"}


@pytest.mark.guardrail
def test_live_belief_manager_relaxes_thresholds_after_spike() -> None:
    base_prices = [1.1002 + idx * 0.00004 for idx in range(80)]
    frame = _frame_from_prices(base_prices)
    bus = _RecordingBus()

    manager, snapshot, _belief_state, _signal = LiveBeliefManager.from_market_data(
        market_data=frame,
        symbol="EURUSD",
        belief_id="relax-belief",
        event_bus=bus,
    )

    base_health = manager.regime_fsm.healthcheck()
    base_calm = float(base_health["calm_threshold"])
    base_storm = float(base_health["storm_threshold"])

    calibration = manager.calibration
    assert calibration is not None

    spike_snapshot = deepcopy(snapshot)
    spike_snapshot["generated_at"] = snapshot["generated_at"] + timedelta(minutes=1)

    spike_magnitude = float(max(calibration.calm_threshold * 12.0, base_calm * 2.0, 1e-4))
    how_payload = spike_snapshot["dimensions"]["HOW"]
    how_payload["signal"] = spike_magnitude
    how_payload.setdefault("confidence", 0.7)
    how_value = how_payload.setdefault("value", {})
    how_value["volatility"] = spike_magnitude
    how_value["strength"] = spike_magnitude

    anomaly_payload = spike_snapshot["dimensions"]["ANOMALY"]
    anomaly_payload["signal"] = spike_magnitude * 0.9
    anomaly_payload.setdefault("confidence", 0.6)
    anomaly_value = anomaly_payload.setdefault("value", {})
    anomaly_value["is_anomaly"] = True
    anomaly_value["z_score"] = spike_magnitude * 4.5
    anomaly_meta = anomaly_payload.setdefault("metadata", {})
    anomaly_meta["is_anomaly"] = True
    audit_payload = anomaly_meta.setdefault("audit", {})
    audit_payload["z_score"] = spike_magnitude * 4.5

    integrated = spike_snapshot["integrated_signal"]
    integrated.strength = float(min(max(integrated.strength + 0.1, -1.0), 1.0))
    integrated.confidence = float(min(max(integrated.confidence, 0.35), 0.95))

    _, _belief_state, spike_signal = manager.process_snapshot(
        spike_snapshot,
        apply_threshold_scaling=True,
    )

    spike_health = manager.regime_fsm.healthcheck()
    assert spike_health["calm_threshold"] > base_calm
    assert spike_health["storm_threshold"] > base_storm
    assert spike_signal.regime_state.volatility_state in {"calm", "normal", "storm"}

    calm_snapshot = deepcopy(spike_snapshot)
    calm_snapshot["dimensions"]["HOW"]["signal"] = float(calibration.calm_threshold * 0.75)
    calm_snapshot["dimensions"]["HOW"]["value"]["volatility"] = float(calibration.calm_threshold * 0.75)
    calm_snapshot["dimensions"]["ANOMALY"]["signal"] = float(calibration.calm_threshold * 0.6)
    calm_snapshot["dimensions"]["ANOMALY"]["value"]["z_score"] = float(calibration.calm_threshold * 3.0)
    calm_snapshot["dimensions"]["ANOMALY"]["metadata"]["audit"]["z_score"] = float(calibration.calm_threshold * 3.0)
    integrated_after = calm_snapshot["integrated_signal"]
    integrated_after.strength = float(max(min(integrated_after.strength - 0.2, 1.0), -1.0))
    integrated_after.confidence = float(min(max(integrated_after.confidence, 0.35), 0.95))

    calm_start = spike_snapshot["generated_at"]
    for idx in range(1, 36):
        calm_snapshot["generated_at"] = calm_start + timedelta(minutes=idx + 1)
        _, _belief_state, calm_signal = manager.process_snapshot(
            calm_snapshot,
            apply_threshold_scaling=True,
        )
        assert calm_signal.regime_state.volatility_state in {"calm", "normal", "storm"}

    relaxed_health = manager.regime_fsm.healthcheck()
    assert relaxed_health["calm_threshold"] < spike_health["calm_threshold"]
    assert relaxed_health["storm_threshold"] < spike_health["storm_threshold"]
    assert relaxed_health["calm_threshold"] >= calibration.calm_threshold
    assert relaxed_health["storm_threshold"] >= calibration.storm_threshold
