"""Belief/regime calibration helpers for real-market data integration.

The roadmap task *Belief & Regime Integration with Real Data* requires
that BeliefState buffers stay numerically stable when driven by live
market feeds and that the regime finite-state machine transitions
between calm/normal/storm phases in response to observed volatility.

This module inspects historical price series to derive Hebbian learning
rates, decay factors, and volatility thresholds that can be injected
into :mod:`src.understanding.belief` components.  The heuristics favour
conservative defaults – minimum learning/decay floors prevent the buffer
from freezing during low-volatility periods, while percentile-derived
volatility bands ensure "storm" is only triggered when dispersion
meaningfully departs from baseline levels.  Diagnostics are returned so
operators can document how the calibration was produced before enabling
the real-data ingest path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, MutableMapping, Sequence

import numpy as np

from src.core.event_bus import EventBus
from src.understanding.belief import BeliefBuffer, BeliefEmitter, RegimeFSM

__all__ = [
    "BeliefRegimeCalibration",
    "calibrate_belief_and_regime",
    "build_calibrated_belief_components",
]


@dataclass(slots=True, frozen=True)
class BeliefRegimeCalibration:
    """Container holding derived parameters for belief/regime components."""

    learning_rate: float
    decay: float
    max_variance: float
    min_variance: float
    calm_threshold: float
    storm_threshold: float
    volatility_window: int
    volatility_feature: str
    volatility_features: tuple[str, ...] = field(default_factory=tuple)
    diagnostics: Mapping[str, float] = field(default_factory=dict)

    def as_dict(self) -> Mapping[str, float]:
        payload: MutableMapping[str, float] = {
            "learning_rate": self.learning_rate,
            "decay": self.decay,
            "max_variance": self.max_variance,
            "min_variance": self.min_variance,
            "calm_threshold": self.calm_threshold,
            "storm_threshold": self.storm_threshold,
            "volatility_window": float(self.volatility_window),
        }
        payload.update(self.diagnostics)
        return payload


def _as_returns(series: Sequence[float]) -> np.ndarray:
    if not isinstance(series, Sequence) or len(series) < 3:
        raise ValueError("price series must contain at least three samples")
    array = np.asarray(series, dtype=float)
    if np.any(~np.isfinite(array)):
        raise ValueError("price series contains non-finite values")
    returns = np.diff(array) / np.clip(array[:-1], 1e-12, None)
    returns = np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)
    return returns


def _clamp(value: float, low: float, high: float) -> float:
    return float(max(low, min(high, value)))


def calibrate_belief_and_regime(
    prices: Sequence[float] | Iterable[float],
    *,
    volatility_features: Sequence[str] | None = None,
    min_learning_rate: float = 0.05,
    max_learning_rate: float = 0.35,
    min_decay: float = 0.02,
    max_decay: float = 0.25,
    target_half_life: int = 48,
) -> BeliefRegimeCalibration:
    """Derive stable Hebbian/volatility parameters from a price series."""

    series = list(prices)
    returns = _as_returns(series)
    abs_returns = np.abs(returns)
    dispersion = float(np.std(returns, ddof=0))
    median_abs = float(np.median(abs_returns))
    mad = float(np.median(np.abs(abs_returns - median_abs)))
    spread = float(np.std(abs_returns, ddof=0))
    epsilon = max(spread, 1e-4)

    learning_scale = dispersion * 6.0 + median_abs * 2.0
    learning_rate = _clamp(learning_scale, min_learning_rate, max_learning_rate)

    half_life = max(target_half_life, 1)
    base_decay = 1.0 - pow(0.5, 1.0 / half_life)
    decay_scale = 1.0 + min(1.5, dispersion * 12.0)
    decay = _clamp(base_decay * decay_scale, min_decay, max_decay)

    variance_high = float(np.quantile(abs_returns ** 2, 0.95))
    max_variance = max(variance_high * 10.0, 0.05)
    min_variance = min(max(max_variance * 1e-4, 1e-8), max_variance * 0.1)

    calm_quantile = float(np.quantile(abs_returns, 0.3))
    storm_quantile = float(np.quantile(abs_returns, 0.9))
    calm_threshold = max(calm_quantile + epsilon * 0.5, epsilon * 0.5)
    storm_threshold = storm_quantile + epsilon * 3.0
    if storm_threshold <= calm_threshold:
        storm_threshold = calm_threshold + max(epsilon, 0.05)

    volatility_window = int(
        _clamp(round(len(series) * 0.2), 8, 96)
    )

    features = tuple(str(name) for name in (volatility_features or ("HOW_signal", "ANOMALY_signal")))
    volatility_feature = features[0] if features else "HOW_signal"

    diagnostics: MutableMapping[str, float] = {
        "returns_std": dispersion,
        "median_abs_return": median_abs,
        "mad_abs_return": mad,
    }

    return BeliefRegimeCalibration(
        learning_rate=learning_rate,
        decay=decay,
        max_variance=max_variance,
        min_variance=min_variance,
        calm_threshold=calm_threshold,
        storm_threshold=storm_threshold,
        volatility_window=volatility_window,
        volatility_feature=volatility_feature,
        volatility_features=features,
        diagnostics=diagnostics,
    )


def build_calibrated_belief_components(
    calibration: BeliefRegimeCalibration,
    *,
    belief_id: str,
    regime_signal_id: str,
    event_bus: EventBus,
) -> tuple[BeliefBuffer, BeliefEmitter, RegimeFSM]:
    """Instantiate belief/regime components using calibration parameters."""

    buffer = BeliefBuffer(
        belief_id=belief_id,
        learning_rate=calibration.learning_rate,
        decay=calibration.decay,
        max_variance=calibration.max_variance,
        min_variance=calibration.min_variance,
        volatility_features=calibration.volatility_features,
        volatility_window=calibration.volatility_window,
    )
    emitter = BeliefEmitter(buffer=buffer, event_bus=event_bus)
    fsm = RegimeFSM(
        event_bus=event_bus,
        signal_id=regime_signal_id,
        calm_threshold=calibration.calm_threshold,
        storm_threshold=calibration.storm_threshold,
        volatility_window=calibration.volatility_window,
        volatility_feature=calibration.volatility_feature,
    )
    return buffer, emitter, fsm

