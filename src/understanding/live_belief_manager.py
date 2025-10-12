"""Runtime helper that keeps belief/regime components aligned with real data."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Mapping

import numpy as np
import pandas as pd

from src.core.event_bus import EventBus
from src.understanding.belief import BeliefBuffer, BeliefEmitter, BeliefState, RegimeFSM, RegimeSignal
from src.understanding.belief_regime_calibrator import BeliefRegimeCalibration, build_calibrated_belief_components
from src.understanding.belief_real_data_utils import calibrate_from_market_data, extract_snapshot_volatility, resolve_threshold_scale

if TYPE_CHECKING:  # pragma: no cover - typing helpers only
    from src.sensory.real_sensory_organ import RealSensoryOrgan


class LiveBeliefManager:
    """Maintain calibrated belief/regime components while processing live data."""

    def __init__(
        self,
        *,
        belief_id: str,
        symbol: str,
        organ: "RealSensoryOrgan",
        emitter: BeliefEmitter,
        regime_fsm: RegimeFSM,
        calibration: BeliefRegimeCalibration | None = None,
        scaling_hysteresis: float = 0.15,
        scaling_trigger: float = 10.0,
        relaxation_rate: float = 0.08,
    ) -> None:
        self._belief_id = belief_id
        self._symbol = symbol
        self._organ = organ
        self._emitter = emitter
        self._regime_fsm = regime_fsm
        self._calibration = calibration
        self._threshold_scale = 1.0
        self._scaling_hysteresis = scaling_hysteresis if scaling_hysteresis >= 0.0 else 0.0
        if scaling_trigger <= 1.0:
            raise ValueError("scaling_trigger must be greater than 1.0")
        if relaxation_rate <= 0.0 or relaxation_rate >= 1.0:
            raise ValueError("relaxation_rate must be in (0, 1)")
        self._scaling_trigger = float(scaling_trigger)
        self._relaxation_rate = float(relaxation_rate)
        health = regime_fsm.healthcheck()
        self._base_calm_threshold = float(health.get("calm_threshold", calibration.calm_threshold if calibration else 0.0))
        self._base_storm_threshold = float(health.get("storm_threshold", calibration.storm_threshold if calibration else 0.0))
        self._last_snapshot: Mapping[str, object] | None = None
        self._last_belief_state: BeliefState | None = None
        self._last_regime_signal: RegimeSignal | None = None

    @property
    def calibration(self) -> BeliefRegimeCalibration | None:
        return self._calibration

    @property
    def last_snapshot(self) -> Mapping[str, object] | None:
        return self._last_snapshot

    @property
    def last_belief_state(self) -> BeliefState | None:
        return self._last_belief_state

    @property
    def last_regime_signal(self) -> RegimeSignal | None:
        return self._last_regime_signal

    @property
    def regime_fsm(self) -> RegimeFSM:
        return self._regime_fsm

    @property
    def emitter(self) -> BeliefEmitter:
        return self._emitter

    @property
    def symbol(self) -> str:
        return self._symbol

    @classmethod
    def from_market_data(
        cls,
        *,
        market_data: pd.DataFrame,
        symbol: str,
        belief_id: str,
        event_bus: EventBus,
        organ: "RealSensoryOrgan" | None = None,
    ) -> tuple["LiveBeliefManager", Mapping[str, object], BeliefState, RegimeSignal]:
        if market_data.empty:
            raise ValueError("Market data frame is empty; cannot initialise live belief manager")

        calibration = calibrate_from_market_data(market_data)
        if calibration is not None:
            buffer, emitter, regime_fsm = build_calibrated_belief_components(
                calibration,
                belief_id=belief_id,
                regime_signal_id=f"{belief_id}-regime",
                event_bus=event_bus,
            )
        else:
            buffer = BeliefBuffer(belief_id=belief_id)
            emitter = BeliefEmitter(buffer=buffer, event_bus=event_bus)
            regime_fsm = RegimeFSM(event_bus=event_bus, signal_id=f"{belief_id}-regime")

        if organ is None:
            from src.sensory.real_sensory_organ import RealSensoryOrgan  # local import to avoid circular load

            organ = RealSensoryOrgan(event_bus=None)

        live_manager = cls(
            belief_id=belief_id,
            symbol=symbol,
            organ=organ,
            emitter=emitter,
            regime_fsm=regime_fsm,
            calibration=calibration,
        )
        snapshot, belief_state, regime_signal = live_manager.process_market_data(
            market_data,
            symbol=symbol,
            apply_threshold_scaling=True,
        )
        return live_manager, snapshot, belief_state, regime_signal

    def process_market_data(
        self,
        market_data: pd.DataFrame,
        *,
        symbol: str | None = None,
        apply_threshold_scaling: bool | None = None,
    ) -> tuple[Mapping[str, object], BeliefState, RegimeSignal]:
        resolved_symbol = symbol or self._symbol
        snapshot = self._organ.observe(market_data, symbol=resolved_symbol)
        return self.process_snapshot(snapshot, apply_threshold_scaling=apply_threshold_scaling)

    def process_with_fetcher(
        self,
        fetcher: Callable[..., pd.DataFrame],
        *,
        symbol: str | None = None,
        apply_threshold_scaling: bool | None = None,
        **kwargs,
    ) -> tuple[Mapping[str, object], BeliefState, RegimeSignal]:
        resolved_symbol = symbol or self._symbol
        frame = fetcher(resolved_symbol, **kwargs)
        if frame.empty:
            raise ValueError("fetcher returned empty market data frame")
        return self.process_market_data(
            frame,
            symbol=resolved_symbol,
            apply_threshold_scaling=apply_threshold_scaling,
        )

    def recalibrate_from_market_data(
        self,
        market_data: pd.DataFrame,
        *,
        reset_states: bool = True,
    ) -> BeliefRegimeCalibration | None:
        """Re-derive calibration parameters from fresh market data."""

        if market_data.empty:
            raise ValueError("Market data frame is empty; cannot recalibrate belief state")

        calibration = calibrate_from_market_data(market_data)
        if calibration is None:
            return None

        buffer = self._emitter.buffer
        buffer.apply_hyperparameters(
            learning_rate=calibration.learning_rate,
            decay=calibration.decay,
            max_variance=calibration.max_variance,
            min_variance=calibration.min_variance,
            volatility_features=calibration.volatility_features,
            volatility_window=calibration.volatility_window,
            reset_states=reset_states,
            reset_volatility=True,
        )
        self._regime_fsm.reconfigure(
            calm_threshold=calibration.calm_threshold,
            storm_threshold=calibration.storm_threshold,
            volatility_feature=calibration.volatility_feature,
            volatility_window=calibration.volatility_window,
            reset_history=True,
            reset_dynamic_thresholds=True,
        )
        self._calibration = calibration
        self._threshold_scale = 1.0
        self._base_calm_threshold = calibration.calm_threshold
        self._base_storm_threshold = calibration.storm_threshold
        return calibration

    def process_snapshot(
        self,
        snapshot: Mapping[str, object],
        *,
        apply_threshold_scaling: bool | None = None,
    ) -> tuple[Mapping[str, object], BeliefState, RegimeSignal]:
        if apply_threshold_scaling is None:
            should_scale = self._calibration is not None
        else:
            should_scale = bool(apply_threshold_scaling)

        if should_scale and self._calibration is not None:
            sample = extract_snapshot_volatility(snapshot, self._calibration.volatility_feature)
            scale = resolve_threshold_scale(
                self._calibration,
                sample,
                trigger=self._scaling_trigger,
            )
            scaled = False
            if scale is not None:
                target_scale = float(max(scale, 1.0))
                hysteresis = 1.0 + self._scaling_hysteresis
                if target_scale > self._threshold_scale * hysteresis:
                    current = self._threshold_scale if self._threshold_scale > 0.0 else 1.0
                    factor = target_scale / current
                    if factor > 0.0:
                        self._regime_fsm.apply_threshold_scale(factor)
                        self._threshold_scale = target_scale
                        scaled = True
            if not scaled:
                self._maybe_relax_thresholds(sample)

        belief_state = self._emitter.emit(snapshot)
        regime_signal = self._regime_fsm.publish(belief_state)
        self._last_snapshot = snapshot
        self._last_belief_state = belief_state
        self._last_regime_signal = regime_signal
        return snapshot, belief_state, regime_signal

    def _maybe_relax_thresholds(self, sample: float | None) -> None:
        if self._calibration is None:
            return
        if self._threshold_scale <= 1.0:
            return

        baseline = max(self._calibration.calm_threshold, 1e-8)
        sample_ratio: float | None = None
        if sample is not None:
            try:
                sample_value = abs(float(sample))
            except (TypeError, ValueError):
                sample_ratio = None
            else:
                if np.isfinite(sample_value) and baseline > 0.0:
                    sample_ratio = sample_value / baseline

        if sample_ratio is not None:
            relax_threshold = self._scaling_trigger * (1.0 - self._scaling_hysteresis)
            if sample_ratio > relax_threshold:
                return

        target_scale = max(1.0, self._threshold_scale * (1.0 - self._relaxation_rate))
        factor = target_scale / self._threshold_scale
        health = self._regime_fsm.healthcheck()
        current_calm = float(health.get("calm_threshold", 0.0))
        current_storm = float(health.get("storm_threshold", 0.0))
        calm_floor = max(self._base_calm_threshold, self._calibration.calm_threshold)
        storm_floor = max(self._base_storm_threshold, self._calibration.storm_threshold)

        floor_ratio = 0.0
        if current_calm > 0.0:
            floor_ratio = max(floor_ratio, calm_floor / current_calm)
        if current_storm > 0.0:
            floor_ratio = max(floor_ratio, storm_floor / current_storm)
        floor_ratio = float(min(max(floor_ratio, 0.0), 1.0))
        if floor_ratio > factor:
            factor = floor_ratio
            target_scale = self._threshold_scale * factor

        if target_scale >= self._threshold_scale:
            return

        if factor <= 0.0 or not np.isfinite(factor):
            return
        self._regime_fsm.apply_threshold_scale(factor)
        self._threshold_scale = target_scale
