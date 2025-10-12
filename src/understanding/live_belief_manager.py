"""Runtime helper that keeps belief/regime components aligned with real data."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Mapping

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
    ) -> None:
        self._belief_id = belief_id
        self._symbol = symbol
        self._organ = organ
        self._emitter = emitter
        self._regime_fsm = regime_fsm
        self._calibration = calibration
        self._threshold_scaled = calibration is None
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

    def process_snapshot(
        self,
        snapshot: Mapping[str, object],
        *,
        apply_threshold_scaling: bool | None = None,
    ) -> tuple[Mapping[str, object], BeliefState, RegimeSignal]:
        should_scale = apply_threshold_scaling
        if should_scale is None:
            should_scale = not self._threshold_scaled

        if (
            should_scale
            and not self._threshold_scaled
            and self._calibration is not None
        ):
            sample = extract_snapshot_volatility(snapshot, self._calibration.volatility_feature)
            scale = resolve_threshold_scale(self._calibration, sample)
            if scale is not None:
                self._regime_fsm.apply_threshold_scale(scale)
                self._threshold_scaled = True

        belief_state = self._emitter.emit(snapshot)
        regime_signal = self._regime_fsm.publish(belief_state)
        self._last_snapshot = snapshot
        self._last_belief_state = belief_state
        self._last_regime_signal = regime_signal
        return snapshot, belief_state, regime_signal
