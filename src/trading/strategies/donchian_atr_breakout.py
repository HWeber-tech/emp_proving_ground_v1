"""Donchian channel breakout with ATR-based trailing stop."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

import numpy as np

from src.core.strategy.engine import BaseStrategy
from src.risk.analytics.volatility_target import (
    calculate_realised_volatility,
    determine_target_allocation,
)

from .models import StrategySignal

__all__ = ["DonchianATRBreakoutConfig", "DonchianATRBreakoutStrategy"]

_EPSILON = 1e-12


def _extract_array(payload: Mapping[str, object], key: str) -> np.ndarray | None:
    values = payload.get(key)
    if values is None:
        return None
    return np.asarray(values, dtype=float)


@dataclass(slots=True)
class DonchianATRBreakoutConfig:
    """Configuration for Donchian breakout strategy with ATR guardrails."""

    channel_lookback: int = 20
    atr_lookback: int = 14
    breakout_buffer_atr: float = 0.5
    trailing_stop_atr: float = 2.0
    target_volatility: float = 0.11
    max_leverage: float = 2.0
    annualisation_factor: float = math.sqrt(252.0)


class DonchianATRBreakoutStrategy(BaseStrategy):
    """Generate breakout signals using Donchian channels and ATR trailing stops."""

    def __init__(
        self,
        strategy_id: str,
        symbols: list[str],
        *,
        capital: float,
        config: DonchianATRBreakoutConfig | None = None,
    ) -> None:
        super().__init__(strategy_id=strategy_id, symbols=symbols)
        self._capital = float(capital)
        self._config = config or DonchianATRBreakoutConfig()

    async def generate_signal(
        self, market_data: Mapping[str, object], symbol: str
    ) -> StrategySignal:
        payload_obj = market_data.get(symbol)
        if not isinstance(payload_obj, Mapping):
            return StrategySignal(
                symbol=symbol,
                action="FLAT",
                confidence=0.0,
                notional=0.0,
                metadata={"reason": "missing_market_data"},
            )

        closes = _extract_array(payload_obj, "close")
        if closes is None or closes.size < max(self._config.channel_lookback, self._config.atr_lookback) + 1:
            return StrategySignal(
                symbol=symbol,
                action="FLAT",
                confidence=0.0,
                notional=0.0,
                metadata={"reason": "insufficient_data"},
            )

        highs = _extract_array(payload_obj, "high")
        lows = _extract_array(payload_obj, "low")
        if highs is None:
            highs = closes
        if lows is None:
            lows = closes

        channel_window = closes[-(self._config.channel_lookback + 1) : -1]
        if channel_window.size == 0:
            channel_window = closes[-self._config.channel_lookback :]
        channel_high = float(np.max(channel_window))
        channel_low = float(np.min(channel_window))
        last_close = float(closes[-1])

        atr = self._compute_atr(closes, highs, lows)

        breakout_buffer = self._config.breakout_buffer_atr * max(atr, _EPSILON)
        upper_trigger = channel_high + breakout_buffer
        lower_trigger = channel_low - breakout_buffer

        if last_close >= upper_trigger:
            action = "BUY"
        elif last_close <= lower_trigger:
            action = "SELL"
        else:
            action = "FLAT"

        returns = np.diff(closes) / closes[:-1]
        realised_vol = calculate_realised_volatility(
            returns[-max(self._config.channel_lookback, self._config.atr_lookback) :],
            annualisation_factor=self._config.annualisation_factor,
        )

        confidence = 0.0
        notional = 0.0
        trailing_stop: float | None = None

        if action != "FLAT":
            if action == "BUY":
                excess = last_close - upper_trigger
                trailing_stop = last_close - self._config.trailing_stop_atr * atr
            else:
                excess = lower_trigger - last_close
                trailing_stop = last_close + self._config.trailing_stop_atr * atr

            threshold = max(breakout_buffer, _EPSILON)
            ratio = max(excess, 0.0) / threshold
            confidence = float(min(ratio, 2.0) / 2.0)
            allocation = determine_target_allocation(
                capital=self._capital,
                target_volatility=self._config.target_volatility,
                realised_volatility=realised_vol,
                max_leverage=self._config.max_leverage,
            )
            notional = allocation.target_notional
            if action == "SELL":
                notional *= -1.0

        metadata = {
            "channel_high": channel_high,
            "channel_low": channel_low,
            "atr": atr,
            "upper_trigger": upper_trigger,
            "lower_trigger": lower_trigger,
            "target_volatility": self._config.target_volatility,
            "max_leverage": self._config.max_leverage,
            "trailing_stop": trailing_stop,
        }

        return StrategySignal(
            symbol=symbol,
            action=action,
            confidence=confidence,
            notional=notional,
            metadata=metadata,
        )

    def _compute_atr(
        self, closes: np.ndarray, highs: np.ndarray, lows: np.ndarray
    ) -> float:
        lookback = self._config.atr_lookback
        if lookback <= 1:
            return float(np.std(np.diff(closes)))

        highs = highs[-lookback:]
        lows = lows[-lookback:]
        closes_window = closes[-(lookback + 1) :]

        true_ranges = []
        for idx in range(1, closes_window.size):
            current_high = float(highs[idx - 1])
            current_low = float(lows[idx - 1])
            prev_close = float(closes_window[idx - 1])
            current_close = float(closes_window[idx])
            tr_components = [
                current_high - current_low,
                abs(current_high - prev_close),
                abs(current_low - prev_close),
                abs(current_close - prev_close),
            ]
            true_ranges.append(max(tr_components))

        if not true_ranges:
            return 0.0
        return float(np.mean(true_ranges))
