"""Momentum strategy implementation aligned with the roadmap milestones."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from src.core.strategy.engine import BaseStrategy
from src.risk.analytics.volatility_target import (
    calculate_realised_volatility,
    determine_target_allocation,
)

from .models import StrategyAction, StrategySignal

__all__ = ["MomentumStrategyConfig", "MomentumStrategy"]

_EPSILON = 1e-12


def _extract_closes(market_data: Mapping[str, Any], symbol: str) -> np.ndarray:
    payload = market_data.get(symbol)
    if payload is None or "close" not in payload:
        raise ValueError(f"Missing close prices for symbol {symbol}")
    closes = np.asarray(payload["close"], dtype=float)
    if closes.size < 2:
        raise ValueError("At least two price points are required")
    return closes


def _window_returns(closes: np.ndarray) -> np.ndarray:
    returns = np.diff(closes) / closes[:-1]
    return returns


@dataclass(slots=True)
class MomentumStrategyConfig:
    """Configuration for the momentum strategy."""

    lookback: int = 20
    entry_threshold: float = 0.5
    target_volatility: float = 0.10
    max_leverage: float = 2.0
    annualisation_factor: float = math.sqrt(252.0)


class MomentumStrategy(BaseStrategy):
    """Mean-variance style momentum strategy using realised volatility sizing."""

    def __init__(
        self,
        strategy_id: str,
        symbols: list[str],
        *,
        capital: float,
        config: MomentumStrategyConfig | None = None,
    ) -> None:
        super().__init__(strategy_id=strategy_id, symbols=symbols)
        self._config = config or MomentumStrategyConfig()
        self._capital = float(capital)

    async def generate_signal(
        self, market_data: Mapping[str, Any], symbol: str
    ) -> StrategySignal:
        try:
            closes = _extract_closes(market_data, symbol)
            returns = _window_returns(closes)
            window = returns[-self._config.lookback :]
        except Exception as exc:
            return StrategySignal(
                symbol=symbol,
                action="FLAT",
                confidence=0.0,
                notional=0.0,
                metadata={"reason": "insufficient_data", "error": str(exc)},
            )

        mean_ret = float(np.mean(window))
        std_ret = float(np.std(window, ddof=1 if window.size > 1 else 0))
        sharpe_like = mean_ret / max(std_ret, _EPSILON)

        if sharpe_like >= self._config.entry_threshold:
            action: StrategyAction = "BUY"
        elif sharpe_like <= -self._config.entry_threshold:
            action = "SELL"
        else:
            action = "FLAT"

        confidence = 0.0
        notional = 0.0
        realised_vol = calculate_realised_volatility(
            window,
            annualisation_factor=self._config.annualisation_factor,
        )

        if action != "FLAT":
            ratio = abs(sharpe_like) / max(self._config.entry_threshold, _EPSILON)
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
            "momentum_mean": mean_ret,
            "momentum_std": std_ret,
            "momentum_score": sharpe_like,
            "realised_volatility": realised_vol,
            "entry_threshold": self._config.entry_threshold,
            "target_volatility": self._config.target_volatility,
            "max_leverage": self._config.max_leverage,
        }

        return StrategySignal(
            symbol=symbol,
            action=action,
            confidence=confidence,
            notional=notional,
            metadata=metadata,
        )
