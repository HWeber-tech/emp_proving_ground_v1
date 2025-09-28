"""Multi-timeframe momentum stack for roadmap phase 2A."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from src.core.strategy.engine import BaseStrategy
from src.risk.analytics.volatility_target import (
    calculate_realised_volatility,
    determine_target_allocation,
)

from .models import StrategyAction, StrategySignal

__all__ = [
    "MultiTimeframeMomentumConfig",
    "MultiTimeframeMomentumStrategy",
]

_EPSILON = 1e-12


def _extract_closes(market_data: Mapping[str, object], symbol: str) -> np.ndarray:
    payload = market_data.get(symbol)
    if not isinstance(payload, Mapping):
        raise ValueError(f"Missing payload for symbol {symbol}")
    closes = payload.get("close")
    if closes is None:
        raise ValueError(f"Missing close prices for symbol {symbol}")
    array = np.asarray(closes, dtype=float)
    if array.size < 3:
        raise ValueError("At least three price points are required")
    return array


def _returns(closes: np.ndarray) -> np.ndarray:
    return np.diff(closes) / closes[:-1]


def _default_weights(lookbacks: Sequence[int]) -> np.ndarray:
    weights = np.asarray([1.0 / math.sqrt(max(lb, 1)) for lb in lookbacks], dtype=float)
    total = float(np.sum(np.abs(weights)))
    if total <= _EPSILON:
        return np.ones(len(lookbacks), dtype=float)
    return weights / total


@dataclass(slots=True)
class MultiTimeframeMomentumConfig:
    """Configuration for the multi-timeframe momentum stack."""

    lookbacks: tuple[int, ...] = (10, 30, 90)
    entry_threshold: float = 0.35
    min_alignment: int = 2
    target_volatility: float = 0.10
    max_leverage: float = 2.0
    annualisation_factor: float = math.sqrt(252.0)
    weights: tuple[float, ...] | None = None

    def normalised_weights(self) -> np.ndarray:
        if self.weights is not None:
            if len(self.weights) != len(self.lookbacks):
                raise ValueError("weights must match lookbacks length")
            weights = np.asarray(self.weights, dtype=float)
            total = float(np.sum(np.abs(weights)))
            if total <= _EPSILON:
                raise ValueError("weights must not all be zero")
            return weights / total
        return _default_weights(self.lookbacks)


class MultiTimeframeMomentumStrategy(BaseStrategy):
    """Aggregates momentum signals across multiple timeframes."""

    def __init__(
        self,
        strategy_id: str,
        symbols: list[str],
        *,
        capital: float,
        config: MultiTimeframeMomentumConfig | None = None,
    ) -> None:
        super().__init__(strategy_id=strategy_id, symbols=symbols)
        self._config = config or MultiTimeframeMomentumConfig()
        self._capital = float(capital)

    async def generate_signal(
        self, market_data: Mapping[str, object], symbol: str
    ) -> StrategySignal:
        try:
            closes = _extract_closes(market_data, symbol)
            returns = _returns(closes)
        except Exception as exc:
            return StrategySignal(
                symbol=symbol,
                action="FLAT",
                confidence=0.0,
                notional=0.0,
                metadata={"reason": "insufficient_data", "error": str(exc)},
            )

        lookbacks = self._config.lookbacks
        weights = self._config.normalised_weights()
        if len(lookbacks) == 0:
            return StrategySignal(
                symbol=symbol,
                action="FLAT",
                confidence=0.0,
                notional=0.0,
                metadata={"reason": "no_lookbacks_configured"},
            )

        scores: list[float] = []
        momentum_summary: dict[str, float] = {}
        for lookback, weight in zip(lookbacks, weights):
            if lookback <= 0:
                continue
            if returns.size < lookback:
                raise ValueError(
                    f"Not enough return observations for lookback {lookback}"
                )
            window = returns[-lookback:]
            mean_ret = float(np.mean(window))
            std_ret = float(np.std(window, ddof=1 if window.size > 1 else 0))
            score = mean_ret / max(std_ret, _EPSILON)
            scores.append(score * weight)
            momentum_summary[str(lookback)] = score

        if not scores:
            return StrategySignal(
                symbol=symbol,
                action="FLAT",
                confidence=0.0,
                notional=0.0,
                metadata={"reason": "no_valid_scores"},
            )

        aggregated_score = float(np.sum(scores))
        positive = sum(1 for score in momentum_summary.values() if score > 0.0)
        negative = sum(1 for score in momentum_summary.values() if score < 0.0)

        alignment_required = max(1, self._config.min_alignment)
        action: StrategyAction = "FLAT"
        if aggregated_score >= self._config.entry_threshold and positive >= alignment_required:
            action = "BUY"
        elif aggregated_score <= -self._config.entry_threshold and negative >= alignment_required:
            action = "SELL"

        confidence = 0.0
        notional = 0.0
        longest_lookback = max(lookbacks)
        realised_vol = calculate_realised_volatility(
            returns[-longest_lookback:],
            annualisation_factor=self._config.annualisation_factor,
        )

        if action != "FLAT":
            ratio = abs(aggregated_score) / max(self._config.entry_threshold, _EPSILON)
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
            "aggregated_score": aggregated_score,
            "window_scores": momentum_summary,
            "alignment": {"positive": positive, "negative": negative},
            "target_volatility": self._config.target_volatility,
            "max_leverage": self._config.max_leverage,
            "realised_volatility": realised_vol,
        }

        return StrategySignal(
            symbol=symbol,
            action=action,
            confidence=confidence,
            notional=notional,
            metadata=metadata,
        )
