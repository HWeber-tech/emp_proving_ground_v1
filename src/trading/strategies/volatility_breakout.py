"""Volatility breakout strategy supporting Phase 2 roadmap milestones."""

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
from src.trading.strategies.signals.ict_microstructure import (
    ICTMicrostructureAnalyzer,
)

from .models import StrategySignal

__all__ = ["VolatilityBreakoutConfig", "VolatilityBreakoutStrategy"]

_EPSILON = 1e-12


def _extract_closes(market_data: Mapping[str, Any], symbol: str) -> np.ndarray:
    payload = market_data.get(symbol)
    if payload is None or "close" not in payload:
        raise ValueError(f"Missing close prices for symbol {symbol}")
    closes = np.asarray(payload["close"], dtype=float)
    if closes.size < 2:
        raise ValueError("At least two price points are required")
    return closes


def _returns(closes: np.ndarray) -> np.ndarray:
    return np.diff(closes) / closes[:-1]


@dataclass(slots=True)
class VolatilityBreakoutConfig:
    """Configuration describing breakout detection thresholds."""

    breakout_lookback: int = 14
    baseline_lookback: int = 30
    volatility_multiplier: float = 1.4
    price_channel_lookback: int = 10
    target_volatility: float = 0.12
    max_leverage: float = 2.5
    annualisation_factor: float = math.sqrt(252.0)


class VolatilityBreakoutStrategy(BaseStrategy):
    """Detects volatility regime shifts combined with price channel breaks."""

    def __init__(
        self,
        strategy_id: str,
        symbols: list[str],
        *,
        capital: float,
        config: VolatilityBreakoutConfig | None = None,
        microstructure_analyzer: ICTMicrostructureAnalyzer | None = None,
    ) -> None:
        super().__init__(strategy_id=strategy_id, symbols=symbols)
        self._capital = float(capital)
        self._config = config or VolatilityBreakoutConfig()
        self._microstructure_analyzer = (
            microstructure_analyzer or ICTMicrostructureAnalyzer()
        )

    async def generate_signal(
        self, market_data: Mapping[str, Any], symbol: str
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

        recent_returns = returns[-self._config.breakout_lookback :]
        baseline_returns = returns[-self._config.baseline_lookback :]

        recent_vol = calculate_realised_volatility(
            recent_returns,
            annualisation_factor=self._config.annualisation_factor,
        )
        baseline_vol = calculate_realised_volatility(
            baseline_returns,
            annualisation_factor=self._config.annualisation_factor,
        )
        vol_ratio = recent_vol / max(baseline_vol, _EPSILON)

        channel_prices = closes[-self._config.price_channel_lookback :]
        channel_high = float(np.max(channel_prices))
        channel_low = float(np.min(channel_prices))
        last_price = float(closes[-1])

        if vol_ratio >= self._config.volatility_multiplier:
            if last_price >= channel_high:
                action = "BUY"
            elif last_price <= channel_low:
                action = "SELL"
            else:
                action = "FLAT"
        else:
            action = "FLAT"

        confidence = 0.0
        notional = 0.0
        if action != "FLAT":
            confidence = float(
                min(vol_ratio / max(self._config.volatility_multiplier, _EPSILON), 2.0) / 2.0
            )
            allocation = determine_target_allocation(
                capital=self._capital,
                target_volatility=self._config.target_volatility,
                realised_volatility=recent_vol,
                max_leverage=self._config.max_leverage,
            )
            notional = allocation.target_notional
            if action == "SELL":
                notional *= -1.0

        metadata = {
            "recent_vol": recent_vol,
            "baseline_vol": baseline_vol,
            "volatility_ratio": vol_ratio,
            "channel_high": channel_high,
            "channel_low": channel_low,
            "last_price": last_price,
            "volatility_multiplier": self._config.volatility_multiplier,
            "target_volatility": self._config.target_volatility,
        }

        if self._microstructure_analyzer is not None:
            features = await self._microstructure_analyzer.summarise(market_data, symbol)
            if features is not None:
                alignment_score, breakdown = features.alignment_assessment(action)
                microstructure_metadata = features.to_metadata()
                microstructure_metadata["alignment"] = {
                    "score": alignment_score,
                    "breakdown": breakdown,
                }
                metadata["microstructure"] = microstructure_metadata
                if action in {"BUY", "SELL"} and confidence > 0.0 and alignment_score != 0.0:
                    adjustment = 1.0 + alignment_score * features.confidence * 0.25
                    confidence = float(
                        max(0.0, min(1.0, confidence * adjustment))
                    )

        return StrategySignal(
            symbol=symbol,
            action=action,
            confidence=confidence,
            notional=notional,
            metadata=metadata,
        )
