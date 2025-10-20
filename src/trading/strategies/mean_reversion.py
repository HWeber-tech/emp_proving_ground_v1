"""Mean reversion strategy fulfilling roadmap Phase 2 commitments."""

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
from src.trading.strategies.capacity import (
    DEFAULT_L1_CAPACITY_RATIO,
    resolve_l1_depth_cap,
)

from .models import StrategySignal

__all__ = ["MeanReversionStrategyConfig", "MeanReversionStrategy"]

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
class MeanReversionStrategyConfig:
    """Configuration for the mean reversion strategy."""

    lookback: int = 30
    zscore_entry: float = 1.0
    target_volatility: float = 0.08
    max_leverage: float = 1.5
    annualisation_factor: float = math.sqrt(252.0)


class MeanReversionStrategy(BaseStrategy):
    """Bollinger-band inspired mean reversion implementation."""

    def __init__(
        self,
        strategy_id: str,
        symbols: list[str],
        *,
        capital: float,
        config: MeanReversionStrategyConfig | None = None,
        microstructure_analyzer: ICTMicrostructureAnalyzer | None = None,
    ) -> None:
        super().__init__(strategy_id=strategy_id, symbols=symbols)
        self._capital = float(capital)
        self._config = config or MeanReversionStrategyConfig()
        self._microstructure_analyzer = (
            microstructure_analyzer or ICTMicrostructureAnalyzer()
        )

    async def generate_signal(
        self, market_data: Mapping[str, Any], symbol: str
    ) -> StrategySignal:
        try:
            closes = _extract_closes(market_data, symbol)
            prices = closes[-self._config.lookback :]
            returns = _returns(closes)
            return_window = returns[-self._config.lookback :]
        except Exception as exc:
            return StrategySignal(
                symbol=symbol,
                action="FLAT",
                confidence=0.0,
                notional=0.0,
                metadata={"reason": "insufficient_data", "error": str(exc)},
            )

        mean_price = float(np.mean(prices))
        std_price = float(np.std(prices, ddof=1 if prices.size > 1 else 0))
        last_price = float(prices[-1])
        zscore = (last_price - mean_price) / max(std_price, _EPSILON)

        if zscore >= self._config.zscore_entry:
            action = "SELL"
        elif zscore <= -self._config.zscore_entry:
            action = "BUY"
        else:
            action = "FLAT"

        confidence = 0.0
        notional = 0.0
        capacity_meta: dict[str, object] | None = None
        cap_limit, cap_details = resolve_l1_depth_cap(
            market_data, symbol, ratio=DEFAULT_L1_CAPACITY_RATIO
        )
        realised_vol = calculate_realised_volatility(
            return_window,
            annualisation_factor=self._config.annualisation_factor,
        )

        if action != "FLAT":
            ratio = abs(zscore) / max(self._config.zscore_entry, _EPSILON)
            confidence = float(min(ratio, 2.0) / 2.0)
            allocation = determine_target_allocation(
                capital=self._capital,
                target_volatility=self._config.target_volatility,
                realised_volatility=realised_vol,
                max_leverage=self._config.max_leverage,
                max_notional=cap_limit,
            )
            notional = allocation.target_notional
            if action == "SELL":
                notional *= -1.0
            if cap_details:
                raw_target = (
                    allocation.raw_target_notional
                    if allocation.raw_target_notional is not None
                    else allocation.target_notional
                )
                capacity_meta = {
                    **cap_details,
                    "raw_target_notional": raw_target,
                    "notional_after_cap": allocation.target_notional,
                    "cap_applied": bool(raw_target > allocation.target_notional + 1e-9),
                }

        metadata = {
            "mean_price": mean_price,
            "price_std": std_price,
            "last_price": last_price,
            "zscore": zscore,
            "target_volatility": self._config.target_volatility,
            "max_leverage": self._config.max_leverage,
            "realised_volatility": realised_vol,
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
                    adjustment = 1.0 + alignment_score * features.confidence * 0.3
                    confidence = float(
                        max(0.0, min(1.0, confidence * adjustment))
                    )

        if capacity_meta is not None:
            metadata["liquidity_capacity"] = capacity_meta

        return StrategySignal(
            symbol=symbol,
            action=action,
            confidence=confidence,
            notional=notional,
            metadata=metadata,
        )
