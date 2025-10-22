"""Synthetic volatility trading strategy combining realised vs implied signals."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from src.core.strategy.engine import BaseStrategy
from src.risk.analytics.volatility_target import (
    calculate_realised_volatility,
    determine_target_allocation,
)
from src.trading.strategies.capacity import (
    DEFAULT_L1_CAPACITY_RATIO,
    resolve_l1_depth_cap,
)
from src.trading.strategies.signals.ict_microstructure import (
    ICTMicrostructureAnalyzer,
)

from ..models import StrategySignal

__all__ = ["VolatilityTradingConfig", "VolatilityTradingStrategy"]

_EPSILON = 1e-12


def _extract_closes(market_data: Mapping[str, Any], symbol: str) -> np.ndarray:
    payload = market_data.get(symbol)
    if payload is None or "close" not in payload:
        raise ValueError(f"Missing close prices for symbol {symbol}")
    closes = np.asarray(payload["close"], dtype=float)
    closes = closes[np.isfinite(closes)]
    if closes.size < 2:
        raise ValueError("At least two price points are required")
    return closes


def _returns(closes: np.ndarray) -> np.ndarray:
    return np.diff(closes) / closes[:-1]


def _to_float(value: Any) -> float | None:
    if isinstance(value, Mapping):
        for inner in value.values():
            candidate = _to_float(inner)
            if candidate is not None:
                return candidate
        return None
    if isinstance(value, (list, tuple, np.ndarray)):
        arr = np.asarray(list(value), dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return None
        return float(arr[-1])
    try:
        candidate = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(candidate):
        return None
    return candidate


def _extract_implied_vol(payload: Mapping[str, Any], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        if key in payload:
            candidate = _to_float(payload[key])
            if candidate is not None and candidate > 0.0:
                return candidate
    for key, value in payload.items():
        if isinstance(value, Mapping):
            candidate = _extract_implied_vol(value, keys)
            if candidate is not None:
                return candidate
    return None


@dataclass(slots=True)
class VolatilityTradingConfig:
    """Configuration for the volatility trading strategy."""

    realised_lookback: int = 30
    implied_vol_keys: tuple[str, ...] = (
        "implied_volatility",
        "iv",
        "sigma",
        "volatility",
    )
    vol_spread_entry: float = 0.04
    confidence_scale: float = 0.75
    target_volatility: float = 0.18
    max_leverage: float = 3.0
    annualisation_factor: float = math.sqrt(252.0)
    gamma_lookback: int = 10
    gamma_sensitivity: float = 3.0
    gamma_cap: float = 0.35
    microstructure_weight: float = 0.25


class VolatilityTradingStrategy(BaseStrategy):
    """Trades synthetic volatility using futures exposures and gamma scalping."""

    def __init__(
        self,
        strategy_id: str,
        symbols: list[str],
        *,
        capital: float,
        config: VolatilityTradingConfig | None = None,
        microstructure_analyzer: ICTMicrostructureAnalyzer | None = None,
    ) -> None:
        super().__init__(strategy_id=strategy_id, symbols=symbols)
        self._capital = float(capital)
        self._config = config or VolatilityTradingConfig()
        self._microstructure_analyzer = (
            microstructure_analyzer or ICTMicrostructureAnalyzer()
        )

    async def generate_signal(
        self, market_data: Mapping[str, Any], symbol: str
    ) -> StrategySignal:
        try:
            closes = _extract_closes(market_data, symbol)
            returns = _returns(closes)
        except Exception as exc:  # pragma: no cover - defensive programming
            return StrategySignal(
                symbol=symbol,
                action="FLAT",
                confidence=0.0,
                notional=0.0,
                metadata={"reason": "insufficient_data", "error": str(exc)},
            )

        payload = market_data.get(symbol)
        if not isinstance(payload, Mapping):
            return StrategySignal(
                symbol=symbol,
                action="FLAT",
                confidence=0.0,
                notional=0.0,
                metadata={"reason": "invalid_payload"},
            )

        implied_vol = _extract_implied_vol(payload, self._config.implied_vol_keys)
        if implied_vol is None:
            return StrategySignal(
                symbol=symbol,
                action="FLAT",
                confidence=0.0,
                notional=0.0,
                metadata={"reason": "missing_implied_vol"},
            )

        realised_vol = calculate_realised_volatility(
            returns,
            window=max(2, self._config.realised_lookback),
            annualisation_factor=self._config.annualisation_factor,
        )
        vol_spread = realised_vol - implied_vol

        action = "FLAT"
        direction = 0.0
        if vol_spread >= self._config.vol_spread_entry:
            action = "BUY"
            direction = 1.0
        elif vol_spread <= -self._config.vol_spread_entry:
            action = "SELL"
            direction = -1.0

        cap_limit, cap_details = resolve_l1_depth_cap(
            market_data, symbol, ratio=DEFAULT_L1_CAPACITY_RATIO
        )

        confidence = 0.0
        notional = 0.0
        gamma_metadata: dict[str, Any] | None = None

        if direction != 0.0:
            excess = abs(vol_spread) - self._config.vol_spread_entry
            if excess > 0.0:
                confidence = float(
                    min(
                        1.0,
                        (excess / max(implied_vol, _EPSILON))
                        * self._config.confidence_scale,
                    )
                )

            reference_vol = max(realised_vol, implied_vol, _EPSILON)
            allocation = determine_target_allocation(
                capital=self._capital,
                target_volatility=self._config.target_volatility,
                realised_volatility=reference_vol,
                max_leverage=self._config.max_leverage,
                max_notional=cap_limit,
            )
            notional = allocation.target_notional * direction

            gamma_factor = self._compute_gamma_factor(closes, direction)
            if gamma_factor != 0.0:
                adjusted_notional = notional * (1.0 + gamma_factor)
                gamma_metadata = {
                    "gamma_factor": gamma_factor,
                    "adjusted_notional": adjusted_notional,
                }
                notional = adjusted_notional

            if cap_details:
                raw_target = (
                    allocation.raw_target_notional
                    if allocation.raw_target_notional is not None
                    else allocation.target_notional
                )
                gamma_metadata = gamma_metadata or {}
                gamma_metadata.update(
                    {
                        "liquidity_cap": cap_details,
                        "raw_target_notional": raw_target * direction,
                    }
                )

        metadata: dict[str, Any] = {
            "realised_vol": realised_vol,
            "implied_vol": implied_vol,
            "volatility_spread": vol_spread,
            "action_basis": action,
        }

        if gamma_metadata is not None:
            metadata["gamma_scalping"] = gamma_metadata

        synthetic_payoff = 0.5 * (realised_vol**2 - implied_vol**2)
        metadata["synthetic_payoff_estimate"] = synthetic_payoff

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
                    adjustment = 1.0 + alignment_score * features.confidence * self._config.microstructure_weight
                    confidence = float(max(0.0, min(1.0, confidence * adjustment)))

        return StrategySignal(
            symbol=symbol,
            action=action,
            confidence=confidence,
            notional=notional,
            metadata=metadata,
        )

    def _compute_gamma_factor(self, closes: np.ndarray, direction: float) -> float:
        lookback = max(2, min(self._config.gamma_lookback, closes.size))
        window = closes[-lookback:]
        last_price = float(window[-1])
        base_price = float(np.mean(window))
        if base_price <= 0.0:
            return 0.0
        deviation = (last_price - base_price) / base_price
        bias = -deviation if direction > 0 else deviation
        raw_factor = bias * self._config.gamma_sensitivity
        return float(max(-self._config.gamma_cap, min(self._config.gamma_cap, raw_factor)))
