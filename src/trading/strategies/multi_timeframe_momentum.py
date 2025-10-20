"""Multi-timeframe momentum stack with confirmation logic.

This module implements the roadmap requirement for a momentum strategy that
looks across intraday and daily horizons.  Each timeframe contributes a
weighted momentum score and the resulting aggregate drives the trading action
once a configurable confirmation ratio is met.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, MutableMapping

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

from .models import StrategyAction, StrategySignal

__all__ = [
    "TimeframeMomentumLegConfig",
    "MultiTimeframeMomentumConfig",
    "MultiTimeframeMomentumStrategy",
]

_EPSILON = 1e-12


@dataclass(slots=True)
class TimeframeMomentumLegConfig:
    """Configuration for a single timeframe leg in the stack."""

    timeframe: str
    lookback: int
    weight: float = 1.0
    minimum_observations: int | None = None

    def required_observations(self) -> int:
        """Return the minimum number of closes required for this leg."""

        minimum = self.minimum_observations
        # Need ``lookback`` returns which implies ``lookback + 1`` closes.
        implied = max(self.lookback + 1, 2)
        if minimum is None:
            return implied
        return max(int(minimum), implied)


@dataclass(slots=True)
class MultiTimeframeMomentumConfig:
    """Configuration for the multi-timeframe momentum strategy."""

    timeframes: tuple[TimeframeMomentumLegConfig, ...] = (
        TimeframeMomentumLegConfig("15m", lookback=32, weight=0.25),
        TimeframeMomentumLegConfig("1h", lookback=48, weight=0.35),
        TimeframeMomentumLegConfig("1d", lookback=60, weight=0.40),
    )
    entry_threshold: float = 0.65
    confirmation_ratio: float = 0.6
    target_volatility: float = 0.10
    max_leverage: float = 2.0
    annualisation_factor: float = math.sqrt(252.0)
    volatility_timeframe: str | None = "1d"
    volatility_lookback: int | None = 60

    def total_weight(self) -> float:
        weight = sum(max(0.0, leg.weight) for leg in self.timeframes)
        return weight if weight > 0 else 1.0


def _normalise_closes(series: Any, *, required: int) -> np.ndarray:
    closes = np.asarray(series, dtype=float)
    closes = closes[np.isfinite(closes)]
    if closes.size < required:
        raise ValueError(
            f"expected >= {required} observations, received {closes.size}"
        )
    return closes


def _locate_close_series(payload: Mapping[str, Any], timeframe: str) -> np.ndarray:
    """Extract a closing price series for the requested timeframe."""

    timeframe_key = str(timeframe)
    candidates: list[Any] = []

    timeframes = payload.get("timeframes")
    if isinstance(timeframes, Mapping):
        inner = timeframes.get(timeframe_key)
        if isinstance(inner, Mapping):
            candidates.extend([inner.get("close"), inner.get("closes")])
        elif inner is not None:
            candidates.append(inner)

    # Alternate schema where timeframe is top-level mapping
    direct = payload.get(timeframe_key)
    if isinstance(direct, Mapping):
        candidates.extend([direct.get("close"), direct.get("closes")])
    elif direct is not None:
        candidates.append(direct)

    # Explicit column naming convention e.g. ``close_1h``
    for key in (f"close_{timeframe_key}", f"{timeframe_key}_close"):
        alt = payload.get(key)
        if isinstance(alt, Mapping):
            candidates.extend([alt.get("close"), alt.get("closes")])
        elif alt is not None:
            candidates.append(alt)

    if timeframe_key in {"1d", "daily", "default"}:
        candidates.append(payload.get("close"))

    for candidate in candidates:
        if candidate is None:
            continue
        closes = np.asarray(candidate, dtype=float)
        closes = closes[np.isfinite(closes)]
        if closes.size:
            return closes
    raise ValueError(f"no close series found for timeframe {timeframe_key}")


def _momentum_score(closes: np.ndarray, *, lookback: int) -> tuple[float, float, float]:
    returns = np.diff(closes) / closes[:-1]
    window = returns[-lookback:]
    mean_ret = float(np.mean(window))
    std_ret = float(np.std(window, ddof=1 if window.size > 1 else 0))
    score = mean_ret / max(std_ret, _EPSILON)
    return score, mean_ret, std_ret


class MultiTimeframeMomentumStrategy(BaseStrategy):
    """Stack momentum signals across multiple timeframes."""

    def __init__(
        self,
        strategy_id: str,
        symbols: list[str],
        *,
        capital: float,
        config: MultiTimeframeMomentumConfig | None = None,
    ) -> None:
        super().__init__(strategy_id=strategy_id, symbols=symbols)
        self._capital = float(capital)
        self._config = config or MultiTimeframeMomentumConfig()

    async def generate_signal(
        self, market_data: Mapping[str, Any], symbol: str
    ) -> StrategySignal:
        payload = market_data.get(symbol)
        if not isinstance(payload, Mapping):
            return StrategySignal(
                symbol=symbol,
                action="FLAT",
                confidence=0.0,
                notional=0.0,
                metadata={"reason": "missing_symbol_payload"},
            )

        leg_metrics: MutableMapping[str, Mapping[str, float]] = {}
        weighted_scores: list[float] = []
        weights: list[float] = []
        individual_scores: list[float] = []
        issues: list[str] = []

        for leg in self._config.timeframes:
            try:
                closes = _locate_close_series(payload, leg.timeframe)
                closes = _normalise_closes(closes, required=leg.required_observations())
            except ValueError as exc:
                issues.append(f"{leg.timeframe}: {exc}")
                continue

            try:
                score, mean_ret, std_ret = _momentum_score(closes, lookback=leg.lookback)
            except Exception as exc:
                issues.append(f"{leg.timeframe}: {exc}")
                continue

            leg_metrics[leg.timeframe] = {
                "score": score,
                "mean_return": mean_ret,
                "std_return": std_ret,
                "lookback": float(leg.lookback),
                "weight": float(leg.weight),
            }

            weight = max(0.0, float(leg.weight))
            weighted_scores.append(score * weight)
            weights.append(weight)
            individual_scores.append(score)

        if not weighted_scores:
            return StrategySignal(
                symbol=symbol,
                action="FLAT",
                confidence=0.0,
                notional=0.0,
                metadata={
                    "reason": "insufficient_timeframe_data",
                    "issues": issues,
                },
            )

        total_weight = sum(weights) or 1.0
        aggregate_score = float(sum(weighted_scores) / total_weight)
        valid_count = len(individual_scores)

        if aggregate_score > 0:
            supporters = sum(1 for score in individual_scores if score > 0)
            direction: StrategyAction = "BUY"
        elif aggregate_score < 0:
            supporters = sum(1 for score in individual_scores if score < 0)
            direction = "SELL"
        else:
            supporters = 0
            direction = "FLAT"

        support_ratio = supporters / valid_count if valid_count else 0.0
        action: StrategyAction = direction

        if direction == "BUY" and aggregate_score < self._config.entry_threshold:
            action = "FLAT"
        elif direction == "SELL" and -aggregate_score < self._config.entry_threshold:
            action = "FLAT"

        if action != "FLAT" and support_ratio < self._config.confirmation_ratio:
            action = "FLAT"

        realised_vol = self._compute_realised_volatility(payload)
        cap_limit, cap_details = resolve_l1_depth_cap(
            market_data, symbol, ratio=DEFAULT_L1_CAPACITY_RATIO
        )
        capacity_meta: dict[str, object] | None = None

        confidence = 0.0
        notional = 0.0
        if action != "FLAT":
            base_ratio = abs(aggregate_score) / max(self._config.entry_threshold, _EPSILON)
            confidence = float(min(base_ratio, 2.0) / 2.0)
            confidence *= float(min(max(support_ratio, 0.0), 1.0))

            allocation = determine_target_allocation(
                capital=self._capital,
                target_volatility=self._config.target_volatility,
                realised_volatility=max(realised_vol, 0.0),
                max_leverage=self._config.max_leverage,
                max_notional=cap_limit,
            )
            notional = allocation.target_notional
            if direction == "SELL":
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

        metadata: MutableMapping[str, Any] = {
            "aggregate_score": aggregate_score,
            "support_ratio": support_ratio,
            "timeframes": dict(leg_metrics),
            "issues": issues,
            "volatility_estimate": realised_vol,
            "entry_threshold": self._config.entry_threshold,
            "confirmation_ratio": self._config.confirmation_ratio,
            "target_volatility": self._config.target_volatility,
            "max_leverage": self._config.max_leverage,
        }

        if action == "FLAT" and direction != "FLAT":
            metadata["reason"] = "insufficient_confirmation"

        if capacity_meta is not None:
            metadata["liquidity_capacity"] = capacity_meta

        return StrategySignal(
            symbol=symbol,
            action=action,
            confidence=confidence,
            notional=notional,
            metadata=dict(metadata),
        )

    # ------------------------------------------------------------------
    def _compute_realised_volatility(self, payload: Mapping[str, Any]) -> float:
        timeframe = self._config.volatility_timeframe
        if timeframe is None:
            return 0.0
        try:
            closes = _locate_close_series(payload, timeframe)
            closes = _normalise_closes(
                closes,
                required=max(2, (self._config.volatility_lookback or 0) + 1),
            )
        except ValueError:
            return 0.0

        returns = np.diff(closes) / closes[:-1]
        return calculate_realised_volatility(
            returns,
            window=self._config.volatility_lookback,
            annualisation_factor=self._config.annualisation_factor,
        )
