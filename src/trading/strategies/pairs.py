"""Cointegration-based pair trading strategy aligned with the roadmap."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import math
import numpy as np

from src.core.strategy.engine import BaseStrategy

from .models import StrategySignal

__all__ = ["PairTradingConfig", "PairTradingStrategy"]


@dataclass(slots=True)
class PairTradingConfig:
    """Configuration parameters for the pair trading strategy."""

    lookback: int = 120
    zscore_entry: float = 2.0
    zscore_exit: float = 0.5
    adf_stat_threshold: float = -2.5
    min_half_life: float = 0.1
    max_half_life: float = 120.0
    max_notional_fraction: float = 0.1
    min_spread_std: float = 1e-4

    def __post_init__(self) -> None:
        if self.lookback < 30:
            raise ValueError("lookback must be at least 30 observations")
        if self.zscore_entry <= 0:
            raise ValueError("zscore_entry must be positive")
        if self.zscore_exit < 0:
            raise ValueError("zscore_exit must be non-negative")
        if self.max_notional_fraction <= 0:
            raise ValueError("max_notional_fraction must be positive")
        if self.min_spread_std < 0:
            raise ValueError("min_spread_std must be non-negative")
        if self.min_half_life <= 0:
            raise ValueError("min_half_life must be positive")
        if self.max_half_life <= self.min_half_life:
            raise ValueError("max_half_life must exceed min_half_life")


class PairTradingStrategy(BaseStrategy):
    """Simple Engle–Granger style pair-trading implementation."""

    def __init__(
        self,
        strategy_id: str,
        symbols: list[str],
        *,
        capital: float,
        config: PairTradingConfig | None = None,
    ) -> None:
        if len(symbols) != 2:
            raise ValueError("PairTradingStrategy expects exactly two symbols")
        super().__init__(strategy_id=strategy_id, symbols=symbols)
        self._capital = float(capital)
        if not math.isfinite(self._capital) or self._capital <= 0:
            raise ValueError("capital must be a positive finite number")
        self._config = config or PairTradingConfig()
        self._primary_symbol = symbols[0]
        self._hedge_symbol = symbols[1]

    async def generate_signal(
        self, market_data: Mapping[str, Any], symbol: str
    ) -> StrategySignal:
        if symbol not in {self._primary_symbol, self._hedge_symbol}:
            return StrategySignal(
                symbol=symbol,
                action="FLAT",
                confidence=0.0,
                notional=0.0,
                metadata={"reason": "symbol_not_in_pair"},
            )

        try:
            primary_prices = _extract_prices(market_data, self._primary_symbol)
            hedge_prices = _extract_prices(market_data, self._hedge_symbol)
        except Exception as exc:  # pragma: no cover - defensive conversion guard
            return StrategySignal(
                symbol=symbol,
                action="FLAT",
                confidence=0.0,
                notional=0.0,
                metadata={"reason": "data_error", "error": str(exc)},
            )

        cfg = self._config
        lookback = cfg.lookback
        if (
            primary_prices.size < lookback
            or hedge_prices.size < lookback
        ):
            return StrategySignal(
                symbol=symbol,
                action="FLAT",
                confidence=0.0,
                notional=0.0,
                metadata={"reason": "insufficient_history", "required": lookback},
            )

        primary_window = primary_prices[-lookback:]
        hedge_window = hedge_prices[-lookback:]

        analysis = _analyse_pair(primary_window, hedge_window)
        if analysis is None:
            return StrategySignal(
                symbol=symbol,
                action="FLAT",
                confidence=0.0,
                notional=0.0,
                metadata={"reason": "degenerate_spread"},
            )

        (
            hedge_ratio,
            spread_mean,
            spread_std,
            zscore,
            adf_stat,
            half_life,
        ) = analysis

        metadata = {
            "pair": f"{self._primary_symbol}/{self._hedge_symbol}",
            "hedge_ratio": hedge_ratio,
            "spread_mean": spread_mean,
            "spread_std": spread_std,
            "zscore": zscore,
            "adf_stat": adf_stat,
            "half_life": half_life,
        }

        if spread_std < cfg.min_spread_std:
            metadata["reason"] = "spread_variance_too_low"
            return StrategySignal(symbol, "FLAT", 0.0, 0.0, metadata)

        if adf_stat > cfg.adf_stat_threshold:
            metadata["reason"] = "failed_cointegration_test"
            return StrategySignal(symbol, "FLAT", 0.0, 0.0, metadata)

        if not (cfg.min_half_life <= half_life <= cfg.max_half_life):
            metadata["reason"] = "half_life_out_of_bounds"
            return StrategySignal(symbol, "FLAT", 0.0, 0.0, metadata)

        if abs(zscore) < cfg.zscore_entry:
            metadata["reason"] = "zscore_below_entry"
            return StrategySignal(symbol, "FLAT", 0.0, 0.0, metadata)

        trade_direction = "LONG_SPREAD" if zscore <= -cfg.zscore_entry else "SHORT_SPREAD"
        if abs(zscore) < cfg.zscore_exit:
            trade_direction = "FLAT"

        intensity = min(abs(zscore) / cfg.zscore_entry, 2.0)
        confidence = float(min(intensity / 2.0, 1.0))

        base_allocation = self._capital * cfg.max_notional_fraction * min(intensity, 1.0)
        hedge_notional = abs(hedge_ratio) * base_allocation

        primary_action: str
        hedge_action: str
        primary_notional: float
        hedge_leg_notional: float

        if trade_direction == "LONG_SPREAD":
            primary_action = "BUY"
            hedge_action = "SELL"
            primary_notional = base_allocation
            hedge_leg_notional = -hedge_notional
        elif trade_direction == "SHORT_SPREAD":
            primary_action = "SELL"
            hedge_action = "BUY"
            primary_notional = -base_allocation
            hedge_leg_notional = hedge_notional
        else:
            metadata["reason"] = "within_exit_threshold"
            return StrategySignal(symbol, "FLAT", 0.0, 0.0, metadata)

        if symbol == self._primary_symbol:
            return StrategySignal(
                symbol=symbol,
                action=primary_action,
                confidence=confidence,
                notional=primary_notional,
                metadata=metadata,
            )
        else:
            return StrategySignal(
                symbol=symbol,
                action=hedge_action,
                confidence=confidence,
                notional=hedge_leg_notional,
                metadata=metadata,
            )


def _extract_prices(market_data: Mapping[str, Any], symbol: str) -> np.ndarray:
    payload = market_data.get(symbol)
    if payload is None:
        raise KeyError(f"market data missing symbol {symbol}")
    if isinstance(payload, Mapping):
        for key in ("close", "closing_prices", "prices"):
            if key in payload:
                payload = payload[key]
                break
    if isinstance(payload, np.ndarray):
        prices = payload.astype(float)
    else:
        prices = np.asarray(list(payload), dtype=float)
    if prices.ndim != 1:
        raise ValueError("price series must be one-dimensional")
    if np.isnan(prices).any():
        prices = prices[~np.isnan(prices)]
    return prices


def _analyse_pair(primary: np.ndarray, hedge: np.ndarray) -> tuple[float, float, float, float, float, float] | None:
    if primary.size != hedge.size:
        length = min(primary.size, hedge.size)
        primary = primary[-length:]
        hedge = hedge[-length:]
    if primary.size < 5:
        return None

    ones = np.ones_like(primary)
    x = np.column_stack([hedge, ones])
    try:
        beta, alpha = np.linalg.lstsq(x, primary, rcond=None)[0]
    except np.linalg.LinAlgError:
        return None

    spread = primary - (beta * hedge + alpha)
    if spread.size < 5:
        return None

    spread_mean = float(np.mean(spread))
    spread_std = float(np.std(spread, ddof=1))
    if not math.isfinite(spread_std) or spread_std <= 0:
        return None

    zscore = float((spread[-1] - spread_mean) / spread_std)
    adf_stat = _augmented_dickey_fuller(spread)
    half_life = _compute_half_life(spread)

    return float(beta), spread_mean, spread_std, zscore, adf_stat, half_life


def _augmented_dickey_fuller(series: np.ndarray) -> float:
    if series.size < 10:
        return float("inf")
    y = series[1:] - series[:-1]
    x = series[:-1]
    X = np.column_stack([np.ones_like(x), x])
    try:
        coeffs, residuals, _, _ = np.linalg.lstsq(X, y, rcond=None)
    except np.linalg.LinAlgError:
        return float("inf")
    n = y.size
    k = X.shape[1]
    if n <= k:
        return float("inf")
    if residuals.size == 0:
        sigma2 = float(np.sum((y - X @ coeffs) ** 2) / (n - k))
    else:
        sigma2 = float(residuals[0] / (n - k))
    XtX_inv = np.linalg.inv(X.T @ X)
    phi = coeffs[1]
    se_phi = math.sqrt(max(XtX_inv[1, 1] * sigma2, 1e-12))
    return float(phi / se_phi)


def _compute_half_life(series: np.ndarray) -> float:
    if series.size < 10:
        return float("inf")
    y = series[1:]
    x = series[:-1]
    X = np.column_stack([np.ones_like(x), x])
    try:
        coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    except np.linalg.LinAlgError:
        return float("inf")
    rho = float(coeffs[1])
    if not math.isfinite(rho):
        return float("inf")
    rho_abs = abs(rho)
    if rho_abs >= 1:
        return float("inf")
    if rho_abs <= 1e-6:
        return 0.0
    return float(-math.log(2.0) / math.log(rho_abs))
