"""PCA-based statistical arbitrage strategy implementation.

This module satisfies the roadmap requirement for *Strategy 4: Statistical
Arbitrage* by building a market-neutral portfolio using principal component
analysis.  The workflow is:

1. Convert recent closing prices into demeaned log returns.
2. Extract the dominant systematic factors with PCA.
3. Treat the residual component as a mean-reverting signal.
4. Allocate a multi-asset long/short book based on residual z-scores while
   respecting a configurable gross exposure limit.

The design purposefully avoids external dependencies beyond NumPy so that the
strategy can operate inside our existing lightweight backtesting harness.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import math
import numpy as np

from src.core.strategy.engine import BaseStrategy

from ..models import StrategySignal

__all__ = ["PCAStatArbConfig", "PCAStatArbStrategy"]


_EPSILON = 1e-12
_DEFAULT_KEYS = ("close", "closing_prices", "prices", "closes")


@dataclass(slots=True)
class PCAStatArbConfig:
    """Configuration for the PCA statistical arbitrage strategy."""

    lookback: int = 120
    n_components: int = 3
    entry_zscore: float = 1.5
    exit_zscore: float = 0.5
    max_gross_exposure: float = 1.5
    residual_std_floor: float = 1e-4

    def __post_init__(self) -> None:
        if self.lookback < 20:
            raise ValueError("lookback must be at least 20 observations")
        if self.n_components < 1:
            raise ValueError("n_components must be positive")
        if self.entry_zscore <= 0:
            raise ValueError("entry_zscore must be positive")
        if self.exit_zscore < 0:
            raise ValueError("exit_zscore must be non-negative")
        if self.exit_zscore >= self.entry_zscore:
            raise ValueError("exit_zscore must be strictly below entry_zscore")
        if self.max_gross_exposure <= 0:
            raise ValueError("max_gross_exposure must be positive")
        if self.residual_std_floor <= 0:
            raise ValueError("residual_std_floor must be positive")


class PCAStatArbStrategy(BaseStrategy):
    """Market-neutral PCA residual strategy across a basket of instruments."""

    def __init__(
        self,
        strategy_id: str,
        symbols: list[str],
        *,
        capital: float,
        config: PCAStatArbConfig | None = None,
    ) -> None:
        if len(symbols) < 3:
            raise ValueError("PCAStatArbStrategy expects at least three symbols")
        super().__init__(strategy_id=strategy_id, symbols=symbols)
        self._capital = float(capital)
        if not math.isfinite(self._capital) or self._capital <= 0:
            raise ValueError("capital must be a positive finite number")
        self._config = config or PCAStatArbConfig()

    async def generate_signal(
        self, market_data: Mapping[str, Any], symbol: str
    ) -> StrategySignal:
        if symbol not in self.symbols:
            return StrategySignal(
                symbol=symbol,
                action="FLAT",
                confidence=0.0,
                notional=0.0,
                metadata={"reason": "symbol_not_managed"},
            )

        try:
            signals = _construct_portfolio(
                market_data,
                symbols=self.symbols,
                capital=self._capital,
                config=self._config,
            )
        except ValueError as exc:
            message = str(exc)
            reason = (
                message
                if message.startswith("insufficient_history")
                else "insufficient_history"
            )
            return StrategySignal(
                symbol=symbol,
                action="FLAT",
                confidence=0.0,
                notional=0.0,
                metadata={"reason": reason, "details": message},
            )
        except Exception as exc:  # pragma: no cover - defensive fallback
            return StrategySignal(
                symbol=symbol,
                action="FLAT",
                confidence=0.0,
                notional=0.0,
                metadata={"reason": "data_error", "error": str(exc)},
            )

        action, confidence, notional, metadata = signals.get(
            symbol,
            ("FLAT", 0.0, 0.0, {"reason": "symbol_missing_from_portfolio"}),
        )
        return StrategySignal(
            symbol=symbol,
            action=action,
            confidence=confidence,
            notional=notional,
            metadata=metadata,
        )


def _construct_portfolio(
    market_data: Mapping[str, Any],
    *,
    symbols: list[str],
    capital: float,
    config: PCAStatArbConfig,
) -> dict[str, tuple[str, float, float, dict[str, Any]]]:
    price_matrix, ordered_symbols = _extract_price_matrix(
        market_data,
        symbols=symbols,
        required_length=config.lookback + 1,
    )

    returns = np.diff(np.log(price_matrix), axis=0)
    returns = returns[-config.lookback :, :]
    returns -= np.mean(returns, axis=0, keepdims=True)

    (residuals, latest_residual, residual_std, components_used, variance_ratio) = (
        _pca_residuals(returns, n_components=config.n_components)
    )

    zscores = latest_residual / np.maximum(residual_std, config.residual_std_floor)
    active_mask = np.abs(zscores) >= config.entry_zscore

    base_metadata: dict[str, Any] = {
        "lookback": float(config.lookback),
        "components": float(components_used),
        "variance_explained": float(variance_ratio),
    }

    signals: dict[str, tuple[str, float, float, dict[str, Any]]] = {}

    active_count = int(np.count_nonzero(active_mask))
    if active_count < 2:
        reason = "no_active_assets" if active_count == 0 else "insufficient_active_assets"
        for idx, sym in enumerate(ordered_symbols):
            metadata = {
                **base_metadata,
                "zscore": float(zscores[idx]),
                "residual": float(latest_residual[idx]),
                "residual_std": float(residual_std[idx]),
                "weight": 0.0,
                "reason": reason,
                "active_assets": float(active_count),
            }
            signals[sym] = ("FLAT", 0.0, 0.0, metadata)
        return signals

    weights = np.zeros_like(zscores)
    weights[active_mask] = -zscores[active_mask]

    # Enforce market neutrality by demeaning the active weights.
    active_weights = weights[active_mask]
    weights[active_mask] -= np.mean(active_weights)

    gross = float(np.sum(np.abs(weights)))
    if gross <= _EPSILON:
        for idx, sym in enumerate(ordered_symbols):
            metadata = {
                **base_metadata,
                "zscore": float(zscores[idx]),
                "residual": float(latest_residual[idx]),
                "residual_std": float(residual_std[idx]),
                "weight": 0.0,
                "reason": "neutralised_weights",
                "active_assets": float(active_count),
            }
            signals[sym] = ("FLAT", 0.0, 0.0, metadata)
        return signals

    weights /= gross
    gross_target = float(capital * config.max_gross_exposure)
    notionals = weights * gross_target

    for idx, sym in enumerate(ordered_symbols):
        metadata = {
            **base_metadata,
            "zscore": float(zscores[idx]),
            "residual": float(latest_residual[idx]),
            "residual_std": float(residual_std[idx]),
            "weight": float(weights[idx]),
            "gross_target": gross_target,
            "active_assets": float(active_count),
        }

        if not active_mask[idx]:
            metadata["reason"] = "zscore_below_entry"
            signals[sym] = ("FLAT", 0.0, 0.0, metadata)
            continue

        if abs(zscores[idx]) <= config.exit_zscore:
            metadata["reason"] = "within_exit_threshold"
            signals[sym] = ("FLAT", 0.0, 0.0, metadata)
            continue

        notional = float(notionals[idx])
        if abs(notional) <= _EPSILON:
            metadata["reason"] = "zero_notional"
            signals[sym] = ("FLAT", 0.0, 0.0, metadata)
            continue

        intensity = min(abs(zscores[idx]) / config.entry_zscore, 2.0)
        confidence = float(min(intensity / 2.0, 1.0))
        action = "BUY" if notional > 0 else "SELL"

        signals[sym] = (action, confidence, notional, metadata)

    return signals


def _extract_price_matrix(
    market_data: Mapping[str, Any],
    *,
    symbols: list[str],
    required_length: int,
) -> tuple[np.ndarray, list[str]]:
    series: list[np.ndarray] = []
    ordered_symbols: list[str] = []

    for symbol in symbols:
        payload = market_data.get(symbol)
        if payload is None:
            raise KeyError(f"market data missing symbol {symbol}")

        prices = _extract_series(payload)
        prices = prices[np.isfinite(prices)]
        if prices.size < required_length:
            raise ValueError(
                f"insufficient_history:{symbol}:{prices.size}:{required_length}"
            )

        series.append(prices[-required_length:])
        ordered_symbols.append(symbol)

    matrix = np.column_stack(series)
    return matrix, ordered_symbols


def _extract_series(payload: Any) -> np.ndarray:
    if isinstance(payload, Mapping):
        for key in _DEFAULT_KEYS:
            if key in payload:
                return _as_array(payload[key])
        return _as_array(payload.values())
    return _as_array(payload)


def _as_array(values: Any) -> np.ndarray:
    if isinstance(values, np.ndarray):
        arr = values.astype(float, copy=False)
    else:
        arr = np.asarray(list(values), dtype=float)
    if arr.ndim != 1:
        raise ValueError("price series must be one-dimensional")
    return arr


def _pca_residuals(
    returns: np.ndarray,
    *,
    n_components: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, float]:
    if returns.ndim != 2:
        raise ValueError("returns matrix must be two-dimensional")
    if returns.shape[0] < 2:
        raise ValueError("returns matrix requires at least two observations")

    cov = np.cov(returns, rowvar=False)
    if not np.all(np.isfinite(cov)):
        raise ValueError("covariance matrix contains non-finite values")

    try:
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
    except np.linalg.LinAlgError as exc:  # pragma: no cover - numerical guard
        raise ValueError("failed to compute eigen decomposition") from exc

    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    total_variance = float(np.sum(np.maximum(eigenvalues, 0.0)))
    usable_components = int(
        min(
            n_components,
            returns.shape[1] - 1,
            np.count_nonzero(eigenvalues > _EPSILON),
        )
    )

    if usable_components <= 0 or total_variance <= _EPSILON:
        # Fallback: treat each asset's demeaned return as residual.
        residuals = returns - np.mean(returns, axis=1, keepdims=True)
        latest_residual = residuals[-1]
        residual_std = np.std(
            residuals, axis=0, ddof=1 if residuals.shape[0] > 1 else 0
        )
        variance_ratio = 0.0
        return residuals, latest_residual, residual_std, 0, variance_ratio

    basis = eigenvectors[:, :usable_components]
    factor_scores = returns @ basis
    reconstructed = factor_scores @ basis.T
    residuals = returns - reconstructed
    latest_residual = residuals[-1]
    residual_std = np.std(
        residuals, axis=0, ddof=1 if residuals.shape[0] > 1 else 0
    )

    explained = float(np.sum(eigenvalues[:usable_components]))
    variance_ratio = explained / total_variance if total_variance > 0 else 0.0

    return residuals, latest_residual, residual_std, usable_components, variance_ratio
