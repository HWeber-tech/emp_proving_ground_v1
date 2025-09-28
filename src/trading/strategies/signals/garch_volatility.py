"""Lightweight GARCH(1,1) volatility estimator for strategy signals.

The roadmap calls for a volatility toolkit that can be consumed by strategies
and risk modules without introducing heavyweight third-party dependencies. The
implementation below intentionally favours a deterministic grid-search
calibration over numerical optimisation libraries so it can run inside CI and
during dry runs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

__all__ = [
    "GARCHCalibrationError",
    "GARCHVolatilityConfig",
    "GARCHVolatilityResult",
    "compute_garch_volatility",
]

_EPS = 1e-12


class GARCHCalibrationError(RuntimeError):
    """Raised when the calibration loop fails to find a stable parameter set."""


@dataclass(frozen=True)
class GARCHVolatilityConfig:
    """Configuration for the lightweight GARCH volatility estimator."""

    lookback: int = 260
    trading_days_per_year: int = 252
    return_type: str = "log"  # or "simple"
    input_kind: str = "price"  # or "returns"
    candidate_alphas: Sequence[float] = (0.05, 0.075, 0.1, 0.125, 0.15)
    candidate_betas: Sequence[float] = (0.7, 0.75, 0.8, 0.85, 0.9)
    min_alpha: float = 0.01
    max_alpha: float = 0.25
    min_beta: float = 0.5
    max_beta: float = 0.98

    def validate(self) -> None:
        if self.lookback < 30:
            raise ValueError("lookback must be at least 30 observations")
        if self.trading_days_per_year <= 0:
            raise ValueError("trading_days_per_year must be positive")
        for alpha in self.candidate_alphas:
            if not (self.min_alpha <= alpha <= self.max_alpha):
                raise ValueError("candidate alpha out of bounds")
        for beta in self.candidate_betas:
            if not (self.min_beta <= beta <= self.max_beta):
                raise ValueError("candidate beta out of bounds")
        if self.return_type not in {"log", "simple"}:
            raise ValueError("return_type must be 'log' or 'simple'")
        if self.input_kind not in {"price", "returns"}:
            raise ValueError("input_kind must be 'price' or 'returns'")


@dataclass(frozen=True)
class GARCHVolatilityResult:
    """Container with conditional volatility outputs."""

    conditional_volatility: pd.Series
    annualised_volatility: pd.Series
    parameters: Mapping[str, float]
    log_likelihood: float
    last_update: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    @property
    def latest_sigma(self) -> float:
        if self.conditional_volatility.empty:
            raise ValueError("no volatility observations available")
        return float(self.conditional_volatility.iloc[-1])

    @property
    def latest_annualised_sigma(self) -> float:
        if self.annualised_volatility.empty:
            raise ValueError("no volatility observations available")
        return float(self.annualised_volatility.iloc[-1])


def compute_garch_volatility(
    series: pd.Series,
    *,
    config: GARCHVolatilityConfig | None = None,
) -> GARCHVolatilityResult:
    """Compute conditional volatility using a compact GARCH(1,1) model.

    Args:
        series: Price or return series indexed by timestamp.  When
            ``config.input_kind`` is ``"price"`` the series is converted to
            returns using ``config.return_type``; when it is ``"returns"`` the
            series is treated as pre-computed returns.
        config: Optional :class:`GARCHVolatilityConfig` overriding defaults.

    Returns:
        :class:`GARCHVolatilityResult` containing conditional and annualised
        volatility series with fitted parameters.
    """

    cfg = config or GARCHVolatilityConfig()
    cfg.validate()

    returns = _prepare_returns(series, cfg.return_type, cfg.input_kind)
    if returns.empty or len(returns) < cfg.lookback:
        raise ValueError(
            "Insufficient history for GARCH estimation; "
            f"have {len(returns)}, need >= {cfg.lookback}"
        )

    history = returns.iloc[-cfg.lookback :].copy()
    history -= history.mean()  # centre returns to remove drift

    variance = float(history.var(ddof=1))
    if variance <= 0:
        raise GARCHCalibrationError("Return variance must be positive for calibration")

    params, cond_var, ll = _calibrate_parameters(
        history.to_numpy(), variance, cfg.candidate_alphas, cfg.candidate_betas
    )

    sigma = pd.Series(np.sqrt(cond_var), index=history.index)
    annualised = sigma * np.sqrt(cfg.trading_days_per_year)

    return GARCHVolatilityResult(
        conditional_volatility=sigma,
        annualised_volatility=annualised,
        parameters=params,
        log_likelihood=ll,
    )


def _prepare_returns(series: pd.Series, return_type: str, input_kind: str) -> pd.Series:
    if not isinstance(series, pd.Series):
        raise TypeError("series must be a pandas Series")
    if series.empty:
        return series.astype(float)

    series = series.astype(float)
    if input_kind == "returns":
        returns = series.copy()
    else:
        if return_type == "log":
            returns = np.log(series / series.shift(1))
        elif return_type == "simple":
            returns = series.pct_change()
        else:  # pragma: no cover - guarded in validation
            raise ValueError(f"Unsupported return_type: {return_type}")
    returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
    returns.index = pd.DatetimeIndex(returns.index)
    return returns


def _calibrate_parameters(
    returns: np.ndarray,
    sample_variance: float,
    candidate_alphas: Iterable[float],
    candidate_betas: Iterable[float],
) -> tuple[dict[str, float], np.ndarray, float]:
    best_ll = -np.inf
    best_params: dict[str, float] | None = None
    best_conditional: np.ndarray | None = None

    for alpha in candidate_alphas:
        if alpha <= 0:
            continue
        for beta in candidate_betas:
            if beta <= 0 or alpha + beta >= 0.995:
                continue
            omega = max((1.0 - alpha - beta) * sample_variance, _EPS)
            cond_var, ll = _simulate_garch(returns, omega, alpha, beta)
            if not np.isfinite(ll):
                continue
            if ll > best_ll:
                best_ll = ll
                best_params = {
                    "omega": float(omega),
                    "alpha": float(alpha),
                    "beta": float(beta),
                }
                best_conditional = cond_var

    if best_params is None or best_conditional is None:
        raise GARCHCalibrationError("Failed to calibrate GARCH parameters")

    best_params["log_likelihood"] = float(best_ll)
    return best_params, best_conditional, float(best_ll)


def _simulate_garch(
    returns: np.ndarray, omega: float, alpha: float, beta: float
) -> tuple[np.ndarray, float]:
    cond_var = np.zeros_like(returns, dtype=float)
    cond_var[0] = max(np.var(returns, ddof=1), _EPS)
    ll = 0.0
    for idx in range(1, len(returns)):
        cond_var[idx] = max(
            omega + alpha * returns[idx - 1] ** 2 + beta * cond_var[idx - 1], _EPS
        )
    # log-likelihood under conditional normality assumption
    ll = -0.5 * np.sum(np.log(cond_var + _EPS) + returns**2 / (cond_var + _EPS))
    return cond_var, float(ll)
