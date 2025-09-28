from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from src.trading.strategies.signals import (
    GARCHVolatilityConfig,
    GARCHCalibrationError,
    compute_garch_volatility,
)


def _build_price_series(low_sigma: float, high_sigma: float, *, seed: int = 7) -> pd.Series:
    rng = np.random.default_rng(seed)
    low = rng.normal(0.0, low_sigma, size=180)
    high = rng.normal(0.0, high_sigma, size=180)
    returns = np.concatenate([low, high])
    prices = 100.0 * np.exp(np.cumsum(returns))
    start = datetime(2023, 1, 1)
    index = pd.date_range(start, periods=returns.size, freq="D")
    return pd.Series(prices, index=index)


def test_garch_detects_volatility_regime_shift() -> None:
    series = _build_price_series(0.003, 0.018)
    cfg = GARCHVolatilityConfig(lookback=300)
    result = compute_garch_volatility(series, config=cfg)

    assert result.conditional_volatility.index.is_monotonic_increasing
    early = result.annualised_volatility.iloc[:150].mean()
    late = result.annualised_volatility.iloc[-60:].mean()

    assert late > early * 1.8
    params = result.parameters
    assert params["alpha"] + params["beta"] < 0.995
    assert result.log_likelihood == pytest.approx(params["log_likelihood"])


def test_garch_accepts_return_series() -> None:
    rng = np.random.default_rng(11)
    returns = pd.Series(
        rng.normal(0.0, 0.01, size=320),
        index=pd.date_range(datetime(2022, 1, 1), periods=320, freq="D"),
    )
    cfg = GARCHVolatilityConfig(
        lookback=300,
        input_kind="returns",
        return_type="log",
    )
    result = compute_garch_volatility(returns, config=cfg)
    assert result.conditional_volatility.iloc[-1] > 0
    assert result.annualised_volatility.iloc[-1] > 0


def test_calibration_failure_raises() -> None:
    series = pd.Series(
        100.0,
        index=pd.date_range(datetime(2024, 1, 1), periods=400, freq="D"),
    )
    cfg = GARCHVolatilityConfig(lookback=300)
    with pytest.raises(GARCHCalibrationError):
        compute_garch_volatility(series, config=cfg)

    noisy_returns = pd.Series(
        np.zeros(320),
        index=pd.date_range(datetime(2022, 1, 1), periods=320, freq="D"),
    )
    cfg_returns = GARCHVolatilityConfig(lookback=300, input_kind="returns")
    with pytest.raises(GARCHCalibrationError):
        compute_garch_volatility(noisy_returns, config=cfg_returns)
