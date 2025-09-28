"""Trading strategy helpers and signal generators."""

from __future__ import annotations

from .mean_reversion import MeanReversionStrategy, MeanReversionStrategyConfig
from .models import StrategyAction, StrategySignal
from .momentum import MomentumStrategy, MomentumStrategyConfig
from .pairs import PairTradingConfig, PairTradingStrategy
from .signals import (
    GARCHCalibrationError,
    GARCHVolatilityConfig,
    GARCHVolatilityResult,
    compute_garch_volatility,
)
from .volatility_breakout import VolatilityBreakoutConfig, VolatilityBreakoutStrategy

__all__ = [
    "GARCHCalibrationError",
    "GARCHVolatilityConfig",
    "GARCHVolatilityResult",
    "MeanReversionStrategy",
    "MeanReversionStrategyConfig",
    "MomentumStrategy",
    "MomentumStrategyConfig",
    "PairTradingConfig",
    "PairTradingStrategy",
    "StrategyAction",
    "StrategySignal",
    "VolatilityBreakoutConfig",
    "VolatilityBreakoutStrategy",
    "compute_garch_volatility",
]
