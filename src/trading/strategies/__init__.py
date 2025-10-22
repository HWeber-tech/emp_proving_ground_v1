"""Trading strategy helpers and signal generators."""

from __future__ import annotations

from .analytics import (
    AttributionResult,
    FeatureContribution,
    compute_performance_attribution,
    result_to_dataframe,
)
from .mean_reversion import MeanReversionStrategy, MeanReversionStrategyConfig
from .models import StrategyAction, StrategySignal
from .momentum import MomentumStrategy, MomentumStrategyConfig
from .multi_timeframe_momentum import (
    MultiTimeframeMomentumConfig,
    MultiTimeframeMomentumStrategy,
    TimeframeMomentumLegConfig,
)
from .pairs import PairTradingConfig, PairTradingStrategy
from .signals import (
    GARCHCalibrationError,
    GARCHVolatilityConfig,
    GARCHVolatilityResult,
    compute_garch_volatility,
)
from .volatility_breakout import VolatilityBreakoutConfig, VolatilityBreakoutStrategy
from .volatility import VolatilityTradingConfig, VolatilityTradingStrategy

__all__ = [
    "AttributionResult",
    "FeatureContribution",
    "GARCHCalibrationError",
    "GARCHVolatilityConfig",
    "GARCHVolatilityResult",
    "MeanReversionStrategy",
    "MeanReversionStrategyConfig",
    "MultiTimeframeMomentumConfig",
    "MultiTimeframeMomentumStrategy",
    "TimeframeMomentumLegConfig",
    "MomentumStrategy",
    "MomentumStrategyConfig",
    "PairTradingConfig",
    "PairTradingStrategy",
    "StrategyAction",
    "StrategySignal",
    "VolatilityTradingConfig",
    "VolatilityTradingStrategy",
    "VolatilityBreakoutConfig",
    "VolatilityBreakoutStrategy",
    "compute_performance_attribution",
    "compute_garch_volatility",
    "result_to_dataframe",
]
