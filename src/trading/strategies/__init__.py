"""Trading strategy helpers and signal generators."""

from __future__ import annotations

from .donchian_atr_breakout import (
    DonchianATRBreakoutConfig,
    DonchianATRBreakoutStrategy,
)
from .mean_reversion import MeanReversionStrategy, MeanReversionStrategyConfig
from .models import StrategyAction, StrategySignal
from .momentum import MomentumStrategy, MomentumStrategyConfig
from .multi_timeframe_momentum import (
    MultiTimeframeMomentumConfig,
    MultiTimeframeMomentumStrategy,
)
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
    "DonchianATRBreakoutConfig",
    "DonchianATRBreakoutStrategy",
    "MeanReversionStrategy",
    "MeanReversionStrategyConfig",
    "MomentumStrategy",
    "MomentumStrategyConfig",
    "MultiTimeframeMomentumConfig",
    "MultiTimeframeMomentumStrategy",
    "PairTradingConfig",
    "PairTradingStrategy",
    "StrategyAction",
    "StrategySignal",
    "VolatilityBreakoutConfig",
    "VolatilityBreakoutStrategy",
    "compute_garch_volatility",
]
