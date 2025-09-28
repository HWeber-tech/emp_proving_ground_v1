from __future__ import annotations

from .mean_reversion import MeanReversionStrategy
from .moving_average import MovingAverageStrategy
from .trend_strategies import TrendFollowing

__all__ = [
    "MeanReversionStrategy",
    "MovingAverageStrategy",
    "TrendFollowing",
]
