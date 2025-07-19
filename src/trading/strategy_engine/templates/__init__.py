"""
Strategy Templates Package

Pre-built strategy templates for common trading strategies.
"""

from .trend_following import TrendFollowingStrategy
from .mean_reversion import MeanReversionStrategy
from .momentum import MomentumStrategy

__all__ = [
    'TrendFollowingStrategy',
    'MeanReversionStrategy',
    'MomentumStrategy'
] 