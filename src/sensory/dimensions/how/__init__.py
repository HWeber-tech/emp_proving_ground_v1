"""
How Sense - Technical Analysis and Market Mechanics

This sense handles all technical analysis, indicators, and market mechanics.
It processes market data to understand HOW the market is moving.

Sub-modules:
- indicators: Technical indicators (RSI, MACD, Bollinger Bands, etc.)
- patterns: Chart patterns and formations  
- signals: Buy/sell signal generation
- momentum: Momentum analysis and strength indicators
- volatility: Volatility analysis and measures

Author: EMP Development Team
Date: July 18, 2024
Phase: 2 - Sensory Cortex Refactoring
"""

# Import the main engine
from .how_engine import HowEngine

# Import sub-modules (will be created as needed)
# from . import indicators
# from . import patterns
# from . import signals
# from . import momentum
# from . import volatility

__all__ = [
    'HowEngine'
] 