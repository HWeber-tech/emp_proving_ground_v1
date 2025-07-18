"""
Trading package for EMP system.

This package contains:
- IC Markets cTrader OpenAPI integration (Mock for testing)
- Live trading execution
- Order management
- Position tracking
"""

from .mock_ctrader_interface import (
    CTraderInterface, TokenManager, TradingConfig,
    MarketData, Order, Position,
    TradingMode, OrderType, OrderSide
)
from .live_trading_executor import (
    LiveTradingExecutor, LiveRiskManager,
    TradingSignal, TradingPerformance
)

__all__ = [
    'CTraderInterface', 'TokenManager', 'TradingConfig',
    'MarketData', 'Order', 'Position',
    'TradingMode', 'OrderType', 'OrderSide',
    'LiveTradingExecutor', 'LiveRiskManager',
    'TradingSignal', 'TradingPerformance'
] 