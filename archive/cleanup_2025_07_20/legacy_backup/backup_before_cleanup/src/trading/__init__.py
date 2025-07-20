"""
Trading package for EMP system.

This package contains:
- IC Markets cTrader OpenAPI integration (Real and Mock)
- Live trading execution
- Order management
- Position tracking
"""

# Import real interface (primary)
try:
    from .real_ctrader_interface import (
        RealCTraderInterface,
        TradingConfig as RealTradingConfig,
        OrderType as RealOrderType,
        OrderSide as RealOrderSide,
        OrderStatus as RealOrderStatus,
        MarketData as RealMarketData,
        Order as RealOrder,
        Position as RealPosition,
        create_demo_config,
        create_live_config
    )
    REAL_CTRADER_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Real cTrader interface not available: {e}")
    REAL_CTRADER_AVAILABLE = False

# Import mock interface (fallback)
from .mock_ctrader_interface import (
    CTraderInterface, TokenManager, TradingConfig,
    MarketData, Order, Position,
    TradingMode, OrderType, OrderSide
)
from .live_trading_executor import (
    LiveTradingExecutor, LiveRiskManager,
    TradingSignal, TradingPerformance
)

# Export real interface if available, otherwise mock
if REAL_CTRADER_AVAILABLE:
    __all__ = [
        'RealCTraderInterface',
        'TradingConfig',
        'OrderType',
        'OrderSide',
        'OrderStatus',
        'MarketData',
        'Order',
        'Position',
        'create_demo_config',
        'create_live_config',
        'CTraderInterface',  # Keep mock for testing
        'TokenManager',
        'TradingMode',
        'LiveTradingExecutor',
        'LiveRiskManager',
        'TradingSignal',
        'TradingPerformance'
    ]
else:
    # Fallback to mock interface
    __all__ = [
        'CTraderInterface', 'TokenManager', 'TradingConfig',
        'MarketData', 'Order', 'Position',
        'TradingMode', 'OrderType', 'OrderSide',
        'LiveTradingExecutor', 'LiveRiskManager',
        'TradingSignal', 'TradingPerformance'
    ] 