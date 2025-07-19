"""
Base Strategy Class

Abstract base class for all trading strategies with common functionality.

Author: EMP Development Team
Date: July 18, 2024
Phase: 3 - Advanced Trading Strategies and Risk Management
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum

from src.sensory.core.base import MarketData

logger = logging.getLogger(__name__)


class StrategyType(Enum):
    """Types of trading strategies"""
    TREND_FOLLOWING = "trend_following"
    MEAN_REVERSION = "mean_reversion"
    MOMENTUM = "momentum"
    ARBITRAGE = "arbitrage"
    STATISTICAL_ARBITRAGE = "statistical_arbitrage"
    BREAKOUT = "breakout"
    SCALPING = "scalping"
    SWING_TRADING = "swing_trading"
    CUSTOM = "custom"


class StrategyStatus(Enum):
    """Strategy execution status"""
    INACTIVE = "inactive"
    ACTIVE = "active"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"


class SignalType(Enum):
    """Trading signal types"""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    CLOSE = "close"


@dataclass
class StrategySignal:
    """Trading signal from strategy"""
    strategy_id: str
    signal_type: SignalType
    symbol: str
    timestamp: datetime
    price: float
    quantity: float
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StrategyPerformance:
    """Strategy performance metrics"""
    total_return: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    average_win: float = 0.0
    average_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    consecutive_wins: int = 0
    consecutive_losses: int = 0
    calmar_ratio: float = 0.0
    sortino_ratio: float = 0.0


class BaseStrategy(ABC):
    """
    Base class for all trading strategies.
    All strategies must inherit from this class and implement required methods.
    """
    
    def __init__(self, strategy_id: str, strategy_type: StrategyType, 
                 parameters: Dict[str, Any], symbols: List[str]):
        self.strategy_id = strategy_id
        self.strategy_type = strategy_type
        self.parameters = parameters
        self.symbols = symbols
        self.status = StrategyStatus.INACTIVE
        self.performance = StrategyPerformance()
        
        # Strategy state
        self.positions: Dict[str, float] = {}
        self.signals: List[StrategySignal] = []
        self.market_data: Dict[str, List[MarketData]] = {}
        self.last_update = datetime.utcnow()
        
        # Performance tracking
        self.trades: List[Dict[str, Any]] = []
        self.equity_curve: List[float] = []
        self.drawdown_curve: List[float] = []
        
        logger.info(f"Strategy {strategy_id} initialized: {strategy_type.value}")
    
    @abstractmethod
    async def generate_signal(self, market_data: List[MarketData], 
                            symbol: str) -> Optional[StrategySignal]:
        """Generate trading signal based on market data"""
        pass
    
    @abstractmethod
    def calculate_position_size(self, signal: StrategySignal, 
                              available_capital: float) -> float:
        """Calculate position size for the signal"""
        pass
    
    @abstractmethod
    def should_exit_position(self, position: Dict[str, Any], 
                           market_data: List[MarketData]) -> bool:
        """Determine if position should be closed"""
        pass
    
    async def update(self, market_data: List[MarketData], symbol: str) -> None:
        """Update strategy with new market data"""
        if self.status != StrategyStatus.ACTIVE:
            return
        
        # Store market data
        if symbol not in self.market_data:
            self.market_data[symbol] = []
        self.market_data[symbol].extend(market_data)
        
        # Keep only recent data
        lookback_period = self.parameters.get('lookback_period', 100)
        max_data_points = max(lookback_period * 2, 1000)
        if len(self.market_data[symbol]) > max_data_points:
            self.market_data[symbol] = self.market_data[symbol][-max_data_points:]
        
        # Generate signal
        signal = await self.generate_signal(market_data, symbol)
        if signal:
            self.signals.append(signal)
            logger.info(f"Strategy {self.strategy_id} generated {signal.signal_type.value} signal for {symbol}")
        
        # Check existing positions
        await self._check_positions(market_data, symbol)
        
        self.last_update = datetime.utcnow()
    
    async def _check_positions(self, market_data: List[MarketData], symbol: str) -> None:
        """Check existing positions for exit conditions"""
        if symbol in self.positions:
            position = {
                'symbol': symbol,
                'quantity': self.positions[symbol],
                'entry_price': self._get_entry_price(symbol),
                'entry_time': self._get_entry_time(symbol)
            }
            
            if self.should_exit_position(position, market_data):
                exit_signal = StrategySignal(
                    strategy_id=self.strategy_id,
                    signal_type=SignalType.CLOSE,
                    symbol=symbol,
                    timestamp=datetime.utcnow(),
                    price=market_data[-1].close if market_data else 0.0,
                    quantity=abs(self.positions[symbol]),
                    confidence=1.0
                )
                self.signals.append(exit_signal)
                logger.info(f"Strategy {self.strategy_id} closing position for {symbol}")
    
    def _get_entry_price(self, symbol: str) -> float:
        """Get entry price for position"""
        # This would be implemented based on trade history
        return 0.0
    
    def _get_entry_time(self, symbol: str) -> datetime:
        """Get entry time for position"""
        # This would be implemented based on trade history
        return datetime.utcnow()
    
    def start(self) -> None:
        """Start strategy execution"""
        self.status = StrategyStatus.ACTIVE
        logger.info(f"Strategy {self.strategy_id} started")
    
    def stop(self) -> None:
        """Stop strategy execution"""
        self.status = StrategyStatus.STOPPED
        logger.info(f"Strategy {self.strategy_id} stopped")
    
    def pause(self) -> None:
        """Pause strategy execution"""
        self.status = StrategyStatus.PAUSED
        logger.info(f"Strategy {self.strategy_id} paused")
    
    def get_performance(self) -> StrategyPerformance:
        """Get current performance metrics"""
        return self.performance
    
    def update_performance(self, trade_result: Dict[str, Any]) -> None:
        """Update performance metrics with trade result"""
        # This would implement performance calculation logic
        pass 