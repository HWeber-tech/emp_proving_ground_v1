"""
Moving Average Strategy Template
==============================

Simple moving average crossover strategy implementation.
Provides a complete example of how to implement trading strategies.
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import numpy as np

from src.trading.strategy_engine.base_strategy import BaseStrategy, StrategySignal, StrategyType, SignalType

logger = logging.getLogger(__name__)


class MovingAverageStrategy(BaseStrategy):
    """
    Simple moving average crossover strategy.
    
    This strategy uses two moving averages (fast and slow) to generate
    trading signals based on crossover events.
    """
    
    def __init__(self, strategy_id: str, symbols: List[str], parameters: Dict[str, Any]):
        """
        Initialize the moving average strategy.
        
        Args:
            strategy_id: Unique identifier for this strategy
            symbols: List of symbols to trade
            parameters: Strategy parameters including periods and thresholds
        """
        super().__init__(
            strategy_id=strategy_id,
            strategy_type=StrategyType.TREND_FOLLOWING,
            parameters=parameters,
            symbols=symbols
        )
        
        # Set default parameters
        self.parameters.setdefault('fast_period', 20)
        self.parameters.setdefault('slow_period', 50)
        self.parameters.setdefault('rsi_period', 14)
        self.parameters.setdefault('rsi_overbought', 70)
        self.parameters.setdefault('rsi_oversold', 30)
        self.parameters.setdefault('min_confidence', 0.6)
        self.parameters.setdefault('max_positions', 1)
        
        # Track previous signals for crossover detection
        self.previous_signals: Dict[str, Dict[str, Any]] = {}
        
        logger.info(f"MovingAverageStrategy {strategy_id} initialized with parameters: {self.parameters}")
    
    async def generate_signal(self, market_data: List[Dict[str, Any]], symbol: str) -> Optional[StrategySignal]:
        """
        Generate trading signal based on moving average crossover.
        
        Args:
            market_data: List of market data points
            symbol: Trading symbol
            
        Returns:
            Trading signal or None if no signal
        """
        try:
            if not market_data or len(market_data) < self.parameters['slow_period']:
                return None
            
            # Extract prices and volumes
            closes = [float(d.get('close', 0)) for d in market_data]
            volumes = [float(d.get('volume', 0)) for d in market_data]
            
            if len(closes) < max(self.parameters['fast_period'], self.parameters['slow_period']):
                return None
            
            # Calculate moving averages
            fast_ma = self._calculate_sma(closes, self.parameters['fast_period'])
            slow_ma = self._calculate_sma(closes, self.parameters['slow_period'])
            rsi = self._calculate_rsi(closes, self.parameters['rsi_period'])
            
            # Get current price
            current_price = closes[-1]
            
            # Generate signal based on crossover
            signal_type = self._determine_signal_type(
                fast_ma, slow_ma, rsi, current_price
            )
            
            if signal_type == SignalType.HOLD:
                return None
            
            # Calculate confidence
            confidence = self._calculate_confidence(
                fast_ma, slow_ma, rsi, current_price
            )
            
            if confidence < self.parameters['min_confidence']:
                return None
            
            # Calculate position size
            position_size = self.calculate_position_size(
                StrategySignal(
                    strategy_id=self.strategy_id,
                    signal_type=signal_type,
                    symbol=symbol,
                    timestamp=datetime.now(),
                    price=current_price,
                    quantity=0,  # Will be calculated
                    confidence=confidence
                ),
                10000.0  # Available capital
            )
            
            return StrategySignal(
                strategy_id=self.strategy_id,
                signal_type=signal_type,
                symbol=symbol,
                timestamp=datetime.now(),
                price=current_price,
                quantity=position_size,
                confidence=confidence,
                metadata={
                    'fast_ma': fast_ma,
                    'slow_ma': slow_ma,
                    'rsi': rsi,
                    'crossover': self._detect_crossover(closes)
                }
            )
            
        except Exception as e:
            logger.error(f"Error generating signal for {symbol}: {e}")
            return None
    
    def calculate_position_size(self, signal: StrategySignal, available_capital: float) -> float:
        """
        Calculate position size based on risk parameters.
        
        Args:
            signal: Trading signal
            available_capital: Available capital for trading
            
        Returns:
            Position size
        """
        try:
            # Risk 2% of capital per trade
            risk_amount = available_capital * 0.02
            
            # Calculate position size based on stop loss
            stop_loss_pct = 0.02  # 2% stop loss
            position_size = risk_amount / stop_loss_pct
            
            # Apply maximum position limit
            max_position = available_capital * 0.1  # 10% max position
            position_size = min(position_size, max_position)
            
            return position_size
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 1000.0
    
    def should_exit_position(self, position: Dict[str, Any], market_data: List[Dict[str, Any]]) -> bool:
        """
        Determine if position should be closed based on exit conditions.
        
        Args:
            position: Current position details
            market_data: Recent market data
            
        Returns:
            True if position should be closed
        """
        try:
            if not market_data or len(market_data) < self.parameters['fast_period']:
                return False
            
            # Extract prices
            closes = [float(d.get('close', 0)) for d in market_data]
            
            # Calculate moving averages
            fast_ma = self._calculate_sma(closes, self.parameters['fast_period'])
            slow_ma = self._calculate_sma(closes, self.parameters['slow_period'])
            
            # Exit on reverse crossover
            if position['quantity'] > 0:  # Long position
                return fast_ma < slow_ma
            else:  # Short position
                return fast_ma > slow_ma
                
        except Exception as e:
            logger.error(f"Error checking exit conditions: {e}")
            return False
    
    def _calculate_sma(self, prices: List[float], period: int) -> float:
        """
        Calculate simple moving average.
        
        Args:
            prices: List of prices
            period: Period for calculation
            
        Returns:
            Moving average value
        """
        if len(prices) < period:
            return prices[-1] if prices else 0.0
        
        return float(np.mean(prices[-period:]))
    
    def _calculate_ema(self, prices: List[float], period: int) -> float:
        """
        Calculate exponential moving average.
        
        Args:
            prices: List of prices
            period: Period for calculation
            
        Returns:
            Exponential moving average value
        """
        if len(prices) < period:
            return prices[-1] if prices else 0.0
        
        alpha = 2 / (period + 1)
        ema = prices[0]
        
        for price in prices[1:]:
            ema = alpha * price + (1 - alpha) * ema
        
        return float(ema)
    
    def _calculate_rsi(self, prices: List[float], period: int) -> float:
        """
        Calculate Relative Strength Index.
        
        Args:
            prices: List of prices
            period: Period for calculation
            
        Returns:
            RSI value between 0 and 100
        """
        if len(prices) < period + 1:
            return 50.0
        
        prices_array = np.array(prices)
        delta = np.diff(prices_array)
        
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.mean(gain[-period:])
        avg_loss = np.mean(loss[-period:])
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return float(rsi)
    
    def _determine_signal_type(self, fast_ma: float, slow_ma: float, rsi: float, current_price: float) -> SignalType:
        """
        Determine signal type based on indicators.
        
        Args:
            fast_ma: Fast moving average
            slow_ma: Slow moving average
            rsi: RSI value
            current_price: Current price
            
        Returns:
            Signal type
        """
        # Bullish crossover
        if fast_ma > slow_ma and rsi < self.parameters['rsi_oversold']:
            return SignalType.BUY
        
        # Bearish crossover
        elif fast_ma < slow_ma and rsi > self.parameters['rsi_overbought']:
            return SignalType.SELL
        
        return SignalType.HOLD
    
    def _calculate_confidence(self, fast_ma: float, slow_ma: float, rsi: float, current_price: float) -> float:
        """
        Calculate confidence level for the signal.
        
        Args:
            fast_ma: Fast moving average
            slow_ma: Slow moving average
            rsi: RSI value
            current_price: Current price
            
        Returns:
            Confidence level between 0 and 1
        """
        confidence = 0.5
        
        # Distance between MAs
        ma_distance = abs(fast_ma - slow_ma) / slow_ma
        confidence += min(ma_distance * 10, 0.3)
        
        # RSI extreme
        if rsi < self.parameters['rsi_oversold'] or rsi > self.parameters['rsi_overbought']:
            confidence += 0.2
        
        # Price position relative to MAs
        if current_price > fast_ma > slow_ma:
            confidence += 0.1
        elif current_price < fast_ma < slow_ma:
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _detect_crossover(self, prices: List[float]) -> bool:
        """
        Detect if a crossover has occurred recently.
        
        Args:
            prices: List of prices
            
        Returns:
            True if crossover detected
        """
        if len(prices) < self.parameters['slow_period'] + 2:
            return False
        
        # Calculate previous and current MAs
        prev_fast = self._calculate_sma(prices[:-1], self.parameters['fast_period'])
        prev_slow = self._calculate_sma(prices[:-1], self.parameters['slow_period'])
        curr_fast = self._calculate_sma(prices, self.parameters['fast_period'])
        curr_slow = self._calculate_sma(prices, self.parameters['slow_period'])
        
        # Detect crossover
        return (prev_fast <= prev_slow and curr_fast > curr_slow) or \
               (prev_fast >= prev_slow and curr_fast < curr_slow)
    
    def update_parameters(self, new_parameters: Dict[str, Any]) -> bool:
        """
        Update strategy parameters.
        
        Args:
            new_parameters: New parameter values
            
        Returns:
            True if update successful
        """
        try:
            # Validate parameters
            for key, value in new_parameters.items():
                if key in self.parameters:
                    if key.endswith('_period') and not isinstance(value, int):
                        return False
                    if key.endswith('_pct') and not isinstance(value, (int, float)):
                        return False
            
            # Update parameters
            self.parameters.update(new_parameters)
            logger.info(f"Strategy parameters updated: {new_parameters}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating parameters: {e}")
            return False
    
    def get_parameters(self) -> Dict[str, Any]:
        """
        Get current strategy parameters.
        
        Returns:
            Current parameters
        """
        return self.parameters.copy()
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """
        Get comprehensive strategy information.
        
        Returns:
            Strategy information
        """
        return {
            'strategy_id': self.strategy_id,
            'strategy_type': self.strategy_type.value,
            'symbols': self.symbols,
            'parameters': self.parameters,
            'status': self.status.value,
            'positions': len(self.positions),
            'signals': len(self.signals)
        }


# Factory function for easy creation
def create_moving_average_strategy(
    strategy_id: str,
    symbols: List[str],
    parameters: Dict[str, Any] = None
) -> MovingAverageStrategy:
    """
    Create a new MovingAverageStrategy instance.
    
    Args:
        strategy_id: Unique strategy identifier
        symbols: List of symbols to trade
        parameters: Strategy parameters
        
    Returns:
        Configured MovingAverageStrategy instance
    """
    if parameters is None:
        parameters = {}
    
    return MovingAverageStrategy(
        strategy_id=strategy_id,
        symbols=symbols,
        parameters=parameters
    )


if __name__ == "__main__":
    # Test the strategy
    print("Testing MovingAverageStrategy...")
    
    strategy = create_moving_average_strategy(
        strategy_id="test_ma",
        symbols=["EURUSD", "GBPUSD"],
        parameters={
            'fast_period': 10,
            'slow_period': 20,
            'rsi_period': 14
        }
    )
    
    # Test strategy info
    info = strategy.get_strategy_info()
    print(f"Strategy info: {info}")
    
    # Test parameter update
    success = strategy.update_parameters({'fast_period': 15})
    print(f"Parameter update: {success}")
    
    print("MovingAverageStrategy test completed successfully!")
