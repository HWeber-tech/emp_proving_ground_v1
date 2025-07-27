"""
Trend Following Strategy Template

Specialized trend following strategy implementation using sensory layer indicators.

Author: EMP Development Team
Date: July 18, 2024
Phase: 3 - Advanced Trading Strategies and Risk Management
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

from src.sensory.core.base import MarketData
from src.sensory.dimensions.indicators import TechnicalIndicators
from src.trading.strategy_engine.base_strategy import BaseStrategy, StrategyType, StrategySignal, SignalType

logger = logging.getLogger(__name__)


class TrendFollowingStrategy(BaseStrategy):
    """
    Trend Following Strategy Template
    
    Implements trend following using sensory layer indicators:
    - Moving Average Crossovers (from sensory layer)
    - Momentum Indicators (RSI, MACD from sensory layer)
    - Trend Strength Analysis
    - Volume Confirmation
    """
    
    def __init__(self, strategy_id: str, parameters: Dict[str, Any], symbols: List[str]):
        super().__init__(strategy_id, StrategyType.TREND_FOLLOWING, parameters, symbols)
        
        # Strategy-specific parameters
        self.short_ma_period = parameters.get('short_ma_period', 10)
        self.long_ma_period = parameters.get('long_ma_period', 20)
        self.rsi_period = parameters.get('rsi_period', 14)
        self.volume_threshold = parameters.get('volume_threshold', 1.5)
        
        # Initialize sensory layer indicators
        self.indicators = TechnicalIndicators()
        
        logger.info(f"TrendFollowingStrategy {strategy_id} initialized")
    
    async def generate_signal(self, market_data: List[MarketData], symbol: str) -> Optional[StrategySignal]:
        """Generate trend following signal using sensory layer indicators"""
        if len(market_data) < self.long_ma_period:
            return None
        
        # Convert market data to DataFrame for sensory layer
        df = self._market_data_to_dataframe(market_data)
        
        # Get indicators from sensory layer
        indicators = self.indicators.calculate_indicators(df, ['sma', 'rsi', 'macd'])
        
        # Extract indicator values
        short_ma = indicators.get(f'sma_{self.short_ma_period}', None)
        long_ma = indicators.get(f'sma_{self.long_ma_period}', None)
        rsi = indicators.get('rsi', None)
        macd_line = indicators.get('macd', None)
        macd_signal = indicators.get('macd_signal', None)
        
        if not all([short_ma is not None, long_ma is not None, rsi is not None, 
                   macd_line is not None, macd_signal is not None]):
            return None
        
        # Get current values
        current_short_ma = short_ma.iloc[-1] if short_ma is not None else 0.0
        current_long_ma = long_ma.iloc[-1] if long_ma is not None else 0.0
        current_rsi = rsi.iloc[-1] if rsi is not None else 50.0
        current_macd_line = macd_line.iloc[-1] if macd_line is not None else 0.0
        current_macd_signal = macd_signal.iloc[-1] if macd_signal is not None else 0.0
        
        # Volume analysis
        volume_ratio = self._analyze_volume(market_data)
        
        # Trend strength
        trend_strength = self._calculate_trend_strength(short_ma, long_ma)
        
        current_price = market_data[-1].close
        
        # Generate signal based on multiple confirmations
        signal = self._evaluate_signals(
            current_price, current_short_ma, current_long_ma, current_rsi, 
            current_macd_line, current_macd_signal, volume_ratio, trend_strength
        )
        
        return signal
    
    def _market_data_to_dataframe(self, market_data: List[MarketData]):
        """Convert market data to DataFrame for sensory layer"""
        import pandas as pd
        
        data = {
            'open': [md.open for md in market_data],
            'high': [md.high for md in market_data],
            'low': [md.low for md in market_data],
            'close': [md.close for md in market_data],
            'volume': [md.volume for md in market_data]
        }
        
        return pd.DataFrame(data)
    
    def _analyze_volume(self, market_data: List[MarketData]) -> float:
        """Analyze volume relative to average"""
        if len(market_data) < 20:
            return 1.0
        
        volumes = [md.volume for md in market_data]
        avg_volume = sum(volumes[-20:]) / 20
        current_volume = volumes[-1]
        
        return current_volume / avg_volume if avg_volume > 0 else 1.0
    
    def _calculate_trend_strength(self, short_ma, long_ma) -> float:
        """Calculate trend strength using moving averages"""
        if len(short_ma) < 10 or len(long_ma) < 10:
            return 0.0
        
        # Calculate slope of moving averages
        short_slope = (short_ma.iloc[-1] - short_ma.iloc[-10]) / short_ma.iloc[-10]
        long_slope = (long_ma.iloc[-1] - long_ma.iloc[-10]) / long_ma.iloc[-10]
        
        # Trend strength is average of both slopes
        trend_strength = (short_slope + long_slope) / 2
        
        return trend_strength
    
    def _evaluate_signals(self, current_price: float, short_ma: float, long_ma: float,
                         rsi: float, macd_line: float, macd_signal: float,
                         volume_ratio: float, trend_strength: float) -> Optional[StrategySignal]:
        """Evaluate multiple signals to generate trading decision"""
        
        # Buy conditions
        buy_signals = 0
        total_signals = 0
        
        # Moving average crossover
        if short_ma > long_ma:
            buy_signals += 1
        total_signals += 1
        
        # RSI not overbought
        if rsi < 70:
            buy_signals += 1
        total_signals += 1
        
        # MACD bullish
        if macd_line > macd_signal:
            buy_signals += 1
        total_signals += 1
        
        # Volume confirmation
        if volume_ratio > self.volume_threshold:
            buy_signals += 1
        total_signals += 1
        
        # Trend strength
        if trend_strength > 0.001:
            buy_signals += 1
        total_signals += 1
        
        # Sell conditions
        sell_signals = 0
        
        # Moving average crossover
        if short_ma < long_ma:
            sell_signals += 1
        
        # RSI not oversold
        if rsi > 30:
            sell_signals += 1
        
        # MACD bearish
        if macd_line < macd_signal:
            sell_signals += 1
        
        # Generate signal based on signal strength
        buy_confidence = buy_signals / total_signals if total_signals > 0 else 0
        sell_confidence = sell_signals / 3 if sell_signals > 0 else 0
        
        if buy_confidence > 0.6:  # 60% of signals are bullish
            return StrategySignal(
                strategy_id=self.strategy_id,
                signal_type=SignalType.BUY,
                symbol=self.symbols[0],
                timestamp=datetime.utcnow(),
                price=current_price,
                quantity=1.0,
                confidence=buy_confidence,
                metadata={
                    'short_ma': short_ma,
                    'long_ma': long_ma,
                    'rsi': rsi,
                    'macd_line': macd_line,
                    'macd_signal': macd_signal,
                    'volume_ratio': volume_ratio,
                    'trend_strength': trend_strength
                }
            )
        elif sell_confidence > 0.6:  # 60% of signals are bearish
            return StrategySignal(
                strategy_id=self.strategy_id,
                signal_type=SignalType.SELL,
                symbol=self.symbols[0],
                timestamp=datetime.utcnow(),
                price=current_price,
                quantity=1.0,
                confidence=sell_confidence,
                metadata={
                    'short_ma': short_ma,
                    'long_ma': long_ma,
                    'rsi': rsi,
                    'macd_line': macd_line,
                    'macd_signal': macd_signal,
                    'volume_ratio': volume_ratio,
                    'trend_strength': trend_strength
                }
            )
        
        return None
    
    def calculate_position_size(self, signal: StrategySignal, available_capital: float) -> float:
        """Calculate position size based on trend strength and confidence"""
        base_size = available_capital * self.parameters.get('risk_per_trade', 0.02)
        
        # Adjust based on signal confidence
        confidence_multiplier = signal.confidence
        
        # Adjust based on trend strength
        trend_strength = signal.metadata.get('trend_strength', 0.0)
        strength_multiplier = min(abs(trend_strength) * 100, 2.0)
        
        position_size = base_size * confidence_multiplier * strength_multiplier
        
        # Apply maximum position size limit
        max_size = available_capital * self.parameters.get('max_position_size', 0.1)
        return min(position_size, max_size)
    
    def should_exit_position(self, position: Dict[str, Any], market_data: List[MarketData]) -> bool:
        """Determine if position should be closed based on trend reversal"""
        if not market_data:
            return False
        
        current_price = market_data[-1].close
        entry_price = position['entry_price']
        
        # Check stop loss
        stop_loss = self.parameters.get('stop_loss', 0.02)
        if position['quantity'] > 0:  # Long position
            if current_price <= entry_price * (1 - stop_loss):
                return True
        else:  # Short position
            if current_price >= entry_price * (1 + stop_loss):
                return True
        
        # Check trend reversal using sensory layer
        df = self._market_data_to_dataframe(market_data[-20:])
        indicators = self.indicators.calculate_indicators(df, ['sma'])
        
        short_ma = indicators.get(f'sma_{self.short_ma_period}', None)
        long_ma = indicators.get(f'sma_{self.long_ma_period}', None)
        
        if short_ma is not None and long_ma is not None:
            current_short_ma = short_ma.iloc[-1]
            current_long_ma = long_ma.iloc[-1]
            
            # Trend reversal signal
            if position['quantity'] > 0 and current_short_ma < current_long_ma:
                return True
            elif position['quantity'] < 0 and current_short_ma > current_long_ma:
                return True
        
        return False
