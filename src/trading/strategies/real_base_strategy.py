"""
Real Base Strategy Implementation
Replaces the stub with functional signal generation using technical indicators
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np
import pandas as pd
import json

from ..models import TradingSignal, SignalType, MarketData
from ...config.strategy_config import StrategyConfig
from ...sensory.indicators import TechnicalIndicators

logger = logging.getLogger(__name__)

@dataclass
class StrategyParameters:
    """Container for strategy parameters with validation"""
    
    # Technical indicators
    fast_ma_period: int = 12
    slow_ma_period: int = 26
    rsi_period: int = 14
    rsi_overbought: float = 70.0
    rsi_oversold: float = 30.0
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    
    # Risk management
    stop_loss_pct: float = 0.02  # 2%
    take_profit_pct: float = 0.04  # 4%
    max_position_size: float = 0.1  # 10% of portfolio
    
    # Entry/Exit conditions
    min_volume_ratio: float = 1.5  # Minimum volume vs average
    trend_confirmation_bars: int = 3
    min_price_change: float = 0.001  # Minimum price change for signal
    
    def validate(self) -> bool:
        """Validate parameter ranges"""
        validations = [
            0 < self.fast_ma_period < self.slow_ma_period,
            0 < self.rsi_period <= 50,
            0 < self.rsi_oversold < self.rsi_overbought < 100,
            0 < self.stop_loss_pct < 0.1,
            0 < self.take_profit_pct < 0.2,
            0 < self.max_position_size <= 1.0,
            self.min_volume_ratio > 0,
            self.trend_confirmation_bars > 0,
            self.min_price_change > 0
        ]
        
        return all(validations)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert parameters to dictionary"""
        return {
            'fast_ma_period': self.fast_ma_period,
            'slow_ma_period': self.slow_ma_period,
            'rsi_period': self.rsi_period,
            'rsi_overbought': self.rsi_overbought,
            'rsi_oversold': self.rsi_oversold,
            'macd_fast': self.macd_fast,
            'macd_slow': self.macd_slow,
            'macd_signal': self.macd_signal,
            'stop_loss_pct': self.stop_loss_pct,
            'take_profit_pct': self.take_profit_pct,
            'max_position_size': self.max_position_size,
            'min_volume_ratio': self.min_volume_ratio,
            'trend_confirmation_bars': self.trend_confirmation_bars,
            'min_price_change': self.min_price_change
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StrategyParameters':
        """Create parameters from dictionary"""
        return cls(**data)

class RealBaseStrategy:
    """
    Real implementation of base trading strategy.
    Replaces the stub with functional signal generation using technical analysis.
    """
    
    def __init__(self, name: str, config: StrategyConfig):
        self.name = name
        self.config = config
        self.parameters = StrategyParameters()
        self.indicators = TechnicalIndicators()
        
        # Strategy state
        self.is_active = True
        self.last_signal_time: Optional[datetime] = None
        self.signal_history: List[TradingSignal] = []
        self.market_data_buffer: List[MarketData] = []
        self.max_buffer_size = 200
        
        # Performance tracking
        self.signals_generated = 0
        self.signals_executed = 0
        self.total_pnl = 0.0
        
        logger.info(f"RealBaseStrategy '{self.name}' initialized")
    
    def generate_signal(self, market_data: MarketData) -> Optional[TradingSignal]:
        """Generate trading signal based on market data and technical indicators"""
        try:
            if not self.is_active:
                return None
            
            # Update market data buffer
            self.update_market_data_buffer(market_data)
            
            # Ensure we have enough data
            if len(self.market_data_buffer) < max(
                self.parameters.slow_ma_period, 
                self.parameters.rsi_period,
                self.parameters.macd_slow
            ):
                return None
            
            # Calculate technical indicators
            indicators = self.calculate_indicators()
            if not indicators:
                return None
            
            # Generate signal based on strategy logic
            signal = self.evaluate_signal_conditions(market_data, indicators)
            
            # Validate signal
            if signal and self.validate_signal(signal, market_data):
                self.last_signal_time = market_data.timestamp
                self.signals_generated += 1
                self.signal_history.append(signal)
                
                logger.info(f"Signal generated: {signal.signal_type} for {signal.symbol} at {signal.price}")
                return signal
            
            return None
            
        except Exception as e:
            logger.error(f"Error generating signal: {e}")
            return None
    
    def update_market_data_buffer(self, market_data: MarketData) -> None:
        """Update the market data buffer with new data"""
        self.market_data_buffer.append(market_data)
        
        # Maintain buffer size
        if len(self.market_data_buffer) > self.max_buffer_size:
            self.market_data_buffer = self.market_data_buffer[-self.max_buffer_size:]
    
    def calculate_indicators(self) -> Dict[str, Any]:
        """Calculate technical indicators based on current market data"""
        if len(self.market_data_buffer) < 2:
            return {}
        
        try:
            # Convert to pandas DataFrame for easier calculation
            df = self.market_data_to_dataframe()
            
            # Moving Averages
            fast_ma = df['close'].rolling(window=self.parameters.fast_ma_period).mean().iloc[-1]
            slow_ma = df['close'].rolling(window=self.parameters.slow_ma_period).mean().iloc[-1]
            
            # RSI
            rsi = self.calculate_rsi(df['close'], self.parameters.rsi_period)
            
            # MACD
            macd_line, signal_line = self.calculate_macd(
                df['close'], 
                self.parameters.macd_fast, 
                self.parameters.macd_slow, 
                self.parameters.macd_signal
            )
            
            # Volume analysis
            avg_volume = df['volume'].rolling(window=20).mean().iloc[-1]
            current_volume = df['volume'].iloc[-1]
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
            
            # Price momentum
            price_change = (df['close'].iloc[-1] - df['close'].iloc[-2]) / df['close'].iloc[-2]
            
            # Support and resistance levels
            support_level = df['low'].rolling(window=20).min().iloc[-1]
            resistance_level = df['high'].rolling(window=20).max().iloc[-1]
            
            return {
                'fast_ma': fast_ma,
                'slow_ma': slow_ma,
                'rsi': rsi,
                'macd_line': macd_line,
                'signal_line': signal_line,
                'volume_ratio': volume_ratio,
                'price_change': price_change,
                'support_level': support_level,
                'resistance_level': resistance_level,
                'current_price': df['close'].iloc[-1]
            }
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
            return {}
    
    def market_data_to_dataframe(self) -> pd.DataFrame:
        """Convert market data buffer to pandas DataFrame"""
        data = []
        for md in self.market_data_buffer:
            data.append({
                'timestamp': md.timestamp,
                'open': md.open,
                'high': md.high,
                'low': md.low,
                'close': md.close,
                'volume': md.volume
            })
        
        return pd.DataFrame(data)
    
    def calculate_rsi(self, prices: pd.Series, period: int) -> float:
        """Calculate RSI (Relative Strength Index)"""
        if len(prices) < period + 1:
            return 50.0
        
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return float(rsi.iloc[-1]) if not rsi.empty else 50.0
    
    def calculate_macd(self, prices: pd.Series, fast: int, slow: int, signal: int) -> tuple:
        """Calculate MACD (Moving Average Convergence Divergence)"""
        if len(prices) < slow + signal:
            return 0.0, 0.0
        
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        
        return float(macd_line.iloc[-1]), float(signal_line.iloc[-1])
    
    def evaluate_signal_conditions(self, market_data: MarketData, indicators: Dict[str, Any]) -> Optional[TradingSignal]:
        """Evaluate signal conditions based on technical indicators"""
        if not indicators:
            return None
        
        current_price = indicators['current_price']
        
        # Check minimum price change
        if abs(indicators['price_change']) < self.parameters.min_price_change:
            return None
        
        # Check volume
        if indicators['volume_ratio'] < self.parameters.min_volume_ratio:
            return None
        
        # Moving Average Crossover
        if indicators['fast_ma'] > indicators['slow_ma'] and indicators['rsi'] < self.parameters.rsi_oversold:
            # Bullish signal
            stop_loss = current_price * (1 - self.parameters.stop_loss_pct)
            take_profit = current_price * (1 + self.parameters.take_profit_pct)
            
            return TradingSignal(
                symbol=market_data.symbol,
                signal_type=SignalType.BUY,
                price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                strategy_name=self.name,
                confidence=0.75,
                timestamp=market_data.timestamp
            )
        
        elif indicators['fast_ma'] < indicators['slow_ma'] and indicators['rsi'] > self.parameters.rsi_overbought:
            # Bearish signal
            stop_loss = current_price * (1 + self.parameters.stop_loss_pct)
            take_profit = current_price * (1 - self.parameters.take_profit_pct)
            
            return TradingSignal(
                symbol=market_data.symbol,
                signal_type=SignalType.SELL,
                price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                strategy_name=self.name,
                confidence=0.75,
                timestamp=market_data.timestamp
            )
        
        # MACD Crossover
        elif indicators['macd_line'] > indicators['signal_line'] and indicators['rsi'] < 50:
            # Bullish MACD signal
            stop_loss = current_price * (1 - self.parameters.stop_loss_pct)
            take_profit = current_price * (1 + self.parameters.take_profit_pct)
            
            return TradingSignal(
                symbol=market_data.symbol,
                signal_type=SignalType.BUY,
                price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                strategy_name=self.name,
                confidence=0.70,
                timestamp=market_data.timestamp
            )
        
        elif indicators['macd_line'] < indicators['signal_line'] and indicators['rsi'] > 50:
            # Bearish MACD signal
            stop_loss = current_price * (1 + self.parameters.stop_loss_pct)
            take_profit = current_price * (1 - self.parameters.take_profit_pct)
            
            return TradingSignal(
                symbol=market_data.symbol,
                signal_type=SignalType.SELL,
                price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                strategy_name=self.name,
                confidence=0.70,
                timestamp=market_data.timestamp
            )
        
        return None
    
    def validate_signal(self, signal: TradingSignal, market_data: MarketData) -> bool:
        """Validate the generated signal"""
        # Check if price is reasonable
        if signal.price <= 0:
            return False
        
        # Check if stop loss and take profit are reasonable
        if signal.stop_loss <= 0 or signal.take_profit <= 0:
            return False
        
        # Check if stop loss is in correct direction
        if signal.signal_type == SignalType.BUY and signal.stop_loss >= signal.price:
            return False
        
        if signal.signal_type == SignalType.SELL and signal.stop_loss <= signal.price:
            return False
        
        # Check if take profit is in correct direction
        if signal.signal_type == SignalType.BUY and signal.take_profit <= signal.price:
            return False
        
        if signal.signal_type == SignalType.SELL and signal.take_profit >= signal.price:
            return False
        
        return True
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get current strategy parameters"""
        return self.parameters.to_dict()
    
    def set_parameters(self, parameters: Dict[str, Any]) -> bool:
        """Update strategy parameters"""
        try:
            new_params = StrategyParameters.from_dict(parameters)
            if new_params.validate():
                self.parameters = new_params
                logger.info(f"Strategy parameters updated for {self.name}")
                return True
            else:
                logger.warning("Invalid strategy parameters provided")
                return False
        except Exception as e:
            logger.error(f"Error updating parameters: {e}")
            return False
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get strategy performance metrics"""
        return {
            'name': self.name,
            'signals_generated': self.signals_generated,
            'signals_executed': self.signals_executed,
            'total_pnl': self.total_pnl,
            'is_active': self.is_active,
            'last_signal_time': self.last_signal_time.isoformat() if self.last_signal_time else None
        }
    
    def reset(self) -> None:
        """Reset strategy state"""
        self.signals_generated = 0
        self.signals_executed = 0
        self.total_pnl = 0.0
        self.signal_history.clear()
        self.market_data_buffer.clear()
        self.last_signal_time = None
        logger.info(f"Strategy {self.name} reset")
