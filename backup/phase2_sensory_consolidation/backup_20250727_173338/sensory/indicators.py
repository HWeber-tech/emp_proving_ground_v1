"""
Technical Indicators for Sensory Processing
Real implementations of technical analysis indicators
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from abc import ABC, abstractmethod


class BaseIndicator(ABC):
    """Base class for all technical indicators"""
    
    @abstractmethod
    def calculate(self, data: pd.DataFrame) -> Dict:
        """Calculate indicator values"""
        pass


class RSIIndicator(BaseIndicator):
    """Relative Strength Index indicator"""
    
    def __init__(self, period: int = 14):
        self.period = period
    
    def calculate(self, data: pd.DataFrame) -> Dict:
        """Calculate RSI values"""
        if len(data) < self.period:
            return None
        
        close_prices = data['close'].values
        delta = np.diff(close_prices)
        
        gains = np.where(delta > 0, delta, 0)
        losses = np.where(delta < 0, -delta, 0)
        
        avg_gains = np.convolve(gains, np.ones(self.period)/self.period, mode='valid')
        avg_losses = np.convolve(losses, np.ones(self.period)/self.period, mode='valid')
        
        rs = avg_gains / (avg_losses + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        
        current_rsi = rsi[-1] if len(rsi) > 0 else 50
        
        # Determine signal
        if current_rsi > 70:
            signal = 'bearish'
            strength = (current_rsi - 70) / 30
        elif current_rsi < 30:
            signal = 'bullish'
            strength = (30 - current_rsi) / 30
        else:
            signal = 'neutral'
            strength = 0.5
        
        return {
            'value': float(current_rsi),
            'signal': signal,
            'strength': float(strength)
        }


class MACDIndicator(BaseIndicator):
    """MACD (Moving Average Convergence Divergence) indicator"""
    
    def __init__(self, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9):
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
    
    def calculate(self, data: pd.DataFrame) -> Dict:
        """Calculate MACD values"""
        if len(data) < self.slow_period:
            return None
        
        close_prices = data['close'].values
        
        # Calculate EMAs
        ema_fast = self._ema(close_prices, self.fast_period)
        ema_slow = self._ema(close_prices, self.slow_period)
        
        macd_line = ema_fast - ema_slow
        signal_line = self._ema(macd_line, self.signal_period)
        histogram = macd_line - signal_line
        
        # Get current values
        current_macd = macd_line[-1] if len(macd_line) > 0 else 0
        current_signal = signal_line[-1] if len(signal_line) > 0 else 0
        
        # Determine signal
        if current_macd > current_signal and current_macd > 0:
            signal = 'bullish'
            strength = min(1.0, abs(current_macd) / 0.01)
        elif current_macd < current_signal and current_macd < 0:
            signal = 'bearish'
            strength = min(1.0, abs(current_macd) / 0.01)
        else:
            signal = 'neutral'
            strength = 0.5
        
        return {
            'value': float(current_macd),
            'signal': signal,
            'strength': float(strength)
        }
    
    def _ema(self, data: np.ndarray, period: int) -> np.ndarray:
        """Calculate Exponential Moving Average"""
        if len(data) < period:
            return np.array([])
        
        alpha = 2 / (period + 1)
        ema = np.zeros_like(data)
        ema[period-1] = np.mean(data[:period])
        
        for i in range(period, len(data)):
            ema[i] = alpha * data[i] + (1 - alpha) * ema[i-1]
        
        return ema


class BollingerBandsIndicator(BaseIndicator):
    """Bollinger Bands indicator"""
    
    def __init__(self, period: int = 20, num_std: float = 2.0):
        self.period = period
        self.num_std = num_std
    
    def calculate(self, data: pd.DataFrame) -> Dict:
        """Calculate Bollinger Bands values"""
        if len(data) < self.period:
            return None
        
        close_prices = data['close'].values
        
        # Calculate moving average and standard deviation
        sma = np.convolve(close_prices, np.ones(self.period)/self.period, mode='valid')
        rolling_std = np.array([np.std(close_prices[i:i+self.period]) 
                               for i in range(len(close_prices) - self.period + 1)])
        
        upper_band = sma + (self.num_std * rolling_std)
        lower_band = sma - (self.num_std * rolling_std)
        
        current_price = close_prices[-1]
        current_upper = upper_band[-1] if len(upper_band) > 0 else 0
        current_lower = lower_band[-1] if len(lower_band) > 0 else 0
        
        # Determine signal
        if current_price >= current_upper:
            signal = 'bearish'
            strength = min(1.0, (current_price - current_upper) / current_upper)
        elif current_price <= current_lower:
            signal = 'bullish'
            strength = min(1.0, (current_lower - current_price) / current_lower)
        else:
            signal = 'neutral'
            strength = 0.5
        
        return {
            'value': float(current_price),
            'signal': signal,
            'strength': float(strength)
        }


class VolumeIndicator(BaseIndicator):
    """Volume analysis indicator"""
    
    def __init__(self, period: int = 20):
        self.period = period
    
    def calculate(self, data: pd.DataFrame) -> Dict:
        """Calculate volume-based signals"""
        if len(data) < self.period:
            return None
        
        volumes = data['volume'].values
        avg_volume = np.mean(volumes[-self.period:])
        current_volume = volumes[-1]
        
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
        
        # Determine signal
        if volume_ratio > 2.0:
            signal = 'high_volume'
            strength = min(1.0, (volume_ratio - 2.0) / 2.0)
        elif volume_ratio < 0.5:
            signal = 'low_volume'
            strength = min(1.0, (0.5 - volume_ratio) / 0.5)
        else:
            signal = 'normal_volume'
            strength = 0.5
        
        return {
            'value': float(current_volume),
            'signal': signal,
            'strength': float(strength)
        }


class MomentumIndicator(BaseIndicator):
    """Momentum indicator"""
    
    def __init__(self, period: int = 10):
        self.period = period
    
    def calculate(self, data: pd.DataFrame) -> Dict:
        """Calculate momentum values"""
        if len(data) < self.period + 1:
            return None
        
        close_prices = data['close'].values
        momentum = close_prices[-1] - close_prices[-self.period]
        
        # Normalize momentum
        price_range = np.max(close_prices[-self.period:]) - np.min(close_prices[-self.period:])
        normalized_momentum = momentum / (price_range + 1e-10)
        
        # Determine signal
        if normalized_momentum > 0.02:
            signal = 'bullish'
            strength = min(1.0, normalized_momentum / 0.1)
        elif normalized_momentum < -0.02:
            signal = 'bearish'
            strength = min(1.0, abs(normalized_momentum) / 0.1)
        else:
            signal = 'neutral'
            strength = 0.5
        
        return {
            'value': float(momentum),
            'signal': signal,
            'strength': float(strength)
        }


class SupportResistanceIndicator(BaseIndicator):
    """Support and Resistance levels indicator"""
    
    def __init__(self, period: int = 50, tolerance: float = 0.01):
        self.period = period
        self.tolerance = tolerance
    
    def calculate(self, data: pd.DataFrame) -> Dict:
        """Calculate support and resistance levels"""
        if len(data) < self.period:
            return None
        
        high_prices = data['high'].values[-self.period:]
        low_prices = data['low'].values[-self.period:]
        close_prices = data['close'].values
        
        # Find support and resistance levels
        resistance = np.max(high_prices)
        support = np.min(low_prices)
        
        current_price = close_prices[-1]
        
        # Determine signal
        resistance_distance = abs(current_price - resistance) / resistance
        support_distance = abs(current_price - support) / support
        
        if resistance_distance < self.tolerance:
            signal = 'resistance'
            strength = 1.0 - (resistance_distance / self.tolerance)
        elif support_distance < self.tolerance:
            signal = 'support'
            strength = 1.0 - (support_distance / self.tolerance)
        else:
            signal = 'neutral'
            strength = 0.5
        
        return {
            'value': float(current_price),
            'signal': signal,
            'strength': float(strength),
            'support': float(support),
            'resistance': float(resistance)
        }
