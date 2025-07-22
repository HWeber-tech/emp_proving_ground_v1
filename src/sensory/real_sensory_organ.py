#!/usr/bin/env python3
"""
Real Sensory Organ Implementation
=================================

Complete functional sensory processing with actual technical indicators.
Replaces all mock implementations with genuine indicator calculations.
"""

import logging
from typing import Dict, Any
import pandas as pd
import numpy as np

from src.data import MarketData

logger = logging.getLogger(__name__)


class RealSensoryOrgan:
    """Real sensory organ with actual indicator calculations."""
    
    def __init__(self):
        logger.info("RealSensoryOrgan initialized")
        
    def process(self, market_data: MarketData) -> Dict[str, Any]:
        """Process market data and return indicators."""
        if not hasattr(market_data, 'data') or market_data.data.empty:
            return {}
            
        df = market_data.data
        
        # Calculate technical indicators
        indicators = {
            'sma_20': self.calculate_sma(df['close'], 20),
            'ema_12': self.calculate_ema(df['close'], 12),
            'rsi_14': self.calculate_rsi(df['close'], 14),
            'macd': self.calculate_macd(df['close']),
            'bollinger_bands': self.calculate_bollinger_bands(df['close'], 20),
            'volume_sma': self.calculate_sma(df['volume'], 20) if 'volume' in df.columns else None
        }
        
        return indicators
        
    def calculate_sma(self, prices: pd.Series, period: int) -> float:
        """Calculate Simple Moving Average."""
        if len(prices) < period:
            return float(prices.iloc[-1]) if len(prices) > 0 else 0.0
        return float(prices.rolling(window=period).mean().iloc[-1])
        
    def calculate_ema(self, prices: pd.Series, period: int) -> float:
        """Calculate Exponential Moving Average."""
        if len(prices) < period:
            return float(prices.iloc[-1]) if len(prices) > 0 else 0.0
        return float(prices.ewm(span=period, adjust=False).mean().iloc[-1])
        
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate Relative Strength Index."""
        if len(prices) < period + 1:
            return 50.0
            
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return float(rsi.iloc[-1]) if not rsi.empty else 50.0
        
    def calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, float]:
        """Calculate MACD indicator."""
        if len(prices) < slow:
            return {'macd': 0.0, 'signal': 0.0, 'histogram': 0.0}
            
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        histogram = macd - signal_line
        
        return {
            'macd': float(macd.iloc[-1]),
            'signal': float(signal_line.iloc[-1]),
            'histogram': float(histogram.iloc[-1])
        }
        
    def calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: float = 2.0) -> Dict[str, float]:
        """Calculate Bollinger Bands."""
        if len(prices) < period:
            return {'upper': 0.0, 'middle': 0.0, 'lower': 0.0}
            
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        
        return {
            'upper': float(upper_band.iloc[-1]),
            'middle': float(sma.iloc[-1]),
            'lower': float(lower_band.iloc[-1])
        }
        
    def calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> float:
        """Calculate Average True Range."""
        if len(high) < period:
            return 0.0
            
        high_low = high - low
        high_close = np.abs(high - close.shift())
        low_close = np.abs(low - close.shift())
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        atr = true_range.rolling(window=period).mean()
        
        return float(atr.iloc[-1]) if not atr.empty else 0.0
        
    def calculate_stochastic(self, high: pd.Series, low: pd.Series, close: pd.Series, k_period: int = 14, d_period: int = 3) -> Dict[str, float]:
        """Calculate Stochastic Oscillator."""
        if len(high) < k_period:
            return {'k': 50.0, 'd': 50.0}
            
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        
        k = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d = k.rolling(window=d_period).mean()
        
        return {
            'k': float(k.iloc[-1]),
            'd': float(d.iloc[-1])
        }
