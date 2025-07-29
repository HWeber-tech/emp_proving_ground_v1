#!/usr/bin/env python3
"""
Real Base Strategy Implementation
=================================

Complete functional trading strategy with actual signal generation.
Replaces all mock implementations with genuine strategy logic.
"""

import logging
from typing import Dict, Any
import pandas as pd
import numpy as np

from src.core.market_data import MarketData

logger = logging.getLogger(__name__)


class RealBaseStrategy:
    """Real base strategy with actual signal generation."""
    
    def __init__(self):
        self.parameters = {
            'rsi_period': 14,
            'rsi_overbought': 70,
            'rsi_oversold': 30,
            'sma_fast': 20,
            'sma_slow': 50,
            'volume_threshold': 1000
        }
        logger.info("RealBaseStrategy initialized")
        
    def generate_signal(self, market_data: MarketData) -> str:
        """Generate trading signal based on market data."""
        if not hasattr(market_data, 'data') or market_data.data.empty:
            return 'HOLD'
            
        df = market_data.data
        
        # Ensure we have enough data
        if len(df) < max(self.parameters['sma_slow'], self.parameters['rsi_period']):
            return 'HOLD'
            
        # Calculate indicators
        close = df['close']
        
        # Simple moving averages
        sma_fast = close.rolling(window=self.parameters['sma_fast']).mean().iloc[-1]
        sma_slow = close.rolling(window=self.parameters['sma_slow']).mean().iloc[-1]
        
        # RSI calculation
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.parameters['rsi_period']).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.parameters['rsi_period']).mean()
        
        # Handle division by zero
        if loss.iloc[-1] == 0:
            rsi = 100.0
        else:
            rs = gain.iloc[-1] / loss.iloc[-1]
            rsi = 100 - (100 / (1 + rs))
        
        # Volume filter
        volume_ok = True
        if 'volume' in df.columns:
            avg_volume = df['volume'].rolling(window=20).mean().iloc[-1]
            volume_ok = avg_volume > self.parameters['volume_threshold']
        
        # Generate signal
        current_price = close.iloc[-1]
        
        # Trend following with RSI confirmation
        if current_price > sma_fast > sma_slow and rsi < self.parameters['rsi_oversold'] and volume_ok:
            return 'BUY'
        elif current_price < sma_fast < sma_slow and rsi > self.parameters['rsi_overbought'] and volume_ok:
            return 'SELL'
        else:
            return 'HOLD'
            
    def get_parameters(self) -> Dict[str, Any]:
        """Get strategy parameters."""
        return self.parameters.copy()
        
    def set_parameters(self, parameters: Dict[str, Any]):
        """Set strategy parameters with validation."""
        for key, value in parameters.items():
            if key in self.parameters:
                if isinstance(value, (int, float)) and value > 0:
                    self.parameters[key] = value
                else:
                    raise ValueError(f"Invalid parameter value for {key}: {value}")
            else:
                raise ValueError(f"Unknown parameter: {key}")
                
    def calculate_indicators(self, market_data: MarketData) -> Dict[str, Any]:
        """Calculate and return all indicators for analysis."""
        if not hasattr(market_data, 'data') or market_data.data.empty:
            return {}
            
        df = market_data.data
        
        # Ensure we have enough data
        if len(df) < 50:
            return {}
            
        close = df['close']
        
        # Calculate all indicators
        sma_20 = close.rolling(window=20).mean().iloc[-1]
        sma_50 = close.rolling(window=50).mean().iloc[-1]
        
        # RSI
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        
        if loss.iloc[-1] == 0:
            rsi = 100.0
        else:
            rs = gain.iloc[-1] / loss.iloc[-1]
            rsi = 100 - (100 / (1 + rs))
        
        # MACD
        ema_12 = close.ewm(span=12, adjust=False).mean()
        ema_26 = close.ewm(span=26, adjust=False).mean()
        macd = ema_12.iloc[-1] - ema_26.iloc[-1]
        
        # Bollinger Bands
        sma_20_bb = close.rolling(window=20).mean().iloc[-1]
        std_20 = close.rolling(window=20).std().iloc[-1]
        
        return {
            'current_price': float(close.iloc[-1]),
            'sma_20': float(sma_20),
            'sma_50': float(sma_50),
            'rsi': float(rsi),
            'macd': float(macd),
            'bollinger_upper': float(sma_20_bb + 2 * std_20),
            'bollinger_lower': float(sma_20_bb - 2 * std_20),
            'bollinger_middle': float(sma_20_bb)
        }
