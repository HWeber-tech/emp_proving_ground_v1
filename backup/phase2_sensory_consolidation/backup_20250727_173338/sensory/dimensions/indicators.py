"""
Basic Technical Indicators
==========================

Simple technical indicator calculations for strategy templates.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List


class TechnicalIndicators:
    """Basic technical indicator calculations."""
    
    def calculate_indicators(self, df: pd.DataFrame, indicators: List[str]) -> Dict[str, Any]:
        """Calculate requested technical indicators."""
        results = {}
        
        for indicator in indicators:
            if indicator == 'sma':
                for period in [5, 10, 20, 50]:
                    key = f'sma_{period}'
                    results[key] = self._calculate_sma(df['close'], period)
            
            elif indicator == 'rsi':
                results['rsi'] = self._calculate_rsi(df['close'], 14)
            
            elif indicator == 'macd':
                macd_line, macd_signal = self._calculate_macd(df['close'])
                results['macd'] = macd_line
                results['macd_signal'] = macd_signal
        
        return results
    
    def _calculate_sma(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate Simple Moving Average."""
        return prices.rolling(window=period).mean()
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple:
        """Calculate MACD line and signal line."""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        
        macd_line = ema_fast - ema_slow
        macd_signal = macd_line.ewm(span=signal).mean()
        
        return macd_line, macd_signal
