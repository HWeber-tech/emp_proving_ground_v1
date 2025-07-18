"""
Technical Indicators Module - How Sense

This module contains all technical indicators for market analysis.
All indicators are implemented as part of the "how" sense.

Author: EMP Development Team
Date: July 18, 2024
Phase: 2 - Sensory Cortex Refactoring
"""

import logging
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class TechnicalIndicators:
    """
    Technical indicators for market analysis.
    
    This class implements various technical indicators including:
    - Trend indicators (SMA, EMA, MACD)
    - Momentum indicators (RSI, Stochastic, Williams %R)
    - Volatility indicators (Bollinger Bands, ATR)
    - Volume indicators (OBV, VWAP)
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize technical indicators with configuration"""
        self.config = config or {}
        logger.info("Technical Indicators module initialized")
    
    def calculate_all(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate all available technical indicators.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Dictionary containing all indicator values
        """
        if df.empty:
            return {}
        
        try:
            indicators = {}
            
            # Trend indicators
            indicators.update(self._calculate_trend_indicators(df))
            
            # Momentum indicators
            indicators.update(self._calculate_momentum_indicators(df))
            
            # Volatility indicators
            indicators.update(self._calculate_volatility_indicators(df))
            
            # Volume indicators
            indicators.update(self._calculate_volume_indicators(df))
            
            logger.info(f"Calculated {len(indicators)} technical indicators")
            return indicators
            
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {e}")
            return {}
    
    def calculate_indicators(self, df: pd.DataFrame, 
                           indicators_list: List[str] = None) -> Dict[str, Any]:
        """
        Calculate specific technical indicators.
        
        Args:
            df: DataFrame with OHLCV data
            indicators_list: List of indicators to calculate (None for all)
            
        Returns:
            Dictionary of indicator values
        """
        if df.empty:
            return {}
        
        if indicators_list is None:
            return self.calculate_all(df)
        
        try:
            indicators = {}
            
            for indicator in indicators_list:
                if indicator in ['sma', 'ema', 'macd']:
                    indicators.update(self._calculate_trend_indicators(df, [indicator]))
                elif indicator in ['rsi', 'stochastic', 'williams_r']:
                    indicators.update(self._calculate_momentum_indicators(df, [indicator]))
                elif indicator in ['bollinger_bands', 'atr']:
                    indicators.update(self._calculate_volatility_indicators(df, [indicator]))
                elif indicator in ['obv', 'vwap']:
                    indicators.update(self._calculate_volume_indicators(df, [indicator]))
                else:
                    logger.warning(f"Unknown indicator: {indicator}")
            
            return indicators
            
        except Exception as e:
            logger.error(f"Error calculating specific indicators: {e}")
            return {}
    
    def _calculate_trend_indicators(self, df: pd.DataFrame, 
                                  indicators: List[str] = None) -> Dict[str, Any]:
        """Calculate trend indicators"""
        if indicators is None:
            indicators = ['sma', 'ema', 'macd']
        
        result = {}
        
        try:
            if 'sma' in indicators:
                result['sma_20'] = self._simple_moving_average(df['close'], 20)
                result['sma_50'] = self._simple_moving_average(df['close'], 50)
                result['sma_200'] = self._simple_moving_average(df['close'], 200)
            
            if 'ema' in indicators:
                result['ema_12'] = self._exponential_moving_average(df['close'], 12)
                result['ema_26'] = self._exponential_moving_average(df['close'], 26)
            
            if 'macd' in indicators:
                macd_data = self._macd(df['close'])
                result['macd'] = macd_data['macd']
                result['macd_signal'] = macd_data['signal']
                result['macd_histogram'] = macd_data['histogram']
                
        except Exception as e:
            logger.error(f"Error calculating trend indicators: {e}")
        
        return result
    
    def _calculate_momentum_indicators(self, df: pd.DataFrame, 
                                     indicators: List[str] = None) -> Dict[str, Any]:
        """Calculate momentum indicators"""
        if indicators is None:
            indicators = ['rsi', 'stochastic', 'williams_r']
        
        result = {}
        
        try:
            if 'rsi' in indicators:
                result['rsi'] = self._relative_strength_index(df['close'])
            
            if 'stochastic' in indicators:
                stoch_data = self._stochastic(df)
                result['stoch_k'] = stoch_data['k']
                result['stoch_d'] = stoch_data['d']
            
            if 'williams_r' in indicators:
                result['williams_r'] = self._williams_percent_r(df)
                
        except Exception as e:
            logger.error(f"Error calculating momentum indicators: {e}")
        
        return result
    
    def _calculate_volatility_indicators(self, df: pd.DataFrame, 
                                       indicators: List[str] = None) -> Dict[str, Any]:
        """Calculate volatility indicators"""
        if indicators is None:
            indicators = ['bollinger_bands', 'atr']
        
        result = {}
        
        try:
            if 'bollinger_bands' in indicators:
                bb_data = self._bollinger_bands(df['close'])
                result['bb_upper'] = bb_data['upper']
                result['bb_middle'] = bb_data['middle']
                result['bb_lower'] = bb_data['lower']
                result['bb_width'] = bb_data['width']
            
            if 'atr' in indicators:
                result['atr'] = self._average_true_range(df)
                
        except Exception as e:
            logger.error(f"Error calculating volatility indicators: {e}")
        
        return result
    
    def _calculate_volume_indicators(self, df: pd.DataFrame, 
                                   indicators: List[str] = None) -> Dict[str, Any]:
        """Calculate volume indicators"""
        if indicators is None:
            indicators = ['obv', 'vwap']
        
        result = {}
        
        try:
            if 'obv' in indicators:
                result['obv'] = self._on_balance_volume(df)
            
            if 'vwap' in indicators:
                result['vwap'] = self._volume_weighted_average_price(df)
                
        except Exception as e:
            logger.error(f"Error calculating volume indicators: {e}")
        
        return result
    
    # Individual indicator implementations
    def _simple_moving_average(self, series: pd.Series, period: int) -> pd.Series:
        """Calculate Simple Moving Average"""
        return series.rolling(window=period).mean()
    
    def _exponential_moving_average(self, series: pd.Series, period: int) -> pd.Series:
        """Calculate Exponential Moving Average"""
        return series.ewm(span=period).mean()
    
    def _macd(self, series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
        """Calculate MACD (Moving Average Convergence Divergence)"""
        ema_fast = self._exponential_moving_average(series, fast)
        ema_slow = self._exponential_moving_average(series, slow)
        macd_line = ema_fast - ema_slow
        signal_line = self._exponential_moving_average(macd_line, signal)
        histogram = macd_line - signal_line
        
        return {
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        }
    
    def _relative_strength_index(self, series: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index (RSI)"""
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _stochastic(self, df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> Dict[str, pd.Series]:
        """Calculate Stochastic Oscillator"""
        low_min = df['low'].rolling(window=k_period).min()
        high_max = df['high'].rolling(window=k_period).max()
        k = 100 * ((df['close'] - low_min) / (high_max - low_min))
        d = k.rolling(window=d_period).mean()
        
        return {'k': k, 'd': d}
    
    def _williams_percent_r(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Williams %R"""
        low_min = df['low'].rolling(window=period).min()
        high_max = df['high'].rolling(window=period).max()
        williams_r = -100 * ((high_max - df['close']) / (high_max - low_min))
        return williams_r
    
    def _bollinger_bands(self, series: pd.Series, period: int = 20, std_dev: float = 2) -> Dict[str, pd.Series]:
        """Calculate Bollinger Bands"""
        middle = self._simple_moving_average(series, period)
        std = series.rolling(window=period).std()
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        width = (upper - lower) / middle
        
        return {
            'upper': upper,
            'middle': middle,
            'lower': lower,
            'width': width
        }
    
    def _average_true_range(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range (ATR)"""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()
        
        return atr
    
    def _on_balance_volume(self, df: pd.DataFrame) -> pd.Series:
        """Calculate On-Balance Volume (OBV)"""
        obv = pd.Series(index=df.index, dtype=float)
        obv.iloc[0] = df['volume'].iloc[0]
        
        for i in range(1, len(df)):
            if df['close'].iloc[i] > df['close'].iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] + df['volume'].iloc[i]
            elif df['close'].iloc[i] < df['close'].iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] - df['volume'].iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]
        
        return obv
    
    def _volume_weighted_average_price(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Volume Weighted Average Price (VWAP)"""
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        vwap = (typical_price * df['volume']).cumsum() / df['volume'].cumsum()
        return vwap


# Example usage
if __name__ == "__main__":
    # Test technical indicators
    indicators = TechnicalIndicators()
    print("Technical Indicators module initialized successfully") 