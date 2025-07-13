"""
Sensory Cortex for the EMP Proving Ground system.

This module provides the 4D+1 sensory system:
- SensoryCortex: Main sensory processing engine
- SensoryReading: Complete sensory reading output
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from enum import Enum

logger = logging.getLogger(__name__)


@dataclass
class SensoryReading:
    """Represents a complete sensory reading from the 4D+1 cortex"""
    timestamp: datetime
    symbol: str
    why_score: float      # Fundamental/macro momentum
    how_score: float      # Institutional footprint
    what_score: float     # Technical patterns
    when_score: float     # Timing/session analysis
    anomaly_score: float  # Deviation detection
    raw_components: Dict[str, Any]
    confidence: float


class SensoryCortex:
    """4D+1 Sensory Cortex for market perception"""
    
    def __init__(self, symbol: str, data_storage):
        """
        Initialize sensory cortex
        
        Args:
            symbol: Trading symbol
            data_storage: Data storage instance
        """
        self.symbol = symbol
        self.data_storage = data_storage
        
        # Calibration data
        self.calibration_data = None
        self.anomaly_detector = None
        
        # Technical indicators cache
        self.ohlcv_cache = {}
        
        # Session definitions
        self.sessions = {
            'asian': (0, 8),      # 00:00-08:00 UTC
            'london': (8, 16),    # 08:00-16:00 UTC
            'new_york': (13, 21), # 13:00-21:00 UTC
            'overlap': (13, 16)   # London-NY overlap
        }
        
        logger.info(f"Initialized sensory cortex for {symbol}")
    
    def calibrate(self, start_time: datetime, end_time: datetime):
        """
        Calibrate the sensory cortex with historical data
        
        Args:
            start_time: Start time for calibration
            end_time: End time for calibration
        """
        try:
            # Load calibration data
            self.calibration_data = self.data_storage.load_tick_data(
                self.symbol, start_time, end_time
            )
            
            if self.calibration_data.empty:
                logger.warning(f"No calibration data available for {self.symbol}")
                return
            
            # Precompute OHLCV data
            self._precompute_ohlcv_data()
            
            # Train anomaly detector
            self._train_anomaly_detector()
            
            logger.info(f"Calibrated sensory cortex with {len(self.calibration_data)} ticks")
            
        except Exception as e:
            logger.error(f"Error calibrating sensory cortex: {e}")
    
    def _precompute_ohlcv_data(self):
        """Precompute OHLCV data for different timeframes"""
        if self.calibration_data is None or self.calibration_data.empty:
            return
        
        timeframes = ['M1', 'M5', 'M15', 'H1', 'H4', 'D1']
        
        for tf in timeframes:
            try:
                ohlcv = self.data_storage.get_ohlcv(
                    self.symbol, 
                    self.calibration_data['timestamp'].min(),
                    self.calibration_data['timestamp'].max(),
                    tf
                )
                
                if not ohlcv.empty:
                    ohlcv = self._add_technical_indicators(ohlcv)
                    self.ohlcv_cache[tf] = ohlcv
                    
            except Exception as e:
                logger.warning(f"Error precomputing {tf} data: {e}")
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to OHLCV data"""
        if df is None or df.empty:
            return df  # Return input unchanged if invalid
        
        df = df.copy()
        
        # Moving averages
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        df['ema_12'] = df['close'].ewm(span=12).mean()
        df['ema_26'] = df['close'].ewm(span=26).mean()
        
        # MACD
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        
        # ATR
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        df['atr'] = true_range.rolling(window=14).mean()
        
        # Volume indicators
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        return df
    
    def _train_anomaly_detector(self):
        """Train anomaly detection model"""
        if self.calibration_data is None or self.calibration_data.empty:
            return
        
        try:
            # Simple statistical anomaly detection
            # In a real implementation, you might use more sophisticated methods
            # like isolation forest, autoencoder, etc.
            
            # Calculate price volatility
            returns = self.calibration_data['bid'].pct_change().dropna()
            
            # Store statistics for anomaly detection
            self.anomaly_stats = {
                'price_mean': self.calibration_data['bid'].mean(),
                'price_std': self.calibration_data['bid'].std(),
                'returns_mean': returns.mean(),
                'returns_std': returns.std(),
                'volume_mean': self.calibration_data['bid_volume'].mean(),
                'volume_std': self.calibration_data['bid_volume'].std()
            }
            
            logger.info("Trained anomaly detector")
            
        except Exception as e:
            logger.error(f"Error training anomaly detector: {e}")
    
    def perceive(self, market_state) -> SensoryReading:
        """
        Generate a complete sensory reading
        
        Args:
            market_state: Current market state
            
        Returns:
            SensoryReading with all sensory scores
        """
        try:
            # Get current OHLCV data
            current_ohlcv = self._get_current_ohlcv(market_state)
            
            # Calculate each sensory dimension
            why_score = self._calculate_why_score(market_state, current_ohlcv)
            how_score = self._calculate_how_score(market_state, current_ohlcv)
            what_score = self._calculate_what_score(market_state, current_ohlcv)
            when_score = self._calculate_when_score(market_state)
            anomaly_score = self._calculate_anomaly_score(market_state)
            
            # Calculate overall confidence
            confidence = self._calculate_confidence(market_state, current_ohlcv)
            
            # Collect raw components
            raw_components = {
                'why': self._get_why_components(market_state, current_ohlcv),
                'how': self._get_how_components(market_state, current_ohlcv),
                'what': self._get_what_components(market_state, current_ohlcv),
                'when': self._get_when_components(market_state),
                'anomaly': self._get_anomaly_components(market_state)
            }
            
            return SensoryReading(
                timestamp=market_state.timestamp,
                symbol=self.symbol,
                why_score=why_score,
                how_score=how_score,
                what_score=what_score,
                when_score=when_score,
                anomaly_score=anomaly_score,
                raw_components=raw_components,
                confidence=confidence
            )
            
        except Exception as e:
            logger.error(f"Error generating sensory reading: {e}")
            # Return neutral reading on error
            return SensoryReading(
                timestamp=market_state.timestamp,
                symbol=self.symbol,
                why_score=0.0,
                how_score=0.0,
                what_score=0.0,
                when_score=0.0,
                anomaly_score=0.0,
                raw_components={},
                confidence=0.0
            )
    
    def _get_current_ohlcv(self, market_state) -> Optional[Dict[str, pd.DataFrame]]:
        """Get current OHLCV data for different timeframes"""
        if not self.ohlcv_cache:
            return None
        
        current_time = market_state.timestamp
        
        # Get recent data for each timeframe
        recent_data = {}
        for tf, ohlcv in self.ohlcv_cache.items():
            # Get last 100 bars
            recent = ohlcv[ohlcv['timestamp'] <= current_time].tail(100)
            if not recent.empty:
                recent_data[tf] = recent
        
        return recent_data if recent_data else None
    
    def _calculate_why_score(self, market_state, current_ohlcv) -> float:
        """Calculate fundamental/macro momentum score"""
        try:
            score = 0.0
            components = []
            
            # Trend analysis
            if current_ohlcv and 'D1' in current_ohlcv:
                daily_data = current_ohlcv['D1']
                if len(daily_data) >= 20:
                    # Long-term trend
                    sma_20 = daily_data['sma_20'].iloc[-1]
                    current_price = market_state.mid_price
                    trend_score = 1.0 if current_price > sma_20 else -1.0
                    score += trend_score * 0.3
                    components.append(('trend', trend_score))
            
            # Momentum indicators
            if current_ohlcv and 'H4' in current_ohlcv:
                h4_data = current_ohlcv['H4']
                if len(h4_data) >= 14:
                    # RSI momentum
                    rsi = h4_data['rsi'].iloc[-1]
                    if rsi > 70:
                        momentum_score = -0.5  # Overbought
                    elif rsi < 30:
                        momentum_score = 0.5   # Oversold
                    else:
                        momentum_score = (rsi - 50) / 50  # Normalized
                    
                    score += momentum_score * 0.2
                    components.append(('momentum', momentum_score))
            
            # Volume analysis
            volume_ratio = market_state.bid_volume / (market_state.ask_volume + 1e-6)
            volume_score = (volume_ratio - 1.0) * 0.1  # Normalize around 1.0
            score += volume_score * 0.1
            components.append(('volume', volume_score))
            
            # Normalize to [-1, 1] range
            score = max(-1.0, min(1.0, score))
            
            return score
            
        except Exception as e:
            logger.error(f"Error calculating why score: {e}")
            return 0.0
    
    def _calculate_how_score(self, market_state, current_ohlcv) -> float:
        """Calculate institutional footprint score"""
        try:
            score = 0.0
            components = []
            
            # Spread analysis (institutional activity indicator)
            spread_bps = market_state.spread_bps
            if spread_bps < 1.0:
                spread_score = 0.5  # Tight spread = institutional activity
            elif spread_bps > 5.0:
                spread_score = -0.5  # Wide spread = low liquidity
            else:
                spread_score = (5.0 - spread_bps) / 4.0  # Normalized
            
            score += spread_score * 0.3
            components.append(('spread', spread_score))
            
            # Volume profile
            if current_ohlcv and 'H1' in current_ohlcv:
                h1_data = current_ohlcv['H1']
                if len(h1_data) >= 20:
                    avg_volume = h1_data['volume'].mean()
                    current_volume = market_state.bid_volume + market_state.ask_volume
                    volume_score = (current_volume - avg_volume) / (avg_volume + 1e-6)
                    volume_score = max(-1.0, min(1.0, volume_score))
                    
                    score += volume_score * 0.2
                    components.append(('volume_profile', volume_score))
            
            # Price action patterns
            if current_ohlcv and 'M15' in current_ohlcv:
                m15_data = current_ohlcv['M15']
                if len(m15_data) >= 10:
                    # Check for large moves
                    recent_highs = m15_data['high'].tail(10)
                    recent_lows = m15_data['low'].tail(10)
                    
                    range_expansion = (recent_highs.max() - recent_lows.min()) / m15_data['close'].mean()
                    if range_expansion > 0.01:  # 1% range
                        range_score = 0.3
                    else:
                        range_score = -0.1
                    
                    score += range_score * 0.2
                    components.append(('range_expansion', range_score))
            
            # Normalize to [-1, 1] range
            score = max(-1.0, min(1.0, score))
            
            return score
            
        except Exception as e:
            logger.error(f"Error calculating how score: {e}")
            return 0.0
    
    def _calculate_what_score(self, market_state, current_ohlcv) -> float:
        """Calculate technical patterns score"""
        try:
            score = 0.0
            components = []
            
            # MACD analysis
            if current_ohlcv and 'H1' in current_ohlcv:
                h1_data = current_ohlcv['H1']
                if len(h1_data) >= 26:
                    macd = h1_data['macd'].iloc[-1]
                    macd_signal = h1_data['macd_signal'].iloc[-1]
                    macd_hist = h1_data['macd_histogram'].iloc[-1]
                    
                    # MACD crossover
                    if macd > macd_signal and macd_hist > 0:
                        macd_score = 0.4
                    elif macd < macd_signal and macd_hist < 0:
                        macd_score = -0.4
                    else:
                        macd_score = 0.0
                    
                    score += macd_score * 0.3
                    components.append(('macd', macd_score))
            
            # Bollinger Bands
            if current_ohlcv and 'M15' in current_ohlcv:
                m15_data = current_ohlcv['M15']
                if len(m15_data) >= 20:
                    current_price = market_state.mid_price
                    bb_upper = m15_data['bb_upper'].iloc[-1]
                    bb_lower = m15_data['bb_lower'].iloc[-1]
                    bb_middle = m15_data['bb_middle'].iloc[-1]
                    
                    if current_price > bb_upper:
                        bb_score = -0.3  # Overbought
                    elif current_price < bb_lower:
                        bb_score = 0.3   # Oversold
                    else:
                        # Position within bands
                        bb_score = (current_price - bb_middle) / (bb_upper - bb_lower)
                    
                    score += bb_score * 0.2
                    components.append(('bollinger_bands', bb_score))
            
            # Support/Resistance levels
            if current_ohlcv and 'H4' in current_ohlcv:
                h4_data = current_ohlcv['H4']
                if len(h4_data) >= 50:
                    # Simple S/R detection
                    highs = h4_data['high'].rolling(window=10).max()
                    lows = h4_data['low'].rolling(window=10).min()
                    
                    current_price = market_state.mid_price
                    
                    # Check if near recent high/low
                    recent_high = highs.tail(20).max()
                    recent_low = lows.tail(20).min()
                    
                    if abs(current_price - recent_high) / recent_high < 0.001:
                        sr_score = -0.2  # Near resistance
                    elif abs(current_price - recent_low) / recent_low < 0.001:
                        sr_score = 0.2   # Near support
                    else:
                        sr_score = 0.0
                    
                    score += sr_score * 0.1
                    components.append(('support_resistance', sr_score))
            
            # Normalize to [-1, 1] range
            score = max(-1.0, min(1.0, score))
            
            return score
            
        except Exception as e:
            logger.error(f"Error calculating what score: {e}")
            return 0.0
    
    def _calculate_when_score(self, market_state) -> float:
        """Calculate timing/session analysis score"""
        try:
            score = 0.0
            components = []
            
            current_hour = market_state.timestamp.hour
            
            # Session analysis
            if self.sessions['asian'][0] <= current_hour < self.sessions['asian'][1]:
                session_score = 0.1  # Asian session
            elif self.sessions['london'][0] <= current_hour < self.sessions['london'][1]:
                session_score = 0.3  # London session
            elif self.sessions['new_york'][0] <= current_hour < self.sessions['new_york'][1]:
                session_score = 0.3  # New York session
            elif self.sessions['overlap'][0] <= current_hour < self.sessions['overlap'][1]:
                session_score = 0.5  # London-NY overlap (highest activity)
            else:
                session_score = -0.2  # Low activity hours
            
            score += session_score * 0.4
            components.append(('session', session_score))
            
            # Time of day analysis
            if 6 <= current_hour <= 18:
                time_score = 0.2  # Business hours
            else:
                time_score = -0.1  # Off hours
            
            score += time_score * 0.2
            components.append(('time_of_day', time_score))
            
            # Day of week analysis
            weekday = market_state.timestamp.weekday()
            if weekday < 5:  # Monday to Friday
                day_score = 0.1
            else:
                day_score = -0.3  # Weekend
            
            score += day_score * 0.2
            components.append(('day_of_week', day_score))
            
            # Normalize to [-1, 1] range
            score = max(-1.0, min(1.0, score))
            
            return score
            
        except Exception as e:
            logger.error(f"Error calculating when score: {e}")
            return 0.0
    
    def _calculate_anomaly_score(self, market_state) -> float:
        """Calculate anomaly detection score"""
        try:
            if not hasattr(self, 'anomaly_stats'):
                return 0.0
            
            score = 0.0
            components = []
            
            # Price anomaly
            price_z_score = abs(market_state.mid_price - self.anomaly_stats['price_mean']) / (self.anomaly_stats['price_std'] + 1e-6)
            if price_z_score > 3.0:
                price_anomaly = 0.5
            elif price_z_score > 2.0:
                price_anomaly = 0.2
            else:
                price_anomaly = 0.0
            
            score += price_anomaly * 0.3
            components.append(('price_anomaly', price_anomaly))
            
            # Volume anomaly
            current_volume = market_state.bid_volume + market_state.ask_volume
            volume_z_score = abs(current_volume - self.anomaly_stats['volume_mean']) / (self.anomaly_stats['volume_std'] + 1e-6)
            if volume_z_score > 3.0:
                volume_anomaly = 0.4
            elif volume_z_score > 2.0:
                volume_anomaly = 0.2
            else:
                volume_anomaly = 0.0
            
            score += volume_anomaly * 0.2
            components.append(('volume_anomaly', volume_anomaly))
            
            # Spread anomaly
            if market_state.spread_bps > 10.0:  # Very wide spread
                spread_anomaly = 0.3
            elif market_state.spread_bps < 0.5:  # Very tight spread
                spread_anomaly = 0.2
            else:
                spread_anomaly = 0.0
            
            score += spread_anomaly * 0.2
            components.append(('spread_anomaly', spread_anomaly))
            
            # Normalize to [0, 1] range (anomaly is always positive)
            score = max(0.0, min(1.0, score))
            
            return score
            
        except Exception as e:
            logger.error(f"Error calculating anomaly score: {e}")
            return 0.0
    
    def _calculate_confidence(self, market_state, current_ohlcv) -> float:
        """Calculate overall confidence in the sensory reading"""
        try:
            confidence = 0.5  # Base confidence
            
            # Data quality factors
            if current_ohlcv:
                confidence += 0.2  # Have recent data
            
            # Spread quality
            if market_state.spread_bps < 2.0:
                confidence += 0.1  # Tight spread = good data
            
            # Volume quality
            if market_state.bid_volume > 1000 and market_state.ask_volume > 1000:
                confidence += 0.1  # Good volume
            
            # Time factors
            current_hour = market_state.timestamp.hour
            if self.sessions['overlap'][0] <= current_hour < self.sessions['overlap'][1]:
                confidence += 0.1  # High activity session
            
            return min(1.0, confidence)
            
        except Exception as e:
            logger.error(f"Error calculating confidence: {e}")
            return 0.5
    
    def _get_why_components(self, market_state, current_ohlcv) -> Dict:
        """Get raw components for why score"""
        return {'trend': 0.0, 'momentum': 0.0, 'volume': 0.0}
    
    def _get_how_components(self, market_state, current_ohlcv) -> Dict:
        """Get raw components for how score"""
        return {'spread': 0.0, 'volume_profile': 0.0, 'range_expansion': 0.0}
    
    def _get_what_components(self, market_state, current_ohlcv) -> Dict:
        """Get raw components for what score"""
        return {'macd': 0.0, 'bollinger_bands': 0.0, 'support_resistance': 0.0}
    
    def _get_when_components(self, market_state) -> Dict:
        """Get raw components for when score"""
        return {'session': 0.0, 'time_of_day': 0.0, 'day_of_week': 0.0}
    
    def _get_anomaly_components(self, market_state) -> Dict:
        """Get raw components for anomaly score"""
        return {'price_anomaly': 0.0, 'volume_anomaly': 0.0, 'spread_anomaly': 0.0} 