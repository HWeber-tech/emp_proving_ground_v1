"""
WHAT Dimension - Technical Analysis (Simplified for EMP)

This dimension analyzes technical patterns and price action:
- Market structure analysis
- Support and resistance levels
- Momentum and trend analysis
- Volatility patterns

Simplified version adapted for EMP trading system.
"""

import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple
from collections import deque
import pandas as pd

from ..core.base import DimensionalSensor, DimensionalReading, MarketData


class WhatDimension(DimensionalSensor):
    """
    WHAT Dimension - Understanding technical patterns
    
    Simplified version adapted for EMP trading system.
    """
    
    def __init__(self):
        super().__init__("WHAT")
        
        # Calibration data
        self.calibrated = False
        self.price_baseline = 0.0
        self.volatility_baseline = 0.0
        
        # Technical analysis parameters
        self.ema_periods = [9, 21, 50, 200]
        self.rsi_period = 14
        self.bb_period = 20
        self.bb_std = 2
        
        # Price history
        self.price_history = deque(maxlen=200)
        
    def calibrate(self, df: pd.DataFrame) -> None:
        """Calibrate with historical data"""
        if df is None or df.empty:
            return
            
        try:
            # Calculate baselines
            self.price_baseline = df['close'].mean()
            self.volatility_baseline = df['close'].pct_change().std()
            
            # Validate baselines
            if self.price_baseline > 0 and self.volatility_baseline > 0:
                self.calibrated = True
                
        except Exception as e:
            print(f"WHAT dimension calibration failed: {e}")
    
    def process(self, data: MarketData, peer_readings: Dict[str, DimensionalReading]) -> DimensionalReading:
        """Process market data and return WHAT dimension reading"""
        
        if not self.calibrated:
            return self._create_default_reading(data.timestamp)
        
        try:
            # Update price history
            self.price_history.append({
                'timestamp': data.timestamp,
                'open': data.open,
                'high': data.high,
                'low': data.low,
                'close': data.close,
                'volume': data.volume
            })
            
            # Analyze technical patterns
            
            # 1. Trend analysis
            trend_analysis = self._analyze_trend(data)
            
            # 2. Support/Resistance analysis
            support_resistance = self._analyze_support_resistance(data)
            
            # 3. Momentum analysis
            momentum_analysis = self._analyze_momentum(data)
            
            # 4. Volatility analysis
            volatility_analysis = self._analyze_volatility(data)
            
            # Combine factors
            what_score = (
                trend_analysis * 0.4 +
                support_resistance * 0.3 +
                momentum_analysis * 0.2 +
                volatility_analysis * 0.1
            )
            
            # Calculate confidence
            confidence = self._calculate_confidence(data, peer_readings)
            
            # Create context
            context = {
                'trend_analysis': trend_analysis,
                'support_resistance': support_resistance,
                'momentum_analysis': momentum_analysis,
                'volatility_analysis': volatility_analysis,
                'primary_pattern': self._identify_primary_pattern(trend_analysis, support_resistance),
                'support_level': self._calculate_support_level(),
                'resistance_level': self._calculate_resistance_level(),
                'volatility_score': abs(volatility_analysis)
            }
            
            # Calculate peer influences
            influences = self._calculate_peer_influences(peer_readings)
            
            # Create reading
            reading = DimensionalReading(
                dimension=self.name,
                value=what_score,
                confidence=confidence,
                timestamp=data.timestamp,
                context=context,
                influences=influences
            )
            
            # Store in history
            with self._lock:
                self.history.append(reading)
            
            return reading
            
        except Exception as e:
            print(f"WHAT dimension processing error: {e}")
            return self._create_default_reading(data.timestamp)
    
    def _analyze_trend(self, data: MarketData) -> float:
        """Analyze trend direction and strength"""
        if len(self.price_history) < 20:
            return 0.0
        
        # Calculate EMAs
        closes = [p['close'] for p in self.price_history]
        ema_signals = []
        
        for period in self.ema_periods:
            if len(closes) >= period:
                ema = self._calculate_ema(closes, period)
                current_price = data.close
                ema_signals.append(1 if current_price > ema else -1)
        
        if not ema_signals:
            return 0.0
        
        # Trend score based on EMA alignment
        trend_score = np.mean(ema_signals)
        
        # Add trend strength based on price movement
        if len(closes) >= 20:
            recent_trend = (closes[-1] - closes[-20]) / closes[-20]
            trend_strength = min(abs(recent_trend) * 10, 1.0)
            trend_score *= trend_strength
        
        return max(-1.0, min(1.0, trend_score))
    
    def _analyze_support_resistance(self, data: MarketData) -> float:
        """Analyze support and resistance levels"""
        if len(self.price_history) < 20:
            return 0.0
        
        # Find recent highs and lows
        recent_highs = [p['high'] for p in list(self.price_history)[-20:]]
        recent_lows = [p['low'] for p in list(self.price_history)[-20:]]
        
        current_price = data.close
        
        # Calculate distance to nearest support/resistance
        nearest_high = min(recent_highs, key=lambda x: abs(x - current_price))
        nearest_low = min(recent_lows, key=lambda x: abs(x - current_price))
        
        # Position relative to support/resistance
        high_distance = (nearest_high - current_price) / current_price
        low_distance = (current_price - nearest_low) / current_price
        
        # Score based on position
        if high_distance < 0.001:  # Near resistance
            return -0.5
        elif low_distance < 0.001:  # Near support
            return 0.5
        else:
            # Neutral position
            return 0.0
    
    def _analyze_momentum(self, data: MarketData) -> float:
        """Analyze momentum indicators"""
        if len(self.price_history) < self.rsi_period:
            return 0.0
        
        # Calculate RSI
        closes = [p['close'] for p in self.price_history]
        rsi = self._calculate_rsi(closes, self.rsi_period)
        
        # Convert RSI to momentum score
        if rsi > 70:
            momentum_score = -0.5  # Overbought
        elif rsi < 30:
            momentum_score = 0.5   # Oversold
        else:
            momentum_score = (rsi - 50) / 50  # Normalized to [-1, 1]
        
        return max(-1.0, min(1.0, momentum_score))
    
    def _analyze_volatility(self, data: MarketData) -> float:
        """Analyze volatility patterns"""
        if self.volatility_baseline <= 0:
            return 0.0
        
        # Calculate current volatility
        if len(self.price_history) >= 20:
            recent_closes = [p['close'] for p in list(self.price_history)[-20:]]
            current_volatility = np.std(np.diff(recent_closes) / recent_closes[:-1])
        else:
            current_volatility = self.volatility_baseline
        
        # Compare to baseline
        vol_ratio = current_volatility / self.volatility_baseline
        
        # High volatility can be bullish or bearish depending on context
        # For now, return neutral for high volatility
        if vol_ratio > 1.5:
            return 0.0  # High volatility - neutral
        elif vol_ratio < 0.7:
            return 0.0  # Low volatility - neutral
        else:
            return 0.0  # Normal volatility - neutral
    
    def _calculate_ema(self, prices: List[float], period: int) -> float:
        """Calculate Exponential Moving Average"""
        if len(prices) < period:
            return prices[-1]
        
        alpha = 2.0 / (period + 1)
        ema = prices[0]
        
        for price in prices[1:]:
            ema = alpha * price + (1 - alpha) * ema
        
        return ema
    
    def _calculate_rsi(self, prices: List[float], period: int) -> float:
        """Calculate Relative Strength Index"""
        if len(prices) < period + 1:
            return 50.0
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return float(rsi)
    
    def _calculate_support_level(self) -> Optional[float]:
        """Calculate nearest support level"""
        if len(self.price_history) < 10:
            return None
        
        recent_lows = [p['low'] for p in list(self.price_history)[-10:]]
        return min(recent_lows)
    
    def _calculate_resistance_level(self) -> Optional[float]:
        """Calculate nearest resistance level"""
        if len(self.price_history) < 10:
            return None
        
        recent_highs = [p['high'] for p in list(self.price_history)[-10:]]
        return max(recent_highs)
    
    def _calculate_confidence(self, data: MarketData, peer_readings: Dict[str, DimensionalReading]) -> float:
        """Calculate confidence in the reading"""
        base_confidence = 0.6
        
        # Boost confidence with sufficient data
        if len(self.price_history) >= 50:
            base_confidence += 0.1
        
        # Boost confidence if trend is strong
        if abs(self._analyze_trend(data)) > 0.5:
            base_confidence += 0.1
        
        # Reduce confidence if anomalies detected
        if 'anomaly' in peer_readings:
            anomaly_reading = peer_readings['anomaly']
            if anomaly_reading.value > 0.5:
                base_confidence -= 0.2
        
        return max(0.1, min(1.0, base_confidence))
    
    def _identify_primary_pattern(self, trend: float, support_resistance: float) -> str:
        """Identify the primary technical pattern"""
        if abs(trend) > 0.5:
            if trend > 0:
                return "strong_uptrend"
            else:
                return "strong_downtrend"
        elif abs(support_resistance) > 0.3:
            if support_resistance > 0:
                return "support_bounce"
            else:
                return "resistance_rejection"
        else:
            return "consolidation"
    
    def _calculate_peer_influences(self, peer_readings: Dict[str, DimensionalReading]) -> Dict[str, float]:
        """Calculate how other dimensions influence this one"""
        influences = {}
        
        # WHY dimension can confirm or contradict technical signals
        if 'why' in peer_readings:
            why_reading = peer_readings['why']
            influences['why'] = why_reading.value * 0.2
        
        # HOW dimension can show institutional activity at technical levels
        if 'how' in peer_readings:
            how_reading = peer_readings['how']
            influences['how'] = how_reading.value * 0.3
        
        # WHEN dimension can affect technical interpretation
        if 'when' in peer_readings:
            when_reading = peer_readings['when']
            influences['when'] = when_reading.value * 0.1
        
        # ANOMALY dimension can invalidate technical signals
        if 'anomaly' in peer_readings:
            anomaly_reading = peer_readings['anomaly']
            influences['anomaly'] = -anomaly_reading.value * 0.4  # Negative influence
        
        return influences
    
    def _create_default_reading(self, timestamp: datetime) -> DimensionalReading:
        """Create default reading when processing fails"""
        return DimensionalReading(
            dimension=self.name,
            value=0.0,
            confidence=0.3,
            timestamp=timestamp,
            context={'error': 'Default reading'},
            influences={}
        )

