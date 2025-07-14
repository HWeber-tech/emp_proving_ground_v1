"""
HOW Dimension - Institutional Mechanics (Simplified for EMP)

This dimension analyzes institutional behavior and execution patterns:
- Order flow analysis
- Volume profile analysis
- Institutional footprint detection
- Market maker vs. taker behavior

Simplified version adapted for EMP trading system.
"""

import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple
from collections import deque
import pandas as pd

from ..core.base import DimensionalSensor, DimensionalReading, MarketData


class HowDimension(DimensionalSensor):
    """
    HOW Dimension - Understanding institutional mechanics
    
    Simplified version adapted for EMP trading system.
    """
    
    def __init__(self):
        super().__init__("HOW")
        
        # Calibration data
        self.calibrated = False
        self.volume_baseline = 0.0
        self.spread_baseline = 0.0
        self.price_baseline = 0.0
        
        # Order flow tracking
        self.recent_volumes = deque(maxlen=20)
        self.recent_spreads = deque(maxlen=20)
        
    def calibrate(self, df: pd.DataFrame) -> None:
        """Calibrate with historical data"""
        if df is None or df.empty:
            return
            
        try:
            # Calculate baselines
            self.volume_baseline = df['volume'].mean() if 'volume' in df.columns else 1000
            self.price_baseline = df['close'].mean()
            
            # Calculate spread baseline (simulated)
            if 'bid' in df.columns and 'ask' in df.columns:
                spreads = (df['ask'] - df['bid']) / df['bid']
                self.spread_baseline = spreads.mean()
            else:
                self.spread_baseline = 0.0001  # 1 pip default
            
            # Validate baselines
            if (self.volume_baseline > 0 and 
                self.price_baseline > 0 and 
                self.spread_baseline > 0):
                self.calibrated = True
                
        except Exception as e:
            print(f"HOW dimension calibration failed: {e}")
    
    def process(self, data: MarketData, peer_readings: Dict[str, DimensionalReading]) -> DimensionalReading:
        """Process market data and return HOW dimension reading"""
        
        if not self.calibrated:
            return self._create_default_reading(data.timestamp)
        
        try:
            # Update recent data
            self.recent_volumes.append(data.volume)
            self.recent_spreads.append(data.spread_bps)
            
            # Analyze institutional behavior
            
            # 1. Order flow bias
            order_flow_bias = self._calculate_order_flow_bias(data)
            
            # 2. Volume analysis
            volume_analysis = self._analyze_volume_patterns(data)
            
            # 3. Spread analysis
            spread_analysis = self._analyze_spread_patterns(data)
            
            # 4. Institutional footprint
            institutional_footprint = self._detect_institutional_footprint(data)
            
            # Combine factors
            how_score = (
                order_flow_bias * 0.4 +
                volume_analysis * 0.3 +
                spread_analysis * 0.2 +
                institutional_footprint * 0.1
            )
            
            # Calculate confidence
            confidence = self._calculate_confidence(data, peer_readings)
            
            # Create context
            context = {
                'order_flow_bias': order_flow_bias,
                'volume_analysis': volume_analysis,
                'spread_analysis': spread_analysis,
                'institutional_footprint': institutional_footprint,
                'dominant_pattern': self._identify_dominant_pattern(order_flow_bias, volume_analysis)
            }
            
            # Calculate peer influences
            influences = self._calculate_peer_influences(peer_readings)
            
            # Create reading
            reading = DimensionalReading(
                dimension=self.name,
                value=how_score,
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
            print(f"HOW dimension processing error: {e}")
            return self._create_default_reading(data.timestamp)
    
    def _calculate_order_flow_bias(self, data: MarketData) -> float:
        """Calculate order flow bias (buying vs selling pressure)"""
        if self.volume_baseline <= 0:
            return 0.0
        
        # Simulate order flow bias based on price movement and volume
        volume_ratio = data.volume / self.volume_baseline
        
        # Higher volume with price increase suggests buying pressure
        # Higher volume with price decrease suggests selling pressure
        # This is a simplified simulation - real implementation would use actual order book data
        
        # Simulate based on time and volatility
        hour = data.timestamp.hour
        time_factor = np.sin(2 * np.pi * hour / 24) * 0.3
        
        # Volume factor
        volume_factor = (volume_ratio - 1.0) * 0.2
        
        # Spread factor (tighter spreads suggest more activity)
        spread_factor = -(data.spread_bps / max(self.spread_baseline * 10000, 1)) * 0.1
        
        return max(-1.0, min(1.0, time_factor + volume_factor + spread_factor))
    
    def _analyze_volume_patterns(self, data: MarketData) -> float:
        """Analyze volume patterns for institutional activity"""
        if len(self.recent_volumes) < 5:
            return 0.0
        
        # Calculate volume trend
        recent_volumes = list(self.recent_volumes)
        volume_trend = np.polyfit(range(len(recent_volumes)), recent_volumes, 1)[0]
        
        # Normalize trend
        avg_volume = np.mean(recent_volumes)
        if avg_volume > 0:
            normalized_trend = volume_trend / avg_volume
        else:
            normalized_trend = 0.0
        
        # Volume spike detection
        current_volume = data.volume
        avg_recent_volume = np.mean(recent_volumes[:-1]) if len(recent_volumes) > 1 else current_volume
        
        if avg_recent_volume > 0:
            volume_spike = (current_volume - avg_recent_volume) / avg_recent_volume
        else:
            volume_spike = 0.0
        
        # Combine trend and spike
        volume_score = normalized_trend * 0.6 + volume_spike * 0.4
        
        return max(-1.0, min(1.0, volume_score))
    
    def _analyze_spread_patterns(self, data: MarketData) -> float:
        """Analyze spread patterns for liquidity and activity"""
        if len(self.recent_spreads) < 5:
            return 0.0
        
        # Tighter spreads suggest more activity/liquidity
        current_spread = data.spread_bps
        avg_spread = np.mean(self.recent_spreads)
        
        if avg_spread > 0:
            spread_ratio = current_spread / avg_spread
            # Invert: tighter spreads = positive score
            spread_score = -(spread_ratio - 1.0)
        else:
            spread_score = 0.0
        
        return max(-1.0, min(1.0, spread_score))
    
    def _detect_institutional_footprint(self, data: MarketData) -> float:
        """Detect institutional trading patterns"""
        # Simulate institutional footprint detection
        # In real implementation, this would analyze:
        # - Large order blocks
        # - VWAP hugging
        # - Time-based patterns
        # - Cross-market correlations
        
        hour = data.timestamp.hour
        minute = data.timestamp.minute
        
        # Simulate institutional activity during certain times
        institutional_score = 0.0
        
        # Higher activity during major session opens
        if hour in [8, 14, 20] and minute < 30:
            institutional_score += 0.3
        
        # Higher activity during first/last hour of sessions
        if hour in [9, 15, 21] or hour in [7, 13, 19]:
            institutional_score += 0.2
        
        # Volume-based institutional detection
        volume_ratio = data.volume / self.volume_baseline
        if volume_ratio > 2.0:
            institutional_score += 0.3
        
        return max(-1.0, min(1.0, institutional_score))
    
    def _calculate_confidence(self, data: MarketData, peer_readings: Dict[str, DimensionalReading]) -> float:
        """Calculate confidence in the reading"""
        base_confidence = 0.6
        
        # Boost confidence with good data quality
        if data.volume > 0 and data.spread_bps > 0:
            base_confidence += 0.1
        
        # Boost confidence if volume is significant
        if data.volume > self.volume_baseline * 1.5:
            base_confidence += 0.1
        
        # Reduce confidence if anomalies detected
        if 'anomaly' in peer_readings:
            anomaly_reading = peer_readings['anomaly']
            if anomaly_reading.value > 0.5:
                base_confidence -= 0.2
        
        return max(0.1, min(1.0, base_confidence))
    
    def _identify_dominant_pattern(self, order_flow: float, volume: float) -> str:
        """Identify the dominant institutional pattern"""
        if abs(order_flow) > 0.5:
            if order_flow > 0:
                return "aggressive_buying"
            else:
                return "aggressive_selling"
        elif abs(volume) > 0.5:
            if volume > 0:
                return "high_volume_activity"
            else:
                return "low_volume_activity"
        else:
            return "balanced_flow"
    
    def _calculate_peer_influences(self, peer_readings: Dict[str, DimensionalReading]) -> Dict[str, float]:
        """Calculate how other dimensions influence this one"""
        influences = {}
        
        # WHY dimension can indicate fundamental shifts affecting institutions
        if 'why' in peer_readings:
            why_reading = peer_readings['why']
            influences['why'] = why_reading.value * 0.2
        
        # WHAT dimension can show technical levels attracting institutional activity
        if 'what' in peer_readings:
            what_reading = peer_readings['what']
            influences['what'] = what_reading.value * 0.3
        
        # WHEN dimension can indicate timing-based institutional patterns
        if 'when' in peer_readings:
            when_reading = peer_readings['when']
            influences['when'] = when_reading.value * 0.2
        
        # ANOMALY dimension can indicate manipulation affecting institutional behavior
        if 'anomaly' in peer_readings:
            anomaly_reading = peer_readings['anomaly']
            influences['anomaly'] = -anomaly_reading.value * 0.3  # Negative influence
        
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

