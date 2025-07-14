"""
ANOMALY Dimension - Chaos and Manipulation Detection (Simplified for EMP)

This dimension detects market anomalies and manipulation:
- Unusual price movements
- Volume anomalies
- Spread anomalies
- Manipulation patterns

Simplified version adapted for EMP trading system.
"""

import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple
from collections import deque

from ..core.base import DimensionalSensor, DimensionalReading, MarketData


class AnomalyDimension(DimensionalSensor):
    """
    ANOMALY Dimension - Detecting market anomalies and manipulation
    
    Simplified version adapted for EMP trading system.
    """
    
    def __init__(self):
        super().__init__("ANOMALY")
        
        # Calibration data
        self.calibrated = False
        self.price_baseline = 0.0
        self.volatility_baseline = 0.0
        self.volume_baseline = 0.0
        self.spread_baseline = 0.0
        
        # Anomaly tracking
        self.recent_prices = deque(maxlen=50)
        self.recent_volumes = deque(maxlen=50)
        self.recent_spreads = deque(maxlen=50)
        
    def calibrate(self, df) -> None:
        """Calibrate with historical data"""
        if df is None or df.empty:
            return
            
        try:
            # Calculate baselines
            self.price_baseline = df['close'].mean()
            self.volatility_baseline = df['close'].pct_change().std()
            self.volume_baseline = df['volume'].mean() if 'volume' in df.columns else 1000
            
            # Calculate spread baseline (simulated)
            if 'bid' in df.columns and 'ask' in df.columns:
                spreads = (df['ask'] - df['bid']) / df['bid']
                self.spread_baseline = spreads.mean()
            else:
                self.spread_baseline = 0.0001  # 1 pip default
            
            # Validate baselines
            if (self.price_baseline > 0 and 
                self.volatility_baseline > 0 and 
                self.volume_baseline > 0 and
                self.spread_baseline > 0):
                self.calibrated = True
                
        except Exception as e:
            print(f"ANOMALY dimension calibration failed: {e}")
    
    def process(self, data: MarketData, peer_readings: Dict[str, DimensionalReading]) -> DimensionalReading:
        """Process market data and return ANOMALY dimension reading"""
        
        if not self.calibrated:
            return self._create_default_reading(data.timestamp)
        
        try:
            # Update recent data
            self.recent_prices.append(data.mid_price)
            self.recent_volumes.append(data.volume)
            self.recent_spreads.append(data.spread_bps)
            
            # Analyze anomalies
            
            # 1. Price anomalies
            price_anomaly = self._detect_price_anomalies(data)
            
            # 2. Volume anomalies
            volume_anomaly = self._detect_volume_anomalies(data)
            
            # 3. Spread anomalies
            spread_anomaly = self._detect_spread_anomalies(data)
            
            # 4. Manipulation patterns
            manipulation = self._detect_manipulation_patterns(data)
            
            # Combine factors
            anomaly_score = (
                price_anomaly * 0.4 +
                volume_anomaly * 0.3 +
                spread_anomaly * 0.2 +
                manipulation * 0.1
            )
            
            # Calculate confidence
            confidence = self._calculate_confidence(data, peer_readings)
            
            # Create context
            context = {
                'price_anomaly': price_anomaly,
                'volume_anomaly': volume_anomaly,
                'spread_anomaly': spread_anomaly,
                'manipulation': manipulation,
                'stop_hunt_probability': self._calculate_stop_hunt_probability(data),
                'spoofing_detected': self._detect_spoofing(data),
                'liquidity_activity': self._analyze_liquidity_activity(data),
                'unusual_volume': volume_anomaly
            }
            
            # Calculate peer influences
            influences = self._calculate_peer_influences(peer_readings)
            
            # Create reading
            reading = DimensionalReading(
                dimension=self.name,
                value=anomaly_score,
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
            print(f"ANOMALY dimension processing error: {e}")
            return self._create_default_reading(data.timestamp)
    
    def _detect_price_anomalies(self, data: MarketData) -> float:
        """Detect unusual price movements"""
        if len(self.recent_prices) < 10:
            return 0.0
        
        current_price = data.mid_price
        recent_prices = list(self.recent_prices)[-10:]
        
        # Calculate price change
        if len(recent_prices) > 1:
            price_change = abs(current_price - recent_prices[-2]) / recent_prices[-2]
        else:
            price_change = 0.0
        
        # Compare to baseline volatility
        if self.volatility_baseline > 0:
            volatility_ratio = price_change / self.volatility_baseline
        else:
            volatility_ratio = 0.0
        
        # Detect anomalies
        if volatility_ratio > 3.0:  # 3x normal volatility
            return min(1.0, volatility_ratio / 5.0)
        elif volatility_ratio > 2.0:  # 2x normal volatility
            return 0.5
        else:
            return 0.0
    
    def _detect_volume_anomalies(self, data: MarketData) -> float:
        """Detect unusual volume patterns"""
        if len(self.recent_volumes) < 10:
            return 0.0
        
        current_volume = data.volume
        recent_volumes = list(self.recent_volumes)[-10:]
        avg_volume = np.mean(recent_volumes[:-1]) if len(recent_volumes) > 1 else current_volume
        
        if avg_volume > 0:
            volume_ratio = current_volume / avg_volume
        else:
            volume_ratio = 1.0
        
        # Detect volume anomalies
        if volume_ratio > 3.0:  # 3x normal volume
            return min(1.0, volume_ratio / 5.0)
        elif volume_ratio > 2.0:  # 2x normal volume
            return 0.5
        elif volume_ratio < 0.3:  # Very low volume
            return 0.3
        else:
            return 0.0
    
    def _detect_spread_anomalies(self, data: MarketData) -> float:
        """Detect unusual spread patterns"""
        if len(self.recent_spreads) < 10:
            return 0.0
        
        current_spread = data.spread_bps
        recent_spreads = list(self.recent_spreads)[-10:]
        avg_spread = np.mean(recent_spreads[:-1]) if len(recent_spreads) > 1 else current_spread
        
        if avg_spread > 0:
            spread_ratio = current_spread / avg_spread
        else:
            spread_ratio = 1.0
        
        # Detect spread anomalies
        if spread_ratio > 2.0:  # 2x normal spread
            return min(1.0, spread_ratio / 3.0)
        elif spread_ratio < 0.5:  # Unusually tight spread
            return 0.3
        else:
            return 0.0
    
    def _detect_manipulation_patterns(self, data: MarketData) -> float:
        """Detect potential manipulation patterns"""
        # Simulate manipulation detection
        # In real implementation, this would analyze:
        # - Stop hunting patterns
        # - Spoofing detection
        # - Wash trading
        # - Pump and dump patterns
        
        hour = data.timestamp.hour
        minute = data.timestamp.minute
        
        # Simulate manipulation during certain times
        manipulation_score = 0.0
        
        # Higher manipulation risk during low liquidity hours
        if hour in [2, 3, 4, 5]:  # Early Asian session
            manipulation_score += 0.2
        
        # Higher risk during specific minutes (simulating coordinated moves)
        if minute in [0, 15, 30, 45]:
            manipulation_score += 0.1
        
        # Volume-based manipulation detection
        if len(self.recent_volumes) >= 5:
            recent_volumes = list(self.recent_volumes)[-5:]
            volume_pattern = np.std(recent_volumes) / np.mean(recent_volumes) if np.mean(recent_volumes) > 0 else 0
            
            if volume_pattern > 1.0:  # Irregular volume pattern
                manipulation_score += 0.3
        
        return min(1.0, manipulation_score)
    
    def _calculate_stop_hunt_probability(self, data: MarketData) -> float:
        """Calculate probability of stop hunting"""
        # Simulate stop hunt detection
        # In real implementation, this would analyze:
        # - Price spikes near support/resistance
        # - Volume patterns
        # - Reversal patterns
        
        hour = data.timestamp.hour
        
        # Higher stop hunt probability during certain hours
        if hour in [8, 14, 20]:  # Session opens
            return 0.4
        elif hour in [7, 13, 19]:  # Pre-session
            return 0.3
        else:
            return 0.1
    
    def _detect_spoofing(self, data: MarketData) -> bool:
        """Detect potential spoofing activity"""
        # Simulate spoofing detection
        # In real implementation, this would analyze:
        # - Large orders that don't execute
        # - Order book manipulation
        # - Price impact vs order size
        
        # For now, return False (no spoofing detected)
        return False
    
    def _analyze_liquidity_activity(self, data: MarketData) -> float:
        """Analyze liquidity zone activity"""
        # Simulate liquidity analysis
        # In real implementation, this would analyze:
        # - Order book depth
        # - Liquidity pools
        # - Market maker activity
        
        # Simulate based on time and volume
        hour = data.timestamp.hour
        volume_ratio = data.volume / self.volume_baseline if self.volume_baseline > 0 else 1.0
        
        liquidity_score = 0.0
        
        # Higher liquidity activity during major sessions
        if hour in [8, 14, 20]:
            liquidity_score += 0.3
        
        # Volume-based liquidity activity
        if volume_ratio > 1.5:
            liquidity_score += 0.2
        
        return min(1.0, liquidity_score)
    
    def _calculate_confidence(self, data: MarketData, peer_readings: Dict[str, DimensionalReading]) -> float:
        """Calculate confidence in the reading"""
        base_confidence = 0.6
        
        # Boost confidence with more data
        if len(self.recent_prices) >= 20:
            base_confidence += 0.1
        
        # Boost confidence if multiple anomalies detected
        anomaly_count = 0
        if self._detect_price_anomalies(data) > 0.3:
            anomaly_count += 1
        if self._detect_volume_anomalies(data) > 0.3:
            anomaly_count += 1
        if self._detect_spread_anomalies(data) > 0.3:
            anomaly_count += 1
        
        if anomaly_count >= 2:
            base_confidence += 0.1
        
        return max(0.1, min(1.0, base_confidence))
    
    def _calculate_peer_influences(self, peer_readings: Dict[str, DimensionalReading]) -> Dict[str, float]:
        """Calculate how other dimensions influence this one"""
        influences = {}
        
        # WHY dimension can indicate fundamental anomalies
        if 'why' in peer_readings:
            why_reading = peer_readings['why']
            influences['why'] = why_reading.value * 0.1
        
        # HOW dimension can show institutional anomalies
        if 'how' in peer_readings:
            how_reading = peer_readings['how']
            influences['how'] = how_reading.value * 0.2
        
        # WHAT dimension can show technical anomalies
        if 'what' in peer_readings:
            what_reading = peer_readings['what']
            influences['what'] = what_reading.value * 0.1
        
        # WHEN dimension can show timing anomalies
        if 'when' in peer_readings:
            when_reading = peer_readings['when']
            influences['when'] = when_reading.value * 0.1
        
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

