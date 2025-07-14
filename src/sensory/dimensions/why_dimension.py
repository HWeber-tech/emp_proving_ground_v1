"""
WHY Dimension - Fundamental Market Forces (Simplified for EMP)

This dimension seeks to understand the fundamental forces driving market behavior:
- Economic momentum and cycles
- Central bank policy and monetary flows
- Risk sentiment and capital allocation
- Geopolitical events and structural changes

Simplified version adapted for EMP trading system.
"""

import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple
from collections import deque
from dataclasses import dataclass
import pandas as pd

from ..core.base import DimensionalSensor, DimensionalReading, MarketData


class WhyDimension(DimensionalSensor):
    """
    WHY Dimension - Understanding fundamental market forces
    
    Simplified version adapted for EMP trading system.
    """
    
    def __init__(self):
        super().__init__("WHY")
        
        # Calibration data
        self.calibrated = False
        self.price_baseline = 0.0
        self.volatility_baseline = 0.0
        self.volume_baseline = 0.0
        
        # Economic indicators (simulated)
        self.economic_momentum = 0.0
        self.policy_divergence = 0.0
        self.risk_sentiment = 0.0
        
    def calibrate(self, df: pd.DataFrame) -> None:
        """Calibrate with historical data"""
        if df is None or df.empty:
            return
            
        try:
            # Calculate baselines
            self.price_baseline = df['close'].mean()
            self.volatility_baseline = df['close'].pct_change().std()
            self.volume_baseline = df['volume'].mean() if 'volume' in df.columns else 1000
            
            # Validate baselines
            if (self.price_baseline > 0 and 
                self.volatility_baseline > 0 and 
                self.volume_baseline > 0):
                self.calibrated = True
                
        except Exception as e:
            print(f"WHY dimension calibration failed: {e}")
    
    def process(self, data: MarketData, peer_readings: Dict[str, DimensionalReading]) -> DimensionalReading:
        """Process market data and return WHY dimension reading"""
        
        if not self.calibrated:
            return self._create_default_reading(data.timestamp)
        
        try:
            # Analyze fundamental factors (simplified)
            
            # 1. Economic momentum from price action
            price_momentum = self._calculate_price_momentum(data)
            
            # 2. Policy divergence (simulated)
            policy_divergence = self._simulate_policy_divergence(data)
            
            # 3. Risk sentiment from volatility
            risk_sentiment = self._calculate_risk_sentiment(data)
            
            # 4. Geopolitical impact (simulated)
            geopolitical_impact = self._simulate_geopolitical_impact(data)
            
            # Combine factors
            why_score = (
                price_momentum * 0.4 +
                policy_divergence * 0.3 +
                risk_sentiment * 0.2 +
                geopolitical_impact * 0.1
            )
            
            # Calculate confidence
            confidence = self._calculate_confidence(data, peer_readings)
            
            # Create context
            context = {
                'economic_momentum': price_momentum,
                'policy_divergence': policy_divergence,
                'risk_sentiment': risk_sentiment,
                'geopolitical_impact': geopolitical_impact,
                'primary_driver': self._identify_primary_driver(price_momentum, policy_divergence, risk_sentiment)
            }
            
            # Calculate peer influences
            influences = self._calculate_peer_influences(peer_readings)
            
            # Create reading
            reading = DimensionalReading(
                dimension=self.name,
                value=why_score,
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
            print(f"WHY dimension processing error: {e}")
            return self._create_default_reading(data.timestamp)
    
    def _calculate_price_momentum(self, data: MarketData) -> float:
        """Calculate economic momentum from price action"""
        if self.price_baseline <= 0:
            return 0.0
        
        # Simple momentum based on current price vs baseline
        price_ratio = data.mid_price / self.price_baseline
        momentum = (price_ratio - 1.0) * 10  # Scale to [-1, 1]
        return max(-1.0, min(1.0, momentum))
    
    def _simulate_policy_divergence(self, data: MarketData) -> float:
        """Simulate central bank policy divergence"""
        # In a real implementation, this would use actual policy data
        # For now, simulate based on time and volatility
        hour = data.timestamp.hour
        volatility_factor = data.volatility / max(self.volatility_baseline, 0.001)
        
        # Simulate policy cycles
        policy_cycle = np.sin(2 * np.pi * hour / 24) * 0.3
        volatility_effect = (volatility_factor - 1.0) * 0.2
        
        return max(-1.0, min(1.0, policy_cycle + volatility_effect))
    
    def _calculate_risk_sentiment(self, data: MarketData) -> float:
        """Calculate risk sentiment from volatility"""
        if self.volatility_baseline <= 0:
            return 0.0
        
        # Higher volatility = risk-off sentiment
        vol_ratio = data.volatility / self.volatility_baseline
        risk_sentiment = -(vol_ratio - 1.0)  # Invert: high vol = risk-off
        
        return max(-1.0, min(1.0, risk_sentiment))
    
    def _simulate_geopolitical_impact(self, data: MarketData) -> float:
        """Simulate geopolitical impact"""
        # In a real implementation, this would use news feeds
        # For now, simulate random geopolitical events
        hour = data.timestamp.hour
        day = data.timestamp.weekday()
        
        # Simulate higher impact during certain hours/days
        base_impact = 0.0
        if hour in [8, 14, 20]:  # Major session opens
            base_impact = 0.1
        if day in [0, 4]:  # Monday, Friday
            base_impact += 0.05
        
        # Add some randomness
        random_factor = np.random.normal(0, 0.1)
        
        return max(-1.0, min(1.0, base_impact + random_factor))
    
    def _calculate_confidence(self, data: MarketData, peer_readings: Dict[str, DimensionalReading]) -> float:
        """Calculate confidence in the reading"""
        base_confidence = 0.6
        
        # Boost confidence if data quality is good
        if data.volatility > 0 and data.volume > 0:
            base_confidence += 0.1
        
        # Reduce confidence if anomalies detected
        if 'anomaly' in peer_readings:
            anomaly_reading = peer_readings['anomaly']
            if anomaly_reading.value > 0.5:
                base_confidence -= 0.2
        
        # Boost confidence if technical signals align
        if 'what' in peer_readings:
            what_reading = peer_readings['what']
            if abs(what_reading.value) > 0.5:
                base_confidence += 0.1
        
        return max(0.1, min(1.0, base_confidence))
    
    def _identify_primary_driver(self, momentum: float, policy: float, sentiment: float) -> str:
        """Identify the primary fundamental driver"""
        drivers = [
            ('economic_momentum', abs(momentum)),
            ('policy_divergence', abs(policy)),
            ('risk_sentiment', abs(sentiment))
        ]
        
        primary_driver = max(drivers, key=lambda x: x[1])
        
        if primary_driver[1] < 0.3:
            return "mixed_fundamentals"
        else:
            return primary_driver[0]
    
    def _calculate_peer_influences(self, peer_readings: Dict[str, DimensionalReading]) -> Dict[str, float]:
        """Calculate how other dimensions influence this one"""
        influences = {}
        
        # HOW dimension can indicate fundamental shifts
        if 'how' in peer_readings:
            how_reading = peer_readings['how']
            influences['how'] = how_reading.value * 0.3
        
        # WHAT dimension can confirm fundamental trends
        if 'what' in peer_readings:
            what_reading = peer_readings['what']
            influences['what'] = what_reading.value * 0.2
        
        # ANOMALY dimension can override fundamental analysis
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

