"""
WHEN Dimension - Temporal Intelligence (Simplified for EMP)

This dimension analyzes timing and session dynamics:
- Session analysis (Asian, London, New York)
- Time-based patterns
- Event timing
- Cyclical patterns

Simplified version adapted for EMP trading system.
"""

import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple
from collections import deque

from ..core.base import DimensionalSensor, DimensionalReading, MarketData


class WhenDimension(DimensionalSensor):
    """
    WHEN Dimension - Understanding temporal dynamics
    
    Simplified version adapted for EMP trading system.
    """
    
    def __init__(self):
        super().__init__("WHEN")
        
        # Session definitions (UTC)
        self.sessions = {
            'ASIAN': (0, 8),      # 00:00-08:00 UTC
            'LONDON': (8, 16),    # 08:00-16:00 UTC
            'NEW_YORK': (13, 21), # 13:00-21:00 UTC
            'OVERLAP': [(8, 13), (16, 21)]  # London-New York overlaps
        }
        
        # Session characteristics
        self.session_characteristics = {
            'ASIAN': {'volatility': 0.3, 'trend': 0.0},
            'LONDON': {'volatility': 0.8, 'trend': 0.2},
            'NEW_YORK': {'volatility': 0.8, 'trend': 0.2},
            'OVERLAP': {'volatility': 1.0, 'trend': 0.4}
        }
        
    def calibrate(self, df) -> None:
        """Calibrate with historical data"""
        # WHEN dimension doesn't need complex calibration
        pass
    
    def process(self, data: MarketData, peer_readings: Dict[str, DimensionalReading]) -> DimensionalReading:
        """Process market data and return WHEN dimension reading"""
        
        try:
            # Analyze temporal factors
            
            # 1. Session analysis
            session_analysis = self._analyze_session(data.timestamp)
            
            # 2. Time of day analysis
            time_analysis = self._analyze_time_of_day(data.timestamp)
            
            # 3. Day of week analysis
            day_analysis = self._analyze_day_of_week(data.timestamp)
            
            # 4. Event timing (simulated)
            event_timing = self._analyze_event_timing(data.timestamp)
            
            # Combine factors
            when_score = (
                session_analysis * 0.4 +
                time_analysis * 0.3 +
                day_analysis * 0.2 +
                event_timing * 0.1
            )
            
            # Calculate confidence
            confidence = self._calculate_confidence(data, peer_readings)
            
            # Create context
            context = {
                'session_analysis': session_analysis,
                'time_analysis': time_analysis,
                'day_analysis': day_analysis,
                'event_timing': event_timing,
                'session_phase': self._get_current_session(data.timestamp),
                'session_volatility': self._get_session_volatility(data.timestamp),
                'day_of_week_score': day_analysis,
                'news_impact_score': event_timing
            }
            
            # Calculate peer influences
            influences = self._calculate_peer_influences(peer_readings)
            
            # Create reading
            reading = DimensionalReading(
                dimension=self.name,
                value=when_score,
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
            print(f"WHEN dimension processing error: {e}")
            return self._create_default_reading(data.timestamp)
    
    def _analyze_session(self, timestamp: datetime) -> float:
        """Analyze current session characteristics"""
        current_session = self._get_current_session(timestamp)
        characteristics = self.session_characteristics.get(current_session, {})
        
        # Session bias score
        session_bias = characteristics.get('trend', 0.0)
        
        # Add time within session factor
        hour = timestamp.hour
        if current_session == 'ASIAN':
            # Asian session: early hours more active
            time_factor = np.sin(2 * np.pi * hour / 8) * 0.2
        elif current_session == 'LONDON':
            # London session: opening and closing more active
            time_factor = np.sin(2 * np.pi * (hour - 8) / 8) * 0.3
        elif current_session == 'NEW_YORK':
            # New York session: opening more active
            time_factor = np.sin(2 * np.pi * (hour - 13) / 8) * 0.3
        else:
            time_factor = 0.0
        
        return max(-1.0, min(1.0, session_bias + time_factor))
    
    def _analyze_time_of_day(self, timestamp: datetime) -> float:
        """Analyze time of day patterns"""
        hour = timestamp.hour
        
        # Global forex patterns
        # Peak activity during major session opens
        if hour in [8, 14, 20]:  # Session opens
            return 0.5
        elif hour in [7, 13, 19]:  # Pre-session
            return 0.2
        elif hour in [9, 15, 21]:  # Post-open
            return 0.3
        else:
            return 0.0
    
    def _analyze_day_of_week(self, timestamp: datetime) -> float:
        """Analyze day of week patterns"""
        weekday = timestamp.weekday()
        
        # Monday: often gap opens, Friday: often profit taking
        if weekday == 0:  # Monday
            return 0.3
        elif weekday == 4:  # Friday
            return 0.2
        elif weekday == 6:  # Sunday
            return -0.1  # Lower activity
        else:
            return 0.0
    
    def _analyze_event_timing(self, timestamp: datetime) -> float:
        """Analyze event timing (simulated)"""
        # In real implementation, this would check economic calendar
        # For now, simulate based on time patterns
        
        hour = timestamp.hour
        minute = timestamp.minute
        
        # Simulate news events during certain hours
        if hour in [8, 13, 15] and minute < 30:
            return 0.3  # Potential news time
        else:
            return 0.0
    
    def _get_current_session(self, timestamp: datetime) -> str:
        """Get current trading session"""
        hour = timestamp.hour
        
        if 0 <= hour < 8:
            return 'ASIAN'
        elif 8 <= hour < 16:
            return 'LONDON'
        elif 13 <= hour < 21:
            return 'NEW_YORK'
        else:
            return 'OVERLAP'
    
    def _get_session_volatility(self, timestamp: datetime) -> float:
        """Get current session volatility"""
        current_session = self._get_current_session(timestamp)
        characteristics = self.session_characteristics.get(current_session, {})
        return characteristics.get('volatility', 0.5)
    
    def _calculate_confidence(self, data: MarketData, peer_readings: Dict[str, DimensionalReading]) -> float:
        """Calculate confidence in the reading"""
        base_confidence = 0.7  # Time-based analysis is generally reliable
        
        # Boost confidence during major session opens
        hour = data.timestamp.hour
        if hour in [8, 14, 20]:
            base_confidence += 0.1
        
        # Reduce confidence if anomalies detected
        if 'anomaly' in peer_readings:
            anomaly_reading = peer_readings['anomaly']
            if anomaly_reading.value > 0.5:
                base_confidence -= 0.1
        
        return max(0.1, min(1.0, base_confidence))
    
    def _calculate_peer_influences(self, peer_readings: Dict[str, DimensionalReading]) -> Dict[str, float]:
        """Calculate how other dimensions influence this one"""
        influences = {}
        
        # WHY dimension can affect timing interpretation
        if 'why' in peer_readings:
            why_reading = peer_readings['why']
            influences['why'] = why_reading.value * 0.1
        
        # HOW dimension can show institutional timing patterns
        if 'how' in peer_readings:
            how_reading = peer_readings['how']
            influences['how'] = how_reading.value * 0.2
        
        # WHAT dimension can show technical timing
        if 'what' in peer_readings:
            what_reading = peer_readings['what']
            influences['what'] = what_reading.value * 0.1
        
        # ANOMALY dimension can indicate unusual timing
        if 'anomaly' in peer_readings:
            anomaly_reading = peer_readings['anomaly']
            influences['anomaly'] = anomaly_reading.value * 0.1
        
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

