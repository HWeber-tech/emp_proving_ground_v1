"""
Intelligence Engine - Orchestrated Dimensional Synthesis for EMP

This is the central orchestration engine that synthesizes all five dimensions
(WHY, HOW, WHAT, WHEN, ANOMALY) into unified market intelligence, adapted
for the EMP trading system.
"""

import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple, NamedTuple
from collections import deque, defaultdict
from dataclasses import dataclass
from enum import Enum, auto
import threading
import math
import pandas as pd
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

from ..core.base import (
    DimensionalSensor, DimensionalReading, MarketRegime,
    MarketNarrative, MemoryBank, DimensionalCorrelationMatrix, RegimeDetector
)
from ..dimensions.why_dimension import WhyDimension
from ..dimensions.how_dimension import HowDimension
from ..dimensions.what_dimension import WhatDimension
from ..dimensions.when_dimension import WhenDimension
from ..dimensions.anomaly_dimension import AnomalyDimension


class IntelligenceLevel(Enum):
    CONFUSED = auto()      # Conflicting signals, low confidence
    UNCERTAIN = auto()     # Some clarity but missing pieces
    AWARE = auto()         # Good understanding of current state
    INSIGHTFUL = auto()    # Deep understanding with predictive power
    PRESCIENT = auto()     # Exceptional understanding across all dimensions


@dataclass
class MarketUnderstanding:
    """Unified market understanding from all dimensions"""
    timestamp: datetime
    regime: MarketRegime
    confidence: float
    
    # Dimensional readings
    why_reading: DimensionalReading
    how_reading: DimensionalReading
    what_reading: DimensionalReading
    when_reading: DimensionalReading
    anomaly_reading: DimensionalReading
    
    # Synthesis
    consensus_direction: float  # -1 to 1
    consensus_strength: float   # 0 to 1
    dimensional_alignment: float  # How well dimensions agree
    narrative: str = ""
    
    # Additional context for EMP compatibility
    context: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.context is None:
            self.context = {}
    
    @property
    def all_readings(self) -> List[DimensionalReading]:
        return [self.why_reading, self.how_reading, self.what_reading,
                self.when_reading, self.anomaly_reading]


class IntelligenceEngine:
    """
    Central orchestration engine for multidimensional market intelligence.
    
    This engine synthesizes all five dimensions into unified market understanding,
    providing the enhanced sensory capabilities for the EMP trading system.
    """
    
    def __init__(self, symbol: str, data_storage):
        """
        Initialize the intelligence engine.
        
        Args:
            symbol: Trading symbol (e.g., 'EURUSD')
            data_storage: EMP data storage instance
        """
        self.symbol = symbol
        self.data_storage = data_storage
        
        # Initialize dimensional sensors
        self.why_sensor = WhyDimension()
        self.how_sensor = HowDimension()
        self.what_sensor = WhatDimension()
        self.when_sensor = WhenDimension()
        self.anomaly_sensor = AnomalyDimension()
        
        # Register peer sensors for cross-dimensional awareness
        self.why_sensor.register_peer('how', self.how_sensor)
        self.why_sensor.register_peer('what', self.what_sensor)
        self.why_sensor.register_peer('when', self.when_sensor)
        self.why_sensor.register_peer('anomaly', self.anomaly_sensor)
        
        self.how_sensor.register_peer('why', self.why_sensor)
        self.how_sensor.register_peer('what', self.what_sensor)
        self.how_sensor.register_peer('when', self.when_sensor)
        self.how_sensor.register_peer('anomaly', self.anomaly_sensor)
        
        self.what_sensor.register_peer('why', self.why_sensor)
        self.what_sensor.register_peer('how', self.how_sensor)
        self.what_sensor.register_peer('when', self.when_sensor)
        self.what_sensor.register_peer('anomaly', self.anomaly_sensor)
        
        self.when_sensor.register_peer('why', self.why_sensor)
        self.when_sensor.register_peer('how', self.how_sensor)
        self.when_sensor.register_peer('what', self.what_sensor)
        self.when_sensor.register_peer('anomaly', self.anomaly_sensor)
        
        self.anomaly_sensor.register_peer('why', self.why_sensor)
        self.anomaly_sensor.register_peer('how', self.how_sensor)
        self.anomaly_sensor.register_peer('what', self.what_sensor)
        self.anomaly_sensor.register_peer('when', self.when_sensor)
        
        # Core components
        self.narrative_constructor = MarketNarrative()
        self.correlation_matrix = DimensionalCorrelationMatrix()
        self.regime_detector = RegimeDetector()
        self.memory_bank = MemoryBank()
        
        # State tracking
        self.current_understanding: Optional[MarketUnderstanding] = None
        self.calibrated = False
        self._lock = threading.RLock()
        
    def calibrate(self, start_time: datetime, end_time: datetime) -> bool:
        """
        Calibrate the intelligence engine with historical data.
        
        Args:
            start_time: Start of calibration period
            end_time: End of calibration period
            
        Returns:
            True if calibration successful
        """
        try:
            # Load historical data for calibration
            df = self.data_storage.get_data_range(self.symbol, start_time, end_time)
            if df is None or df.empty:
                return False
            
            # Calibrate each sensor
            self.why_sensor.calibrate(df)
            self.how_sensor.calibrate(df)
            self.what_sensor.calibrate(df)
            self.when_sensor.calibrate(df)
            self.anomaly_sensor.calibrate(df)
            
            self.calibrated = True
            return True
            
        except Exception as e:
            print(f"Calibration failed: {e}")
            return False
    
    def process_market_data(self, current_data) -> 'SensoryReading':
        """
        Process market data and return enhanced sensory reading.
        
        Args:
            current_data: Current market data point (pandas Series)
            
        Returns:
            SensoryReading object with multidimensional intelligence
        """
        if not self.calibrated:
            # Return default reading if not calibrated
            from .. import SensoryReading
            return self._create_default_reading(current_data.name)
        
        try:
            # Get historical data for context
            end_time = current_data.name
            start_time_hist = end_time - timedelta(hours=24)
            
            df = self.data_storage.get_data_range(self.symbol, start_time_hist, end_time)
            if df is None or df.empty:
                return self._create_default_reading(current_data.name)
            
            # Create market data object
            market_data = self._create_market_data(current_data, df)
            
            # Process through all dimensions
            peer_readings = {}
            
            why_reading = self.why_sensor.process(market_data, peer_readings)
            peer_readings['why'] = why_reading
            
            how_reading = self.how_sensor.process(market_data, peer_readings)
            peer_readings['how'] = how_reading
            
            what_reading = self.what_sensor.process(market_data, peer_readings)
            peer_readings['what'] = what_reading
            
            when_reading = self.when_sensor.process(market_data, peer_readings)
            peer_readings['when'] = when_reading
            
            anomaly_reading = self.anomaly_sensor.process(market_data, peer_readings)
            peer_readings['anomaly'] = anomaly_reading
            
            # Update correlation matrix
            self.correlation_matrix.update_correlations(peer_readings)
            
            # Detect market regime
            regime = self.regime_detector.update_regime(peer_readings)
            
            # Calculate consensus and alignment
            consensus_direction = self._calculate_consensus_direction(peer_readings)
            consensus_strength = self._calculate_consensus_strength(peer_readings)
            dimensional_alignment = self.correlation_matrix.get_dimensional_alignment(peer_readings)
            
            # Construct narrative
            narrative = self.narrative_constructor.construct_narrative(peer_readings)
            
            # Create market understanding
            understanding = MarketUnderstanding(
                timestamp=current_data.name,
                regime=regime,
                confidence=np.mean([r.confidence for r in peer_readings.values()]),
                why_reading=why_reading,
                how_reading=how_reading,
                what_reading=what_reading,
                when_reading=when_reading,
                anomaly_reading=anomaly_reading,
                consensus_direction=consensus_direction,
                consensus_strength=consensus_strength,
                dimensional_alignment=dimensional_alignment,
                narrative=narrative,
                context={
                    'processing_time_ms': 0.0,  # Could track actual processing time
                    'data_quality_score': 1.0
                }
            )
            
            # Store in memory bank
            self.memory_bank.store_episode(understanding)
            
            # Update current understanding
            with self._lock:
                self.current_understanding = understanding
            
            # Return as SensoryReading for backward compatibility
            from .. import SensoryReading
            return SensoryReading(understanding)
            
        except Exception as e:
            print(f"Error processing market data: {e}")
            return self._create_default_reading(current_data.name)
    
    def get_current_understanding(self) -> Optional[MarketUnderstanding]:
        """Get the current market understanding."""
        with self._lock:
            return self.current_understanding
    
    def get_dimensional_readings(self) -> Dict[str, DimensionalReading]:
        """Get individual dimensional readings."""
        understanding = self.get_current_understanding()
        if understanding is None:
            return {}
        
        return {
            'why': understanding.why_reading,
            'how': understanding.how_reading,
            'what': understanding.what_reading,
            'when': understanding.when_reading,
            'anomaly': understanding.anomaly_reading
        }
    
    def _create_market_data(self, current_data, historical_df: pd.DataFrame):
        """Create MarketData object from current data and historical context."""
        # This is a simplified adapter - in the full system, MarketData would be more sophisticated
        class SimpleMarketData:
            def __init__(self, timestamp, current_data, historical_df):
                self.timestamp = timestamp
                self.current_data = current_data
                self.historical_df = historical_df
                
                # Extract OHLCV from current data
                self.open = current_data.get('open', current_data.get('close', 0))
                self.high = current_data.get('high', current_data.get('close', 0))
                self.low = current_data.get('low', current_data.get('close', 0))
                self.close = current_data.get('close', 0)
                self.volume = current_data.get('volume', 1000)
        
        return SimpleMarketData(current_data.name, current_data, historical_df)
    
    def _calculate_consensus_direction(self, readings: Dict[str, DimensionalReading]) -> float:
        """Calculate consensus direction across dimensions."""
        values = [reading.value for reading in readings.values()]
        return np.mean(values)
    
    def _calculate_consensus_strength(self, readings: Dict[str, DimensionalReading]) -> float:
        """Calculate consensus strength across dimensions."""
        # Higher strength when readings are more aligned
        values = [reading.value for reading in readings.values()]
        return 1.0 - np.std(values)  # Lower std = higher strength
    
    def _create_default_reading(self, timestamp: datetime):
        """Create a default sensory reading when processing fails."""
        from .. import SensoryReading
        
        # Create a default market understanding
        default_reading = DimensionalReading(
            dimension="default",
            value=0.0,
            confidence=0.3,
            timestamp=timestamp,
            context={},
            influences={}
        )
        
        understanding = MarketUnderstanding(
            timestamp=timestamp,
            regime=MarketRegime.RANGING_LOW_VOL,
            confidence=0.3,
            why_reading=default_reading,
            how_reading=default_reading,
            what_reading=default_reading,
            when_reading=default_reading,
            anomaly_reading=default_reading,
            consensus_direction=0.0,
            consensus_strength=0.3,
            dimensional_alignment=0.0,
            narrative="Default reading - system not fully calibrated",
            context={
                'processing_time_ms': 0.0,
                'data_quality_score': 0.5
            }
        )
        
        return SensoryReading(understanding)

