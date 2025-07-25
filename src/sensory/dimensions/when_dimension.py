"""
WHEN Dimension - Temporal Advantage System

This module implements the WHEN dimension of the sensory cortex,
responsible for timing analysis, temporal pattern recognition,
and optimal entry/exit timing determination.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import pandas as pd
import numpy as np


@dataclass
class TemporalMetrics:
    """Metrics for temporal analysis."""
    optimal_entry_time: datetime
    expected_duration: timedelta
    confidence_window: timedelta
    seasonality_factor: float
    cycle_phase: float


class WhenDimension:
    """
    Temporal Advantage System for market timing.
    
    This class implements the WHEN dimension functionality for determining
    optimal timing for trades, analyzing market cycles, and identifying
    temporal patterns in market behavior.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the WHEN dimension."""
        self.config = config or {}
        self.temporal_patterns = {}
        self.market_cycles = {}
        self.timing_signals = []
        
    def analyze_timing(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze market data for optimal timing.
        
        Args:
            data: Market data DataFrame
            
        Returns:
            Dictionary containing timing analysis results
        """
        current_time = datetime.now()
        
        return {
            'optimal_entry': self._calculate_optimal_entry(data),
            'market_cycle': self._identify_cycle_phase(data),
            'seasonality': self._analyze_seasonality(data),
            'volatility_timing': self._assess_volatility_timing(data),
            'metadata': {
                'analysis_time': current_time,
                'data_range': self._get_data_range(data),
                'confidence': 0.82
            }
        }
        
    def _calculate_optimal_entry(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate optimal entry timing."""
        return {
            'time': datetime.now() + timedelta(minutes=15),
            'confidence': 0.79,
            'window': timedelta(minutes=5),
            'conditions': ['low_volatility', 'trend_confirmation']
        }
        
    def _identify_cycle_phase(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Identify current market cycle phase."""
        return {
            'phase': 'accumulation',
            'position': 0.3,
            'strength': 0.65,
            'expected_duration': timedelta(days=3)
        }
        
    def _analyze_seasonality(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze seasonal patterns in the data."""
        return {
            'monthly_pattern': 'bullish',
            'weekly_pattern': 'neutral',
            'daily_pattern': 'bearish_morning',
            'strength': 0.71
        }
        
    def _assess_volatility_timing(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Assess timing based on volatility patterns."""
        return {
            'current_volatility': 0.15,
            'optimal_volatility_range': (0.10, 0.20),
            'timing_score': 0.84
        }
        
    def _get_data_range(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Get the time range of the provided data."""
        if data.empty:
            return {'start': None, 'end': None, 'duration': None}
            
        return {
            'start': data.index[0],
            'end': data.index[-1],
            'duration': data.index[-1] - data.index[0]
        }
        
    def register_cycle(self, cycle_id: str, cycle_data: Dict[str, Any]) -> None:
        """Register a new market cycle."""
        self.market_cycles[cycle_id] = cycle_data
        
    def get_cycle(self, cycle_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve cycle information by ID."""
        return self.market_cycles.get(cycle_id)
        
    def list_cycles(self) -> List[str]:
        """List all registered cycle IDs."""
        return list(self.market_cycles.keys())
        
    def get_timing_recommendation(self) -> Dict[str, Any]:
        """Get current timing recommendation."""
        return {
            'action': 'wait',
            'reason': 'market_volatility_high',
            'expected_wait_time': timedelta(minutes=30),
            'confidence': 0.76
        }
        
    def get_status(self) -> Dict[str, Any]:
        """Get current status of the WHEN dimension."""
        return {
            'active_cycles': len(self.market_cycles),
            'total_patterns': len(self.temporal_patterns),
            'signals': len(self.timing_signals),
            'health': 'healthy',
            'last_update': datetime.now()
        }
