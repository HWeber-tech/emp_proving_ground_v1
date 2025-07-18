"""
Anomaly Engine - Chaos Intelligence and Anomaly Detection Engine

This is the main engine for the "anomaly" sense that handles chaos intelligence,
anomaly detection, and pattern recognition.

Author: EMP Development Team
Date: July 18, 2024
Phase: 2 - Sensory Cortex Refactoring
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import pandas as pd
import numpy as np

from src.sensory.core.base import MarketData, DimensionalReading, MarketRegime

logger = logging.getLogger(__name__)


class AnomalyEngine:
    """
    Main engine for chaos intelligence and anomaly detection.
    
    This engine processes market data to understand anomalies and chaos patterns,
    including statistical anomalies, pattern recognition, and manipulation detection.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the anomaly engine with configuration"""
        self.config = config or {}
        
        # Initialize sub-modules
        try:
            from .pattern_recognition import AdvancedPatternRecognition
            from .anomaly_detection import StatisticalAnomalyDetector, ChaosDetector, ManipulationDetector
            
            self.pattern_recognition = AdvancedPatternRecognition()
            self.anomaly_detector = StatisticalAnomalyDetector()
            self.chaos_detector = ChaosDetector()
            self.manipulation_detector = ManipulationDetector()
            
            logger.info("Anomaly Engine initialized with sub-modules")
        except ImportError as e:
            logger.warning(f"Some sub-modules not available: {e}")
            self.pattern_recognition = None
            self.anomaly_detector = None
            self.chaos_detector = None
            self.manipulation_detector = None
    
    def analyze_market_data(self, market_data: List[MarketData], 
                          symbol: str = "UNKNOWN") -> Dict[str, Any]:
        """
        Perform comprehensive anomaly analysis on market data.
        
        Args:
            market_data: List of market data points
            symbol: Symbol being analyzed
            
        Returns:
            Dictionary containing all analysis results
        """
        if not market_data:
            logger.warning(f"No market data provided for {symbol}")
            return {}
        
        try:
            # Convert to DataFrame for easier analysis
            df = self._market_data_to_dataframe(market_data)
            
            # Perform all analyses
            analysis_results = {
                'symbol': symbol,
                'timestamp': datetime.now(),
                'data_points': len(market_data),
                'pattern_recognition': self._analyze_patterns(df),
                'anomaly_detection': self._analyze_anomalies(df),
                'chaos_analysis': self._analyze_chaos(df),
                'manipulation_detection': self._analyze_manipulation(df)
            }
            
            logger.info(f"Anomaly analysis completed for {symbol}")
            
            return analysis_results
            
        except Exception as e:
            logger.error(f"Error in anomaly analysis for {symbol}: {e}")
            return {}
    
    def analyze_anomaly_intelligence(self, market_data: List[MarketData], 
                                   symbol: str = "UNKNOWN") -> Dict[str, Any]:
        """
        Analyze anomaly intelligence and chaos patterns.
        
        Args:
            market_data: List of market data points
            symbol: Symbol being analyzed
            
        Returns:
            Anomaly intelligence analysis results
        """
        if not market_data:
            return {}
        
        try:
            df = self._market_data_to_dataframe(market_data)
            
            analysis = {
                'symbol': symbol,
                'timestamp': datetime.now(),
                'patterns_detected': self._detect_patterns(df),
                'anomalies_detected': self._detect_anomalies(df),
                'chaos_patterns': self._detect_chaos_patterns(df),
                'manipulation_patterns': self._detect_manipulation_patterns(df)
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error in anomaly intelligence analysis: {e}")
            return {}
    
    def get_dimensional_reading(self, market_data: List[MarketData], 
                              symbol: str = "UNKNOWN") -> DimensionalReading:
        """
        Get a dimensional reading for the anomaly sense.
        
        Args:
            market_data: List of market data points
            symbol: Symbol being analyzed
            
        Returns:
            DimensionalReading with anomaly sense analysis
        """
        analysis = self.analyze_market_data(market_data, symbol)
        
        # Calculate signal strength based on analysis
        signal_strength = self._calculate_signal_strength(analysis)
        confidence = self._calculate_confidence(analysis)
        
        return DimensionalReading(
            dimension="ANOMALY",
            signal_strength=signal_strength,
            confidence=confidence,
            regime=MarketRegime.UNKNOWN,
            context=analysis,
            data_quality=1.0,
            processing_time_ms=0.0,
            evidence={},
            warnings=[]
        )
    
    def _analyze_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze pattern recognition"""
        if self.pattern_recognition is None:
            return {}
        
        try:
            return self.pattern_recognition.update_data(df)
        except Exception as e:
            logger.error(f"Error analyzing patterns: {e}")
            return {}
    
    def _analyze_anomalies(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze statistical anomalies"""
        if self.anomaly_detector is None:
            return {}
        
        try:
            return self.anomaly_detector.update_data(df)
        except Exception as e:
            logger.error(f"Error analyzing anomalies: {e}")
            return {}
    
    def _analyze_chaos(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze chaos patterns"""
        if self.chaos_detector is None:
            return {}
        
        try:
            return self.chaos_detector.update_data(df)
        except Exception as e:
            logger.error(f"Error analyzing chaos: {e}")
            return {}
    
    def _analyze_manipulation(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze manipulation patterns"""
        if self.manipulation_detector is None:
            return {}
        
        try:
            return self.manipulation_detector.update_data(df)
        except Exception as e:
            logger.error(f"Error analyzing manipulation: {e}")
            return {}
    
    def _detect_patterns(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect patterns"""
        if self.pattern_recognition is None:
            return []
        
        try:
            return self.pattern_recognition.detect_patterns(df)
        except Exception as e:
            logger.error(f"Error detecting patterns: {e}")
            return []
    
    def _detect_anomalies(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect anomalies"""
        if self.anomaly_detector is None:
            return []
        
        try:
            return self.anomaly_detector.detect_statistical_anomalies(df)
        except Exception as e:
            logger.error(f"Error detecting anomalies: {e}")
            return []
    
    def _detect_chaos_patterns(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect chaos patterns"""
        if self.chaos_detector is None:
            return []
        
        try:
            return self.chaos_detector.detect_chaos_patterns(df)
        except Exception as e:
            logger.error(f"Error detecting chaos patterns: {e}")
            return []
    
    def _detect_manipulation_patterns(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect manipulation patterns"""
        if self.manipulation_detector is None:
            return []
        
        try:
            return self.manipulation_detector.detect_manipulation_patterns(df)
        except Exception as e:
            logger.error(f"Error detecting manipulation patterns: {e}")
            return []
    
    def _calculate_signal_strength(self, analysis: Dict[str, Any]) -> float:
        """Calculate signal strength from analysis results"""
        try:
            # Combine various analysis components
            pattern_score = analysis.get('pattern_recognition', {}).get('pattern_confidence', 0.0)
            anomaly_score = analysis.get('anomaly_detection', {}).get('anomaly_score', 0.0)
            chaos_score = analysis.get('chaos_analysis', {}).get('chaos_score', 0.0)
            manipulation_score = analysis.get('manipulation_detection', {}).get('manipulation_score', 0.0)
            
            # Calculate weighted signal strength
            signal_strength = (
                pattern_score * 0.3 + 
                anomaly_score * 0.3 + 
                chaos_score * 0.2 + 
                manipulation_score * 0.2
            )
            return min(max(signal_strength, -1.0), 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating signal strength: {e}")
            return 0.0
    
    def _calculate_confidence(self, analysis: Dict[str, Any]) -> float:
        """Calculate confidence from analysis results"""
        try:
            # Base confidence on data quality and analysis completeness
            data_points = analysis.get('data_points', 0)
            base_confidence = min(data_points / 100.0, 1.0)  # Higher confidence with more data
            
            # Adjust based on analysis quality
            if (analysis.get('pattern_recognition') and 
                analysis.get('anomaly_detection') and
                analysis.get('chaos_analysis')):
                base_confidence *= 1.2  # Boost confidence if we have good analysis
            
            return min(max(base_confidence, 0.0), 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating confidence: {e}")
            return 0.5
    
    def _market_data_to_dataframe(self, market_data: List[MarketData]) -> pd.DataFrame:
        """Convert market data list to pandas DataFrame"""
        data = []
        for md in market_data:
            data.append({
                'timestamp': md.timestamp,
                'open': md.open,
                'high': md.high,
                'low': md.low,
                'close': md.close,
                'volume': md.volume,
                'bid': md.bid,
                'ask': md.ask,
                'spread': md.spread,
                'mid_price': md.mid_price
            })
        
        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        return df


# Example usage
if __name__ == "__main__":
    # Test the anomaly engine
    engine = AnomalyEngine()
    print("Anomaly Engine initialized successfully") 