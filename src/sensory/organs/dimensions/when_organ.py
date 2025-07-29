"""
When Engine - Temporal Intelligence and Market Timing Engine

This is the main engine for the "when" sense that handles temporal intelligence,
market timing, and regime detection.

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


class WhenEngine:
    """
    Main engine for temporal intelligence and market timing.
    
    This engine processes market data to understand WHEN to act,
    including market regime detection, temporal analysis, and timing signals.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the when engine with configuration"""
        self.config = config or {}
        
        # Initialize sub-modules
        try:
            from .regime_detection import MarketRegimeDetector, TemporalAnalyzer
            
            self.regime_detector = MarketRegimeDetector()
            self.temporal_analyzer = TemporalAnalyzer()
            
            logger.info("When Engine initialized with sub-modules")
        except ImportError as e:
            logger.warning(f"Some sub-modules not available: {e}")
            self.regime_detector = None
            self.temporal_analyzer = None
    
    def analyze_market_data(self, market_data: List[MarketData], 
                          symbol: str = "UNKNOWN") -> Dict[str, Any]:
        """
        Perform comprehensive temporal analysis on market data.
        
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
                'regime_detection': self._analyze_regime_detection(df),
                'temporal_analysis': self._analyze_temporal_analysis(df),
                'market_timing': self._analyze_market_timing(df),
                'session_analysis': self._analyze_session_behavior(df)
            }
            
            logger.info(f"Temporal analysis completed for {symbol}")
            
            return analysis_results
            
        except Exception as e:
            logger.error(f"Error in temporal analysis for {symbol}: {e}")
            return {}
    
    def analyze_temporal_intelligence(self, market_data: List[MarketData], 
                                    symbol: str = "UNKNOWN") -> Dict[str, Any]:
        """
        Analyze temporal intelligence and market timing.
        
        Args:
            market_data: List of market data points
            symbol: Symbol being analyzed
            
        Returns:
            Temporal intelligence analysis results
        """
        if not market_data:
            return {}
        
        try:
            df = self._market_data_to_dataframe(market_data)
            
            analysis = {
                'symbol': symbol,
                'timestamp': datetime.now(),
                'market_regime': self._detect_market_regime(df),
                'temporal_confidence': self._calculate_temporal_confidence(df),
                'temporal_strength': self._calculate_temporal_strength(df),
                'chrono_behavior': self._get_chrono_behavior(df)
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error in temporal intelligence analysis: {e}")
            return {}
    
    def get_dimensional_reading(self, market_data: List[MarketData], 
                              symbol: str = "UNKNOWN") -> DimensionalReading:
        """
        Get a dimensional reading for the when sense.
        
        Args:
            market_data: List of market data points
            symbol: Symbol being analyzed
            
        Returns:
            DimensionalReading with when sense analysis
        """
        analysis = self.analyze_market_data(market_data, symbol)
        
        # Calculate signal strength based on analysis
        signal_strength = self._calculate_signal_strength(analysis)
        confidence = self._calculate_confidence(analysis)
        
        return DimensionalReading(
            dimension="WHEN",
            signal_strength=signal_strength,
            confidence=confidence,
            regime=MarketRegime.UNKNOWN,
            context=analysis,
            data_quality=1.0,
            processing_time_ms=0.0,
            evidence={},
            warnings=[]
        )
    
    def _analyze_regime_detection(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze market regime detection"""
        if self.regime_detector is None:
            return {}
        
        try:
            return self.regime_detector.get_temporal_regime(df)
        except Exception as e:
            logger.error(f"Error analyzing regime detection: {e}")
            return {}
    
    def _analyze_temporal_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze temporal patterns"""
        if self.temporal_analyzer is None:
            return {}
        
        try:
            return self.temporal_analyzer.get_temporal_regime(df)
        except Exception as e:
            logger.error(f"Error analyzing temporal patterns: {e}")
            return {}
    
    def _analyze_market_timing(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze market timing signals"""
        try:
            if len(df) < 20:
                return {}
            
            # Calculate timing indicators
            volatility = self._calculate_volatility(df)
            trend_strength = self._calculate_trend_strength(df)
            
            # Determine optimal timing
            if volatility > 0.02:  # High volatility - be cautious
                timing_signal = "wait"
                timing_confidence = 0.3
            elif trend_strength > 0.01:  # Strong trend - good timing
                timing_signal = "act"
                timing_confidence = 0.8
            else:  # Low volatility, weak trend - neutral
                timing_signal = "neutral"
                timing_confidence = 0.5
            
            return {
                'timing_signal': timing_signal,
                'timing_confidence': timing_confidence,
                'volatility': volatility,
                'trend_strength': trend_strength
            }
            
        except Exception as e:
            logger.error(f"Error analyzing market timing: {e}")
            return {}
    
    def _analyze_session_behavior(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze trading session behavior"""
        if self.regime_detector is None:
            return {}
        
        try:
            return self.regime_detector.get_chrono_behavior(df)
        except Exception as e:
            logger.error(f"Error analyzing session behavior: {e}")
            return {}
    
    def _detect_market_regime(self, df: pd.DataFrame) -> str:
        """Detect market regime"""
        if self.regime_detector is None:
            return "unknown"
        
        try:
            return self.regime_detector.detect_market_regime(df)
        except Exception as e:
            logger.error(f"Error detecting market regime: {e}")
            return "unknown"
    
    def _calculate_temporal_confidence(self, df: pd.DataFrame) -> float:
        """Calculate temporal confidence"""
        if self.regime_detector is None:
            return 0.0
        
        try:
            return self.regime_detector.calculate_temporal_confidence(df)
        except Exception as e:
            logger.error(f"Error calculating temporal confidence: {e}")
            return 0.0
    
    def _calculate_temporal_strength(self, df: pd.DataFrame) -> float:
        """Calculate temporal strength"""
        if self.regime_detector is None:
            return 0.0
        
        try:
            return self.regime_detector.calculate_temporal_strength(df)
        except Exception as e:
            logger.error(f"Error calculating temporal strength: {e}")
            return 0.0
    
    def _get_chrono_behavior(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get chronological behavior"""
        if self.regime_detector is None:
            return {}
        
        try:
            return self.regime_detector.get_chrono_behavior(df)
        except Exception as e:
            logger.error(f"Error getting chrono behavior: {e}")
            return {}
    
    def _calculate_volatility(self, df: pd.DataFrame) -> float:
        """Calculate volatility metric"""
        try:
            if len(df) < 5:
                return 0.0
            
            returns = df['close'].pct_change().dropna()
            volatility = returns.std()
            return min(max(volatility, 0.0), 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating volatility: {e}")
            return 0.0
    
    def _calculate_trend_strength(self, df: pd.DataFrame) -> float:
        """Calculate trend strength metric"""
        try:
            if len(df) < 10:
                return 0.0
            
            # Calculate trend using linear regression
            x = np.arange(len(df))
            y = df['close'].values
            
            # Linear regression
            slope, intercept = np.polyfit(x, y, 1)
            
            # Calculate R-squared (trend strength)
            y_pred = slope * x + intercept
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            trend_strength = abs(slope) * r_squared
            return min(max(trend_strength, 0.0), 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating trend strength: {e}")
            return 0.0
    
    def _calculate_signal_strength(self, analysis: Dict[str, Any]) -> float:
        """Calculate signal strength from analysis results"""
        try:
            # Combine various analysis components
            temporal_confidence = analysis.get('regime_detection', {}).get('temporal_confidence', 0.0)
            temporal_strength = analysis.get('regime_detection', {}).get('temporal_strength', 0.0)
            timing_confidence = analysis.get('market_timing', {}).get('timing_confidence', 0.0)
            
            # Calculate weighted signal strength
            signal_strength = (
                temporal_confidence * 0.4 + 
                temporal_strength * 0.3 + 
                timing_confidence * 0.3
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
            if analysis.get('regime_detection') and analysis.get('temporal_analysis'):
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
    # Test the when engine
    engine = WhenEngine()
    print("When Engine initialized successfully") 
