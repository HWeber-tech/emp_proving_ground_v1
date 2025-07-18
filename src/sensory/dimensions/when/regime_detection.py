"""
Regime Detection Module - When Sense

This module handles market regime detection and temporal analysis
for the "when" sense.

Author: EMP Development Team
Date: July 18, 2024
Phase: 2 - Missing Function Implementation
"""

import logging
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)


class MarketRegimeDetector:
    """
    Market Regime Detector
    
    Detects and classifies market regimes based on temporal patterns.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize market regime detector"""
        self.config = config or {}
        self.regime_history = []
        logger.info("MarketRegimeDetector initialized")
    
    def detect_market_regime(self, df: pd.DataFrame) -> str:
        """
        Detect the current market regime.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Market regime classification
        """
        if df.empty:
            return "unknown"
        
        try:
            # Calculate regime indicators
            volatility = self._calculate_volatility(df)
            trend_strength = self._calculate_trend_strength(df)
            momentum = self._calculate_momentum(df)
            
            # Classify regime based on indicators
            if volatility > 0.02:  # High volatility
                if trend_strength > 0.01:  # Strong trend
                    return "trending_volatile"
                else:
                    return "ranging_volatile"
            else:  # Low volatility
                if trend_strength > 0.01:  # Strong trend
                    return "trending_stable"
                else:
                    return "ranging_stable"
                    
        except Exception as e:
            logger.error(f"Error detecting market regime: {e}")
            return "unknown"
    
    def get_temporal_regime(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get detailed temporal regime analysis.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Temporal regime analysis results
        """
        if df.empty:
            return {}
        
        try:
            analysis = {
                'regime': self.detect_market_regime(df),
                'volatility': self._calculate_volatility(df),
                'trend_strength': self._calculate_trend_strength(df),
                'momentum': self._calculate_momentum(df),
                'temporal_confidence': self.calculate_temporal_confidence(df),
                'temporal_strength': self.calculate_temporal_strength(df),
                'timestamp': datetime.now()
            }
            
            self.regime_history.append(analysis)
            return analysis
            
        except Exception as e:
            logger.error(f"Error getting temporal regime: {e}")
            return {}
    
    def update_market_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Update market data and detect regime changes.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Regime detection results
        """
        return self.get_temporal_regime(df)
    
    def update_temporal_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Update temporal data for regime analysis.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Temporal analysis results
        """
        return self.get_temporal_regime(df)
    
    def calculate_temporal_confidence(self, df: pd.DataFrame) -> float:
        """
        Calculate confidence in temporal regime detection.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Temporal confidence score (0-1)
        """
        try:
            if len(df) < 20:
                return 0.0
            
            # Calculate confidence based on data quality and regime consistency
            volatility = self._calculate_volatility(df)
            trend_strength = self._calculate_trend_strength(df)
            
            # Higher confidence with more data and clearer patterns
            data_confidence = min(len(df) / 100.0, 1.0)
            pattern_confidence = (volatility + trend_strength) / 2
            
            temporal_confidence = (data_confidence * 0.6 + pattern_confidence * 0.4)
            return min(max(temporal_confidence, 0.0), 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating temporal confidence: {e}")
            return 0.0
    
    def calculate_temporal_strength(self, df: pd.DataFrame) -> float:
        """
        Calculate temporal strength of current regime.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Temporal strength score (0-1)
        """
        try:
            if len(df) < 10:
                return 0.0
            
            # Calculate temporal strength based on regime persistence
            regime = self.detect_market_regime(df)
            
            # Analyze regime consistency over time
            recent_data = df.tail(10)
            volatility_consistency = 1.0 - recent_data['close'].pct_change().std()
            trend_consistency = abs(recent_data['close'].iloc[-1] - recent_data['close'].iloc[0]) / recent_data['close'].iloc[0]
            
            temporal_strength = (volatility_consistency + trend_consistency) / 2
            return min(max(temporal_strength, 0.0), 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating temporal strength: {e}")
            return 0.0
    
    def get_chrono_behavior(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze chronological behavior patterns.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Chronological behavior analysis
        """
        try:
            if len(df) < 20:
                return {}
            
            # Analyze time-based patterns
            df_with_time = df.copy()
            df_with_time['hour'] = pd.to_datetime(df_with_time['timestamp']).dt.hour
            
            # Session analysis
            london_session = df_with_time[(df_with_time['hour'] >= 8) & (df_with_time['hour'] < 16)]
            ny_session = df_with_time[(df_with_time['hour'] >= 13) & (df_with_time['hour'] < 21)]
            asia_session = df_with_time[(df_with_time['hour'] >= 0) & (df_with_time['hour'] < 8)]
            
            session_volatility = {
                'london': london_session['close'].pct_change().std() if len(london_session) > 0 else 0.0,
                'new_york': ny_session['close'].pct_change().std() if len(ny_session) > 0 else 0.0,
                'asia': asia_session['close'].pct_change().std() if len(asia_session) > 0 else 0.0
            }
            
            return {
                'session_volatility': session_volatility,
                'most_volatile_session': max(session_volatility, key=session_volatility.get),
                'session_patterns': self._analyze_session_patterns(df_with_time)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing chrono behavior: {e}")
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
    
    def _calculate_momentum(self, df: pd.DataFrame) -> float:
        """Calculate momentum metric"""
        try:
            if len(df) < 5:
                return 0.0
            
            # Calculate momentum based on recent price changes
            recent_returns = df['close'].pct_change().tail(5)
            momentum = recent_returns.mean()
            
            return min(max(abs(momentum), 0.0), 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating momentum: {e}")
            return 0.0
    
    def _analyze_session_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze trading session patterns"""
        try:
            # Analyze volume patterns by session
            session_volume = df.groupby('hour')['volume'].mean()
            
            # Find peak volume hours
            peak_hours = session_volume.nlargest(3).index.tolist()
            
            return {
                'peak_volume_hours': peak_hours,
                'volume_distribution': session_volume.to_dict()
            }
            
        except Exception as e:
            logger.error(f"Error analyzing session patterns: {e}")
            return {}


class TemporalAnalyzer:
    """
    Temporal Analyzer
    
    Analyzes temporal patterns and market timing.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize temporal analyzer"""
        self.config = config or {}
        logger.info("TemporalAnalyzer initialized")
    
    def update_market_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Update market data for temporal analysis.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Temporal analysis results
        """
        return self.get_temporal_regime(df)
    
    def get_temporal_regime(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get temporal regime analysis.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Temporal regime analysis
        """
        detector = MarketRegimeDetector()
        return detector.get_temporal_regime(df)
    
    def detect_market_regime(self, df: pd.DataFrame) -> str:
        """
        Detect market regime.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Market regime classification
        """
        detector = MarketRegimeDetector()
        return detector.detect_market_regime(df)
    
    def calculate_temporal_confidence(self, df: pd.DataFrame) -> float:
        """
        Calculate temporal confidence.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Temporal confidence score
        """
        detector = MarketRegimeDetector()
        return detector.calculate_temporal_confidence(df)
    
    def calculate_temporal_strength(self, df: pd.DataFrame) -> float:
        """
        Calculate temporal strength.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Temporal strength score
        """
        detector = MarketRegimeDetector()
        return detector.calculate_temporal_strength(df)
    
    def get_chrono_behavior(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get chronological behavior analysis.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Chronological behavior analysis
        """
        detector = MarketRegimeDetector()
        return detector.get_chrono_behavior(df)
    
    def update_temporal_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Update temporal data.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Temporal analysis results
        """
        return self.get_temporal_regime(df)


# Example usage
if __name__ == "__main__":
    # Test regime detector
    detector = MarketRegimeDetector()
    analyzer = TemporalAnalyzer()
    print("Regime detection modules initialized successfully") 