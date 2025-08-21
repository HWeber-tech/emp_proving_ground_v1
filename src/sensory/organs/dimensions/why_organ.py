"""
Why Engine - Fundamental Intelligence and Economic Analysis Engine

This is the main engine for the "why" sense that handles fundamental intelligence,
economic analysis, and market drivers.

Author: EMP Development Team
Date: July 18, 2024
Phase: 2 - Sensory Cortex Refactoring
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.data_foundation.config.why_config import load_why_config
from src.core.telemetry import get_metrics_sink
from src.core.base import DimensionalReading, MarketData, MarketRegime
from src.sensory.dimensions.why.yield_signal import YieldSlopeTracker

logger = logging.getLogger(__name__)


class WhyEngine:
    """
    Main engine for fundamental intelligence and economic analysis.
    
    This engine processes market data to understand WHY the market moves,
    including economic factors, fundamental analysis, and market drivers.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the why engine with configuration"""
        self.config = config or {}
        try:
            self.why_cfg = load_why_config()
        except Exception:
            self.why_cfg = None
        self._ytracker = YieldSlopeTracker()
        
        # Initialize sub-modules
        try:
            from .economic_analysis import EconomicDataProvider, FundamentalAnalyzer
            
            self.economic_provider = EconomicDataProvider()
            self.fundamental_analyzer = FundamentalAnalyzer()
            
            logger.info("Why Engine initialized with sub-modules")
        except ImportError as e:
            logger.warning(f"Some sub-modules not available: {e}")
            self.economic_provider = None
            self.fundamental_analyzer = None
    
    def analyze_market_data(self, market_data: List[MarketData], 
                          symbol: str = "UNKNOWN") -> Dict[str, Any]:
        """
        Perform comprehensive fundamental analysis on market data.
        
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
                'economic_analysis': self._analyze_economic_factors(df),
                'fundamental_analysis': self._analyze_fundamentals(df),
                'market_drivers': self._analyze_market_drivers(df),
                'sentiment_analysis': self._analyze_sentiment(df)
            }
            
            logger.info(f"Fundamental analysis completed for {symbol}")
            
            return analysis_results
            
        except Exception as e:
            logger.error(f"Error in fundamental analysis for {symbol}: {e}")
            return {}
    
    def analyze_fundamental_intelligence(self, market_data: List[MarketData], 
                                       symbol: str = "UNKNOWN") -> Dict[str, Any]:
        """
        Analyze fundamental intelligence and economic factors.
        
        Args:
            market_data: List of market data points
            symbol: Symbol being analyzed
            
        Returns:
            Fundamental intelligence analysis results
        """
        if not market_data:
            return {}
        
        try:
            df = self._market_data_to_dataframe(market_data)
            
            analysis = {
                'symbol': symbol,
                'timestamp': datetime.now(),
                'economic_calendar': self._get_economic_calendar(),
                'central_bank_policies': self._get_central_bank_policies(),
                'economic_momentum': self._analyze_economic_momentum(df),
                'risk_sentiment': self._analyze_risk_sentiment(df),
                'yield_differentials': self._analyze_yield_differentials(df)
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error in fundamental intelligence analysis: {e}")
            return {}
    
    def get_dimensional_reading(self, market_data: List[MarketData], 
                              symbol: str = "UNKNOWN") -> DimensionalReading:
        """
        Get a dimensional reading for the why sense.
        
        Args:
            market_data: List of market data points
            symbol: Symbol being analyzed
            
        Returns:
            DimensionalReading with why sense analysis
        """
        analysis = self.analyze_market_data(market_data, symbol)
        
        # Calculate signal strength based on analysis and yield tracker if enabled
        signal_strength = self._calculate_signal_strength(analysis)
        confidence = self._calculate_confidence(analysis)
        try:
            sink = get_metrics_sink()
            sink.set_gauge("why_composite_signal", float(signal_strength), {"symbol": symbol})
            sink.set_gauge("why_confidence", float(confidence), {"symbol": symbol})
            sink.set_gauge(
                "why_feature_available",
                1.0 if (self.why_cfg.enable_yields if self.why_cfg else True) else 0.0,
                {"feature": "yields"},
            )
            sink.set_gauge(
                "why_feature_available",
                1.0 if (self.why_cfg.enable_macro_proximity if self.why_cfg else True) else 0.0,
                {"feature": "macro"},
            )
        except Exception:
            pass
        
        return DimensionalReading(
            dimension="WHY",
            signal_strength=signal_strength,
            confidence=confidence,
            regime=MarketRegime.UNKNOWN,
            context=analysis,
            data_quality=1.0,
            processing_time_ms=0.0,
            evidence={},
            warnings=[]
        )
    
    def _analyze_economic_factors(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze economic factors"""
        if self.economic_provider is None:
            return {}
        
        try:
            return {
                'currency_strength': self.economic_provider.analyze_currency_strength(df),
                'economic_calendar': self.economic_provider.get_economic_calendar(),
                'central_bank_policies': self.economic_provider.get_central_bank_policies(),
                'geopolitical_events': self.economic_provider.get_geopolitical_events()
            }
        except Exception as e:
            logger.error(f"Error analyzing economic factors: {e}")
            return {}
    
    def _analyze_fundamentals(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze fundamental factors"""
        if self.fundamental_analyzer is None:
            return {}
        
        try:
            return {
                'economic_momentum': self.fundamental_analyzer.analyze_economic_momentum(df),
                'risk_sentiment': self.fundamental_analyzer.analyze_risk_sentiment(df),
                'yield_differentials': self.fundamental_analyzer.analyze_yield_differentials(df),
                'central_bank_divergence': self.fundamental_analyzer.analyze_central_bank_divergence(df),
                'economic_calendar_impact': self.fundamental_analyzer.analyze_economic_calendar_impact(df)
            }
        except Exception as e:
            logger.error(f"Error analyzing fundamentals: {e}")
            return {}
    
    def _analyze_market_drivers(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze market drivers"""
        try:
            if df.empty:
                return {}
            
            # Identify key market drivers
            price_trend = self._calculate_price_trend(df)
            volume_trend = self._calculate_volume_trend(df)
            
            drivers = {
                'price_trend': price_trend,
                'volume_trend': volume_trend,
                'primary_driver': 'technical' if abs(price_trend) > 0.01 else 'fundamental',
                'driver_strength': min(max(abs(price_trend) + abs(volume_trend), 0.0), 1.0)
            }
            
            return drivers
            
        except Exception as e:
            logger.error(f"Error analyzing market drivers: {e}")
            return {}
    
    def _analyze_sentiment(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze market sentiment"""
        try:
            if df.empty:
                return {}
            
            # Calculate sentiment indicators
            price_momentum = df['close'].pct_change().tail(10).mean()
            volatility = df['close'].pct_change().std()
            
            # Determine sentiment
            if price_momentum > 0.001 and volatility < 0.02:
                sentiment = 'bullish'
                sentiment_score = 0.8
            elif price_momentum < -0.001 and volatility < 0.02:
                sentiment = 'bearish'
                sentiment_score = 0.2
            else:
                sentiment = 'neutral'
                sentiment_score = 0.5
            
            return {
                'sentiment': sentiment,
                'sentiment_score': sentiment_score,
                'price_momentum': price_momentum,
                'volatility': volatility
            }
            
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {e}")
            return {}
    
    def _get_economic_calendar(self) -> List[Dict[str, Any]]:
        """Get economic calendar"""
        if self.economic_provider is None:
            return []
        
        try:
            return self.economic_provider.get_economic_calendar()
        except Exception as e:
            logger.error(f"Error getting economic calendar: {e}")
            return []
    
    def _get_central_bank_policies(self) -> Dict[str, Any]:
        """Get central bank policies"""
        if self.economic_provider is None:
            return {}
        
        try:
            return self.economic_provider.get_central_bank_policies()
        except Exception as e:
            logger.error(f"Error getting central bank policies: {e}")
            return {}
    
    def _analyze_economic_momentum(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze economic momentum"""
        if self.fundamental_analyzer is None:
            return {}
        
        try:
            return self.fundamental_analyzer.analyze_economic_momentum(df)
        except Exception as e:
            logger.error(f"Error analyzing economic momentum: {e}")
            return {}
    
    def _analyze_risk_sentiment(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze risk sentiment"""
        if self.fundamental_analyzer is None:
            return {}
        
        try:
            return self.fundamental_analyzer.analyze_risk_sentiment(df)
        except Exception as e:
            logger.error(f"Error analyzing risk sentiment: {e}")
            return {}
    
    def _analyze_yield_differentials(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze yield differentials"""
        if self.fundamental_analyzer is None:
            return {}
        
        try:
            return self.fundamental_analyzer.analyze_yield_differentials(df)
        except Exception as e:
            logger.error(f"Error analyzing yield differentials: {e}")
            return {}
    
    def _calculate_price_trend(self, df: pd.DataFrame) -> float:
        """Calculate price trend"""
        try:
            if len(df) < 10:
                return 0.0
            
            # Calculate trend using linear regression
            x = np.arange(len(df))
            y = df['close'].values
            
            # Linear regression
            slope, intercept = np.polyfit(x, y, 1)
            
            # Normalize trend
            trend = slope / df['close'].mean() if df['close'].mean() > 0 else 0
            return min(max(trend, -1.0), 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating price trend: {e}")
            return 0.0
    
    def _calculate_volume_trend(self, df: pd.DataFrame) -> float:
        """Calculate volume trend"""
        try:
            if len(df) < 10:
                return 0.0
            
            # Calculate volume trend
            volume_trend = df['volume'].pct_change().tail(10).mean()
            return min(max(volume_trend, -1.0), 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating volume trend: {e}")
            return 0.0
    
    def _calculate_signal_strength(self, analysis: Dict[str, Any]) -> float:
        """Calculate signal strength from analysis results"""
        try:
            # Combine various analysis components
            economic_momentum = analysis.get('fundamental_analysis', {}).get('economic_momentum', {}).get('momentum_score', 0.0)
            risk_sentiment = analysis.get('fundamental_analysis', {}).get('risk_sentiment', {}).get('risk_sentiment', 0.0)
            sentiment_score = analysis.get('sentiment_analysis', {}).get('sentiment_score', 0.5)
            # Yield signal
            y_sig, y_conf = (0.0, 0.0)
            if self.why_cfg is None or self.why_cfg.enable_yields:
                # Use tracker last known slope direction as signal proxy
                y_sig, y_conf = self._ytracker.signal()
            # Macro proximity is not available in live engine here; keep neutral
            w_macro = (self.why_cfg.weight_macro if self.why_cfg else 0.4)
            w_yield = (self.why_cfg.weight_yields if self.why_cfg else 0.6)
            # Calculate weighted signal strength (macro prox -> neutral 0)
            base = (
                economic_momentum * 0.4 +
                (1 - risk_sentiment) * 0.3 +
                sentiment_score * 0.3
            )
            why_weighted = (0.0 * w_macro) + (y_sig * y_conf * w_yield)
            signal_strength = float(max(-1.0, min(1.0, base * 0.5 + why_weighted)))
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
            if analysis.get('economic_analysis') and analysis.get('fundamental_analysis'):
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
    # Test the why engine
    engine = WhyEngine()
    print("Why Engine initialized successfully") 
