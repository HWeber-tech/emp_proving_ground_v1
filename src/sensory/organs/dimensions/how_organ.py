"""
How Engine - Main Technical Analysis Engine

This is the main engine for the "how" sense that orchestrates all technical analysis,
indicators, patterns, and market mechanics analysis.

Author: EMP Development Team
Date: July 18, 2024
Phase: 2 - Sensory Cortex Refactoring
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

from src.core.base import DimensionalReading, MarketData, MarketRegime

logger = logging.getLogger(__name__)


class HowEngine:
    """
    Main engine for technical analysis and market mechanics.
    
    This engine processes market data to understand HOW the market is moving,
    including technical indicators, patterns, momentum, and volatility analysis.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the how engine with configuration"""
        self.config = config or {}
        
        # Initialize sub-modules
        try:
            from .indicators import TechnicalIndicators
            from .order_flow import MarketMicrostructureAnalyzer, OrderFlowAnalyzer
            from .patterns import ICTPatternDetector, OrderFlowDataProvider
            
            self.indicators = TechnicalIndicators()
            self.patterns = ICTPatternDetector()
            self.order_flow_provider = OrderFlowDataProvider()
            self.order_flow_analyzer = OrderFlowAnalyzer()
            self.microstructure = MarketMicrostructureAnalyzer()
            
            logger.info("How Engine initialized with all sub-modules")
        except ImportError as e:
            logger.warning(f"Some sub-modules not available: {e}")
            self.indicators = None
            self.patterns = None
            self.order_flow_provider = None
            self.order_flow_analyzer = None
            self.microstructure = None
    
    def analyze_market_data(self, market_data: List[MarketData], 
                          symbol: str = "UNKNOWN") -> Dict[str, Any]:
        """
        Perform comprehensive technical analysis on market data.
        
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
                'indicators': self._analyze_indicators(df),
                'patterns': self._analyze_patterns(df),
                'order_flow': self._analyze_order_flow(df),
                'microstructure': self._analyze_microstructure(df),
                'institutional_mechanics': self.analyze_institutional_mechanics(market_data, symbol)
            }
            
            logger.info(f"Technical analysis completed for {symbol}")
            
            return analysis_results
            
        except Exception as e:
            logger.error(f"Error in technical analysis for {symbol}: {e}")
            return {}
    
    def analyze_institutional_mechanics(self, market_data: List[MarketData], 
                                      symbol: str = "UNKNOWN") -> Dict[str, Any]:
        """
        Analyze institutional mechanics and ICT patterns.
        
        Args:
            market_data: List of market data points
            symbol: Symbol being analyzed
            
        Returns:
            Institutional mechanics analysis results
        """
        if not market_data:
            return {}
        
        try:
            df = self._market_data_to_dataframe(market_data)
            
            analysis = {
                'symbol': symbol,
                'timestamp': datetime.now(),
                'ict_patterns': self._analyze_patterns(df),
                'order_flow': self._analyze_order_flow(df),
                'footprint_score': self._get_footprint_score(df),
                'institutional_pressure': self._get_institutional_pressure(df)
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error in institutional mechanics analysis: {e}")
            return {}
    
    def get_dimensional_reading(self, market_data: List[MarketData], 
                              symbol: str = "UNKNOWN") -> DimensionalReading:
        """
        Get a dimensional reading for the how sense.
        
        Args:
            market_data: List of market data points
            symbol: Symbol being analyzed
            
        Returns:
            DimensionalReading with how sense analysis
        """
        analysis = self.analyze_market_data(market_data, symbol)
        
        # Calculate signal strength based on analysis
        signal_strength = self._calculate_signal_strength(analysis)
        confidence = self._calculate_confidence(analysis)
        
        return DimensionalReading(
            dimension="HOW",
            signal_strength=signal_strength,
            confidence=confidence,
            regime=MarketRegime.UNKNOWN,
            context=analysis,
            data_quality=1.0,
            processing_time_ms=0.0,
            evidence={},
            warnings=[]
        )
    
    def _analyze_indicators(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze technical indicators"""
        if self.indicators is None:
            return {}
        
        try:
            return self.indicators.calculate_all(df)
        except Exception as e:
            logger.error(f"Error analyzing indicators: {e}")
            return {}
    
    def _analyze_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze ICT patterns"""
        if self.patterns is None:
            return {}
        
        try:
            return self.patterns.update_market_data(df)
        except Exception as e:
            logger.error(f"Error analyzing patterns: {e}")
            return {}
    
    def _analyze_order_flow(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze order flow"""
        if self.order_flow_analyzer is None:
            return {}
        
        try:
            return self.order_flow_analyzer.analyze_institutional_flow(df)
        except Exception as e:
            logger.error(f"Error analyzing order flow: {e}")
            return {}
    
    def _analyze_microstructure(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze market microstructure"""
        if self.microstructure is None:
            return {}
        
        try:
            return self.microstructure.analyze_microstructure(df)
        except Exception as e:
            logger.error(f"Error analyzing microstructure: {e}")
            return {}
    
    def _get_footprint_score(self, df: pd.DataFrame) -> float:
        """Get institutional footprint score"""
        if self.patterns is None:
            return 0.0
        
        try:
            return self.patterns.get_institutional_footprint_score(df)
        except Exception as e:
            logger.error(f"Error getting footprint score: {e}")
            return 0.0
    
    def _get_institutional_pressure(self, df: pd.DataFrame) -> Dict[str, float]:
        """Get institutional pressure analysis"""
        if self.order_flow_analyzer is None:
            return {'buying_pressure': 0.0, 'selling_pressure': 0.0}
        
        try:
            flow_analysis = self.order_flow_analyzer.analyze_institutional_flow(df)
            return flow_analysis.get('institutional_pressure', 
                                   {'buying_pressure': 0.0, 'selling_pressure': 0.0})
        except Exception as e:
            logger.error(f"Error getting institutional pressure: {e}")
            return {'buying_pressure': 0.0, 'selling_pressure': 0.0}
    
    def _calculate_signal_strength(self, analysis: Dict[str, Any]) -> float:
        """Calculate signal strength from analysis results"""
        try:
            # Combine various analysis components
            footprint_score = analysis.get('patterns', {}).get('footprint_score', 0.0)
            flow_strength = analysis.get('order_flow', {}).get('flow_strength', 0.0)
            
            # Calculate weighted signal strength
            signal_strength = (footprint_score * 0.6 + flow_strength * 0.4)
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
            if analysis.get('patterns') and analysis.get('order_flow'):
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
    # Test the how engine
    engine = HowEngine()
    print("How Engine initialized successfully") 
