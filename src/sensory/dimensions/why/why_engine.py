"""
Why Engine - Fundamental Intelligence and Market Drivers Engine

This is the main engine for the "why" sense that handles fundamental analysis,
market drivers, and economic intelligence.

Author: EMP Development Team
Date: July 18, 2024
Phase: 2 - Sensory Cortex Refactoring
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import pandas as pd
import numpy as np

from src.sensory.core.base import MarketData, DimensionalReading, MarketRegime, ConfidenceLevel

logger = logging.getLogger(__name__)


class WhyEngine:
    """
    Main engine for fundamental intelligence and market driver analysis.
    
    This engine processes market data to understand WHY the market moves,
    including fundamental analysis, economic drivers, and sentiment.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the why engine with configuration"""
        self.config = config or {}
        logger.info("Why Engine initialized")
    
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
                'fundamental_intelligence': {}, # Will be implemented in fundamental_intelligence sub-module
                'economic_analysis': {},        # Will be implemented in economic_analysis sub-module
                'market_drivers': {},           # Will be implemented in market_drivers sub-module
                'sentiment_analysis': {}        # Will be implemented in sentiment_analysis sub-module
            }
            
            logger.info(f"Fundamental analysis completed for {symbol}")
            
            return analysis_results
            
        except Exception as e:
            logger.error(f"Error in fundamental analysis for {symbol}: {e}")
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
        
        # Calculate signal strength based on analysis
        signal_strength = 0.0  # Will be calculated based on analysis results
        confidence = 0.5       # Will be calculated based on data quality
        
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