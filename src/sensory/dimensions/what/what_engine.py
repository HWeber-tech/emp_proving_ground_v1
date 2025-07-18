"""
What Engine - Technical Reality and Market Structure Engine

This is the main engine for the "what" sense that handles technical reality analysis,
market structure, and price action analysis.

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


class WhatEngine:
    """
    Main engine for technical reality and market structure analysis.
    
    This engine processes market data to understand WHAT the market is doing,
    including technical analysis, market structure, and regime detection.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the what engine with configuration"""
        self.config = config or {}
        logger.info("What Engine initialized")
    
    def analyze_market_data(self, market_data: List[MarketData], 
                          symbol: str = "UNKNOWN") -> Dict[str, Any]:
        """
        Perform comprehensive technical reality analysis on market data.
        
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
                'technical_reality': {},  # Will be implemented in technical_reality sub-module
                'price_action': {},       # Will be implemented in price_action sub-module
                'market_structure': {},   # Will be implemented in market_structure sub-module
                'regime_detection': {}    # Will be implemented in regime_detection sub-module
            }
            
            logger.info(f"Technical reality analysis completed for {symbol}")
            
            return analysis_results
            
        except Exception as e:
            logger.error(f"Error in technical reality analysis for {symbol}: {e}")
            return {}
    
    def get_dimensional_reading(self, market_data: List[MarketData], 
                              symbol: str = "UNKNOWN") -> DimensionalReading:
        """
        Get a dimensional reading for the what sense.
        
        Args:
            market_data: List of market data points
            symbol: Symbol being analyzed
            
        Returns:
            DimensionalReading with what sense analysis
        """
        analysis = self.analyze_market_data(market_data, symbol)
        
        # Calculate signal strength based on analysis
        signal_strength = 0.0  # Will be calculated based on analysis results
        confidence = 0.5       # Will be calculated based on data quality
        
        return DimensionalReading(
            dimension="WHAT",
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
    # Test the what engine
    engine = WhatEngine()
    print("What Engine initialized successfully") 