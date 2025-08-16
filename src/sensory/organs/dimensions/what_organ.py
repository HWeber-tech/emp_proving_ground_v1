"""
What Engine - Technical Reality and Market Structure Engine

This is the main engine for the "what" sense that handles technical reality analysis,
market structure, and price action analysis.

Author: EMP Development Team
Date: July 18, 2024
Phase: 2 - Sensory Cortex Refactoring
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

from src.sensory.core.base import DimensionalReading, MarketData, MarketRegime
from src.sensory.what.patterns.orchestrator import PatternOrchestrator

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
        
        # Initialize sub-modules
        try:
            from .price_action import PriceActionAnalyzer
            
            self.price_action = PriceActionAnalyzer()
            
            logger.info("What Engine initialized with sub-modules")
        except ImportError as e:
            logger.warning(f"Some sub-modules not available: {e}")
            self.price_action = None

        # Pattern synthesis orchestrator (async engine behind a thin façade)
        self.pattern_orchestrator = PatternOrchestrator()
    
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
                'price_action': self._analyze_price_action(df),
                'technical_reality': self._analyze_technical_reality(df),
                'market_structure': self._analyze_market_structure(df),
                'support_resistance': self._identify_support_resistance(df),
            }

            # Best-effort pattern synthesis (async); avoid nesting event loops
            try:
                import asyncio

                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    loop = None

                if loop and loop.is_running():
                    # In an async context already; skip sync orchestration to avoid nested loop issues.
                    patterns: Dict[str, Any] = {}
                else:
                    patterns = asyncio.run(self.pattern_orchestrator.analyze(df))
                analysis_results['pattern_synthesis'] = patterns
            except Exception as _ex:
                # Non-fatal: keep the rest of the analysis
                analysis_results['pattern_synthesis'] = {}

            logger.info(f"Technical reality analysis completed for {symbol}")
            
            return analysis_results
            
        except Exception as e:
            logger.error(f"Error in technical reality analysis for {symbol}: {e}")
            return {}
    
    def analyze_technical_reality(self, market_data: List[MarketData], 
                                symbol: str = "UNKNOWN") -> Dict[str, Any]:
        """
        Analyze technical reality and market structure.
        
        Args:
            market_data: List of market data points
            symbol: Symbol being analyzed
            
        Returns:
            Technical reality analysis results
        """
        if not market_data:
            return {}
        
        try:
            df = self._market_data_to_dataframe(market_data)
            
            analysis = {
                'symbol': symbol,
                'timestamp': datetime.now(),
                'price_action_score': self._get_price_action_score(df),
                'market_structure': self._analyze_market_structure(df),
                'technical_indicators': self._calculate_technical_indicators(df),
                'market_regime': self._determine_market_regime(df)
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error in technical reality analysis: {e}")
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
        signal_strength = self._calculate_signal_strength(analysis)
        confidence = self._calculate_confidence(analysis)
        
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
    
    def _analyze_price_action(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze price action"""
        if self.price_action is None:
            return {}
        
        try:
            return self.price_action.update_market_data(df)
        except Exception as e:
            logger.error(f"Error analyzing price action: {e}")
            return {}
    
    def _analyze_technical_reality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze technical reality"""
        try:
            return {
                'price_action_score': self._get_price_action_score(df),
                'market_structure': self._analyze_market_structure(df),
                'technical_indicators': self._calculate_technical_indicators(df),
                'market_regime': self._determine_market_regime(df)
            }
        except Exception as e:
            logger.error(f"Error analyzing technical reality: {e}")
            return {}
    
    def _analyze_market_structure(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze market structure"""
        try:
            if len(df) < 10:
                return {}
            
            # Analyze market structure
            highs = df['high'].rolling(5).max()
            lows = df['low'].rolling(5).min()
            
            # Structure analysis
            higher_highs = (highs.diff() > 0).sum()
            higher_lows = (lows.diff() > 0).sum()
            lower_highs = (highs.diff() < 0).sum()
            lower_lows = (lows.diff() < 0).sum()
            
            # Determine structure type
            if higher_highs > lower_highs and higher_lows > lower_lows:
                structure_type = "uptrend"
            elif lower_highs > higher_highs and lower_lows > higher_lows:
                structure_type = "downtrend"
            else:
                structure_type = "sideways"
            
            return {
                'structure_type': structure_type,
                'higher_highs': higher_highs,
                'higher_lows': higher_lows,
                'lower_highs': lower_highs,
                'lower_lows': lower_lows,
                'structure_strength': self._calculate_structure_strength(df)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing market structure: {e}")
            return {}
    
    def _identify_support_resistance(self, df: pd.DataFrame) -> Dict[str, List[float]]:
        """Identify support and resistance levels"""
        if self.price_action is None:
            return {'support_levels': [], 'resistance_levels': []}
        
        try:
            return self.price_action._identify_support_resistance(df)
        except Exception as e:
            logger.error(f"Error identifying support/resistance: {e}")
            return {'support_levels': [], 'resistance_levels': []}
    
    def _get_price_action_score(self, df: pd.DataFrame) -> float:
        """Get price action score"""
        if self.price_action is None:
            return 0.0
        
        try:
            return self.price_action.get_price_action_score(df)
        except Exception as e:
            logger.error(f"Error getting price action score: {e}")
            return 0.0
    
    def _calculate_technical_indicators(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate basic technical indicators"""
        try:
            if len(df) < 20:
                return {}
            
            # Simple moving averages
            sma_20 = df['close'].rolling(20).mean().iloc[-1]
            sma_50 = df['close'].rolling(50).mean().iloc[-1] if len(df) >= 50 else sma_20
            
            # RSI approximation
            returns = df['close'].pct_change().dropna()
            gains = returns.where(returns > 0, 0)
            losses = -returns.where(returns < 0, 0)
            avg_gain = gains.rolling(14).mean().iloc[-1]
            avg_loss = losses.rolling(14).mean().iloc[-1]
            rs = avg_gain / avg_loss if avg_loss > 0 else 0
            rsi = 100 - (100 / (1 + rs))
            
            return {
                'sma_20': sma_20,
                'sma_50': sma_50,
                'rsi': rsi,
                'current_price': df['close'].iloc[-1]
            }
            
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {e}")
            return {}
    
    def _determine_market_regime(self, df: pd.DataFrame) -> str:
        """Determine market regime"""
        try:
            if len(df) < 20:
                return "unknown"
            
            # Calculate volatility
            returns = df['close'].pct_change().dropna()
            volatility = returns.std()
            
            # Calculate trend strength
            sma_20 = df['close'].rolling(20).mean()
            trend_strength = abs(df['close'].iloc[-1] - sma_20.iloc[-1]) / sma_20.iloc[-1]
            
            # Determine regime
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
            logger.error(f"Error determining market regime: {e}")
            return "unknown"
    
    def _calculate_structure_strength(self, df: pd.DataFrame) -> float:
        """Calculate market structure strength"""
        try:
            if len(df) < 10:
                return 0.0
            
            # Calculate structure consistency
            highs = df['high'].rolling(5).max()
            lows = df['low'].rolling(5).min()
            
            higher_highs = (highs.diff() > 0).sum()
            higher_lows = (lows.diff() > 0).sum()
            lower_highs = (highs.diff() < 0).sum()
            lower_lows = (lows.diff() < 0).sum()
            
            # Structure strength based on consistency
            total_patterns = higher_highs + higher_lows + lower_highs + lower_lows
            if total_patterns == 0:
                return 0.0
            
            # Calculate consistency score
            max_pattern = max(higher_highs, higher_lows, lower_highs, lower_lows)
            consistency = max_pattern / total_patterns
            
            return min(max(consistency, 0.0), 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating structure strength: {e}")
            return 0.0
    
    def _calculate_signal_strength(self, analysis: Dict[str, Any]) -> float:
        """Calculate signal strength from analysis results"""
        try:
            # Combine various analysis components
            price_action_score = analysis.get('price_action', {}).get('price_action_score', 0.0)
            structure_strength = analysis.get('technical_reality', {}).get('market_structure', {}).get('structure_strength', 0.0)
            
            # Calculate weighted signal strength
            signal_strength = (price_action_score * 0.6 + structure_strength * 0.4)
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
            if analysis.get('price_action') and analysis.get('technical_reality'):
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
    # Test the what engine
    engine = WhatEngine()
    print("What Engine initialized successfully") 
