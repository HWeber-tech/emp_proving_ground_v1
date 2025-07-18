"""
Price Action Module - What Sense

This module handles price action analysis and technical reality assessment
for the "what" sense.

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


class PriceActionAnalyzer:
    """
    Price Action Analyzer
    
    Analyzes pure price action patterns and market structure.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize price action analyzer"""
        self.config = config or {}
        self.analysis_history = []
        logger.info("PriceActionAnalyzer initialized")
    
    def get_price_action_score(self, df: pd.DataFrame) -> float:
        """
        Calculate price action score based on market structure.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Price action score (0-1)
        """
        if df.empty:
            return 0.0
        
        try:
            # Calculate various price action components
            trend_strength = self._calculate_trend_strength(df)
            momentum_score = self._calculate_momentum_score(df)
            volatility_score = self._calculate_volatility_score(df)
            structure_score = self._calculate_structure_score(df)
            
            # Weighted average of all scores
            price_action_score = (
                trend_strength * 0.3 +
                momentum_score * 0.25 +
                volatility_score * 0.2 +
                structure_score * 0.25
            )
            
            return min(max(price_action_score, 0.0), 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating price action score: {e}")
            return 0.0
    
    def update_market_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Update market data and analyze price action.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Price action analysis results
        """
        if df.empty:
            return {}
        
        try:
            analysis = {
                'price_action_score': self.get_price_action_score(df),
                'trend_analysis': self._analyze_trend(df),
                'momentum_analysis': self._analyze_momentum(df),
                'volatility_analysis': self._analyze_volatility(df),
                'structure_analysis': self._analyze_structure(df),
                'support_resistance': self._identify_support_resistance(df),
                'timestamp': datetime.now()
            }
            
            self.analysis_history.append(analysis)
            return analysis
            
        except Exception as e:
            logger.error(f"Error updating market data: {e}")
            return {}
    
    def _calculate_trend_strength(self, df: pd.DataFrame) -> float:
        """Calculate trend strength"""
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
            
            # Normalize and consider slope direction
            trend_strength = abs(slope) * r_squared
            return min(max(trend_strength, 0.0), 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating trend strength: {e}")
            return 0.0
    
    def _calculate_momentum_score(self, df: pd.DataFrame) -> float:
        """Calculate momentum score"""
        try:
            if len(df) < 5:
                return 0.0
            
            # Calculate momentum indicators
            price_momentum = df['close'].pct_change().tail(5).mean()
            volume_momentum = df['volume'].pct_change().tail(5).mean()
            
            # Combine momentum signals
            momentum_score = (abs(price_momentum) + abs(volume_momentum)) / 2
            return min(max(momentum_score, 0.0), 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating momentum score: {e}")
            return 0.0
    
    def _calculate_volatility_score(self, df: pd.DataFrame) -> float:
        """Calculate volatility score"""
        try:
            if len(df) < 10:
                return 0.0
            
            # Calculate volatility metrics
            returns = df['close'].pct_change().dropna()
            volatility = returns.std()
            
            # Normalize volatility
            volatility_score = min(volatility * 10, 1.0)  # Scale factor
            return volatility_score
            
        except Exception as e:
            logger.error(f"Error calculating volatility score: {e}")
            return 0.0
    
    def _calculate_structure_score(self, df: pd.DataFrame) -> float:
        """Calculate market structure score"""
        try:
            if len(df) < 10:
                return 0.0
            
            # Analyze market structure
            highs = df['high'].rolling(5).max()
            lows = df['low'].rolling(5).min()
            
            # Higher highs and higher lows (uptrend structure)
            higher_highs = (highs.diff() > 0).sum() / len(df)
            higher_lows = (lows.diff() > 0).sum() / len(df)
            
            # Lower highs and lower lows (downtrend structure)
            lower_highs = (highs.diff() < 0).sum() / len(df)
            lower_lows = (lows.diff() < 0).sum() / len(df)
            
            # Structure score based on trend consistency
            uptrend_score = (higher_highs + higher_lows) / 2
            downtrend_score = (lower_highs + lower_lows) / 2
            
            structure_score = max(uptrend_score, downtrend_score)
            return min(max(structure_score, 0.0), 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating structure score: {e}")
            return 0.0
    
    def _analyze_trend(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze trend characteristics"""
        try:
            if len(df) < 10:
                return {}
            
            # Calculate trend metrics
            trend_strength = self._calculate_trend_strength(df)
            
            # Determine trend direction
            recent_prices = df['close'].tail(5)
            trend_direction = "bullish" if recent_prices.iloc[-1] > recent_prices.iloc[0] else "bearish"
            
            return {
                'trend_strength': trend_strength,
                'trend_direction': trend_direction,
                'trend_duration': len(df)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing trend: {e}")
            return {}
    
    def _analyze_momentum(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze momentum characteristics"""
        try:
            if len(df) < 5:
                return {}
            
            # Calculate momentum metrics
            price_momentum = df['close'].pct_change().tail(5).mean()
            volume_momentum = df['volume'].pct_change().tail(5).mean()
            
            return {
                'price_momentum': price_momentum,
                'volume_momentum': volume_momentum,
                'momentum_score': self._calculate_momentum_score(df)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing momentum: {e}")
            return {}
    
    def _analyze_volatility(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze volatility characteristics"""
        try:
            if len(df) < 10:
                return {}
            
            # Calculate volatility metrics
            returns = df['close'].pct_change().dropna()
            volatility = returns.std()
            realized_vol = returns.abs().mean()
            
            return {
                'volatility': volatility,
                'realized_volatility': realized_vol,
                'volatility_score': self._calculate_volatility_score(df)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing volatility: {e}")
            return {}
    
    def _analyze_structure(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze market structure"""
        try:
            if len(df) < 10:
                return {}
            
            # Analyze structure patterns
            highs = df['high'].rolling(5).max()
            lows = df['low'].rolling(5).min()
            
            higher_highs = (highs.diff() > 0).sum()
            higher_lows = (lows.diff() > 0).sum()
            lower_highs = (highs.diff() < 0).sum()
            lower_lows = (lows.diff() < 0).sum()
            
            return {
                'higher_highs': higher_highs,
                'higher_lows': higher_lows,
                'lower_highs': lower_highs,
                'lower_lows': lower_lows,
                'structure_score': self._calculate_structure_score(df)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing structure: {e}")
            return {}
    
    def _identify_support_resistance(self, df: pd.DataFrame) -> Dict[str, List[float]]:
        """Identify support and resistance levels"""
        try:
            if len(df) < 20:
                return {'support_levels': [], 'resistance_levels': []}
            
            # Simple support/resistance identification
            support_levels = []
            resistance_levels = []
            
            # Look for local minima (support) and maxima (resistance)
            for i in range(2, len(df) - 2):
                current_low = df.iloc[i]['low']
                current_high = df.iloc[i]['high']
                
                # Support level (local minimum)
                if (current_low < df.iloc[i-1]['low'] and 
                    current_low < df.iloc[i-2]['low'] and
                    current_low < df.iloc[i+1]['low'] and 
                    current_low < df.iloc[i+2]['low']):
                    support_levels.append(current_low)
                
                # Resistance level (local maximum)
                if (current_high > df.iloc[i-1]['high'] and 
                    current_high > df.iloc[i-2]['high'] and
                    current_high > df.iloc[i+1]['high'] and 
                    current_high > df.iloc[i+2]['high']):
                    resistance_levels.append(current_high)
            
            # Remove duplicates and sort
            support_levels = sorted(list(set(support_levels)))
            resistance_levels = sorted(list(set(resistance_levels)))
            
            return {
                'support_levels': support_levels[:5],  # Top 5 levels
                'resistance_levels': resistance_levels[:5]  # Top 5 levels
            }
            
        except Exception as e:
            logger.error(f"Error identifying support/resistance: {e}")
            return {'support_levels': [], 'resistance_levels': []}


# Example usage
if __name__ == "__main__":
    # Test price action analyzer
    analyzer = PriceActionAnalyzer()
    print("Price Action Analyzer initialized successfully") 