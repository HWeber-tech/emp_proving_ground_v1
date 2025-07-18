"""
Order Flow Module - How Sense

This module handles order flow analysis and institutional flow detection
for the "how" sense.

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


class OrderFlowAnalyzer:
    """
    Order Flow Analyzer
    
    Analyzes institutional order flow and market microstructure.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize order flow analyzer"""
        self.config = config or {}
        self.flow_history = []
        logger.info("OrderFlowAnalyzer initialized")
    
    def analyze_institutional_flow(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze institutional order flow patterns.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Institutional flow analysis results
        """
        if df.empty:
            return {}
        
        try:
            analysis = {
                'flow_direction': self._determine_flow_direction(df),
                'flow_strength': self._calculate_flow_strength(df),
                'institutional_pressure': self._calculate_institutional_pressure(df),
                'absorption_levels': self._identify_absorption_levels(df),
                'distribution_levels': self._identify_distribution_levels(df),
                'timestamp': datetime.now()
            }
            
            self.flow_history.append(analysis)
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing institutional flow: {e}")
            return {}
    
    def _determine_flow_direction(self, df: pd.DataFrame) -> str:
        """Determine the direction of institutional flow"""
        try:
            if len(df) < 5:
                return "neutral"
            
            # Calculate volume-weighted price movement
            recent_data = df.tail(5)
            volume_weighted_change = (
                (recent_data['close'] - recent_data['open']) * recent_data['volume']
            ).sum()
            
            if volume_weighted_change > 0:
                return "bullish"
            elif volume_weighted_change < 0:
                return "bearish"
            else:
                return "neutral"
                
        except Exception as e:
            logger.error(f"Error determining flow direction: {e}")
            return "neutral"
    
    def _calculate_flow_strength(self, df: pd.DataFrame) -> float:
        """Calculate the strength of institutional flow"""
        try:
            if len(df) < 10:
                return 0.0
            
            # Calculate volume momentum
            volume_momentum = df['volume'].pct_change().tail(10).mean()
            
            # Calculate price momentum
            price_momentum = df['close'].pct_change().tail(10).mean()
            
            # Combine metrics
            flow_strength = (abs(volume_momentum) + abs(price_momentum)) / 2
            return min(max(flow_strength, 0.0), 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating flow strength: {e}")
            return 0.0
    
    def _calculate_institutional_pressure(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate institutional buying/selling pressure"""
        try:
            if len(df) < 5:
                return {'buying_pressure': 0.0, 'selling_pressure': 0.0}
            
            recent_data = df.tail(5)
            
            # Calculate buying pressure (volume on up moves)
            up_moves = recent_data[recent_data['close'] > recent_data['open']]
            buying_pressure = up_moves['volume'].sum() / recent_data['volume'].sum()
            
            # Calculate selling pressure (volume on down moves)
            down_moves = recent_data[recent_data['close'] < recent_data['open']]
            selling_pressure = down_moves['volume'].sum() / recent_data['volume'].sum()
            
            return {
                'buying_pressure': min(max(buying_pressure, 0.0), 1.0),
                'selling_pressure': min(max(selling_pressure, 0.0), 1.0)
            }
            
        except Exception as e:
            logger.error(f"Error calculating institutional pressure: {e}")
            return {'buying_pressure': 0.0, 'selling_pressure': 0.0}
    
    def _identify_absorption_levels(self, df: pd.DataFrame) -> List[float]:
        """Identify levels where selling is being absorbed"""
        try:
            absorption_levels = []
            
            for i in range(2, len(df) - 2):
                current = df.iloc[i]
                prev = df.iloc[i-1]
                next_candle = df.iloc[i+1]
                
                # Absorption pattern: high volume down move followed by reversal
                if (current['close'] < current['open'] and  # Down candle
                    current['volume'] > prev['volume'] * 1.2 and  # High volume
                    next_candle['close'] > current['close']):  # Reversal
                    
                    absorption_levels.append(current['low'])
            
            return absorption_levels[:5]  # Return top 5 levels
            
        except Exception as e:
            logger.error(f"Error identifying absorption levels: {e}")
            return []
    
    def _identify_distribution_levels(self, df: pd.DataFrame) -> List[float]:
        """Identify levels where buying is being distributed"""
        try:
            distribution_levels = []
            
            for i in range(2, len(df) - 2):
                current = df.iloc[i]
                prev = df.iloc[i-1]
                next_candle = df.iloc[i+1]
                
                # Distribution pattern: high volume up move followed by reversal
                if (current['close'] > current['open'] and  # Up candle
                    current['volume'] > prev['volume'] * 1.2 and  # High volume
                    next_candle['close'] < current['close']):  # Reversal
                    
                    distribution_levels.append(current['high'])
            
            return distribution_levels[:5]  # Return top 5 levels
            
        except Exception as e:
            logger.error(f"Error identifying distribution levels: {e}")
            return []


class MarketMicrostructureAnalyzer:
    """
    Market Microstructure Analyzer
    
    Analyzes market microstructure and institutional behavior patterns.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize market microstructure analyzer"""
        self.config = config or {}
        logger.info("MarketMicrostructureAnalyzer initialized")
    
    def analyze_microstructure(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze market microstructure patterns.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Microstructure analysis results
        """
        if df.empty:
            return {}
        
        try:
            analysis = {
                'spread_analysis': self._analyze_spreads(df),
                'depth_analysis': self._analyze_market_depth(df),
                'liquidity_analysis': self._analyze_liquidity(df),
                'volatility_analysis': self._analyze_volatility(df),
                'timestamp': datetime.now()
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing microstructure: {e}")
            return {}
    
    def _analyze_spreads(self, df: pd.DataFrame) -> Dict[str, float]:
        """Analyze bid-ask spread patterns"""
        try:
            # Calculate spread metrics (approximated from OHLC)
            spreads = []
            for _, candle in df.iterrows():
                # Approximate spread from candle range
                spread = (candle['high'] - candle['low']) / candle['close']
                spreads.append(spread)
            
            return {
                'avg_spread': np.mean(spreads) if spreads else 0.0,
                'spread_volatility': np.std(spreads) if spreads else 0.0,
                'max_spread': max(spreads) if spreads else 0.0,
                'min_spread': min(spreads) if spreads else 0.0
            }
            
        except Exception as e:
            logger.error(f"Error analyzing spreads: {e}")
            return {}
    
    def _analyze_market_depth(self, df: pd.DataFrame) -> Dict[str, float]:
        """Analyze market depth indicators"""
        try:
            # Calculate depth indicators
            volume_concentration = df['volume'].tail(10).std() / df['volume'].tail(10).mean()
            price_impact = df['close'].pct_change().abs().tail(10).mean()
            
            return {
                'volume_concentration': min(max(volume_concentration, 0.0), 1.0),
                'price_impact': min(max(price_impact, 0.0), 1.0),
                'depth_score': 1.0 - min(max(volume_concentration + price_impact, 0.0), 1.0)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing market depth: {e}")
            return {}
    
    def _analyze_liquidity(self, df: pd.DataFrame) -> Dict[str, float]:
        """Analyze liquidity conditions"""
        try:
            # Calculate liquidity metrics
            avg_volume = df['volume'].mean()
            volume_trend = df['volume'].tail(5).mean() / avg_volume if avg_volume > 0 else 1.0
            
            # Price efficiency (how quickly price reflects information)
            price_efficiency = 1.0 - df['close'].pct_change().abs().mean()
            
            return {
                'volume_trend': min(max(volume_trend, 0.0), 2.0),
                'price_efficiency': min(max(price_efficiency, 0.0), 1.0),
                'liquidity_score': min(max((volume_trend + price_efficiency) / 2, 0.0), 1.0)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing liquidity: {e}")
            return {}
    
    def _analyze_volatility(self, df: pd.DataFrame) -> Dict[str, float]:
        """Analyze volatility patterns"""
        try:
            # Calculate volatility metrics
            returns = df['close'].pct_change().dropna()
            
            if len(returns) < 5:
                return {}
            
            volatility = returns.std()
            realized_vol = returns.abs().mean()
            
            # Volatility clustering
            vol_clustering = returns.rolling(5).std().std()
            
            return {
                'volatility': min(max(volatility, 0.0), 1.0),
                'realized_volatility': min(max(realized_vol, 0.0), 1.0),
                'volatility_clustering': min(max(vol_clustering, 0.0), 1.0)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing volatility: {e}")
            return {}


# Example usage
if __name__ == "__main__":
    # Test order flow analyzer
    analyzer = OrderFlowAnalyzer()
    microstructure = MarketMicrostructureAnalyzer()
    print("Order flow analysis modules initialized successfully") 