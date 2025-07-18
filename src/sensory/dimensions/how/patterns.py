"""
Patterns Module - How Sense

This module handles ICT (Institutional Candle Theory) patterns and institutional
footprint analysis for the "how" sense.

Author: EMP Development Team
Date: July 18, 2024
Phase: 2 - Missing Function Implementation
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)


class ICTPatternDetector:
    """
    ICT (Institutional Candle Theory) Pattern Detector
    
    Detects institutional patterns and provides institutional footprint analysis.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize ICT pattern detector"""
        self.config = config or {}
        self.patterns_detected = []
        logger.info("ICTPatternDetector initialized")
    
    def get_institutional_footprint_score(self, df: pd.DataFrame) -> float:
        """
        Calculate institutional footprint score based on ICT patterns.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Institutional footprint score (0-1)
        """
        if df.empty:
            return 0.0
        
        try:
            # Calculate various ICT pattern scores
            fair_value_gap_score = self._detect_fair_value_gaps(df)
            order_block_score = self._detect_order_blocks(df)
            liquidity_grab_score = self._detect_liquidity_grabs(df)
            institutional_candle_score = self._detect_institutional_candles(df)
            
            # Weighted average of all scores
            footprint_score = (
                fair_value_gap_score * 0.3 +
                order_block_score * 0.3 +
                liquidity_grab_score * 0.2 +
                institutional_candle_score * 0.2
            )
            
            return min(max(footprint_score, 0.0), 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating institutional footprint score: {e}")
            return 0.0
    
    def update_market_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Update market data and detect ICT patterns.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Dictionary with pattern detection results
        """
        if df.empty:
            return {}
        
        try:
            patterns = {
                'fair_value_gaps': self._detect_fair_value_gaps(df),
                'order_blocks': self._detect_order_blocks(df),
                'liquidity_grabs': self._detect_liquidity_grabs(df),
                'institutional_candles': self._detect_institutional_candles(df),
                'footprint_score': self.get_institutional_footprint_score(df),
                'timestamp': datetime.now()
            }
            
            self.patterns_detected.append(patterns)
            return patterns
            
        except Exception as e:
            logger.error(f"Error updating market data: {e}")
            return {}
    
    def _detect_fair_value_gaps(self, df: pd.DataFrame) -> float:
        """Detect fair value gaps (FVG) in price action"""
        if len(df) < 3:
            return 0.0
        
        try:
            fvg_count = 0
            
            for i in range(1, len(df) - 1):
                # Bullish FVG: low of current candle > high of previous candle
                if df.iloc[i]['low'] > df.iloc[i-1]['high']:
                    fvg_count += 1
                
                # Bearish FVG: high of current candle < low of previous candle
                elif df.iloc[i]['high'] < df.iloc[i-1]['low']:
                    fvg_count += 1
            
            # Normalize by number of candles
            fvg_score = min(fvg_count / len(df), 1.0)
            return fvg_score
            
        except Exception as e:
            logger.error(f"Error detecting fair value gaps: {e}")
            return 0.0
    
    def _detect_order_blocks(self, df: pd.DataFrame) -> float:
        """Detect order blocks (institutional order zones)"""
        if len(df) < 5:
            return 0.0
        
        try:
            order_block_count = 0
            
            for i in range(2, len(df) - 2):
                current_candle = df.iloc[i]
                prev_candle = df.iloc[i-1]
                
                # Bullish order block: strong move up after consolidation
                if (current_candle['close'] > current_candle['open'] and
                    current_candle['close'] - current_candle['open'] > 
                    prev_candle['high'] - prev_candle['low'] * 0.5):
                    order_block_count += 1
                
                # Bearish order block: strong move down after consolidation
                elif (current_candle['close'] < current_candle['open'] and
                      current_candle['open'] - current_candle['close'] > 
                      prev_candle['high'] - prev_candle['low'] * 0.5):
                    order_block_count += 1
            
            # Normalize by number of candles
            ob_score = min(order_block_count / len(df), 1.0)
            return ob_score
            
        except Exception as e:
            logger.error(f"Error detecting order blocks: {e}")
            return 0.0
    
    def _detect_liquidity_grabs(self, df: pd.DataFrame) -> float:
        """Detect liquidity grabs (stop hunts)"""
        if len(df) < 3:
            return 0.0
        
        try:
            liquidity_grab_count = 0
            
            for i in range(1, len(df) - 1):
                current_candle = df.iloc[i]
                prev_candle = df.iloc[i-1]
                next_candle = df.iloc[i+1]
                
                # Bullish liquidity grab: wick below support then reversal
                if (current_candle['low'] < prev_candle['low'] and
                    current_candle['close'] > current_candle['open'] and
                    next_candle['close'] > current_candle['high']):
                    liquidity_grab_count += 1
                
                # Bearish liquidity grab: wick above resistance then reversal
                elif (current_candle['high'] > prev_candle['high'] and
                      current_candle['close'] < current_candle['open'] and
                      next_candle['close'] < current_candle['low']):
                    liquidity_grab_count += 1
            
            # Normalize by number of candles
            lg_score = min(liquidity_grab_count / len(df), 1.0)
            return lg_score
            
        except Exception as e:
            logger.error(f"Error detecting liquidity grabs: {e}")
            return 0.0
    
    def _detect_institutional_candles(self, df: pd.DataFrame) -> float:
        """Detect institutional candles (large volume, strong moves)"""
        if df.empty:
            return 0.0
        
        try:
            # Calculate average volume and price range
            avg_volume = df['volume'].mean()
            avg_range = (df['high'] - df['low']).mean()
            
            institutional_candle_count = 0
            
            for _, candle in df.iterrows():
                # Institutional candle criteria
                volume_factor = candle['volume'] / avg_volume if avg_volume > 0 else 1
                range_factor = (candle['high'] - candle['low']) / avg_range if avg_range > 0 else 1
                
                # Strong institutional candle
                if volume_factor > 1.5 and range_factor > 1.2:
                    institutional_candle_count += 1
            
            # Normalize by number of candles
            ic_score = min(institutional_candle_count / len(df), 1.0)
            return ic_score
            
        except Exception as e:
            logger.error(f"Error detecting institutional candles: {e}")
            return 0.0


class OrderFlowDataProvider:
    """
    Order Flow Data Provider
    
    Provides order book analysis and institutional flow data.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize order flow data provider"""
        self.config = config or {}
        self.order_book_snapshots = []
        logger.info("OrderFlowDataProvider initialized")
    
    def get_latest_snapshot(self) -> Optional[Dict[str, Any]]:
        """
        Get the latest order book snapshot.
        
        Returns:
            Latest order book snapshot or None
        """
        if not self.order_book_snapshots:
            return None
        
        return self.order_book_snapshots[-1]
    
    def update_order_book(self, snapshot: Dict[str, Any]) -> None:
        """
        Update order book with new snapshot.
        
        Args:
            snapshot: Order book snapshot data
        """
        try:
            self.order_book_snapshots.append({
                **snapshot,
                'timestamp': datetime.now()
            })
            
            # Keep only recent snapshots
            if len(self.order_book_snapshots) > 100:
                self.order_book_snapshots = self.order_book_snapshots[-100:]
                
        except Exception as e:
            logger.error(f"Error updating order book: {e}")
    
    def analyze_order_flow(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze order flow patterns.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Order flow analysis results
        """
        if df.empty:
            return {}
        
        try:
            analysis = {
                'volume_profile': self._calculate_volume_profile(df),
                'order_imbalance': self._calculate_order_imbalance(df),
                'flow_strength': self._calculate_flow_strength(df),
                'timestamp': datetime.now()
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing order flow: {e}")
            return {}
    
    def _calculate_volume_profile(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate volume profile analysis"""
        try:
            # Volume-weighted average price
            vwap = (df['close'] * df['volume']).sum() / df['volume'].sum()
            
            # Volume distribution
            above_vwap = df[df['close'] > vwap]['volume'].sum()
            below_vwap = df[df['close'] < vwap]['volume'].sum()
            
            return {
                'vwap': vwap,
                'above_vwap_volume': above_vwap,
                'below_vwap_volume': below_vwap,
                'volume_ratio': above_vwap / below_vwap if below_vwap > 0 else 1.0
            }
            
        except Exception as e:
            logger.error(f"Error calculating volume profile: {e}")
            return {}
    
    def _calculate_order_imbalance(self, df: pd.DataFrame) -> float:
        """Calculate order imbalance score"""
        try:
            # Simple order imbalance based on price movement vs volume
            price_changes = df['close'].diff().abs()
            volume_weights = df['volume'] / df['volume'].sum()
            
            imbalance = (price_changes * volume_weights).sum()
            return min(max(imbalance, 0.0), 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating order imbalance: {e}")
            return 0.0
    
    def _calculate_flow_strength(self, df: pd.DataFrame) -> float:
        """Calculate institutional flow strength"""
        try:
            # Flow strength based on volume and price momentum
            volume_momentum = df['volume'].pct_change().mean()
            price_momentum = df['close'].pct_change().mean()
            
            flow_strength = (volume_momentum + price_momentum) / 2
            return min(max(flow_strength, 0.0), 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating flow strength: {e}")
            return 0.0


# Example usage
if __name__ == "__main__":
    # Test ICT pattern detector
    detector = ICTPatternDetector()
    provider = OrderFlowDataProvider()
    print("Pattern detection modules initialized successfully") 