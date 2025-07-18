#!/usr/bin/env python3
"""
Advanced Pattern Recognition - Real Market Pattern Detection

This module implements real pattern recognition using actual market data
to identify trading patterns: triangles, flags, head & shoulders, etc.
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import warnings

logger = logging.getLogger(__name__)


class PatternType(Enum):
    """Trading pattern types."""
    ASCENDING_TRIANGLE = "ascending_triangle"
    DESCENDING_TRIANGLE = "descending_triangle"
    SYMMETRICAL_TRIANGLE = "symmetrical_triangle"
    BULL_FLAG = "bull_flag"
    BEAR_FLAG = "bear_flag"
    HEAD_SHOULDERS = "head_shoulders"
    INVERSE_HEAD_SHOULDERS = "inverse_head_shoulders"
    DOUBLE_TOP = "double_top"
    DOUBLE_BOTTOM = "double_bottom"
    WEDGE = "wedge"
    CHANNEL = "channel"
    UNKNOWN = "unknown"


@dataclass
class PatternResult:
    """Result of pattern recognition analysis."""
    pattern_type: PatternType
    confidence: float
    start_time: datetime
    end_time: datetime
    breakout_price: Optional[float]
    target_price: Optional[float]
    stop_loss: Optional[float]
    description: str
    metrics: Dict[str, float]


class AdvancedPatternRecognition:
    """
    Advanced pattern recognition using real market data.
    
    This replaces synthetic pattern detection with real analysis
    of price action and chart patterns.
    """
    
    def __init__(self, min_pattern_bars: int = 10, max_pattern_bars: int = 100):
        """
        Initialize the pattern recognition system.
        
        Args:
            min_pattern_bars: Minimum bars for pattern formation
            max_pattern_bars: Maximum bars for pattern formation
        """
        self.min_pattern_bars = min_pattern_bars
        self.max_pattern_bars = max_pattern_bars
        
        logger.info(f"Advanced pattern recognition initialized")
    
    def detect_patterns(self, data: pd.DataFrame, symbol: str = "UNKNOWN") -> List[PatternResult]:
        """
        Detect trading patterns in market data.
        
        Args:
            data: OHLCV market data
            symbol: Trading symbol
            
        Returns:
            List of detected patterns
        """
        if data.empty or len(data) < self.min_pattern_bars:
            logger.warning(f"Insufficient data for pattern detection: {len(data)} records")
            return []
        
        try:
            patterns = []
            
            # Detect different pattern types
            patterns.extend(self._detect_triangles(data, symbol))
            patterns.extend(self._detect_flags(data, symbol))
            patterns.extend(self._detect_head_shoulders(data, symbol))
            patterns.extend(self._detect_double_patterns(data, symbol))
            patterns.extend(self._detect_channels(data, symbol))
            
            # Sort by confidence
            patterns.sort(key=lambda p: p.confidence, reverse=True)
            
            logger.info(f"Detected {len(patterns)} patterns for {symbol}")
            return patterns
            
        except Exception as e:
            logger.error(f"Error detecting patterns for {symbol}: {e}")
            return []
    
    def _detect_triangles(self, data: pd.DataFrame, symbol: str) -> List[PatternResult]:
        """Detect triangle patterns."""
        patterns = []
        
        # Look for triangle patterns in recent data
        for i in range(self.min_pattern_bars, min(len(data), self.max_pattern_bars)):
            window_data = data.iloc[-i:]
            
            # Get highs and lows
            highs = window_data['high'].values
            lows = window_data['low'].values
            closes = window_data['close'].values
            
            # Find peaks and troughs
            peaks = self._find_peaks(highs)
            troughs = self._find_peaks(-lows)  # Invert lows to find troughs
            
            if len(peaks) >= 3 and len(troughs) >= 3:
                # Analyze trend lines
                upper_trend = self._fit_trend_line(peaks, highs[peaks])
                lower_trend = self._fit_trend_line(troughs, lows[troughs])
                
                if upper_trend and lower_trend:
                    upper_slope = upper_trend['slope']
                    lower_slope = lower_trend['slope']
                    
                    # Determine triangle type
                    if abs(upper_slope) < 0.001 and lower_slope > 0.001:
                        # Ascending triangle
                        pattern = self._create_triangle_pattern(
                            PatternType.ASCENDING_TRIANGLE, window_data, symbol,
                            upper_trend, lower_trend, "Ascending triangle formation"
                        )
                        patterns.append(pattern)
                    
                    elif upper_slope < -0.001 and abs(lower_slope) < 0.001:
                        # Descending triangle
                        pattern = self._create_triangle_pattern(
                            PatternType.DESCENDING_TRIANGLE, window_data, symbol,
                            upper_trend, lower_trend, "Descending triangle formation"
                        )
                        patterns.append(pattern)
                    
                    elif abs(upper_slope - lower_slope) < 0.002:
                        # Symmetrical triangle
                        pattern = self._create_triangle_pattern(
                            PatternType.SYMMETRICAL_TRIANGLE, window_data, symbol,
                            upper_trend, lower_trend, "Symmetrical triangle formation"
                        )
                        patterns.append(pattern)
        
        return patterns
    
    def _detect_flags(self, data: pd.DataFrame, symbol: str) -> List[PatternResult]:
        """Detect flag patterns."""
        patterns = []
        
        for i in range(self.min_pattern_bars, min(len(data), self.max_pattern_bars)):
            window_data = data.iloc[-i:]
            
            # Check for strong move followed by consolidation
            first_half = window_data.iloc[:len(window_data)//2]
            second_half = window_data.iloc[len(window_data)//2:]
            
            # Calculate momentum
            first_momentum = self._calculate_momentum(first_half)
            second_momentum = self._calculate_momentum(second_half)
            
            # Check for flag conditions
            if abs(first_momentum) > 0.02 and abs(second_momentum) < 0.005:
                # Strong move followed by consolidation
                if first_momentum > 0:
                    pattern_type = PatternType.BULL_FLAG
                    description = "Bull flag pattern"
                else:
                    pattern_type = PatternType.BEAR_FLAG
                    description = "Bear flag pattern"
                
                pattern = PatternResult(
                    pattern_type=pattern_type,
                    confidence=0.7,
                    start_time=window_data.index[0],
                    end_time=window_data.index[-1],
                    breakout_price=window_data['close'].iloc[-1],
                    target_price=self._calculate_flag_target(window_data, first_momentum),
                    stop_loss=self._calculate_flag_stop_loss(window_data, first_momentum),
                    description=description,
                    metrics={'momentum': first_momentum, 'consolidation': second_momentum}
                )
                patterns.append(pattern)
        
        return patterns
    
    def _detect_head_shoulders(self, data: pd.DataFrame, symbol: str) -> List[PatternResult]:
        """Detect head and shoulders patterns."""
        patterns = []
        
        for i in range(self.min_pattern_bars, min(len(data), self.max_pattern_bars)):
            window_data = data.iloc[-i:]
            highs = window_data['high'].values
            lows = window_data['low'].values
            
            # Find peaks for potential shoulders and head
            peaks = self._find_peaks(highs)
            troughs = self._find_peaks(-lows)
            
            if len(peaks) >= 5 and len(troughs) >= 3:
                # Check for head and shoulders pattern
                hs_pattern = self._identify_head_shoulders(peaks, highs, troughs, lows)
                if hs_pattern:
                    pattern = PatternResult(
                        pattern_type=PatternType.HEAD_SHOULDERS,
                        confidence=hs_pattern['confidence'],
                        start_time=window_data.index[0],
                        end_time=window_data.index[-1],
                        breakout_price=hs_pattern['breakout_price'],
                        target_price=hs_pattern['target_price'],
                        stop_loss=hs_pattern['stop_loss'],
                        description="Head and shoulders pattern",
                        metrics=hs_pattern['metrics']
                    )
                    patterns.append(pattern)
        
        return patterns
    
    def _detect_double_patterns(self, data: pd.DataFrame, symbol: str) -> List[PatternResult]:
        """Detect double top and double bottom patterns."""
        patterns = []
        
        for i in range(self.min_pattern_bars, min(len(data), self.max_pattern_bars)):
            window_data = data.iloc[-i:]
            highs = window_data['high'].values
            lows = window_data['low'].values
            
            # Find peaks and troughs
            peaks = self._find_peaks(highs)
            troughs = self._find_peaks(-lows)
            
            if len(peaks) >= 4:
                # Check for double top
                double_top = self._identify_double_top(peaks, highs)
                if double_top:
                    pattern = PatternResult(
                        pattern_type=PatternType.DOUBLE_TOP,
                        confidence=double_top['confidence'],
                        start_time=window_data.index[0],
                        end_time=window_data.index[-1],
                        breakout_price=double_top['breakout_price'],
                        target_price=double_top['target_price'],
                        stop_loss=double_top['stop_loss'],
                        description="Double top pattern",
                        metrics=double_top['metrics']
                    )
                    patterns.append(pattern)
            
            if len(troughs) >= 4:
                # Check for double bottom
                double_bottom = self._identify_double_bottom(troughs, lows)
                if double_bottom:
                    pattern = PatternResult(
                        pattern_type=PatternType.DOUBLE_BOTTOM,
                        confidence=double_bottom['confidence'],
                        start_time=window_data.index[0],
                        end_time=window_data.index[-1],
                        breakout_price=double_bottom['breakout_price'],
                        target_price=double_bottom['target_price'],
                        stop_loss=double_bottom['stop_loss'],
                        description="Double bottom pattern",
                        metrics=double_bottom['metrics']
                    )
                    patterns.append(pattern)
        
        return patterns
    
    def _detect_channels(self, data: pd.DataFrame, symbol: str) -> List[PatternResult]:
        """Detect channel patterns."""
        patterns = []
        
        for i in range(self.min_pattern_bars, min(len(data), self.max_pattern_bars)):
            window_data = data.iloc[-i:]
            
            # Fit parallel lines to highs and lows
            highs = window_data['high'].values
            lows = window_data['low'].values
            
            # Find trend lines
            upper_trend = self._fit_trend_line_to_highs(window_data)
            lower_trend = self._fit_trend_line_to_lows(window_data)
            
            if upper_trend and lower_trend:
                # Check if lines are parallel
                slope_diff = abs(upper_trend['slope'] - lower_trend['slope'])
                if slope_diff < 0.001:  # Parallel lines
                    pattern = PatternResult(
                        pattern_type=PatternType.CHANNEL,
                        confidence=0.75,
                        start_time=window_data.index[0],
                        end_time=window_data.index[-1],
                        breakout_price=upper_trend['current_price'],
                        target_price=self._calculate_channel_target(upper_trend, lower_trend),
                        stop_loss=lower_trend['current_price'],
                        description="Channel pattern",
                        metrics={
                            'upper_slope': upper_trend['slope'],
                            'lower_slope': lower_trend['slope'],
                            'channel_width': upper_trend['current_price'] - lower_trend['current_price']
                        }
                    )
                    patterns.append(pattern)
        
        return patterns
    
    def _find_peaks(self, data: np.ndarray, window: int = 3) -> List[int]:
        """Find peaks in data using sliding window."""
        peaks = []
        for i in range(window, len(data) - window):
            if all(data[i] >= data[j] for j in range(i - window, i)) and \
               all(data[i] >= data[j] for j in range(i + 1, i + window + 1)):
                peaks.append(i)
        return peaks
    
    def _fit_trend_line(self, x_points: List[int], y_points: List[float]) -> Optional[Dict[str, float]]:
        """Fit a trend line to points."""
        if len(x_points) < 2:
            return None
        
        try:
            x = np.array(x_points)
            y = np.array(y_points)
            
            # Linear regression
            slope, intercept = np.polyfit(x, y, 1)
            
            # Calculate R-squared
            y_pred = slope * x + intercept
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            
            return {
                'slope': slope,
                'intercept': intercept,
                'r_squared': r_squared,
                'current_price': slope * (len(x_points) - 1) + intercept
            }
        except:
            return None
    
    def _fit_trend_line_to_highs(self, data: pd.DataFrame) -> Optional[Dict[str, float]]:
        """Fit trend line to highs."""
        highs = data['high'].values
        peaks = self._find_peaks(highs)
        
        if len(peaks) >= 2:
            return self._fit_trend_line(peaks, highs[peaks])
        return None
    
    def _fit_trend_line_to_lows(self, data: pd.DataFrame) -> Optional[Dict[str, float]]:
        """Fit trend line to lows."""
        lows = data['low'].values
        troughs = self._find_peaks(-lows)
        
        if len(troughs) >= 2:
            return self._fit_trend_line(troughs, lows[troughs])
        return None
    
    def _calculate_momentum(self, data: pd.DataFrame) -> float:
        """Calculate price momentum."""
        if len(data) < 2:
            return 0
        
        start_price = data['close'].iloc[0]
        end_price = data['close'].iloc[-1]
        return (end_price - start_price) / start_price
    
    def _create_triangle_pattern(self, pattern_type: PatternType, data: pd.DataFrame, 
                               symbol: str, upper_trend: Dict, lower_trend: Dict, 
                               description: str) -> PatternResult:
        """Create triangle pattern result."""
        current_price = data['close'].iloc[-1]
        
        # Calculate breakout price (intersection of trend lines)
        if abs(upper_trend['slope'] - lower_trend['slope']) > 0.001:
            # Lines will intersect
            x_intersect = (lower_trend['intercept'] - upper_trend['intercept']) / (upper_trend['slope'] - lower_trend['slope'])
            breakout_price = upper_trend['slope'] * x_intersect + upper_trend['intercept']
        else:
            breakout_price = current_price
        
        # Calculate target and stop loss
        pattern_height = abs(upper_trend['current_price'] - lower_trend['current_price'])
        
        if pattern_type == PatternType.ASCENDING_TRIANGLE:
            target_price = breakout_price + pattern_height
            stop_loss = lower_trend['current_price']
        elif pattern_type == PatternType.DESCENDING_TRIANGLE:
            target_price = breakout_price - pattern_height
            stop_loss = upper_trend['current_price']
        else:  # Symmetrical
            target_price = breakout_price + pattern_height
            stop_loss = lower_trend['current_price']
        
        return PatternResult(
            pattern_type=pattern_type,
            confidence=min(0.9, (upper_trend['r_squared'] + lower_trend['r_squared']) / 2),
            start_time=data.index[0],
            end_time=data.index[-1],
            breakout_price=breakout_price,
            target_price=target_price,
            stop_loss=stop_loss,
            description=description,
            metrics={
                'upper_r_squared': upper_trend['r_squared'],
                'lower_r_squared': lower_trend['r_squared'],
                'pattern_height': pattern_height
            }
        )
    
    def _calculate_flag_target(self, data: pd.DataFrame, momentum: float) -> float:
        """Calculate flag pattern target."""
        current_price = data['close'].iloc[-1]
        flag_height = abs(data['high'].max() - data['low'].min())
        
        if momentum > 0:  # Bull flag
            return current_price + flag_height
        else:  # Bear flag
            return current_price - flag_height
    
    def _calculate_flag_stop_loss(self, data: pd.DataFrame, momentum: float) -> float:
        """Calculate flag pattern stop loss."""
        current_price = data['close'].iloc[-1]
        flag_height = abs(data['high'].max() - data['low'].min())
        
        if momentum > 0:  # Bull flag
            return current_price - flag_height * 0.5
        else:  # Bear flag
            return current_price + flag_height * 0.5
    
    def _identify_head_shoulders(self, peaks: List[int], highs: np.ndarray, 
                               troughs: List[int], lows: np.ndarray) -> Optional[Dict[str, Any]]:
        """Identify head and shoulders pattern."""
        if len(peaks) < 5:
            return None
        
        # Look for three peaks with middle peak higher
        for i in range(len(peaks) - 2):
            left_shoulder = highs[peaks[i]]
            head = highs[peaks[i + 1]]
            right_shoulder = highs[peaks[i + 2]]
            
            # Check head and shoulders criteria
            if (head > left_shoulder and head > right_shoulder and 
                abs(left_shoulder - right_shoulder) / left_shoulder < 0.05):
                
                # Calculate neckline
                neckline_price = (left_shoulder + right_shoulder) / 2
                
                return {
                    'confidence': 0.8,
                    'breakout_price': neckline_price,
                    'target_price': neckline_price - (head - neckline_price),
                    'stop_loss': head,
                    'metrics': {
                        'head_height': head - neckline_price,
                        'shoulder_symmetry': abs(left_shoulder - right_shoulder) / left_shoulder
                    }
                }
        
        return None
    
    def _identify_double_top(self, peaks: List[int], highs: np.ndarray) -> Optional[Dict[str, Any]]:
        """Identify double top pattern."""
        if len(peaks) < 4:
            return None
        
        for i in range(len(peaks) - 1):
            first_top = highs[peaks[i]]
            second_top = highs[peaks[i + 1]]
            
            # Check double top criteria
            if abs(first_top - second_top) / first_top < 0.02:  # Within 2%
                return {
                    'confidence': 0.75,
                    'breakout_price': first_top,
                    'target_price': first_top - (first_top - min(highs[peaks[i]:peaks[i+1]+1])),
                    'stop_loss': max(first_top, second_top),
                    'metrics': {
                        'top_symmetry': abs(first_top - second_top) / first_top,
                        'trough_depth': first_top - min(highs[peaks[i]:peaks[i+1]+1])
                    }
                }
        
        return None
    
    def _identify_double_bottom(self, troughs: List[int], lows: np.ndarray) -> Optional[Dict[str, Any]]:
        """Identify double bottom pattern."""
        if len(troughs) < 4:
            return None
        
        for i in range(len(troughs) - 1):
            first_bottom = lows[troughs[i]]
            second_bottom = lows[troughs[i + 1]]
            
            # Check double bottom criteria
            if abs(first_bottom - second_bottom) / first_bottom < 0.02:  # Within 2%
                return {
                    'confidence': 0.75,
                    'breakout_price': first_bottom,
                    'target_price': first_bottom + (max(lows[troughs[i]:troughs[i+1]+1]) - first_bottom),
                    'stop_loss': min(first_bottom, second_bottom),
                    'metrics': {
                        'bottom_symmetry': abs(first_bottom - second_bottom) / first_bottom,
                        'peak_height': max(lows[troughs[i]:troughs[i+1]+1]) - first_bottom
                    }
                }
        
        return None
    
    def _calculate_channel_target(self, upper_trend: Dict, lower_trend: Dict) -> float:
        """Calculate channel pattern target."""
        channel_width = upper_trend['current_price'] - lower_trend['current_price']
        return upper_trend['current_price'] + channel_width


def main():
    """Test the pattern recognition system."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test pattern recognition")
    parser.add_argument("symbol", help="Trading symbol (e.g., EURUSD)")
    parser.add_argument("--days", type=int, default=60, help="Number of days to analyze")
    
    args = parser.parse_args()
    
    # Get real data
    from src.data.real_data_ingestor import RealDataIngestor
    
    ingestor = RealDataIngestor()
    end_date = datetime.now()
    start_date = end_date - timedelta(days=args.days)
    
    data = ingestor.load_symbol_data(args.symbol, start_date, end_date)
    
    if data is None or data.empty:
        print(f"❌ No data available for {args.symbol}")
        return
    
    # Initialize pattern recognition
    pattern_recognition = AdvancedPatternRecognition()
    
    # Detect patterns
    patterns = pattern_recognition.detect_patterns(data, args.symbol)
    
    print(f"\n🎯 Pattern Recognition Analysis for {args.symbol}")
    print("=" * 50)
    
    if patterns:
        print(f"Found {len(patterns)} patterns:")
        for i, pattern in enumerate(patterns[:5]):  # Show top 5
            print(f"\n{i+1}. {pattern.pattern_type.value.upper()}")
            print(f"   Confidence: {pattern.confidence:.2%}")
            print(f"   Description: {pattern.description}")
            print(f"   Breakout Price: {pattern.breakout_price:.4f}")
            print(f"   Target Price: {pattern.target_price:.4f}")
            print(f"   Stop Loss: {pattern.stop_loss:.4f}")
    else:
        print("No patterns detected in the current data.")


if __name__ == "__main__":
    main() 