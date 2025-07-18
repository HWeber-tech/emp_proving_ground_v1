#!/usr/bin/env python3
"""
Real Market Regime Detector - Advanced Market Analysis

This module implements real market regime detection using actual market data
to identify different market conditions: trending, ranging, volatile, calm.
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


class MarketRegime(Enum):
    """Market regime types."""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    VOLATILE = "volatile"
    CALM = "calm"
    BREAKOUT = "breakout"
    REVERSAL = "reversal"
    UNKNOWN = "unknown"


@dataclass
class RegimeResult:
    """Result of market regime analysis."""
    regime: MarketRegime
    confidence: float
    start_time: datetime
    end_time: datetime
    metrics: Dict[str, float]
    description: str


class MarketRegimeDetector:
    """
    Real market regime detector using actual market data.
    
    This replaces synthetic regime detection with real analysis
    of market conditions using technical indicators and statistical methods.
    """
    
    def __init__(self, lookback_period: int = 50, volatility_threshold: float = 0.02):
        """
        Initialize the market regime detector.
        
        Args:
            lookback_period: Number of periods to analyze
            volatility_threshold: Threshold for volatility classification
        """
        self.lookback_period = lookback_period
        self.volatility_threshold = volatility_threshold
        
        logger.info(f"Market regime detector initialized with {lookback_period} period lookback")
    
    def detect_regime(self, data: pd.DataFrame, symbol: str = "UNKNOWN") -> RegimeResult:
        """
        Detect the current market regime from real market data.
        
        Args:
            data: OHLCV market data
            symbol: Trading symbol
            
        Returns:
            RegimeResult with detected regime and confidence
        """
        if data.empty or len(data) < self.lookback_period:
            logger.warning(f"Insufficient data for regime detection: {len(data)} records")
            return self._create_unknown_regime(data)
        
        try:
            # Calculate technical indicators
            indicators = self._calculate_indicators(data)
            
            # Analyze different aspects
            trend_analysis = self._analyze_trend(indicators)
            volatility_analysis = self._analyze_volatility(indicators)
            momentum_analysis = self._analyze_momentum(indicators)
            pattern_analysis = self._analyze_patterns(indicators)
            
            # Combine analyses to determine regime
            regime, confidence, description = self._combine_analyses(
                trend_analysis, volatility_analysis, momentum_analysis, pattern_analysis
            )
            
            # Calculate regime metrics
            metrics = self._calculate_regime_metrics(indicators, regime)
            
            return RegimeResult(
                regime=regime,
                confidence=confidence,
                start_time=data.index[0],
                end_time=data.index[-1],
                metrics=metrics,
                description=description
            )
            
        except Exception as e:
            logger.error(f"Error detecting regime for {symbol}: {e}")
            return self._create_unknown_regime(data)
    
    def _calculate_indicators(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate technical indicators for regime analysis."""
        indicators = {}
        
        # Price-based indicators
        indicators['returns'] = data['close'].pct_change()
        indicators['log_returns'] = np.log(data['close'] / data['close'].shift(1))
        
        # Moving averages
        indicators['sma_20'] = data['close'].rolling(window=20).mean()
        indicators['sma_50'] = data['close'].rolling(window=50).mean()
        indicators['ema_12'] = data['close'].ewm(span=12).mean()
        indicators['ema_26'] = data['close'].ewm(span=26).mean()
        
        # Volatility indicators
        indicators['atr'] = self._calculate_atr(data)
        indicators['volatility'] = indicators['returns'].rolling(window=20).std()
        
        # Momentum indicators
        indicators['rsi'] = self._calculate_rsi(data['close'])
        indicators['macd'] = indicators['ema_12'] - indicators['ema_26']
        indicators['macd_signal'] = indicators['macd'].ewm(span=9).mean()
        
        # Volume indicators
        if 'volume' in data.columns:
            indicators['volume_sma'] = data['volume'].rolling(window=20).mean()
            indicators['volume_ratio'] = data['volume'] / indicators['volume_sma']
        else:
            indicators['volume_ratio'] = pd.Series(1.0, index=data.index)
        
        # Support and resistance
        indicators['support'] = data['low'].rolling(window=20).min()
        indicators['resistance'] = data['high'].rolling(window=20).max()
        
        return indicators
    
    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range."""
        high_low = data['high'] - data['low']
        high_close = np.abs(data['high'] - data['close'].shift())
        low_close = np.abs(data['low'] - data['close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return true_range.rolling(window=period).mean()
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _analyze_trend(self, indicators: Dict[str, pd.Series]) -> Dict[str, Any]:
        """Analyze market trend."""
        recent_data = {k: v.tail(self.lookback_period) for k, v in indicators.items()}
        
        # Trend strength using linear regression
        prices = recent_data['sma_20'].dropna()
        if len(prices) < 10:
            return {'trend_strength': 0, 'trend_direction': 0, 'trend_confidence': 0}
        
        x = np.arange(len(prices))
        slope, intercept = np.polyfit(x, prices, 1)
        
        # Calculate R-squared for trend confidence
        y_pred = slope * x + intercept
        ss_res = np.sum((prices - y_pred) ** 2)
        ss_tot = np.sum((prices - prices.mean()) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        # Moving average alignment
        ma_alignment = 0
        if not recent_data['sma_20'].isna().all() and not recent_data['sma_50'].isna().all():
            ma_alignment = 1 if recent_data['sma_20'].iloc[-1] > recent_data['sma_50'].iloc[-1] else -1
        
        return {
            'trend_strength': abs(slope),
            'trend_direction': np.sign(slope),
            'trend_confidence': r_squared,
            'ma_alignment': ma_alignment
        }
    
    def _analyze_volatility(self, indicators: Dict[str, pd.Series]) -> Dict[str, Any]:
        """Analyze market volatility."""
        recent_data = {k: v.tail(self.lookback_period) for k, v in indicators.items()}
        
        # Current volatility
        current_vol = recent_data['volatility'].iloc[-1] if not recent_data['volatility'].isna().all() else 0
        
        # Volatility trend
        vol_trend = 0
        if len(recent_data['volatility'].dropna()) > 10:
            vol_values = recent_data['volatility'].dropna()
            vol_slope = np.polyfit(range(len(vol_values)), vol_values, 1)[0]
            vol_trend = np.sign(vol_slope)
        
        # Volatility regime
        vol_regime = "high" if current_vol > self.volatility_threshold else "low"
        
        return {
            'current_volatility': current_vol,
            'volatility_trend': vol_trend,
            'volatility_regime': vol_regime
        }
    
    def _analyze_momentum(self, indicators: Dict[str, pd.Series]) -> Dict[str, Any]:
        """Analyze market momentum."""
        recent_data = {k: v.tail(self.lookback_period) for k, v in indicators.items()}
        
        # RSI analysis
        current_rsi = recent_data['rsi'].iloc[-1] if not recent_data['rsi'].isna().all() else 50
        rsi_trend = 0
        if len(recent_data['rsi'].dropna()) > 5:
            rsi_values = recent_data['rsi'].dropna()
            rsi_slope = np.polyfit(range(len(rsi_values)), rsi_values, 1)[0]
            rsi_trend = np.sign(rsi_slope)
        
        # MACD analysis
        macd_current = recent_data['macd'].iloc[-1] if not recent_data['macd'].isna().all() else 0
        macd_signal = recent_data['macd_signal'].iloc[-1] if not recent_data['macd_signal'].isna().all() else 0
        macd_bullish = macd_current > macd_signal
        
        return {
            'rsi_current': current_rsi,
            'rsi_trend': rsi_trend,
            'macd_bullish': macd_bullish,
            'momentum_strength': abs(current_rsi - 50) / 50
        }
    
    def _analyze_patterns(self, indicators: Dict[str, pd.Series]) -> Dict[str, Any]:
        """Analyze price patterns."""
        recent_data = {k: v.tail(self.lookback_period) for k, v in indicators.items()}
        
        # Support/resistance analysis
        current_price = recent_data['sma_20'].iloc[-1] if not recent_data['sma_20'].isna().all() else 0
        support = recent_data['support'].iloc[-1] if not recent_data['support'].isna().all() else 0
        resistance = recent_data['resistance'].iloc[-1] if not recent_data['resistance'].isna().all() else 0
        
        # Distance to support/resistance
        support_distance = (current_price - support) / current_price if current_price > 0 else 0
        resistance_distance = (resistance - current_price) / current_price if current_price > 0 else 0
        
        # Breakout detection
        breakout_up = resistance_distance < 0.01  # Within 1% of resistance
        breakout_down = support_distance < 0.01   # Within 1% of support
        
        return {
            'support_distance': support_distance,
            'resistance_distance': resistance_distance,
            'breakout_up': breakout_up,
            'breakout_down': breakout_down,
            'range_width': (resistance - support) / current_price if current_price > 0 else 0
        }
    
    def _combine_analyses(self, trend: Dict, volatility: Dict, momentum: Dict, patterns: Dict) -> Tuple[MarketRegime, float, str]:
        """Combine all analyses to determine the market regime."""
        
        # Trend analysis
        trend_strength = trend.get('trend_strength', 0)
        trend_direction = trend.get('trend_direction', 0)
        trend_confidence = trend.get('trend_confidence', 0)
        
        # Volatility analysis
        current_vol = volatility.get('current_volatility', 0)
        vol_regime = volatility.get('volatility_regime', 'low')
        
        # Momentum analysis
        rsi_current = momentum.get('rsi_current', 50)
        macd_bullish = momentum.get('macd_bullish', False)
        
        # Pattern analysis
        breakout_up = patterns.get('breakout_up', False)
        breakout_down = patterns.get('breakout_down', False)
        range_width = patterns.get('range_width', 0)
        
        # Determine regime based on combined analysis
        if trend_confidence > 0.7 and trend_strength > 0.001:
            if trend_direction > 0:
                regime = MarketRegime.TRENDING_UP
                confidence = min(0.9, trend_confidence + 0.1)
                description = f"Strong uptrend (confidence: {confidence:.2f})"
            else:
                regime = MarketRegime.TRENDING_DOWN
                confidence = min(0.9, trend_confidence + 0.1)
                description = f"Strong downtrend (confidence: {confidence:.2f})"
        
        elif breakout_up or breakout_down:
            regime = MarketRegime.BREAKOUT
            confidence = 0.8
            description = f"Breakout detected (up: {breakout_up}, down: {breakout_down})"
        
        elif vol_regime == 'high' and current_vol > self.volatility_threshold * 1.5:
            regime = MarketRegime.VOLATILE
            confidence = 0.85
            description = f"High volatility regime (vol: {current_vol:.4f})"
        
        elif range_width < 0.05 and trend_confidence < 0.3:  # Narrow range, low trend
            regime = MarketRegime.RANGING
            confidence = 0.75
            description = f"Ranging market (range width: {range_width:.2%})"
        
        elif vol_regime == 'low' and current_vol < self.volatility_threshold * 0.5:
            regime = MarketRegime.CALM
            confidence = 0.8
            description = f"Low volatility, calm market (vol: {current_vol:.4f})"
        
        else:
            regime = MarketRegime.UNKNOWN
            confidence = 0.5
            description = "Mixed signals, unclear regime"
        
        return regime, confidence, description
    
    def _calculate_regime_metrics(self, indicators: Dict[str, pd.Series], regime: MarketRegime) -> Dict[str, float]:
        """Calculate regime-specific metrics."""
        recent_data = {k: v.tail(self.lookback_period) for k, v in indicators.items()}
        
        metrics = {
            'avg_volatility': recent_data['volatility'].mean() if not recent_data['volatility'].isna().all() else 0,
            'avg_volume_ratio': recent_data['volume_ratio'].mean() if not recent_data['volume_ratio'].isna().all() else 1,
            'price_range': (recent_data['sma_20'].max() - recent_data['sma_20'].min()) / recent_data['sma_20'].mean() if not recent_data['sma_20'].isna().all() else 0,
            'trend_consistency': recent_data['returns'].rolling(window=5).apply(lambda x: np.sign(x).sum() / len(x)).mean() if not recent_data['returns'].isna().all() else 0
        }
        
        return metrics
    
    def _create_unknown_regime(self, data: pd.DataFrame) -> RegimeResult:
        """Create unknown regime result when analysis fails."""
        return RegimeResult(
            regime=MarketRegime.UNKNOWN,
            confidence=0.0,
            start_time=data.index[0] if not data.empty else datetime.now(),
            end_time=data.index[-1] if not data.empty else datetime.now(),
            metrics={},
            description="Insufficient data for regime detection"
        )
    
    def detect_regime_history(self, data: pd.DataFrame, window_size: int = 50, step_size: int = 10) -> List[RegimeResult]:
        """
        Detect regime changes over time using sliding window.
        
        Args:
            data: OHLCV market data
            window_size: Size of analysis window
            step_size: Step size for sliding window
            
        Returns:
            List of RegimeResult objects over time
        """
        if len(data) < window_size:
            logger.warning(f"Insufficient data for regime history: {len(data)} records")
            return []
        
        regime_history = []
        
        for i in range(0, len(data) - window_size + 1, step_size):
            window_data = data.iloc[i:i + window_size]
            regime_result = self.detect_regime(window_data)
            regime_history.append(regime_result)
        
        return regime_history
    
    def get_regime_transitions(self, regime_history: List[RegimeResult]) -> List[Dict[str, Any]]:
        """
        Identify regime transitions from regime history.
        
        Args:
            regime_history: List of RegimeResult objects
            
        Returns:
            List of regime transition events
        """
        transitions = []
        
        for i in range(1, len(regime_history)):
            prev_regime = regime_history[i-1].regime
            curr_regime = regime_history[i].regime
            
            if prev_regime != curr_regime:
                transition = {
                    'from_regime': prev_regime,
                    'to_regime': curr_regime,
                    'transition_time': regime_history[i].start_time,
                    'confidence': regime_history[i].confidence,
                    'description': f"Transition from {prev_regime.value} to {curr_regime.value}"
                }
                transitions.append(transition)
        
        return transitions


def main():
    """Test the market regime detector."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test market regime detection")
    parser.add_argument("symbol", help="Trading symbol (e.g., EURUSD)")
    parser.add_argument("--days", type=int, default=30, help="Number of days to analyze")
    
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
    
    # Initialize detector
    detector = MarketRegimeDetector()
    
    # Detect current regime
    regime_result = detector.detect_regime(data, args.symbol)
    
    print(f"\n🎯 Market Regime Analysis for {args.symbol}")
    print("=" * 50)
    print(f"Regime: {regime_result.regime.value.upper()}")
    print(f"Confidence: {regime_result.confidence:.2%}")
    print(f"Description: {regime_result.description}")
    print(f"Period: {regime_result.start_time.date()} to {regime_result.end_time.date()}")
    
    print(f"\n📊 Regime Metrics:")
    for key, value in regime_result.metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # Detect regime history
    print(f"\n📈 Regime History:")
    regime_history = detector.detect_regime_history(data)
    
    if regime_history:
        regime_counts = {}
        for result in regime_history:
            regime = result.regime.value
            regime_counts[regime] = regime_counts.get(regime, 0) + 1
        
        for regime, count in regime_counts.items():
            percentage = (count / len(regime_history)) * 100
            print(f"  {regime}: {count} periods ({percentage:.1f}%)")
        
        # Find transitions
        transitions = detector.get_regime_transitions(regime_history)
        if transitions:
            print(f"\n🔄 Regime Transitions ({len(transitions)}):")
            for transition in transitions[-3:]:  # Show last 3 transitions
                print(f"  {transition['transition_time'].date()}: {transition['description']}")


if __name__ == "__main__":
    main() 