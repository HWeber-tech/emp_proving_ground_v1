"""
EMP Regime Classifier v1.1

Market regime classification and detection for the thinking layer.
Migrated from sensory layer to thinking layer where cognitive functions belong.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from enum import Enum
import logging

from ...core.events import MarketData, AnalysisResult

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """Market regime classifications."""
    BULL_MARKET = "bull_market"
    BEAR_MARKET = "bear_market"
    SIDEWAYS_MARKET = "sideways_market"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    CONSOLIDATION = "consolidation"
    BREAKOUT = "breakout"
    BREAKDOWN = "breakdown"


class RegimeClassifier:
    """Market regime classifier for cognitive analysis."""
    
    def __init__(self, lookback_period: int = 252, volatility_threshold: float = 0.2):
        self.lookback_period = lookback_period
        self.volatility_threshold = volatility_threshold
        self.regime_history: List[Dict[str, Any]] = []
        
        logger.info(f"Regime Classifier initialized with {lookback_period} day lookback")
        
    def classify_regime(self, market_data: List[MarketData]) -> AnalysisResult:
        """Classify the current market regime."""
        try:
            if len(market_data) < self.lookback_period:
                logger.warning(f"Insufficient data for regime classification: {len(market_data)} < {self.lookback_period}")
                return self._create_default_analysis()
                
            # Convert to DataFrame
            df = self._market_data_to_dataframe(market_data)
            
            # Calculate regime indicators
            regime_indicators = self._calculate_regime_indicators(df)
            
            # Determine primary regime
            primary_regime = self._determine_primary_regime(regime_indicators)
            
            # Calculate confidence
            confidence = self._calculate_regime_confidence(regime_indicators)
            
            # Create analysis result
            result = AnalysisResult(
                timestamp=datetime.now(),
                analysis_type="market_regime_classification",
                result={
                    "primary_regime": primary_regime.value,
                    "regime_indicators": regime_indicators,
                    "confidence": confidence,
                    "lookback_period": self.lookback_period,
                    "data_points": len(market_data)
                },
                confidence=confidence,
                metadata={
                    "classifier_version": "1.1.0",
                    "method": "multi_factor_regime_analysis"
                }
            )
            
            # Store in history
            self.regime_history.append({
                "timestamp": result.timestamp,
                "regime": primary_regime.value,
                "confidence": confidence,
                "indicators": regime_indicators
            })
            
            logger.debug(f"Regime classified as {primary_regime.value} with {confidence:.2f} confidence")
            return result
            
        except Exception as e:
            logger.error(f"Error classifying regime: {e}")
            return self._create_default_analysis()
            
    def _market_data_to_dataframe(self, market_data: List[MarketData]) -> pd.DataFrame:
        """Convert market data to DataFrame."""
        data = []
        for md in market_data:
            data.append({
                'timestamp': md.timestamp,
                'open': md.open,
                'high': md.high,
                'low': md.low,
                'close': md.close,
                'volume': md.volume
            })
            
        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        df.sort_index(inplace=True)
        
        return df
        
    def _calculate_regime_indicators(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate regime classification indicators."""
        indicators = {}
        
        # Price-based indicators
        indicators['price_trend'] = self._calculate_price_trend(df)
        indicators['price_momentum'] = self._calculate_price_momentum(df)
        indicators['price_volatility'] = self._calculate_price_volatility(df)
        
        # Volume-based indicators
        indicators['volume_trend'] = self._calculate_volume_trend(df)
        indicators['volume_volatility'] = self._calculate_volume_volatility(df)
        
        # Technical indicators
        indicators['rsi'] = self._calculate_rsi(df)
        indicators['macd'] = self._calculate_macd(df)
        indicators['bollinger_position'] = self._calculate_bollinger_position(df)
        
        # Volatility regime
        indicators['volatility_regime'] = self._classify_volatility_regime(df)
        
        return indicators
        
    def _calculate_price_trend(self, df: pd.DataFrame) -> float:
        """Calculate price trend strength."""
        if len(df) < 20:
            return 0.0
            
        # Linear regression slope
        x = np.arange(len(df))
        y = df['close'].values
        slope = np.polyfit(x, y, 1)[0]
        
        # Normalize slope
        price_range = df['close'].max() - df['close'].min()
        if price_range == 0:
            return 0.0
            
        normalized_slope = slope / price_range
        return np.clip(normalized_slope * 100, -1, 1)
        
    def _calculate_price_momentum(self, df: pd.DataFrame) -> float:
        """Calculate price momentum."""
        if len(df) < 14:
            return 0.0
            
        # Rate of change over 14 periods
        roc = (df['close'].iloc[-1] - df['close'].iloc[-14]) / df['close'].iloc[-14]
        return np.clip(roc, -1, 1)
        
    def _calculate_price_volatility(self, df: pd.DataFrame) -> float:
        """Calculate price volatility."""
        if len(df) < 20:
            return 0.0
            
        # Rolling standard deviation
        returns = df['close'].pct_change().dropna()
        volatility = returns.rolling(window=20).std().iloc[-1]
        return np.clip(volatility, 0, 1)
        
    def _calculate_volume_trend(self, df: pd.DataFrame) -> float:
        """Calculate volume trend."""
        if len(df) < 20:
            return 0.0
            
        # Volume moving average ratio
        current_volume = df['volume'].iloc[-1]
        avg_volume = df['volume'].rolling(window=20).mean().iloc[-1]
        
        if avg_volume == 0:
            return 0.0
            
        volume_ratio = (current_volume - avg_volume) / avg_volume
        return np.clip(volume_ratio, -1, 1)
        
    def _calculate_volume_volatility(self, df: pd.DataFrame) -> float:
        """Calculate volume volatility."""
        if len(df) < 20:
            return 0.0
            
        # Volume standard deviation
        volume_std = df['volume'].rolling(window=20).std().iloc[-1]
        volume_mean = df['volume'].rolling(window=20).mean().iloc[-1]
        
        if volume_mean == 0:
            return 0.0
            
        volume_cv = volume_std / volume_mean
        return np.clip(volume_cv, 0, 1)
        
    def _calculate_rsi(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate RSI indicator."""
        if len(df) < period + 1:
            return 50.0
            
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi.iloc[-1] / 100  # Normalize to [0, 1]
        
    def _calculate_macd(self, df: pd.DataFrame) -> float:
        """Calculate MACD indicator."""
        if len(df) < 26:
            return 0.0
            
        ema12 = df['close'].ewm(span=12).mean()
        ema26 = df['close'].ewm(span=26).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9).mean()
        
        macd_value = macd.iloc[-1]
        signal_value = signal.iloc[-1]
        
        # Normalize MACD
        price_range = df['close'].max() - df['close'].min()
        if price_range == 0:
            return 0.0
            
        normalized_macd = (macd_value - signal_value) / price_range
        return np.clip(normalized_macd * 100, -1, 1)
        
    def _calculate_bollinger_position(self, df: pd.DataFrame, period: int = 20) -> float:
        """Calculate position within Bollinger Bands."""
        if len(df) < period:
            return 0.5
            
        sma = df['close'].rolling(window=period).mean()
        std = df['close'].rolling(window=period).std()
        
        upper_band = sma + (std * 2)
        lower_band = sma - (std * 2)
        
        current_price = df['close'].iloc[-1]
        current_upper = upper_band.iloc[-1]
        current_lower = lower_band.iloc[-1]
        
        if current_upper == current_lower:
            return 0.5
            
        position = (current_price - current_lower) / (current_upper - current_lower)
        return np.clip(position, 0, 1)
        
    def _classify_volatility_regime(self, df: pd.DataFrame) -> float:
        """Classify volatility regime."""
        if len(df) < 20:
            return 0.5
            
        current_volatility = self._calculate_price_volatility(df)
        
        if current_volatility > self.volatility_threshold:
            return 1.0  # High volatility
        elif current_volatility < self.volatility_threshold * 0.5:
            return 0.0  # Low volatility
        else:
            return 0.5  # Medium volatility
            
    def _determine_primary_regime(self, indicators: Dict[str, float]) -> MarketRegime:
        """Determine the primary market regime based on indicators."""
        price_trend = indicators.get('price_trend', 0)
        price_momentum = indicators.get('price_momentum', 0)
        volatility = indicators.get('price_volatility', 0)
        volume_trend = indicators.get('volume_trend', 0)
        rsi = indicators.get('rsi', 0.5)
        macd = indicators.get('macd', 0)
        
        # High volatility regime
        if volatility > self.volatility_threshold:
            return MarketRegime.HIGH_VOLATILITY
            
        # Low volatility regime
        if volatility < self.volatility_threshold * 0.3:
            return MarketRegime.LOW_VOLATILITY
            
        # Trending regimes
        if price_trend > 0.1 and price_momentum > 0.05:
            return MarketRegime.TRENDING_UP
        elif price_trend < -0.1 and price_momentum < -0.05:
            return MarketRegime.TRENDING_DOWN
            
        # Bull/Bear market based on RSI and MACD
        if rsi > 0.7 and macd > 0.1:
            return MarketRegime.BULL_MARKET
        elif rsi < 0.3 and macd < -0.1:
            return MarketRegime.BEAR_MARKET
            
        # Consolidation
        if abs(price_trend) < 0.05 and abs(price_momentum) < 0.02:
            return MarketRegime.CONSOLIDATION
            
        # Default to sideways
        return MarketRegime.SIDEWAYS_MARKET
        
    def _calculate_regime_confidence(self, indicators: Dict[str, float]) -> float:
        """Calculate confidence in regime classification."""
        # Base confidence on indicator consistency
        confidence_factors = []
        
        # Price trend consistency
        price_trend = abs(indicators.get('price_trend', 0))
        confidence_factors.append(price_trend)
        
        # Momentum consistency
        price_momentum = abs(indicators.get('price_momentum', 0))
        confidence_factors.append(price_momentum)
        
        # Volume confirmation
        volume_trend = abs(indicators.get('volume_trend', 0))
        confidence_factors.append(volume_trend * 0.5)
        
        # Technical indicator alignment
        rsi_extreme = abs(indicators.get('rsi', 0.5) - 0.5) * 2
        confidence_factors.append(rsi_extreme)
        
        # Average confidence
        avg_confidence = np.mean(confidence_factors)
        return np.clip(avg_confidence, 0.1, 1.0)
        
    def _create_default_analysis(self) -> AnalysisResult:
        """Create default analysis when classification fails."""
        return AnalysisResult(
            timestamp=datetime.now(),
            analysis_type="market_regime_classification",
            result={
                "primary_regime": MarketRegime.SIDEWAYS_MARKET.value,
                "regime_indicators": {},
                "confidence": 0.1,
                "lookback_period": self.lookback_period,
                "data_points": 0
            },
            confidence=0.1,
            metadata={
                "classifier_version": "1.1.0",
                "method": "default_fallback",
                "error": "Insufficient data for classification"
            }
        )
        
    def get_regime_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get regime classification history."""
        if limit:
            return self.regime_history[-limit:]
        return self.regime_history.copy()
        
    def get_regime_statistics(self) -> Dict[str, Any]:
        """Get statistics about regime classifications."""
        if not self.regime_history:
            return {}
            
        regimes = [h['regime'] for h in self.regime_history]
        confidences = [h['confidence'] for h in self.regime_history]
        
        regime_counts = {}
        for regime in regimes:
            regime_counts[regime] = regime_counts.get(regime, 0) + 1
            
        return {
            'total_classifications': len(self.regime_history),
            'regime_distribution': regime_counts,
            'average_confidence': np.mean(confidences),
            'confidence_std': np.std(confidences),
            'most_common_regime': max(regime_counts.items(), key=lambda x: x[1])[0] if regime_counts else None
        } 