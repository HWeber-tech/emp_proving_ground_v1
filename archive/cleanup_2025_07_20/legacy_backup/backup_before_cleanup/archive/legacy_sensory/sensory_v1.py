"""
Sensory Cortex Module - v2.0 Implementation

This module implements the innovative 4D+1 Sensory Cortex as specified in v2.0,
providing the "brain" of each trading organism with multi-dimensional market perception.
"""

import logging
from datetime import datetime, timedelta
from decimal import Decimal, getcontext, ROUND_HALF_UP
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler

from .core import Instrument
from .data import TickDataStorage

# Configure decimal precision for financial calculations
getcontext().prec = 12
getcontext().rounding = ROUND_HALF_UP

logger = logging.getLogger(__name__)


@dataclass
class SensoryReading:
    """
    Complete sensory reading from the 4D+1 Sensory Cortex.
    
    This represents the organism's perception of the market across all dimensions,
    providing the foundation for intelligent decision-making.
    """
    timestamp: datetime
    
    # WHY dimension (Macro Context)
    macro_trend: str  # 'BULLISH', 'BEARISH', 'NEUTRAL'
    macro_strength: float  # 0.0 to 1.0
    macro_volatility: float  # Current volatility regime
    macro_regime: str  # 'TRENDING', 'RANGING', 'VOLATILE'
    
    # HOW dimension (Institutional Behavior)
    institutional_flow: float  # -1.0 to 1.0 (negative = selling, positive = buying)
    institutional_confidence: float  # 0.0 to 1.0
    large_order_activity: float  # 0.0 to 1.0
    order_flow_imbalance: float  # -1.0 to 1.0
    
    # WHAT dimension (Technical Analysis)
    technical_signal: str  # 'BUY', 'SELL', 'HOLD'
    technical_strength: float  # 0.0 to 1.0
    support_level: Optional[float] = None
    resistance_level: Optional[float] = None
    momentum_score: float  # -1.0 to 1.0
    volatility_score: float  # 0.0 to 1.0
    
    # WHEN dimension (Session/Time Context)
    session_phase: str  # 'ASIAN', 'LONDON', 'NEW_YORK', 'OVERLAP'
    session_volatility: float  # 0.0 to 1.0
    time_of_day_score: float  # -1.0 to 1.0
    day_of_week_score: float  # -1.0 to 1.0
    news_impact_score: float  # 0.0 to 1.0
    
    # ANOMALY dimension (Manipulation Detection)
    manipulation_probability: float  # 0.0 to 1.0
    stop_hunt_probability: float  # 0.0 to 1.0
    spoofing_detected: bool = False
    liquidity_zone_activity: float  # 0.0 to 1.0
    unusual_volume: float  # 0.0 to 1.0
    
    # Composite scores
    overall_sentiment: float  # -1.0 to 1.0
    confidence_level: float  # 0.0 to 1.0
    risk_level: float  # 0.0 to 1.0
    
    # Metadata
    processing_time_ms: float = 0.0
    data_quality_score: float = 1.0


class SensoryCortex:
    """
    v2.0 4D+1 Sensory Cortex - The "Brain" of Trading Organisms.
    
    This class implements the innovative multi-dimensional perception system
    from the original unified file, providing organisms with sophisticated
    market understanding across five key dimensions.
    """
    
    def __init__(self, symbol: str, data_storage: TickDataStorage):
        """
        Initialize the sensory cortex.
        
        Args:
            symbol: Trading symbol (e.g., 'EURUSD')
            data_storage: Data storage for market data access
        """
        self.symbol = symbol
        self.data_storage = data_storage
        
        # Calibration parameters
        self.calibrated = False
        self.volatility_baseline = 0.0
        self.volume_baseline = 0.0
        self.price_baseline = 0.0
        
        # Technical analysis parameters
        self.ema_periods = [9, 21, 50, 200]
        self.rsi_period = 14
        self.bb_period = 20
        self.bb_std = 2
        
        # Session definitions (UTC)
        self.sessions = {
            'ASIAN': (0, 8),      # 00:00-08:00 UTC
            'LONDON': (8, 16),    # 08:00-16:00 UTC
            'NEW_YORK': (13, 21), # 13:00-21:00 UTC
            'OVERLAP': [(8, 13), (16, 21)]  # London-New York overlaps
        }
        
        # Manipulation detection parameters
        self.stop_hunt_threshold = 0.7
        self.spoofing_threshold = 0.6
        self.liquidity_threshold = 0.5
        
        logger.info(f"SensoryCortex initialized for {symbol}")
    
    def calibrate(self, start_time: datetime, end_time: datetime) -> bool:
        """
        Calibrate the sensory cortex with historical data.
        
        Args:
            start_time: Start of calibration period
            end_time: End of calibration period
            
        Returns:
            True if calibration successful
        """
        try:
            logger.info(f"Calibrating sensory cortex from {start_time} to {end_time}")
            
            # Load historical data for calibration
            df = self.data_storage.get_data_range(self.symbol, start_time, end_time)
            if df is None or df.empty:
                logger.warning("No data available for calibration")
                return False
            
            # Calculate baseline metrics
            self.volatility_baseline = df['close'].pct_change().std()
            self.volume_baseline = df['volume'].mean() if 'volume' in df.columns else 1000
            self.price_baseline = df['close'].mean()
            
            # Validate baselines
            if (self.volatility_baseline <= 0 or 
                self.volume_baseline <= 0 or 
                self.price_baseline <= 0):
                logger.warning("Invalid baseline values calculated")
                return False
            
            self.calibrated = True
            logger.info(f"Calibration complete - Volatility: {self.volatility_baseline:.6f}, "
                       f"Volume: {self.volume_baseline:.2f}, Price: {self.price_baseline:.5f}")
            return True
            
        except Exception as e:
            logger.error(f"Calibration failed: {e}")
            return False
    
    def perceive(self, current_data: pd.Series, historical_window: int = 1000) -> SensoryReading:
        """
        Generate a complete sensory reading of the current market state.
        
        Args:
            current_data: Current market data point
            historical_window: Number of historical points to analyze
            
        Returns:
            Complete sensory reading
        """
        start_time = datetime.now()
        
        if not self.calibrated:
            logger.warning("Sensory cortex not calibrated, using default values")
            return self._create_default_reading(current_data.name)
        
        try:
            # Get historical data for analysis
            end_time = current_data.name
            start_time_hist = end_time - timedelta(hours=24)  # 24 hours of data
            
            df = self.data_storage.get_data_range(self.symbol, start_time_hist, end_time)
            if df is None or df.empty:
                logger.warning("No historical data available for perception")
                return self._create_default_reading(current_data.name)
            
            # Ensure we have enough data
            if len(df) < 50:
                logger.warning("Insufficient data for perception")
                return self._create_default_reading(current_data.name)
            
            # Generate sensory readings for each dimension
            why_reading = self._analyze_why_dimension(df, current_data)
            how_reading = self._analyze_how_dimension(df, current_data)
            what_reading = self._analyze_what_dimension(df, current_data)
            when_reading = self._analyze_when_dimension(df, current_data)
            anomaly_reading = self._analyze_anomaly_dimension(df, current_data)
            
            # Calculate composite scores
            overall_sentiment = self._calculate_overall_sentiment(
                why_reading, how_reading, what_reading, when_reading, anomaly_reading
            )
            confidence_level = self._calculate_confidence_level(
                why_reading, how_reading, what_reading, when_reading, anomaly_reading
            )
            risk_level = self._calculate_risk_level(
                why_reading, how_reading, what_reading, when_reading, anomaly_reading
            )
            
            # Create complete sensory reading
            reading = SensoryReading(
                timestamp=current_data.name,
                
                # WHY dimension
                macro_trend=why_reading['trend'],
                macro_strength=why_reading['strength'],
                macro_volatility=why_reading['volatility'],
                macro_regime=why_reading['regime'],
                
                # HOW dimension
                institutional_flow=how_reading['flow'],
                institutional_confidence=how_reading['confidence'],
                large_order_activity=how_reading['large_orders'],
                order_flow_imbalance=how_reading['imbalance'],
                
                # WHAT dimension
                technical_signal=what_reading['signal'],
                technical_strength=what_reading['strength'],
                support_level=what_reading['support'],
                resistance_level=what_reading['resistance'],
                momentum_score=what_reading['momentum'],
                volatility_score=what_reading['volatility'],
                
                # WHEN dimension
                session_phase=when_reading['session'],
                session_volatility=when_reading['volatility'],
                time_of_day_score=when_reading['time_score'],
                day_of_week_score=when_reading['day_score'],
                news_impact_score=when_reading['news_impact'],
                
                # ANOMALY dimension
                manipulation_probability=anomaly_reading['manipulation'],
                stop_hunt_probability=anomaly_reading['stop_hunt'],
                spoofing_detected=anomaly_reading['spoofing'],
                liquidity_zone_activity=anomaly_reading['liquidity'],
                unusual_volume=anomaly_reading['unusual_volume'],
                
                # Composite scores
                overall_sentiment=overall_sentiment,
                confidence_level=confidence_level,
                risk_level=risk_level,
                
                # Metadata
                processing_time_ms=(datetime.now() - start_time).total_seconds() * 1000,
                data_quality_score=self._calculate_data_quality(df)
            )
            
            logger.debug(f"Sensory reading generated in {reading.processing_time_ms:.2f}ms")
            return reading
            
        except Exception as e:
            logger.error(f"Error generating sensory reading: {e}")
            return self._create_default_reading(current_data.name)
    
    def _analyze_why_dimension(self, df: pd.DataFrame, current_data: pd.Series) -> Dict[str, Any]:
        """Analyze the WHY dimension (Macro Context)."""
        try:
            # Calculate trend using multiple timeframes
            short_trend = self._calculate_trend(df.tail(20))
            medium_trend = self._calculate_trend(df.tail(50))
            long_trend = self._calculate_trend(df.tail(200))
            
            # Determine macro trend
            if short_trend > 0.6 and medium_trend > 0.5 and long_trend > 0.4:
                trend = 'BULLISH'
                strength = (short_trend + medium_trend + long_trend) / 3
            elif short_trend < -0.6 and medium_trend < -0.5 and long_trend < -0.4:
                trend = 'BEARISH'
                strength = abs(short_trend + medium_trend + long_trend) / 3
            else:
                trend = 'NEUTRAL'
                strength = 0.3
            
            # Calculate volatility regime
            current_vol = df['close'].pct_change().tail(20).std()
            vol_ratio = current_vol / self.volatility_baseline
            
            if vol_ratio > 1.5:
                regime = 'VOLATILE'
            elif vol_ratio < 0.7:
                regime = 'RANGING'
            else:
                regime = 'TRENDING'
            
            return {
                'trend': trend,
                'strength': min(strength, 1.0),
                'volatility': vol_ratio,
                'regime': regime
            }
            
        except Exception as e:
            logger.error(f"Error analyzing WHY dimension: {e}")
            return {'trend': 'NEUTRAL', 'strength': 0.3, 'volatility': 1.0, 'regime': 'TRENDING'}
    
    def _analyze_how_dimension(self, df: pd.DataFrame, current_data: pd.Series) -> Dict[str, Any]:
        """Analyze the HOW dimension (Institutional Behavior)."""
        try:
            # Simulate institutional flow analysis
            # In a real implementation, this would use order book data
            price_momentum = df['close'].pct_change().tail(10).mean()
            volume_trend = df['volume'].tail(10).mean() / self.volume_baseline if 'volume' in df.columns else 1.0
            
            # Institutional flow based on price and volume patterns
            if price_momentum > 0.001 and volume_trend > 1.2:
                flow = min(price_momentum * 1000, 1.0)
                confidence = min(volume_trend - 1.0, 1.0)
            elif price_momentum < -0.001 and volume_trend > 1.2:
                flow = max(price_momentum * 1000, -1.0)
                confidence = min(volume_trend - 1.0, 1.0)
            else:
                flow = 0.0
                confidence = 0.3
            
            # Large order activity (simulated)
            large_order_activity = min(volume_trend - 1.0, 1.0) if volume_trend > 1.0 else 0.0
            
            # Order flow imbalance
            recent_volatility = df['close'].pct_change().tail(5).std()
            imbalance = (recent_volatility - self.volatility_baseline) / self.volatility_baseline
            imbalance = max(min(imbalance, 1.0), -1.0)
            
            return {
                'flow': flow,
                'confidence': confidence,
                'large_orders': large_order_activity,
                'imbalance': imbalance
            }
            
        except Exception as e:
            logger.error(f"Error analyzing HOW dimension: {e}")
            return {'flow': 0.0, 'confidence': 0.3, 'large_orders': 0.0, 'imbalance': 0.0}
    
    def _analyze_what_dimension(self, df: pd.DataFrame, current_data: pd.Series) -> Dict[str, Any]:
        """Analyze the WHAT dimension (Technical Analysis)."""
        try:
            # Calculate EMAs
            ema_signals = []
            for period in self.ema_periods:
                if len(df) >= period:
                    ema = df['close'].ewm(span=period).mean().iloc[-1]
                    current_price = current_data['close']
                    ema_signals.append(1 if current_price > ema else -1)
            
            # RSI
            if len(df) >= self.rsi_period:
                delta = df['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))
                current_rsi = rsi.iloc[-1]
            else:
                current_rsi = 50
            
            # Bollinger Bands
            if len(df) >= self.bb_period:
                bb_middle = df['close'].rolling(window=self.bb_period).mean()
                bb_std = df['close'].rolling(window=self.bb_period).std()
                bb_upper = bb_middle + (bb_std * self.bb_std)
                bb_lower = bb_middle - (bb_std * self.bb_std)
                
                current_price = current_data['close']
                bb_position = (current_price - bb_lower.iloc[-1]) / (bb_upper.iloc[-1] - bb_lower.iloc[-1])
                bb_position = max(min(bb_position, 1.0), 0.0)
            else:
                bb_position = 0.5
            
            # Generate technical signal
            bullish_signals = sum(1 for signal in ema_signals if signal > 0)
            bearish_signals = sum(1 for signal in ema_signals if signal < 0)
            
            if bullish_signals >= 3 and current_rsi < 70 and bb_position < 0.8:
                signal = 'BUY'
                strength = (bullish_signals / len(ema_signals)) * (1 - bb_position)
            elif bearish_signals >= 3 and current_rsi > 30 and bb_position > 0.2:
                signal = 'SELL'
                strength = (bearish_signals / len(ema_signals)) * bb_position
            else:
                signal = 'HOLD'
                strength = 0.3
            
            # Support and resistance levels
            recent_highs = df['high'].tail(20).nlargest(3)
            recent_lows = df['low'].tail(20).nsmallest(3)
            
            resistance = recent_highs.mean() if not recent_highs.empty else None
            support = recent_lows.mean() if not recent_lows.empty else None
            
            # Momentum and volatility scores
            momentum = (current_data['close'] - df['close'].iloc[-20]) / df['close'].iloc[-20]
            momentum = max(min(momentum, 1.0), -1.0)
            
            volatility = df['close'].pct_change().tail(20).std() / self.volatility_baseline
            volatility = min(volatility, 1.0)
            
            return {
                'signal': signal,
                'strength': strength,
                'support': support,
                'resistance': resistance,
                'momentum': momentum,
                'volatility': volatility
            }
            
        except Exception as e:
            logger.error(f"Error analyzing WHAT dimension: {e}")
            return {
                'signal': 'HOLD', 'strength': 0.3, 'support': None, 'resistance': None,
                'momentum': 0.0, 'volatility': 1.0
            }
    
    def _analyze_when_dimension(self, df: pd.DataFrame, current_data: pd.Series) -> Dict[str, Any]:
        """Analyze the WHEN dimension (Session/Time Context)."""
        try:
            current_time = current_data.name
            current_hour = current_time.hour
            
            # Determine session
            if 0 <= current_hour < 8:
                session = 'ASIAN'
            elif 8 <= current_hour < 16:
                session = 'LONDON'
            elif 13 <= current_hour < 21:
                session = 'NEW_YORK'
            else:
                session = 'OVERLAP'
            
            # Session volatility (simulated)
            session_volatility = 0.5  # Base volatility
            if session in ['LONDON', 'NEW_YORK']:
                session_volatility = 0.8
            elif session == 'OVERLAP':
                session_volatility = 1.0
            elif session == 'ASIAN':
                session_volatility = 0.3
            
            # Time of day score
            time_score = np.sin(2 * np.pi * current_hour / 24)  # Cyclical pattern
            
            # Day of week score
            day_score = 0.5  # Neutral for now
            if current_time.weekday() in [0, 4]:  # Monday, Friday
                day_score = 0.8  # Higher activity
            elif current_time.weekday() == 6:  # Sunday
                day_score = 0.2  # Lower activity
            
            # News impact (simulated)
            news_impact = 0.3  # Base level
            # In a real implementation, this would check economic calendar
            
            return {
                'session': session,
                'volatility': session_volatility,
                'time_score': time_score,
                'day_score': day_score,
                'news_impact': news_impact
            }
            
        except Exception as e:
            logger.error(f"Error analyzing WHEN dimension: {e}")
            return {
                'session': 'LONDON', 'volatility': 0.5, 'time_score': 0.0,
                'day_score': 0.5, 'news_impact': 0.3
            }
    
    def _analyze_anomaly_dimension(self, df: pd.DataFrame, current_data: pd.Series) -> Dict[str, Any]:
        """Analyze the ANOMALY dimension (Manipulation Detection)."""
        try:
            # Stop hunt detection
            recent_highs = df['high'].tail(10)
            recent_lows = df['low'].tail(10)
            current_price = current_data['close']
            
            # Check if price is near recent extremes
            near_high = any(abs(current_price - high) / high < 0.001 for high in recent_highs)
            near_low = any(abs(current_price - low) / low < 0.001 for low in recent_lows)
            
            stop_hunt_prob = 0.0
            if near_high or near_low:
                stop_hunt_prob = 0.6
            
            # Spoofing detection (simulated)
            volume_spike = False
            if 'volume' in df.columns:
                recent_volume = df['volume'].tail(5).mean()
                current_volume = current_data.get('volume', recent_volume)
                volume_spike = current_volume > recent_volume * 2
            
            spoofing_detected = volume_spike and stop_hunt_prob > 0.5
            
            # Liquidity zone activity
            liquidity_activity = 0.3  # Base level
            if stop_hunt_prob > 0.5:
                liquidity_activity = 0.8
            
            # Unusual volume
            unusual_volume = 0.3  # Base level
            if volume_spike:
                unusual_volume = 0.8
            
            # Overall manipulation probability
            manipulation_prob = max(stop_hunt_prob, 0.3 if spoofing_detected else 0.0)
            
            return {
                'manipulation': manipulation_prob,
                'stop_hunt': stop_hunt_prob,
                'spoofing': spoofing_detected,
                'liquidity': liquidity_activity,
                'unusual_volume': unusual_volume
            }
            
        except Exception as e:
            logger.error(f"Error analyzing ANOMALY dimension: {e}")
            return {
                'manipulation': 0.0, 'stop_hunt': 0.0, 'spoofing': False,
                'liquidity': 0.3, 'unusual_volume': 0.3
            }
    
    def _calculate_trend(self, df: pd.DataFrame) -> float:
        """Calculate trend strength using linear regression."""
        if len(df) < 2:
            return 0.0
        
        x = np.arange(len(df))
        y = df['close'].values
        
        slope, _, r_value, _, _ = stats.linregress(x, y)
        
        # Normalize slope and multiply by R-squared for confidence
        normalized_slope = slope / df['close'].mean()
        trend_strength = normalized_slope * (r_value ** 2)
        
        return max(min(trend_strength, 1.0), -1.0)
    
    def _calculate_overall_sentiment(self, why: Dict, how: Dict, what: Dict, 
                                   when: Dict, anomaly: Dict) -> float:
        """Calculate overall market sentiment."""
        # Weight the different dimensions
        why_weight = 0.3
        how_weight = 0.25
        what_weight = 0.25
        when_weight = 0.1
        anomaly_weight = 0.1
        
        # Convert trend to sentiment
        why_sentiment = 0.5
        if why['trend'] == 'BULLISH':
            why_sentiment = 0.5 + (why['strength'] * 0.5)
        elif why['trend'] == 'BEARISH':
            why_sentiment = 0.5 - (why['strength'] * 0.5)
        
        # How dimension sentiment
        how_sentiment = 0.5 + (how['flow'] * 0.5)
        
        # What dimension sentiment
        what_sentiment = 0.5
        if what['signal'] == 'BUY':
            what_sentiment = 0.5 + (what['strength'] * 0.5)
        elif what['signal'] == 'SELL':
            what_sentiment = 0.5 - (what['strength'] * 0.5)
        
        # When dimension sentiment
        when_sentiment = 0.5 + (when['time_score'] * 0.3) + (when['day_score'] * 0.2)
        
        # Anomaly dimension sentiment (manipulation reduces confidence)
        anomaly_sentiment = 0.5 - (anomaly['manipulation'] * 0.3)
        
        # Calculate weighted average
        overall = (why_sentiment * why_weight + 
                  how_sentiment * how_weight + 
                  what_sentiment * what_weight + 
                  when_sentiment * when_weight + 
                  anomaly_sentiment * anomaly_weight)
        
        return max(min(overall, 1.0), 0.0)
    
    def _calculate_confidence_level(self, why: Dict, how: Dict, what: Dict, 
                                  when: Dict, anomaly: Dict) -> float:
        """Calculate confidence level in the reading."""
        # Higher confidence when signals align and manipulation is low
        confidence = 0.5
        
        # Boost confidence if technical and macro signals align
        if (why['trend'] == 'BULLISH' and what['signal'] == 'BUY') or \
           (why['trend'] == 'BEARISH' and what['signal'] == 'SELL'):
            confidence += 0.2
        
        # Boost confidence with institutional flow
        confidence += abs(how['flow']) * 0.1
        
        # Reduce confidence with manipulation
        confidence -= anomaly['manipulation'] * 0.3
        
        return max(min(confidence, 1.0), 0.0)
    
    def _calculate_risk_level(self, why: Dict, how: Dict, what: Dict, 
                            when: Dict, anomaly: Dict) -> float:
        """Calculate risk level."""
        risk = 0.5  # Base risk
        
        # Higher risk with high volatility
        risk += why['volatility'] * 0.2
        
        # Higher risk with manipulation
        risk += anomaly['manipulation'] * 0.3
        
        # Higher risk with high session volatility
        risk += when['volatility'] * 0.1
        
        # Higher risk with unusual volume
        risk += anomaly['unusual_volume'] * 0.1
        
        return max(min(risk, 1.0), 0.0)
    
    def _calculate_data_quality(self, df: pd.DataFrame) -> float:
        """Calculate data quality score."""
        if df is None or df.empty:
            return 0.0
        
        # Check for missing data
        missing_ratio = df.isnull().sum().sum() / (len(df) * len(df.columns))
        quality = 1.0 - missing_ratio
        
        # Check for sufficient data points
        if len(df) < 50:
            quality *= 0.8
        
        return max(quality, 0.0)
    
    def _create_default_reading(self, timestamp: datetime) -> SensoryReading:
        """Create a default sensory reading when analysis fails."""
        return SensoryReading(
            timestamp=timestamp,
            macro_trend='NEUTRAL',
            macro_strength=0.3,
            macro_volatility=1.0,
            macro_regime='TRENDING',
            institutional_flow=0.0,
            institutional_confidence=0.3,
            large_order_activity=0.0,
            order_flow_imbalance=0.0,
            technical_signal='HOLD',
            technical_strength=0.3,
            support_level=None,
            resistance_level=None,
            momentum_score=0.0,
            volatility_score=1.0,
            session_phase='LONDON',
            session_volatility=0.5,
            time_of_day_score=0.0,
            day_of_week_score=0.5,
            news_impact_score=0.3,
            manipulation_probability=0.0,
            stop_hunt_probability=0.0,
            spoofing_detected=False,
            liquidity_zone_activity=0.3,
            unusual_volume=0.3,
            overall_sentiment=0.5,
            confidence_level=0.3,
            risk_level=0.5,
            processing_time_ms=0.0,
            data_quality_score=0.5
        ) 