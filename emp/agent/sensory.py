"""
SensoryCortex: 4D+1 cortex implementation for market perception.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import logging
from scipy import stats
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import NotFittedError

logger = logging.getLogger(__name__)

@dataclass
class SensoryReading:
    """Represents a complete sensory reading from the 4D+1 cortex"""
    timestamp: datetime
    symbol: str
    why_score: float      # Fundamental/macro momentum
    how_score: float      # Institutional footprint
    what_score: float     # Technical patterns
    when_score: float     # Timing/session analysis
    anomaly_score: float  # Deviation detection
    raw_components: Dict[str, Any]
    confidence: float

class SensoryCortex:
    """
    4D+1 Sensory Cortex for comprehensive market perception.
    
    The 4D+1 cortex consists of:
    - WHY: Fundamental/macro momentum analysis
    - HOW: Institutional footprint detection
    - WHAT: Technical pattern recognition
    - WHEN: Timing and session analysis
    - ANOMALY: Deviation and outlier detection
    """
    
    def __init__(self, symbol: str, data_storage):
        self.symbol = symbol
        self.data_storage = data_storage
        
        # Precomputed data
        self.ohlcv_data: Dict[str, pd.DataFrame] = {}
        self.technical_indicators: Dict[str, pd.DataFrame] = {}
        
        # Anomaly detection
        self.anomaly_detector: Optional[MLPRegressor] = None
        self.scaler: Optional[StandardScaler] = None
        self.anomaly_threshold = 0.7
        
        # Calibration data
        self.calibration_period: Optional[Dict] = None
        self.baseline_metrics: Dict[str, Dict] = {}
        
        # Session definitions
        self.sessions = {
            'asian': (datetime(2000, 1, 1, 0, 0), datetime(2000, 1, 1, 8, 0)),
            'london': (datetime(2000, 1, 1, 8, 0), datetime(2000, 1, 1, 16, 0)),
            'new_york': (datetime(2000, 1, 1, 13, 0), datetime(2000, 1, 1, 21, 0)),
            'overnight': (datetime(2000, 1, 1, 21, 0), datetime(2000, 1, 1, 0, 0))
        }
        
        logger.info(f"Initialized SensoryCortex for {symbol}")
    
    def calibrate(self, start_time: datetime, end_time: datetime):
        """
        Calibrate the sensory cortex with historical data.
        
        Args:
            start_time: Start of calibration period
            end_time: End of calibration period
        """
        logger.info(f"Calibrating SensoryCortex for {self.symbol} from {start_time} to {end_time}")
        
        # Load calibration data
        tick_data = self.data_storage.load_tick_data(self.symbol, start_time, end_time)
        
        if tick_data.empty:
            logger.warning("No calibration data available")
            return
        
        # Precompute OHLCV data
        self._precompute_ohlcv_data(tick_data)
        
        # Calculate baseline metrics
        self._calculate_baseline_metrics()
        
        # Train anomaly detector
        self._train_anomaly_detector()
        
        self.calibration_period = {
            'start_time': start_time,
            'end_time': end_time,
            'data_points': len(tick_data)
        }
        
        logger.info("SensoryCortex calibration complete")
    
    def _precompute_ohlcv_data(self, tick_data: pd.DataFrame):
        """Precompute OHLCV data at different timeframes."""
        # Convert to different timeframes
        timeframes = ['1T', '5T', '15T', '1H', '4H', '1D']
        
        for tf in timeframes:
            ohlcv = self._ticks_to_ohlcv(tick_data, tf)
            if not ohlcv.empty:
                # Add technical indicators
                ohlcv_with_indicators = self._add_technical_indicators(ohlcv)
                self.ohlcv_data[tf] = ohlcv_with_indicators
    
    def _ticks_to_ohlcv(self, tick_data: pd.DataFrame, freq: str) -> pd.DataFrame:
        """Convert tick data to OHLCV format."""
        if tick_data.empty:
            return pd.DataFrame()
        
        # Ensure timestamp is index
        tick_data = tick_data.set_index('timestamp')
        
        # Resample to OHLCV
        ohlcv = tick_data.resample(freq).agg({
            'bid': 'ohlc',
            'ask': 'ohlc',
            'bid_volume': 'sum',
            'ask_volume': 'sum'
        })
        
        if ohlcv.empty:
            return pd.DataFrame()
        
        # Flatten multi-index columns
        ohlcv.columns = [f"{col[0]}_{col[1]}" for col in ohlcv.columns]
        
        # Calculate OHLC from bid/ask
        ohlcv['open'] = (ohlcv['bid_open'] + ohlcv['ask_open']) / 2
        ohlcv['high'] = ohlcv['ask_high']
        ohlcv['low'] = ohlcv['bid_low']
        ohlcv['close'] = (ohlcv['bid_close'] + ohlcv['ask_close']) / 2
        ohlcv['volume'] = ohlcv['bid_volume'] + ohlcv['ask_volume']
        
        return ohlcv[['open', 'high', 'low', 'close', 'volume']].copy()
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to OHLCV data."""
        result = df.copy()
        
        # Calculate returns
        result['returns'] = result['close'].pct_change()
        result['log_returns'] = np.log(result['close'] / result['close'].shift(1))
        
        # Moving averages
        result['sma_20'] = result['close'].rolling(window=20).mean()
        result['sma_50'] = result['close'].rolling(window=50).mean()
        result['ema_12'] = result['close'].ewm(span=12).mean()
        result['ema_26'] = result['close'].ewm(span=26).mean()
        
        # MACD
        result['macd'] = result['ema_12'] - result['ema_26']
        result['macd_signal'] = result['macd'].ewm(span=9).mean()
        result['macd_histogram'] = result['macd'] - result['macd_signal']
        
        # RSI
        delta = result['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        result['rsi'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        result['bb_middle'] = result['close'].rolling(window=20).mean()
        bb_std = result['close'].rolling(window=20).std()
        result['bb_upper'] = result['bb_middle'] + (bb_std * 2)
        result['bb_lower'] = result['bb_middle'] - (bb_std * 2)
        result['bb_width'] = (result['bb_upper'] - result['bb_lower']) / result['bb_middle']
        
        # ATR
        high_low = result['high'] - result['low']
        high_close = np.abs(result['high'] - result['close'].shift())
        low_close = np.abs(result['low'] - result['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        result['atr'] = true_range.rolling(window=14).mean()
        
        # Volume indicators
        result['volume_sma'] = result['volume'].rolling(window=20).mean()
        result['volume_ratio'] = result['volume'] / result['volume_sma']
        
        # Volatility
        result['volatility'] = result['returns'].rolling(window=20).std() * np.sqrt(252)
        
        # Momentum indicators
        result['momentum'] = result['close'] - result['close'].shift(10)
        result['roc'] = (result['close'] / result['close'].shift(10) - 1) * 100
        
        # Always return as DataFrame
        return pd.DataFrame(result)
    
    def _calculate_baseline_metrics(self):
        """Calculate baseline metrics for each timeframe."""
        for tf, df in self.ohlcv_data.items():
            if df.empty:
                continue
            
            # Calculate baseline statistics
            baseline = {
                'mean_return': df['returns'].mean(),
                'std_return': df['returns'].std(),
                'mean_volume': df['volume'].mean(),
                'std_volume': df['volume'].std(),
                'mean_atr': df['atr'].mean(),
                'mean_volatility': df['volatility'].mean(),
                'price_range': (df['high'].max() - df['low'].min()) / df['close'].mean(),
                'volume_profile': df['volume'].quantile([0.25, 0.5, 0.75]).to_dict()
            }
            
            self.baseline_metrics[tf] = baseline
    
    def _train_anomaly_detector(self):
        """Train the anomaly detection model."""
        try:
            # Prepare training data
            features = []
            for tf, df in self.ohlcv_data.items():
                if df.empty:
                    continue
                
                # Select relevant features for anomaly detection
                feature_cols = ['returns', 'volume_ratio', 'rsi', 'bb_width', 'atr', 'volatility']
                available_cols = [col for col in feature_cols if col in df.columns]
                
                if len(available_cols) >= 3:
                    feature_data = df[available_cols].dropna()
                    features.append(feature_data)
            
            if not features:
                logger.warning("No features available for anomaly detection training")
                return
            
            # Combine all features
            all_features = pd.concat(features, ignore_index=True)
            
            # Remove outliers for training
            if not isinstance(all_features, pd.DataFrame):
                all_features = pd.DataFrame(all_features)
            # Only use numeric columns for zscore
            numeric_features = all_features.select_dtypes(include=[np.number])
            z_scores = np.abs(stats.zscore(numeric_features, axis=0))
            clean_features = all_features[(z_scores < 3).all(axis=1)]
            
            if len(clean_features) < 100:
                logger.warning("Insufficient clean data for anomaly detection training")
                return
            
            # Train model
            self.scaler = StandardScaler()
            scaled_features = self.scaler.fit_transform(clean_features)
            
            self.anomaly_detector = MLPRegressor(
                hidden_layer_sizes=(50, 25),
                max_iter=1000,
                random_state=42
            )
            
            # Train on reconstruction task
            self.anomaly_detector.fit(scaled_features, scaled_features)
            
            logger.info(f"Anomaly detector trained on {len(clean_features)} samples")
            
        except Exception as e:
            logger.error(f"Failed to train anomaly detector: {e}")
    
    def perceive(self, market_state) -> SensoryReading:
        """
        Generate a complete sensory reading from current market state.
        
        Args:
            market_state: Current market state
            
        Returns:
            SensoryReading with all 4D+1 scores
        """
        # Get current OHLCV data
        current_ohlcv = self._get_current_ohlcv(market_state)
        
        # Calculate 4D+1 scores
        why_score = self._calculate_why_score(market_state, current_ohlcv)
        how_score = self._calculate_how_score(market_state, current_ohlcv)
        what_score = self._calculate_what_score(market_state, current_ohlcv)
        when_score = self._calculate_when_score(market_state)
        anomaly_score = self._calculate_anomaly_score(market_state)
        
        # Calculate confidence
        confidence = self._calculate_confidence(market_state, current_ohlcv)
        
        # Create raw components
        raw_components = {
            'why': self._get_why_components(market_state, current_ohlcv),
            'how': self._get_how_components(market_state, current_ohlcv),
            'what': self._get_what_components(market_state, current_ohlcv),
            'when': self._get_when_components(market_state),
            'anomaly': self._get_anomaly_components(market_state)
        }
        
        return SensoryReading(
            timestamp=market_state.timestamp,
            symbol=self.symbol,
            why_score=why_score,
            how_score=how_score,
            what_score=what_score,
            when_score=when_score,
            anomaly_score=anomaly_score,
            raw_components=raw_components,
            confidence=confidence
        )
    
    def _get_current_ohlcv(self, market_state) -> Optional[Dict[str, pd.DataFrame]]:
        """Get current OHLCV data for all timeframes."""
        if not self.ohlcv_data:
            return None
        
        # Return the most recent data for each timeframe
        current_data = {}
        for tf, df in self.ohlcv_data.items():
            if not df.empty:
                # Get recent data (last 100 bars)
                recent_data = df.tail(100)
                current_data[tf] = recent_data
        
        return current_data
    
    def _calculate_why_score(self, market_state, current_ohlcv: Optional[Dict[str, pd.DataFrame]]) -> float:
        """
        Calculate WHY score (Fundamental/macro momentum).
        
        This analyzes the underlying momentum and trend strength.
        """
        if not current_ohlcv:
            return 0.5
        
        why_components = []
        
        # Trend strength analysis
        for tf, df in current_ohlcv.items():
            if len(df) < 20:
                continue
            
            # Price trend
            price_trend = (df['close'].iloc[-1] - df['close'].iloc[-20]) / df['close'].iloc[-20]
            why_components.append(price_trend)
            
            # Moving average alignment
            if 'sma_20' in df.columns and 'sma_50' in df.columns:
                ma_alignment = 1.0 if df['sma_20'].iloc[-1] > df['sma_50'].iloc[-1] else -1.0
                why_components.append(ma_alignment * 0.5)
            
            # MACD momentum
            if 'macd' in df.columns and 'macd_signal' in df.columns:
                macd_momentum = 1.0 if df['macd'].iloc[-1] > df['macd_signal'].iloc[-1] else -1.0
                why_components.append(macd_momentum * 0.3)
            
            # RSI momentum
            if 'rsi' in df.columns:
                rsi = df['rsi'].iloc[-1]
                if rsi > 70:
                    rsi_momentum = -0.5  # Overbought
                elif rsi < 30:
                    rsi_momentum = 0.5   # Oversold
                else:
                    rsi_momentum = (rsi - 50) / 50  # Neutral
                why_components.append(rsi_momentum)
        
        if not why_components:
            return 0.5
        
        # Calculate weighted average
        why_score = np.mean(why_components)
        
        # Normalize to 0-1 range
        why_score = (why_score + 1) / 2
        why_score = max(0.0, min(float(why_score), 1.0))
        
        return why_score
    
    def _calculate_how_score(self, market_state, current_ohlcv: Optional[Dict[str, pd.DataFrame]]) -> float:
        """
        Calculate HOW score (Institutional footprint).
        
        This analyzes volume patterns and institutional activity.
        """
        if not current_ohlcv:
            return 0.5
        
        how_components = []
        
        # Volume analysis
        for tf, df in current_ohlcv.items():
            if len(df) < 20:
                continue
            
            # Volume surge detection
            if 'volume_ratio' in df.columns:
                volume_ratio = df['volume_ratio'].iloc[-1]
                if volume_ratio > 2.0:
                    how_components.append(0.8)  # High institutional activity
                elif volume_ratio > 1.5:
                    how_components.append(0.6)
                elif volume_ratio < 0.5:
                    how_components.append(0.2)  # Low activity
                else:
                    how_components.append(0.5)
            
            # Price-volume relationship
            if 'returns' in df.columns and 'volume_ratio' in df.columns:
                price_change = df['returns'].iloc[-1]
                volume_change = df['volume_ratio'].iloc[-1]
                
                # Strong price movement with volume confirms institutional activity
                if abs(price_change) > 0.001 and volume_change > 1.2:
                    how_components.append(0.7)
                elif abs(price_change) < 0.0001 and volume_change < 0.8:
                    how_components.append(0.3)
                else:
                    how_components.append(0.5)
            
            # ATR analysis (volatility expansion indicates institutional activity)
            if 'atr' in df.columns:
                atr = df['atr'].iloc[-1]
                if tf in self.baseline_metrics:
                    baseline_atr = self.baseline_metrics[tf]['mean_atr']
                    if baseline_atr > 0:
                        atr_ratio = atr / baseline_atr
                        if atr_ratio > 1.5:
                            how_components.append(0.8)
                        elif atr_ratio < 0.7:
                            how_components.append(0.3)
                        else:
                            how_components.append(0.5)
        
        if not how_components:
            return 0.5
        
        how_score = np.mean(how_components)
        return max(0.0, min(float(how_score), 1.0))
    
    def _calculate_what_score(self, market_state, current_ohlcv: Optional[Dict[str, pd.DataFrame]]) -> float:
        """
        Calculate WHAT score (Technical patterns).
        
        This analyzes technical patterns and chart formations.
        """
        if not current_ohlcv:
            return 0.5
        
        what_components = []
        
        for tf, df in current_ohlcv.items():
            if len(df) < 50:
                continue
            
            # Bollinger Band position
            if all(col in df.columns for col in ['close', 'bb_upper', 'bb_lower']):
                close = df['close'].iloc[-1]
                bb_upper = df['bb_upper'].iloc[-1]
                bb_lower = df['bb_lower'].iloc[-1]
                
                if close > bb_upper:
                    bb_signal = 0.2  # Overbought
                elif close < bb_lower:
                    bb_signal = 0.8  # Oversold
                else:
                    bb_signal = 0.5  # Neutral
                what_components.append(bb_signal)
            
            # RSI patterns
            if 'rsi' in df.columns:
                rsi = df['rsi'].iloc[-1]
                if rsi > 80:
                    rsi_signal = 0.1  # Extremely overbought
                elif rsi > 70:
                    rsi_signal = 0.3  # Overbought
                elif rsi < 20:
                    rsi_signal = 0.9  # Extremely oversold
                elif rsi < 30:
                    rsi_signal = 0.7  # Oversold
                else:
                    rsi_signal = 0.5  # Neutral
                what_components.append(rsi_signal)
            
            # MACD patterns
            if all(col in df.columns for col in ['macd', 'macd_signal', 'macd_histogram']):
                macd = df['macd'].iloc[-1]
                macd_signal = df['macd_signal'].iloc[-1]
                macd_hist = df['macd_histogram'].iloc[-1]
                
                # MACD crossover
                if macd > macd_signal and macd_hist > 0:
                    macd_signal_score = 0.7  # Bullish
                elif macd < macd_signal and macd_hist < 0:
                    macd_signal_score = 0.3  # Bearish
                else:
                    macd_signal_score = 0.5  # Neutral
                what_components.append(macd_signal_score)
            
            # Support/Resistance levels
            if len(df) >= 20:
                current_price = df['close'].iloc[-1]
                recent_high = df['high'].tail(20).max()
                recent_low = df['low'].tail(20).min()
                
                # Distance to support/resistance
                distance_to_high = (recent_high - current_price) / current_price
                distance_to_low = (current_price - recent_low) / current_price
                
                if distance_to_high < 0.001:  # Near resistance
                    level_signal = 0.3
                elif distance_to_low < 0.001:  # Near support
                    level_signal = 0.7
                else:
                    level_signal = 0.5
                what_components.append(level_signal)
        
        if not what_components:
            return 0.5
        
        what_score = np.mean(what_components)
        return max(0.0, min(float(what_score), 1.0))
    
    def _calculate_when_score(self, market_state) -> float:
        """
        Calculate WHEN score (Timing/session analysis).
        
        This analyzes the timing and session context.
        """
        # Get current session
        current_time = market_state.timestamp.time()
        current_session = self._get_current_session(current_time)
        
        # Session-based scoring
        session_scores = {
            'asian': 0.4,      # Lower volatility
            'london': 0.7,     # High activity
            'new_york': 0.8,   # Highest activity
            'overnight': 0.3   # Low activity
        }
        
        base_score = session_scores.get(current_session, 0.5)
        
        # Time-based adjustments
        hour = current_time.hour
        
        # Overlap periods (higher activity)
        if 8 <= hour <= 12:  # London-Asian overlap
            base_score *= 1.1
        elif 13 <= hour <= 17:  # London-NY overlap
            base_score *= 1.2
        
        # Weekend effect (if applicable)
        if market_state.timestamp.weekday() >= 5:  # Weekend
            base_score *= 0.5
        
        return max(0.0, min(1.0, base_score))
    
    def _get_current_session(self, current_time) -> str:
        """Get current trading session."""
        for session, (start, end) in self.sessions.items():
            if start.time() <= current_time <= end.time():
                return session
        return 'overnight'
    
    def _calculate_anomaly_score(self, market_state) -> float:
        """
        Calculate ANOMALY score (Deviation detection).
        
        This detects unusual market behavior and outliers.
        """
        # Simple anomaly detection based on price and volume
        anomaly_components = []
        
        # Price anomaly (if we have recent price history)
        if hasattr(market_state, 'mid_price'):
            # This would be more sophisticated with historical context
            anomaly_components.append(0.5)
        
        # Volume anomaly
        if hasattr(market_state, 'bid_volume') and hasattr(market_state, 'ask_volume'):
            total_volume = market_state.bid_volume + market_state.ask_volume
            # Simple volume anomaly detection
            if total_volume > 10000:  # High volume
                anomaly_components.append(0.8)
            elif total_volume < 1000:  # Low volume
                anomaly_components.append(0.3)
            else:
                anomaly_components.append(0.5)
        
        # Spread anomaly
        if hasattr(market_state, 'spread'):
            spread_bps = market_state.spread / market_state.mid_price * 10000
            if spread_bps > 10:  # Wide spread
                anomaly_components.append(0.8)
            elif spread_bps < 1:  # Tight spread
                anomaly_components.append(0.3)
            else:
                anomaly_components.append(0.5)
        
        if not anomaly_components:
            return 0.5
        
        anomaly_score = np.mean(anomaly_components)
        return max(0.0, min(float(anomaly_score), 1.0))
    
    def _calculate_confidence(self, market_state, current_ohlcv: Optional[Dict[str, pd.DataFrame]]) -> float:
        """
        Calculate confidence in the sensory reading.
        
        Higher confidence when we have more data and consistent signals.
        """
        confidence_components = []
        
        # Data availability
        if current_ohlcv and len(current_ohlcv) >= 3:
            confidence_components.append(0.8)
        elif current_ohlcv and len(current_ohlcv) >= 1:
            confidence_components.append(0.6)
        else:
            confidence_components.append(0.3)
        
        # Data quality (no missing values)
        if current_ohlcv:
            for tf, df in current_ohlcv.items():
                if not df.empty and not df.isnull().values.any():
                    confidence_components.append(0.7)
                else:
                    confidence_components.append(0.4)
        
        # Calibration status
        if self.calibration_period:
            confidence_components.append(0.8)
        else:
            confidence_components.append(0.3)
        
        if not confidence_components:
            return 0.5
        
        confidence = np.mean(confidence_components)
        return max(0.0, min(float(confidence), 1.0))
    
    def _get_why_components(self, market_state, current_ohlcv: Optional[Dict[str, pd.DataFrame]]) -> Dict:
        """Get detailed WHY components."""
        return {
            'trend_strength': 0.5,  # Placeholder
            'momentum': 0.5,        # Placeholder
            'ma_alignment': 0.5     # Placeholder
        }
    
    def _get_how_components(self, market_state, current_ohlcv: Optional[Dict[str, pd.DataFrame]]) -> Dict:
        """Get detailed HOW components."""
        return {
            'volume_profile': 0.5,  # Placeholder
            'institutional_activity': 0.5,  # Placeholder
            'volatility_regime': 0.5  # Placeholder
        }
    
    def _get_what_components(self, market_state, current_ohlcv: Optional[Dict[str, pd.DataFrame]]) -> Dict:
        """Get detailed WHAT components."""
        return {
            'technical_patterns': 0.5,  # Placeholder
            'support_resistance': 0.5,  # Placeholder
            'oscillator_signals': 0.5   # Placeholder
        }
    
    def _get_when_components(self, market_state) -> Dict:
        """Get detailed WHEN components."""
        current_time = market_state.timestamp.time()
        current_session = self._get_current_session(current_time)
        
        return {
            'current_session': current_session,
            'session_activity': 0.5,  # Placeholder
            'time_of_day': current_time.hour
        }
    
    def _get_anomaly_components(self, market_state) -> Dict:
        """Get detailed ANOMALY components."""
        return {
            'price_anomaly': 0.5,    # Placeholder
            'volume_anomaly': 0.5,   # Placeholder
            'spread_anomaly': 0.5    # Placeholder
        } 