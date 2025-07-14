"""
Enhanced ANOMALY Dimension - Self-Refuting Chaos Intelligence Engine

This module implements sophisticated anomaly detection that goes beyond simple statistical outliers
to understand market manipulation, chaos patterns, and self-refuting behaviors. It serves as the
system's reality check and strengthens from market stress (antifragile design).
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple, NamedTuple
from collections import deque, defaultdict
from dataclasses import dataclass, field
from enum import Enum, auto
import math
from scipy import stats
from scipy.signal import find_peaks
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import logging

from ..core.base import DimensionalReading, MarketData, MarketRegime

logger = logging.getLogger(__name__)

class AnomalyType(Enum):
    """Types of market anomalies"""
    STATISTICAL_OUTLIER = auto()      # Price/volume statistical anomalies
    MANIPULATION_PATTERN = auto()     # Market manipulation signatures
    STRUCTURAL_BREAK = auto()         # Regime change anomalies
    LIQUIDITY_ANOMALY = auto()        # Unusual liquidity patterns
    TEMPORAL_ANOMALY = auto()         # Time-based irregularities
    CORRELATION_BREAK = auto()        # Cross-dimensional correlation breaks
    SELF_REFUTATION = auto()          # Model prediction failures
    CHAOS_EMERGENCE = auto()          # Chaotic behavior patterns

class ManipulationSignature(Enum):
    """Known manipulation patterns"""
    STOP_HUNTING = auto()             # Liquidity sweeps beyond key levels
    SPOOFING = auto()                 # Fake order placement patterns
    LAYERING = auto()                 # Multiple order layers to mislead
    WASH_TRADING = auto()             # Artificial volume creation
    PUMP_AND_DUMP = auto()            # Coordinated price manipulation
    ICEBERG_HUNTING = auto()          # Large order detection and targeting
    MOMENTUM_IGNITION = auto()        # Artificial momentum creation

class ChaosPattern(Enum):
    """Chaotic behavior patterns"""
    BUTTERFLY_EFFECT = auto()         # Small changes, large consequences
    STRANGE_ATTRACTOR = auto()        # Unusual price attraction points
    FRACTAL_BREAKDOWN = auto()        # Self-similarity breakdown
    PHASE_TRANSITION = auto()         # Sudden state changes
    EMERGENCE = auto()                # New behavior patterns emerging

@dataclass
class AnomalyEvent:
    """Detected anomaly event"""
    anomaly_type: AnomalyType
    timestamp: datetime
    severity: float  # 0-1
    confidence: float  # 0-1
    description: str
    affected_dimensions: List[str]
    market_impact: float  # Expected impact on market behavior
    duration_estimate: Optional[timedelta] = None
    resolution_probability: float = 0.5  # Probability of self-resolution

@dataclass
class ManipulationEvent:
    """Detected manipulation event"""
    signature: ManipulationSignature
    timestamp: datetime
    strength: float  # 0-1
    target_level: Optional[float]  # Price level being targeted
    volume_signature: float  # Volume pattern strength
    success_probability: float  # Probability manipulation will succeed
    counter_strategy: str  # Suggested counter-strategy

@dataclass
class ChaosEvent:
    """Detected chaos event"""
    pattern: ChaosPattern
    timestamp: datetime
    intensity: float  # 0-1
    predictability: float  # 0-1 (lower = more chaotic)
    emergence_rate: float  # Rate of new pattern emergence
    system_stress: float  # Stress on market system

@dataclass
class SelfRefutationEvent:
    """Model self-refutation event"""
    prediction_type: str
    predicted_value: float
    actual_value: float
    error_magnitude: float
    confidence_was: float
    learning_opportunity: str

class StatisticalAnomalyDetector:
    """
    Advanced statistical anomaly detection using multiple methods
    """
    
    def __init__(self, lookback_periods: int = 200):
        self.lookback_periods = lookback_periods
        
        # Data storage
        self.price_history = deque(maxlen=lookback_periods)
        self.volume_history = deque(maxlen=lookback_periods)
        self.volatility_history = deque(maxlen=lookback_periods)
        self.timestamp_history = deque(maxlen=lookback_periods)
        
        # Anomaly detection models
        self.isolation_forest = IsolationForest(contamination=0.1, random_state=42)
        self.scaler = StandardScaler()
        
        # Adaptive thresholds
        self.price_threshold = 3.0  # Standard deviations
        self.volume_threshold = 3.0
        self.volatility_threshold = 2.5
        
        # Model state
        self.model_trained = False
        self.last_training_time = None
        
    def update_data(self, market_data: MarketData) -> None:
        """Update with new market data"""
        
        mid_price = (market_data.bid + market_data.ask) / 2
        
        self.price_history.append(mid_price)
        self.volume_history.append(market_data.volume)
        self.volatility_history.append(market_data.volatility)
        self.timestamp_history.append(market_data.timestamp)
        
        # Retrain model periodically
        if (len(self.price_history) >= 50 and 
            (not self.model_trained or 
             (self.last_training_time and 
              market_data.timestamp - self.last_training_time > timedelta(hours=6)))):
            self._train_anomaly_models()
    
    def _train_anomaly_models(self) -> None:
        """Train anomaly detection models"""
        
        if len(self.price_history) < 50:
            return
        
        try:
            # Prepare features
            features = self._extract_features()
            
            if len(features) > 10:
                # Scale features
                scaled_features = self.scaler.fit_transform(features)
                
                # Train isolation forest
                self.isolation_forest.fit(scaled_features)
                
                self.model_trained = True
                self.last_training_time = list(self.timestamp_history)[-1]
                
                # Update adaptive thresholds
                self._update_adaptive_thresholds(features)
                
        except Exception as e:
            logger.warning(f"Failed to train anomaly models: {e}")
    
    def _extract_features(self) -> np.ndarray:
        """Extract features for anomaly detection"""
        
        if len(self.price_history) < 20:
            return np.array([])
        
        prices = np.array(list(self.price_history))
        volumes = np.array(list(self.volume_history))
        volatilities = np.array(list(self.volatility_history))
        
        features = []
        
        # Use sliding window to create feature vectors
        window_size = 20
        for i in range(window_size, len(prices)):
            window_prices = prices[i-window_size:i]
            window_volumes = volumes[i-window_size:i]
            window_volatilities = volatilities[i-window_size:i]
            
            feature_vector = [
                # Price features
                np.mean(window_prices),
                np.std(window_prices),
                np.max(window_prices) - np.min(window_prices),
                (window_prices[-1] - window_prices[0]) / window_prices[0],
                
                # Volume features
                np.mean(window_volumes),
                np.std(window_volumes),
                np.max(window_volumes) / np.mean(window_volumes),
                
                # Volatility features
                np.mean(window_volatilities),
                np.std(window_volatilities),
                
                # Cross-correlations
                np.corrcoef(window_prices, window_volumes)[0, 1] if len(window_prices) > 1 else 0,
                np.corrcoef(window_prices, window_volatilities)[0, 1] if len(window_prices) > 1 else 0,
                
                # Momentum features
                np.sum(np.diff(window_prices) > 0) / len(window_prices),  # Positive momentum ratio
                
                # Microstructure features
                np.mean(np.abs(np.diff(window_prices))),  # Average price change
                len(find_peaks(window_prices)[0]) / len(window_prices),  # Peak density
            ]
            
            # Handle NaN values
            feature_vector = [f if not np.isnan(f) and not np.isinf(f) else 0.0 for f in feature_vector]
            features.append(feature_vector)
        
        return np.array(features)
    
    def _update_adaptive_thresholds(self, features: np.ndarray) -> None:
        """Update adaptive thresholds based on recent data"""
        
        if len(features) < 10:
            return
        
        # Calculate recent volatility of features
        recent_features = features[-20:] if len(features) >= 20 else features
        
        # Adapt thresholds based on feature volatility
        feature_volatility = np.mean(np.std(recent_features, axis=0))
        
        # Higher feature volatility = higher thresholds (less sensitive)
        volatility_factor = min(feature_volatility * 10, 2.0)
        
        self.price_threshold = 3.0 * volatility_factor
        self.volume_threshold = 3.0 * volatility_factor
        self.volatility_threshold = 2.5 * volatility_factor
    
    def detect_statistical_anomalies(self, market_data: MarketData) -> List[AnomalyEvent]:
        """Detect statistical anomalies in market data"""
        
        anomalies = []
        
        if len(self.price_history) < 20:
            return anomalies
        
        # Z-score based detection
        z_score_anomalies = self._detect_zscore_anomalies(market_data)
        anomalies.extend(z_score_anomalies)
        
        # Isolation forest detection
        if self.model_trained:
            isolation_anomalies = self._detect_isolation_anomalies(market_data)
            anomalies.extend(isolation_anomalies)
        
        # Distribution-based detection
        distribution_anomalies = self._detect_distribution_anomalies(market_data)
        anomalies.extend(distribution_anomalies)
        
        return anomalies
    
    def _detect_zscore_anomalies(self, market_data: MarketData) -> List[AnomalyEvent]:
        """Detect anomalies using Z-score method"""
        
        anomalies = []
        
        if len(self.price_history) < 10:
            return anomalies
        
        mid_price = (market_data.bid + market_data.ask) / 2
        
        # Price anomalies
        prices = np.array(list(self.price_history))
        price_mean = np.mean(prices)
        price_std = np.std(prices)
        
        if price_std > 0:
            price_zscore = abs(mid_price - price_mean) / price_std
            
            if price_zscore > self.price_threshold:
                anomalies.append(AnomalyEvent(
                    anomaly_type=AnomalyType.STATISTICAL_OUTLIER,
                    timestamp=market_data.timestamp,
                    severity=min(price_zscore / self.price_threshold - 1.0, 1.0),
                    confidence=min(price_zscore / 5.0, 1.0),
                    description=f"Price Z-score anomaly: {price_zscore:.2f}",
                    affected_dimensions=['WHAT'],
                    market_impact=min(price_zscore / 10.0, 0.5)
                ))
        
        # Volume anomalies
        volumes = np.array(list(self.volume_history))
        volume_mean = np.mean(volumes)
        volume_std = np.std(volumes)
        
        if volume_std > 0:
            volume_zscore = abs(market_data.volume - volume_mean) / volume_std
            
            if volume_zscore > self.volume_threshold:
                anomalies.append(AnomalyEvent(
                    anomaly_type=AnomalyType.STATISTICAL_OUTLIER,
                    timestamp=market_data.timestamp,
                    severity=min(volume_zscore / self.volume_threshold - 1.0, 1.0),
                    confidence=min(volume_zscore / 5.0, 1.0),
                    description=f"Volume Z-score anomaly: {volume_zscore:.2f}",
                    affected_dimensions=['HOW'],
                    market_impact=min(volume_zscore / 15.0, 0.3)
                ))
        
        # Volatility anomalies
        volatilities = np.array(list(self.volatility_history))
        volatility_mean = np.mean(volatilities)
        volatility_std = np.std(volatilities)
        
        if volatility_std > 0:
            volatility_zscore = abs(market_data.volatility - volatility_mean) / volatility_std
            
            if volatility_zscore > self.volatility_threshold:
                anomalies.append(AnomalyEvent(
                    anomaly_type=AnomalyType.STATISTICAL_OUTLIER,
                    timestamp=market_data.timestamp,
                    severity=min(volatility_zscore / self.volatility_threshold - 1.0, 1.0),
                    confidence=min(volatility_zscore / 4.0, 1.0),
                    description=f"Volatility Z-score anomaly: {volatility_zscore:.2f}",
                    affected_dimensions=['WHAT', 'WHEN'],
                    market_impact=min(volatility_zscore / 8.0, 0.4)
                ))
        
        return anomalies
    
    def _detect_isolation_anomalies(self, market_data: MarketData) -> List[AnomalyEvent]:
        """Detect anomalies using Isolation Forest"""
        
        anomalies = []
        
        if not self.model_trained or len(self.price_history) < 20:
            return anomalies
        
        try:
            # Extract current features
            current_features = self._extract_current_features(market_data)
            
            if len(current_features) > 0:
                # Scale features
                scaled_features = self.scaler.transform([current_features])
                
                # Get anomaly score
                anomaly_score = self.isolation_forest.decision_function(scaled_features)[0]
                is_anomaly = self.isolation_forest.predict(scaled_features)[0] == -1
                
                if is_anomaly:
                    # Convert anomaly score to severity (more negative = more anomalous)
                    severity = min(abs(anomaly_score) / 0.5, 1.0)
                    confidence = min(abs(anomaly_score) / 0.3, 1.0)
                    
                    anomalies.append(AnomalyEvent(
                        anomaly_type=AnomalyType.STRUCTURAL_BREAK,
                        timestamp=market_data.timestamp,
                        severity=severity,
                        confidence=confidence,
                        description=f"Isolation Forest anomaly: score {anomaly_score:.3f}",
                        affected_dimensions=['WHAT', 'HOW'],
                        market_impact=severity * 0.4
                    ))
        
        except Exception as e:
            logger.warning(f"Isolation Forest detection failed: {e}")
        
        return anomalies
    
    def _extract_current_features(self, market_data: MarketData) -> List[float]:
        """Extract features for current market data"""
        
        if len(self.price_history) < 20:
            return []
        
        mid_price = (market_data.bid + market_data.ask) / 2
        
        # Get recent window
        window_size = 20
        recent_prices = np.array(list(self.price_history)[-window_size:])
        recent_volumes = np.array(list(self.volume_history)[-window_size:])
        recent_volatilities = np.array(list(self.volatility_history)[-window_size:])
        
        # Add current data point
        current_prices = np.append(recent_prices, mid_price)
        current_volumes = np.append(recent_volumes, market_data.volume)
        current_volatilities = np.append(recent_volatilities, market_data.volatility)
        
        features = [
            # Price features
            np.mean(current_prices),
            np.std(current_prices),
            np.max(current_prices) - np.min(current_prices),
            (current_prices[-1] - current_prices[0]) / current_prices[0],
            
            # Volume features
            np.mean(current_volumes),
            np.std(current_volumes),
            np.max(current_volumes) / np.mean(current_volumes),
            
            # Volatility features
            np.mean(current_volatilities),
            np.std(current_volatilities),
            
            # Cross-correlations
            np.corrcoef(current_prices, current_volumes)[0, 1] if len(current_prices) > 1 else 0,
            np.corrcoef(current_prices, current_volatilities)[0, 1] if len(current_prices) > 1 else 0,
            
            # Momentum features
            np.sum(np.diff(current_prices) > 0) / len(current_prices),
            
            # Microstructure features
            np.mean(np.abs(np.diff(current_prices))),
            len(find_peaks(current_prices)[0]) / len(current_prices),
        ]
        
        # Handle NaN values
        features = [f if not np.isnan(f) and not np.isinf(f) else 0.0 for f in features]
        
        return features
    
    def _detect_distribution_anomalies(self, market_data: MarketData) -> List[AnomalyEvent]:
        """Detect anomalies using distribution analysis"""
        
        anomalies = []
        
        if len(self.price_history) < 30:
            return anomalies
        
        # Test for normality breaks
        recent_prices = np.array(list(self.price_history)[-30:])
        price_returns = np.diff(recent_prices) / recent_prices[:-1]
        
        # Shapiro-Wilk test for normality
        try:
            statistic, p_value = stats.shapiro(price_returns)
            
            if p_value < 0.01:  # Strong evidence against normality
                anomalies.append(AnomalyEvent(
                    anomaly_type=AnomalyType.STRUCTURAL_BREAK,
                    timestamp=market_data.timestamp,
                    severity=1.0 - p_value,
                    confidence=1.0 - p_value,
                    description=f"Distribution normality break: p-value {p_value:.4f}",
                    affected_dimensions=['WHAT'],
                    market_impact=0.3
                ))
        except Exception as e:
            logger.debug(f"Statistical anomaly detection failed: {e}")
            # Continue with other anomaly detection methods
        
        # Test for volatility clustering breaks
        volatilities = np.array(list(self.volatility_history)[-20:])
        if len(volatilities) >= 10:
            # ARCH test (simplified)
            vol_squared = volatilities ** 2
            vol_mean = np.mean(vol_squared)
            vol_var = np.var(vol_squared)
            
            if vol_var > vol_mean * 5:  # High variance in volatility
                anomalies.append(AnomalyEvent(
                    anomaly_type=AnomalyType.STRUCTURAL_BREAK,
                    timestamp=market_data.timestamp,
                    severity=min(vol_var / (vol_mean * 5) - 1.0, 1.0),
                    confidence=0.6,
                    description="Volatility clustering break detected",
                    affected_dimensions=['WHAT', 'WHEN'],
                    market_impact=0.4
                ))
        
        return anomalies

class ManipulationDetector:
    """
    Advanced manipulation detection using pattern recognition
    """
    
    def __init__(self, lookback_periods: int = 100):
        self.lookback_periods = lookback_periods
        
        # Data storage
        self.price_history = deque(maxlen=lookback_periods)
        self.volume_history = deque(maxlen=lookback_periods)
        self.bid_history = deque(maxlen=lookback_periods)
        self.ask_history = deque(maxlen=lookback_periods)
        self.timestamp_history = deque(maxlen=lookback_periods)
        
        # Manipulation tracking
        self.active_manipulations: List[ManipulationEvent] = []
        
        # Key levels for stop hunting detection
        self.key_levels = []
        self.level_test_count = defaultdict(int)
        
    def update_data(self, market_data: MarketData) -> None:
        """Update with new market data"""
        
        mid_price = (market_data.bid + market_data.ask) / 2
        
        self.price_history.append(mid_price)
        self.volume_history.append(market_data.volume)
        self.bid_history.append(market_data.bid)
        self.ask_history.append(market_data.ask)
        self.timestamp_history.append(market_data.timestamp)
        
        # Update key levels
        self._update_key_levels()
        
        # Clean expired manipulations
        self._clean_expired_manipulations(market_data.timestamp)
    
    def _update_key_levels(self) -> None:
        """Update key support/resistance levels"""
        
        if len(self.price_history) < 20:
            return
        
        prices = np.array(list(self.price_history))
        
        # Find swing highs and lows
        highs, _ = find_peaks(prices, distance=5)
        lows, _ = find_peaks(-prices, distance=5)
        
        # Extract significant levels
        new_levels = []
        
        for idx in highs:
            if idx < len(prices):
                new_levels.append(prices[idx])
        
        for idx in lows:
            if idx < len(prices):
                new_levels.append(prices[idx])
        
        # Cluster nearby levels
        if new_levels:
            new_levels = sorted(new_levels)
            clustered_levels = []
            current_cluster = [new_levels[0]]
            
            for level in new_levels[1:]:
                if abs(level - current_cluster[-1]) < 0.001:  # Within 10 pips
                    current_cluster.append(level)
                else:
                    clustered_levels.append(np.mean(current_cluster))
                    current_cluster = [level]
            
            clustered_levels.append(np.mean(current_cluster))
            
            # Update key levels (keep most recent)
            self.key_levels = clustered_levels[-10:]  # Keep top 10 levels
    
    def _clean_expired_manipulations(self, current_time: datetime) -> None:
        """Remove expired manipulation events"""
        
        cutoff_time = current_time - timedelta(hours=2)
        self.active_manipulations = [
            manip for manip in self.active_manipulations
            if manip.timestamp > cutoff_time
        ]
    
    def detect_manipulation_patterns(self, market_data: MarketData) -> List[ManipulationEvent]:
        """Detect various manipulation patterns"""
        
        manipulations = []
        
        # Stop hunting detection
        stop_hunt = self._detect_stop_hunting(market_data)
        if stop_hunt:
            manipulations.append(stop_hunt)
        
        # Spoofing detection
        spoofing = self._detect_spoofing(market_data)
        if spoofing:
            manipulations.append(spoofing)
        
        # Wash trading detection
        wash_trading = self._detect_wash_trading(market_data)
        if wash_trading:
            manipulations.append(wash_trading)
        
        # Momentum ignition detection
        momentum_ignition = self._detect_momentum_ignition(market_data)
        if momentum_ignition:
            manipulations.append(momentum_ignition)
        
        # Update active manipulations
        self.active_manipulations.extend(manipulations)
        
        return manipulations
    
    def _detect_stop_hunting(self, market_data: MarketData) -> Optional[ManipulationEvent]:
        """Detect stop hunting patterns"""
        
        if len(self.price_history) < 10 or not self.key_levels:
            return None
        
        current_price = (market_data.bid + market_data.ask) / 2
        recent_prices = np.array(list(self.price_history)[-10:])
        
        # Check if price has moved beyond key levels and reversed
        for level in self.key_levels:
            distance_to_level = abs(current_price - level)
            
            if distance_to_level < 0.0005:  # Within 5 pips of key level
                self.level_test_count[level] += 1
                
                # Check for quick reversal after breaking level
                if len(recent_prices) >= 5:
                    # Check if price broke above level and reversed down
                    if (np.max(recent_prices[-5:]) > level and 
                        current_price < level and
                        np.max(recent_prices[-5:]) - level > 0.0003):  # Broke by at least 3 pips
                        
                        return ManipulationEvent(
                            signature=ManipulationSignature.STOP_HUNTING,
                            timestamp=market_data.timestamp,
                            strength=min(self.level_test_count[level] / 3.0, 1.0),
                            target_level=level,
                            volume_signature=self._calculate_volume_signature(),
                            success_probability=0.7,
                            counter_strategy="fade_the_break"
                        )
                    
                    # Check if price broke below level and reversed up
                    elif (np.min(recent_prices[-5:]) < level and 
                          current_price > level and
                          level - np.min(recent_prices[-5:]) > 0.0003):
                        
                        return ManipulationEvent(
                            signature=ManipulationSignature.STOP_HUNTING,
                            timestamp=market_data.timestamp,
                            strength=min(self.level_test_count[level] / 3.0, 1.0),
                            target_level=level,
                            volume_signature=self._calculate_volume_signature(),
                            success_probability=0.7,
                            counter_strategy="fade_the_break"
                        )
        
        return None
    
    def _detect_spoofing(self, market_data: MarketData) -> Optional[ManipulationEvent]:
        """Detect spoofing patterns (simplified - would need order book data)"""
        
        if len(self.bid_history) < 10:
            return None
        
        # Analyze bid-ask spread patterns
        recent_spreads = []
        for i in range(-10, 0):
            if abs(i) <= len(self.bid_history):
                spread = list(self.ask_history)[i] - list(self.bid_history)[i]
                recent_spreads.append(spread)
        
        if len(recent_spreads) < 5:
            return None
        
        current_spread = market_data.ask - market_data.bid
        avg_spread = np.mean(recent_spreads)
        
        # Detect unusual spread compression followed by expansion
        if (current_spread > avg_spread * 2.0 and  # Current spread is wide
            min(recent_spreads[-3:]) < avg_spread * 0.5):  # Recent spreads were tight
            
            return ManipulationEvent(
                signature=ManipulationSignature.SPOOFING,
                timestamp=market_data.timestamp,
                strength=min(current_spread / avg_spread - 1.0, 1.0),
                target_level=None,
                volume_signature=self._calculate_volume_signature(),
                success_probability=0.5,
                counter_strategy="wait_for_normalization"
            )
        
        return None
    
    def _detect_wash_trading(self, market_data: MarketData) -> Optional[ManipulationEvent]:
        """Detect wash trading patterns"""
        
        if len(self.volume_history) < 20:
            return None
        
        volumes = np.array(list(self.volume_history))
        prices = np.array(list(self.price_history))
        
        # Look for high volume with minimal price movement
        recent_volumes = volumes[-10:]
        recent_prices = prices[-10:]
        
        avg_volume = np.mean(volumes[:-10]) if len(volumes) > 10 else np.mean(volumes)
        price_range = np.max(recent_prices) - np.min(recent_prices)
        volume_spike = np.max(recent_volumes) / avg_volume if avg_volume > 0 else 1
        
        # High volume, low price movement = potential wash trading
        if volume_spike > 3.0 and price_range < 0.0005:  # 3x volume, <5 pip range
            
            return ManipulationEvent(
                signature=ManipulationSignature.WASH_TRADING,
                timestamp=market_data.timestamp,
                strength=min(volume_spike / 5.0, 1.0),
                target_level=None,
                volume_signature=volume_spike,
                success_probability=0.6,
                counter_strategy="ignore_volume_signals"
            )
        
        return None
    
    def _detect_momentum_ignition(self, market_data: MarketData) -> Optional[ManipulationEvent]:
        """Detect momentum ignition patterns"""
        
        if len(self.price_history) < 15:
            return None
        
        prices = np.array(list(self.price_history))
        volumes = np.array(list(self.volume_history))
        
        # Look for sudden price acceleration with volume spike
        recent_prices = prices[-10:]
        recent_volumes = volumes[-10:]
        
        # Calculate price acceleration
        if len(recent_prices) >= 5:
            early_momentum = (recent_prices[-5] - recent_prices[-10]) / recent_prices[-10]
            late_momentum = (recent_prices[-1] - recent_prices[-5]) / recent_prices[-5]
            
            acceleration = late_momentum - early_momentum
            
            # Volume confirmation
            early_volume = np.mean(recent_volumes[:5])
            late_volume = np.mean(recent_volumes[5:])
            volume_acceleration = late_volume / early_volume if early_volume > 0 else 1
            
            # Momentum ignition: accelerating price + accelerating volume
            if acceleration > 0.001 and volume_acceleration > 2.0:  # 10 pip acceleration + 2x volume
                
                return ManipulationEvent(
                    signature=ManipulationSignature.MOMENTUM_IGNITION,
                    timestamp=market_data.timestamp,
                    strength=min(acceleration * 1000 + volume_acceleration / 5, 1.0),
                    target_level=None,
                    volume_signature=volume_acceleration,
                    success_probability=0.4,  # Often fails
                    counter_strategy="fade_after_exhaustion"
                )
        
        return None
    
    def _calculate_volume_signature(self) -> float:
        """Calculate volume signature strength"""
        
        if len(self.volume_history) < 10:
            return 0.5
        
        volumes = np.array(list(self.volume_history))
        recent_volume = np.mean(volumes[-5:])
        historical_volume = np.mean(volumes[:-5])
        
        return min(recent_volume / historical_volume, 3.0) / 3.0 if historical_volume > 0 else 0.5

class ChaosDetector:
    """
    Advanced chaos and emergence detection
    """
    
    def __init__(self, lookback_periods: int = 200):
        self.lookback_periods = lookback_periods
        
        # Data storage
        self.price_history = deque(maxlen=lookback_periods)
        self.timestamp_history = deque(maxlen=lookback_periods)
        
        # Chaos metrics
        self.lyapunov_exponent = 0.0
        self.correlation_dimension = 0.0
        self.hurst_exponent = 0.5
        
        # Emergence tracking
        self.pattern_memory = deque(maxlen=50)
        self.emergence_events: List[ChaosEvent] = []
        
    def update_data(self, market_data: MarketData) -> None:
        """Update with new market data"""
        
        mid_price = (market_data.bid + market_data.ask) / 2
        
        self.price_history.append(mid_price)
        self.timestamp_history.append(market_data.timestamp)
        
        # Update chaos metrics periodically
        if len(self.price_history) >= 50 and len(self.price_history) % 10 == 0:
            self._update_chaos_metrics()
        
        # Clean old emergence events
        cutoff_time = market_data.timestamp - timedelta(hours=4)
        self.emergence_events = [
            event for event in self.emergence_events
            if event.timestamp > cutoff_time
        ]
    
    def _update_chaos_metrics(self) -> None:
        """Update chaos theory metrics"""
        
        if len(self.price_history) < 50:
            return
        
        prices = np.array(list(self.price_history))
        
        # Calculate Hurst exponent (simplified R/S analysis)
        self.hurst_exponent = self._calculate_hurst_exponent(prices)
        
        # Calculate approximate Lyapunov exponent
        self.lyapunov_exponent = self._calculate_lyapunov_exponent(prices)
        
        # Calculate correlation dimension (simplified)
        self.correlation_dimension = self._calculate_correlation_dimension(prices)
    
    def _calculate_hurst_exponent(self, prices: np.ndarray) -> float:
        """Calculate Hurst exponent using R/S analysis"""
        
        if len(prices) < 20:
            return 0.5
        
        try:
            # Calculate log returns
            returns = np.diff(np.log(prices))
            
            # R/S analysis for different time scales
            scales = [5, 10, 20, 30]
            rs_values = []
            
            for scale in scales:
                if scale < len(returns):
                    # Divide into non-overlapping windows
                    n_windows = len(returns) // scale
                    rs_window = []
                    
                    for i in range(n_windows):
                        window = returns[i*scale:(i+1)*scale]
                        
                        # Calculate mean
                        mean_return = np.mean(window)
                        
                        # Calculate cumulative deviations
                        deviations = np.cumsum(window - mean_return)
                        
                        # Calculate range
                        R = np.max(deviations) - np.min(deviations)
                        
                        # Calculate standard deviation
                        S = np.std(window)
                        
                        if S > 0:
                            rs_window.append(R / S)
                    
                    if rs_window:
                        rs_values.append(np.mean(rs_window))
            
            if len(rs_values) >= 2:
                # Fit log(R/S) vs log(n) to get Hurst exponent
                log_scales = np.log(scales[:len(rs_values)])
                log_rs = np.log(rs_values)
                
                # Linear regression
                slope, _ = np.polyfit(log_scales, log_rs, 1)
                return max(0.0, min(slope, 1.0))
        
        except Exception as e:
            logger.debug(f"Hurst exponent calculation failed: {e}")
            # Return neutral value indicating no clear trend persistence
        
        return 0.5
    
    def _calculate_lyapunov_exponent(self, prices: np.ndarray) -> float:
        """Calculate approximate Lyapunov exponent"""
        
        if len(prices) < 30:
            return 0.0
        
        try:
            # Use price returns
            returns = np.diff(prices) / prices[:-1]
            
            # Calculate divergence of nearby trajectories (simplified)
            divergences = []
            
            for i in range(len(returns) - 10):
                # Find nearby points
                current_return = returns[i]
                
                # Look for similar returns in recent history
                for j in range(max(0, i-20), i):
                    if abs(returns[j] - current_return) < 0.0001:  # Similar initial conditions
                        # Calculate divergence after 10 steps
                        if i + 10 < len(returns) and j + 10 < len(returns):
                            initial_diff = abs(returns[j] - current_return)
                            final_diff = abs(returns[j+10] - returns[i+10])
                            
                            if initial_diff > 0 and final_diff > 0:
                                divergence = np.log(final_diff / initial_diff) / 10
                                divergences.append(divergence)
            
            if divergences:
                return np.mean(divergences)
        
        except Exception as e:
            logger.debug(f"Lyapunov exponent calculation failed: {e}")
            # Return zero indicating no detected chaos
        
        return 0.0
    
    def _calculate_correlation_dimension(self, prices: np.ndarray) -> float:
        """Calculate correlation dimension (simplified)"""
        
        if len(prices) < 50:
            return 1.0
        
        try:
            # Embed the time series in higher dimensions
            embedding_dim = 3
            delay = 1
            
            # Create embedded vectors
            embedded = []
            for i in range(len(prices) - (embedding_dim - 1) * delay):
                vector = [prices[i + j * delay] for j in range(embedding_dim)]
                embedded.append(vector)
            
            embedded = np.array(embedded)
            
            if len(embedded) < 10:
                return 1.0
            
            # Calculate correlation integral for different radii
            radii = np.logspace(-4, -1, 10)  # Different distance scales
            correlations = []
            
            for radius in radii:
                count = 0
                total_pairs = 0
                
                # Count pairs within radius (sample to avoid O(n²) complexity)
                sample_size = min(len(embedded), 100)
                indices = np.random.choice(len(embedded), sample_size, replace=False)
                
                for i in range(len(indices)):
                    for j in range(i + 1, len(indices)):
                        distance = np.linalg.norm(embedded[indices[i]] - embedded[indices[j]])
                        total_pairs += 1
                        if distance < radius:
                            count += 1
                
                if total_pairs > 0:
                    correlation = count / total_pairs
                    correlations.append(correlation)
            
            # Estimate dimension from slope of log(C) vs log(r)
            if len(correlations) >= 3:
                log_radii = np.log(radii[:len(correlations)])
                log_correlations = np.log(np.maximum(correlations, 1e-10))
                
                # Linear regression
                slope, _ = np.polyfit(log_radii, log_correlations, 1)
                return max(1.0, min(slope, 5.0))
        
        except:
            pass
        
        return 2.0
    
    def detect_chaos_patterns(self, market_data: MarketData) -> List[ChaosEvent]:
        """Detect chaotic behavior patterns"""
        
        chaos_events = []
        
        # Strange attractor detection
        strange_attractor = self._detect_strange_attractor(market_data)
        if strange_attractor:
            chaos_events.append(strange_attractor)
        
        # Phase transition detection
        phase_transition = self._detect_phase_transition(market_data)
        if phase_transition:
            chaos_events.append(phase_transition)
        
        # Emergence detection
        emergence = self._detect_emergence(market_data)
        if emergence:
            chaos_events.append(emergence)
        
        # Fractal breakdown detection
        fractal_breakdown = self._detect_fractal_breakdown(market_data)
        if fractal_breakdown:
            chaos_events.append(fractal_breakdown)
        
        # Update emergence events
        self.emergence_events.extend(chaos_events)
        
        return chaos_events
    
    def _detect_strange_attractor(self, market_data: MarketData) -> Optional[ChaosEvent]:
        """Detect strange attractor behavior"""
        
        if len(self.price_history) < 50:
            return None
        
        # Check if price is being attracted to unusual levels
        current_price = (market_data.bid + market_data.ask) / 2
        prices = np.array(list(self.price_history))
        
        # Look for price clustering around non-obvious levels
        price_clusters = []
        for i in range(len(prices) - 10):
            window = prices[i:i+10]
            if np.std(window) < 0.0002:  # Very tight clustering
                price_clusters.append(np.mean(window))
        
        if len(price_clusters) >= 3:
            # Check if current price is near a cluster
            for cluster_price in price_clusters:
                if abs(current_price - cluster_price) < 0.0003:
                    
                    return ChaosEvent(
                        pattern=ChaosPattern.STRANGE_ATTRACTOR,
                        timestamp=market_data.timestamp,
                        intensity=min(len(price_clusters) / 5.0, 1.0),
                        predictability=0.3,  # Strange attractors are hard to predict
                        emergence_rate=0.2,
                        system_stress=0.4
                    )
        
        return None
    
    def _detect_phase_transition(self, market_data: MarketData) -> Optional[ChaosEvent]:
        """Detect sudden phase transitions"""
        
        if len(self.price_history) < 30:
            return None
        
        # Look for sudden changes in market behavior
        prices = np.array(list(self.price_history))
        
        # Compare recent behavior to historical
        recent_volatility = np.std(prices[-10:])
        historical_volatility = np.std(prices[:-10])
        
        recent_autocorr = self._calculate_autocorrelation(prices[-20:])
        historical_autocorr = self._calculate_autocorrelation(prices[:-20])
        
        # Detect sudden changes
        volatility_change = abs(recent_volatility - historical_volatility) / historical_volatility if historical_volatility > 0 else 0
        autocorr_change = abs(recent_autocorr - historical_autocorr)
        
        if volatility_change > 1.0 or autocorr_change > 0.5:  # Significant behavior change
            
            return ChaosEvent(
                pattern=ChaosPattern.PHASE_TRANSITION,
                timestamp=market_data.timestamp,
                intensity=min(max(volatility_change, autocorr_change), 1.0),
                predictability=0.1,  # Phase transitions are highly unpredictable
                emergence_rate=0.8,  # High emergence during transitions
                system_stress=min(volatility_change + autocorr_change, 1.0)
            )
        
        return None
    
    def _detect_emergence(self, market_data: MarketData) -> Optional[ChaosEvent]:
        """Detect emergence of new patterns"""
        
        if len(self.price_history) < 40:
            return None
        
        # Analyze recent price patterns
        prices = np.array(list(self.price_history))
        recent_pattern = self._extract_pattern(prices[-20:])
        
        # Compare to historical patterns
        pattern_similarity = 0.0
        for historical_pattern in self.pattern_memory:
            similarity = self._calculate_pattern_similarity(recent_pattern, historical_pattern)
            pattern_similarity = max(pattern_similarity, similarity)
        
        # Store current pattern
        self.pattern_memory.append(recent_pattern)
        
        # Low similarity = new pattern emerging
        if pattern_similarity < 0.3:
            
            return ChaosEvent(
                pattern=ChaosPattern.EMERGENCE,
                timestamp=market_data.timestamp,
                intensity=1.0 - pattern_similarity,
                predictability=pattern_similarity,  # New patterns are less predictable
                emergence_rate=1.0 - pattern_similarity,
                system_stress=0.5
            )
        
        return None
    
    def _detect_fractal_breakdown(self, market_data: MarketData) -> Optional[ChaosEvent]:
        """Detect breakdown of fractal self-similarity"""
        
        if len(self.price_history) < 60:
            return None
        
        # Check Hurst exponent for fractal breakdown
        if abs(self.hurst_exponent - 0.5) < 0.1:  # Close to random walk
            
            return ChaosEvent(
                pattern=ChaosPattern.FRACTAL_BREAKDOWN,
                timestamp=market_data.timestamp,
                intensity=0.5 - abs(self.hurst_exponent - 0.5),
                predictability=abs(self.hurst_exponent - 0.5) * 2,
                emergence_rate=0.3,
                system_stress=0.6
            )
        
        return None
    
    def _calculate_autocorrelation(self, prices: np.ndarray) -> float:
        """Calculate autocorrelation of price series"""
        
        if len(prices) < 10:
            return 0.0
        
        try:
            returns = np.diff(prices) / prices[:-1]
            
            if len(returns) < 2:
                return 0.0
            
            # Lag-1 autocorrelation
            correlation = np.corrcoef(returns[:-1], returns[1:])[0, 1]
            return correlation if not np.isnan(correlation) else 0.0
        
        except:
            return 0.0
    
    def _extract_pattern(self, prices: np.ndarray) -> np.ndarray:
        """Extract pattern features from price series"""
        
        if len(prices) < 5:
            return np.array([])
        
        # Normalize prices
        normalized = (prices - np.mean(prices)) / np.std(prices) if np.std(prices) > 0 else prices
        
        # Extract features
        features = [
            np.mean(normalized),
            np.std(normalized),
            np.max(normalized) - np.min(normalized),
            len(find_peaks(normalized)[0]) / len(normalized),  # Peak density
            self._calculate_autocorrelation(normalized),
        ]
        
        return np.array(features)
    
    def _calculate_pattern_similarity(self, pattern1: np.ndarray, pattern2: np.ndarray) -> float:
        """Calculate similarity between two patterns"""
        
        if len(pattern1) == 0 or len(pattern2) == 0:
            return 0.0
        
        try:
            # Euclidean distance normalized
            distance = np.linalg.norm(pattern1 - pattern2)
            max_distance = np.linalg.norm(pattern1) + np.linalg.norm(pattern2)
            
            if max_distance > 0:
                similarity = 1.0 - (distance / max_distance)
                return max(0.0, similarity)
        
        except:
            pass
        
        return 0.0

class SelfRefutationEngine:
    """
    Self-refutation and meta-learning engine
    """
    
    def __init__(self):
        # Prediction tracking
        self.predictions: List[Dict] = []
        self.refutation_events: List[SelfRefutationEvent] = []
        
        # Learning metrics
        self.prediction_accuracy = 0.5
        self.confidence_calibration = 0.5
        self.overconfidence_bias = 0.0
        
    def record_prediction(self, prediction_type: str, predicted_value: float, 
                         confidence: float, timestamp: datetime) -> None:
        """Record a prediction for later validation"""
        
        self.predictions.append({
            'type': prediction_type,
            'predicted_value': predicted_value,
            'confidence': confidence,
            'timestamp': timestamp,
            'validated': False
        })
        
        # Keep only recent predictions
        cutoff_time = timestamp - timedelta(hours=24)
        self.predictions = [
            pred for pred in self.predictions
            if pred['timestamp'] > cutoff_time
        ]
    
    def validate_predictions(self, market_data: MarketData) -> List[SelfRefutationEvent]:
        """Validate previous predictions and detect refutations"""
        
        refutations = []
        current_time = market_data.timestamp
        
        for prediction in self.predictions:
            if not prediction['validated']:
                # Check if enough time has passed for validation
                time_elapsed = current_time - prediction['timestamp']
                
                if time_elapsed > timedelta(minutes=30):  # 30 minutes for validation
                    actual_value = self._get_actual_value(prediction['type'], market_data)
                    
                    if actual_value is not None:
                        error_magnitude = abs(actual_value - prediction['predicted_value'])
                        relative_error = error_magnitude / abs(prediction['predicted_value']) if prediction['predicted_value'] != 0 else error_magnitude
                        
                        # Check for significant prediction failure
                        if relative_error > 0.5 and prediction['confidence'] > 0.7:  # High confidence, high error
                            
                            refutation = SelfRefutationEvent(
                                prediction_type=prediction['type'],
                                predicted_value=prediction['predicted_value'],
                                actual_value=actual_value,
                                error_magnitude=error_magnitude,
                                confidence_was=prediction['confidence'],
                                learning_opportunity=self._generate_learning_opportunity(prediction, actual_value)
                            )
                            
                            refutations.append(refutation)
                            self.refutation_events.append(refutation)
                        
                        # Update learning metrics
                        self._update_learning_metrics(prediction, actual_value)
                        prediction['validated'] = True
        
        return refutations
    
    def _get_actual_value(self, prediction_type: str, market_data: MarketData) -> Optional[float]:
        """Get actual value for comparison with prediction"""
        
        if prediction_type == 'price':
            return (market_data.bid + market_data.ask) / 2
        elif prediction_type == 'volatility':
            return market_data.volatility
        elif prediction_type == 'volume':
            return market_data.volume
        
        return None
    
    def _update_learning_metrics(self, prediction: Dict, actual_value: float) -> None:
        """Update learning metrics based on prediction accuracy"""
        
        error = abs(actual_value - prediction['predicted_value'])
        relative_error = error / abs(prediction['predicted_value']) if prediction['predicted_value'] != 0 else error
        
        # Update accuracy (exponential moving average)
        accuracy = 1.0 - min(relative_error, 1.0)
        self.prediction_accuracy = 0.9 * self.prediction_accuracy + 0.1 * accuracy
        
        # Update confidence calibration
        confidence_error = abs(prediction['confidence'] - accuracy)
        calibration = 1.0 - confidence_error
        self.confidence_calibration = 0.9 * self.confidence_calibration + 0.1 * calibration
        
        # Update overconfidence bias
        overconfidence = prediction['confidence'] - accuracy
        self.overconfidence_bias = 0.9 * self.overconfidence_bias + 0.1 * overconfidence
    
    def _generate_learning_opportunity(self, prediction: Dict, actual_value: float) -> str:
        """Generate learning opportunity description"""
        
        error_direction = "overestimated" if prediction['predicted_value'] > actual_value else "underestimated"
        error_magnitude = abs(actual_value - prediction['predicted_value'])
        
        return f"Model {error_direction} {prediction['type']} by {error_magnitude:.4f} with {prediction['confidence']:.2f} confidence"
    
    def get_meta_learning_insights(self) -> Dict[str, Any]:
        """Get meta-learning insights about model performance"""
        
        return {
            'prediction_accuracy': self.prediction_accuracy,
            'confidence_calibration': self.confidence_calibration,
            'overconfidence_bias': self.overconfidence_bias,
            'recent_refutations': len(self.refutation_events),
            'learning_rate': max(0.1, 1.0 - self.prediction_accuracy),  # Learn faster when less accurate
            'reliability_score': (self.prediction_accuracy + self.confidence_calibration) / 2
        }

class AnomalyIntelligenceEngine:
    """
    Main engine for anomaly detection and chaos intelligence
    Orchestrates all anomaly detection components
    """
    
    def __init__(self):
        self.statistical_detector = StatisticalAnomalyDetector()
        self.manipulation_detector = ManipulationDetector()
        self.chaos_detector = ChaosDetector()
        self.self_refutation_engine = SelfRefutationEngine()
        
        # Current state
        self.current_anomaly_level = 0.0
        self.system_stress_level = 0.0
        self.antifragility_score = 0.5
        
        # Anomaly history
        self.recent_anomalies: List[AnomalyEvent] = []
        
    async def analyze_anomaly_intelligence(self, market_data: MarketData) -> DimensionalReading:
        """Analyze anomaly intelligence and return dimensional reading"""
        
        # Update all detectors
        self.statistical_detector.update_data(market_data)
        self.manipulation_detector.update_data(market_data)
        self.chaos_detector.update_data(market_data)
        
        # Detect anomalies
        statistical_anomalies = self.statistical_detector.detect_statistical_anomalies(market_data)
        manipulation_events = self.manipulation_detector.detect_manipulation_patterns(market_data)
        chaos_events = self.chaos_detector.detect_chaos_patterns(market_data)
        refutation_events = self.self_refutation_engine.validate_predictions(market_data)
        
        # Combine all anomalies
        all_anomalies = statistical_anomalies.copy()
        
        # Convert manipulation events to anomaly events
        for manip in manipulation_events:
            anomaly = AnomalyEvent(
                anomaly_type=AnomalyType.MANIPULATION_PATTERN,
                timestamp=manip.timestamp,
                severity=manip.strength,
                confidence=manip.success_probability,
                description=f"Manipulation: {manip.signature.name}",
                affected_dimensions=['HOW'],
                market_impact=manip.strength * 0.6
            )
            all_anomalies.append(anomaly)
        
        # Convert chaos events to anomaly events
        for chaos in chaos_events:
            anomaly = AnomalyEvent(
                anomaly_type=AnomalyType.CHAOS_EMERGENCE,
                timestamp=chaos.timestamp,
                severity=chaos.intensity,
                confidence=1.0 - chaos.predictability,
                description=f"Chaos: {chaos.pattern.name}",
                affected_dimensions=['WHAT', 'WHEN'],
                market_impact=chaos.system_stress
            )
            all_anomalies.append(anomaly)
        
        # Convert refutation events to anomaly events
        for refutation in refutation_events:
            anomaly = AnomalyEvent(
                anomaly_type=AnomalyType.SELF_REFUTATION,
                timestamp=market_data.timestamp,
                severity=min(refutation.error_magnitude * 10, 1.0),
                confidence=refutation.confidence_was,
                description=f"Self-refutation: {refutation.learning_opportunity}",
                affected_dimensions=['WHY', 'WHAT', 'HOW', 'WHEN'],
                market_impact=0.2
            )
            all_anomalies.append(anomaly)
        
        # Update recent anomalies
        self.recent_anomalies.extend(all_anomalies)
        cutoff_time = market_data.timestamp - timedelta(hours=2)
        self.recent_anomalies = [
            anomaly for anomaly in self.recent_anomalies
            if anomaly.timestamp > cutoff_time
        ]
        
        # Calculate anomaly strength
        anomaly_strength = self._calculate_anomaly_strength()
        
        # Update system stress and antifragility
        self._update_system_metrics()
        
        # Calculate confidence
        confidence = self._calculate_anomaly_confidence()
        
        # Generate context
        context = self._generate_anomaly_context()
        
        return DimensionalReading(
            dimension='ANOMALY',
            value=anomaly_strength,
            confidence=confidence,
            context=context,
            timestamp=market_data.timestamp
        )
    
    def _calculate_anomaly_strength(self) -> float:
        """Calculate overall anomaly strength"""
        
        if not self.recent_anomalies:
            return 0.0
        
        # Weight anomalies by recency and severity
        total_weight = 0.0
        weighted_severity = 0.0
        
        current_time = datetime.now()
        
        for anomaly in self.recent_anomalies:
            # Time decay factor
            time_elapsed = current_time - anomaly.timestamp
            decay_factor = math.exp(-time_elapsed.total_seconds() / 3600)  # 1-hour half-life
            
            # Weight by severity and confidence
            weight = anomaly.severity * anomaly.confidence * decay_factor
            
            total_weight += weight
            weighted_severity += weight * anomaly.severity
        
        if total_weight > 0:
            self.current_anomaly_level = weighted_severity / total_weight
        else:
            self.current_anomaly_level = 0.0
        
        return self.current_anomaly_level
    
    def _update_system_metrics(self) -> None:
        """Update system stress and antifragility metrics"""
        
        # Calculate system stress from recent anomalies
        if self.recent_anomalies:
            stress_contributions = [
                anomaly.market_impact * anomaly.confidence
                for anomaly in self.recent_anomalies
            ]
            self.system_stress_level = min(np.sum(stress_contributions), 1.0)
        else:
            self.system_stress_level = 0.0
        
        # Calculate antifragility (system gets stronger from stress)
        # High stress with good learning = higher antifragility
        meta_insights = self.self_refutation_engine.get_meta_learning_insights()
        learning_rate = meta_insights['learning_rate']
        
        if self.system_stress_level > 0.3:  # Significant stress
            # Antifragility increases with stress if we're learning
            stress_benefit = self.system_stress_level * learning_rate
            self.antifragility_score = min(self.antifragility_score + stress_benefit * 0.1, 1.0)
        else:
            # Gradual decay without stress
            self.antifragility_score = max(self.antifragility_score * 0.999, 0.3)
    
    def _calculate_anomaly_confidence(self) -> float:
        """Calculate confidence in anomaly detection"""
        
        confidence_factors = []
        
        # Data quality (amount of historical data)
        data_quality = min(len(self.statistical_detector.price_history) / 100, 1.0)
        confidence_factors.append(data_quality * 0.3)
        
        # Model training status
        if self.statistical_detector.model_trained:
            confidence_factors.append(0.3)
        
        # Meta-learning insights
        meta_insights = self.self_refutation_engine.get_meta_learning_insights()
        confidence_factors.append(meta_insights['confidence_calibration'] * 0.2)
        
        # Chaos metrics reliability
        if len(self.chaos_detector.price_history) >= 50:
            chaos_confidence = min(len(self.chaos_detector.price_history) / 200, 1.0)
            confidence_factors.append(chaos_confidence * 0.2)
        
        return np.sum(confidence_factors) if confidence_factors else 0.5
    
    def _generate_anomaly_context(self) -> Dict[str, Any]:
        """Generate contextual information about anomaly analysis"""
        
        context = {
            'current_anomaly_level': self.current_anomaly_level,
            'system_stress_level': self.system_stress_level,
            'antifragility_score': self.antifragility_score,
            'recent_anomalies_count': len(self.recent_anomalies)
        }
        
        # Add anomaly breakdown by type
        anomaly_types = defaultdict(int)
        for anomaly in self.recent_anomalies:
            anomaly_types[anomaly.anomaly_type.name] += 1
        
        context['anomaly_breakdown'] = dict(anomaly_types)
        
        # Add chaos metrics
        context['chaos_metrics'] = {
            'hurst_exponent': self.chaos_detector.hurst_exponent,
            'lyapunov_exponent': self.chaos_detector.lyapunov_exponent,
            'correlation_dimension': self.chaos_detector.correlation_dimension
        }
        
        # Add manipulation status
        active_manipulations = [
            {
                'signature': manip.signature.name,
                'strength': manip.strength,
                'target_level': manip.target_level
            }
            for manip in self.manipulation_detector.active_manipulations
        ]
        context['active_manipulations'] = active_manipulations
        
        # Add meta-learning insights
        context['meta_learning'] = self.self_refutation_engine.get_meta_learning_insights()
        
        # Add recent significant anomalies
        significant_anomalies = [
            {
                'type': anomaly.anomaly_type.name,
                'severity': anomaly.severity,
                'description': anomaly.description,
                'affected_dimensions': anomaly.affected_dimensions
            }
            for anomaly in self.recent_anomalies
            if anomaly.severity > 0.5
        ]
        context['significant_anomalies'] = significant_anomalies[-5:]  # Last 5
        
        return context

# Example usage
async def main():
    """Example usage of the enhanced anomaly intelligence engine"""
    
    # Initialize engine
    engine = AnomalyIntelligenceEngine()
    
    # Simulate market data with various anomalies
    base_price = 1.0950
    
    for i in range(300):
        
        current_time = datetime.now() + timedelta(minutes=i)
        
        # Create market data with occasional anomalies
        if i == 100:  # Inject price spike anomaly
            price_change = 0.005  # 50 pip spike
            volume_multiplier = 5.0
        elif i == 150:  # Inject manipulation pattern
            price_change = -0.003  # 30 pip drop
            volume_multiplier = 0.3  # Low volume
        elif i == 200:  # Inject chaos
            price_change = np.random.normal(0, 0.002)  # High volatility
            volume_multiplier = np.random.exponential(2.0)
        else:  # Normal market behavior
            price_change = np.random.normal(0, 0.0005)
            volume_multiplier = 1.0 + np.random.normal(0, 0.3)
        
        current_price = base_price + price_change
        volume = max(1000 * volume_multiplier, 100)
        volatility = abs(price_change) + np.random.exponential(0.002)
        
        market_data = MarketData(
            timestamp=current_time,
            bid=current_price - 0.0001,
            ask=current_price + 0.0001,
            volume=volume,
            volatility=volatility
        )
        
        # Record some predictions for self-refutation testing
        if i % 50 == 0:
            predicted_price = current_price + np.random.normal(0, 0.001)
            engine.self_refutation_engine.record_prediction(
                'price', predicted_price, 0.8, current_time
            )
        
        # Analyze anomaly intelligence
        reading = await engine.analyze_anomaly_intelligence(market_data)
        
        if i % 50 == 0 or reading.value > 0.3:  # Print every 50th or when anomalies detected
            print(f"Anomaly Intelligence Reading (Period {i}):")
            print(f"  Value: {reading.value:.3f}")
            print(f"  Confidence: {reading.confidence:.3f}")
            print(f"  System Stress: {reading.context.get('system_stress_level', 0):.3f}")
            print(f"  Antifragility: {reading.context.get('antifragility_score', 0):.3f}")
            
            if reading.context.get('recent_anomalies_count', 0) > 0:
                print(f"  Recent Anomalies: {reading.context['recent_anomalies_count']}")
                
                if 'anomaly_breakdown' in reading.context:
                    breakdown = reading.context['anomaly_breakdown']
                    for anomaly_type, count in breakdown.items():
                        print(f"    {anomaly_type}: {count}")
            
            if reading.context.get('active_manipulations'):
                print(f"  Active Manipulations: {len(reading.context['active_manipulations'])}")
            
            print()

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())

