"""
Sensory Cortex v2.2 - ANOMALY Dimension Engine (Manipulation Detection)

Masterful implementation of market manipulation and anomaly detection.
Uses Isolation Forest, statistical analysis, and behavioral pattern recognition.
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, NamedTuple
from dataclasses import dataclass
from enum import Enum
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

from src.sensory.core.base import (
    DimensionalSensor, DimensionalReading, MarketData, InstrumentMeta,
    MarketRegime, OrderBookSnapshot
)
from src.sensory.core.utils import (
    EMA, WelfordVar, compute_confidence, normalize_signal,
    calculate_momentum, PerformanceTracker
)

logger = logging.getLogger(__name__)


class AnomalyType(Enum):
    """Types of market anomalies."""
    PRICE_SPIKE = "price_spike"
    VOLUME_ANOMALY = "volume_anomaly"
    SPREAD_MANIPULATION = "spread_manipulation"
    ORDER_BOOK_IMBALANCE = "order_book_imbalance"
    MOMENTUM_DIVERGENCE = "momentum_divergence"
    STATISTICAL_OUTLIER = "statistical_outlier"
    BEHAVIORAL_ANOMALY = "behavioral_anomaly"


class ManipulationPattern(Enum):
    """Known manipulation patterns."""
    PUMP_AND_DUMP = "pump_and_dump"
    BEAR_RAID = "bear_raid"
    SPOOFING = "spoofing"
    LAYERING = "layering"
    WASH_TRADING = "wash_trading"
    STOP_HUNTING = "stop_hunting"


@dataclass
class AnomalyEvent:
    """
    Anomaly event structure.
    """
    timestamp: datetime
    anomaly_type: AnomalyType
    severity: float  # 0-1
    confidence: float  # 0-1
    affected_metrics: List[str]
    description: str
    manipulation_pattern: Optional[ManipulationPattern] = None
    duration_seconds: Optional[int] = None


@dataclass
class StatisticalProfile:
    """
    Statistical profile for anomaly detection.
    """
    mean: float
    std: float
    skewness: float
    kurtosis: float
    percentile_95: float
    percentile_99: float
    z_score_threshold: float = 3.0


class IsolationForestDetector:
    """
    Isolation Forest-based anomaly detection.
    """
    
    def __init__(self, contamination: float = 0.1, n_estimators: int = 100):
        """
        Initialize Isolation Forest detector.
        
        Args:
            contamination: Expected proportion of anomalies
            n_estimators: Number of trees in the forest
        """
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=42,
            n_jobs=-1
        )
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.feature_history: List[np.ndarray] = []
        self.min_samples = 50
        
    def update(self, features: np.ndarray) -> Tuple[bool, float]:
        """
        Update detector with new features and detect anomalies.
        
        Args:
            features: Feature vector for current observation
            
        Returns:
            Tuple of (is_anomaly, anomaly_score)
        """
        self.feature_history.append(features)
        
        # Maintain reasonable history size
        if len(self.feature_history) > 1000:
            self.feature_history.pop(0)
        
        # Need minimum samples to fit model
        if len(self.feature_history) < self.min_samples:
            return False, 0.0
        
        # Refit model periodically with recent data
        if len(self.feature_history) % 20 == 0 or not self.is_fitted:
            self._refit_model()
        
        if not self.is_fitted:
            return False, 0.0
        
        # Scale features
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        
        # Get anomaly prediction and score
        prediction = self.model.predict(features_scaled)[0]
        anomaly_score = self.model.decision_function(features_scaled)[0]
        
        # Convert to probability-like score (0-1)
        anomaly_score_normalized = max(0.0, min(1.0, (0.5 - anomaly_score) * 2))
        
        is_anomaly = prediction == -1
        
        return is_anomaly, anomaly_score_normalized
    
    def _refit_model(self) -> None:
        """Refit the Isolation Forest model with recent data."""
        if len(self.feature_history) < self.min_samples:
            return
        
        try:
            # Use recent data for training
            training_data = np.array(self.feature_history[-200:])  # Last 200 observations
            
            # Fit scaler
            self.scaler.fit(training_data)
            training_data_scaled = self.scaler.transform(training_data)
            
            # Fit Isolation Forest
            self.model.fit(training_data_scaled)
            self.is_fitted = True
            
            logger.debug(f"Isolation Forest refitted with {len(training_data)} samples")
            
        except Exception as e:
            logger.warning(f"Failed to refit Isolation Forest: {e}")
            self.is_fitted = False


class StatisticalAnomalyDetector:
    """
    Statistical anomaly detection using Z-scores and distribution analysis.
    """
    
    def __init__(self, window_size: int = 100):
        """
        Initialize statistical detector.
        
        Args:
            window_size: Window size for statistical calculations
        """
        self.window_size = window_size
        self.price_tracker = WelfordVar()
        self.volume_tracker = WelfordVar()
        self.spread_tracker = WelfordVar()
        self.momentum_tracker = WelfordVar()
        
        self.price_history: List[float] = []
        self.volume_history: List[float] = []
        self.spread_history: List[float] = []
        self.momentum_history: List[float] = []
        
    def update(self, market_data: MarketData) -> Dict[str, any]:
        """
        Update statistical profiles and detect anomalies.
        
        Args:
            market_data: Current market data
            
        Returns:
            Statistical anomaly analysis
        """
        # Calculate metrics
        price = market_data.close
        volume = market_data.volume
        spread = market_data.ask - market_data.bid
        
        # Calculate momentum if we have history
        if len(self.price_history) > 0:
            momentum = (price - self.price_history[-1]) / self.price_history[-1]
        else:
            momentum = 0.0
        
        # Update trackers
        self.price_tracker.update(price)
        self.volume_tracker.update(volume)
        self.spread_tracker.update(spread)
        self.momentum_tracker.update(momentum)
        
        # Update histories
        self.price_history.append(price)
        self.volume_history.append(volume)
        self.spread_history.append(spread)
        self.momentum_history.append(momentum)
        
        # Maintain window size
        for history in [self.price_history, self.volume_history, self.spread_history, self.momentum_history]:
            if len(history) > self.window_size:
                history.pop(0)
        
        # Detect anomalies
        anomalies = {}
        
        if len(self.price_history) >= 20:  # Minimum samples for detection
            anomalies['price'] = self._detect_price_anomalies(price)
            anomalies['volume'] = self._detect_volume_anomalies(volume)
            anomalies['spread'] = self._detect_spread_anomalies(spread)
            anomalies['momentum'] = self._detect_momentum_anomalies(momentum)
        
        return anomalies
    
    def _detect_price_anomalies(self, current_price: float) -> Dict[str, any]:
        """Detect price-based anomalies."""
        if self.price_tracker.count < 20:
            return {'is_anomaly': False, 'z_score': 0.0, 'severity': 0.0}
        
        mean, _, std = self.price_tracker.get_stats()
        
        if std == 0:
            return {'is_anomaly': False, 'z_score': 0.0, 'severity': 0.0}
        
        z_score = abs(current_price - mean) / std
        is_anomaly = z_score > 3.0
        severity = min(1.0, z_score / 5.0)  # Normalize to 0-1
        
        return {
            'is_anomaly': is_anomaly,
            'z_score': z_score,
            'severity': severity,
            'type': 'price_spike' if is_anomaly else 'normal'
        }
    
    def _detect_volume_anomalies(self, current_volume: float) -> Dict[str, any]:
        """Detect volume-based anomalies."""
        if self.volume_tracker.count < 20:
            return {'is_anomaly': False, 'z_score': 0.0, 'severity': 0.0}
        
        mean, _, std = self.volume_tracker.get_stats()
        
        if std == 0:
            return {'is_anomaly': False, 'z_score': 0.0, 'severity': 0.0}
        
        z_score = abs(current_volume - mean) / std
        is_anomaly = z_score > 2.5  # Lower threshold for volume
        severity = min(1.0, z_score / 4.0)
        
        return {
            'is_anomaly': is_anomaly,
            'z_score': z_score,
            'severity': severity,
            'type': 'volume_anomaly' if is_anomaly else 'normal'
        }
    
    def _detect_spread_anomalies(self, current_spread: float) -> Dict[str, any]:
        """Detect spread-based anomalies."""
        if self.spread_tracker.count < 20:
            return {'is_anomaly': False, 'z_score': 0.0, 'severity': 0.0}
        
        mean, _, std = self.spread_tracker.get_stats()
        
        if std == 0:
            return {'is_anomaly': False, 'z_score': 0.0, 'severity': 0.0}
        
        z_score = abs(current_spread - mean) / std
        is_anomaly = z_score > 2.0  # Lower threshold for spread
        severity = min(1.0, z_score / 3.0)
        
        return {
            'is_anomaly': is_anomaly,
            'z_score': z_score,
            'severity': severity,
            'type': 'spread_manipulation' if is_anomaly else 'normal'
        }
    
    def _detect_momentum_anomalies(self, current_momentum: float) -> Dict[str, any]:
        """Detect momentum-based anomalies."""
        if self.momentum_tracker.count < 20:
            return {'is_anomaly': False, 'z_score': 0.0, 'severity': 0.0}
        
        mean, _, std = self.momentum_tracker.get_stats()
        
        if std == 0:
            return {'is_anomaly': False, 'z_score': 0.0, 'severity': 0.0}
        
        z_score = abs(current_momentum - mean) / std
        is_anomaly = z_score > 2.5
        severity = min(1.0, z_score / 4.0)
        
        return {
            'is_anomaly': is_anomaly,
            'z_score': z_score,
            'severity': severity,
            'type': 'momentum_divergence' if is_anomaly else 'normal'
        }


class BehavioralPatternDetector:
    """
    Behavioral pattern detection for manipulation identification.
    """
    
    def __init__(self):
        """Initialize behavioral pattern detector."""
        self.price_history: List[MarketData] = []
        self.pattern_memory: Dict[str, List[datetime]] = {}
        
    def update(self, market_data: MarketData) -> Dict[str, any]:
        """
        Update behavioral analysis and detect manipulation patterns.
        
        Args:
            market_data: Current market data
            
        Returns:
            Behavioral pattern analysis
        """
        self.price_history.append(market_data)
        if len(self.price_history) > 200:
            self.price_history.pop(0)
        
        patterns = {}
        
        if len(self.price_history) >= 20:
            patterns['pump_and_dump'] = self._detect_pump_and_dump()
            patterns['bear_raid'] = self._detect_bear_raid()
            patterns['stop_hunting'] = self._detect_stop_hunting()
            patterns['wash_trading'] = self._detect_wash_trading()
        
        return patterns
    
    def _detect_pump_and_dump(self) -> Dict[str, any]:
        """Detect pump and dump patterns."""
        if len(self.price_history) < 50:
            return {'detected': False, 'confidence': 0.0}
        
        recent_data = self.price_history[-50:]
        prices = [d.close for d in recent_data]
        volumes = [d.volume for d in recent_data]
        
        # Look for rapid price increase followed by decline with volume spike
        price_changes = np.diff(prices) / prices[:-1]
        
        # Find potential pump phase (rapid increase)
        pump_threshold = 0.02  # 2% rapid increase
        pump_indices = np.where(price_changes > pump_threshold)[0]
        
        if len(pump_indices) == 0:
            return {'detected': False, 'confidence': 0.0}
        
        # Check for subsequent dump (rapid decline)
        for pump_idx in pump_indices[-5:]:  # Check recent pumps
            if pump_idx < len(price_changes) - 10:  # Need room for dump
                subsequent_changes = price_changes[pump_idx+1:pump_idx+11]
                if np.sum(subsequent_changes) < -0.015:  # 1.5% decline after pump
                    # Check volume spike during pump
                    pump_volume = volumes[pump_idx]
                    avg_volume = np.mean(volumes[:pump_idx]) if pump_idx > 0 else pump_volume
                    
                    if pump_volume > avg_volume * 2:  # Volume spike
                        confidence = min(1.0, (pump_volume / avg_volume - 1) / 3)
                        return {'detected': True, 'confidence': confidence, 'pump_index': pump_idx}
        
        return {'detected': False, 'confidence': 0.0}
    
    def _detect_bear_raid(self) -> Dict[str, any]:
        """Detect bear raid patterns."""
        if len(self.price_history) < 30:
            return {'detected': False, 'confidence': 0.0}
        
        recent_data = self.price_history[-30:]
        prices = [d.close for d in recent_data]
        volumes = [d.volume for d in recent_data]
        
        # Look for coordinated selling pressure with volume spikes
        price_changes = np.diff(prices) / prices[:-1]
        
        # Find consecutive negative moves with increasing volume
        negative_moves = 0
        volume_increase = 0
        
        for i in range(len(price_changes) - 5, len(price_changes)):
            if price_changes[i] < -0.005:  # 0.5% decline
                negative_moves += 1
                if i > 0 and volumes[i] > volumes[i-1]:
                    volume_increase += 1
        
        if negative_moves >= 3 and volume_increase >= 2:
            confidence = min(1.0, (negative_moves + volume_increase) / 8)
            return {'detected': True, 'confidence': confidence}
        
        return {'detected': False, 'confidence': 0.0}
    
    def _detect_stop_hunting(self) -> Dict[str, any]:
        """Detect stop hunting patterns."""
        if len(self.price_history) < 20:
            return {'detected': False, 'confidence': 0.0}
        
        recent_data = self.price_history[-20:]
        
        # Look for sharp moves that quickly reverse
        for i in range(len(recent_data) - 5):
            current = recent_data[i]
            next_few = recent_data[i+1:i+4]
            
            if not next_few:
                continue
            
            # Check for sharp downward move followed by quick recovery
            max_decline = min((d.low - current.close) / current.close for d in next_few)
            final_recovery = (recent_data[i+3].close - current.close) / current.close
            
            if max_decline < -0.01 and final_recovery > -0.002:  # Sharp drop, quick recovery
                confidence = min(1.0, abs(max_decline) * 50)
                return {'detected': True, 'confidence': confidence}
            
            # Check for sharp upward move followed by quick decline
            max_rise = max((d.high - current.close) / current.close for d in next_few)
            final_decline = (recent_data[i+3].close - current.close) / current.close
            
            if max_rise > 0.01 and final_decline < 0.002:  # Sharp rise, quick decline
                confidence = min(1.0, max_rise * 50)
                return {'detected': True, 'confidence': confidence}
        
        return {'detected': False, 'confidence': 0.0}
    
    def _detect_wash_trading(self) -> Dict[str, any]:
        """Detect wash trading patterns."""
        if len(self.price_history) < 30:
            return {'detected': False, 'confidence': 0.0}
        
        recent_data = self.price_history[-30:]
        prices = [d.close for d in recent_data]
        volumes = [d.volume for d in recent_data]
        
        # Look for high volume with minimal price movement
        price_volatility = np.std(prices) / np.mean(prices)
        avg_volume = np.mean(volumes)
        recent_volume = np.mean(volumes[-10:])
        
        # High volume, low volatility suggests wash trading
        if recent_volume > avg_volume * 1.5 and price_volatility < 0.005:
            volume_ratio = recent_volume / avg_volume
            confidence = min(1.0, (volume_ratio - 1) / 2)
            return {'detected': True, 'confidence': confidence}
        
        return {'detected': False, 'confidence': 0.0}


class OrderBookAnomalyDetector:
    """
    Order book anomaly detection for spoofing and layering.
    """
    
    def __init__(self):
        """Initialize order book anomaly detector."""
        self.order_book_history: List[OrderBookSnapshot] = []
        self.imbalance_tracker = EMA(20)
        
    def update(self, order_book: Optional[OrderBookSnapshot]) -> Dict[str, any]:
        """
        Update order book analysis and detect anomalies.
        
        Args:
            order_book: Current order book snapshot
            
        Returns:
            Order book anomaly analysis
        """
        if order_book is None:
            return {'imbalance_anomaly': False, 'spoofing_detected': False}
        
        self.order_book_history.append(order_book)
        if len(self.order_book_history) > 100:
            self.order_book_history.pop(0)
        
        analysis = {}
        
        # Detect order book imbalance anomalies
        analysis['imbalance'] = self._detect_imbalance_anomaly(order_book)
        
        # Detect spoofing patterns
        if len(self.order_book_history) >= 10:
            analysis['spoofing'] = self._detect_spoofing()
            analysis['layering'] = self._detect_layering()
        
        return analysis
    
    def _detect_imbalance_anomaly(self, order_book: OrderBookSnapshot) -> Dict[str, any]:
        """Detect order book imbalance anomalies."""
        bid_volume = sum(level.volume for level in order_book.bids[:5])  # Top 5 levels
        ask_volume = sum(level.volume for level in order_book.asks[:5])
        
        total_volume = bid_volume + ask_volume
        if total_volume == 0:
            return {'detected': False, 'imbalance_ratio': 0.0}
        
        imbalance_ratio = (bid_volume - ask_volume) / total_volume
        self.imbalance_tracker.update(abs(imbalance_ratio))
        
        avg_imbalance = self.imbalance_tracker.get_value() or 0.0
        current_imbalance = abs(imbalance_ratio)
        
        # Anomaly if current imbalance is much higher than average
        is_anomaly = current_imbalance > avg_imbalance * 2.5 and current_imbalance > 0.3
        
        return {
            'detected': is_anomaly,
            'imbalance_ratio': imbalance_ratio,
            'severity': min(1.0, current_imbalance / 0.8)
        }
    
    def _detect_spoofing(self) -> Dict[str, any]:
        """Detect spoofing patterns in order book."""
        if len(self.order_book_history) < 10:
            return {'detected': False, 'confidence': 0.0}
        
        # Look for large orders that appear and disappear quickly
        spoof_events = 0
        
        for i in range(len(self.order_book_history) - 5):
            current_book = self.order_book_history[i]
            future_books = self.order_book_history[i+1:i+5]
            
            # Check for large bid/ask that disappears
            for bid in current_book.bids[:3]:
                if bid.volume > 1000000:  # Large order
                    # Check if it disappears in subsequent books
                    disappeared = True
                    for future_book in future_books:
                        for future_bid in future_book.bids[:5]:
                            if abs(future_bid.price - bid.price) < 0.0001 and future_bid.volume > bid.volume * 0.5:
                                disappeared = False
                                break
                        if not disappeared:
                            break
                    
                    if disappeared:
                        spoof_events += 1
        
        confidence = min(1.0, spoof_events / 3)
        return {'detected': spoof_events > 0, 'confidence': confidence, 'events': spoof_events}
    
    def _detect_layering(self) -> Dict[str, any]:
        """Detect layering patterns in order book."""
        if len(self.order_book_history) < 5:
            return {'detected': False, 'confidence': 0.0}
        
        current_book = self.order_book_history[-1]
        
        # Look for multiple large orders on one side
        bid_layers = sum(1 for bid in current_book.bids[:10] if bid.volume > 500000)
        ask_layers = sum(1 for ask in current_book.asks[:10] if ask.volume > 500000)
        
        # Layering if many large orders on one side
        if bid_layers >= 5 or ask_layers >= 5:
            confidence = min(1.0, max(bid_layers, ask_layers) / 8)
            return {'detected': True, 'confidence': confidence, 'side': 'bid' if bid_layers > ask_layers else 'ask'}
        
        return {'detected': False, 'confidence': 0.0}


class ANOMALYEngine(DimensionalSensor):
    """
    Masterful ANOMALY dimension engine for market manipulation detection.
    Implements Isolation Forest, statistical analysis, and behavioral pattern recognition.
    """
    
    def __init__(self, instrument_meta: InstrumentMeta):
        """
        Initialize ANOMALY engine.
        
        Args:
            instrument_meta: Instrument metadata
        """
        super().__init__(instrument_meta)
        
        # Initialize detectors
        self.isolation_forest = IsolationForestDetector(contamination=0.05)
        self.statistical_detector = StatisticalAnomalyDetector(window_size=100)
        self.behavioral_detector = BehavioralPatternDetector()
        self.order_book_detector = OrderBookAnomalyDetector()
        
        # Performance tracking
        self.performance_tracker = PerformanceTracker()
        
        # State variables
        self.anomaly_history: List[AnomalyEvent] = []
        self.anomaly_score = EMA(20)
        self.manipulation_confidence = EMA(15)
        
        logger.info(f"ANOMALY Engine initialized for {instrument_meta.symbol}")
    
    async def update(self, market_data: MarketData, order_book: Optional[OrderBookSnapshot] = None) -> DimensionalReading:
        """
        Process market data and detect anomalies and manipulation.
        
        Args:
            market_data: Latest market data
            order_book: Optional order book snapshot
            
        Returns:
            Dimensional reading with anomaly analysis
        """
        start_time = datetime.utcnow()
        
        try:
            # Extract features for Isolation Forest
            features = self._extract_features(market_data, order_book)
            
            # Run Isolation Forest detection
            is_anomaly_ml, anomaly_score_ml = self.isolation_forest.update(features)
            
            # Run statistical anomaly detection
            statistical_anomalies = self.statistical_detector.update(market_data)
            
            # Run behavioral pattern detection
            behavioral_patterns = self.behavioral_detector.update(market_data)
            
            # Run order book anomaly detection
            order_book_anomalies = self.order_book_detector.update(order_book)
            
            # Comprehensive anomaly analysis
            anomaly_analysis = self._analyze_anomalies(
                is_anomaly_ml, anomaly_score_ml, statistical_anomalies,
                behavioral_patterns, order_book_anomalies, market_data
            )
            
            # Calculate signal strength and confidence
            signal_strength = self._calculate_signal_strength(anomaly_analysis)
            confidence = self._calculate_confidence(anomaly_analysis)
            
            # Detect market regime
            regime = self._detect_market_regime(anomaly_analysis)
            
            # Create dimensional reading
            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            reading = DimensionalReading(
                dimension="ANOMALY",
                timestamp=market_data.timestamp,
                signal_strength=signal_strength,
                confidence=confidence,
                regime=regime,
                context={
                    'anomaly_analysis': anomaly_analysis,
                    'statistical_anomalies': statistical_anomalies,
                    'behavioral_patterns': behavioral_patterns,
                    'order_book_anomalies': order_book_anomalies,
                    'ml_anomaly_score': anomaly_score_ml,
                    'anomaly_count': len(self.anomaly_history)
                },
                data_quality=1.0,  # Market data always available
                processing_time_ms=processing_time,
                evidence=self._extract_evidence(anomaly_analysis),
                warnings=self._generate_warnings(anomaly_analysis, behavioral_patterns)
            )
            
            self.last_reading = reading
            self.is_initialized = True
            
            logger.debug(f"ANOMALY analysis complete: signal={signal_strength:.3f}, "
                        f"confidence={confidence:.3f}, anomalies={len(self.anomaly_history)}")
            
            return reading
            
        except Exception as e:
            logger.error(f"Error in ANOMALY engine update: {e}")
            return self._create_error_reading(market_data.timestamp, str(e))
    
    def _extract_features(self, market_data: MarketData, order_book: Optional[OrderBookSnapshot]) -> np.ndarray:
        """Extract features for machine learning anomaly detection."""
        features = []
        
        # Price-based features
        features.extend([
            market_data.close,
            market_data.high - market_data.low,  # Range
            (market_data.close - market_data.open) / market_data.open if market_data.open > 0 else 0,  # Return
            market_data.ask - market_data.bid,  # Spread
        ])
        
        # Volume features
        features.extend([
            market_data.volume,
            np.log1p(market_data.volume),  # Log volume
        ])
        
        # Time-based features
        hour = market_data.timestamp.hour
        minute = market_data.timestamp.minute
        weekday = market_data.timestamp.weekday()
        
        features.extend([
            hour / 24.0,  # Normalized hour
            minute / 60.0,  # Normalized minute
            weekday / 7.0,  # Normalized weekday
        ])
        
        # Order book features (if available)
        if order_book and order_book.bids and order_book.asks:
            bid_volume = sum(level.volume for level in order_book.bids[:5])
            ask_volume = sum(level.volume for level in order_book.asks[:5])
            total_volume = bid_volume + ask_volume
            
            if total_volume > 0:
                imbalance = (bid_volume - ask_volume) / total_volume
            else:
                imbalance = 0.0
            
            features.extend([
                bid_volume,
                ask_volume,
                imbalance,
                len(order_book.bids),
                len(order_book.asks),
            ])
        else:
            # Default values when order book not available
            features.extend([0.0, 0.0, 0.0, 0.0, 0.0])
        
        return np.array(features)
    
    def _analyze_anomalies(
        self,
        is_anomaly_ml: bool,
        anomaly_score_ml: float,
        statistical_anomalies: Dict[str, any],
        behavioral_patterns: Dict[str, any],
        order_book_anomalies: Dict[str, any],
        market_data: MarketData
    ) -> Dict[str, any]:
        """
        Analyze comprehensive anomaly detection results.
        
        Args:
            is_anomaly_ml: ML-based anomaly detection result
            anomaly_score_ml: ML anomaly score
            statistical_anomalies: Statistical anomaly results
            behavioral_patterns: Behavioral pattern results
            order_book_anomalies: Order book anomaly results
            market_data: Current market data
            
        Returns:
            Comprehensive anomaly analysis
        """
        analysis = {}
        
        # Aggregate anomaly scores
        anomaly_scores = []
        detected_anomalies = []
        
        # ML-based anomaly
        if is_anomaly_ml:
            anomaly_scores.append(anomaly_score_ml)
            detected_anomalies.append(AnomalyType.STATISTICAL_OUTLIER)
        
        # Statistical anomalies
        for metric, result in statistical_anomalies.items():
            if isinstance(result, dict) and result.get('is_anomaly', False):
                anomaly_scores.append(result.get('severity', 0.0))
                if metric == 'price':
                    detected_anomalies.append(AnomalyType.PRICE_SPIKE)
                elif metric == 'volume':
                    detected_anomalies.append(AnomalyType.VOLUME_ANOMALY)
                elif metric == 'spread':
                    detected_anomalies.append(AnomalyType.SPREAD_MANIPULATION)
                elif metric == 'momentum':
                    detected_anomalies.append(AnomalyType.MOMENTUM_DIVERGENCE)
        
        # Behavioral patterns
        manipulation_patterns = []
        for pattern, result in behavioral_patterns.items():
            if isinstance(result, dict) and result.get('detected', False):
                confidence = result.get('confidence', 0.0)
                anomaly_scores.append(confidence)
                detected_anomalies.append(AnomalyType.BEHAVIORAL_ANOMALY)
                
                if pattern == 'pump_and_dump':
                    manipulation_patterns.append(ManipulationPattern.PUMP_AND_DUMP)
                elif pattern == 'bear_raid':
                    manipulation_patterns.append(ManipulationPattern.BEAR_RAID)
                elif pattern == 'stop_hunting':
                    manipulation_patterns.append(ManipulationPattern.STOP_HUNTING)
                elif pattern == 'wash_trading':
                    manipulation_patterns.append(ManipulationPattern.WASH_TRADING)
        
        # Order book anomalies
        for anomaly_type, result in order_book_anomalies.items():
            if isinstance(result, dict) and result.get('detected', False):
                severity = result.get('severity', result.get('confidence', 0.0))
                anomaly_scores.append(severity)
                detected_anomalies.append(AnomalyType.ORDER_BOOK_IMBALANCE)
                
                if anomaly_type == 'spoofing':
                    manipulation_patterns.append(ManipulationPattern.SPOOFING)
                elif anomaly_type == 'layering':
                    manipulation_patterns.append(ManipulationPattern.LAYERING)
        
        # Calculate overall anomaly metrics
        overall_anomaly_score = np.mean(anomaly_scores) if anomaly_scores else 0.0
        self.anomaly_score.update(overall_anomaly_score)
        
        # Calculate manipulation confidence
        manipulation_score = 0.0
        if manipulation_patterns:
            pattern_scores = [s for s in anomaly_scores[-len(manipulation_patterns):]]
            manipulation_score = np.mean(pattern_scores) if pattern_scores else 0.0
        
        self.manipulation_confidence.update(manipulation_score)
        
        # Create anomaly events for significant detections
        if overall_anomaly_score > 0.5:
            anomaly_event = AnomalyEvent(
                timestamp=market_data.timestamp,
                anomaly_type=detected_anomalies[0] if detected_anomalies else AnomalyType.STATISTICAL_OUTLIER,
                severity=overall_anomaly_score,
                confidence=min(1.0, overall_anomaly_score * 1.2),
                affected_metrics=list(statistical_anomalies.keys()),
                description=f"Anomaly detected with score {overall_anomaly_score:.3f}",
                manipulation_pattern=manipulation_patterns[0] if manipulation_patterns else None
            )
            
            self.anomaly_history.append(anomaly_event)
            if len(self.anomaly_history) > 100:
                self.anomaly_history.pop(0)
        
        analysis['overall_anomaly_score'] = self.anomaly_score.get_value() or 0.0
        analysis['manipulation_confidence'] = self.manipulation_confidence.get_value() or 0.0
        analysis['detected_anomalies'] = detected_anomalies
        analysis['manipulation_patterns'] = manipulation_patterns
        analysis['anomaly_count'] = len(detected_anomalies)
        analysis['recent_anomaly_events'] = len([e for e in self.anomaly_history 
                                               if (market_data.timestamp - e.timestamp).total_seconds() < 3600])
        
        return analysis
    
    def _calculate_signal_strength(self, analysis: Dict[str, any]) -> float:
        """Calculate overall signal strength from anomaly analysis."""
        anomaly_score = analysis.get('overall_anomaly_score', 0.0)
        manipulation_confidence = analysis.get('manipulation_confidence', 0.0)
        anomaly_count = analysis.get('anomaly_count', 0)
        
        # Signal strength increases with anomaly detection
        # Negative signal indicates potential manipulation/anomaly
        signal_magnitude = (
            anomaly_score * 0.6 +
            manipulation_confidence * 0.3 +
            min(1.0, anomaly_count / 3.0) * 0.1
        )
        
        # Return negative signal (anomaly detection is a warning signal)
        return -signal_magnitude
    
    def _calculate_confidence(self, analysis: Dict[str, any]) -> float:
        """Calculate confidence in anomaly analysis."""
        anomaly_score = analysis.get('overall_anomaly_score', 0.0)
        anomaly_count = analysis.get('anomaly_count', 0)
        recent_events = analysis.get('recent_anomaly_events', 0)
        
        # Confidence increases with multiple detection methods agreeing
        confluence_signals = [
            anomaly_score,
            min(1.0, anomaly_count / 2.0),
            min(1.0, recent_events / 3.0)
        ]
        
        return compute_confidence(
            signal_strength=abs(self._calculate_signal_strength(analysis)),
            data_quality=1.0,  # Market data always available
            historical_accuracy=self.performance_tracker.get_accuracy(),
            confluence_signals=confluence_signals
        )
    
    def _detect_market_regime(self, analysis: Dict[str, any]) -> MarketRegime:
        """Detect market regime from anomaly analysis."""
        anomaly_score = analysis.get('overall_anomaly_score', 0.0)
        manipulation_confidence = analysis.get('manipulation_confidence', 0.0)
        recent_events = analysis.get('recent_anomaly_events', 0)
        
        # High anomaly activity suggests manipulated/exhausted market
        if anomaly_score > 0.7 or manipulation_confidence > 0.6:
            return MarketRegime.EXHAUSTED
        
        # Moderate anomaly activity suggests breakout potential
        elif anomaly_score > 0.4 or recent_events > 2:
            return MarketRegime.BREAKOUT
        
        # Low anomaly activity suggests normal market
        else:
            return MarketRegime.CONSOLIDATING
    
    def _extract_evidence(self, analysis: Dict[str, any]) -> Dict[str, float]:
        """Extract evidence scores for transparency."""
        evidence = {}
        
        evidence['anomaly_score'] = analysis.get('overall_anomaly_score', 0.0)
        evidence['manipulation_confidence'] = analysis.get('manipulation_confidence', 0.0)
        evidence['anomaly_count'] = min(1.0, analysis.get('anomaly_count', 0) / 3.0)
        evidence['recent_events'] = min(1.0, analysis.get('recent_anomaly_events', 0) / 5.0)
        evidence['pattern_diversity'] = min(1.0, len(analysis.get('manipulation_patterns', [])) / 3.0)
        
        return evidence
    
    def _generate_warnings(self, analysis: Dict[str, any], behavioral_patterns: Dict[str, any]) -> List[str]:
        """Generate warnings about detected anomalies."""
        warnings = []
        
        # High anomaly score warning
        anomaly_score = analysis.get('overall_anomaly_score', 0.0)
        if anomaly_score > 0.6:
            warnings.append(f"High anomaly score detected: {anomaly_score:.2f}")
        
        # Manipulation pattern warnings
        manipulation_patterns = analysis.get('manipulation_patterns', [])
        for pattern in manipulation_patterns:
            warnings.append(f"Potential manipulation pattern: {pattern.value}")
        
        # Specific behavioral pattern warnings
        for pattern, result in behavioral_patterns.items():
            if isinstance(result, dict) and result.get('detected', False):
                confidence = result.get('confidence', 0.0)
                if confidence > 0.5:
                    warnings.append(f"Behavioral anomaly: {pattern} (confidence: {confidence:.2f})")
        
        # Recent anomaly activity warning
        recent_events = analysis.get('recent_anomaly_events', 0)
        if recent_events > 3:
            warnings.append(f"High anomaly activity: {recent_events} events in last hour")
        
        return warnings
    
    def _create_error_reading(self, timestamp: datetime, error_msg: str) -> DimensionalReading:
        """Create reading when error occurs."""
        return DimensionalReading(
            dimension="ANOMALY",
            timestamp=timestamp,
            signal_strength=0.0,
            confidence=0.0,
            regime=MarketRegime.CONSOLIDATING,
            context={'error': error_msg},
            data_quality=0.0,
            processing_time_ms=0.0,
            evidence={},
            warnings=[f'Analysis error: {error_msg}']
        )
    
    def snapshot(self) -> DimensionalReading:
        """Return current dimensional state."""
        if self.last_reading:
            return self.last_reading
        
        return DimensionalReading(
            dimension="ANOMALY",
            timestamp=datetime.utcnow(),
            signal_strength=0.0,
            confidence=0.0,
            regime=MarketRegime.CONSOLIDATING,
            context={},
            data_quality=0.0,
            processing_time_ms=0.0,
            evidence={},
            warnings=['Engine not initialized']
        )
    
    def reset(self) -> None:
        """Reset engine state."""
        self.last_reading = None
        self.is_initialized = False
        self.anomaly_history.clear()
        
        # Reset components
        self.isolation_forest = IsolationForestDetector(contamination=0.05)
        self.statistical_detector = StatisticalAnomalyDetector(window_size=100)
        self.behavioral_detector = BehavioralPatternDetector()
        self.order_book_detector = OrderBookAnomalyDetector()
        self.anomaly_score = EMA(20)
        self.manipulation_confidence = EMA(15)
        
        logger.info("ANOMALY Engine reset completed")

