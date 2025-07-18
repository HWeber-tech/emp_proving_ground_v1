"""
ANOMALY Dimension - Chaos and Manipulation Detection

This dimension detects market anomalies, manipulation, and chaotic behavior:
- Statistical anomalies and outliers
- Market manipulation patterns
- Flash crashes and sudden volatility spikes
- Coordinated trading activity
- Regime breaks and structural changes
- Self-refuting patterns that indicate manipulation

The ANOMALY dimension serves as a reality check for other dimensions,
identifying when normal market behavior breaks down.
"""

import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple, NamedTuple
from collections import deque, defaultdict
from dataclasses import dataclass
from enum import Enum, auto
import math
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

from ..core.base import (
    DimensionalSensor, DimensionalReading, MarketData, MarketRegime
)


class AnomalyType(Enum):
    PRICE_SPIKE = auto()
    VOLUME_SPIKE = auto()
    SPREAD_ANOMALY = auto()
    VELOCITY_ANOMALY = auto()
    CORRELATION_BREAK = auto()
    REGIME_SHIFT = auto()
    MANIPULATION_PATTERN = auto()
    FLASH_EVENT = auto()
    COORDINATED_ACTIVITY = auto()


class ManipulationPattern(Enum):
    SPOOFING = auto()
    LAYERING = auto()
    WASH_TRADING = auto()
    PUMP_AND_DUMP = auto()
    BEAR_RAID = auto()
    QUOTE_STUFFING = auto()
    MOMENTUM_IGNITION = auto()


@dataclass
class AnomalyEvent:
    """Detected anomaly event"""
    anomaly_type: AnomalyType
    timestamp: datetime
    severity: float  # 0-1 scale
    confidence: float  # 0-1 scale
    description: str
    affected_metrics: Dict[str, float]
    duration_seconds: Optional[float] = None
    
    @property
    def age_minutes(self) -> float:
        return (datetime.now() - self.timestamp).total_seconds() / 60
    
    @property
    def is_recent(self) -> bool:
        return self.age_minutes < 60  # Within last hour


@dataclass
class StatisticalBaseline:
    """Statistical baseline for anomaly detection"""
    mean: float
    std: float
    median: float
    q25: float
    q75: float
    sample_count: int
    last_updated: datetime
    
    @property
    def iqr(self) -> float:
        return self.q75 - self.q25
    
    def is_outlier(self, value: float, threshold: float = 3.0) -> bool:
        """Check if value is statistical outlier"""
        if self.std > 0:
            z_score = abs(value - self.mean) / self.std
            return z_score > threshold
        return False
    
    def is_extreme_outlier(self, value: float) -> bool:
        """Check if value is extreme outlier using IQR method"""
        lower_bound = self.q25 - 3.0 * self.iqr
        upper_bound = self.q75 + 3.0 * self.iqr
        return value < lower_bound or value > upper_bound


class StatisticalAnomalyDetector:
    """Detects statistical anomalies in market data"""
    
    def __init__(self, baseline_periods: int = 200):
        self.baseline_periods = baseline_periods
        
        # Data history for baseline calculation
        self.price_changes: deque = deque(maxlen=baseline_periods)
        self.volumes: deque = deque(maxlen=baseline_periods)
        self.spreads: deque = deque(maxlen=baseline_periods)
        self.velocities: deque = deque(maxlen=baseline_periods)  # Price change per unit time
        
        # Statistical baselines
        self.baselines: Dict[str, StatisticalBaseline] = {}
        
        # Anomaly tracking
        self.detected_anomalies: deque = deque(maxlen=100)
        
    def update(self, data: MarketData, price_change: float) -> None:
        """Update statistical data and baselines"""
        # Calculate velocity (price change per minute)
        velocity = abs(price_change) * 60  # Assuming data comes every second
        
        # Update data history
        self.price_changes.append(price_change)
        self.volumes.append(data.volume)
        self.spreads.append(data.spread)
        self.velocities.append(velocity)
        
        # Update baselines if we have enough data
        if len(self.price_changes) >= 50:  # Minimum for meaningful statistics
            self._update_baselines()
    
    def _update_baselines(self) -> None:
        """Update statistical baselines"""
        datasets = {
            'price_change': list(self.price_changes),
            'volume': list(self.volumes),
            'spread': list(self.spreads),
            'velocity': list(self.velocities)
        }
        
        for name, data in datasets.items():
            if len(data) >= 10:  # Minimum for statistics
                self.baselines[name] = StatisticalBaseline(
                    mean=np.mean(data),
                    std=np.std(data),
                    median=np.median(data),
                    q25=np.percentile(data, 25),
                    q75=np.percentile(data, 75),
                    sample_count=len(data),
                    last_updated=datetime.now()
                )
    
    def detect_anomalies(self, data: MarketData, price_change: float) -> List[AnomalyEvent]:
        """Detect current anomalies"""
        anomalies = []
        
        if not self.baselines:
            return anomalies
        
        # Check price change anomaly
        if 'price_change' in self.baselines:
            baseline = self.baselines['price_change']
            if baseline.is_extreme_outlier(price_change):
                severity = min(1.0, abs(price_change - baseline.mean) / (baseline.std * 5))
                anomalies.append(AnomalyEvent(
                    anomaly_type=AnomalyType.PRICE_SPIKE,
                    timestamp=data.timestamp,
                    severity=severity,
                    confidence=0.8,
                    description=f"Extreme price change: {price_change:.6f}",
                    affected_metrics={'price_change': price_change, 'baseline_mean': baseline.mean}
                ))
        
        # Check volume anomaly
        if 'volume' in self.baselines:
            baseline = self.baselines['volume']
            if baseline.is_extreme_outlier(data.volume):
                severity = min(1.0, (data.volume - baseline.mean) / (baseline.std * 3))
                anomalies.append(AnomalyEvent(
                    anomaly_type=AnomalyType.VOLUME_SPIKE,
                    timestamp=data.timestamp,
                    severity=abs(severity),
                    confidence=0.7,
                    description=f"Extreme volume: {data.volume}",
                    affected_metrics={'volume': data.volume, 'baseline_mean': baseline.mean}
                ))
        
        # Check spread anomaly
        if 'spread' in self.baselines:
            baseline = self.baselines['spread']
            if baseline.is_extreme_outlier(data.spread):
                severity = min(1.0, (data.spread - baseline.mean) / (baseline.std * 3))
                anomalies.append(AnomalyEvent(
                    anomaly_type=AnomalyType.SPREAD_ANOMALY,
                    timestamp=data.timestamp,
                    severity=abs(severity),
                    confidence=0.6,
                    description=f"Extreme spread: {data.spread:.6f}",
                    affected_metrics={'spread': data.spread, 'baseline_mean': baseline.mean}
                ))
        
        # Check velocity anomaly
        velocity = abs(price_change) * 60
        if 'velocity' in self.baselines:
            baseline = self.baselines['velocity']
            if baseline.is_extreme_outlier(velocity):
                severity = min(1.0, (velocity - baseline.mean) / (baseline.std * 3))
                anomalies.append(AnomalyEvent(
                    anomaly_type=AnomalyType.VELOCITY_ANOMALY,
                    timestamp=data.timestamp,
                    severity=severity,
                    confidence=0.7,
                    description=f"Extreme velocity: {velocity:.6f}",
                    affected_metrics={'velocity': velocity, 'baseline_mean': baseline.mean}
                ))
        
        # Store detected anomalies
        for anomaly in anomalies:
            self.detected_anomalies.append(anomaly)
        
        return anomalies
    
    def get_recent_anomalies(self, minutes: float = 60.0) -> List[AnomalyEvent]:
        """Get anomalies from recent time period"""
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        return [
            anomaly for anomaly in self.detected_anomalies
            if anomaly.timestamp > cutoff_time
        ]
    
    def calculate_anomaly_score(self) -> Tuple[float, float]:
        """Calculate overall anomaly score"""
        recent_anomalies = self.get_recent_anomalies(30.0)  # Last 30 minutes
        
        if not recent_anomalies:
            return 0.0, 1.0
        
        # Weight anomalies by recency and severity
        total_score = 0.0
        total_weight = 0.0
        
        for anomaly in recent_anomalies:
            # Recency weight (more recent = higher weight)
            recency_weight = max(0.1, 1.0 - anomaly.age_minutes / 30.0)
            
            # Severity weight
            severity_weight = anomaly.severity * anomaly.confidence
            
            total_score += severity_weight * recency_weight
            total_weight += recency_weight
        
        if total_weight > 0:
            score = min(1.0, total_score / total_weight)
            confidence = min(1.0, len(recent_anomalies) / 5.0)  # More anomalies = higher confidence
        else:
            score, confidence = 0.0, 0.0
        
        return score, confidence


class ManipulationDetector:
    """Detects market manipulation patterns"""
    
    def __init__(self):
        self.order_flow_history: deque = deque(maxlen=100)
        self.price_volume_history: deque = deque(maxlen=100)
        self.detected_patterns: deque = deque(maxlen=50)
        
    def update(self, data: MarketData, price_change: float, volume_delta: float) -> None:
        """Update manipulation detection data"""
        flow_data = {
            'timestamp': data.timestamp,
            'price': data.mid_price,
            'price_change': price_change,
            'volume': data.volume,
            'volume_delta': volume_delta,
            'spread': data.spread
        }
        
        self.order_flow_history.append(flow_data)
        self.price_volume_history.append((data.mid_price, data.volume, data.timestamp))
    
    def detect_manipulation(self) -> List[AnomalyEvent]:
        """Detect manipulation patterns"""
        anomalies = []
        
        # Check for spoofing patterns
        spoofing_anomaly = self._detect_spoofing()
        if spoofing_anomaly:
            anomalies.append(spoofing_anomaly)
        
        # Check for wash trading
        wash_trading_anomaly = self._detect_wash_trading()
        if wash_trading_anomaly:
            anomalies.append(wash_trading_anomaly)
        
        # Check for momentum ignition
        momentum_anomaly = self._detect_momentum_ignition()
        if momentum_anomaly:
            anomalies.append(momentum_anomaly)
        
        # Check for coordinated activity
        coordinated_anomaly = self._detect_coordinated_activity()
        if coordinated_anomaly:
            anomalies.append(coordinated_anomaly)
        
        # Store detected patterns
        for anomaly in anomalies:
            self.detected_patterns.append(anomaly)
        
        return anomalies
    
    def _detect_spoofing(self) -> Optional[AnomalyEvent]:
        """Detect spoofing patterns (large orders that get cancelled)"""
        if len(self.order_flow_history) < 10:
            return None
        
        recent_data = list(self.order_flow_history)[-10:]
        
        # Look for pattern: large volume followed by quick reversal
        for i in range(1, len(recent_data) - 1):
            current = recent_data[i]
            previous = recent_data[i - 1]
            next_item = recent_data[i + 1]
            
            # Large volume spike
            if current['volume'] > previous['volume'] * 3:
                # Followed by quick price reversal
                if (current['price_change'] > 0 and next_item['price_change'] < -current['price_change'] * 0.5):
                    return AnomalyEvent(
                        anomaly_type=AnomalyType.MANIPULATION_PATTERN,
                        timestamp=current['timestamp'],
                        severity=0.7,
                        confidence=0.6,
                        description="Potential spoofing pattern detected",
                        affected_metrics={
                            'volume_spike': current['volume'] / previous['volume'],
                            'price_reversal': next_item['price_change'] / current['price_change']
                        }
                    )
        
        return None
    
    def _detect_wash_trading(self) -> Optional[AnomalyEvent]:
        """Detect wash trading patterns (artificial volume)"""
        if len(self.price_volume_history) < 20:
            return None
        
        recent_data = list(self.price_volume_history)[-20:]
        
        # Group by price levels
        price_groups = defaultdict(list)
        for price, volume, timestamp in recent_data:
            price_level = round(price, 5)  # Group to 5 decimal places
            price_groups[price_level].append((volume, timestamp))
        
        # Look for excessive volume at same price levels
        for price_level, volume_data in price_groups.items():
            if len(volume_data) >= 5:  # At least 5 trades at same price
                volumes = [v[0] for v in volume_data]
                avg_volume = np.mean(volumes)
                
                # Check if volume is unusually consistent (potential wash trading)
                if len(volumes) > 1:
                    volume_consistency = 1.0 - (np.std(volumes) / avg_volume) if avg_volume > 0 else 0.0
                    
                    if volume_consistency > 0.8:  # Very consistent volumes
                        return AnomalyEvent(
                            anomaly_type=AnomalyType.MANIPULATION_PATTERN,
                            timestamp=volume_data[-1][1],
                            severity=volume_consistency,
                            confidence=0.5,
                            description="Potential wash trading pattern detected",
                            affected_metrics={
                                'price_level': price_level,
                                'trade_count': len(volume_data),
                                'volume_consistency': volume_consistency
                            }
                        )
        
        return None
    
    def _detect_momentum_ignition(self) -> Optional[AnomalyEvent]:
        """Detect momentum ignition patterns"""
        if len(self.order_flow_history) < 15:
            return None
        
        recent_data = list(self.order_flow_history)[-15:]
        
        # Look for sudden volume spike followed by sustained price movement
        for i in range(5, len(recent_data) - 5):
            current = recent_data[i]
            
            # Calculate average volume before and after
            before_volumes = [d['volume'] for d in recent_data[i-5:i]]
            after_volumes = [d['volume'] for d in recent_data[i+1:i+6]]
            
            avg_before = np.mean(before_volumes)
            avg_after = np.mean(after_volumes)
            
            # Volume spike followed by sustained higher volume
            if (current['volume'] > avg_before * 5 and  # 5x volume spike
                avg_after > avg_before * 2):  # Sustained higher volume
                
                # Check for sustained price movement in same direction
                price_changes_after = [d['price_change'] for d in recent_data[i+1:i+6]]
                direction_consistency = len([pc for pc in price_changes_after if pc * current['price_change'] > 0]) / len(price_changes_after)
                
                if direction_consistency > 0.6:  # 60% of moves in same direction
                    return AnomalyEvent(
                        anomaly_type=AnomalyType.MANIPULATION_PATTERN,
                        timestamp=current['timestamp'],
                        severity=0.8,
                        confidence=0.7,
                        description="Potential momentum ignition detected",
                        affected_metrics={
                            'volume_spike_ratio': current['volume'] / avg_before,
                            'direction_consistency': direction_consistency,
                            'sustained_volume_ratio': avg_after / avg_before
                        }
                    )
        
        return None
    
    def _detect_coordinated_activity(self) -> Optional[AnomalyEvent]:
        """Detect coordinated trading activity"""
        if len(self.order_flow_history) < 20:
            return None
        
        recent_data = list(self.order_flow_history)[-20:]
        
        # Look for synchronized volume spikes
        volume_spikes = []
        for i, data in enumerate(recent_data):
            if i > 0:
                prev_volume = recent_data[i-1]['volume']
                if data['volume'] > prev_volume * 2:  # 2x volume increase
                    volume_spikes.append(i)
        
        # Check if spikes are coordinated (similar timing patterns)
        if len(volume_spikes) >= 3:
            spike_intervals = []
            for i in range(1, len(volume_spikes)):
                interval = volume_spikes[i] - volume_spikes[i-1]
                spike_intervals.append(interval)
            
            if spike_intervals:
                interval_consistency = 1.0 - (np.std(spike_intervals) / np.mean(spike_intervals)) if np.mean(spike_intervals) > 0 else 0.0
                
                if interval_consistency > 0.7:  # Consistent intervals
                    return AnomalyEvent(
                        anomaly_type=AnomalyType.COORDINATED_ACTIVITY,
                        timestamp=recent_data[-1]['timestamp'],
                        severity=interval_consistency,
                        confidence=0.6,
                        description="Potential coordinated trading activity",
                        affected_metrics={
                            'spike_count': len(volume_spikes),
                            'interval_consistency': interval_consistency,
                            'avg_interval': np.mean(spike_intervals)
                        }
                    )
        
        return None
    
    def get_manipulation_score(self) -> Tuple[float, float]:
        """Calculate manipulation probability score"""
        recent_patterns = [
            pattern for pattern in self.detected_patterns
            if pattern.age_minutes < 30  # Last 30 minutes
        ]
        
        if not recent_patterns:
            return 0.0, 1.0
        
        # Weight by severity and recency
        total_score = 0.0
        total_weight = 0.0
        
        for pattern in recent_patterns:
            recency_weight = max(0.1, 1.0 - pattern.age_minutes / 30.0)
            severity_weight = pattern.severity * pattern.confidence
            
            total_score += severity_weight * recency_weight
            total_weight += recency_weight
        
        if total_weight > 0:
            score = min(1.0, total_score / total_weight)
            confidence = min(1.0, len(recent_patterns) / 3.0)
        else:
            score, confidence = 0.0, 0.0
        
        return score, confidence


class RegimeChangeDetector:
    """Detects market regime changes and structural breaks"""
    
    def __init__(self, lookback_periods: int = 100):
        self.lookback_periods = lookback_periods
        self.price_history: deque = deque(maxlen=lookback_periods)
        self.volatility_history: deque = deque(maxlen=lookback_periods)
        self.correlation_history: deque = deque(maxlen=lookback_periods)
        
        self.detected_breaks: deque = deque(maxlen=20)
        
    def update(self, price: float, volatility: float) -> None:
        """Update regime detection data"""
        self.price_history.append(price)
        self.volatility_history.append(volatility)
        
        # Calculate rolling correlation (simplified)
        if len(self.price_history) >= 20:
            recent_prices = list(self.price_history)[-20:]
            price_changes = [recent_prices[i] - recent_prices[i-1] for i in range(1, len(recent_prices))]
            recent_vols = list(self.volatility_history)[-19:]  # Match length
            
            if len(price_changes) == len(recent_vols) and len(price_changes) > 1:
                correlation = np.corrcoef(price_changes, recent_vols)[0, 1]
                if not np.isnan(correlation):
                    self.correlation_history.append(correlation)
    
    def detect_regime_change(self) -> Optional[AnomalyEvent]:
        """Detect regime changes"""
        if len(self.volatility_history) < 50:
            return None
        
        # Split data into two halves
        vol_data = list(self.volatility_history)
        mid_point = len(vol_data) // 2
        
        first_half = vol_data[:mid_point]
        second_half = vol_data[mid_point:]
        
        # Test for significant difference in volatility regimes
        if len(first_half) > 5 and len(second_half) > 5:
            try:
                # Perform t-test
                t_stat, p_value = stats.ttest_ind(first_half, second_half)
                
                # Significant difference indicates regime change
                if p_value < 0.05:  # 95% confidence
                    mean_diff = abs(np.mean(second_half) - np.mean(first_half))
                    severity = min(1.0, mean_diff / np.mean(first_half)) if np.mean(first_half) > 0 else 0.5
                    
                    regime_change = AnomalyEvent(
                        anomaly_type=AnomalyType.REGIME_SHIFT,
                        timestamp=datetime.now(),
                        severity=severity,
                        confidence=1.0 - p_value,  # Lower p-value = higher confidence
                        description=f"Volatility regime change detected (p={p_value:.4f})",
                        affected_metrics={
                            'p_value': p_value,
                            't_statistic': t_stat,
                            'mean_difference': mean_diff,
                            'first_half_mean': np.mean(first_half),
                            'second_half_mean': np.mean(second_half)
                        }
                    )
                    
                    self.detected_breaks.append(regime_change)
                    return regime_change
                    
            except Exception:
                pass  # Skip if statistical test fails
        
        return None
    
    def get_regime_stability_score(self) -> Tuple[float, float]:
        """Calculate regime stability score (lower = more stable)"""
        recent_breaks = [
            break_event for break_event in self.detected_breaks
            if break_event.age_minutes < 120  # Last 2 hours
        ]
        
        if not recent_breaks:
            return 0.0, 1.0  # Stable regime
        
        # More recent breaks = less stability
        instability_score = 0.0
        for break_event in recent_breaks:
            recency_weight = max(0.1, 1.0 - break_event.age_minutes / 120.0)
            instability_score += break_event.severity * recency_weight
        
        instability_score = min(1.0, instability_score / len(recent_breaks))
        confidence = min(1.0, len(recent_breaks) / 3.0)
        
        return instability_score, confidence


class AnomalyDimension(DimensionalSensor):
    """
    ANOMALY Dimension - Chaos and Manipulation Detection
    
    Detects market anomalies, manipulation, and chaotic behavior:
    - Statistical anomalies and outliers
    - Market manipulation patterns
    - Regime changes and structural breaks
    - Flash events and coordinated activity
    - Self-refuting patterns
    """
    
    def __init__(self):
        super().__init__("ANOMALY")
        
        # Component detectors
        self.statistical_detector = StatisticalAnomalyDetector()
        self.manipulation_detector = ManipulationDetector()
        self.regime_detector = RegimeChangeDetector()
        
        # Previous data for change calculation
        self.previous_price: Optional[float] = None
        
        # Synthesis weights
        self.component_weights = {
            'statistical': 0.40,
            'manipulation': 0.35,
            'regime_change': 0.25
        }
        
        # Peer influence weights (anomalies can disrupt other dimensions)
        self.peer_influences = {
            'why': 0.05,    # Anomalies can override fundamentals
            'how': 0.10,    # Manipulation affects institutional patterns
            'what': 0.15,   # Anomalies can break technical patterns
            'when': 0.05    # Anomalies can disrupt timing patterns
        }
    
    def process(self, data: MarketData, peer_readings: Dict[str, DimensionalReading]) -> DimensionalReading:
        """Process market data to detect anomalies"""
        
        # Calculate price change
        price_change = 0.0
        if self.previous_price is not None:
            price_change = (data.mid_price - self.previous_price) / self.previous_price
        self.previous_price = data.mid_price
        
        # Estimate volume delta (simplified)
        volume_delta = 0.0  # Would need tick data for accurate calculation
        
        # Update detectors
        self.statistical_detector.update(data, price_change)
        self.manipulation_detector.update(data, price_change, volume_delta)
        self.regime_detector.update(data.mid_price, data.volatility if data.volatility > 0 else abs(price_change))
        
        # Detect anomalies
        statistical_anomalies = self.statistical_detector.detect_anomalies(data, price_change)
        manipulation_anomalies = self.manipulation_detector.detect_manipulation()
        regime_change = self.regime_detector.detect_regime_change()
        
        # Calculate component scores
        stat_score, stat_conf = self.statistical_detector.calculate_anomaly_score()
        manip_score, manip_conf = self.manipulation_detector.get_manipulation_score()
        regime_score, regime_conf = self.regime_detector.get_regime_stability_score()
        
        # Weighted synthesis
        components = {
            'statistical': (stat_score, stat_conf),
            'manipulation': (manip_score, manip_conf),
            'regime_change': (regime_score, regime_conf)
        }
        
        weighted_score = 0.0
        weighted_confidence = 0.0
        total_weight = 0.0
        
        for component, (score, conf) in components.items():
            weight = self.component_weights[component]
            weighted_score += score * weight * conf
            weighted_confidence += conf * weight
            total_weight += weight
        
        if total_weight > 0:
            base_score = weighted_score / total_weight
            base_confidence = weighted_confidence / total_weight
        else:
            base_score, base_confidence = 0.0, 0.0
        
        # Apply peer influences (anomalies can disrupt other dimensions)
        peer_disruption = 0.0
        peer_confidence_boost = 0.0
        
        for peer_name, influence_weight in self.peer_influences.items():
            if peer_name in peer_readings:
                peer_reading = peer_readings[peer_name]
                
                # High peer confidence with high anomaly score suggests disruption
                if base_score > 0.5 and peer_reading.confidence > 0.7:
                    peer_disruption += influence_weight * base_score
                
                # Peer confidence boost when anomalies are detected
                if base_score > 0.3:
                    peer_confidence_boost += influence_weight * base_score
        
        # Final score and confidence
        final_score = base_score + peer_disruption * 0.2
        final_confidence = base_confidence + peer_confidence_boost * 0.1
        
        # Normalize
        final_score = max(0.0, min(1.0, final_score))  # Anomaly score is always positive
        final_confidence = max(0.0, min(1.0, final_confidence))
        
        # Build context
        all_anomalies = statistical_anomalies + manipulation_anomalies
        if regime_change:
            all_anomalies.append(regime_change)
        
        recent_statistical = self.statistical_detector.get_recent_anomalies(60.0)
        recent_manipulation = [p for p in self.manipulation_detector.detected_patterns if p.age_minutes < 60]
        
        context = {
            'current_anomalies': len(all_anomalies),
            'recent_statistical_anomalies': len(recent_statistical),
            'recent_manipulation_patterns': len(recent_manipulation),
            'regime_change_detected': regime_change is not None,
            'component_scores': {k: v[0] for k, v in components.items()},
            'component_confidences': {k: v[1] for k, v in components.items()},
            'peer_disruption': peer_disruption,
            'price_change': price_change,
            'anomaly_types': [a.anomaly_type.name for a in all_anomalies],
            'max_severity': max([a.severity for a in all_anomalies], default=0.0)
        }
        
        # Track peer influences
        influences = {}
        for peer_name in self.peer_influences:
            if peer_name in peer_readings:
                peer_reading = peer_readings[peer_name]
                # Anomalies influence other dimensions by disrupting them
                influence_strength = base_score * self.peer_influences[peer_name]
                influences[peer_name] = influence_strength
        
        reading = DimensionalReading(
            dimension=self.name,
            value=final_score,
            confidence=final_confidence,
            timestamp=data.timestamp,
            context=context,
            influences=influences
        )
        
        # Store in history
        with self._lock:
            self.history.append(reading)
        
        return reading
    
    def get_anomaly_summary(self) -> Dict[str, Any]:
        """Get comprehensive anomaly analysis summary"""
        recent_statistical = self.statistical_detector.get_recent_anomalies(60.0)
        recent_manipulation = [p for p in self.manipulation_detector.detected_patterns if p.age_minutes < 60]
        recent_regime_breaks = [b for b in self.regime_detector.detected_breaks if b.age_minutes < 120]
        
        # Group anomalies by type
        anomaly_counts = defaultdict(int)
        for anomaly in recent_statistical:
            anomaly_counts[anomaly.anomaly_type.name] += 1
        
        for pattern in recent_manipulation:
            anomaly_counts[pattern.anomaly_type.name] += 1
        
        return {
            'recent_anomalies': {
                'statistical': len(recent_statistical),
                'manipulation': len(recent_manipulation),
                'regime_breaks': len(recent_regime_breaks),
                'total': len(recent_statistical) + len(recent_manipulation) + len(recent_regime_breaks)
            },
            'anomaly_breakdown': dict(anomaly_counts),
            'severity_distribution': {
                'low': len([a for a in recent_statistical if a.severity < 0.3]),
                'medium': len([a for a in recent_statistical if 0.3 <= a.severity < 0.7]),
                'high': len([a for a in recent_statistical if a.severity >= 0.7])
            },
            'latest_anomalies': [
                {
                    'type': anomaly.anomaly_type.name,
                    'severity': anomaly.severity,
                    'confidence': anomaly.confidence,
                    'description': anomaly.description,
                    'age_minutes': anomaly.age_minutes
                }
                for anomaly in (recent_statistical + recent_manipulation)[-5:]  # Last 5
            ],
            'market_stability': {
                'statistical_baseline_available': bool(self.statistical_detector.baselines),
                'regime_stability': 1.0 - self.regime_detector.get_regime_stability_score()[0],
                'manipulation_risk': self.manipulation_detector.get_manipulation_score()[0]
            }
        }

