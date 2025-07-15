"""
Multidimensional Market Intelligence - Core Architecture

This system understands markets through five interconnected dimensions:
- WHY: Fundamental forces driving market behavior
- HOW: Institutional mechanics and execution patterns  
- WHAT: Technical manifestations and price action
- WHEN: Temporal patterns and timing dynamics
- ANOMALY: Chaos, manipulation, and stress responses

Each dimension maintains awareness of others, creating emergent market understanding.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple, Protocol
from enum import Enum, auto
import numpy as np
from collections import deque
import threading


class MarketRegime(Enum):
    UNKNOWN = auto()
    TRENDING_BULL = auto()
    TRENDING_BEAR = auto()
    RANGING_HIGH_VOL = auto()
    RANGING_LOW_VOL = auto()
    TRANSITION = auto()
    CRISIS = auto()
    VOLATILE = auto()


class ConfidenceLevel(Enum):
    """Confidence levels for dimensional readings"""
    VERY_LOW = 0.2
    LOW = 0.4
    MEDIUM = 0.6
    HIGH = 0.8
    VERY_HIGH = 0.95


@dataclass
class MarketData:
    """Comprehensive market data structure"""
    timestamp: datetime
    symbol: str
    bid: float
    ask: float
    volume: int
    spread: float
    
    # Extended market context
    session: str = ""
    regime: Optional[MarketRegime] = None
    volatility: float = 0.0
    liquidity_score: float = 0.0
    
    @property
    def mid_price(self) -> float:
        return (self.bid + self.ask) / 2
    
    @property
    def spread_bps(self) -> float:
        return (self.spread / self.mid_price) * 10000


@dataclass
class DimensionalReading:
    """Reading from a single dimension with context"""
    dimension: str
    value: float
    confidence: float
    timestamp: datetime
    context: Dict[str, Any] = field(default_factory=dict)
    influences: Dict[str, float] = field(default_factory=dict)  # How other dimensions affect this
    
    def __post_init__(self):
        self.confidence = max(0.0, min(1.0, self.confidence))
        self.value = max(-1.0, min(1.0, self.value))


@dataclass
class MarketUnderstanding:
    """Synthesized understanding across all dimensions"""
    timestamp: datetime
    regime: MarketRegime
    confidence: float
    
    # Dimensional readings
    why_reading: DimensionalReading
    how_reading: DimensionalReading
    what_reading: DimensionalReading
    when_reading: DimensionalReading
    anomaly_reading: DimensionalReading
    
    # Synthesis
    consensus_direction: float  # -1 to 1
    consensus_strength: float   # 0 to 1
    dimensional_alignment: float  # How well dimensions agree
    narrative: str = ""
    
    @property
    def all_readings(self) -> List[DimensionalReading]:
        return [self.why_reading, self.how_reading, self.what_reading, 
                self.when_reading, self.anomaly_reading]


class DimensionalSensor(ABC):
    """Base class for dimensional sensors"""
    
    def __init__(self, name: str):
        self.name = name
        self.history: deque[DimensionalReading] = deque(maxlen=1000)
        self.peer_sensors: Dict[str, 'DimensionalSensor'] = {}
        self._lock = threading.RLock()
        
    @abstractmethod
    def process(self, data: MarketData, peer_readings: Dict[str, DimensionalReading]) -> DimensionalReading:
        """Process market data considering peer dimensional readings"""
        pass
    
    def register_peer(self, name: str, sensor: 'DimensionalSensor') -> None:
        """Register peer sensor for cross-dimensional awareness"""
        self.peer_sensors[name] = sensor
    
    def get_recent_reading(self) -> Optional[DimensionalReading]:
        """Get most recent reading"""
        with self._lock:
            return self.history[-1] if self.history else None
    
    def get_trend(self, lookback_periods: int = 20) -> Tuple[float, float]:
        """Get trend direction and strength over lookback periods"""
        with self._lock:
            if len(self.history) < 2:
                return 0.0, 0.0
            
            recent = list(self.history)[-lookback_periods:]
            values = [r.value for r in recent]
            
            if len(values) < 2:
                return 0.0, 0.0
            
            # Simple linear regression for trend
            x = np.arange(len(values))
            slope = np.polyfit(x, values, 1)[0]
            
            # R-squared for trend strength
            y_pred = np.polyval([slope, values[0]], x)
            ss_res = np.sum((values - y_pred) ** 2)
            ss_tot = np.sum((values - np.mean(values)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            return float(slope), float(r_squared)


class MarketNarrative:
    """Constructs coherent market narratives from dimensional readings"""
    
    def __init__(self):
        self.narrative_templates = {
            'bullish_momentum': "Strong fundamental drivers ({why:.2f}) supported by institutional flow ({how:.2f}) "
                              "manifesting in technical breakout ({what:.2f}) during favorable timing ({when:.2f})",
            
            'bearish_pressure': "Fundamental headwinds ({why:.2f}) with institutional selling ({how:.2f}) "
                              "creating technical breakdown ({what:.2f}) amplified by timing factors ({when:.2f})",
            
            'range_bound': "Mixed fundamental signals ({why:.2f}) with balanced institutional activity ({how:.2f}) "
                         "resulting in sideways technical action ({what:.2f}) during neutral timing ({when:.2f})",
            
            'anomaly_driven': "Market disrupted by anomalous conditions ({anomaly:.2f}) overriding normal "
                            "fundamental ({why:.2f}) and technical ({what:.2f}) relationships",
            
            'transition': "Market in transition with conflicting signals across dimensions - "
                        "fundamentals ({why:.2f}), institutions ({how:.2f}), technicals ({what:.2f}), timing ({when:.2f})"
        }
    
    def construct_narrative(self, understanding: MarketUnderstanding) -> str:
        """Construct coherent narrative from dimensional readings"""
        readings = {
            'why': understanding.why_reading.value,
            'how': understanding.how_reading.value,
            'what': understanding.what_reading.value,
            'when': understanding.when_reading.value,
            'anomaly': understanding.anomaly_reading.value
        }
        
        # Determine dominant narrative type
        anomaly_strength = abs(readings['anomaly'])
        alignment = understanding.dimensional_alignment
        consensus_strength = understanding.consensus_strength
        
        if anomaly_strength > 0.7:
            template_key = 'anomaly_driven'
        elif alignment < 0.3:
            template_key = 'transition'
        elif consensus_strength > 0.6:
            if understanding.consensus_direction > 0.3:
                template_key = 'bullish_momentum'
            elif understanding.consensus_direction < -0.3:
                template_key = 'bearish_pressure'
            else:
                template_key = 'range_bound'
        else:
            template_key = 'range_bound'
        
        template = self.narrative_templates.get(template_key, self.narrative_templates['transition'])
        return template.format(**readings)


class DimensionalCorrelationMatrix:
    """Tracks correlations and influences between dimensions"""
    
    def __init__(self):
        self.dimensions = ['why', 'how', 'what', 'when', 'anomaly']
        self.correlation_matrix = np.eye(5)  # Start with identity matrix
        self.influence_weights = np.ones((5, 5)) * 0.2  # Equal influence initially
        self.update_count = 0
        self._lock = threading.RLock()
    
    def update_correlations(self, readings: Dict[str, DimensionalReading]) -> None:
        """Update correlation matrix based on new readings"""
        with self._lock:
            if len(readings) != 5:
                return
            
            values = np.array([readings[dim].value for dim in self.dimensions])
            
            # Exponential moving average update
            alpha = 0.1  # Learning rate
            
            if self.update_count > 0:
                # Update correlation matrix
                outer_product = np.outer(values, values)
                self.correlation_matrix = (1 - alpha) * self.correlation_matrix + alpha * outer_product
            
            self.update_count += 1
    
    def get_influence_weights(self, target_dimension: str) -> Dict[str, float]:
        """Get influence weights for target dimension from other dimensions"""
        with self._lock:
            if target_dimension not in self.dimensions:
                return {}
            
            target_idx = self.dimensions.index(target_dimension)
            weights = {}
            
            for i, dim in enumerate(self.dimensions):
                if dim != target_dimension:
                    # Influence based on correlation strength
                    correlation = abs(self.correlation_matrix[target_idx, i])
                    weights[dim] = correlation
            
            # Normalize weights
            total_weight = sum(weights.values())
            if total_weight > 0:
                weights = {k: v / total_weight for k, v in weights.items()}
            
            return weights
    
    def get_dimensional_alignment(self, readings: Dict[str, DimensionalReading]) -> float:
        """Calculate how well dimensions are aligned"""
        with self._lock:
            if len(readings) != 5:
                return 0.0
            
            values = np.array([readings[dim].value for dim in self.dimensions])
            
            # Calculate pairwise correlations for current readings
            alignment_scores = []
            for i in range(len(values)):
                for j in range(i + 1, len(values)):
                    # Correlation between current values
                    correlation = values[i] * values[j]  # Simple correlation proxy
                    alignment_scores.append(correlation)
            
            # Return average alignment
            return float(np.mean(alignment_scores)) if alignment_scores else 0.0


class MarketRegimeDetector:
    """Sophisticated market regime detection based on multiple factors"""
    
    def __init__(self, lookback_periods: int = 50):
        self.lookback_periods = lookback_periods
        self.price_history = deque(maxlen=lookback_periods)
        self.volume_history = deque(maxlen=lookback_periods)
        self.volatility_history = deque(maxlen=lookback_periods)
        self.trend_history = deque(maxlen=lookback_periods)
        
    def update_market_data(self, market_data: MarketData) -> None:
        """Update with new market data"""
        mid_price = market_data.mid_price
        self.price_history.append(mid_price)
        self.volume_history.append(market_data.volume)
        
        # Calculate volatility (rolling standard deviation of returns)
        if len(self.price_history) >= 2:
            returns = np.diff(list(self.price_history))
            volatility = np.std(returns) if len(returns) > 0 else 0.0
            self.volatility_history.append(volatility)
        
        # Calculate trend strength
        if len(self.price_history) >= 20:
            recent_prices = list(self.price_history)[-20:]
            x = np.arange(len(recent_prices))
            slope = np.polyfit(x, recent_prices, 1)[0]
            trend_strength = abs(slope) / np.mean(recent_prices) if np.mean(recent_prices) > 0 else 0.0
            self.trend_history.append(trend_strength)
    
    def determine_regime(self, market_data: MarketData) -> MarketRegime:
        """Determine current market regime based on comprehensive analysis"""
        
        self.update_market_data(market_data)
        
        if len(self.price_history) < 10:
            return MarketRegime.UNKNOWN
        
        # Calculate key metrics
        volatility = self._calculate_current_volatility()
        trend_strength = self._calculate_trend_strength()
        volume_profile = self._analyze_volume_profile()
        price_momentum = self._calculate_price_momentum()
        
        # Crisis detection (extreme conditions)
        if self._is_crisis_condition(volatility, volume_profile):
            return MarketRegime.CRISIS
        
        # Volatile regime detection
        if self._is_volatile_condition(volatility, volume_profile):
            return MarketRegime.VOLATILE
        
        # Trending regime detection
        if self._is_trending_condition(trend_strength, price_momentum):
            if price_momentum > 0:
                return MarketRegime.TRENDING_BULL
            else:
                return MarketRegime.TRENDING_BEAR
        
        # Ranging regime detection
        if self._is_ranging_condition(trend_strength, volatility):
            if volatility > self._get_volatility_threshold() * 1.5:
                return MarketRegime.RANGING_HIGH_VOL
            else:
                return MarketRegime.RANGING_LOW_VOL
        
        # Transition regime
        if self._is_transition_condition(trend_strength, volatility, volume_profile):
            return MarketRegime.TRANSITION
        
        return MarketRegime.UNKNOWN
    
    def _calculate_current_volatility(self) -> float:
        """Calculate current volatility level"""
        if len(self.volatility_history) < 5:
            return 0.0
        return np.mean(list(self.volatility_history)[-5:])
    
    def _calculate_trend_strength(self) -> float:
        """Calculate current trend strength"""
        if len(self.trend_history) < 5:
            return 0.0
        return np.mean(list(self.trend_history)[-5:])
    
    def _analyze_volume_profile(self) -> Dict[str, float]:
        """Analyze volume characteristics"""
        if len(self.volume_history) < 10:
            return {'avg_volume': 0.0, 'volume_trend': 0.0, 'volume_volatility': 0.0}
        
        volumes = list(self.volume_history)[-10:]
        avg_volume = np.mean(volumes)
        volume_trend = np.polyfit(range(len(volumes)), volumes, 1)[0] / avg_volume if avg_volume > 0 else 0.0
        volume_volatility = np.std(volumes) / avg_volume if avg_volume > 0 else 0.0
        
        return {
            'avg_volume': avg_volume,
            'volume_trend': volume_trend,
            'volume_volatility': volume_volatility
        }
    
    def _calculate_price_momentum(self) -> float:
        """Calculate price momentum (short-term trend)"""
        if len(self.price_history) < 5:
            return 0.0
        
        recent_prices = list(self.price_history)[-5:]
        return (recent_prices[-1] - recent_prices[0]) / recent_prices[0] if recent_prices[0] > 0 else 0.0
    
    def _get_volatility_threshold(self) -> float:
        """Get adaptive volatility threshold"""
        if len(self.volatility_history) < 20:
            return 0.001  # Default threshold
        
        historical_vol = list(self.volatility_history)
        return np.percentile(historical_vol, 75)  # 75th percentile as threshold
    
    def _is_crisis_condition(self, volatility: float, volume_profile: Dict[str, float]) -> bool:
        """Detect crisis conditions"""
        vol_threshold = self._get_volatility_threshold()
        return (volatility > vol_threshold * 3.0 or 
                volume_profile['volume_volatility'] > 2.0)
    
    def _is_volatile_condition(self, volatility: float, volume_profile: Dict[str, float]) -> bool:
        """Detect volatile conditions"""
        vol_threshold = self._get_volatility_threshold()
        return (volatility > vol_threshold * 2.0 or 
                volume_profile['volume_volatility'] > 1.5)
    
    def _is_trending_condition(self, trend_strength: float, price_momentum: float) -> bool:
        """Detect trending conditions"""
        return (trend_strength > 0.0001 and  # Strong trend
                abs(price_momentum) > 0.002)  # Clear momentum
    
    def _is_ranging_condition(self, trend_strength: float, volatility: float) -> bool:
        """Detect ranging conditions"""
        vol_threshold = self._get_volatility_threshold()
        return (trend_strength < 0.00005 and  # Weak trend
                volatility < vol_threshold * 2.0)  # Moderate volatility
    
    def _is_transition_condition(self, trend_strength: float, volatility: float, 
                               volume_profile: Dict[str, float]) -> bool:
        """Detect transition conditions"""
        vol_threshold = self._get_volatility_threshold()
        return (0.00005 <= trend_strength <= 0.0001 and  # Moderate trend
                vol_threshold <= volatility <= vol_threshold * 2.0)  # Moderate volatility


class MemoryBank:
    """Stores and retrieves similar market episodes for pattern recognition"""
    
    def __init__(self, max_episodes: int = 1000):
        self.episodes: deque[MarketUnderstanding] = deque(maxlen=max_episodes)
        self.regime_episodes: Dict[MarketRegime, List[MarketUnderstanding]] = {
            regime: [] for regime in MarketRegime
        }
        self._lock = threading.RLock()
    
    def store_episode(self, understanding: MarketUnderstanding) -> None:
        """Store market understanding episode"""
        with self._lock:
            self.episodes.append(understanding)
            
            # Store by regime
            regime_list = self.regime_episodes[understanding.regime]
            regime_list.append(understanding)
            
            # Keep regime lists bounded
            if len(regime_list) > 200:
                regime_list.pop(0)
    
    def find_similar_episodes(self, current: MarketUnderstanding, 
                            similarity_threshold: float = 0.8) -> List[MarketUnderstanding]:
        """Find episodes similar to current market state"""
        with self._lock:
            similar = []
            
            # First filter by regime
            candidates = self.regime_episodes.get(current.regime, [])
            
            for episode in candidates:
                similarity = self._calculate_similarity(current, episode)
                if similarity >= similarity_threshold:
                    similar.append(episode)
            
            # Sort by similarity (most similar first)
            similar.sort(key=lambda ep: self._calculate_similarity(current, ep), reverse=True)
            
            return similar[:10]  # Return top 10 most similar
    
    def _calculate_similarity(self, ep1: MarketUnderstanding, ep2: MarketUnderstanding) -> float:
        """Calculate similarity between two episodes"""
        
        # Compare dimensional readings
        dim_similarities = []
        for r1, r2 in zip(ep1.all_readings, ep2.all_readings):
            value_sim = 1.0 - abs(r1.value - r2.value) / 2.0  # Normalize to [0,1]
            confidence_sim = 1.0 - abs(r1.confidence - r2.confidence)
            dim_similarities.append((value_sim + confidence_sim) / 2.0)
        
        # Compare consensus metrics
        direction_sim = 1.0 - abs(ep1.consensus_direction - ep2.consensus_direction) / 2.0
        strength_sim = 1.0 - abs(ep1.consensus_strength - ep2.consensus_strength)
        alignment_sim = 1.0 - abs(ep1.dimensional_alignment - ep2.dimensional_alignment)
        
        # Weighted average
        dim_weight = 0.6
        consensus_weight = 0.4
        
        dim_avg = np.mean(dim_similarities)
        consensus_avg = np.mean([direction_sim, strength_sim, alignment_sim])
        
        return dim_weight * dim_avg + consensus_weight * consensus_avg


@dataclass
class EconomicData:
    indicator: str
    value: float
    timestamp: datetime
    frequency: str
    surprise_factor: float
    importance: float

