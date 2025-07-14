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
    """Market regime classification"""
    TRENDING_BULL = auto()
    TRENDING_BEAR = auto()
    RANGING_HIGH_VOL = auto()
    RANGING_LOW_VOL = auto()
    TRANSITION = auto()
    CRISIS = auto()


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


class RegimeDetector:
    """Detects and tracks market regime changes"""
    
    def __init__(self):
        self.current_regime = MarketRegime.RANGING_LOW_VOL
        self.regime_confidence = 0.5
        self.regime_history: deque[Tuple[datetime, MarketRegime, float]] = deque(maxlen=100)
        self.volatility_window: deque[float] = deque(maxlen=50)
        self.trend_window: deque[float] = deque(maxlen=50)
        
    def update_regime(self, understanding: MarketUnderstanding) -> MarketRegime:
        """Update market regime based on dimensional understanding"""
        
        # Extract key metrics
        consensus_direction = understanding.consensus_direction
        consensus_strength = understanding.consensus_strength
        anomaly_level = abs(understanding.anomaly_reading.value)
        
        # Estimate volatility from dimensional readings
        volatility = np.std([r.value for r in understanding.all_readings])
        self.volatility_window.append(volatility)
        
        # Track trend
        self.trend_window.append(consensus_direction)
        
        # Calculate regime indicators
        avg_volatility = np.mean(self.volatility_window) if self.volatility_window else 0.5
        trend_consistency = np.std(self.trend_window) if len(self.trend_window) > 1 else 1.0
        
        # Regime classification logic
        if anomaly_level > 0.8:
            new_regime = MarketRegime.CRISIS
            confidence = anomaly_level
        elif avg_volatility > 0.7 and trend_consistency < 0.3:
            if abs(consensus_direction) > 0.5:
                new_regime = MarketRegime.TRENDING_BULL if consensus_direction > 0 else MarketRegime.TRENDING_BEAR
            else:
                new_regime = MarketRegime.RANGING_HIGH_VOL
            confidence = consensus_strength
        elif avg_volatility < 0.3:
            new_regime = MarketRegime.RANGING_LOW_VOL
            confidence = 1.0 - avg_volatility
        elif trend_consistency > 0.7:
            new_regime = MarketRegime.TRANSITION
            confidence = trend_consistency
        else:
            if abs(consensus_direction) > 0.4:
                new_regime = MarketRegime.TRENDING_BULL if consensus_direction > 0 else MarketRegime.TRENDING_BEAR
            else:
                new_regime = MarketRegime.RANGING_HIGH_VOL
            confidence = consensus_strength
        
        # Update regime if confidence is sufficient
        if confidence > 0.6:
            self.current_regime = new_regime
            self.regime_confidence = confidence
            self.regime_history.append((understanding.timestamp, new_regime, confidence))
        
        return self.current_regime


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

