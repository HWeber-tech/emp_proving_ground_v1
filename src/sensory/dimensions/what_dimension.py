"""
WHAT Dimension - Technical Reality

This dimension understands what the market is actually doing through price action:
- Market structure and trend analysis
- Support/resistance dynamics
- Momentum and mean reversion patterns
- Volume and participation analysis
- Pattern recognition and fractal analysis

The WHAT dimension translates fundamental forces (WHY) and institutional activity (HOW)
into observable technical reality.
"""

import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple, NamedTuple
from collections import deque
from dataclasses import dataclass
from enum import Enum, auto
import math

from ..core.base import (
    DimensionalSensor, DimensionalReading, MarketData, MarketRegime
)


class TrendDirection(Enum):
    STRONG_UP = auto()
    WEAK_UP = auto()
    SIDEWAYS = auto()
    WEAK_DOWN = auto()
    STRONG_DOWN = auto()


class MarketStructure(Enum):
    HIGHER_HIGHS_HIGHER_LOWS = auto()
    LOWER_HIGHS_LOWER_LOWS = auto()
    HIGHER_HIGHS_LOWER_LOWS = auto()  # Divergence
    LOWER_HIGHS_HIGHER_LOWS = auto()  # Convergence
    CHOPPY = auto()


@dataclass
class SwingPoint:
    """Swing high or low point"""
    price: float
    timestamp: datetime
    is_high: bool
    strength: float  # How significant this swing is
    volume: int = 0
    
    @property
    def type_str(self) -> str:
        return "High" if self.is_high else "Low"


@dataclass
class SupportResistanceLevel:
    """Support or resistance level"""
    price: float
    strength: float
    touch_count: int
    first_touch: datetime
    last_touch: datetime
    is_support: bool
    
    @property
    def age_hours(self) -> float:
        return (datetime.now() - self.first_touch).total_seconds() / 3600
    
    @property
    def recency_hours(self) -> float:
        return (datetime.now() - self.last_touch).total_seconds() / 3600


class PriceActionAnalyzer:
    """Analyzes price action patterns and market structure"""
    
    def __init__(self, lookback_periods: int = 100):
        self.lookback_periods = lookback_periods
        self.price_history: deque[Tuple[float, datetime, int]] = deque(maxlen=lookback_periods)
        self.swing_points: deque[SwingPoint] = deque(maxlen=50)
        self.support_resistance: List[SupportResistanceLevel] = []
        
    def update_price(self, price: float, timestamp: datetime, volume: int) -> None:
        """Update price history"""
        self.price_history.append((price, timestamp, volume))
        self._detect_swing_points()
        self._update_support_resistance()
    
    def _detect_swing_points(self) -> None:
        """Detect swing highs and lows"""
        if len(self.price_history) < 5:
            return
        
        prices = [p[0] for p in self.price_history]
        timestamps = [p[1] for p in self.price_history]
        volumes = [p[2] for p in self.price_history]
        
        # Look for swing points using a simple peak/trough detection
        for i in range(2, len(prices) - 2):
            current_price = prices[i]
            
            # Check for swing high
            if (current_price > prices[i-1] and current_price > prices[i-2] and
                current_price > prices[i+1] and current_price > prices[i+2]):
                
                # Calculate strength based on how much higher than surrounding prices
                strength = min(
                    current_price - max(prices[i-2:i]),
                    current_price - max(prices[i+1:i+3])
                ) / current_price
                
                swing = SwingPoint(
                    price=current_price,
                    timestamp=timestamps[i],
                    is_high=True,
                    strength=strength,
                    volume=volumes[i]
                )
                
                # Only add if significantly different from last swing high
                if not self.swing_points or abs(current_price - self._last_swing_high()) > current_price * 0.001:
                    self.swing_points.append(swing)
            
            # Check for swing low
            elif (current_price < prices[i-1] and current_price < prices[i-2] and
                  current_price < prices[i+1] and current_price < prices[i+2]):
                
                strength = min(
                    min(prices[i-2:i]) - current_price,
                    min(prices[i+1:i+3]) - current_price
                ) / current_price
                
                swing = SwingPoint(
                    price=current_price,
                    timestamp=timestamps[i],
                    is_high=False,
                    strength=strength,
                    volume=volumes[i]
                )
                
                if not self.swing_points or abs(current_price - self._last_swing_low()) > current_price * 0.001:
                    self.swing_points.append(swing)
    
    def _last_swing_high(self) -> float:
        """Get last swing high price"""
        for swing in reversed(self.swing_points):
            if swing.is_high:
                return swing.price
        return 0.0
    
    def _last_swing_low(self) -> float:
        """Get last swing low price"""
        for swing in reversed(self.swing_points):
            if not swing.is_high:
                return swing.price
        return float('inf')
    
    def _update_support_resistance(self) -> None:
        """Update support and resistance levels"""
        if len(self.swing_points) < 3:
            return
        
        # Group swing points by price proximity
        tolerance = 0.002  # 20 pips for EURUSD
        
        for swing in self.swing_points:
            # Find existing level within tolerance
            existing_level = None
            for level in self.support_resistance:
                if abs(level.price - swing.price) / swing.price < tolerance:
                    existing_level = level
                    break
            
            if existing_level:
                # Update existing level
                existing_level.touch_count += 1
                existing_level.last_touch = swing.timestamp
                existing_level.strength = min(1.0, existing_level.strength + 0.1)
            else:
                # Create new level
                new_level = SupportResistanceLevel(
                    price=swing.price,
                    strength=swing.strength,
                    touch_count=1,
                    first_touch=swing.timestamp,
                    last_touch=swing.timestamp,
                    is_support=not swing.is_high
                )
                self.support_resistance.append(new_level)
        
        # Clean old levels
        current_time = datetime.now()
        self.support_resistance = [
            level for level in self.support_resistance
            if (current_time - level.last_touch).total_seconds() < 7 * 24 * 3600  # Keep for 1 week
        ]
    
    def get_market_structure(self) -> Tuple[MarketStructure, float]:
        """Analyze market structure"""
        if len(self.swing_points) < 4:
            return MarketStructure.CHOPPY, 0.5
        
        # Get recent swing points
        recent_swings = list(self.swing_points)[-6:]
        
        # Separate highs and lows
        highs = [s for s in recent_swings if s.is_high]
        lows = [s for s in recent_swings if not s.is_high]
        
        if len(highs) < 2 or len(lows) < 2:
            return MarketStructure.CHOPPY, 0.5
        
        # Sort by timestamp
        highs.sort(key=lambda x: x.timestamp)
        lows.sort(key=lambda x: x.timestamp)
        
        # Analyze trends in highs and lows
        high_trend = self._calculate_trend([h.price for h in highs])
        low_trend = self._calculate_trend([l.price for l in lows])
        
        # Determine structure
        if high_trend > 0.1 and low_trend > 0.1:
            structure = MarketStructure.HIGHER_HIGHS_HIGHER_LOWS
            confidence = min(high_trend, low_trend)
        elif high_trend < -0.1 and low_trend < -0.1:
            structure = MarketStructure.LOWER_HIGHS_LOWER_LOWS
            confidence = min(abs(high_trend), abs(low_trend))
        elif high_trend > 0.1 and low_trend < -0.1:
            structure = MarketStructure.HIGHER_HIGHS_LOWER_LOWS
            confidence = (high_trend + abs(low_trend)) / 2
        elif high_trend < -0.1 and low_trend > 0.1:
            structure = MarketStructure.LOWER_HIGHS_HIGHER_LOWS
            confidence = (abs(high_trend) + low_trend) / 2
        else:
            structure = MarketStructure.CHOPPY
            confidence = 1.0 - (abs(high_trend) + abs(low_trend)) / 2
        
        return structure, min(1.0, confidence)
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend slope for a series of values"""
        if len(values) < 2:
            return 0.0
        
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        
        # Normalize by average value
        avg_value = np.mean(values)
        return slope / avg_value if avg_value > 0 else 0.0
    
    def get_nearest_levels(self, current_price: float, max_distance: float = 0.01) -> List[SupportResistanceLevel]:
        """Get support/resistance levels near current price"""
        nearby_levels = []
        
        for level in self.support_resistance:
            distance = abs(level.price - current_price) / current_price
            if distance <= max_distance:
                nearby_levels.append(level)
        
        # Sort by distance
        nearby_levels.sort(key=lambda l: abs(l.price - current_price))
        return nearby_levels


class MomentumAnalyzer:
    """Analyzes momentum and mean reversion characteristics"""
    
    def __init__(self):
        self.price_changes: deque[float] = deque(maxlen=50)
        self.volume_changes: deque[float] = deque(maxlen=50)
        self.rsi_period = 14
        self.momentum_period = 10
        
    def update(self, price_change: float, volume_change: float) -> None:
        """Update momentum data"""
        self.price_changes.append(price_change)
        self.volume_changes.append(volume_change)
    
    def calculate_rsi(self) -> Tuple[float, float]:
        """Calculate RSI and confidence"""
        if len(self.price_changes) < self.rsi_period:
            return 50.0, 0.0
        
        changes = list(self.price_changes)[-self.rsi_period:]
        gains = [max(0, change) for change in changes]
        losses = [max(0, -change) for change in changes]
        
        avg_gain = np.mean(gains) if gains else 0.0
        avg_loss = np.mean(losses) if losses else 0.0
        
        if avg_loss == 0:
            rsi = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
        
        # Confidence based on data quality
        confidence = min(1.0, len(self.price_changes) / self.rsi_period)
        
        return rsi, confidence
    
    def calculate_momentum(self) -> Tuple[float, float]:
        """Calculate price momentum"""
        if len(self.price_changes) < self.momentum_period:
            return 0.0, 0.0
        
        recent_changes = list(self.price_changes)[-self.momentum_period:]
        
        # Momentum as cumulative change
        momentum = sum(recent_changes)
        
        # Confidence based on consistency
        consistency = 1.0 - np.std(recent_changes) if len(recent_changes) > 1 else 0.5
        confidence = max(0.0, min(1.0, consistency))
        
        return momentum, confidence
    
    def calculate_volume_momentum(self) -> Tuple[float, float]:
        """Calculate volume momentum"""
        if len(self.volume_changes) < self.momentum_period:
            return 0.0, 0.0
        
        recent_changes = list(self.volume_changes)[-self.momentum_period:]
        volume_momentum = np.mean(recent_changes)
        
        consistency = 1.0 - np.std(recent_changes) if len(recent_changes) > 1 else 0.5
        confidence = max(0.0, min(1.0, consistency))
        
        return volume_momentum, confidence


class PatternRecognizer:
    """Recognizes technical patterns"""
    
    def __init__(self):
        self.pattern_history: deque[Tuple[str, float, datetime]] = deque(maxlen=20)
        
    def analyze_patterns(self, price_analyzer: PriceActionAnalyzer) -> Tuple[str, float]:
        """Analyze current patterns"""
        if len(price_analyzer.swing_points) < 5:
            return "insufficient_data", 0.0
        
        # Get recent swing points
        recent_swings = list(price_analyzer.swing_points)[-5:]
        
        # Check for common patterns
        pattern, confidence = self._check_triangle_pattern(recent_swings)
        if confidence > 0.6:
            return pattern, confidence
        
        pattern, confidence = self._check_flag_pattern(recent_swings)
        if confidence > 0.6:
            return pattern, confidence
        
        pattern, confidence = self._check_double_top_bottom(recent_swings)
        if confidence > 0.6:
            return pattern, confidence
        
        return "no_clear_pattern", 0.3
    
    def _check_triangle_pattern(self, swings: List[SwingPoint]) -> Tuple[str, float]:
        """Check for triangle patterns"""
        if len(swings) < 4:
            return "triangle", 0.0
        
        highs = [s for s in swings if s.is_high]
        lows = [s for s in swings if not s.is_high]
        
        if len(highs) < 2 or len(lows) < 2:
            return "triangle", 0.0
        
        # Sort by time
        highs.sort(key=lambda x: x.timestamp)
        lows.sort(key=lambda x: x.timestamp)
        
        # Check if highs are descending and lows are ascending (symmetrical triangle)
        high_trend = self._calculate_swing_trend([h.price for h in highs])
        low_trend = self._calculate_swing_trend([l.price for l in lows])
        
        if high_trend < -0.05 and low_trend > 0.05:
            convergence = abs(high_trend) + low_trend
            return "symmetrical_triangle", min(1.0, convergence)
        elif high_trend < -0.05 and abs(low_trend) < 0.02:
            return "descending_triangle", min(1.0, abs(high_trend))
        elif abs(high_trend) < 0.02 and low_trend > 0.05:
            return "ascending_triangle", min(1.0, low_trend)
        
        return "triangle", 0.0
    
    def _check_flag_pattern(self, swings: List[SwingPoint]) -> Tuple[str, float]:
        """Check for flag patterns"""
        if len(swings) < 4:
            return "flag", 0.0
        
        # Simple flag detection: alternating highs and lows in narrow range
        price_range = max(s.price for s in swings) - min(s.price for s in swings)
        avg_price = np.mean([s.price for s in swings])
        
        if price_range / avg_price < 0.01:  # Narrow range
            return "flag", 0.7
        
        return "flag", 0.0
    
    def _check_double_top_bottom(self, swings: List[SwingPoint]) -> Tuple[str, float]:
        """Check for double top/bottom patterns"""
        highs = [s for s in swings if s.is_high]
        lows = [s for s in swings if not s.is_high]
        
        # Check double top
        if len(highs) >= 2:
            last_two_highs = highs[-2:]
            price_diff = abs(last_two_highs[0].price - last_two_highs[1].price)
            avg_price = np.mean([h.price for h in last_two_highs])
            
            if price_diff / avg_price < 0.005:  # Very similar prices
                return "double_top", 0.8
        
        # Check double bottom
        if len(lows) >= 2:
            last_two_lows = lows[-2:]
            price_diff = abs(last_two_lows[0].price - last_two_lows[1].price)
            avg_price = np.mean([l.price for l in last_two_lows])
            
            if price_diff / avg_price < 0.005:
                return "double_bottom", 0.8
        
        return "double", 0.0
    
    def _calculate_swing_trend(self, prices: List[float]) -> float:
        """Calculate trend in swing prices"""
        if len(prices) < 2:
            return 0.0
        
        x = np.arange(len(prices))
        slope = np.polyfit(x, prices, 1)[0]
        avg_price = np.mean(prices)
        
        return slope / avg_price if avg_price > 0 else 0.0


class WhatDimension(DimensionalSensor):
    """
    WHAT Dimension - Technical Reality
    
    Understands what the market is actually doing through price action:
    - Market structure and trend analysis
    - Support/resistance dynamics
    - Momentum and mean reversion
    - Volume analysis
    - Pattern recognition
    """
    
    def __init__(self):
        super().__init__("WHAT")
        
        # Component analyzers
        self.price_analyzer = PriceActionAnalyzer()
        self.momentum_analyzer = MomentumAnalyzer()
        self.pattern_recognizer = PatternRecognizer()
        
        # Previous price for change calculation
        self.previous_price: Optional[float] = None
        self.previous_volume: Optional[int] = None
        
        # Synthesis weights
        self.component_weights = {
            'structure': 0.30,
            'momentum': 0.25,
            'support_resistance': 0.25,
            'patterns': 0.20
        }
        
        # Peer influence weights
        self.peer_influences = {
            'why': 0.20,    # Fundamentals drive technical moves
            'how': 0.25,    # Institutional activity creates technical patterns
            'when': 0.15,   # Timing affects technical significance
            'anomaly': 0.10 # Anomalies can break technical patterns
        }
    
    def process(self, data: MarketData, peer_readings: Dict[str, DimensionalReading]) -> DimensionalReading:
        """Process market data to understand technical reality"""
        
        # Update analyzers
        current_price = data.mid_price
        self.price_analyzer.update_price(current_price, data.timestamp, data.volume)
        
        # Calculate price and volume changes
        price_change = 0.0
        volume_change = 0.0
        
        if self.previous_price is not None:
            price_change = (current_price - self.previous_price) / self.previous_price
        
        if self.previous_volume is not None and self.previous_volume > 0:
            volume_change = (data.volume - self.previous_volume) / self.previous_volume
        
        self.momentum_analyzer.update(price_change, volume_change)
        
        # Update previous values
        self.previous_price = current_price
        self.previous_volume = data.volume
        
        # Calculate component scores
        structure_score, structure_conf = self._analyze_structure()
        momentum_score, momentum_conf = self._analyze_momentum()
        sr_score, sr_conf = self._analyze_support_resistance(current_price)
        pattern_score, pattern_conf = self._analyze_patterns()
        
        # Weighted synthesis
        components = {
            'structure': (structure_score, structure_conf),
            'momentum': (momentum_score, momentum_conf),
            'support_resistance': (sr_score, sr_conf),
            'patterns': (pattern_score, pattern_conf)
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
        
        # Apply peer influences
        peer_adjustment = 0.0
        peer_confidence_boost = 0.0
        
        for peer_name, influence_weight in self.peer_influences.items():
            if peer_name in peer_readings:
                peer_reading = peer_readings[peer_name]
                
                # Peer influence on score
                peer_adjustment += peer_reading.value * influence_weight * peer_reading.confidence
                
                # Peer influence on confidence
                alignment = 1.0 - abs(base_score - peer_reading.value) / 2.0
                peer_confidence_boost += alignment * influence_weight * peer_reading.confidence
        
        # Final score and confidence
        final_score = base_score + peer_adjustment * 0.2  # Moderate peer influence
        final_confidence = base_confidence + peer_confidence_boost * 0.15
        
        # Normalize
        final_score = max(-1.0, min(1.0, final_score))
        final_confidence = max(0.0, min(1.0, final_confidence))
        
        # Build context
        market_structure, struct_conf = self.price_analyzer.get_market_structure()
        pattern, pattern_conf = self.pattern_recognizer.analyze_patterns(self.price_analyzer)
        
        context = {
            'market_structure': market_structure.name,
            'structure_confidence': struct_conf,
            'current_pattern': pattern,
            'pattern_confidence': pattern_conf,
            'price_change': price_change,
            'volume_change': volume_change,
            'component_scores': {k: v[0] for k, v in components.items()},
            'component_confidences': {k: v[1] for k, v in components.items()},
            'peer_adjustment': peer_adjustment,
            'swing_points_count': len(self.price_analyzer.swing_points),
            'support_resistance_count': len(self.price_analyzer.support_resistance)
        }
        
        # Track peer influences
        influences = {}
        for peer_name in self.peer_influences:
            if peer_name in peer_readings:
                peer_reading = peer_readings[peer_name]
                influence_strength = abs(peer_reading.value * self.peer_influences[peer_name])
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
    
    def _analyze_structure(self) -> Tuple[float, float]:
        """Analyze market structure"""
        structure, confidence = self.price_analyzer.get_market_structure()
        
        # Convert structure to directional score
        structure_scores = {
            MarketStructure.HIGHER_HIGHS_HIGHER_LOWS: 0.8,
            MarketStructure.LOWER_HIGHS_LOWER_LOWS: -0.8,
            MarketStructure.HIGHER_HIGHS_LOWER_LOWS: 0.3,  # Weakening uptrend
            MarketStructure.LOWER_HIGHS_HIGHER_LOWS: -0.3, # Weakening downtrend
            MarketStructure.CHOPPY: 0.0
        }
        
        score = structure_scores.get(structure, 0.0)
        return score, confidence
    
    def _analyze_momentum(self) -> Tuple[float, float]:
        """Analyze momentum characteristics"""
        rsi, rsi_conf = self.momentum_analyzer.calculate_rsi()
        momentum, momentum_conf = self.momentum_analyzer.calculate_momentum()
        vol_momentum, vol_conf = self.momentum_analyzer.calculate_volume_momentum()
        
        # Convert RSI to directional score
        rsi_score = (rsi - 50) / 50  # Normalize to [-1, 1]
        
        # Combine momentum indicators
        momentum_score = (rsi_score * 0.4 + momentum * 100 * 0.4 + vol_momentum * 0.2)
        momentum_score = max(-1.0, min(1.0, momentum_score))
        
        # Average confidence
        avg_confidence = np.mean([rsi_conf, momentum_conf, vol_conf])
        
        return momentum_score, avg_confidence
    
    def _analyze_support_resistance(self, current_price: float) -> Tuple[float, float]:
        """Analyze support/resistance dynamics"""
        nearby_levels = self.price_analyzer.get_nearest_levels(current_price, 0.005)
        
        if not nearby_levels:
            return 0.0, 0.0
        
        # Find strongest nearby level
        strongest_level = max(nearby_levels, key=lambda l: l.strength * l.touch_count)
        
        # Calculate score based on position relative to level
        distance = (current_price - strongest_level.price) / current_price
        
        if strongest_level.is_support:
            # Above support is bullish
            score = min(1.0, distance * 100) if distance > 0 else max(-1.0, distance * 100)
        else:
            # Below resistance is bearish
            score = max(-1.0, distance * 100) if distance < 0 else min(1.0, distance * 100)
        
        # Confidence based on level strength and recency
        recency_factor = max(0.1, 1.0 - strongest_level.recency_hours / 168)  # Week decay
        confidence = strongest_level.strength * recency_factor
        
        return score, min(1.0, confidence)
    
    def _analyze_patterns(self) -> Tuple[float, float]:
        """Analyze technical patterns"""
        pattern, confidence = self.pattern_recognizer.analyze_patterns(self.price_analyzer)
        
        # Convert pattern to directional score
        pattern_scores = {
            'ascending_triangle': 0.6,
            'descending_triangle': -0.6,
            'symmetrical_triangle': 0.0,
            'double_bottom': 0.7,
            'double_top': -0.7,
            'flag': 0.4,  # Continuation pattern
            'no_clear_pattern': 0.0,
            'insufficient_data': 0.0
        }
        
        score = pattern_scores.get(pattern, 0.0)
        return score, confidence
    
    def get_technical_summary(self) -> Dict[str, Any]:
        """Get comprehensive technical summary"""
        if not self.price_analyzer.price_history:
            return {}
        
        current_price = self.price_analyzer.price_history[-1][0]
        structure, struct_conf = self.price_analyzer.get_market_structure()
        pattern, pattern_conf = self.pattern_recognizer.analyze_patterns(self.price_analyzer)
        rsi, rsi_conf = self.momentum_analyzer.calculate_rsi()
        
        nearby_levels = self.price_analyzer.get_nearest_levels(current_price, 0.01)
        
        return {
            'current_price': current_price,
            'market_structure': {
                'type': structure.name,
                'confidence': struct_conf
            },
            'pattern': {
                'type': pattern,
                'confidence': pattern_conf
            },
            'momentum': {
                'rsi': rsi,
                'rsi_confidence': rsi_conf
            },
            'support_resistance': {
                'nearby_levels': len(nearby_levels),
                'strongest_level': nearby_levels[0].price if nearby_levels else None
            },
            'swing_points': len(self.price_analyzer.swing_points),
            'total_levels': len(self.price_analyzer.support_resistance)
        }

