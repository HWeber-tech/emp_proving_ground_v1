"""
Sensory Cortex v2.2 - WHAT Dimension Engine (Market Structure Analysis)

Masterful implementation of pure price action analysis without traditional indicators.
Focuses on market structure, swing points, and momentum dynamics.
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, NamedTuple
from dataclasses import dataclass
from enum import Enum

from src.sensory.core.base import (
    DimensionalSensor, DimensionalReading, MarketData, InstrumentMeta,
    MarketRegime
)
from src.sensory.core.utils import (
    EMA, WelfordVar, compute_confidence, normalize_signal,
    calculate_momentum, PerformanceTracker
)

logger = logging.getLogger(__name__)


class SwingType(Enum):
    """Swing point types for market structure analysis."""
    SWING_HIGH = "swing_high"
    SWING_LOW = "swing_low"
    HIGHER_HIGH = "higher_high"
    LOWER_HIGH = "lower_high"
    HIGHER_LOW = "higher_low"
    LOWER_LOW = "lower_low"


@dataclass
class SwingPoint:
    """
    Swing point structure for market structure analysis.
    """
    timestamp: datetime
    price: float
    swing_type: SwingType
    strength: float  # 0-1
    confirmed: bool = False
    volume: float = 0.0


@dataclass
class TrendStructure:
    """
    Trend structure analysis result.
    """
    direction: int  # 1 for uptrend, -1 for downtrend, 0 for sideways
    strength: float  # 0-1
    quality: float   # 0-1, based on swing point quality
    last_swing_high: Optional[SwingPoint]
    last_swing_low: Optional[SwingPoint]
    structure_breaks: int


class SwingPointDetector:
    """
    Sophisticated swing point detection for market structure analysis.
    """
    
    def __init__(self, lookback_period: int = 5):
        """
        Initialize swing point detector.
        
        Args:
            lookback_period: Number of periods to look back for swing confirmation
        """
        self.lookback_period = lookback_period
        self.price_history: List[MarketData] = []
        self.swing_points: List[SwingPoint] = []
        
    def update(self, market_data: MarketData) -> List[SwingPoint]:
        """
        Update swing point detection with new market data.
        
        Args:
            market_data: Latest market data
            
        Returns:
            List of newly detected swing points
        """
        self.price_history.append(market_data)
        if len(self.price_history) > 1000:  # Maintain reasonable history
            self.price_history.pop(0)
        
        new_swings = []
        
        if len(self.price_history) >= self.lookback_period * 2 + 1:
            # Detect swing highs and lows
            swing_highs = self._detect_swing_highs()
            swing_lows = self._detect_swing_lows()
            
            new_swings.extend(swing_highs)
            new_swings.extend(swing_lows)
            
            # Classify swing relationships
            if new_swings:
                self._classify_swing_relationships(new_swings)
        
        self.swing_points.extend(new_swings)
        if len(self.swing_points) > 200:  # Maintain reasonable swing history
            self.swing_points = self.swing_points[-200:]
        
        return new_swings
    
    def _detect_swing_highs(self) -> List[SwingPoint]:
        """Detect swing high points."""
        swings = []
        
        # Check for swing high at lookback_period from the end
        check_idx = len(self.price_history) - self.lookback_period - 1
        
        if check_idx < self.lookback_period:
            return swings
        
        center_candle = self.price_history[check_idx]
        center_high = center_candle.high
        
        # Check if this is a swing high
        is_swing_high = True
        
        # Check left side
        for i in range(check_idx - self.lookback_period, check_idx):
            if self.price_history[i].high >= center_high:
                is_swing_high = False
                break
        
        # Check right side
        if is_swing_high:
            for i in range(check_idx + 1, check_idx + self.lookback_period + 1):
                if i < len(self.price_history) and self.price_history[i].high >= center_high:
                    is_swing_high = False
                    break
        
        if is_swing_high:
            # Calculate swing strength based on price rejection
            left_max = max(self.price_history[i].high for i in range(check_idx - self.lookback_period, check_idx))
            right_max = max(self.price_history[i].high for i in range(check_idx + 1, min(len(self.price_history), check_idx + self.lookback_period + 1)))
            
            strength = min(1.0, (center_high - max(left_max, right_max)) / center_high * 1000)
            
            swing = SwingPoint(
                timestamp=center_candle.timestamp,
                price=center_high,
                swing_type=SwingType.SWING_HIGH,
                strength=max(0.1, strength),
                confirmed=True,
                volume=center_candle.volume
            )
            swings.append(swing)
        
        return swings
    
    def _detect_swing_lows(self) -> List[SwingPoint]:
        """Detect swing low points."""
        swings = []
        
        # Check for swing low at lookback_period from the end
        check_idx = len(self.price_history) - self.lookback_period - 1
        
        if check_idx < self.lookback_period:
            return swings
        
        center_candle = self.price_history[check_idx]
        center_low = center_candle.low
        
        # Check if this is a swing low
        is_swing_low = True
        
        # Check left side
        for i in range(check_idx - self.lookback_period, check_idx):
            if self.price_history[i].low <= center_low:
                is_swing_low = False
                break
        
        # Check right side
        if is_swing_low:
            for i in range(check_idx + 1, check_idx + self.lookback_period + 1):
                if i < len(self.price_history) and self.price_history[i].low <= center_low:
                    is_swing_low = False
                    break
        
        if is_swing_low:
            # Calculate swing strength based on price rejection
            left_min = min(self.price_history[i].low for i in range(check_idx - self.lookback_period, check_idx))
            right_min = min(self.price_history[i].low for i in range(check_idx + 1, min(len(self.price_history), check_idx + self.lookback_period + 1)))
            
            strength = min(1.0, (min(left_min, right_min) - center_low) / center_low * 1000)
            
            swing = SwingPoint(
                timestamp=center_candle.timestamp,
                price=center_low,
                swing_type=SwingType.SWING_LOW,
                strength=max(0.1, strength),
                confirmed=True,
                volume=center_candle.volume
            )
            swings.append(swing)
        
        return swings
    
    def _classify_swing_relationships(self, new_swings: List[SwingPoint]) -> None:
        """Classify swing relationships (HH, HL, LH, LL)."""
        for swing in new_swings:
            if swing.swing_type == SwingType.SWING_HIGH:
                # Find previous swing high
                prev_high = None
                for prev_swing in reversed(self.swing_points):
                    if prev_swing.swing_type in [SwingType.SWING_HIGH, SwingType.HIGHER_HIGH, SwingType.LOWER_HIGH]:
                        prev_high = prev_swing
                        break
                
                if prev_high:
                    if swing.price > prev_high.price:
                        swing.swing_type = SwingType.HIGHER_HIGH
                    else:
                        swing.swing_type = SwingType.LOWER_HIGH
            
            elif swing.swing_type == SwingType.SWING_LOW:
                # Find previous swing low
                prev_low = None
                for prev_swing in reversed(self.swing_points):
                    if prev_swing.swing_type in [SwingType.SWING_LOW, SwingType.HIGHER_LOW, SwingType.LOWER_LOW]:
                        prev_low = prev_swing
                        break
                
                if prev_low:
                    if swing.price > prev_low.price:
                        swing.swing_type = SwingType.HIGHER_LOW
                    else:
                        swing.swing_type = SwingType.LOWER_LOW


class MomentumAnalyzer:
    """
    Pure price action momentum analysis without traditional indicators.
    """
    
    def __init__(self):
        """Initialize momentum analyzer."""
        self.velocity_tracker = EMA(14)
        self.acceleration_tracker = EMA(10)
        self.momentum_history: List[float] = []
        
    def analyze_momentum(self, price_history: List[MarketData]) -> Dict[str, float]:
        """
        Analyze momentum from pure price action.
        
        Args:
            price_history: Recent price history
            
        Returns:
            Momentum analysis results
        """
        if len(price_history) < 3:
            return {'velocity': 0.0, 'acceleration': 0.0, 'momentum_score': 0.0}
        
        # Calculate price velocity (rate of change)
        current_price = price_history[-1].close
        prev_price = price_history[-2].close
        velocity = (current_price - prev_price) / prev_price
        
        self.velocity_tracker.update(velocity)
        smoothed_velocity = self.velocity_tracker.get_value() or 0.0
        
        # Calculate acceleration (change in velocity)
        if len(self.momentum_history) > 0:
            prev_velocity = self.momentum_history[-1]
            acceleration = velocity - prev_velocity
            self.acceleration_tracker.update(acceleration)
        
        self.momentum_history.append(velocity)
        if len(self.momentum_history) > 100:
            self.momentum_history.pop(0)
        
        # Calculate momentum persistence
        persistence = self._calculate_momentum_persistence()
        
        # Calculate momentum divergence
        divergence = self._calculate_momentum_divergence(price_history)
        
        # Overall momentum score
        momentum_score = (
            abs(smoothed_velocity) * 0.4 +
            persistence * 0.3 +
            (1.0 - abs(divergence)) * 0.3
        )
        
        return {
            'velocity': smoothed_velocity,
            'acceleration': self.acceleration_tracker.get_value() or 0.0,
            'persistence': persistence,
            'divergence': divergence,
            'momentum_score': momentum_score
        }
    
    def _calculate_momentum_persistence(self) -> float:
        """Calculate momentum persistence (consistency of direction)."""
        if len(self.momentum_history) < 10:
            return 0.5
        
        recent_momentum = self.momentum_history[-10:]
        positive_count = sum(1 for m in recent_momentum if m > 0)
        negative_count = sum(1 for m in recent_momentum if m < 0)
        
        # Persistence is higher when momentum is consistently in one direction
        max_count = max(positive_count, negative_count)
        return max_count / len(recent_momentum)
    
    def _calculate_momentum_divergence(self, price_history: List[MarketData]) -> float:
        """Calculate momentum divergence from price action."""
        if len(price_history) < 20 or len(self.momentum_history) < 20:
            return 0.0
        
        # Compare recent price highs/lows with momentum highs/lows
        recent_prices = [candle.close for candle in price_history[-10:]]
        recent_momentum = self.momentum_history[-10:]
        
        price_trend = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]
        momentum_trend = recent_momentum[-1] - recent_momentum[0]
        
        # Divergence when price and momentum move in opposite directions
        if price_trend * momentum_trend < 0:
            return abs(price_trend) + abs(momentum_trend)
        else:
            return 0.0


class TrendAnalyzer:
    """
    Sophisticated trend analysis based on market structure.
    """
    
    def __init__(self):
        """Initialize trend analyzer."""
        self.trend_strength = EMA(20)
        self.structure_quality = EMA(15)
        
    def analyze_trend(self, swing_points: List[SwingPoint]) -> TrendStructure:
        """
        Analyze trend structure from swing points.
        
        Args:
            swing_points: List of swing points
            
        Returns:
            Trend structure analysis
        """
        if len(swing_points) < 4:
            return TrendStructure(
                direction=0,
                strength=0.0,
                quality=0.0,
                last_swing_high=None,
                last_swing_low=None,
                structure_breaks=0
            )
        
        # Find recent swing highs and lows
        recent_highs = [s for s in swing_points[-20:] if 'HIGH' in s.swing_type.value]
        recent_lows = [s for s in swing_points[-20:] if 'LOW' in s.swing_type.value]
        
        last_swing_high = recent_highs[-1] if recent_highs else None
        last_swing_low = recent_lows[-1] if recent_lows else None
        
        # Analyze trend direction
        direction = self._determine_trend_direction(recent_highs, recent_lows)
        
        # Calculate trend strength
        strength = self._calculate_trend_strength(recent_highs, recent_lows, direction)
        
        # Calculate structure quality
        quality = self._calculate_structure_quality(swing_points[-10:])
        
        # Count structure breaks
        structure_breaks = self._count_structure_breaks(swing_points[-20:])
        
        return TrendStructure(
            direction=direction,
            strength=strength,
            quality=quality,
            last_swing_high=last_swing_high,
            last_swing_low=last_swing_low,
            structure_breaks=structure_breaks
        )
    
    def _determine_trend_direction(self, highs: List[SwingPoint], lows: List[SwingPoint]) -> int:
        """Determine overall trend direction from swing analysis."""
        if len(highs) < 2 or len(lows) < 2:
            return 0
        
        # Count higher highs and higher lows for uptrend
        hh_count = sum(1 for h in highs if h.swing_type == SwingType.HIGHER_HIGH)
        hl_count = sum(1 for l in lows if l.swing_type == SwingType.HIGHER_LOW)
        
        # Count lower highs and lower lows for downtrend
        lh_count = sum(1 for h in highs if h.swing_type == SwingType.LOWER_HIGH)
        ll_count = sum(1 for l in lows if l.swing_type == SwingType.LOWER_LOW)
        
        uptrend_score = hh_count + hl_count
        downtrend_score = lh_count + ll_count
        
        if uptrend_score > downtrend_score and uptrend_score >= 2:
            return 1
        elif downtrend_score > uptrend_score and downtrend_score >= 2:
            return -1
        else:
            return 0
    
    def _calculate_trend_strength(self, highs: List[SwingPoint], lows: List[SwingPoint], direction: int) -> float:
        """Calculate trend strength based on swing point analysis."""
        if not highs or not lows:
            return 0.0
        
        # Calculate average swing strength
        all_swings = highs + lows
        avg_swing_strength = np.mean([s.strength for s in all_swings])
        
        # Calculate price progression consistency
        if direction == 1:  # Uptrend
            if len(highs) >= 2:
                price_progression = (highs[-1].price - highs[0].price) / highs[0].price
            else:
                price_progression = 0.0
        elif direction == -1:  # Downtrend
            if len(lows) >= 2:
                price_progression = (lows[0].price - lows[-1].price) / lows[0].price
            else:
                price_progression = 0.0
        else:
            price_progression = 0.0
        
        # Combine factors
        strength = (avg_swing_strength * 0.6 + min(1.0, abs(price_progression) * 100) * 0.4)
        
        self.trend_strength.update(strength)
        return self.trend_strength.get_value() or 0.0
    
    def _calculate_structure_quality(self, recent_swings: List[SwingPoint]) -> float:
        """Calculate quality of market structure."""
        if len(recent_swings) < 3:
            return 0.0
        
        # Quality factors
        avg_strength = np.mean([s.strength for s in recent_swings])
        confirmed_ratio = sum(1 for s in recent_swings if s.confirmed) / len(recent_swings)
        
        quality = avg_strength * confirmed_ratio
        
        self.structure_quality.update(quality)
        return self.structure_quality.get_value() or 0.0
    
    def _count_structure_breaks(self, swing_points: List[SwingPoint]) -> int:
        """Count structure breaks in recent swing points."""
        breaks = 0
        
        for i in range(1, len(swing_points)):
            current = swing_points[i]
            previous = swing_points[i-1]
            
            # Structure break when swing type changes unexpectedly
            if (current.swing_type == SwingType.LOWER_HIGH and 
                previous.swing_type == SwingType.HIGHER_HIGH):
                breaks += 1
            elif (current.swing_type == SwingType.LOWER_LOW and 
                  previous.swing_type == SwingType.HIGHER_LOW):
                breaks += 1
        
        return breaks


class WATEngine(DimensionalSensor):
    """
    Masterful WHAT dimension engine for market structure analysis.
    Implements pure price action analysis without traditional indicators.
    """
    
    def __init__(self, instrument_meta: InstrumentMeta):
        """
        Initialize WHAT engine.
        
        Args:
            instrument_meta: Instrument metadata
        """
        super().__init__(instrument_meta)
        
        # Initialize components
        self.swing_detector = SwingPointDetector(lookback_period=5)
        self.momentum_analyzer = MomentumAnalyzer()
        self.trend_analyzer = TrendAnalyzer()
        
        # Performance tracking
        self.performance_tracker = PerformanceTracker()
        
        # State variables
        self.price_history: List[MarketData] = []
        self.current_trend_structure = None
        self.structure_confidence = EMA(20)
        
        logger.info(f"WHAT Engine initialized for {instrument_meta.symbol}")
    
    async def update(self, market_data: MarketData) -> DimensionalReading:
        """
        Process market data and generate market structure analysis.
        
        Args:
            market_data: Latest market data
            
        Returns:
            Dimensional reading with market structure analysis
        """
        start_time = datetime.utcnow()
        
        try:
            # Update price history
            self.price_history.append(market_data)
            if len(self.price_history) > 500:  # Maintain reasonable history
                self.price_history.pop(0)
            
            # Detect swing points
            new_swings = self.swing_detector.update(market_data)
            
            # Analyze momentum
            momentum_analysis = self.momentum_analyzer.analyze_momentum(self.price_history)
            
            # Analyze trend structure
            trend_structure = self.trend_analyzer.analyze_trend(self.swing_detector.swing_points)
            self.current_trend_structure = trend_structure
            
            # Perform comprehensive market structure analysis
            structure_analysis = self._analyze_market_structure(
                new_swings, momentum_analysis, trend_structure, market_data
            )
            
            # Calculate signal strength and confidence
            signal_strength = self._calculate_signal_strength(structure_analysis)
            confidence = self._calculate_confidence(structure_analysis, trend_structure)
            
            # Detect market regime
            regime = self._detect_market_regime(structure_analysis, trend_structure)
            
            # Create dimensional reading
            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            reading = DimensionalReading(
                dimension="WHAT",
                timestamp=market_data.timestamp,
                signal_strength=signal_strength,
                confidence=confidence,
                regime=regime,
                context={
                    'structure_analysis': structure_analysis,
                    'momentum_analysis': momentum_analysis,
                    'trend_structure': trend_structure.__dict__,
                    'new_swings': [s.__dict__ for s in new_swings],
                    'swing_count': len(self.swing_detector.swing_points)
                },
                data_quality=1.0,  # Price data is always available
                processing_time_ms=processing_time,
                evidence=self._extract_evidence(structure_analysis, trend_structure),
                warnings=self._generate_warnings(structure_analysis, trend_structure)
            )
            
            self.last_reading = reading
            self.is_initialized = True
            
            logger.debug(f"WHAT analysis complete: signal={signal_strength:.3f}, "
                        f"confidence={confidence:.3f}, swings={len(new_swings)}")
            
            return reading
            
        except Exception as e:
            logger.error(f"Error in WHAT engine update: {e}")
            return self._create_error_reading(market_data.timestamp, str(e))
    
    def _analyze_market_structure(
        self,
        new_swings: List[SwingPoint],
        momentum_analysis: Dict[str, float],
        trend_structure: TrendStructure,
        market_data: MarketData
    ) -> Dict[str, any]:
        """
        Analyze comprehensive market structure.
        
        Args:
            new_swings: Newly detected swing points
            momentum_analysis: Momentum analysis results
            trend_structure: Trend structure analysis
            market_data: Current market data
            
        Returns:
            Market structure analysis results
        """
        analysis = {}
        
        # Swing-based analysis
        swing_analysis = self._analyze_swing_structure(new_swings, trend_structure)
        analysis['swing_structure'] = swing_analysis
        
        # Price action quality
        price_action_quality = self._assess_price_action_quality(momentum_analysis, trend_structure)
        analysis['price_action_quality'] = price_action_quality
        
        # Structure strength
        structure_strength = self._calculate_structure_strength(trend_structure, momentum_analysis)
        analysis['structure_strength'] = structure_strength
        
        # Breakout potential
        breakout_potential = self._assess_breakout_potential(trend_structure, momentum_analysis)
        analysis['breakout_potential'] = breakout_potential
        
        # Update structure confidence
        overall_confidence = (
            swing_analysis.get('quality', 0.0) * 0.4 +
            price_action_quality * 0.3 +
            structure_strength * 0.3
        )
        self.structure_confidence.update(overall_confidence)
        analysis['structure_confidence'] = self.structure_confidence.get_value() or 0.0
        
        return analysis
    
    def _analyze_swing_structure(self, new_swings: List[SwingPoint], trend_structure: TrendStructure) -> Dict[str, any]:
        """Analyze swing point structure."""
        if not new_swings:
            return {'quality': 0.0, 'direction_consistency': 0.0, 'strength': 0.0}
        
        # Calculate average swing strength
        avg_strength = np.mean([s.strength for s in new_swings])
        
        # Check direction consistency with trend
        consistent_swings = 0
        for swing in new_swings:
            if trend_structure.direction == 1:  # Uptrend
                if swing.swing_type in [SwingType.HIGHER_HIGH, SwingType.HIGHER_LOW]:
                    consistent_swings += 1
            elif trend_structure.direction == -1:  # Downtrend
                if swing.swing_type in [SwingType.LOWER_HIGH, SwingType.LOWER_LOW]:
                    consistent_swings += 1
        
        direction_consistency = consistent_swings / len(new_swings) if new_swings else 0.0
        
        return {
            'quality': avg_strength,
            'direction_consistency': direction_consistency,
            'strength': min(1.0, avg_strength * direction_consistency),
            'swing_count': len(new_swings)
        }
    
    def _assess_price_action_quality(self, momentum_analysis: Dict[str, float], trend_structure: TrendStructure) -> float:
        """Assess overall price action quality."""
        momentum_score = momentum_analysis.get('momentum_score', 0.0)
        persistence = momentum_analysis.get('persistence', 0.0)
        divergence = momentum_analysis.get('divergence', 0.0)
        structure_quality = trend_structure.quality
        
        # Quality is high when momentum is strong, persistent, and aligned with structure
        quality = (
            momentum_score * 0.3 +
            persistence * 0.3 +
            (1.0 - divergence) * 0.2 +
            structure_quality * 0.2
        )
        
        return min(1.0, quality)
    
    def _calculate_structure_strength(self, trend_structure: TrendStructure, momentum_analysis: Dict[str, float]) -> float:
        """Calculate overall market structure strength."""
        trend_strength = trend_structure.strength
        momentum_strength = momentum_analysis.get('momentum_score', 0.0)
        structure_quality = trend_structure.quality
        
        # Penalize for structure breaks
        break_penalty = min(0.5, trend_structure.structure_breaks * 0.1)
        
        strength = (
            trend_strength * 0.4 +
            momentum_strength * 0.4 +
            structure_quality * 0.2
        ) - break_penalty
        
        return max(0.0, strength)
    
    def _assess_breakout_potential(self, trend_structure: TrendStructure, momentum_analysis: Dict[str, float]) -> Dict[str, float]:
        """Assess potential for breakout from current structure."""
        # Factors that suggest breakout potential
        momentum_acceleration = abs(momentum_analysis.get('acceleration', 0.0))
        momentum_persistence = momentum_analysis.get('persistence', 0.0)
        structure_breaks = trend_structure.structure_breaks
        
        # High acceleration + high persistence + recent structure breaks = breakout potential
        breakout_score = (
            momentum_acceleration * 0.4 +
            momentum_persistence * 0.3 +
            min(1.0, structure_breaks / 3.0) * 0.3
        )
        
        # Direction of potential breakout
        momentum_direction = 1 if momentum_analysis.get('velocity', 0.0) > 0 else -1
        
        return {
            'breakout_score': min(1.0, breakout_score),
            'direction': momentum_direction,
            'confidence': momentum_persistence
        }
    
    def _calculate_signal_strength(self, analysis: Dict[str, any]) -> float:
        """Calculate overall signal strength from market structure analysis."""
        # Component signals
        structure_strength = analysis.get('structure_strength', 0.0)
        swing_strength = analysis.get('swing_structure', {}).get('strength', 0.0)
        breakout_potential = analysis.get('breakout_potential', {}).get('breakout_score', 0.0)
        
        # Direction from trend structure
        if self.current_trend_structure:
            direction = self.current_trend_structure.direction
        else:
            direction = 0
        
        # Combine signals
        signal_magnitude = (
            structure_strength * 0.5 +
            swing_strength * 0.3 +
            breakout_potential * 0.2
        )
        
        return signal_magnitude * direction
    
    def _calculate_confidence(self, analysis: Dict[str, any], trend_structure: TrendStructure) -> float:
        """Calculate confidence in market structure analysis."""
        # Confidence factors
        structure_confidence = analysis.get('structure_confidence', 0.0)
        price_action_quality = analysis.get('price_action_quality', 0.0)
        swing_consistency = analysis.get('swing_structure', {}).get('direction_consistency', 0.0)
        
        # Data availability (always high for price data)
        data_quality = 1.0
        
        return compute_confidence(
            signal_strength=abs(self._calculate_signal_strength(analysis)),
            data_quality=data_quality,
            historical_accuracy=self.performance_tracker.get_accuracy(),
            confluence_signals=[structure_confidence, price_action_quality, swing_consistency]
        )
    
    def _detect_market_regime(self, analysis: Dict[str, any], trend_structure: TrendStructure) -> MarketRegime:
        """Detect market regime from structure analysis."""
        structure_strength = analysis.get('structure_strength', 0.0)
        breakout_potential = analysis.get('breakout_potential', {}).get('breakout_score', 0.0)
        structure_breaks = trend_structure.structure_breaks
        
        # Strong trending regime
        if structure_strength > 0.7 and abs(trend_structure.direction) == 1:
            return MarketRegime.TRENDING_STRONG
        
        # Weak trending regime
        elif structure_strength > 0.4 and abs(trend_structure.direction) == 1:
            return MarketRegime.TRENDING_WEAK
        
        # Breakout regime
        elif breakout_potential > 0.6:
            return MarketRegime.BREAKOUT
        
        # Exhausted regime (many structure breaks)
        elif structure_breaks > 3:
            return MarketRegime.EXHAUSTED
        
        # Default to consolidating
        else:
            return MarketRegime.CONSOLIDATING
    
    def _extract_evidence(self, analysis: Dict[str, any], trend_structure: TrendStructure) -> Dict[str, float]:
        """Extract evidence scores for transparency."""
        evidence = {}
        
        evidence['structure_strength'] = analysis.get('structure_strength', 0.0)
        evidence['swing_quality'] = analysis.get('swing_structure', {}).get('quality', 0.0)
        evidence['price_action_quality'] = analysis.get('price_action_quality', 0.0)
        evidence['trend_strength'] = trend_structure.strength
        evidence['breakout_potential'] = analysis.get('breakout_potential', {}).get('breakout_score', 0.0)
        
        return evidence
    
    def _generate_warnings(self, analysis: Dict[str, any], trend_structure: TrendStructure) -> List[str]:
        """Generate warnings about analysis quality or concerns."""
        warnings = []
        
        # Check for structure deterioration
        if trend_structure.structure_breaks > 3:
            warnings.append(f"High structure break count: {trend_structure.structure_breaks}")
        
        # Check for low quality price action
        price_action_quality = analysis.get('price_action_quality', 0.0)
        if price_action_quality < 0.3:
            warnings.append(f"Low price action quality: {price_action_quality:.2f}")
        
        # Check for conflicting signals
        structure_strength = analysis.get('structure_strength', 0.0)
        breakout_potential = analysis.get('breakout_potential', {}).get('breakout_score', 0.0)
        
        if structure_strength > 0.6 and breakout_potential > 0.6:
            warnings.append("Conflicting signals: strong structure vs high breakout potential")
        
        return warnings
    
    def _create_error_reading(self, timestamp: datetime, error_msg: str) -> DimensionalReading:
        """Create reading when error occurs."""
        return DimensionalReading(
            dimension="WHAT",
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
            dimension="WHAT",
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
        self.price_history.clear()
        self.current_trend_structure = None
        
        # Reset components
        self.swing_detector = SwingPointDetector(lookback_period=5)
        self.momentum_analyzer = MomentumAnalyzer()
        self.trend_analyzer = TrendAnalyzer()
        self.structure_confidence = EMA(20)
        
        logger.info("WHAT Engine reset completed")

