"""
Enhanced WHAT Dimension - Pure Price Action Engine

This module implements advanced price action analysis that moves beyond traditional lagging indicators
to focus on pure market structure, momentum dynamics, and fractal patterns. It identifies what the
market is actually doing through sophisticated price behavior analysis.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, NamedTuple
from dataclasses import dataclass, field
from collections import deque
from scipy import stats
from scipy.signal import find_peaks, savgol_filter
from sklearn.preprocessing import MinMaxScaler
import logging

from ..core.base import DimensionalReading, MarketData, MarketRegime

logger = logging.getLogger(__name__)

@dataclass
class SwingPoint:
    """Represents a swing high or low in price action"""
    timestamp: datetime
    price: float
    type: str  # 'high' or 'low'
    strength: float  # 0-1 based on prominence and volume
    confirmed: bool = False
    broken: bool = False

@dataclass
class TrendStructure:
    """Represents the current trend structure"""
    direction: str  # 'bullish', 'bearish', 'ranging'
    strength: float  # 0-1
    quality: float  # 0-1 based on structure clarity
    key_levels: List[float]
    last_structure_break: Optional[datetime] = None

@dataclass
class MomentumProfile:
    """Represents momentum characteristics"""
    velocity: float  # Rate of price change
    acceleration: float  # Rate of velocity change (2nd derivative)
    persistence: float  # How long momentum has been sustained
    exhaustion_signals: List[str]  # Signs of momentum exhaustion

@dataclass
class VolumeProfile:
    """Volume analysis at different price levels"""
    price_level: float
    volume: float
    transactions: int
    value_area_high: float
    value_area_low: float
    point_of_control: float  # Price with highest volume

class PriceActionAnalyzer:
    """
    Advanced price action analyzer that identifies market structure,
    momentum dynamics, and behavioral patterns without relying on lagging indicators
    """
    
    def __init__(self, lookback_periods: int = 200):
        self.lookback_periods = lookback_periods
        
        # Price and volume history
        self.price_history = deque(maxlen=lookback_periods)
        self.high_history = deque(maxlen=lookback_periods)
        self.low_history = deque(maxlen=lookback_periods)
        self.volume_history = deque(maxlen=lookback_periods)
        self.timestamp_history = deque(maxlen=lookback_periods)
        
        # Structural elements
        self.swing_points: List[SwingPoint] = []
        self.trend_structure: Optional[TrendStructure] = None
        self.momentum_profile: Optional[MomentumProfile] = None
        
        # Adaptive parameters
        self.swing_detection_sensitivity = 0.5  # Adaptive based on volatility
        self.trend_confirmation_periods = 10
        
    def update_market_data(self, market_data: MarketData) -> None:
        """Update with new market data and analyze price action"""
        
        mid_price = (market_data.bid + market_data.ask) / 2
        spread = market_data.ask - market_data.bid
        
        # Estimate high/low from mid and spread (in real implementation, use actual OHLC)
        estimated_high = mid_price + spread * 0.6
        estimated_low = mid_price - spread * 0.6
        
        self.price_history.append(mid_price)
        self.high_history.append(estimated_high)
        self.low_history.append(estimated_low)
        self.volume_history.append(market_data.volume)
        self.timestamp_history.append(market_data.timestamp)
        
        # Update adaptive parameters
        self._update_adaptive_parameters()
        
        # Perform analysis if we have enough data
        if len(self.price_history) >= 20:
            self._detect_swing_points()
            self._analyze_trend_structure()
            self._analyze_momentum_dynamics()
            self._update_swing_confirmations()
    
    def _update_adaptive_parameters(self) -> None:
        """Update analysis parameters based on current market conditions"""
        if len(self.price_history) < 20:
            return
        
        # Calculate recent volatility
        recent_prices = np.array(list(self.price_history)[-20:])
        price_changes = np.diff(recent_prices)
        volatility = np.std(price_changes)
        
        # Adapt swing detection sensitivity to volatility
        # Higher volatility = less sensitive (avoid noise)
        # Lower volatility = more sensitive (catch smaller swings)
        base_sensitivity = 0.5
        volatility_factor = min(float(volatility * 1000), 2.0)  # Scale volatility
        self.swing_detection_sensitivity = base_sensitivity * (2.0 - volatility_factor)
        self.swing_detection_sensitivity = max(0.1, min(self.swing_detection_sensitivity, 1.0))
    
    def _detect_swing_points(self) -> None:
        """Detect swing highs and lows using advanced price action analysis"""
        if len(self.price_history) < 10:
            return
        
        highs = np.array(list(self.high_history))
        lows = np.array(list(self.low_history))
        volumes = np.array(list(self.volume_history))
        timestamps = list(self.timestamp_history)
        
        # Calculate dynamic prominence threshold based on recent volatility
        recent_range = np.max(highs[-20:]) - np.min(lows[-20:]) if len(highs) >= 20 else np.max(highs) - np.min(lows)
        prominence_threshold = recent_range * 0.01 * self.swing_detection_sensitivity
        
        # Find swing highs
        swing_high_indices = find_peaks(
            highs, 
            distance=5,  # Minimum distance between peaks
            prominence=prominence_threshold
        )[0]
        
        # Find swing lows
        swing_low_indices = find_peaks(
            -lows, 
            distance=5,
            prominence=prominence_threshold
        )[0]
        
        # Process new swing highs
        for idx in swing_high_indices:
            if idx >= len(timestamps) - 10:  # Only process recent swings
                swing_price = highs[idx]
                swing_volume = volumes[idx] if idx < len(volumes) else 0
                
                # Calculate swing strength based on prominence and volume
                if idx > 0 and idx < len(highs) - 1:
                    prominence = min(highs[idx] - highs[idx-1], highs[idx] - highs[idx+1])
                    volume_strength = min(float(swing_volume / np.mean(volumes[-20:])), 2.0) if len(volumes) >= 20 else 1.0
                    strength = min((prominence / prominence_threshold) * volume_strength / 2, 1.0)
                else:
                    strength = 0.5
                
                swing_point = SwingPoint(
                    timestamp=timestamps[idx],
                    price=float(swing_price),
                    type='high',
                    strength=strength
                )
                
                # Check if this is a new swing point
                if self._is_new_swing_point(swing_point):
                    self.swing_points.append(swing_point)
        
        # Process new swing lows
        for idx in swing_low_indices:
            if idx >= len(timestamps) - 10:
                swing_price = lows[idx]
                swing_volume = volumes[idx] if idx < len(volumes) else 0
                
                if idx > 0 and idx < len(lows) - 1:
                    prominence = min(lows[idx-1] - lows[idx], lows[idx+1] - lows[idx])
                    volume_strength = min(float(swing_volume / np.mean(volumes[-20:])), 2.0) if len(volumes) >= 20 else 1.0
                    strength = min((prominence / prominence_threshold) * volume_strength / 2, 1.0)
                else:
                    strength = 0.5
                
                swing_point = SwingPoint(
                    timestamp=timestamps[idx],
                    price=float(swing_price),
                    type='low',
                    strength=strength
                )
                
                if self._is_new_swing_point(swing_point):
                    self.swing_points.append(swing_point)
        
        # Clean up old swing points
        cutoff_time = datetime.now() - timedelta(hours=24)
        self.swing_points = [
            sp for sp in self.swing_points 
            if sp.timestamp > cutoff_time
        ]
    
    def _analyze_trend_structure(self) -> None:
        """Analyze current trend structure using swing point analysis"""
        if len(self.swing_points) < 4:
            return
        
        # Get recent swing points
        recent_swings = sorted(self.swing_points, key=lambda x: x.timestamp)[-10:]
        
        if len(recent_swings) < 4:
            return
        
        # Separate highs and lows
        swing_highs = [sp for sp in recent_swings if sp.type == 'high']
        swing_lows = [sp for sp in recent_swings if sp.type == 'low']
        
        # Analyze trend direction
        trend_direction = self._determine_trend_direction(swing_highs, swing_lows)
        trend_strength = self._calculate_trend_strength(swing_highs, swing_lows)
        trend_quality = self._calculate_trend_quality(recent_swings)
        
        # Identify key levels
        key_levels = self._identify_key_levels(swing_highs, swing_lows)
        
        # Check for structure breaks
        last_structure_break = self._detect_structure_break(recent_swings)
        
        self.trend_structure = TrendStructure(
            direction=trend_direction,
            strength=trend_strength,
            quality=trend_quality,
            key_levels=key_levels,
            last_structure_break=last_structure_break
        )
    
    def _analyze_momentum_dynamics(self) -> None:
        """Analyze momentum using velocity and acceleration"""
        if len(self.price_history) < 10:
            return
        
        prices = np.array(list(self.price_history))
        
        # Calculate velocity (1st derivative)
        velocity = self._calculate_velocity(prices)
        
        # Calculate acceleration (2nd derivative)
        acceleration = self._calculate_acceleration(prices)
        
        # Calculate momentum persistence
        persistence = self._calculate_momentum_persistence(prices)
        
        # Detect exhaustion signals
        exhaustion_signals = self._detect_momentum_exhaustion(prices, velocity, acceleration)
        
        self.momentum_profile = MomentumProfile(
            velocity=velocity,
            acceleration=acceleration,
            persistence=persistence,
            exhaustion_signals=exhaustion_signals
        )
    
    def _determine_trend_direction(self, swing_highs: List[SwingPoint], swing_lows: List[SwingPoint]) -> str:
        """Determine trend direction from swing analysis"""
        if len(swing_highs) < 2 or len(swing_lows) < 2:
            return 'ranging'
        
        # Sort by timestamp
        swing_highs = sorted(swing_highs, key=lambda x: x.timestamp)
        swing_lows = sorted(swing_lows, key=lambda x: x.timestamp)
        
        # Check for higher highs and higher lows (bullish)
        higher_highs = all(
            swing_highs[i].price > swing_highs[i-1].price 
            for i in range(1, len(swing_highs))
        )
        higher_lows = all(
            swing_lows[i].price > swing_lows[i-1].price 
            for i in range(1, len(swing_lows))
        )
        
        # Check for lower highs and lower lows (bearish)
        lower_highs = all(
            swing_highs[i].price < swing_highs[i-1].price 
            for i in range(1, len(swing_highs))
        )
        lower_lows = all(
            swing_lows[i].price < swing_lows[i-1].price 
            for i in range(1, len(swing_lows))
        )
        
        if higher_highs and higher_lows:
            return 'bullish'
        elif lower_highs and lower_lows:
            return 'bearish'
        else:
            return 'ranging'
    
    def _calculate_trend_strength(self, swing_highs: List[SwingPoint], swing_lows: List[SwingPoint]) -> float:
        """Calculate trend strength based on swing point analysis"""
        if not swing_highs or not swing_lows:
            return 0.0
        
        # Calculate average swing strength
        all_swings = swing_highs + swing_lows
        avg_strength = np.mean([sp.strength for sp in all_swings])
        
        # Calculate price momentum
        if len(self.price_history) >= 20:
            recent_prices = np.array(list(self.price_history)[-20:])
            price_momentum = abs(recent_prices[-1] - recent_prices[0]) / recent_prices[0]
            momentum_strength = min(price_momentum * 100, 1.0)
        else:
            momentum_strength = 0.5
        
        # Combine swing strength and momentum
        return float((avg_strength + momentum_strength) / 2)
    
    def _calculate_trend_quality(self, recent_swings: List[SwingPoint]) -> float:
        """Calculate trend quality based on swing consistency"""
        if len(recent_swings) < 3:
            return 0.5
        
        # Calculate consistency of swing spacing
        timestamps = [sp.timestamp for sp in recent_swings]
        time_intervals = [
            (timestamps[i] - timestamps[i-1]).total_seconds() 
            for i in range(1, len(timestamps))
        ]
        
        if len(time_intervals) > 1:
            interval_consistency = 1.0 - (np.std(time_intervals) / np.mean(time_intervals))
            interval_consistency = max(0, min(float(interval_consistency), 1.0))
        else:
            interval_consistency = 0.5
        
        # Calculate strength consistency
        strengths = [sp.strength for sp in recent_swings]
        strength_consistency = 1.0 - np.std(strengths) if len(strengths) > 1 else 0.5
        
        return float((interval_consistency + strength_consistency) / 2)
    
    def _identify_key_levels(self, swing_highs: List[SwingPoint], swing_lows: List[SwingPoint]) -> List[float]:
        """Identify key support and resistance levels"""
        key_levels = []
        
        # Add significant swing levels
        for sp in swing_highs + swing_lows:
            if sp.strength > 0.6:  # Only strong swings
                key_levels.append(sp.price)
        
        # Cluster nearby levels
        if key_levels:
            key_levels = sorted(key_levels)
            clustered_levels = []
            current_cluster = [key_levels[0]]
            
            for level in key_levels[1:]:
                if abs(level - current_cluster[-1]) < 0.001:  # Within 10 pips
                    current_cluster.append(level)
                else:
                    # Add average of current cluster
                    clustered_levels.append(np.mean(current_cluster))
                    current_cluster = [level]
            
            # Add final cluster
            clustered_levels.append(np.mean(current_cluster))
            
            return clustered_levels[:5]  # Return top 5 levels
        
        return []
    
    def _detect_structure_break(self, recent_swings: List[SwingPoint]) -> Optional[datetime]:
        """Detect when market structure has been broken"""
        if len(recent_swings) < 3:
            return None
        
        # Look for breaks in swing point patterns
        sorted_swings = sorted(recent_swings, key=lambda x: x.timestamp)
        
        for i in range(2, len(sorted_swings)):
            current_swing = sorted_swings[i]
            previous_swing = sorted_swings[i-1]
            
            # Check for structure break patterns
            if (current_swing.type == 'low' and previous_swing.type == 'high' and
                current_swing.price > previous_swing.price):
                # Bullish structure break
                return current_swing.timestamp
            elif (current_swing.type == 'high' and previous_swing.type == 'low' and
                  current_swing.price < previous_swing.price):
                # Bearish structure break
                return current_swing.timestamp
        
        return None
    
    def _calculate_velocity(self, prices: np.ndarray) -> float:
        """Calculate price velocity (1st derivative)"""
        if len(prices) < 5:
            return 0.0
        
        # Use Savitzky-Golay filter to smooth and differentiate
        try:
            velocity = savgol_filter(prices, window_length=5, polyorder=2, deriv=1)
            return float(velocity[-1])  # Return current velocity
        except:
            # Fallback to simple difference
            return float(prices[-1] - prices[-2])
    
    def _calculate_acceleration(self, prices: np.ndarray) -> float:
        """Calculate price acceleration (2nd derivative)"""
        if len(prices) < 5:
            return 0.0
        
        try:
            acceleration = savgol_filter(prices, window_length=5, polyorder=2, deriv=2)
            return float(acceleration[-1])
        except:
            # Fallback to simple second difference
            if len(prices) >= 3:
                return float(prices[-1] - 2*prices[-2] + prices[-3])
            return 0.0
    
    def _calculate_momentum_persistence(self, prices: np.ndarray) -> float:
        """Calculate how long momentum has been sustained"""
        if len(prices) < 10:
            return 0.0
        
        # Calculate directional consistency over recent periods
        price_changes = np.diff(prices[-10:])
        
        if len(price_changes) == 0:
            return 0.0
        
        # Count consecutive periods in same direction
        current_direction = 1 if price_changes[-1] > 0 else -1
        consecutive_periods = 1
        
        for i in range(len(price_changes) - 2, -1, -1):
            change_direction = 1 if price_changes[i] > 0 else -1
            if change_direction == current_direction:
                consecutive_periods += 1
            else:
                break
        
        # Normalize to 0-1 scale
        return min(consecutive_periods / 10, 1.0)
    
    def _detect_momentum_exhaustion(self, prices: np.ndarray, velocity: float, acceleration: float) -> List[str]:
        """Detect signs of momentum exhaustion"""
        exhaustion_signals = []
        
        if len(prices) < 10:
            return exhaustion_signals
        
        # Divergence between price and momentum
        recent_prices = prices[-5:]
        if len(recent_prices) >= 5:
            price_trend = recent_prices[-1] - recent_prices[0]
            if (price_trend > 0 and velocity < 0) or (price_trend < 0 and velocity > 0):
                exhaustion_signals.append('momentum_divergence')
        
        # Deceleration in strong trend
        if abs(velocity) > 0.001 and acceleration * velocity < 0:  # Velocity and acceleration opposite signs
            exhaustion_signals.append('deceleration')
        
        # Extreme velocity readings
        if len(self.price_history) >= 20:
            recent_velocities = [
                self._calculate_velocity(np.array(list(self.price_history)[i:i+5]))
                for i in range(len(self.price_history) - 20, len(self.price_history) - 4)
            ]
            if recent_velocities:
                velocity_std = np.std(recent_velocities)
                if abs(velocity) > np.mean(recent_velocities) + 2 * velocity_std:
                    exhaustion_signals.append('extreme_velocity')
        
        return exhaustion_signals
    
    def _is_new_swing_point(self, new_swing: SwingPoint) -> bool:
        """Check if swing point is new (not too close to existing ones)"""
        for existing_swing in self.swing_points:
            if (existing_swing.type == new_swing.type and
                abs(existing_swing.price - new_swing.price) < 0.0005 and  # Within 5 pips
                abs((existing_swing.timestamp - new_swing.timestamp).total_seconds()) < 3600):  # Within 1 hour
                return False
        return True
    
    def _update_swing_confirmations(self) -> None:
        """Update swing point confirmations based on subsequent price action"""
        current_price = list(self.price_history)[-1] if self.price_history else 0
        
        for swing in self.swing_points:
            if not swing.confirmed and not swing.broken:
                time_since_swing = datetime.now() - swing.timestamp
                
                # Confirm swing if enough time has passed without being broken
                if time_since_swing > timedelta(minutes=30):
                    if swing.type == 'high' and current_price < swing.price * 0.999:
                        swing.confirmed = True
                    elif swing.type == 'low' and current_price > swing.price * 1.001:
                        swing.confirmed = True
                
                # Mark as broken if price has moved significantly past the swing
                if swing.type == 'high' and current_price > swing.price * 1.001:
                    swing.broken = True
                elif swing.type == 'low' and current_price < swing.price * 0.999:
                    swing.broken = True
    
    def get_price_action_score(self) -> float:
        """Calculate overall price action strength score"""
        scores = []
        
        # Trend structure score
        if self.trend_structure:
            structure_score = self.trend_structure.strength * self.trend_structure.quality
            scores.append(structure_score * 0.4)
        
        # Momentum score
        if self.momentum_profile:
            momentum_score = (
                min(abs(self.momentum_profile.velocity) * 1000, 1.0) * 0.4 +
                self.momentum_profile.persistence * 0.3 +
                (1.0 - len(self.momentum_profile.exhaustion_signals) * 0.2) * 0.3
            )
            scores.append(momentum_score * 0.4)
        
        # Swing point quality score
        if self.swing_points:
            confirmed_swings = [sp for sp in self.swing_points if sp.confirmed]
            if confirmed_swings:
                swing_score = np.mean([sp.strength for sp in confirmed_swings])
                scores.append(swing_score * 0.2)
        
        return np.sum(scores) if scores else 0.0

class TechnicalRealityEngine:
    """
    Main engine for pure price action analysis
    Focuses on what the market is actually doing rather than lagging indicators
    """
    
    def __init__(self):
        self.price_analyzer = PriceActionAnalyzer()
        
        # Market regime detection
        self.current_regime = MarketRegime.RANGING_LOW_VOL
        self.regime_confidence = 0.0
        
        # Pattern recognition
        self.active_patterns = []
        
    async def analyze_technical_reality(self, market_data: MarketData) -> DimensionalReading:
        """Analyze technical reality and return dimensional reading"""
        
        # Update price action analysis
        self.price_analyzer.update_market_data(market_data)
        
        # Detect market regime
        self._detect_market_regime()
        
        # Calculate technical strength
        price_action_score = self.price_analyzer.get_price_action_score()
        
        # Analyze market structure clarity
        structure_clarity = self._calculate_structure_clarity()
        
        # Detect pattern formations
        pattern_strength = self._analyze_pattern_formations()
        
        # Combine components
        technical_strength = (
            price_action_score * 0.5 +
            structure_clarity * 0.3 +
            pattern_strength * 0.2
        )
        
        # Calculate confidence
        confidence = self._calculate_technical_confidence()
        
        # Generate context
        context = self._generate_technical_context()
        
        return DimensionalReading(
            dimension='WHAT',
            signal_strength=technical_strength,
            confidence=confidence,
            regime=self.current_regime,  # Use the detected regime
            context=context,
            timestamp=market_data.timestamp
        )
    
    def _detect_market_regime(self) -> None:
        """Detect current market regime based on price action"""
        if not self.price_analyzer.trend_structure:
            self.current_regime = MarketRegime.UNKNOWN
            self.regime_confidence = 0.0
            return
        
        trend = self.price_analyzer.trend_structure
        
        # Determine regime based on trend characteristics
        if trend.direction == 'bullish' and trend.strength > 0.6:
            self.current_regime = MarketRegime.TRENDING_BULL
            self.regime_confidence = trend.strength * trend.quality
        elif trend.direction == 'bearish' and trend.strength > 0.6:
            self.current_regime = MarketRegime.TRENDING_BEAR
            self.regime_confidence = trend.strength * trend.quality
        elif trend.direction == 'ranging' or trend.strength < 0.4:
            self.current_regime = MarketRegime.RANGING_LOW_VOL
            self.regime_confidence = 1.0 - trend.strength
        else:
            self.current_regime = MarketRegime.TRANSITION
            self.regime_confidence = 0.5
    
    def _calculate_structure_clarity(self) -> float:
        """Calculate how clear the market structure is"""
        if not self.price_analyzer.trend_structure:
            return 0.0
        
        trend = self.price_analyzer.trend_structure
        
        # Structure clarity based on:
        # 1. Trend quality
        # 2. Number of confirmed swing points
        # 3. Key level definition
        
        quality_score = trend.quality
        
        confirmed_swings = [sp for sp in self.price_analyzer.swing_points if sp.confirmed]
        swing_score = min(len(confirmed_swings) / 5, 1.0)  # Normalize to 5 swings
        
        key_level_score = min(len(trend.key_levels) / 3, 1.0)  # Normalize to 3 levels
        
        return (quality_score * 0.5 + swing_score * 0.3 + key_level_score * 0.2)
    
    def _analyze_pattern_formations(self) -> float:
        """Analyze classical and advanced pattern formations"""
        
        # Simplified pattern analysis
        # In production, this would include:
        # - Triangle patterns
        # - Flag/pennant patterns
        # - Head and shoulders
        # - Double tops/bottoms
        # - Fractal patterns
        
        if len(self.price_analyzer.swing_points) < 3:
            return 0.0
        
        # Look for basic pattern formations
        pattern_score = 0.0
        
        # Check for double top/bottom patterns
        highs = [sp for sp in self.price_analyzer.swing_points if sp.type == 'high']
        lows = [sp for sp in self.price_analyzer.swing_points if sp.type == 'low']
        
        if len(highs) >= 2:
            recent_highs = sorted(highs, key=lambda x: x.timestamp)[-2:]
            if abs(recent_highs[0].price - recent_highs[1].price) < 0.001:  # Similar levels
                pattern_score += 0.3
        
        if len(lows) >= 2:
            recent_lows = sorted(lows, key=lambda x: x.timestamp)[-2:]
            if abs(recent_lows[0].price - recent_lows[1].price) < 0.001:  # Similar levels
                pattern_score += 0.3
        
        # Check for trend continuation patterns
        if self.price_analyzer.trend_structure:
            if (self.price_analyzer.trend_structure.direction in ['bullish', 'bearish'] and
                self.price_analyzer.trend_structure.strength > 0.5):
                pattern_score += 0.4
        
        return min(pattern_score, 1.0)
    
    def _calculate_technical_confidence(self) -> float:
        """Calculate confidence in technical analysis"""
        
        confidence_factors = []
        
        # Data quality (amount of price history)
        data_quality = min(len(self.price_analyzer.price_history) / 100, 1.0)
        confidence_factors.append(data_quality * 0.3)
        
        # Swing point confirmation rate
        if self.price_analyzer.swing_points:
            confirmed_rate = len([sp for sp in self.price_analyzer.swing_points if sp.confirmed]) / len(self.price_analyzer.swing_points)
            confidence_factors.append(confirmed_rate * 0.3)
        
        # Trend structure quality
        if self.price_analyzer.trend_structure:
            confidence_factors.append(self.price_analyzer.trend_structure.quality * 0.2)
        
        # Momentum consistency
        if self.price_analyzer.momentum_profile:
            momentum_consistency = 1.0 - len(self.price_analyzer.momentum_profile.exhaustion_signals) * 0.2
            confidence_factors.append(max(0, momentum_consistency) * 0.2)
        
        return np.sum(confidence_factors) if confidence_factors else 0.5
    
    def _generate_technical_context(self) -> Dict[str, Any]:
        """Generate contextual information about technical analysis"""
        
        context = {
            'market_regime': self.current_regime.name,
            'regime_confidence': self.regime_confidence,
            'swing_points_count': len(self.price_analyzer.swing_points),
            'confirmed_swings': len([sp for sp in self.price_analyzer.swing_points if sp.confirmed])
        }
        
        # Add trend structure info
        if self.price_analyzer.trend_structure:
            trend = self.price_analyzer.trend_structure
            context['trend'] = {
                'direction': trend.direction,
                'strength': trend.strength,
                'quality': trend.quality,
                'key_levels_count': len(trend.key_levels)
            }
        
        # Add momentum info
        if self.price_analyzer.momentum_profile:
            momentum = self.price_analyzer.momentum_profile
            context['momentum'] = {
                'velocity': momentum.velocity,
                'acceleration': momentum.acceleration,
                'persistence': momentum.persistence,
                'exhaustion_signals': momentum.exhaustion_signals
            }
        
        # Add recent swing info
        if self.price_analyzer.swing_points:
            recent_swings = sorted(self.price_analyzer.swing_points, key=lambda x: x.timestamp)[-3:]
            context['recent_swings'] = [
                {
                    'type': sp.type,
                    'price': sp.price,
                    'strength': sp.strength,
                    'confirmed': sp.confirmed
                }
                for sp in recent_swings
            ]
        
        return context

# Example usage
async def main():
    """Example usage of the enhanced technical reality engine"""
    
    # Initialize engine
    engine = TechnicalRealityEngine()
    
    # Simulate market data updates with realistic price movement
    base_price = 1.0950
    trend = 0.0001  # Small upward trend
    
    for i in range(200):
        # Create realistic price movement
        noise = np.random.normal(0, 0.0003)
        trend_component = trend * (i / 200)
        price_change = trend_component + noise
        
        current_price = base_price + price_change
        
        market_data = MarketData(
            timestamp=datetime.now() + timedelta(minutes=i),
            bid=current_price - 0.0001,
            ask=current_price + 0.0001,
            volume=1000 + np.random.exponential(500),
            volatility=0.008 + np.random.normal(0, 0.002)
        )
        
        # Analyze technical reality
        reading = await engine.analyze_technical_reality(market_data)
        
        if i % 50 == 0:  # Print every 50th reading
            print(f"Technical Reality Reading (Period {i}):")
            print(f"  Value: {reading.signal_strength:.3f}")
            print(f"  Confidence: {reading.confidence:.3f}")
            print(f"  Market Regime: {reading.context.get('market_regime', 'Unknown')}")
            if 'trend' in reading.context:
                trend_info = reading.context['trend']
                print(f"  Trend: {trend_info['direction']} (strength: {trend_info['strength']:.3f})")
            print()

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())

