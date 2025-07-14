"""
HOW Dimension - Institutional Mechanics

This dimension understands how institutions and large players move markets:
- Order flow analysis and volume delta
- Liquidity analysis and market depth
- ICT concepts: Order blocks, Fair Value Gaps, Liquidity sweeps
- Algorithmic trading pattern detection
- Market maker vs. taker behavior
- Institutional footprint analysis

The HOW dimension reveals the mechanics behind price movements,
showing not just what happened, but how it happened.
"""

import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple, NamedTuple
from collections import deque, defaultdict
from dataclasses import dataclass
from enum import Enum, auto
import math

from ..core.base import (
    DimensionalSensor, DimensionalReading, MarketData, MarketRegime
)


class OrderType(Enum):
    MARKET_BUY = auto()
    MARKET_SELL = auto()
    LIMIT_BUY = auto()
    LIMIT_SELL = auto()
    STOP_BUY = auto()
    STOP_SELL = auto()


class LiquidityEvent(Enum):
    SWEEP_HIGHS = auto()
    SWEEP_LOWS = auto()
    LIQUIDITY_GRAB = auto()
    STOP_HUNT = auto()
    NORMAL_FLOW = auto()


@dataclass
class OrderBlock:
    """ICT Order Block structure"""
    price_high: float
    price_low: float
    timestamp: datetime
    is_bullish: bool
    displacement_strength: float
    volume: int
    tested: bool = False
    test_count: int = 0
    
    @property
    def price_center(self) -> float:
        return (self.price_high + self.price_low) / 2
    
    @property
    def price_range(self) -> float:
        return self.price_high - self.price_low
    
    @property
    def age_hours(self) -> float:
        return (datetime.now() - self.timestamp).total_seconds() / 3600
    
    def is_price_in_block(self, price: float, tolerance: float = 0.0) -> bool:
        """Check if price is within order block"""
        return (self.price_low - tolerance) <= price <= (self.price_high + tolerance)


@dataclass
class FairValueGap:
    """ICT Fair Value Gap structure"""
    gap_high: float
    gap_low: float
    timestamp: datetime
    is_bullish: bool  # True for bullish FVG (gap up), False for bearish FVG (gap down)
    displacement_strength: float
    filled: bool = False
    fill_percentage: float = 0.0
    
    @property
    def gap_size(self) -> float:
        return self.gap_high - self.gap_low
    
    @property
    def gap_center(self) -> float:
        return (self.gap_high + self.gap_low) / 2
    
    @property
    def age_hours(self) -> float:
        return (datetime.now() - self.timestamp).total_seconds() / 3600
    
    def calculate_fill_percentage(self, current_price: float) -> float:
        """Calculate how much of the gap has been filled"""
        if self.is_bullish:
            # Bullish gap fills from top down
            if current_price >= self.gap_high:
                return 0.0
            elif current_price <= self.gap_low:
                return 1.0
            else:
                return (self.gap_high - current_price) / self.gap_size
        else:
            # Bearish gap fills from bottom up
            if current_price <= self.gap_low:
                return 0.0
            elif current_price >= self.gap_high:
                return 1.0
            else:
                return (current_price - self.gap_low) / self.gap_size


@dataclass
class LiquidityPool:
    """Liquidity concentration area"""
    price: float
    liquidity_estimate: float
    timestamp: datetime
    is_buy_side: bool  # True for buy-side liquidity (above highs), False for sell-side (below lows)
    swept: bool = False
    sweep_timestamp: Optional[datetime] = None
    
    @property
    def age_hours(self) -> float:
        return (datetime.now() - self.timestamp).total_seconds() / 3600


class ICTAnalyzer:
    """Analyzes ICT (Inner Circle Trader) concepts"""
    
    def __init__(self, lookback_periods: int = 200):
        self.lookback_periods = lookback_periods
        self.price_history: deque = deque(maxlen=lookback_periods)
        
        # ICT structures
        self.order_blocks: List[OrderBlock] = []
        self.fair_value_gaps: List[FairValueGap] = []
        self.liquidity_pools: List[LiquidityPool] = []
        
        # Market structure tracking
        self.swing_highs: deque = deque(maxlen=50)
        self.swing_lows: deque = deque(maxlen=50)
        
        # Displacement tracking
        self.displacement_threshold = 0.002  # 20 pips for EURUSD
        
    def update(self, data: MarketData) -> None:
        """Update ICT analysis with new market data"""
        price_data = {
            'timestamp': data.timestamp,
            'open': data.mid_price,  # Simplified - using mid as OHLC
            'high': data.mid_price + data.spread / 2,
            'low': data.mid_price - data.spread / 2,
            'close': data.mid_price,
            'volume': data.volume
        }
        
        self.price_history.append(price_data)
        
        if len(self.price_history) >= 3:
            self._detect_swing_points()
            self._detect_order_blocks()
            self._detect_fair_value_gaps()
            self._update_liquidity_pools()
            self._check_liquidity_sweeps(data.mid_price)
            self._cleanup_old_structures()
    
    def _detect_swing_points(self) -> None:
        """Detect swing highs and lows"""
        if len(self.price_history) < 5:
            return
        
        # Get recent price data
        recent_data = list(self.price_history)[-5:]
        
        # Check middle point for swing
        middle_idx = 2
        middle_data = recent_data[middle_idx]
        
        # Swing high detection
        if (middle_data['high'] > recent_data[0]['high'] and
            middle_data['high'] > recent_data[1]['high'] and
            middle_data['high'] > recent_data[3]['high'] and
            middle_data['high'] > recent_data[4]['high']):
            
            swing_high = {
                'price': middle_data['high'],
                'timestamp': middle_data['timestamp'],
                'volume': middle_data['volume']
            }
            
            # Only add if significantly different from last swing high
            if not self.swing_highs or abs(swing_high['price'] - self.swing_highs[-1]['price']) > swing_high['price'] * 0.001:
                self.swing_highs.append(swing_high)
        
        # Swing low detection
        if (middle_data['low'] < recent_data[0]['low'] and
            middle_data['low'] < recent_data[1]['low'] and
            middle_data['low'] < recent_data[3]['low'] and
            middle_data['low'] < recent_data[4]['low']):
            
            swing_low = {
                'price': middle_data['low'],
                'timestamp': middle_data['timestamp'],
                'volume': middle_data['volume']
            }
            
            if not self.swing_lows or abs(swing_low['price'] - self.swing_lows[-1]['price']) > swing_low['price'] * 0.001:
                self.swing_lows.append(swing_low)
    
    def _detect_order_blocks(self) -> None:
        """Detect ICT Order Blocks"""
        if len(self.price_history) < 10:
            return
        
        recent_data = list(self.price_history)[-10:]
        
        # Look for displacement moves
        for i in range(1, len(recent_data) - 1):
            current = recent_data[i]
            previous = recent_data[i - 1]
            next_candle = recent_data[i + 1]
            
            # Calculate displacement
            displacement = abs(current['close'] - previous['close']) / previous['close']
            
            if displacement > self.displacement_threshold:
                # Bullish displacement (strong move up)
                if current['close'] > previous['close']:
                    # Order block is the last down candle before displacement
                    if i >= 2:
                        ob_candle = recent_data[i - 2]
                        if ob_candle['close'] < ob_candle['open']:  # Down candle
                            order_block = OrderBlock(
                                price_high=ob_candle['high'],
                                price_low=ob_candle['low'],
                                timestamp=ob_candle['timestamp'],
                                is_bullish=True,
                                displacement_strength=displacement,
                                volume=ob_candle['volume']
                            )
                            self._add_order_block(order_block)
                
                # Bearish displacement (strong move down)
                elif current['close'] < previous['close']:
                    # Order block is the last up candle before displacement
                    if i >= 2:
                        ob_candle = recent_data[i - 2]
                        if ob_candle['close'] > ob_candle['open']:  # Up candle
                            order_block = OrderBlock(
                                price_high=ob_candle['high'],
                                price_low=ob_candle['low'],
                                timestamp=ob_candle['timestamp'],
                                is_bullish=False,
                                displacement_strength=displacement,
                                volume=ob_candle['volume']
                            )
                            self._add_order_block(order_block)
    
    def _add_order_block(self, new_block: OrderBlock) -> None:
        """Add order block, avoiding duplicates"""
        # Check for overlapping blocks
        for existing_block in self.order_blocks:
            if (abs(new_block.price_center - existing_block.price_center) < new_block.price_range and
                abs((new_block.timestamp - existing_block.timestamp).total_seconds()) < 3600):  # Within 1 hour
                return  # Skip duplicate
        
        self.order_blocks.append(new_block)
        
        # Keep only recent blocks
        self.order_blocks = [ob for ob in self.order_blocks if ob.age_hours < 168]  # 1 week
    
    def _detect_fair_value_gaps(self) -> None:
        """Detect ICT Fair Value Gaps"""
        if len(self.price_history) < 3:
            return
        
        recent_data = list(self.price_history)[-3:]
        
        # 3-candle FVG pattern
        candle1, candle2, candle3 = recent_data
        
        # Bullish FVG: Gap between candle1 high and candle3 low
        if (candle2['close'] > candle2['open'] and  # Middle candle is bullish
            candle3['low'] > candle1['high']):  # Gap exists
            
            gap_size = candle3['low'] - candle1['high']
            displacement = abs(candle2['close'] - candle2['open']) / candle2['open']
            
            if gap_size > 0 and displacement > self.displacement_threshold * 0.5:
                fvg = FairValueGap(
                    gap_high=candle3['low'],
                    gap_low=candle1['high'],
                    timestamp=candle2['timestamp'],
                    is_bullish=True,
                    displacement_strength=displacement
                )
                self._add_fair_value_gap(fvg)
        
        # Bearish FVG: Gap between candle1 low and candle3 high
        elif (candle2['close'] < candle2['open'] and  # Middle candle is bearish
              candle3['high'] < candle1['low']):  # Gap exists
            
            gap_size = candle1['low'] - candle3['high']
            displacement = abs(candle2['close'] - candle2['open']) / candle2['open']
            
            if gap_size > 0 and displacement > self.displacement_threshold * 0.5:
                fvg = FairValueGap(
                    gap_high=candle1['low'],
                    gap_low=candle3['high'],
                    timestamp=candle2['timestamp'],
                    is_bullish=False,
                    displacement_strength=displacement
                )
                self._add_fair_value_gap(fvg)
    
    def _add_fair_value_gap(self, new_fvg: FairValueGap) -> None:
        """Add fair value gap, avoiding duplicates"""
        # Check for overlapping FVGs
        for existing_fvg in self.fair_value_gaps:
            if (abs(new_fvg.gap_center - existing_fvg.gap_center) < new_fvg.gap_size and
                abs((new_fvg.timestamp - existing_fvg.timestamp).total_seconds()) < 1800):  # Within 30 minutes
                return  # Skip duplicate
        
        self.fair_value_gaps.append(new_fvg)
        
        # Keep only recent FVGs
        self.fair_value_gaps = [fvg for fvg in self.fair_value_gaps if fvg.age_hours < 72]  # 3 days
    
    def _update_liquidity_pools(self) -> None:
        """Update liquidity pools based on swing points"""
        # Create liquidity pools at swing highs and lows
        for swing_high in list(self.swing_highs)[-5:]:  # Recent swing highs
            # Buy-side liquidity above swing highs
            pool = LiquidityPool(
                price=swing_high['price'],
                liquidity_estimate=swing_high['volume'] / 1000.0,
                timestamp=swing_high['timestamp'],
                is_buy_side=True
            )
            
            # Check if pool already exists
            existing = False
            for existing_pool in self.liquidity_pools:
                if (existing_pool.is_buy_side and 
                    abs(existing_pool.price - pool.price) < pool.price * 0.001):
                    existing = True
                    break
            
            if not existing:
                self.liquidity_pools.append(pool)
        
        for swing_low in list(self.swing_lows)[-5:]:  # Recent swing lows
            # Sell-side liquidity below swing lows
            pool = LiquidityPool(
                price=swing_low['price'],
                liquidity_estimate=swing_low['volume'] / 1000.0,
                timestamp=swing_low['timestamp'],
                is_buy_side=False
            )
            
            # Check if pool already exists
            existing = False
            for existing_pool in self.liquidity_pools:
                if (not existing_pool.is_buy_side and 
                    abs(existing_pool.price - pool.price) < pool.price * 0.001):
                    existing = True
                    break
            
            if not existing:
                self.liquidity_pools.append(pool)
        
        # Clean old pools
        self.liquidity_pools = [pool for pool in self.liquidity_pools if pool.age_hours < 168]  # 1 week
    
    def _check_liquidity_sweeps(self, current_price: float) -> None:
        """Check for liquidity sweeps"""
        tolerance = current_price * 0.0005  # 5 pips tolerance
        
        for pool in self.liquidity_pools:
            if pool.swept:
                continue
            
            # Check buy-side liquidity sweep (price above swing high)
            if pool.is_buy_side and current_price > pool.price + tolerance:
                pool.swept = True
                pool.sweep_timestamp = datetime.now()
            
            # Check sell-side liquidity sweep (price below swing low)
            elif not pool.is_buy_side and current_price < pool.price - tolerance:
                pool.swept = True
                pool.sweep_timestamp = datetime.now()
    
    def _cleanup_old_structures(self) -> None:
        """Clean up old ICT structures"""
        current_time = datetime.now()
        
        # Remove old order blocks
        self.order_blocks = [
            ob for ob in self.order_blocks 
            if (current_time - ob.timestamp).total_seconds() < 7 * 24 * 3600  # 1 week
        ]
        
        # Remove old FVGs
        self.fair_value_gaps = [
            fvg for fvg in self.fair_value_gaps 
            if (current_time - fvg.timestamp).total_seconds() < 3 * 24 * 3600  # 3 days
        ]
        
        # Remove old liquidity pools
        self.liquidity_pools = [
            pool for pool in self.liquidity_pools 
            if (current_time - pool.timestamp).total_seconds() < 7 * 24 * 3600  # 1 week
        ]
    
    def get_nearest_order_blocks(self, current_price: float, max_distance: float = 0.01) -> List[OrderBlock]:
        """Get order blocks near current price"""
        nearby_blocks = []
        
        for block in self.order_blocks:
            distance = min(
                abs(block.price_high - current_price) / current_price,
                abs(block.price_low - current_price) / current_price
            )
            
            if distance <= max_distance:
                nearby_blocks.append(block)
        
        # Sort by distance
        nearby_blocks.sort(key=lambda b: min(
            abs(b.price_high - current_price),
            abs(b.price_low - current_price)
        ))
        
        return nearby_blocks
    
    def get_unfilled_fvgs(self) -> List[FairValueGap]:
        """Get unfilled fair value gaps"""
        return [fvg for fvg in self.fair_value_gaps if not fvg.filled]
    
    def get_recent_liquidity_sweeps(self, hours: float = 24.0) -> List[LiquidityPool]:
        """Get recent liquidity sweeps"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        return [
            pool for pool in self.liquidity_pools 
            if pool.swept and pool.sweep_timestamp and pool.sweep_timestamp > cutoff_time
        ]


class OrderFlowAnalyzer:
    """Analyzes order flow and volume characteristics"""
    
    def __init__(self):
        self.volume_history: deque = deque(maxlen=100)
        self.price_volume_data: deque = deque(maxlen=100)
        
        # Volume profile data
        self.volume_profile: Dict[float, float] = defaultdict(float)
        self.profile_resolution = 0.0001  # 1 pip resolution for EURUSD
        
    def update(self, price: float, volume: int, is_aggressive_buy: bool) -> None:
        """Update order flow analysis"""
        timestamp = datetime.now()
        
        volume_data = {
            'timestamp': timestamp,
            'price': price,
            'volume': volume,
            'is_aggressive_buy': is_aggressive_buy,
            'volume_delta': volume if is_aggressive_buy else -volume
        }
        
        self.volume_history.append(volume_data)
        self.price_volume_data.append((price, volume))
        
        # Update volume profile
        price_level = round(price / self.profile_resolution) * self.profile_resolution
        self.volume_profile[price_level] += volume
        
        # Clean old volume profile data
        if len(self.volume_profile) > 1000:
            # Keep only recent price levels
            current_price = price
            relevant_levels = {
                level: vol for level, vol in self.volume_profile.items()
                if abs(level - current_price) / current_price < 0.02  # Within 2%
            }
            self.volume_profile = defaultdict(float, relevant_levels)
    
    def calculate_volume_delta(self, periods: int = 20) -> Tuple[float, float]:
        """Calculate cumulative volume delta"""
        if len(self.volume_history) < periods:
            return 0.0, 0.0
        
        recent_data = list(self.volume_history)[-periods:]
        volume_delta = sum(data['volume_delta'] for data in recent_data)
        total_volume = sum(abs(data['volume_delta']) for data in recent_data)
        
        if total_volume > 0:
            normalized_delta = volume_delta / total_volume
            confidence = min(1.0, len(recent_data) / periods)
        else:
            normalized_delta = 0.0
            confidence = 0.0
        
        return normalized_delta, confidence
    
    def detect_algorithmic_activity(self) -> Tuple[float, float]:
        """Detect algorithmic trading patterns"""
        if len(self.volume_history) < 10:
            return 0.0, 0.0
        
        recent_data = list(self.volume_history)[-10:]
        
        # Check for TWAP-like patterns (consistent volume)
        volumes = [data['volume'] for data in recent_data]
        volume_consistency = 1.0 - (np.std(volumes) / np.mean(volumes)) if np.mean(volumes) > 0 else 0.0
        
        # Check for iceberg patterns (large volume at specific price levels)
        price_volume_consistency = self._check_iceberg_pattern(recent_data)
        
        # Combine signals
        algo_score = (volume_consistency * 0.6 + price_volume_consistency * 0.4)
        confidence = min(1.0, len(recent_data) / 10)
        
        return max(0.0, min(1.0, algo_score)), confidence
    
    def _check_iceberg_pattern(self, data: List[Dict]) -> float:
        """Check for iceberg order patterns"""
        if len(data) < 5:
            return 0.0
        
        # Group by price levels
        price_groups = defaultdict(list)
        for item in data:
            price_level = round(item['price'] / self.profile_resolution) * self.profile_resolution
            price_groups[price_level].append(item['volume'])
        
        # Check for repeated large volumes at same price
        iceberg_score = 0.0
        for price_level, volumes in price_groups.items():
            if len(volumes) >= 3:  # At least 3 hits at same level
                volume_consistency = 1.0 - (np.std(volumes) / np.mean(volumes)) if np.mean(volumes) > 0 else 0.0
                iceberg_score = max(iceberg_score, volume_consistency)
        
        return iceberg_score
    
    def get_volume_weighted_price(self, periods: int = 20) -> float:
        """Calculate volume weighted average price"""
        if len(self.price_volume_data) < periods:
            return 0.0
        
        recent_data = list(self.price_volume_data)[-periods:]
        
        total_volume = sum(volume for price, volume in recent_data)
        if total_volume > 0:
            vwap = sum(price * volume for price, volume in recent_data) / total_volume
            return vwap
        
        return 0.0


class HowDimension(DimensionalSensor):
    """
    HOW Dimension - Institutional Mechanics
    
    Understands how institutions and large players move markets:
    - ICT concepts: Order blocks, Fair Value Gaps, Liquidity sweeps
    - Order flow analysis and volume delta
    - Algorithmic trading pattern detection
    - Market maker vs. taker behavior
    - Institutional footprint analysis
    """
    
    def __init__(self):
        super().__init__("HOW")
        
        # Component analyzers
        self.ict_analyzer = ICTAnalyzer()
        self.order_flow_analyzer = OrderFlowAnalyzer()
        
        # Synthesis weights
        self.component_weights = {
            'ict_signals': 0.40,
            'order_flow': 0.35,
            'algorithmic_activity': 0.25
        }
        
        # Peer influence weights
        self.peer_influences = {
            'why': 0.15,    # Fundamentals drive institutional positioning
            'what': 0.25,   # Technical levels guide institutional execution
            'when': 0.20,   # Timing affects institutional activity
            'anomaly': 0.10 # Anomalies can disrupt institutional patterns
        }
    
    def process(self, data: MarketData, peer_readings: Dict[str, DimensionalReading]) -> DimensionalReading:
        """Process market data to understand institutional mechanics"""
        
        # Update analyzers
        self.ict_analyzer.update(data)
        
        # Estimate aggressive buy/sell based on price movement and spread
        # This is simplified - in real implementation, would use tick data
        is_aggressive_buy = data.bid > data.ask  # Simplified heuristic
        self.order_flow_analyzer.update(data.mid_price, data.volume, is_aggressive_buy)
        
        # Calculate component scores
        ict_score, ict_conf = self._analyze_ict_signals(data.mid_price)
        flow_score, flow_conf = self._analyze_order_flow()
        algo_score, algo_conf = self._analyze_algorithmic_activity()
        
        # Weighted synthesis
        components = {
            'ict_signals': (ict_score, ict_conf),
            'order_flow': (flow_score, flow_conf),
            'algorithmic_activity': (algo_score, algo_conf)
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
        final_score = base_score + peer_adjustment * 0.3  # Moderate peer influence
        final_confidence = base_confidence + peer_confidence_boost * 0.2
        
        # Normalize
        final_score = max(-1.0, min(1.0, final_score))
        final_confidence = max(0.0, min(1.0, final_confidence))
        
        # Build context
        nearby_obs = self.ict_analyzer.get_nearest_order_blocks(data.mid_price, 0.01)
        unfilled_fvgs = self.ict_analyzer.get_unfilled_fvgs()
        recent_sweeps = self.ict_analyzer.get_recent_liquidity_sweeps(6.0)
        
        context = {
            'nearby_order_blocks': len(nearby_obs),
            'unfilled_fvgs': len(unfilled_fvgs),
            'recent_liquidity_sweeps': len(recent_sweeps),
            'volume_delta': self.order_flow_analyzer.calculate_volume_delta()[0],
            'algorithmic_activity': algo_score,
            'component_scores': {k: v[0] for k, v in components.items()},
            'component_confidences': {k: v[1] for k, v in components.items()},
            'peer_adjustment': peer_adjustment,
            'total_order_blocks': len(self.ict_analyzer.order_blocks),
            'total_fvgs': len(self.ict_analyzer.fair_value_gaps),
            'total_liquidity_pools': len(self.ict_analyzer.liquidity_pools)
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
    
    def _analyze_ict_signals(self, current_price: float) -> Tuple[float, float]:
        """Analyze ICT signals for directional bias"""
        score = 0.0
        confidence = 0.0
        signal_count = 0
        
        # Order block analysis
        nearby_obs = self.ict_analyzer.get_nearest_order_blocks(current_price, 0.005)
        if nearby_obs:
            strongest_ob = max(nearby_obs, key=lambda ob: ob.displacement_strength)
            
            # Score based on order block type and position
            if strongest_ob.is_bullish and current_price >= strongest_ob.price_low:
                score += 0.6  # Bullish order block support
            elif not strongest_ob.is_bullish and current_price <= strongest_ob.price_high:
                score -= 0.6  # Bearish order block resistance
            
            confidence += 0.3
            signal_count += 1
        
        # Fair Value Gap analysis
        unfilled_fvgs = self.ict_analyzer.get_unfilled_fvgs()
        for fvg in unfilled_fvgs[:3]:  # Top 3 nearest FVGs
            distance = abs(fvg.gap_center - current_price) / current_price
            
            if distance < 0.01:  # Within 1%
                if fvg.is_bullish:
                    score += 0.4 * (1.0 - distance * 100)  # Closer = stronger signal
                else:
                    score -= 0.4 * (1.0 - distance * 100)
                
                confidence += 0.2
                signal_count += 1
        
        # Liquidity sweep analysis
        recent_sweeps = self.ict_analyzer.get_recent_liquidity_sweeps(2.0)  # Last 2 hours
        for sweep in recent_sweeps:
            if sweep.is_buy_side:
                score -= 0.5  # Buy-side sweep often leads to reversal down
            else:
                score += 0.5  # Sell-side sweep often leads to reversal up
            
            confidence += 0.25
            signal_count += 1
        
        # Normalize
        if signal_count > 0:
            score = score / signal_count
            confidence = min(1.0, confidence)
        
        return max(-1.0, min(1.0, score)), confidence
    
    def _analyze_order_flow(self) -> Tuple[float, float]:
        """Analyze order flow characteristics"""
        volume_delta, delta_conf = self.order_flow_analyzer.calculate_volume_delta()
        
        # Volume delta indicates buying/selling pressure
        flow_score = volume_delta  # Already normalized
        flow_confidence = delta_conf
        
        return flow_score, flow_confidence
    
    def _analyze_algorithmic_activity(self) -> Tuple[float, float]:
        """Analyze algorithmic trading activity"""
        algo_activity, algo_conf = self.order_flow_analyzer.detect_algorithmic_activity()
        
        # High algorithmic activity can indicate institutional presence
        # Convert to directional score (neutral for now, could be enhanced)
        algo_score = 0.0  # Algorithmic activity is directionally neutral
        
        return algo_score, algo_conf
    
    def get_ict_summary(self) -> Dict[str, Any]:
        """Get comprehensive ICT analysis summary"""
        current_price = 0.0
        if self.ict_analyzer.price_history:
            current_price = self.ict_analyzer.price_history[-1]['close']
        
        nearby_obs = self.ict_analyzer.get_nearest_order_blocks(current_price, 0.01)
        unfilled_fvgs = self.ict_analyzer.get_unfilled_fvgs()
        recent_sweeps = self.ict_analyzer.get_recent_liquidity_sweeps(24.0)
        
        return {
            'current_price': current_price,
            'order_blocks': {
                'total': len(self.ict_analyzer.order_blocks),
                'nearby': len(nearby_obs),
                'nearest': {
                    'price_range': f"{nearby_obs[0].price_low:.5f}-{nearby_obs[0].price_high:.5f}",
                    'is_bullish': nearby_obs[0].is_bullish,
                    'displacement_strength': nearby_obs[0].displacement_strength
                } if nearby_obs else None
            },
            'fair_value_gaps': {
                'total': len(self.ict_analyzer.fair_value_gaps),
                'unfilled': len(unfilled_fvgs),
                'nearest_unfilled': {
                    'gap_range': f"{unfilled_fvgs[0].gap_low:.5f}-{unfilled_fvgs[0].gap_high:.5f}",
                    'is_bullish': unfilled_fvgs[0].is_bullish,
                    'age_hours': unfilled_fvgs[0].age_hours
                } if unfilled_fvgs else None
            },
            'liquidity_sweeps': {
                'recent_count': len(recent_sweeps),
                'buy_side_sweeps': len([s for s in recent_sweeps if s.is_buy_side]),
                'sell_side_sweeps': len([s for s in recent_sweeps if not s.is_buy_side])
            },
            'order_flow': {
                'volume_delta': self.order_flow_analyzer.calculate_volume_delta()[0],
                'algorithmic_activity': self.order_flow_analyzer.detect_algorithmic_activity()[0]
            }
        }

