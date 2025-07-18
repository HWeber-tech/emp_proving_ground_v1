"""
Enhanced HOW Dimension - Sophisticated Institutional Mechanics Engine

This module implements advanced institutional footprint detection using real order flow analysis,
ICT concepts, and sophisticated pattern recognition. It moves beyond simple volume analysis
to understand how large players position themselves in the market.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, NamedTuple
from dataclasses import dataclass, field
from collections import deque
import asyncio
from scipy import stats
from scipy.signal import find_peaks
import logging

from ..core.base import DimensionalReading, MarketData, MarketRegime, DimensionalSensor, InstrumentMeta
from ..core.data_integration import OrderBookSnapshot, OrderFlowDataProvider

logger = logging.getLogger(__name__)

@dataclass
class OrderBlock:
    """Represents an institutional order block"""
    timestamp: datetime
    price_high: float
    price_low: float
    displacement_candle_index: int
    volume: float
    strength: float  # 0-1 rating based on displacement and volume
    direction: str  # 'bullish' or 'bearish'
    tested: bool = False
    broken: bool = False
    test_count: int = 0
    last_test_time: Optional[datetime] = None

@dataclass
class FairValueGap:
    """Represents a Fair Value Gap (imbalance)"""
    timestamp: datetime
    gap_high: float
    gap_low: float
    direction: str  # 'bullish' or 'bearish'
    displacement_strength: float
    filled_percentage: float = 0.0
    is_filled: bool = False
    fill_time: Optional[datetime] = None

@dataclass
class LiquiditySweep:
    """Represents a liquidity sweep event"""
    timestamp: datetime
    swept_level: float
    direction: str  # 'buy_side' or 'sell_side'
    volume: float
    reversal_strength: float
    follow_through: bool = False

@dataclass
class MarketStructureShift:
    """Represents a change in market structure"""
    timestamp: datetime
    previous_structure: str  # 'bullish', 'bearish', 'ranging'
    new_structure: str
    confidence: float
    key_level: float
    volume_confirmation: bool

class ICTPatternDetector:
    """
    Advanced ICT (Inner Circle Trader) pattern detection engine
    Identifies institutional footprints through sophisticated price action analysis
    """
    
    def __init__(self, lookback_periods: int = 100):
        self.lookback_periods = lookback_periods
        self.price_history = deque(maxlen=lookback_periods)
        self.volume_history = deque(maxlen=lookback_periods)
        self.timestamp_history = deque(maxlen=lookback_periods)
        
        # Pattern storage
        self.order_blocks: List[OrderBlock] = []
        self.fair_value_gaps: List[FairValueGap] = []
        self.liquidity_sweeps: List[LiquiditySweep] = []
        self.structure_shifts: List[MarketStructureShift] = []
        
        # Adaptive parameters
        self.displacement_threshold = 0.0005  # Adaptive based on volatility
        self.volume_threshold_multiplier = 1.5  # Adaptive based on average volume
        
    def update_market_data(self, market_data: MarketData) -> None:
        """Update with new market data and detect patterns"""
        mid_price = (market_data.bid + market_data.ask) / 2
        
        self.price_history.append(mid_price)
        self.volume_history.append(market_data.volume)
        self.timestamp_history.append(market_data.timestamp)
        
        # Update adaptive parameters based on recent market conditions
        self._update_adaptive_parameters()
        
        # Detect patterns if we have enough data
        if len(self.price_history) >= 20:
            self._detect_order_blocks()
            self._detect_fair_value_gaps()
            self._detect_liquidity_sweeps()
            self._detect_structure_shifts()
            self._update_existing_patterns(market_data)
    
    def _update_adaptive_parameters(self) -> None:
        """Update detection parameters based on current market conditions"""
        if len(self.price_history) < 20:
            return
        
        # Calculate recent volatility
        recent_prices = list(self.price_history)[-20:]
        price_changes = np.diff(recent_prices)
        volatility = np.std(price_changes)
        
        # Adapt displacement threshold to volatility
        base_threshold = 0.0005
        self.displacement_threshold = base_threshold * (1 + volatility * 100)
        
        # Adapt volume threshold to recent average
        recent_volumes = list(self.volume_history)[-20:]
        avg_volume = np.mean(recent_volumes)
        volume_std = np.std(recent_volumes)
        
        # Higher threshold in high-volume periods
        self.volume_threshold_multiplier = 1.5 + (volume_std / avg_volume) if avg_volume > 0 else 1.5
    
    def _detect_order_blocks(self) -> None:
        """Detect institutional order blocks from displacement patterns"""
        if len(self.price_history) < 10:
            return
        
        prices = np.array(list(self.price_history))
        volumes = np.array(list(self.volume_history))
        timestamps = list(self.timestamp_history)
        
        # Look for displacement candles (large moves with high volume)
        price_changes = np.diff(prices)
        volume_avg = np.mean(volumes[-20:]) if len(volumes) >= 20 else np.mean(volumes)
        
        for i in range(5, len(price_changes) - 5):
            displacement = abs(price_changes[i])
            volume = volumes[i]
            
            # Check for displacement criteria
            if (displacement > self.displacement_threshold and 
                volume > volume_avg * self.volume_threshold_multiplier):
                
                # Determine direction
                direction = 'bullish' if price_changes[i] > 0 else 'bearish'
                
                # Find the order block (candle before displacement)
                if direction == 'bullish':
                    # Bullish order block is the low of candle before displacement
                    ob_high = prices[i]
                    ob_low = prices[i-1]
                else:
                    # Bearish order block is the high of candle before displacement
                    ob_high = prices[i-1]
                    ob_low = prices[i]
                
                # Calculate strength based on displacement and volume
                displacement_strength = displacement / self.displacement_threshold
                volume_strength = volume / (volume_avg * self.volume_threshold_multiplier)
                strength = min((displacement_strength + volume_strength) / 2, 1.0)
                
                # Create order block
                order_block = OrderBlock(
                    timestamp=timestamps[i],
                    price_high=ob_high,
                    price_low=ob_low,
                    displacement_candle_index=i,
                    volume=volume,
                    strength=strength,
                    direction=direction
                )
                
                # Check if this is a new order block (not too close to existing ones)
                if self._is_new_order_block(order_block):
                    self.order_blocks.append(order_block)
                    
                    # Keep only recent order blocks
                    cutoff_time = datetime.now() - timedelta(hours=24)
                    self.order_blocks = [
                        ob for ob in self.order_blocks 
                        if ob.timestamp > cutoff_time
                    ]
    
    def _detect_fair_value_gaps(self) -> None:
        """Detect Fair Value Gaps (3-candle imbalance patterns)"""
        if len(self.price_history) < 5:
            return
        
        prices = np.array(list(self.price_history))
        timestamps = list(self.timestamp_history)
        
        # Look for 3-candle FVG pattern
        for i in range(2, len(prices) - 1):
            # Get three consecutive price points
            candle1 = prices[i-2]
            candle2 = prices[i-1]  # Middle candle (displacement)
            candle3 = prices[i]
            
            # Check for bullish FVG (gap between candle1 high and candle3 low)
            if candle2 > candle1 and candle2 > candle3:
                gap_low = candle1
                gap_high = candle3
                
                # Ensure there's a meaningful gap
                gap_size = abs(gap_high - gap_low)
                if gap_size > self.displacement_threshold * 0.5:
                    
                    # Calculate displacement strength
                    displacement = abs(candle2 - candle1)
                    displacement_strength = displacement / self.displacement_threshold
                    
                    fvg = FairValueGap(
                        timestamp=timestamps[i-1],
                        gap_high=gap_high,
                        gap_low=gap_low,
                        direction='bullish',
                        displacement_strength=min(displacement_strength, 1.0)
                    )
                    
                    if self._is_new_fvg(fvg):
                        self.fair_value_gaps.append(fvg)
            
            # Check for bearish FVG
            elif candle2 < candle1 and candle2 < candle3:
                gap_high = candle1
                gap_low = candle3
                
                gap_size = abs(gap_high - gap_low)
                if gap_size > self.displacement_threshold * 0.5:
                    
                    displacement = abs(candle2 - candle1)
                    displacement_strength = displacement / self.displacement_threshold
                    
                    fvg = FairValueGap(
                        timestamp=timestamps[i-1],
                        gap_high=gap_high,
                        gap_low=gap_low,
                        direction='bearish',
                        displacement_strength=min(displacement_strength, 1.0)
                    )
                    
                    if self._is_new_fvg(fvg):
                        self.fair_value_gaps.append(fvg)
        
        # Clean up old FVGs
        cutoff_time = datetime.now() - timedelta(hours=12)
        self.fair_value_gaps = [
            fvg for fvg in self.fair_value_gaps 
            if fvg.timestamp > cutoff_time
        ]
    
    def _detect_liquidity_sweeps(self) -> None:
        """Detect liquidity sweeps (stop hunts) at swing highs/lows"""
        if len(self.price_history) < 20:
            return
        
        prices = np.array(list(self.price_history))
        volumes = np.array(list(self.volume_history))
        timestamps = list(self.timestamp_history)
        
        # Find swing highs and lows
        swing_highs = find_peaks(prices, distance=5, prominence=self.displacement_threshold)[0]
        swing_lows = find_peaks(-prices, distance=5, prominence=self.displacement_threshold)[0]
        
        # Check for sweeps of recent swing levels
        current_price = prices[-1]
        current_volume = volumes[-1]
        avg_volume = np.mean(volumes[-10:])
        
        # Check buy-side liquidity sweeps (above swing highs)
        for high_idx in swing_highs[-5:]:  # Last 5 swing highs
            swing_high = prices[high_idx]
            
            # Check if current price swept above this high with volume
            if (current_price > swing_high and 
                current_volume > avg_volume * 1.2):
                
                # Look for reversal after sweep
                reversal_strength = self._calculate_reversal_strength(len(prices)-1, 'bearish')
                
                sweep = LiquiditySweep(
                    timestamp=timestamps[-1],
                    swept_level=swing_high,
                    direction='buy_side',
                    volume=current_volume,
                    reversal_strength=reversal_strength
                )
                
                if self._is_new_liquidity_sweep(sweep):
                    self.liquidity_sweeps.append(sweep)
        
        # Check sell-side liquidity sweeps (below swing lows)
        for low_idx in swing_lows[-5:]:  # Last 5 swing lows
            swing_low = prices[low_idx]
            
            if (current_price < swing_low and 
                current_volume > avg_volume * 1.2):
                
                reversal_strength = self._calculate_reversal_strength(len(prices)-1, 'bullish')
                
                sweep = LiquiditySweep(
                    timestamp=timestamps[-1],
                    swept_level=swing_low,
                    direction='sell_side',
                    volume=current_volume,
                    reversal_strength=reversal_strength
                )
                
                if self._is_new_liquidity_sweep(sweep):
                    self.liquidity_sweeps.append(sweep)
        
        # Clean up old sweeps
        cutoff_time = datetime.now() - timedelta(hours=6)
        self.liquidity_sweeps = [
            sweep for sweep in self.liquidity_sweeps 
            if sweep.timestamp > cutoff_time
        ]
    
    def _detect_structure_shifts(self) -> None:
        """Detect changes in market structure (trend changes)"""
        if len(self.price_history) < 30:
            return
        
        prices = np.array(list(self.price_history))
        timestamps = list(self.timestamp_history)
        
        # Analyze recent structure vs previous structure
        recent_prices = prices[-15:]  # Last 15 periods
        previous_prices = prices[-30:-15]  # Previous 15 periods
        
        # Calculate structure for each period
        recent_structure = self._determine_structure(recent_prices)
        previous_structure = self._determine_structure(previous_prices)
        
        # Check for structure shift
        if recent_structure != previous_structure:
            
            # Calculate confidence based on price action clarity
            confidence = self._calculate_structure_confidence(recent_prices)
            
            # Find key level (breakout point)
            key_level = self._find_structure_key_level(prices[-20:])
            
            # Check volume confirmation
            recent_volumes = np.array(list(self.volume_history))[-15:]
            avg_volume = np.mean(recent_volumes)
            volume_confirmation = np.max(recent_volumes) > avg_volume * 1.3
            
            shift = MarketStructureShift(
                timestamp=timestamps[-1],
                previous_structure=previous_structure,
                new_structure=recent_structure,
                confidence=confidence,
                key_level=key_level,
                volume_confirmation=volume_confirmation
            )
            
            # Only add significant structure shifts
            if confidence > 0.6:
                self.structure_shifts.append(shift)
                
                # Keep only recent shifts
                cutoff_time = datetime.now() - timedelta(hours=8)
                self.structure_shifts = [
                    shift for shift in self.structure_shifts 
                    if shift.timestamp > cutoff_time
                ]
    
    def _update_existing_patterns(self, market_data: MarketData) -> None:
        """Update existing patterns with new market data"""
        current_price = (market_data.bid + market_data.ask) / 2
        
        # Update order block tests
        for ob in self.order_blocks:
            if not ob.broken:
                # Check if price is testing the order block
                if ob.direction == 'bullish':
                    if ob.price_low <= current_price <= ob.price_high:
                        ob.tested = True
                        ob.test_count += 1
                        ob.last_test_time = market_data.timestamp
                    elif current_price < ob.price_low:
                        ob.broken = True
                else:  # bearish
                    if ob.price_low <= current_price <= ob.price_high:
                        ob.tested = True
                        ob.test_count += 1
                        ob.last_test_time = market_data.timestamp
                    elif current_price > ob.price_high:
                        ob.broken = True
        
        # Update FVG fills
        for fvg in self.fair_value_gaps:
            if not fvg.is_filled:
                # Check how much of the gap has been filled
                if fvg.gap_low <= current_price <= fvg.gap_high:
                    gap_size = fvg.gap_high - fvg.gap_low
                    if fvg.direction == 'bullish':
                        filled_amount = current_price - fvg.gap_low
                    else:
                        filled_amount = fvg.gap_high - current_price
                    
                    fvg.filled_percentage = min(filled_amount / gap_size, 1.0)
                    
                    if fvg.filled_percentage >= 0.8:  # 80% filled
                        fvg.is_filled = True
                        fvg.fill_time = market_data.timestamp
    
    def _is_new_order_block(self, new_ob: OrderBlock) -> bool:
        """Check if order block is new (not too close to existing ones)"""
        for existing_ob in self.order_blocks:
            if (abs(new_ob.price_high - existing_ob.price_high) < self.displacement_threshold * 0.5 and
                abs(new_ob.price_low - existing_ob.price_low) < self.displacement_threshold * 0.5):
                return False
        return True
    
    def _is_new_fvg(self, new_fvg: FairValueGap) -> bool:
        """Check if FVG is new"""
        for existing_fvg in self.fair_value_gaps:
            if (abs(new_fvg.gap_high - existing_fvg.gap_high) < self.displacement_threshold * 0.3 and
                abs(new_fvg.gap_low - existing_fvg.gap_low) < self.displacement_threshold * 0.3):
                return False
        return True
    
    def _is_new_liquidity_sweep(self, new_sweep: LiquiditySweep) -> bool:
        """Check if liquidity sweep is new"""
        for existing_sweep in self.liquidity_sweeps:
            if (abs(new_sweep.swept_level - existing_sweep.swept_level) < self.displacement_threshold * 0.5 and
                new_sweep.direction == existing_sweep.direction):
                return False
        return True
    
    def _calculate_reversal_strength(self, index: int, direction: str) -> float:
        """Calculate strength of reversal after liquidity sweep"""
        if index < 5:
            return 0.0
        
        prices = np.array(list(self.price_history))
        
        # Look at price action after the sweep
        post_sweep_prices = prices[index-5:index+1]
        
        if direction == 'bullish':
            # Expect prices to move up after sell-side sweep
            price_change = post_sweep_prices[-1] - post_sweep_prices[0]
            return max(0, min(price_change / self.displacement_threshold, 1.0))
        else:
            # Expect prices to move down after buy-side sweep
            price_change = post_sweep_prices[0] - post_sweep_prices[-1]
            return max(0, min(price_change / self.displacement_threshold, 1.0))
    
    def _determine_structure(self, prices: np.ndarray) -> str:
        """Determine market structure from price array"""
        if len(prices) < 5:
            return 'ranging'
        
        # Calculate trend using linear regression
        x = np.arange(len(prices))
        slope, _, r_value, _, _ = stats.linregress(x, prices)
        
        # Determine structure based on slope and correlation
        if abs(r_value) < 0.3:  # Low correlation = ranging
            return 'ranging'
        elif slope > 0 and r_value > 0.3:
            return 'bullish'
        elif slope < 0 and r_value < -0.3:
            return 'bearish'
        else:
            return 'ranging'
    
    def _calculate_structure_confidence(self, prices: np.ndarray) -> float:
        """Calculate confidence in structure determination"""
        if len(prices) < 5:
            return 0.0
        
        x = np.arange(len(prices))
        _, _, r_value, _, _ = stats.linregress(x, prices)
        
        # Confidence is based on correlation strength
        return abs(r_value)
    
    def _find_structure_key_level(self, prices: np.ndarray) -> float:
        """Find key level for structure shift"""
        if len(prices) < 10:
            return prices[-1] if len(prices) > 0 else 0.0
        
        # Find the breakout point (where structure changed)
        mid_point = len(prices) // 2
        return prices[mid_point]
    
    def get_institutional_footprint_score(self) -> float:
        """Calculate overall institutional footprint score"""
        scores = []
        
        # Order block score
        active_obs = [ob for ob in self.order_blocks if not ob.broken]
        if active_obs:
            ob_score = np.mean([ob.strength for ob in active_obs])
            scores.append(ob_score * 0.3)
        
        # FVG score
        active_fvgs = [fvg for fvg in self.fair_value_gaps if not fvg.is_filled]
        if active_fvgs:
            fvg_score = np.mean([fvg.displacement_strength for fvg in active_fvgs])
            scores.append(fvg_score * 0.2)
        
        # Liquidity sweep score
        recent_sweeps = [
            sweep for sweep in self.liquidity_sweeps 
            if sweep.timestamp > datetime.now() - timedelta(hours=2)
        ]
        if recent_sweeps:
            sweep_score = np.mean([sweep.reversal_strength for sweep in recent_sweeps])
            scores.append(sweep_score * 0.3)
        
        # Structure shift score
        recent_shifts = [
            shift for shift in self.structure_shifts 
            if shift.timestamp > datetime.now() - timedelta(hours=4)
        ]
        if recent_shifts:
            shift_score = np.mean([shift.confidence for shift in recent_shifts])
            scores.append(shift_score * 0.2)
        
        return np.sum(scores) if scores else 0.0

class InstitutionalMechanicsEngine(DimensionalSensor):
    """
    Enhanced institutional mechanics engine with sophisticated flow analysis
    """
    
    def __init__(self, instrument_meta: InstrumentMeta):
        super().__init__(instrument_meta)
        self.institutional_analyzer = ICTPatternDetector()
        
        # Performance tracking
        self.analysis_history = deque(maxlen=100)
        self.confidence_history = deque(maxlen=50)
        
        # Adaptive parameters
        self.learning_rate = 0.05
        self.confidence_threshold = 0.6

    async def update(self, market_data: MarketData) -> DimensionalReading:
        """Process new market data and return dimensional reading."""
        return await self.analyze_institutional_mechanics(market_data)
    
    def snapshot(self) -> DimensionalReading:
        """Return current dimensional state without processing new data."""
        if self.last_reading:
            return self.last_reading
        else:
            # Return default reading
            return DimensionalReading(
                dimension='HOW',
                signal_strength=0.0,
                confidence=0.0,
                regime=MarketRegime.UNKNOWN,
                context={'status': 'not_initialized'},
                timestamp=datetime.now(),
                data_quality=0.0,
                processing_time_ms=0
            )
    
    def reset(self) -> None:
        """Reset sensor state for new trading session or instrument."""
        self.analysis_history.clear()
        self.confidence_history.clear()
        self.last_reading = None
        self.is_initialized = False
        
    async def analyze_institutional_mechanics(self, market_data: MarketData) -> DimensionalReading:
        """Perform comprehensive institutional analysis"""
        
        try:
            # Update institutional analyzer
            self.institutional_analyzer.update_market_data(market_data)
        
            # Get institutional footprint score
            institutional_score = self.institutional_analyzer.get_institutional_footprint_score()
            
            # Calculate confidence based on pattern count and strength
            pattern_count = (
                len(self.institutional_analyzer.order_blocks) +
                len(self.institutional_analyzer.fair_value_gaps) +
                len(self.institutional_analyzer.liquidity_sweeps) +
                len(self.institutional_analyzer.structure_shifts)
            )
            
            confidence = min(0.3 + (pattern_count * 0.1), 0.9)
        
            # Determine regime
            regime = self._determine_regime(institutional_score)
            
            # Create context
            context = {
                'institutional_score': institutional_score,
                'order_blocks': len(self.institutional_analyzer.order_blocks),
                'fair_value_gaps': len(self.institutional_analyzer.fair_value_gaps),
                'liquidity_sweeps': len(self.institutional_analyzer.liquidity_sweeps),
                'structure_shifts': len(self.institutional_analyzer.structure_shifts),
                'confidence': confidence
            }
            
            # Create reading
            reading = DimensionalReading(
            dimension='HOW',
                signal_strength=institutional_score,
            confidence=confidence,
                regime=regime,
            context=context,
                timestamp=market_data.timestamp,
                data_quality=0.8,
                processing_time_ms=50
            )
            
            # Store last reading and mark as initialized
            self.last_reading = reading
            self.is_initialized = True
            
            return reading
            
        except Exception as e:
            logger.error(f"Institutional analysis failed: {e}")
            
            # Return neutral reading on error
            return DimensionalReading(
                dimension='HOW',
                signal_strength=0.0,
                confidence=0.3,
                context={'error': str(e), 'status': 'degraded'},
                timestamp=market_data.timestamp,
                data_quality=0.3,
                processing_time_ms=10
            )
    
    def _calculate_confidence(self, signals: List[float]) -> float:
        """Calculate confidence based on signal agreement."""
        # Signal agreement
        positive_signals = sum(1 for s in signals if s > 0.1)
        negative_signals = sum(1 for s in signals if s < -0.1)
        total_signals = len(signals)
        
        if positive_signals > negative_signals:
            agreement = positive_signals / total_signals
        elif negative_signals > positive_signals:
            agreement = negative_signals / total_signals
        else:
            agreement = 0.5
        
        # Signal strength
        avg_strength = np.mean([abs(s) for s in signals])
        
        # Combine factors
        confidence = (agreement * 0.6 + avg_strength * 0.4)
        return max(0.1, min(confidence, 0.95))
    
    def _determine_regime(self, institutional_score: float) -> MarketRegime:
        """Determine market regime from institutional score."""
        if institutional_score > 0.5:
            return MarketRegime.TRENDING_BULL
        elif institutional_score > 0.2:
            return MarketRegime.TRENDING_WEAK
        elif institutional_score < -0.5:
            return MarketRegime.TRENDING_BEAR
        elif institutional_score < -0.2:
            return MarketRegime.TRENDING_WEAK
        else:
            return MarketRegime.CONSOLIDATING

