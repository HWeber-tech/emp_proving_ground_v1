"""
Sensory Cortex v2.2 - HOW Dimension Engine (Institutional Mechanics)

Masterful implementation of institutional flow analysis using ICT concepts.
Implements sophisticated order flow analysis and smart money detection.
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, NamedTuple
from dataclasses import dataclass
from enum import Enum

from ..core.base import (
    DimensionalSensor, DimensionalReading, MarketData, InstrumentMeta,
    MarketRegime, OrderBookSnapshot, OrderBookLevel
)
from ..core.utils import (
    EMA, WelfordVar, compute_confidence, normalize_signal,
    calculate_momentum, PerformanceTracker, exponential_decay
)

logger = logging.getLogger(__name__)


class ICTPatternType(Enum):
    """ICT pattern types for institutional analysis."""
    ORDER_BLOCK = "order_block"
    FAIR_VALUE_GAP = "fair_value_gap"
    LIQUIDITY_SWEEP = "liquidity_sweep"
    BREAKER_BLOCK = "breaker_block"
    DISPLACEMENT = "displacement"
    MARKET_STRUCTURE_SHIFT = "market_structure_shift"
    OPTIMAL_TRADE_ENTRY = "optimal_trade_entry"
    INDUCEMENT = "inducement"


@dataclass
class ICTPattern:
    """
    ICT pattern detection result.
    """
    pattern_type: ICTPatternType
    timestamp: datetime
    price_level: float
    strength: float  # 0-1
    direction: int   # 1 for bullish, -1 for bearish
    confidence: float  # 0-1
    context: Dict[str, any]


@dataclass
class OrderBlock:
    """
    Order block structure for institutional analysis.
    """
    timestamp: datetime
    high: float
    low: float
    open: float
    close: float
    volume: float
    direction: int  # 1 for bullish, -1 for bearish
    strength: float
    tested: bool = False
    broken: bool = False


@dataclass
class FairValueGap:
    """
    Fair Value Gap (FVG) structure.
    """
    timestamp: datetime
    gap_high: float
    gap_low: float
    direction: int  # 1 for bullish, -1 for bearish
    filled: bool = False
    fill_percentage: float = 0.0


class MockBookProvider:
    """
    Mock order book provider for Level 2 data simulation.
    Generates plausible order book structure for algorithm development.
    """
    
    def __init__(self, symbol: str):
        """
        Initialize mock book provider.
        
        Args:
            symbol: Trading symbol
        """
        self.symbol = symbol
        self.last_price = 1.0850  # Default for EURUSD
        self.spread = 0.0001
        
    def generate_book_snapshot(self, market_data: MarketData) -> OrderBookSnapshot:
        """
        Generate realistic order book snapshot.
        
        Args:
            market_data: Current market data
            
        Returns:
            Mock order book snapshot
        """
        self.last_price = market_data.mid_price
        
        # Generate bid levels (descending prices)
        bids = []
        for i in range(10):
            price = market_data.bid - (i * 0.0001)
            volume = np.random.exponential(1000000) * (1 + np.random.normal(0, 0.1))
            order_count = max(1, int(volume / 100000))
            bids.append(OrderBookLevel(price=price, volume=volume, order_count=order_count))
        
        # Generate ask levels (ascending prices)
        asks = []
        for i in range(10):
            price = market_data.ask + (i * 0.0001)
            volume = np.random.exponential(1000000) * (1 + np.random.normal(0, 0.1))
            order_count = max(1, int(volume / 100000))
            asks.append(OrderBookLevel(price=price, volume=volume, order_count=order_count))
        
        return OrderBookSnapshot(
            symbol=self.symbol,
            timestamp=market_data.timestamp,
            bids=bids,
            asks=asks
        )


class ICTPatternDetector:
    """
    Sophisticated ICT pattern detection engine.
    Implements Inner Circle Trader concepts for institutional analysis.
    """
    
    def __init__(self):
        """Initialize ICT pattern detector."""
        self.price_history: List[MarketData] = []
        self.order_blocks: List[OrderBlock] = []
        self.fair_value_gaps: List[FairValueGap] = []
        self.detected_patterns: List[ICTPattern] = []
        
    def update(self, market_data: MarketData) -> List[ICTPattern]:
        """
        Update pattern detection with new market data.
        
        Args:
            market_data: Latest market data
            
        Returns:
            List of newly detected patterns
        """
        self.price_history.append(market_data)
        if len(self.price_history) > 1000:  # Maintain reasonable history
            self.price_history.pop(0)
        
        new_patterns = []
        
        if len(self.price_history) >= 3:
            # Detect Fair Value Gaps
            fvg_patterns = self._detect_fair_value_gaps()
            new_patterns.extend(fvg_patterns)
            
            # Detect Order Blocks
            ob_patterns = self._detect_order_blocks()
            new_patterns.extend(ob_patterns)
            
            # Detect Displacement
            displacement_patterns = self._detect_displacement()
            new_patterns.extend(displacement_patterns)
            
            # Detect Market Structure Shifts
            if len(self.price_history) >= 20:
                mss_patterns = self._detect_market_structure_shift()
                new_patterns.extend(mss_patterns)
        
        self.detected_patterns.extend(new_patterns)
        if len(self.detected_patterns) > 500:  # Maintain reasonable pattern history
            self.detected_patterns = self.detected_patterns[-500:]
        
        return new_patterns
    
    def _detect_fair_value_gaps(self) -> List[ICTPattern]:
        """Detect Fair Value Gaps (FVG) in price action."""
        patterns = []
        
        if len(self.price_history) < 3:
            return patterns
        
        # Check last 3 candles for FVG
        candle1 = self.price_history[-3]
        candle2 = self.price_history[-2]
        candle3 = self.price_history[-1]
        
        # Bullish FVG: candle1.low > candle3.high (gap between them)
        if candle1.low > candle3.high:
            gap_strength = (candle1.low - candle3.high) / candle2.mid_price
            if gap_strength > 0.0001:  # Minimum gap size
                fvg = FairValueGap(
                    timestamp=candle2.timestamp,
                    gap_high=candle1.low,
                    gap_low=candle3.high,
                    direction=1
                )
                self.fair_value_gaps.append(fvg)
                
                patterns.append(ICTPattern(
                    pattern_type=ICTPatternType.FAIR_VALUE_GAP,
                    timestamp=candle2.timestamp,
                    price_level=(candle1.low + candle3.high) / 2,
                    strength=min(1.0, gap_strength * 10000),  # Normalize
                    direction=1,
                    confidence=0.7,
                    context={'gap_high': candle1.low, 'gap_low': candle3.high}
                ))
        
        # Bearish FVG: candle1.high < candle3.low (gap between them)
        elif candle1.high < candle3.low:
            gap_strength = (candle3.low - candle1.high) / candle2.mid_price
            if gap_strength > 0.0001:  # Minimum gap size
                fvg = FairValueGap(
                    timestamp=candle2.timestamp,
                    gap_high=candle3.low,
                    gap_low=candle1.high,
                    direction=-1
                )
                self.fair_value_gaps.append(fvg)
                
                patterns.append(ICTPattern(
                    pattern_type=ICTPatternType.FAIR_VALUE_GAP,
                    timestamp=candle2.timestamp,
                    price_level=(candle3.low + candle1.high) / 2,
                    strength=min(1.0, gap_strength * 10000),  # Normalize
                    direction=-1,
                    confidence=0.7,
                    context={'gap_high': candle3.low, 'gap_low': candle1.high}
                ))
        
        return patterns
    
    def _detect_order_blocks(self) -> List[ICTPattern]:
        """Detect Order Blocks in price action."""
        patterns = []
        
        if len(self.price_history) < 10:
            return patterns
        
        # Look for strong moves followed by consolidation
        recent_candles = self.price_history[-10:]
        
        for i in range(1, len(recent_candles) - 1):
            current = recent_candles[i]
            prev_candle = recent_candles[i-1]
            next_candle = recent_candles[i+1]
            
            # Bullish Order Block: Strong up move from this candle
            body_size = abs(current.close - current.open)
            candle_range = current.high - current.low
            
            if body_size > candle_range * 0.7:  # Strong body
                # Check for subsequent move away
                move_away = abs(next_candle.close - current.close) / current.mid_price
                
                if move_away > 0.0005:  # Significant move away
                    direction = 1 if current.close > current.open else -1
                    
                    order_block = OrderBlock(
                        timestamp=current.timestamp,
                        high=current.high,
                        low=current.low,
                        open=current.open,
                        close=current.close,
                        volume=current.volume,
                        direction=direction,
                        strength=min(1.0, move_away * 2000)  # Normalize
                    )
                    self.order_blocks.append(order_block)
                    
                    patterns.append(ICTPattern(
                        pattern_type=ICTPatternType.ORDER_BLOCK,
                        timestamp=current.timestamp,
                        price_level=current.close if direction == 1 else current.open,
                        strength=order_block.strength,
                        direction=direction,
                        confidence=0.8,
                        context={
                            'high': current.high,
                            'low': current.low,
                            'body_strength': body_size / candle_range
                        }
                    ))
        
        return patterns
    
    def _detect_displacement(self) -> List[ICTPattern]:
        """Detect displacement moves (strong institutional moves)."""
        patterns = []
        
        if len(self.price_history) < 5:
            return patterns
        
        # Check for strong moves over multiple candles
        recent_candles = self.price_history[-5:]
        
        # Calculate total move
        start_price = recent_candles[0].open
        end_price = recent_candles[-1].close
        total_move = abs(end_price - start_price) / start_price
        
        # Check if move is significant and consistent
        if total_move > 0.002:  # 20 pips for EURUSD
            # Check consistency (most candles in same direction)
            direction_votes = []
            for candle in recent_candles:
                if candle.close > candle.open:
                    direction_votes.append(1)
                elif candle.close < candle.open:
                    direction_votes.append(-1)
                else:
                    direction_votes.append(0)
            
            # Determine overall direction
            direction = 1 if sum(direction_votes) > 0 else -1
            consistency = abs(sum(direction_votes)) / len(direction_votes)
            
            if consistency > 0.6:  # At least 60% consistency
                patterns.append(ICTPattern(
                    pattern_type=ICTPatternType.DISPLACEMENT,
                    timestamp=recent_candles[-1].timestamp,
                    price_level=end_price,
                    strength=min(1.0, total_move * 500),  # Normalize
                    direction=direction,
                    confidence=consistency,
                    context={
                        'total_move': total_move,
                        'consistency': consistency,
                        'candle_count': len(recent_candles)
                    }
                ))
        
        return patterns
    
    def _detect_market_structure_shift(self) -> List[ICTPattern]:
        """Detect Market Structure Shifts (MSS)."""
        patterns = []
        
        if len(self.price_history) < 20:
            return patterns
        
        # Analyze swing highs and lows
        recent_data = self.price_history[-20:]
        highs = [candle.high for candle in recent_data]
        lows = [candle.low for candle in recent_data]
        
        # Find recent swing high and low
        swing_high_idx = np.argmax(highs[-10:]) + len(highs) - 10
        swing_low_idx = np.argmin(lows[-10:]) + len(lows) - 10
        
        swing_high = highs[swing_high_idx]
        swing_low = lows[swing_low_idx]
        current_price = recent_data[-1].close
        
        # Bullish MSS: Break above recent swing high
        if current_price > swing_high:
            strength = (current_price - swing_high) / swing_high
            patterns.append(ICTPattern(
                pattern_type=ICTPatternType.MARKET_STRUCTURE_SHIFT,
                timestamp=recent_data[-1].timestamp,
                price_level=swing_high,
                strength=min(1.0, strength * 1000),
                direction=1,
                confidence=0.8,
                context={'swing_high': swing_high, 'break_level': current_price}
            ))
        
        # Bearish MSS: Break below recent swing low
        elif current_price < swing_low:
            strength = (swing_low - current_price) / swing_low
            patterns.append(ICTPattern(
                pattern_type=ICTPatternType.MARKET_STRUCTURE_SHIFT,
                timestamp=recent_data[-1].timestamp,
                price_level=swing_low,
                strength=min(1.0, strength * 1000),
                direction=-1,
                confidence=0.8,
                context={'swing_low': swing_low, 'break_level': current_price}
            ))
        
        return patterns


class OrderFlowAnalyzer:
    """
    Sophisticated order flow analysis for institutional detection.
    """
    
    def __init__(self):
        """Initialize order flow analyzer."""
        self.volume_profile: Dict[float, float] = {}
        self.imbalance_detector = EMA(10)
        self.liquidity_tracker = WelfordVar()
        
    def analyze_order_flow(self, book_snapshot: OrderBookSnapshot) -> Dict[str, float]:
        """
        Analyze order flow from Level 2 data.
        
        Args:
            book_snapshot: Order book snapshot
            
        Returns:
            Order flow analysis results
        """
        # Calculate bid-ask imbalance
        total_bid_volume = sum(level.volume for level in book_snapshot.bids[:5])  # Top 5 levels
        total_ask_volume = sum(level.volume for level in book_snapshot.asks[:5])
        
        if total_bid_volume + total_ask_volume > 0:
            imbalance = (total_bid_volume - total_ask_volume) / (total_bid_volume + total_ask_volume)
        else:
            imbalance = 0.0
        
        self.imbalance_detector.update(imbalance)
        
        # Calculate liquidity depth
        bid_depth = sum(level.volume for level in book_snapshot.bids)
        ask_depth = sum(level.volume for level in book_snapshot.asks)
        total_depth = bid_depth + ask_depth
        
        self.liquidity_tracker.update(total_depth)
        mean_liquidity, var_liquidity, std_liquidity = self.liquidity_tracker.get_stats()
        
        # Detect large orders (icebergs)
        large_order_threshold = mean_liquidity * 2 if mean_liquidity > 0 else 1000000
        large_bid_orders = sum(1 for level in book_snapshot.bids if level.volume > large_order_threshold)
        large_ask_orders = sum(1 for level in book_snapshot.asks if level.volume > large_order_threshold)
        
        # Calculate spread quality
        spread_bps = (book_snapshot.spread / book_snapshot.mid_price) * 10000
        
        return {
            'signed_imbalance': imbalance,
            'imbalance_ema': self.imbalance_detector.get_value() or 0.0,
            'liquidity_depth': total_depth,
            'liquidity_zscore': (total_depth - mean_liquidity) / std_liquidity if std_liquidity > 0 else 0.0,
            'large_orders_ratio': (large_bid_orders + large_ask_orders) / len(book_snapshot.bids + book_snapshot.asks),
            'spread_bps': spread_bps,
            'bid_ask_ratio': bid_depth / ask_depth if ask_depth > 0 else 1.0
        }


class HOWEngine(DimensionalSensor):
    """
    Masterful HOW dimension engine for institutional mechanics analysis.
    Implements sophisticated ICT concepts and order flow analysis.
    """
    
    def __init__(self, instrument_meta: InstrumentMeta):
        """
        Initialize HOW engine.
        
        Args:
            instrument_meta: Instrument metadata
        """
        super().__init__(instrument_meta)
        
        # Initialize components
        self.ict_detector = ICTPatternDetector()
        self.order_flow_analyzer = OrderFlowAnalyzer()
        self.mock_book_provider = MockBookProvider(instrument_meta.symbol)
        
        # Performance tracking
        self.performance_tracker = PerformanceTracker()
        
        # State variables
        self.institutional_bias = EMA(50)  # Long-term institutional bias
        self.smart_money_flow = EMA(20)    # Smart money flow direction
        self.pattern_confluence = 0.0      # Current pattern confluence
        
        logger.info(f"HOW Engine initialized for {instrument_meta.symbol}")
    
    async def update(self, market_data: MarketData) -> DimensionalReading:
        """
        Process market data and generate institutional mechanics analysis.
        
        Args:
            market_data: Latest market data
            
        Returns:
            Dimensional reading with institutional analysis
        """
        start_time = datetime.utcnow()
        
        try:
            # Generate mock order book for analysis
            book_snapshot = self.mock_book_provider.generate_book_snapshot(market_data)
            
            # Detect ICT patterns
            new_patterns = self.ict_detector.update(market_data)
            
            # Analyze order flow
            order_flow_analysis = self.order_flow_analyzer.analyze_order_flow(book_snapshot)
            
            # Perform institutional analysis
            institutional_analysis = self._analyze_institutional_activity(
                new_patterns, order_flow_analysis, market_data
            )
            
            # Calculate signal strength and confidence
            signal_strength = self._calculate_signal_strength(institutional_analysis)
            confidence = self._calculate_confidence(institutional_analysis, new_patterns)
            
            # Detect market regime
            regime = self._detect_market_regime(institutional_analysis)
            
            # Create dimensional reading
            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            reading = DimensionalReading(
                dimension="HOW",
                timestamp=market_data.timestamp,
                signal_strength=signal_strength,
                confidence=confidence,
                regime=regime,
                context={
                    'institutional_analysis': institutional_analysis,
                    'order_flow': order_flow_analysis,
                    'new_patterns': [p.__dict__ for p in new_patterns],
                    'pattern_count': len(self.ict_detector.detected_patterns)
                },
                data_quality=1.0,  # Mock data is always available
                processing_time_ms=processing_time,
                evidence=self._extract_evidence(institutional_analysis, new_patterns),
                warnings=self._generate_warnings(institutional_analysis)
            )
            
            self.last_reading = reading
            self.is_initialized = True
            
            logger.debug(f"HOW analysis complete: signal={signal_strength:.3f}, "
                        f"confidence={confidence:.3f}, patterns={len(new_patterns)}")
            
            return reading
            
        except Exception as e:
            logger.exception(f"Error in HOW engine update: {e}")
            return self._create_error_reading(market_data.timestamp, str(e))
    
    def _analyze_institutional_activity(
        self,
        new_patterns: List[ICTPattern],
        order_flow: Dict[str, float],
        market_data: MarketData
    ) -> Dict[str, any]:
        """
        Analyze institutional activity from patterns and order flow.
        
        Args:
            new_patterns: Newly detected ICT patterns
            order_flow: Order flow analysis results
            market_data: Current market data
            
        Returns:
            Institutional activity analysis
        """
        analysis = {}
        
        # Pattern-based analysis
        pattern_signals = self._analyze_pattern_signals(new_patterns)
        analysis['pattern_signals'] = pattern_signals
        
        # Order flow institutional signals
        institutional_flow = self._detect_institutional_flow(order_flow)
        analysis['institutional_flow'] = institutional_flow
        
        # Smart money detection
        smart_money_activity = self._detect_smart_money(pattern_signals, institutional_flow)
        analysis['smart_money'] = smart_money_activity
        
        # Update institutional bias
        combined_signal = (
            pattern_signals.get('overall_signal', 0.0) * 0.6 +
            institutional_flow.get('flow_direction', 0.0) * 0.4
        )
        self.institutional_bias.update(combined_signal)
        analysis['institutional_bias'] = self.institutional_bias.get_value() or 0.0
        
        # Pattern confluence analysis
        confluence = self._calculate_pattern_confluence(new_patterns)
        self.pattern_confluence = confluence
        analysis['pattern_confluence'] = confluence
        
        return analysis
    
    def _analyze_pattern_signals(self, patterns: List[ICTPattern]) -> Dict[str, any]:
        """Analyze signals from ICT patterns."""
        if not patterns:
            return {'overall_signal': 0.0, 'pattern_count': 0, 'strongest_pattern': None}
        
        # Weight patterns by type and strength
        pattern_weights = {
            ICTPatternType.ORDER_BLOCK: 0.9,
            ICTPatternType.FAIR_VALUE_GAP: 0.7,
            ICTPatternType.DISPLACEMENT: 0.8,
            ICTPatternType.MARKET_STRUCTURE_SHIFT: 1.0,
            ICTPatternType.LIQUIDITY_SWEEP: 0.6,
            ICTPatternType.BREAKER_BLOCK: 0.8
        }
        
        weighted_signals = []
        strongest_pattern = None
        max_weighted_strength = 0.0
        
        for pattern in patterns:
            weight = pattern_weights.get(pattern.pattern_type, 0.5)
            weighted_strength = pattern.strength * weight * pattern.confidence
            signal = pattern.direction * weighted_strength
            weighted_signals.append(signal)
            
            if weighted_strength > max_weighted_strength:
                max_weighted_strength = weighted_strength
                strongest_pattern = pattern
        
        overall_signal = np.mean(weighted_signals) if weighted_signals else 0.0
        
        return {
            'overall_signal': overall_signal,
            'pattern_count': len(patterns),
            'strongest_pattern': strongest_pattern.pattern_type.value if strongest_pattern else None,
            'max_strength': max_weighted_strength
        }
    
    def _detect_institutional_flow(self, order_flow: Dict[str, float]) -> Dict[str, any]:
        """Detect institutional flow from order flow data."""
        # Institutional flow indicators
        imbalance = order_flow.get('signed_imbalance', 0.0)
        imbalance_ema = order_flow.get('imbalance_ema', 0.0)
        liquidity_zscore = order_flow.get('liquidity_zscore', 0.0)
        large_orders_ratio = order_flow.get('large_orders_ratio', 0.0)
        
        # Flow direction based on sustained imbalance
        flow_direction = normalize_signal(imbalance_ema, -0.5, 0.5)
        
        # Institutional activity score
        institutional_activity = (
            abs(imbalance) * 0.3 +
            abs(liquidity_zscore) * 0.3 +
            large_orders_ratio * 0.4
        )
        
        # Update smart money flow
        self.smart_money_flow.update(flow_direction)
        
        return {
            'flow_direction': flow_direction,
            'institutional_activity': min(1.0, institutional_activity),
            'smart_money_flow': self.smart_money_flow.get_value() or 0.0,
            'liquidity_anomaly': abs(liquidity_zscore) > 2.0
        }
    
    def _detect_smart_money(self, pattern_signals: Dict[str, any], institutional_flow: Dict[str, any]) -> Dict[str, any]:
        """Detect smart money activity."""
        # Smart money indicators
        pattern_strength = pattern_signals.get('max_strength', 0.0)
        flow_strength = institutional_flow.get('institutional_activity', 0.0)
        flow_direction = institutional_flow.get('flow_direction', 0.0)
        
        # Smart money activity when strong patterns align with institutional flow
        pattern_signal = pattern_signals.get('overall_signal', 0.0)
        signal_alignment = 1.0 if (pattern_signal * flow_direction) > 0 else 0.0
        
        smart_money_strength = (
            pattern_strength * 0.4 +
            flow_strength * 0.4 +
            signal_alignment * 0.2
        )
        
        return {
            'activity_strength': smart_money_strength,
            'direction': 1 if flow_direction > 0 else -1 if flow_direction < 0 else 0,
            'pattern_flow_alignment': signal_alignment,
            'confidence': min(1.0, smart_money_strength * 1.2)
        }
    
    def _calculate_pattern_confluence(self, patterns: List[ICTPattern]) -> float:
        """Calculate confluence between multiple patterns."""
        if len(patterns) < 2:
            return 0.0
        
        # Check for patterns pointing in same direction
        directions = [p.direction for p in patterns]
        if len(set(directions)) == 1:  # All same direction
            return min(1.0, len(patterns) / 3.0)  # Max confluence at 3+ patterns
        else:
            return 0.0  # No confluence if mixed directions
    
    def _calculate_signal_strength(self, analysis: Dict[str, any]) -> float:
        """Calculate overall signal strength from institutional analysis."""
        # Component signals
        pattern_signal = analysis.get('pattern_signals', {}).get('overall_signal', 0.0)
        flow_signal = analysis.get('institutional_flow', {}).get('flow_direction', 0.0)
        smart_money_signal = analysis.get('smart_money', {}).get('direction', 0) * \
                           analysis.get('smart_money', {}).get('activity_strength', 0.0)
        
        # Weighted combination
        signal_strength = (
            pattern_signal * 0.5 +
            flow_signal * 0.3 +
            smart_money_signal * 0.2
        )
        
        # Apply confluence boost
        confluence = analysis.get('pattern_confluence', 0.0)
        signal_strength *= (1.0 + confluence * 0.3)  # Up to 30% boost
        
        return np.clip(signal_strength, -1.0, 1.0)
    
    def _calculate_confidence(self, analysis: Dict[str, any], patterns: List[ICTPattern]) -> float:
        """Calculate confidence in institutional analysis."""
        # Base confidence factors
        pattern_count = len(patterns)
        pattern_strength = analysis.get('pattern_signals', {}).get('max_strength', 0.0)
        smart_money_confidence = analysis.get('smart_money', {}).get('confidence', 0.0)
        confluence = analysis.get('pattern_confluence', 0.0)
        
        # Pattern count factor (more patterns = higher confidence)
        pattern_factor = min(1.0, pattern_count / 3.0)
        
        # Confluence factor
        confluence_factor = confluence
        
        # Smart money factor
        smart_money_factor = smart_money_confidence
        
        return compute_confidence(
            signal_strength=abs(self._calculate_signal_strength(analysis)),
            data_quality=1.0,  # Mock data is always available
            historical_accuracy=self.performance_tracker.get_accuracy(),
            confluence_signals=[pattern_factor, confluence_factor, smart_money_factor]
        )
    
    def _detect_market_regime(self, analysis: Dict[str, any]) -> MarketRegime:
        """Detect market regime from institutional activity."""
        institutional_activity = analysis.get('institutional_flow', {}).get('institutional_activity', 0.0)
        pattern_count = analysis.get('pattern_signals', {}).get('pattern_count', 0)
        smart_money_strength = analysis.get('smart_money', {}).get('activity_strength', 0.0)
        
        # High institutional activity suggests trending
        if institutional_activity > 0.7 and smart_money_strength > 0.6:
            return MarketRegime.TRENDING_STRONG
        elif institutional_activity > 0.5:
            return MarketRegime.TRENDING_WEAK
        elif pattern_count > 2:
            return MarketRegime.BREAKOUT
        else:
            return MarketRegime.CONSOLIDATING
    
    def _extract_evidence(self, analysis: Dict[str, any], patterns: List[ICTPattern]) -> Dict[str, float]:
        """Extract evidence scores for transparency."""
        evidence = {}
        
        evidence['pattern_strength'] = analysis.get('pattern_signals', {}).get('max_strength', 0.0)
        evidence['institutional_activity'] = analysis.get('institutional_flow', {}).get('institutional_activity', 0.0)
        evidence['smart_money_activity'] = analysis.get('smart_money', {}).get('activity_strength', 0.0)
        evidence['pattern_confluence'] = analysis.get('pattern_confluence', 0.0)
        evidence['pattern_count'] = len(patterns)
        
        return evidence
    
    def _generate_warnings(self, analysis: Dict[str, any]) -> List[str]:
        """Generate warnings about analysis quality or concerns."""
        warnings = []
        
        # Check for liquidity anomalies
        if analysis.get('institutional_flow', {}).get('liquidity_anomaly', False):
            warnings.append("Liquidity anomaly detected - unusual order book depth")
        
        # Check for conflicting signals
        pattern_signal = analysis.get('pattern_signals', {}).get('overall_signal', 0.0)
        flow_signal = analysis.get('institutional_flow', {}).get('flow_direction', 0.0)
        
        if abs(pattern_signal) > 0.3 and abs(flow_signal) > 0.3 and (pattern_signal * flow_signal) < 0:
            warnings.append("Conflicting signals between patterns and order flow")
        
        return warnings
    
    def _create_error_reading(self, timestamp: datetime, error_msg: str) -> DimensionalReading:
        """Create reading when error occurs."""
        return DimensionalReading(
            dimension="HOW",
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
            dimension="HOW",
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
        
        # Reset components
        self.ict_detector = ICTPatternDetector()
        self.order_flow_analyzer = OrderFlowAnalyzer()
        self.institutional_bias = EMA(50)
        self.smart_money_flow = EMA(20)
        self.pattern_confluence = 0.0
        
        logger.info("HOW Engine reset completed")

