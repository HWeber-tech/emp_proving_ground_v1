"""
WHEN Dimension - Temporal Advantage System
=========================================

Advanced temporal analysis and timing optimization engine for the 5D+1 sensory cortex.
Implements sophisticated timing analysis including:
- Session transition analysis
- Economic calendar impact assessment
- Microstructure timing optimization
- Volatility regime detection
- Optimal entry window identification

Author: EMP Development Team
Phase: 2 - Truth-First Completion
"""

import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


@dataclass
class SessionTransition:
    """Represents a market session transition"""
    session_from: str
    session_to: str
    transition_time: datetime
    transition_score: float
    volatility_change: float
    volume_change: float
    liquidity_change: float


@dataclass
class EconomicImpact:
    """Economic calendar impact assessment"""
    event_name: str
    event_time: datetime
    impact_level: str  # 'low', 'medium', 'high'
    expected_volatility: float
    expected_direction: str  # 'bullish', 'bearish', 'neutral'
    confidence: float


@dataclass
class MicrostructureTiming:
    """Microstructure timing analysis"""
    bid_ask_spread: float
    order_book_depth: float
    trade_frequency: float
    price_impact: float
    optimal_order_size: float
    timing_score: float


@dataclass
class VolatilityRegime:
    """Volatility regime classification"""
    regime_type: str  # 'low', 'normal', 'high', 'extreme'
    regime_strength: float
    expected_duration: timedelta
    regime_change_probability: float
    optimal_strategy: str


@dataclass
class TemporalAdvantage:
    """Complete temporal advantage analysis"""
    session_transitions: List[SessionTransition]
    economic_impacts: List[EconomicImpact]
    microstructure_timing: MicrostructureTiming
    volatility_regime: VolatilityRegime
    optimal_entry_window: Tuple[datetime, datetime]
    optimal_exit_window: Tuple[datetime, datetime]
    confidence: float


class TemporalAdvantageSystem:
    """
    Advanced temporal analysis system for optimal timing.
    
    Provides comprehensive timing analysis including session transitions,
    economic calendar impact, microstructure timing, and volatility regime detection.
    """
    
    def __init__(self):
        self.session_analyzer = SessionAnalyzer()
        self.economic_calendar = EconomicCalendarAnalyzer()
        self.microstructure_analyzer = MicrostructureAnalyzer()
        self.volatility_detector = VolatilityRegimeDetector()
        self.timing_optimizer = TimingOptimizer()
        
    async def analyze_timing(self, market_data: pd.DataFrame, 
                           symbol: str = "EURUSD") -> TemporalAdvantage:
        """
        Comprehensive temporal analysis for optimal timing.
        
        Args:
            market_data: OHLCV DataFrame with datetime index
            symbol: Trading symbol
            
        Returns:
            Complete temporal advantage analysis
        """
        try:
            if market_data.empty:
                logger.warning("No market data provided for temporal analysis")
                return self._get_fallback_analysis()
            
            # Analyze session transitions
            session_transitions = await self.session_analyzer.analyze_transitions(market_data)
            
            # Analyze economic calendar impact
            economic_impacts = await self.economic_calendar.analyze_impact(market_data)
            
            # Analyze microstructure timing
            microstructure = await self.microstructure_analyzer.analyze_timing(market_data)
            
            # Detect volatility regime
            volatility_regime = await self.volatility_detector.detect_regime(market_data)
            
            # Calculate optimal timing windows
            entry_window, exit_window = await self.timing_optimizer.calculate_windows(
                market_data, session_transitions, economic_impacts, volatility_regime
            )
            
            # Calculate overall confidence
            confidence = self._calculate_confidence(
                session_transitions, economic_impacts, microstructure, volatility_regime
            )
            
            return TemporalAdvantage(
                session_transitions=session_transitions,
                economic_impacts=economic_impacts,
                microstructure_timing=microstructure,
                volatility_regime=volatility_regime,
                optimal_entry_window=entry_window,
                optimal_exit_window=exit_window,
                confidence=confidence
            )
            
        except Exception as e:
            logger.error(f"Temporal analysis failed: {e}")
            return self._get_fallback_analysis()
    
    def _calculate_confidence(self, transitions: List, impacts: List, 
                            microstructure: MicrostructureTiming, 
                            regime: VolatilityRegime) -> float:
        """Calculate overall confidence in temporal analysis"""
        factors = [
            len(transitions) / 5.0,  # Normalize to 0-1
            len(impacts) / 3.0,      # Normalize to 0-1
            microstructure.timing_score,
            regime.regime_strength
        ]
        return min(1.0, np.mean(factors))
    
    def _get_fallback_analysis(self) -> TemporalAdvantage:
        """Return fallback temporal analysis"""
        now = datetime.now()
        return TemporalAdvantage(
            session_transitions=[],
            economic_impacts=[],
            microstructure_timing=MicrostructureTiming(
                bid_ask_spread=0.001,
                order_book_depth=1000,
                trade_frequency=50,
                price_impact=0.0001,
                optimal_order_size=10,
                timing_score=0.5
            ),
            volatility_regime=VolatilityRegime(
                regime_type="normal",
                regime_strength=0.5,
                expected_duration=timedelta(hours=4),
                regime_change_probability=0.1,
                optimal_strategy="hold"
            ),
            optimal_entry_window=(now, now + timedelta(hours=1)),
            optimal_exit_window=(now + timedelta(hours=1), now + timedelta(hours=2)),
            confidence=0.1
        )


class SessionAnalyzer:
    """Market session transition analysis"""
    
    async def analyze_transitions(self, data: pd.DataFrame) -> List[SessionTransition]:
        """Analyze market session transitions"""
        transitions = []
        
        # Define major trading sessions
        sessions = {
            'Sydney': {'start': 22, 'end': 6},    # UTC
            'Tokyo': {'start': 0, 'end': 8},      # UTC
            'London': {'start': 8, 'end': 16},    # UTC
            'New York': {'start': 13, 'end': 21}  # UTC
        }
        
        # Analyze transitions between sessions
        for i in range(1, len(data)):
            prev_time = data.index[i-1]
            curr_time = data.index[i]
            
            prev_session = self._get_session(prev_time, sessions)
            curr_session = self._get_session(curr_time, sessions)
            
            if prev_session != curr_session:
                # Calculate transition metrics
                prev_vol = data['high'].iloc[i-1] - data['low'].iloc[i-1]
                curr_vol = data['high'].iloc[i] - data['low'].iloc[i]
                vol_change = (curr_vol - prev_vol) / prev_vol if prev_vol > 0 else 0
                
                prev_vol = data['volume'].iloc[i-1]
                curr_vol = data['volume'].iloc[i]
                volume_change = (curr_vol - prev_vol) / prev_vol if prev_vol > 0 else 0
                
                transition = SessionTransition(
                    session_from=prev_session,
                    session_to=curr_session,
                    transition_time=curr_time,
                    transition_score=0.7,
                    volatility_change=vol_change,
                    volume_change=volume_change,
                    liquidity_change=volume_change
                )
                transitions.append(transition)
        
        return transitions
    
    def _get_session(self, dt: datetime, sessions: Dict) -> str:
        """Determine which trading session a datetime belongs to"""
        hour = dt.hour
        
        # Check each session
        for session_name, times in sessions.items():
            start = times['start']
            end = times['end']
            
            if start <= end:
                if start <= hour < end:
                    return session_name
            else:  # Crosses midnight
                if hour >= start or hour < end:
                    return session_name
        
        return 'Transition'


class EconomicCalendarAnalyzer:
    """Economic calendar impact analysis"""
    
    async def analyze_impact(self, data: pd.DataFrame) -> List[EconomicImpact]:
        """Analyze economic calendar impact on timing"""
        impacts = []
        
        # High-impact events (simplified)
        high_impact_times = [
            (8, 30),  # London open
            (13, 30),  # NY open
            (14, 0),  # ECB/Fed announcements
            (15, 0),  # US data releases
        ]
        
        for dt in data.index:
            hour = dt.hour
            minute = dt.minute
            
            for hi_hour, hi_minute in high_impact_times:
                if hour == hi_hour and abs(minute - hi_minute) <= 15:
                    impact = EconomicImpact(
                        event_name=f"Market_Open_{hi_hour}_{hi_minute}",
                        event_time=dt,
                        impact_level='high',
                        expected_volatility=0.02,
                        expected_direction='neutral',
                        confidence=0.8
                    )
                    impacts.append(impact)
        
        return impacts


class MicrostructureAnalyzer:
    """Microstructure timing analysis"""
    
    async def analyze_timing(self, data: pd.DataFrame) -> MicrostructureTiming:
        """Analyze microstructure for optimal timing"""
        try:
            if data.empty:
                return MicrostructureTiming(
                    bid_ask_spread=0.001,
                    order_book_depth=1000,
                    trade_frequency=50,
                    price_impact=0.0001,
                    optimal_order_size=10,
                    timing_score=0.5
                )
            
            # Calculate bid-ask spread proxy
            spread = (data['high'] - data['low']).mean() / data['close'].mean()
            
            # Calculate order book depth proxy
            volume_depth = data['volume'].mean()
            
            # Calculate trade frequency
            trade_freq = len(data) / ((data.index[-1] - data.index[0]).total_seconds() / 3600)
            
            # Calculate price impact
            price_impact = data['close'].pct_change().abs().mean()
            
            # Calculate optimal order size
            optimal_size = volume_depth * 0.01
            
            # Calculate timing score
            timing_score = self._calculate_timing_score(spread, volume_depth, trade_freq)
            
            return MicrostructureTiming(
                bid_ask_spread=spread,
                order_book_depth=volume_depth,
                trade_frequency=trade_freq,
                price_impact=price_impact,
                optimal_order_size=optimal_size,
                timing_score=timing_score
            )
            
        except Exception as e:
            logger.error(f"Microstructure analysis failed: {e}")
            return MicrostructureTiming(
                bid_ask_spread=0.001,
                order_book_depth=1000,
                trade_frequency=50,
                price_impact=0.0001,
                optimal_order_size=10,
                timing_score=0.5
            )
    
    def _calculate_timing_score(self, spread: float, depth: float, frequency: float) -> float:
        """Calculate microstructure timing score"""
        spread_score = max(0, 1 - spread * 100)
        depth_score = min(1, np.log10(depth + 1) / 5)
        freq_score = min(1, frequency / 100)
        return (spread_score + depth_score + freq_score) / 3


class VolatilityRegimeDetector:
    """Volatility regime detection and classification"""
    
    async def detect_regime(self, data: pd.DataFrame) -> VolatilityRegime:
        """Detect current volatility regime"""
        try:
            if data.empty or len(data) < 20:
                return VolatilityRegime(
                    regime_type='normal',
                    regime_strength=0.5,
                    expected_duration=timedelta(hours=4),
                    regime_change_probability=0.1,
                    optimal_strategy='hold'
                )
            
            # Calculate realized volatility
            returns = data['close'].pct_change().dropna()
            volatility = returns.rolling(20).std().iloc[-1] * np.sqrt(252)
            
            # Classify regime
            if volatility < 0.05:
                regime_type = 'low'
            elif volatility < 0.15:
                regime_type = 'normal'
            elif volatility < 0.25:
                regime_type = 'high'
            else:
                regime_type = 'extreme'
            
            # Calculate regime strength
            regime_strength = min(1.0, abs(volatility - 0.10) / 0.20)
            
            # Estimate expected duration
            expected_duration = timedelta(hours=4)
            
            # Calculate regime change probability
            change_probability = 0.1
            
            # Determine optimal strategy
            optimal_strategy = self._get_optimal_strategy(regime_type)
            
            return VolatilityRegime(
                regime_type=regime_type,
                regime_strength=regime_strength,
                expected_duration=expected_duration,
                regime_change_probability=change_probability,
                optimal_strategy=optimal_strategy
            )
            
        except Exception as e:
            logger.error(f"Volatility regime detection failed: {e}")
            return VolatilityRegime(
                regime_type='normal',
                regime_strength=0.5,
                expected_duration=timedelta(hours=4),
                regime_change_probability=0.1,
                optimal_strategy='hold'
            )
    
    def _get_optimal_strategy(self, regime_type: str) -> str:
        """Get optimal strategy for volatility regime"""
        strategies = {
            'low': 'trend_following',
            'normal': 'mean_reversion',
            'high': 'momentum',
            'extreme': 'breakout'
        }
        return strategies.get(regime_type, 'hold')


class TimingOptimizer:
    """Optimal timing window calculation"""
    
    async def calculate_windows(self, data: pd.DataFrame, 
                              transitions: List[SessionTransition],
                              impacts: List[EconomicImpact],
                              regime: VolatilityRegime) -> Tuple[datetime, datetime]:
        """Calculate optimal entry and exit windows"""
        try:
            if data.empty:
                now = datetime.now()
                return (now, now + timedelta(hours=1))
            
            # Get last timestamp
            last_time = data.index[-1]
            
            # Calculate optimal windows based on analysis
            entry_delay = self._calculate_entry_delay(transitions, impacts, regime)
            exit_delay = self._calculate_exit_delay(regime)
            
            entry_window = (last_time, last_time + timedelta(minutes=entry_delay))
            exit_window = (last_time + timedelta(minutes=entry_delay), 
                          last_time + timedelta(minutes=entry_delay + exit_delay))
            
            return entry_window, exit_window
            
        except Exception as e:
            logger.error(f"Timing optimization failed: {e}")
            now = datetime.now()
            return (now, now + timedelta(hours=1))
    
    def _calculate_entry_delay(self, transitions: List, impacts: List, regime: VolatilityRegime) -> int:
        """Calculate optimal entry delay in minutes"""
        base_delay = 15
        
        # Adjust based on transitions
        if transitions:
            base_delay += 10
        
        # Adjust based on economic impacts
        if impacts:
            base_delay += 20
        
        # Adjust based on volatility regime
        regime_delays = {
            'low': 5,
            'normal': 15,
            'high': 30,
            'extreme': 60
        }
        
        base_delay += regime_delays.get(regime.regime_type, 15)
        return min(base_delay, 120)  # Cap at 2 hours
    
    def _calculate_exit_delay(self, regime: VolatilityRegime) -> int:
        """Calculate optimal exit delay in minutes"""
        delay_map = {
            'low': 60,
            'normal': 120,
            'high': 240,
            'extreme': 480
        }
        return delay_map.get(regime.regime_type, 120)
