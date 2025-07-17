"""
Enhanced WHEN Dimension - Chronon Temporal Intelligence Engine

This module implements sophisticated temporal market intelligence that goes beyond simple
session tracking to understand the deep rhythms and timing patterns that drive market behavior.
It analyzes chronons (temporal market units) and their influence on price action.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta, time, timezone
from typing import Dict, Any, Optional, List, Tuple, NamedTuple
from collections import deque, defaultdict
from dataclasses import dataclass, field
from enum import Enum, auto
import calendar
import math
from scipy import stats
from scipy.signal import find_peaks
import logging

from ..core.base import DimensionalReading, MarketData, MarketRegime

logger = logging.getLogger(__name__)

class TradingSession(Enum):
    """Enhanced trading session classification"""
    ASIAN_EARLY = auto()      # 00:00-02:00 UTC (Tokyo open)
    ASIAN_MAIN = auto()       # 02:00-08:00 UTC (Tokyo main)
    ASIAN_LATE = auto()       # 08:00-09:00 UTC (Tokyo close)
    LONDON_EARLY = auto()     # 08:00-10:00 UTC (London open)
    LONDON_MAIN = auto()      # 10:00-15:00 UTC (London main)
    LONDON_LATE = auto()      # 15:00-17:00 UTC (London close)
    NY_EARLY = auto()         # 13:00-15:00 UTC (NY open)
    NY_MAIN = auto()          # 15:00-20:00 UTC (NY main)
    NY_LATE = auto()          # 20:00-22:00 UTC (NY close)
    OVERLAP_ASIAN_LONDON = auto()  # 08:00-09:00 UTC
    OVERLAP_LONDON_NY = auto()     # 13:00-17:00 UTC
    QUIET_PERIOD = auto()     # 22:00-00:00 UTC

class TemporalRegime(Enum):
    """Temporal market regimes"""
    HIGH_ACTIVITY = auto()    # Major session overlaps
    MEDIUM_ACTIVITY = auto()  # Single major session
    LOW_ACTIVITY = auto()     # Minor sessions or quiet periods
    TRANSITION = auto()       # Session handovers
    EVENT_DRIVEN = auto()     # Around major news events

class ChronoBehavior(Enum):
    """Chronological behavior patterns"""
    MOMENTUM_BUILDING = auto()     # Early session momentum
    MOMENTUM_SUSTAINING = auto()   # Mid-session continuation
    MOMENTUM_EXHAUSTING = auto()   # Late session fatigue
    REVERSAL_PRONE = auto()        # Session transition reversals
    CONSOLIDATION = auto()         # Range-bound periods
    BREAKOUT_LIKELY = auto()       # Pre-breakout accumulation

@dataclass
class SessionProfile:
    """Comprehensive session characteristics"""
    session: TradingSession
    start_time: time
    end_time: time
    typical_volatility: float
    typical_volume: float
    directional_bias: float  # -1 to 1
    reversal_probability: float
    breakout_probability: float
    institutional_activity: float
    retail_activity: float

@dataclass
class TemporalPattern:
    """Detected temporal pattern"""
    pattern_type: str
    strength: float
    duration: timedelta
    next_occurrence: Optional[datetime]
    confidence: float
    historical_accuracy: float

@dataclass
class EventHorizon:
    """Upcoming market events and their temporal influence"""
    event_name: str
    event_time: datetime
    importance: float  # 0-1
    expected_volatility_impact: float
    pre_event_behavior: str
    post_event_behavior: str
    time_to_event: timedelta

class TemporalAnalyzer:
    """
    Advanced temporal analysis engine that understands market chronons
    and their influence on price behavior
    """
    
    def __init__(self, lookback_days: int = 30):
        self.lookback_days = lookback_days
        
        # Historical data storage
        self.session_data = defaultdict(list)
        self.hourly_patterns = defaultdict(list)
        self.daily_patterns = defaultdict(list)
        self.weekly_patterns = defaultdict(list)
        self.monthly_patterns = defaultdict(list)
        
        # Real-time tracking
        self.current_session = TradingSession.QUIET_PERIOD
        self.session_start_time = None
        self.session_volatility = deque(maxlen=100)
        self.session_volume = deque(maxlen=100)
        self.session_price_change = 0.0
        
        # Pattern detection
        self.detected_patterns: List[TemporalPattern] = []
        self.event_horizon: List[EventHorizon] = []
        
        # Adaptive parameters
        self.volatility_threshold = 0.008
        self.volume_threshold = 1000
        
        # Session profiles (learned from historical data)
        self.session_profiles = self._initialize_session_profiles()
        
    def _initialize_session_profiles(self) -> Dict[TradingSession, SessionProfile]:
        """Initialize session profiles with market knowledge"""
        return {
            TradingSession.ASIAN_EARLY: SessionProfile(
                session=TradingSession.ASIAN_EARLY,
                start_time=time(0, 0),
                end_time=time(2, 0),
                typical_volatility=0.004,
                typical_volume=800,
                directional_bias=0.0,
                reversal_probability=0.3,
                breakout_probability=0.2,
                institutional_activity=0.4,
                retail_activity=0.6
            ),
            TradingSession.ASIAN_MAIN: SessionProfile(
                session=TradingSession.ASIAN_MAIN,
                start_time=time(2, 0),
                end_time=time(8, 0),
                typical_volatility=0.006,
                typical_volume=1200,
                directional_bias=0.1,
                reversal_probability=0.4,
                breakout_probability=0.3,
                institutional_activity=0.7,
                retail_activity=0.5
            ),
            TradingSession.LONDON_EARLY: SessionProfile(
                session=TradingSession.LONDON_EARLY,
                start_time=time(8, 0),
                end_time=time(10, 0),
                typical_volatility=0.012,
                typical_volume=2000,
                directional_bias=0.0,
                reversal_probability=0.6,
                breakout_probability=0.7,
                institutional_activity=0.9,
                retail_activity=0.4
            ),
            TradingSession.LONDON_MAIN: SessionProfile(
                session=TradingSession.LONDON_MAIN,
                start_time=time(10, 0),
                end_time=time(15, 0),
                typical_volatility=0.010,
                typical_volume=1800,
                directional_bias=0.0,
                reversal_probability=0.4,
                breakout_probability=0.5,
                institutional_activity=0.8,
                retail_activity=0.6
            ),
            TradingSession.NY_EARLY: SessionProfile(
                session=TradingSession.NY_EARLY,
                start_time=time(13, 0),
                end_time=time(15, 0),
                typical_volatility=0.014,
                typical_volume=2500,
                directional_bias=0.0,
                reversal_probability=0.7,
                breakout_probability=0.8,
                institutional_activity=0.9,
                retail_activity=0.7
            ),
            TradingSession.NY_MAIN: SessionProfile(
                session=TradingSession.NY_MAIN,
                start_time=time(15, 0),
                end_time=time(20, 0),
                typical_volatility=0.011,
                typical_volume=2200,
                directional_bias=0.0,
                reversal_probability=0.5,
                breakout_probability=0.6,
                institutional_activity=0.8,
                retail_activity=0.8
            ),
            TradingSession.OVERLAP_LONDON_NY: SessionProfile(
                session=TradingSession.OVERLAP_LONDON_NY,
                start_time=time(13, 0),
                end_time=time(17, 0),
                typical_volatility=0.016,
                typical_volume=3000,
                directional_bias=0.0,
                reversal_probability=0.6,
                breakout_probability=0.9,
                institutional_activity=0.95,
                retail_activity=0.8
            ),
            TradingSession.QUIET_PERIOD: SessionProfile(
                session=TradingSession.QUIET_PERIOD,
                start_time=time(22, 0),
                end_time=time(0, 0),
                typical_volatility=0.003,
                typical_volume=500,
                directional_bias=0.0,
                reversal_probability=0.2,
                breakout_probability=0.1,
                institutional_activity=0.2,
                retail_activity=0.3
            )
        }
    
    def update_temporal_data(self, market_data: MarketData) -> None:
        """Update temporal analysis with new market data"""
        
        current_time = market_data.timestamp
        
        # Determine current session
        self._update_current_session(current_time)
        
        # Update session tracking
        self._update_session_tracking(market_data)
        
        # Store historical patterns
        self._store_temporal_patterns(market_data)
        
        # Update adaptive parameters
        self._update_adaptive_parameters(market_data)
        
        # Detect temporal patterns
        self._detect_temporal_patterns()
        
        # Update event horizon
        self._update_event_horizon(current_time)
    
    def _update_current_session(self, current_time: datetime) -> None:
        """Determine and update current trading session"""
        utc_time = current_time.replace(tzinfo=timezone.utc).time()
        
        # Check for session overlaps first (highest priority)
        if time(13, 0) <= utc_time < time(17, 0):
            new_session = TradingSession.OVERLAP_LONDON_NY
        elif time(8, 0) <= utc_time < time(9, 0):
            new_session = TradingSession.OVERLAP_ASIAN_LONDON
        # Individual sessions
        elif time(0, 0) <= utc_time < time(2, 0):
            new_session = TradingSession.ASIAN_EARLY
        elif time(2, 0) <= utc_time < time(8, 0):
            new_session = TradingSession.ASIAN_MAIN
        elif time(8, 0) <= utc_time < time(10, 0):
            new_session = TradingSession.LONDON_EARLY
        elif time(10, 0) <= utc_time < time(13, 0):
            new_session = TradingSession.LONDON_MAIN
        elif time(15, 0) <= utc_time < time(17, 0):
            new_session = TradingSession.NY_EARLY
        elif time(17, 0) <= utc_time < time(20, 0):
            new_session = TradingSession.NY_MAIN
        elif time(20, 0) <= utc_time < time(22, 0):
            new_session = TradingSession.NY_LATE
        else:
            new_session = TradingSession.QUIET_PERIOD
        
        # Check for session transition
        if new_session != self.current_session:
            self._handle_session_transition(new_session, current_time)
        
        self.current_session = new_session
    
    def _handle_session_transition(self, new_session: TradingSession, transition_time: datetime) -> None:
        """Handle session transition and store session data"""
        
        if self.session_start_time and self.session_volatility:
            # Calculate session statistics
            start_time = self.session_start_time
            current_time = transition_time
            
            if start_time.tzinfo is not None:
                start_time = start_time.replace(tzinfo=None)
            if current_time.tzinfo is not None:
                current_time = current_time.replace(tzinfo=None)
                
            session_duration = current_time - start_time
            avg_volatility = np.mean(list(self.session_volatility))
            avg_volume = np.mean(list(self.session_volume))
            
            # Store session data for pattern analysis
            session_data = {
                'session': self.current_session,
                'start_time': self.session_start_time,
                'duration': session_duration,
                'avg_volatility': avg_volatility,
                'avg_volume': avg_volume,
                'price_change': self.session_price_change,
                'day_of_week': self.session_start_time.weekday(),
                'month': self.session_start_time.month
            }
            
            self.session_data[self.current_session].append(session_data)
            
            # Keep only recent data
            if len(self.session_data[self.current_session]) > 100:
                self.session_data[self.current_session] = self.session_data[self.current_session][-100:]
        
        # Reset for new session
        self.session_start_time = transition_time
        self.session_volatility.clear()
        self.session_volume.clear()
        self.session_price_change = 0.0
    
    def _update_session_tracking(self, market_data: MarketData) -> None:
        """Update current session tracking metrics"""
        
        if not self.session_start_time:
            self.session_start_time = market_data.timestamp
        
        # Track session metrics
        self.session_volatility.append(market_data.volatility)
        self.session_volume.append(market_data.volume)
        
        # Calculate session price change (simplified)
        if len(self.session_volatility) > 1:
            # Estimate price change from volatility (in real implementation, track actual prices)
            price_change_estimate = market_data.volatility * np.random.choice([-1, 1])
            self.session_price_change += price_change_estimate
    
    def _store_temporal_patterns(self, market_data: MarketData) -> None:
        """Store data for temporal pattern analysis"""
        
        current_time = market_data.timestamp
        
        if current_time.tzinfo is not None:
            current_time = current_time.replace(tzinfo=None)
        
        # Hourly patterns
        hour_key = current_time.hour
        self.hourly_patterns[hour_key].append({
            'volatility': market_data.volatility,
            'volume': market_data.volume,
            'timestamp': current_time
        })
        
        # Daily patterns
        day_key = current_time.weekday()
        self.daily_patterns[day_key].append({
            'volatility': market_data.volatility,
            'volume': market_data.volume,
            'session': self.current_session,
            'timestamp': current_time
        })
        
        # Weekly patterns (week of month)
        week_of_month = (current_time.day - 1) // 7 + 1
        self.weekly_patterns[week_of_month].append({
            'volatility': market_data.volatility,
            'volume': market_data.volume,
            'timestamp': current_time
        })
        
        # Monthly patterns
        month_key = current_time.month
        self.monthly_patterns[month_key].append({
            'volatility': market_data.volatility,
            'volume': market_data.volume,
            'timestamp': current_time
        })
        
        # Clean old data
        cutoff_time = current_time - timedelta(days=self.lookback_days)
        self._clean_old_patterns(cutoff_time)
    
    def _clean_old_patterns(self, cutoff_time: datetime) -> None:
        """Remove old pattern data beyond lookback period"""
        
        if cutoff_time.tzinfo is not None:
            cutoff_time = cutoff_time.replace(tzinfo=None)
        
        for pattern_dict in [self.hourly_patterns, self.daily_patterns, 
                           self.weekly_patterns, self.monthly_patterns]:
            for key in pattern_dict:
                pattern_dict[key] = [
                    item for item in pattern_dict[key]
                    if item['timestamp'] > cutoff_time
                ]
    
    def _update_adaptive_parameters(self, market_data: MarketData) -> None:
        """Update adaptive parameters based on recent market behavior"""
        
        if len(self.session_volatility) >= 10:
            # Adapt volatility threshold to recent conditions
            recent_vol = np.mean(list(self.session_volatility)[-10:])
            self.volatility_threshold = recent_vol * 1.5
        
        if len(self.session_volume) >= 10:
            # Adapt volume threshold to recent conditions
            recent_vol = np.mean(list(self.session_volume)[-10:])
            self.volume_threshold = recent_vol * 0.8
    
    def _detect_temporal_patterns(self) -> None:
        """Detect recurring temporal patterns"""
        
        self.detected_patterns.clear()
        
        # Detect hourly patterns
        hourly_pattern = self._analyze_hourly_patterns()
        if hourly_pattern:
            self.detected_patterns.append(hourly_pattern)
        
        # Detect daily patterns
        daily_pattern = self._analyze_daily_patterns()
        if daily_pattern:
            self.detected_patterns.append(daily_pattern)
        
        # Detect session-specific patterns
        session_pattern = self._analyze_session_patterns()
        if session_pattern:
            self.detected_patterns.append(session_pattern)
        
        # Detect cyclical patterns
        cyclical_patterns = self._analyze_cyclical_patterns()
        self.detected_patterns.extend(cyclical_patterns)
    
    def _analyze_hourly_patterns(self) -> Optional[TemporalPattern]:
        """Analyze hourly volatility and volume patterns"""
        
        if not self.hourly_patterns:
            return None
        
        # Calculate hourly averages
        hourly_volatility = {}
        hourly_volume = {}
        
        for hour in range(24):
            if hour in self.hourly_patterns and self.hourly_patterns[hour]:
                data = self.hourly_patterns[hour]
                hourly_volatility[hour] = np.mean([d['volatility'] for d in data])
                hourly_volume[hour] = np.mean([d['volume'] for d in data])
        
        if len(hourly_volatility) < 12:  # Need at least half the hours
            return None
        
        # Find peak activity hours
        vol_values = list(hourly_volatility.values())
        vol_hours = list(hourly_volatility.keys())
        
        if len(vol_values) > 3:
            peaks, _ = find_peaks(vol_values, height=np.mean(vol_values))
            
            if len(peaks) > 0:
                peak_hour = vol_hours[peaks[0]]
                peak_strength = vol_values[peaks[0]] / np.mean(vol_values)
                
                return TemporalPattern(
                    pattern_type='hourly_volatility_peak',
                    strength=min(peak_strength - 1.0, 1.0),
                    duration=timedelta(hours=1),
                    next_occurrence=self._calculate_next_hour_occurrence(peak_hour),
                    confidence=0.7,
                    historical_accuracy=0.6
                )
        
        return None
    
    def _analyze_daily_patterns(self) -> Optional[TemporalPattern]:
        """Analyze day-of-week patterns"""
        
        if not self.daily_patterns:
            return None
        
        # Calculate daily averages
        daily_volatility = {}
        
        for day in range(7):  # Monday = 0, Sunday = 6
            if day in self.daily_patterns and self.daily_patterns[day]:
                data = self.daily_patterns[day]
                daily_volatility[day] = np.mean([d['volatility'] for d in data])
        
        if len(daily_volatility) < 5:  # Need at least 5 days
            return None
        
        # Find most volatile day
        max_day = max(daily_volatility, key=lambda k: daily_volatility[k])
        max_volatility = daily_volatility[max_day]
        avg_volatility = np.mean(list(daily_volatility.values()))
        
        if max_volatility > avg_volatility * 1.3:  # 30% above average
            day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            
            return TemporalPattern(
                pattern_type=f'{day_names[max_day]}_volatility_spike',
                strength=(max_volatility / avg_volatility) - 1.0,
                duration=timedelta(days=1),
                next_occurrence=self._calculate_next_day_occurrence(max_day),
                confidence=0.6,
                historical_accuracy=0.5
            )
        
        return None
    
    def _analyze_session_patterns(self) -> Optional[TemporalPattern]:
        """Analyze session-specific behavioral patterns"""
        
        if not self.session_data:
            return None
        
        # Analyze current session's typical behavior
        current_session_data = self.session_data.get(self.current_session, [])
        
        if len(current_session_data) < 5:
            return None
        
        # Calculate session characteristics
        volatilities = [d['avg_volatility'] for d in current_session_data]
        price_changes = [d['price_change'] for d in current_session_data]
        
        avg_volatility = np.mean(volatilities)
        volatility_consistency = 1.0 - (np.std(volatilities) / avg_volatility) if avg_volatility > 0 else 0
        
        # Determine directional bias
        positive_sessions = sum(1 for pc in price_changes if pc > 0)
        directional_bias = (positive_sessions / len(price_changes)) - 0.5  # -0.5 to 0.5
        
        if volatility_consistency > 0.6 or abs(directional_bias) > 0.2:
            return TemporalPattern(
                pattern_type=f'{self.current_session.name}_behavior',
                strength=max(float(volatility_consistency), float(abs(directional_bias) * 2)),
                duration=timedelta(hours=2),  # Typical session duration
                next_occurrence=self._calculate_next_session_occurrence(),
                confidence=float(volatility_consistency),
                historical_accuracy=0.7
            )
        
        return None
    
    def _analyze_cyclical_patterns(self) -> List[TemporalPattern]:
        """Analyze longer-term cyclical patterns"""
        
        patterns = []
        
        # Weekly patterns (week of month)
        if self.weekly_patterns:
            for week in range(1, 5):  # Weeks 1-4 of month
                if week in self.weekly_patterns and len(self.weekly_patterns[week]) >= 3:
                    data = self.weekly_patterns[week]
                    avg_volatility = np.mean([d['volatility'] for d in data])
                    
                    # Compare to overall average
                    all_weekly_data = []
                    for w in self.weekly_patterns.values():
                        all_weekly_data.extend(w)
                    
                    if all_weekly_data:
                        overall_avg = np.mean([d['volatility'] for d in all_weekly_data])
                        
                        if avg_volatility > overall_avg * 1.2:  # 20% above average
                            patterns.append(TemporalPattern(
                                pattern_type=f'week_{week}_volatility',
                                strength=float((avg_volatility / overall_avg) - 1.0),
                                duration=timedelta(weeks=1),
                                next_occurrence=self._calculate_next_week_occurrence(week),
                                confidence=0.5,
                                historical_accuracy=0.4
                            ))
        
        # Monthly patterns
        if self.monthly_patterns:
            current_month = datetime.utcnow().month
            if current_month in self.monthly_patterns and len(self.monthly_patterns[current_month]) >= 5:
                data = self.monthly_patterns[current_month]
                avg_volatility = np.mean([d['volatility'] for d in data])
                
                # Compare to other months
                other_months_data = []
                for month, month_data in self.monthly_patterns.items():
                    if month != current_month:
                        other_months_data.extend(month_data)
                
                if other_months_data:
                    other_avg = np.mean([d['volatility'] for d in other_months_data])
                    
                    if avg_volatility > other_avg * 1.15:  # 15% above average
                        month_names = ['', 'January', 'February', 'March', 'April', 'May', 'June',
                                     'July', 'August', 'September', 'October', 'November', 'December']
                        
                        patterns.append(TemporalPattern(
                            pattern_type=f'{month_names[current_month]}_seasonal',
                            strength=float((avg_volatility / other_avg) - 1.0),
                            duration=timedelta(days=30),
                            next_occurrence=self._calculate_next_month_occurrence(current_month),
                            confidence=0.4,
                            historical_accuracy=0.3
                        ))
        
        return patterns
    
    def _calculate_next_hour_occurrence(self, target_hour: int) -> datetime:
        """Calculate next occurrence of specific hour"""
        now = datetime.utcnow()
        target_time = now.replace(hour=target_hour, minute=0, second=0, microsecond=0)
        
        if target_time <= now:
            target_time += timedelta(days=1)
        
        return target_time
    
    def _calculate_next_day_occurrence(self, target_day: int) -> datetime:
        """Calculate next occurrence of specific day of week"""
        now = datetime.utcnow()
        days_ahead = target_day - now.weekday()
        
        if days_ahead <= 0:  # Target day already happened this week
            days_ahead += 7
        
        return now + timedelta(days=days_ahead)
    
    def _calculate_next_session_occurrence(self) -> datetime:
        """Calculate next occurrence of current session"""
        now = datetime.utcnow()
        session_profile = self.session_profiles[self.current_session]
        
        # Calculate next occurrence (tomorrow at session start time)
        next_occurrence = now.replace(
            hour=session_profile.start_time.hour,
            minute=session_profile.start_time.minute,
            second=0,
            microsecond=0
        ) + timedelta(days=1)
        
        return next_occurrence
    
    def _calculate_next_week_occurrence(self, target_week: int) -> datetime:
        """Calculate next occurrence of specific week of month"""
        now = datetime.utcnow()
        
        # Calculate first day of next month
        if now.month == 12:
            next_month = now.replace(year=now.year + 1, month=1, day=1)
        else:
            next_month = now.replace(month=now.month + 1, day=1)
        
        # Calculate target week
        target_date = next_month + timedelta(days=(target_week - 1) * 7)
        
        return target_date
    
    def _calculate_next_month_occurrence(self, target_month: int) -> datetime:
        """Calculate next occurrence of specific month"""
        now = datetime.utcnow()
        
        if target_month > now.month:
            return now.replace(month=target_month, day=1)
        else:
            return now.replace(year=now.year + 1, month=target_month, day=1)
    
    def _update_event_horizon(self, current_time: datetime) -> None:
        """Update upcoming events that may affect temporal behavior"""
        
        if current_time.tzinfo is not None:
            current_time = current_time.replace(tzinfo=None)
        
        # Clear expired events
        self.event_horizon = [
            event for event in self.event_horizon
            if event.event_time > current_time
        ]
        
        # Add upcoming session transitions as events
        next_session_time = self._calculate_next_session_transition(current_time)
        if next_session_time:
            if next_session_time.tzinfo is not None:
                next_session_time = next_session_time.replace(tzinfo=None)
            
            time_to_event = next_session_time - current_time
            
            if time_to_event <= timedelta(hours=2):  # Within 2 hours
                session_event = EventHorizon(
                    event_name='session_transition',
                    event_time=next_session_time,
                    importance=0.7,
                    expected_volatility_impact=0.3,
                    pre_event_behavior='consolidation',
                    post_event_behavior='volatility_spike',
                    time_to_event=time_to_event
                )
                
                # Check if already exists
                if not any(e.event_name == 'session_transition' for e in self.event_horizon):
                    self.event_horizon.append(session_event)
    
    def _calculate_next_session_transition(self, current_time: datetime) -> Optional[datetime]:
        """Calculate next major session transition"""
        
        # Ensure we work with timezone-naive datetime for consistency
        if current_time.tzinfo is not None:
            utc_time = current_time.replace(tzinfo=None)
        else:
            utc_time = current_time
        
        # Major transition times (UTC)
        transition_times = [
            time(8, 0),   # London open
            time(13, 0),  # NY open / London-NY overlap
            time(17, 0),  # London close
            time(22, 0),  # NY close
        ]
        
        # Find next transition
        current_utc_time = utc_time.time()
        
        for transition_time in transition_times:
            if transition_time > current_utc_time:
                return utc_time.replace(
                    hour=transition_time.hour,
                    minute=transition_time.minute,
                    second=0,
                    microsecond=0
                )
        
        # If no transition today, return first transition tomorrow
        tomorrow = utc_time + timedelta(days=1)
        return tomorrow.replace(
            hour=transition_times[0].hour,
            minute=transition_times[0].minute,
            second=0,
            microsecond=0
        )
    
    def get_temporal_regime(self) -> TemporalRegime:
        """Determine current temporal regime"""
        
        session_profile = self.session_profiles[self.current_session]
        
        # Check for overlaps (high activity)
        if self.current_session in [TradingSession.OVERLAP_LONDON_NY, TradingSession.OVERLAP_ASIAN_LONDON]:
            return TemporalRegime.HIGH_ACTIVITY
        
        # Check for major sessions
        if self.current_session in [TradingSession.LONDON_MAIN, TradingSession.NY_MAIN]:
            return TemporalRegime.MEDIUM_ACTIVITY
        
        # Check for quiet periods
        if self.current_session == TradingSession.QUIET_PERIOD:
            return TemporalRegime.LOW_ACTIVITY
        
        # Check for transitions (early/late sessions)
        if 'EARLY' in self.current_session.name or 'LATE' in self.current_session.name:
            return TemporalRegime.TRANSITION
        
        # Check for event-driven periods
        if any(event.time_to_event <= timedelta(minutes=30) for event in self.event_horizon):
            return TemporalRegime.EVENT_DRIVEN
        
        return TemporalRegime.MEDIUM_ACTIVITY
    
    def get_chrono_behavior(self) -> ChronoBehavior:
        """Determine current chronological behavior expectation"""
        
        session_profile = self.session_profiles[self.current_session]
        temporal_regime = self.get_temporal_regime()
        
        # Session-based behavior
        if 'EARLY' in self.current_session.name:
            if session_profile.breakout_probability > 0.6:
                return ChronoBehavior.BREAKOUT_LIKELY
            else:
                return ChronoBehavior.MOMENTUM_BUILDING
        
        elif 'MAIN' in self.current_session.name:
            if temporal_regime == TemporalRegime.HIGH_ACTIVITY:
                return ChronoBehavior.MOMENTUM_SUSTAINING
            else:
                return ChronoBehavior.CONSOLIDATION
        
        elif 'LATE' in self.current_session.name:
            if session_profile.reversal_probability > 0.5:
                return ChronoBehavior.REVERSAL_PRONE
            else:
                return ChronoBehavior.MOMENTUM_EXHAUSTING
        
        # Overlap periods
        elif 'OVERLAP' in self.current_session.name:
            return ChronoBehavior.BREAKOUT_LIKELY
        
        # Quiet periods
        elif self.current_session == TradingSession.QUIET_PERIOD:
            return ChronoBehavior.CONSOLIDATION
        
        return ChronoBehavior.CONSOLIDATION
    
    def calculate_temporal_strength(self) -> float:
        """Calculate overall temporal strength/influence"""
        
        session_profile = self.session_profiles[self.current_session]
        temporal_regime = self.get_temporal_regime()
        
        # Base strength from session characteristics
        base_strength = (
            session_profile.typical_volatility * 0.3 +
            session_profile.institutional_activity * 0.4 +
            session_profile.breakout_probability * 0.3
        )
        
        # Regime multipliers
        regime_multipliers = {
            TemporalRegime.HIGH_ACTIVITY: 1.5,
            TemporalRegime.MEDIUM_ACTIVITY: 1.0,
            TemporalRegime.LOW_ACTIVITY: 0.5,
            TemporalRegime.TRANSITION: 1.2,
            TemporalRegime.EVENT_DRIVEN: 1.8
        }
        
        regime_strength = base_strength * regime_multipliers[temporal_regime]
        
        # Pattern reinforcement
        pattern_strength = 0.0
        if self.detected_patterns:
            pattern_strength = np.mean([p.strength * p.confidence for p in self.detected_patterns])
        
        # Event horizon influence
        event_strength = 0.0
        if self.event_horizon:
            for event in self.event_horizon:
                time_factor = max(0, 1.0 - (event.time_to_event.total_seconds() / 7200))  # 2 hours
                event_strength += event.importance * event.expected_volatility_impact * time_factor
        
        # Combine all factors
        total_strength = (
            regime_strength * 0.5 +
            pattern_strength * 0.3 +
            event_strength * 0.2
        )
        
        return min(float(total_strength), 1.0)
    
    def calculate_temporal_confidence(self) -> float:
        """Calculate confidence in temporal analysis"""
        
        confidence_factors = []
        
        # Data quality (amount of historical data)
        total_data_points = sum(len(data) for data in self.session_data.values())
        data_quality = min(total_data_points / 100, 1.0)  # Normalize to 100 data points
        confidence_factors.append(data_quality * 0.3)
        
        # Pattern consistency
        if self.detected_patterns:
            pattern_confidence = np.mean([p.confidence * p.historical_accuracy for p in self.detected_patterns])
            confidence_factors.append(pattern_confidence * 0.4)
        
        # Session profile reliability
        session_profile = self.session_profiles[self.current_session]
        session_data = self.session_data.get(self.current_session, [])
        
        if len(session_data) >= 5:
            # Calculate how well current session matches historical profile
            recent_volatilities = [d['avg_volatility'] for d in session_data[-5:]]
            expected_volatility = session_profile.typical_volatility
            
            volatility_consistency = 1.0 - abs(np.mean(recent_volatilities) - expected_volatility) / expected_volatility
            volatility_consistency = max(0, min(float(volatility_consistency), 1.0))
            
            confidence_factors.append(volatility_consistency * 0.3)
        
        return np.sum(confidence_factors) if confidence_factors else 0.5

class ChronalIntelligenceEngine:
    """
    Main engine for temporal market intelligence
    Orchestrates all temporal analysis components
    """
    
    def __init__(self):
        self.temporal_analyzer = TemporalAnalyzer()
        
        # Current state
        self.current_temporal_regime = TemporalRegime.MEDIUM_ACTIVITY
        self.current_chrono_behavior = ChronoBehavior.CONSOLIDATION
        
    async def analyze_temporal_intelligence(self, market_data: MarketData) -> DimensionalReading:
        """Analyze temporal intelligence and return dimensional reading"""
        
        # Update temporal analysis
        self.temporal_analyzer.update_temporal_data(market_data)
        
        # Get current temporal state
        self.current_temporal_regime = self.temporal_analyzer.get_temporal_regime()
        self.current_chrono_behavior = self.temporal_analyzer.get_chrono_behavior()
        
        # Calculate temporal strength
        temporal_strength = self.temporal_analyzer.calculate_temporal_strength()
        
        # Calculate confidence
        confidence = self.temporal_analyzer.calculate_temporal_confidence()
        
        # Generate context
        context = self._generate_temporal_context(market_data)
        
        return DimensionalReading(
            dimension='WHEN',
            signal_strength=temporal_strength,
            confidence=confidence,
            regime=MarketRegime.UNKNOWN,  # Default regime for temporal analysis
            context=context,
            timestamp=market_data.timestamp
        )
    
    def _generate_temporal_context(self, market_data: Optional[MarketData] = None) -> Dict[str, Any]:
        """Generate contextual information about temporal analysis"""
        
        session_profile = self.temporal_analyzer.session_profiles[self.temporal_analyzer.current_session]
        
        context = {
            'current_session': self.temporal_analyzer.current_session.name,
            'temporal_regime': self.current_temporal_regime.name,
            'chrono_behavior': self.current_chrono_behavior.name,
            'session_characteristics': {
                'typical_volatility': session_profile.typical_volatility,
                'institutional_activity': session_profile.institutional_activity,
                'breakout_probability': session_profile.breakout_probability,
                'reversal_probability': session_profile.reversal_probability
            }
        }
        
        # Add detected patterns
        if self.temporal_analyzer.detected_patterns:
            context['detected_patterns'] = [
                {
                    'type': pattern.pattern_type,
                    'strength': pattern.strength,
                    'confidence': pattern.confidence,
                    'next_occurrence': pattern.next_occurrence.isoformat() if pattern.next_occurrence else None
                }
                for pattern in self.temporal_analyzer.detected_patterns
            ]
        
        # Add event horizon
        if self.temporal_analyzer.event_horizon:
            context['upcoming_events'] = [
                {
                    'name': event.event_name,
                    'time_to_event_minutes': event.time_to_event.total_seconds() / 60,
                    'importance': event.importance,
                    'expected_impact': event.expected_volatility_impact
                }
                for event in self.temporal_analyzer.event_horizon
            ]
        
        # Add session timing info
        if self.temporal_analyzer.session_start_time and market_data:
            start_time = self.temporal_analyzer.session_start_time
            current_time = market_data.timestamp
            
            if start_time.tzinfo is not None:
                start_time = start_time.replace(tzinfo=None)
            if current_time.tzinfo is not None:
                current_time = current_time.replace(tzinfo=None)
                
            session_duration = current_time - start_time
            context['session_info'] = {
                'duration_minutes': session_duration.total_seconds() / 60,
                'session_volatility': list(self.temporal_analyzer.session_volatility)[-5:] if self.temporal_analyzer.session_volatility else [],
                'session_volume': list(self.temporal_analyzer.session_volume)[-5:] if self.temporal_analyzer.session_volume else []
            }
        
        return context

# Example usage
async def main():
    """Example usage of the enhanced temporal intelligence engine"""
    
    # Initialize engine
    engine = ChronalIntelligenceEngine()
    
    # Simulate market data across different sessions
    base_time = datetime(2024, 1, 15, 8, 0)  # Monday, London open
    
    for i in range(500):  # Simulate 500 periods (about 8 hours)
        
        current_time = base_time + timedelta(minutes=i)
        
        # Create market data with session-appropriate characteristics
        hour = current_time.hour
        
        # Simulate higher volatility during overlaps
        if 13 <= hour < 17:  # London-NY overlap
            base_volatility = 0.015
            base_volume = 2500
        elif 8 <= hour < 13:  # London session
            base_volatility = 0.010
            base_volume = 1800
        else:  # Other periods
            base_volatility = 0.006
            base_volume = 1000
        
        # Add some randomness
        volatility = base_volatility + np.random.normal(0, base_volatility * 0.3)
        volume = base_volume + np.random.exponential(base_volume * 0.5)
        
        market_data = MarketData(
            timestamp=current_time,
            bid=1.0950 + np.random.normal(0, 0.001),
            ask=1.0952 + np.random.normal(0, 0.001),
            volume=volume,
            volatility=volatility
        )
        
        # Analyze temporal intelligence
        reading = await engine.analyze_temporal_intelligence(market_data)
        
        if i % 60 == 0:  # Print every hour
            print(f"Temporal Intelligence Reading (Hour {i//60}):")
            print(f"  Time: {current_time.strftime('%H:%M')}")
            print(f"  Value: {reading.signal_strength:.3f}")
            print(f"  Confidence: {reading.confidence:.3f}")
            print(f"  Session: {reading.context.get('current_session', 'Unknown')}")
            print(f"  Regime: {reading.context.get('temporal_regime', 'Unknown')}")
            print(f"  Behavior: {reading.context.get('chrono_behavior', 'Unknown')}")
            
            if 'detected_patterns' in reading.context:
                print(f"  Patterns: {len(reading.context['detected_patterns'])}")
            
            print()

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())

