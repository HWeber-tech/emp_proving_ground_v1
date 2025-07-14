"""
WHEN Dimension - Temporal Intelligence

This dimension understands the temporal aspects of market behavior:
- Trading session dynamics and overlaps
- Intraday patterns and rhythms
- Weekly and monthly seasonality
- Event timing and scheduled releases
- Market opening/closing effects
- Time-based volatility patterns

The WHEN dimension provides timing context that can amplify or dampen
the effects of other dimensions.
"""

import numpy as np
from datetime import datetime, timedelta, time
from typing import Dict, Any, Optional, List, Tuple, NamedTuple
from collections import deque, defaultdict
from dataclasses import dataclass
from enum import Enum, auto
import calendar
import math

from ..core.base import (
    DimensionalSensor, DimensionalReading, MarketData, MarketRegime
)


class TradingSession(Enum):
    ASIAN = auto()
    LONDON = auto()
    NEW_YORK = auto()
    OVERLAP_LONDON_NY = auto()
    OVERLAP_ASIAN_LONDON = auto()
    QUIET = auto()


class TimeOfDay(Enum):
    EARLY_ASIAN = auto()
    LATE_ASIAN = auto()
    EARLY_LONDON = auto()
    LATE_LONDON = auto()
    EARLY_NY = auto()
    LATE_NY = auto()
    OVERNIGHT = auto()


@dataclass
class SessionCharacteristics:
    """Characteristics of a trading session"""
    session: TradingSession
    avg_volatility: float
    avg_volume: float
    avg_spread: float
    trend_strength: float
    reversal_probability: float
    
    @property
    def activity_score(self) -> float:
        """Overall activity score for the session"""
        return (self.avg_volatility * 0.4 + 
                self.avg_volume * 0.3 + 
                (1.0 - self.avg_spread) * 0.3)


@dataclass
class ScheduledEvent:
    """Scheduled market event"""
    name: str
    timestamp: datetime
    importance: float  # 0-1 scale
    expected_impact: str  # 'bullish', 'bearish', 'neutral'
    currency: str
    
    @property
    def time_until_event(self) -> timedelta:
        return self.timestamp - datetime.now()
    
    @property
    def hours_until_event(self) -> float:
        return self.time_until_event.total_seconds() / 3600


class SessionAnalyzer:
    """Analyzes trading session patterns and characteristics"""
    
    def __init__(self):
        # Session time ranges (UTC)
        self.session_times = {
            TradingSession.ASIAN: (time(22, 0), time(8, 0)),      # 22:00-08:00 UTC
            TradingSession.LONDON: (time(7, 0), time(16, 0)),     # 07:00-16:00 UTC
            TradingSession.NEW_YORK: (time(12, 0), time(21, 0)),  # 12:00-21:00 UTC
        }
        
        # Historical session data
        self.session_data: Dict[TradingSession, deque] = {
            session: deque(maxlen=100) for session in TradingSession
        }
        
        # Current session tracking
        self.current_session_start: Optional[datetime] = None
        self.current_session_data: List[Tuple[float, float, float]] = []  # volatility, volume, spread
        
    def get_current_session(self, timestamp: datetime) -> TradingSession:
        """Determine current trading session"""
        utc_time = timestamp.time()
        
        # Check for overlaps first
        if self._is_in_session(utc_time, TradingSession.LONDON) and \
           self._is_in_session(utc_time, TradingSession.NEW_YORK):
            return TradingSession.OVERLAP_LONDON_NY
        
        if self._is_in_session(utc_time, TradingSession.ASIAN) and \
           self._is_in_session(utc_time, TradingSession.LONDON):
            return TradingSession.OVERLAP_ASIAN_LONDON
        
        # Check individual sessions
        for session in [TradingSession.ASIAN, TradingSession.LONDON, TradingSession.NEW_YORK]:
            if self._is_in_session(utc_time, session):
                return session
        
        return TradingSession.QUIET
    
    def _is_in_session(self, current_time: time, session: TradingSession) -> bool:
        """Check if current time is within session"""
        if session not in self.session_times:
            return False
        
        start_time, end_time = self.session_times[session]
        
        # Handle sessions that cross midnight
        if start_time > end_time:
            return current_time >= start_time or current_time < end_time
        else:
            return start_time <= current_time < end_time
    
    def update_session_data(self, timestamp: datetime, volatility: float, 
                          volume: float, spread: float) -> None:
        """Update session data"""
        current_session = self.get_current_session(timestamp)
        
        # Track session changes
        if (self.current_session_start is None or 
            self.get_current_session(self.current_session_start) != current_session):
            
            # Save previous session data
            if self.current_session_data and self.current_session_start:
                prev_session = self.get_current_session(self.current_session_start)
                self._save_session_summary(prev_session)
            
            # Start new session
            self.current_session_start = timestamp
            self.current_session_data = []
        
        # Add current data
        self.current_session_data.append((volatility, volume, spread))
    
    def _save_session_summary(self, session: TradingSession) -> None:
        """Save summary of completed session"""
        if not self.current_session_data:
            return
        
        volatilities, volumes, spreads = zip(*self.current_session_data)
        
        summary = SessionCharacteristics(
            session=session,
            avg_volatility=np.mean(volatilities),
            avg_volume=np.mean(volumes),
            avg_spread=np.mean(spreads),
            trend_strength=self._calculate_trend_strength(volatilities),
            reversal_probability=self._calculate_reversal_probability(volatilities)
        )
        
        self.session_data[session].append(summary)
    
    def _calculate_trend_strength(self, volatilities: List[float]) -> float:
        """Calculate trend strength during session"""
        if len(volatilities) < 2:
            return 0.5
        
        # Trend strength based on volatility consistency
        volatility_std = np.std(volatilities)
        volatility_mean = np.mean(volatilities)
        
        if volatility_mean > 0:
            consistency = 1.0 - (volatility_std / volatility_mean)
            return max(0.0, min(1.0, consistency))
        
        return 0.5
    
    def _calculate_reversal_probability(self, volatilities: List[float]) -> float:
        """Calculate probability of reversal at session end"""
        if len(volatilities) < 3:
            return 0.5
        
        # Higher volatility at end suggests potential reversal
        early_vol = np.mean(volatilities[:len(volatilities)//3])
        late_vol = np.mean(volatilities[-len(volatilities)//3:])
        
        if early_vol > 0:
            vol_ratio = late_vol / early_vol
            # Higher late volatility = higher reversal probability
            return min(1.0, vol_ratio / 2.0)
        
        return 0.5
    
    def get_session_characteristics(self, session: TradingSession) -> Optional[SessionCharacteristics]:
        """Get average characteristics for a session"""
        if session not in self.session_data or not self.session_data[session]:
            return None
        
        recent_sessions = list(self.session_data[session])[-20:]  # Last 20 sessions
        
        avg_volatility = np.mean([s.avg_volatility for s in recent_sessions])
        avg_volume = np.mean([s.avg_volume for s in recent_sessions])
        avg_spread = np.mean([s.avg_spread for s in recent_sessions])
        avg_trend_strength = np.mean([s.trend_strength for s in recent_sessions])
        avg_reversal_prob = np.mean([s.reversal_probability for s in recent_sessions])
        
        return SessionCharacteristics(
            session=session,
            avg_volatility=avg_volatility,
            avg_volume=avg_volume,
            avg_spread=avg_spread,
            trend_strength=avg_trend_strength,
            reversal_probability=avg_reversal_prob
        )
    
    def calculate_session_score(self, timestamp: datetime) -> Tuple[float, float]:
        """Calculate session-based score and confidence"""
        current_session = self.get_current_session(timestamp)
        characteristics = self.get_session_characteristics(current_session)
        
        if not characteristics:
            return 0.0, 0.0
        
        # Score based on session activity and characteristics
        activity_score = characteristics.activity_score
        
        # Adjust for session type
        session_multipliers = {
            TradingSession.OVERLAP_LONDON_NY: 1.0,    # Highest activity
            TradingSession.LONDON: 0.8,
            TradingSession.NEW_YORK: 0.8,
            TradingSession.OVERLAP_ASIAN_LONDON: 0.6,
            TradingSession.ASIAN: 0.4,
            TradingSession.QUIET: 0.1
        }
        
        multiplier = session_multipliers.get(current_session, 0.5)
        score = (activity_score - 0.5) * 2 * multiplier  # Normalize to [-1, 1]
        
        # Confidence based on data availability
        data_count = len(self.session_data.get(current_session, []))
        confidence = min(1.0, data_count / 20.0)
        
        return max(-1.0, min(1.0, score)), confidence


class IntradayPatternAnalyzer:
    """Analyzes intraday patterns and rhythms"""
    
    def __init__(self):
        # Hourly pattern data (24 hours)
        self.hourly_patterns: Dict[int, deque] = {
            hour: deque(maxlen=50) for hour in range(24)
        }
        
        # Day of week patterns
        self.daily_patterns: Dict[int, deque] = {
            day: deque(maxlen=20) for day in range(7)  # 0=Monday, 6=Sunday
        }
        
    def update_patterns(self, timestamp: datetime, price_change: float, 
                       volatility: float, volume: float) -> None:
        """Update intraday patterns"""
        hour = timestamp.hour
        day_of_week = timestamp.weekday()
        
        pattern_data = {
            'price_change': price_change,
            'volatility': volatility,
            'volume': volume,
            'timestamp': timestamp
        }
        
        self.hourly_patterns[hour].append(pattern_data)
        self.daily_patterns[day_of_week].append(pattern_data)
    
    def get_hourly_bias(self, hour: int) -> Tuple[float, float]:
        """Get directional bias for specific hour"""
        if hour not in self.hourly_patterns or not self.hourly_patterns[hour]:
            return 0.0, 0.0
        
        recent_data = list(self.hourly_patterns[hour])[-20:]
        price_changes = [d['price_change'] for d in recent_data]
        
        if not price_changes:
            return 0.0, 0.0
        
        avg_change = np.mean(price_changes)
        consistency = 1.0 - np.std(price_changes) if len(price_changes) > 1 else 0.5
        
        # Normalize and calculate confidence
        bias = max(-1.0, min(1.0, avg_change * 100))  # Scale up small price changes
        confidence = max(0.0, min(1.0, consistency * len(recent_data) / 20))
        
        return bias, confidence
    
    def get_daily_bias(self, day_of_week: int) -> Tuple[float, float]:
        """Get directional bias for specific day of week"""
        if day_of_week not in self.daily_patterns or not self.daily_patterns[day_of_week]:
            return 0.0, 0.0
        
        recent_data = list(self.daily_patterns[day_of_week])[-10:]
        price_changes = [d['price_change'] for d in recent_data]
        
        if not price_changes:
            return 0.0, 0.0
        
        avg_change = np.mean(price_changes)
        consistency = 1.0 - np.std(price_changes) if len(price_changes) > 1 else 0.5
        
        bias = max(-1.0, min(1.0, avg_change * 50))
        confidence = max(0.0, min(1.0, consistency * len(recent_data) / 10))
        
        return bias, confidence
    
    def calculate_time_momentum(self, timestamp: datetime) -> Tuple[float, float]:
        """Calculate momentum based on time patterns"""
        hour = timestamp.hour
        day_of_week = timestamp.weekday()
        
        hourly_bias, hourly_conf = self.get_hourly_bias(hour)
        daily_bias, daily_conf = self.get_daily_bias(day_of_week)
        
        # Weighted combination
        if hourly_conf > 0 and daily_conf > 0:
            combined_bias = (hourly_bias * hourly_conf + daily_bias * daily_conf) / (hourly_conf + daily_conf)
            combined_conf = (hourly_conf + daily_conf) / 2
        elif hourly_conf > 0:
            combined_bias, combined_conf = hourly_bias, hourly_conf
        elif daily_conf > 0:
            combined_bias, combined_conf = daily_bias, daily_conf
        else:
            combined_bias, combined_conf = 0.0, 0.0
        
        return combined_bias, combined_conf


class EventScheduler:
    """Manages scheduled economic events and their timing"""
    
    def __init__(self):
        self.scheduled_events: List[ScheduledEvent] = []
        self.event_impact_history: deque = deque(maxlen=100)
        
    def add_event(self, name: str, timestamp: datetime, importance: float,
                 expected_impact: str, currency: str) -> None:
        """Add scheduled event"""
        event = ScheduledEvent(
            name=name,
            timestamp=timestamp,
            importance=importance,
            expected_impact=expected_impact,
            currency=currency
        )
        self.scheduled_events.append(event)
        
        # Keep events sorted by time
        self.scheduled_events.sort(key=lambda e: e.timestamp)
        
        # Remove past events
        now = datetime.now()
        self.scheduled_events = [e for e in self.scheduled_events if e.timestamp > now]
    
    def get_upcoming_events(self, hours_ahead: float = 24.0) -> List[ScheduledEvent]:
        """Get events in the next N hours"""
        cutoff_time = datetime.now() + timedelta(hours=hours_ahead)
        return [e for e in self.scheduled_events if e.timestamp <= cutoff_time]
    
    def calculate_event_proximity_score(self, currency_pair: str = "EURUSD") -> Tuple[float, float]:
        """Calculate score based on proximity to important events"""
        relevant_currencies = ['EUR', 'USD'] if currency_pair == "EURUSD" else [currency_pair[:3], currency_pair[3:]]
        
        upcoming_events = self.get_upcoming_events(6.0)  # Next 6 hours
        relevant_events = [e for e in upcoming_events if e.currency in relevant_currencies]
        
        if not relevant_events:
            return 0.0, 0.0
        
        # Calculate weighted impact based on proximity and importance
        total_impact = 0.0
        total_weight = 0.0
        
        for event in relevant_events:
            hours_until = event.hours_until_event
            
            # Proximity weight (closer = higher weight)
            if hours_until <= 0.5:  # Within 30 minutes
                proximity_weight = 1.0
            elif hours_until <= 2.0:  # Within 2 hours
                proximity_weight = 0.8
            elif hours_until <= 6.0:  # Within 6 hours
                proximity_weight = 0.5
            else:
                proximity_weight = 0.1
            
            # Impact direction
            impact_multiplier = {'bullish': 1.0, 'bearish': -1.0, 'neutral': 0.0}.get(event.expected_impact, 0.0)
            
            # Currency weight (EUR events affect EURUSD differently than USD events)
            if currency_pair == "EURUSD":
                currency_weight = 1.0 if event.currency == 'EUR' else -1.0 if event.currency == 'USD' else 0.0
            else:
                currency_weight = 1.0
            
            event_impact = (event.importance * proximity_weight * 
                          impact_multiplier * currency_weight)
            
            total_impact += event_impact
            total_weight += proximity_weight * event.importance
        
        if total_weight > 0:
            score = total_impact / len(relevant_events)  # Average impact
            confidence = min(1.0, total_weight / len(relevant_events))
        else:
            score, confidence = 0.0, 0.0
        
        return max(-1.0, min(1.0, score)), confidence


class SeasonalityAnalyzer:
    """Analyzes seasonal patterns in market behavior"""
    
    def __init__(self):
        # Monthly patterns
        self.monthly_patterns: Dict[int, deque] = {
            month: deque(maxlen=24) for month in range(1, 13)
        }
        
        # Week of month patterns
        self.week_patterns: Dict[int, deque] = {
            week: deque(maxlen=20) for week in range(1, 6)  # 1-5 weeks in month
        }
        
    def update_seasonality(self, timestamp: datetime, price_change: float,
                         volatility: float) -> None:
        """Update seasonal patterns"""
        month = timestamp.month
        week_of_month = (timestamp.day - 1) // 7 + 1
        
        pattern_data = {
            'price_change': price_change,
            'volatility': volatility,
            'timestamp': timestamp
        }
        
        self.monthly_patterns[month].append(pattern_data)
        if week_of_month <= 5:
            self.week_patterns[week_of_month].append(pattern_data)
    
    def get_seasonal_bias(self, timestamp: datetime) -> Tuple[float, float]:
        """Get seasonal bias for current time"""
        month = timestamp.month
        week_of_month = min(5, (timestamp.day - 1) // 7 + 1)
        
        # Monthly bias
        monthly_bias, monthly_conf = self._get_monthly_bias(month)
        
        # Weekly bias
        weekly_bias, weekly_conf = self._get_weekly_bias(week_of_month)
        
        # Combine biases
        if monthly_conf > 0 and weekly_conf > 0:
            combined_bias = (monthly_bias * monthly_conf + weekly_bias * weekly_conf) / (monthly_conf + weekly_conf)
            combined_conf = (monthly_conf + weekly_conf) / 2
        elif monthly_conf > 0:
            combined_bias, combined_conf = monthly_bias, monthly_conf
        elif weekly_conf > 0:
            combined_bias, combined_conf = weekly_bias, weekly_conf
        else:
            combined_bias, combined_conf = 0.0, 0.0
        
        return combined_bias, combined_conf
    
    def _get_monthly_bias(self, month: int) -> Tuple[float, float]:
        """Get bias for specific month"""
        if month not in self.monthly_patterns or not self.monthly_patterns[month]:
            return 0.0, 0.0
        
        recent_data = list(self.monthly_patterns[month])[-12:]  # Last year
        price_changes = [d['price_change'] for d in recent_data]
        
        if not price_changes:
            return 0.0, 0.0
        
        avg_change = np.mean(price_changes)
        consistency = 1.0 - np.std(price_changes) if len(price_changes) > 1 else 0.5
        
        bias = max(-1.0, min(1.0, avg_change * 20))
        confidence = max(0.0, min(1.0, consistency * len(recent_data) / 12))
        
        return bias, confidence
    
    def _get_weekly_bias(self, week: int) -> Tuple[float, float]:
        """Get bias for specific week of month"""
        if week not in self.week_patterns or not self.week_patterns[week]:
            return 0.0, 0.0
        
        recent_data = list(self.week_patterns[week])[-10:]
        price_changes = [d['price_change'] for d in recent_data]
        
        if not price_changes:
            return 0.0, 0.0
        
        avg_change = np.mean(price_changes)
        consistency = 1.0 - np.std(price_changes) if len(price_changes) > 1 else 0.5
        
        bias = max(-1.0, min(1.0, avg_change * 30))
        confidence = max(0.0, min(1.0, consistency * len(recent_data) / 10))
        
        return bias, confidence


class WhenDimension(DimensionalSensor):
    """
    WHEN Dimension - Temporal Intelligence
    
    Understands the temporal aspects of market behavior:
    - Trading session dynamics and characteristics
    - Intraday patterns and rhythms
    - Seasonal effects and calendar patterns
    - Event timing and scheduled releases
    - Time-based volatility and volume patterns
    """
    
    def __init__(self):
        super().__init__("WHEN")
        
        # Component analyzers
        self.session_analyzer = SessionAnalyzer()
        self.pattern_analyzer = IntradayPatternAnalyzer()
        self.event_scheduler = EventScheduler()
        self.seasonality_analyzer = SeasonalityAnalyzer()
        
        # Previous data for change calculation
        self.previous_price: Optional[float] = None
        
        # Synthesis weights
        self.component_weights = {
            'session': 0.35,
            'intraday': 0.25,
            'events': 0.25,
            'seasonality': 0.15
        }
        
        # Peer influence weights
        self.peer_influences = {
            'why': 0.10,    # Fundamentals can be time-sensitive
            'how': 0.15,    # Institutional activity has timing patterns
            'what': 0.20,   # Technical levels can be time-dependent
            'anomaly': 0.05 # Anomalies can disrupt timing patterns
        }
    
    def process(self, data: MarketData, peer_readings: Dict[str, DimensionalReading]) -> DimensionalReading:
        """Process market data to understand temporal dynamics"""
        
        # Calculate price change
        price_change = 0.0
        if self.previous_price is not None:
            price_change = (data.mid_price - self.previous_price) / self.previous_price
        self.previous_price = data.mid_price
        
        # Update analyzers
        volatility = data.volatility if data.volatility > 0 else abs(price_change)
        volume_normalized = data.volume / 1000.0  # Normalize volume
        
        self.session_analyzer.update_session_data(
            data.timestamp, volatility, volume_normalized, data.spread
        )
        
        self.pattern_analyzer.update_patterns(
            data.timestamp, price_change, volatility, volume_normalized
        )
        
        self.seasonality_analyzer.update_seasonality(
            data.timestamp, price_change, volatility
        )
        
        # Calculate component scores
        session_score, session_conf = self.session_analyzer.calculate_session_score(data.timestamp)
        intraday_score, intraday_conf = self.pattern_analyzer.calculate_time_momentum(data.timestamp)
        event_score, event_conf = self.event_scheduler.calculate_event_proximity_score()
        seasonal_score, seasonal_conf = self.seasonality_analyzer.get_seasonal_bias(data.timestamp)
        
        # Weighted synthesis
        components = {
            'session': (session_score, session_conf),
            'intraday': (intraday_score, intraday_conf),
            'events': (event_score, event_conf),
            'seasonality': (seasonal_score, seasonal_conf)
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
        final_score = base_score + peer_adjustment * 0.25  # Moderate peer influence
        final_confidence = base_confidence + peer_confidence_boost * 0.15
        
        # Normalize
        final_score = max(-1.0, min(1.0, final_score))
        final_confidence = max(0.0, min(1.0, final_confidence))
        
        # Build context
        current_session = self.session_analyzer.get_current_session(data.timestamp)
        upcoming_events = self.event_scheduler.get_upcoming_events(2.0)
        
        context = {
            'current_session': current_session.name,
            'hour_of_day': data.timestamp.hour,
            'day_of_week': data.timestamp.weekday(),
            'month': data.timestamp.month,
            'upcoming_events_count': len(upcoming_events),
            'next_event_hours': upcoming_events[0].hours_until_event if upcoming_events else None,
            'component_scores': {k: v[0] for k, v in components.items()},
            'component_confidences': {k: v[1] for k, v in components.items()},
            'peer_adjustment': peer_adjustment,
            'price_change': price_change,
            'volatility': volatility
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
    
    def add_scheduled_event(self, name: str, timestamp: datetime, importance: float,
                          expected_impact: str, currency: str) -> None:
        """Add scheduled economic event"""
        self.event_scheduler.add_event(name, timestamp, importance, expected_impact, currency)
    
    def get_temporal_summary(self) -> Dict[str, Any]:
        """Get comprehensive temporal analysis summary"""
        now = datetime.now()
        current_session = self.session_analyzer.get_current_session(now)
        upcoming_events = self.event_scheduler.get_upcoming_events(24.0)
        
        hourly_bias, hourly_conf = self.pattern_analyzer.get_hourly_bias(now.hour)
        daily_bias, daily_conf = self.pattern_analyzer.get_daily_bias(now.weekday())
        seasonal_bias, seasonal_conf = self.seasonality_analyzer.get_seasonal_bias(now)
        
        return {
            'current_session': current_session.name,
            'session_characteristics': self.session_analyzer.get_session_characteristics(current_session),
            'hourly_bias': {'bias': hourly_bias, 'confidence': hourly_conf},
            'daily_bias': {'bias': daily_bias, 'confidence': daily_conf},
            'seasonal_bias': {'bias': seasonal_bias, 'confidence': seasonal_conf},
            'upcoming_events': [
                {
                    'name': event.name,
                    'hours_until': event.hours_until_event,
                    'importance': event.importance,
                    'expected_impact': event.expected_impact,
                    'currency': event.currency
                }
                for event in upcoming_events[:5]  # Next 5 events
            ],
            'time_context': {
                'hour': now.hour,
                'day_of_week': now.weekday(),
                'month': now.month,
                'week_of_month': (now.day - 1) // 7 + 1
            }
        }

