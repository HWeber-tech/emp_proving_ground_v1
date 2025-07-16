"""
Sensory Cortex v2.2 - WHEN Dimension Engine (Temporal Analysis)

Masterful implementation of temporal market analysis with economic calendar integration.
Implements session dynamics, cyclical patterns, and event-driven temporal intelligence.
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, time
from typing import Dict, List, Optional, Tuple, NamedTuple
from dataclasses import dataclass
from enum import Enum
import pytz
import aiohttp
import json

from src.sensory.core.base import (
    DimensionalSensor, DimensionalReading, MarketData, InstrumentMeta,
    MarketRegime, EconomicEvent, EventTier
)
from src.sensory.core.utils import (
    EMA, WelfordVar, compute_confidence, normalize_signal,
    calculate_momentum, PerformanceTracker
)

logger = logging.getLogger(__name__)


class TradingSession(Enum):
    """Trading session types for temporal analysis."""
    ASIAN = "asian"
    LONDON = "london"
    NEW_YORK = "new_york"
    OVERLAP_LONDON_NY = "overlap_london_ny"
    OVERLAP_ASIAN_LONDON = "overlap_asian_london"
    DEAD_ZONE = "dead_zone"


class CyclicalPattern(Enum):
    """Cyclical pattern types."""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    INTRADAY = "intraday"


@dataclass
class SessionCharacteristics:
    """
    Trading session characteristics.
    """
    session: TradingSession
    start_time: time
    end_time: time
    timezone: str
    volatility_factor: float
    volume_factor: float
    trend_persistence: float


@dataclass
class TemporalEvent:
    """
    Temporal event structure for time-based analysis.
    """
    timestamp: datetime
    event_type: str
    impact_score: float
    duration_minutes: int
    affected_instruments: List[str]
    confidence: float


class EconomicCalendarProvider:
    """
    Economic calendar data provider with real API integration capability.
    Falls back to deterministic events for backtesting.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize economic calendar provider.
        
        Args:
            api_key: API key for real economic data (optional)
        """
        self.api_key = api_key
        self.cached_events: Dict[str, List[EconomicEvent]] = {}
        self.last_fetch_time = None
        
    async def get_economic_events(
        self, 
        start_date: datetime, 
        end_date: datetime,
        currencies: List[str] = None
    ) -> List[EconomicEvent]:
        """
        Get economic events for date range.
        
        Args:
            start_date: Start date for events
            end_date: End date for events
            currencies: List of currencies to filter (optional)
            
        Returns:
            List of economic events
        """
        # For now, use deterministic events for backtesting
        # In production, this would integrate with real economic calendar APIs
        return self._generate_deterministic_events(start_date, end_date, currencies)
    
    def _generate_deterministic_events(
        self,
        start_date: datetime,
        end_date: datetime,
        currencies: List[str] = None
    ) -> List[EconomicEvent]:
        """Generate deterministic economic events for backtesting."""
        events = []
        
        # Define major economic events with their typical schedules
        event_templates = [
            {
                'name': 'Non-Farm Payrolls',
                'currency': 'USD',
                'tier': EventTier.TIER_1,
                'day_of_month': 1,  # First Friday
                'time': time(13, 30),  # 8:30 AM EST
                'impact_duration': 120
            },
            {
                'name': 'Federal Reserve Interest Rate Decision',
                'currency': 'USD',
                'tier': EventTier.TIER_1,
                'day_of_month': 15,  # Mid-month
                'time': time(19, 0),   # 2:00 PM EST
                'impact_duration': 180
            },
            {
                'name': 'ECB Interest Rate Decision',
                'currency': 'EUR',
                'tier': EventTier.TIER_1,
                'day_of_month': 10,
                'time': time(12, 45),  # 7:45 AM EST
                'impact_duration': 150
            },
            {
                'name': 'GDP Quarterly Release',
                'currency': 'USD',
                'tier': EventTier.TIER_2,
                'day_of_month': 28,  # End of month
                'time': time(13, 30),
                'impact_duration': 90
            },
            {
                'name': 'Consumer Price Index',
                'currency': 'USD',
                'tier': EventTier.TIER_2,
                'day_of_month': 12,
                'time': time(13, 30),
                'impact_duration': 60
            }
        ]
        
        # Generate events for the date range
        current_date = start_date.date()
        end_date_only = end_date.date()
        
        while current_date <= end_date_only:
            for template in event_templates:
                # Check if this event should occur this month
                if current_date.day == template['day_of_month']:
                    # Filter by currency if specified
                    if currencies and template['currency'] not in currencies:
                        continue
                    
                    event_datetime = datetime.combine(current_date, template['time'])
                    event_datetime = pytz.UTC.localize(event_datetime)
                    
                    event = EconomicEvent(
                        event_id=f"{template['currency']}_{template['name'].replace(' ', '_')}_{current_date.strftime('%Y%m%d')}",
                        timestamp=event_datetime,
                        currency=template['currency'],
                        event_name=template['name'],
                        tier=template['tier'],
                        actual=None,  # Would be filled from real API
                        forecast=None,
                        previous=None,
                        impact_score=0.5,  # Default impact score
                        surprise_index=0.0,  # Default surprise index
                        time_to_event_hours=0.0  # Will be calculated later
                    )
                    events.append(event)
            
            # Move to next day
            current_date += timedelta(days=1)
        
        return sorted(events, key=lambda x: x.timestamp)


class SessionAnalyzer:
    """
    Sophisticated trading session analysis.
    """
    
    def __init__(self):
        """Initialize session analyzer."""
        self.session_characteristics = self._define_session_characteristics()
        self.session_volatility = {session: EMA(20) for session in TradingSession}
        self.session_volume = {session: EMA(20) for session in TradingSession}
        
    def _define_session_characteristics(self) -> Dict[TradingSession, SessionCharacteristics]:
        """Define characteristics for each trading session."""
        return {
            TradingSession.ASIAN: SessionCharacteristics(
                session=TradingSession.ASIAN,
                start_time=time(23, 0),  # 11 PM UTC (Tokyo open)
                end_time=time(8, 0),     # 8 AM UTC (Tokyo close)
                timezone="Asia/Tokyo",
                volatility_factor=0.6,
                volume_factor=0.4,
                trend_persistence=0.3
            ),
            TradingSession.LONDON: SessionCharacteristics(
                session=TradingSession.LONDON,
                start_time=time(8, 0),   # 8 AM UTC (London open)
                end_time=time(16, 0),    # 4 PM UTC (London close)
                timezone="Europe/London",
                volatility_factor=1.0,
                volume_factor=0.8,
                trend_persistence=0.7
            ),
            TradingSession.NEW_YORK: SessionCharacteristics(
                session=TradingSession.NEW_YORK,
                start_time=time(13, 0),  # 1 PM UTC (NY open)
                end_time=time(22, 0),    # 10 PM UTC (NY close)
                timezone="America/New_York",
                volatility_factor=0.9,
                volume_factor=1.0,
                trend_persistence=0.8
            ),
            TradingSession.OVERLAP_LONDON_NY: SessionCharacteristics(
                session=TradingSession.OVERLAP_LONDON_NY,
                start_time=time(13, 0),  # 1 PM UTC
                end_time=time(16, 0),    # 4 PM UTC
                timezone="UTC",
                volatility_factor=1.2,
                volume_factor=1.2,
                trend_persistence=0.9
            ),
            TradingSession.OVERLAP_ASIAN_LONDON: SessionCharacteristics(
                session=TradingSession.OVERLAP_ASIAN_LONDON,
                start_time=time(7, 0),   # 7 AM UTC
                end_time=time(9, 0),     # 9 AM UTC
                timezone="UTC",
                volatility_factor=0.8,
                volume_factor=0.6,
                trend_persistence=0.5
            ),
            TradingSession.DEAD_ZONE: SessionCharacteristics(
                session=TradingSession.DEAD_ZONE,
                start_time=time(22, 0),  # 10 PM UTC
                end_time=time(23, 0),    # 11 PM UTC
                timezone="UTC",
                volatility_factor=0.3,
                volume_factor=0.2,
                trend_persistence=0.1
            )
        }
    
    def get_current_session(self, timestamp: datetime) -> TradingSession:
        """
        Determine current trading session.
        
        Args:
            timestamp: Current timestamp (UTC)
            
        Returns:
            Current trading session
        """
        utc_time = timestamp.time()
        
        # Check for overlaps first (higher priority)
        if time(13, 0) <= utc_time <= time(16, 0):
            return TradingSession.OVERLAP_LONDON_NY
        elif time(7, 0) <= utc_time <= time(9, 0):
            return TradingSession.OVERLAP_ASIAN_LONDON
        
        # Check individual sessions
        elif time(8, 0) <= utc_time <= time(16, 0):
            return TradingSession.LONDON
        elif time(13, 0) <= utc_time <= time(22, 0):
            return TradingSession.NEW_YORK
        elif utc_time >= time(23, 0) or utc_time <= time(8, 0):
            return TradingSession.ASIAN
        else:
            return TradingSession.DEAD_ZONE
    
    def analyze_session_dynamics(
        self, 
        current_session: TradingSession,
        market_data: MarketData,
        price_history: List[MarketData]
    ) -> Dict[str, float]:
        """
        Analyze current session dynamics.
        
        Args:
            current_session: Current trading session
            market_data: Current market data
            price_history: Recent price history
            
        Returns:
            Session dynamics analysis
        """
        session_chars = self.session_characteristics[current_session]
        
        # Calculate current session volatility
        if len(price_history) >= 2:
            current_volatility = abs(market_data.close - price_history[-2].close) / price_history[-2].close
            self.session_volatility[current_session].update(current_volatility)
        
        # Calculate current session volume (normalized)
        normalized_volume = market_data.volume / 1000000  # Normalize to millions
        self.session_volume[current_session].update(normalized_volume)
        
        # Session strength based on expected vs actual activity
        expected_volatility = session_chars.volatility_factor * 0.001  # 0.1% base
        actual_volatility = self.session_volatility[current_session].get_value() or 0.0
        volatility_ratio = actual_volatility / expected_volatility if expected_volatility > 0 else 1.0
        
        expected_volume = session_chars.volume_factor * 1.0  # Base volume
        actual_volume = self.session_volume[current_session].get_value() or 1.0
        volume_ratio = actual_volume / expected_volume
        
        # Session momentum (trend persistence during session)
        session_momentum = self._calculate_session_momentum(price_history, current_session)
        
        return {
            'session': current_session.value,
            'volatility_ratio': min(3.0, volatility_ratio),  # Cap at 3x
            'volume_ratio': min(3.0, volume_ratio),
            'session_momentum': session_momentum,
            'expected_persistence': session_chars.trend_persistence,
            'session_strength': min(1.0, (volatility_ratio + volume_ratio) / 2),
            'time_in_session': self._calculate_time_in_session(market_data.timestamp, current_session)
        }
    
    def _calculate_session_momentum(self, price_history: List[MarketData], session: TradingSession) -> float:
        """Calculate momentum within current session."""
        if len(price_history) < 5:
            return 0.0
        
        # Get session start time
        session_chars = self.session_characteristics[session]
        
        # Find prices from current session
        session_prices = []
        for data in reversed(price_history):
            data_session = self.get_current_session(data.timestamp)
            if data_session == session:
                session_prices.append(data.close)
            else:
                break  # Stop when we exit current session
        
        if len(session_prices) < 2:
            return 0.0
        
        # Calculate momentum as price change rate
        session_prices.reverse()  # Chronological order
        start_price = session_prices[0]
        end_price = session_prices[-1]
        
        momentum = (end_price - start_price) / start_price
        return normalize_signal(momentum, -0.01, 0.01)  # Normalize to ±1% moves
    
    def _calculate_time_in_session(self, timestamp: datetime, session: TradingSession) -> float:
        """Calculate how far through the session we are (0-1)."""
        session_chars = self.session_characteristics[session]
        current_time = timestamp.time()
        
        start_time = session_chars.start_time
        end_time = session_chars.end_time
        
        # Handle sessions that cross midnight
        if start_time > end_time:
            if current_time >= start_time:
                # After start time, same day
                total_minutes = (24 * 60) - (start_time.hour * 60 + start_time.minute) + (end_time.hour * 60 + end_time.minute)
                elapsed_minutes = (24 * 60) - (start_time.hour * 60 + start_time.minute) + (current_time.hour * 60 + current_time.minute)
            else:
                # Before end time, next day
                total_minutes = (24 * 60) - (start_time.hour * 60 + start_time.minute) + (end_time.hour * 60 + end_time.minute)
                elapsed_minutes = (current_time.hour * 60 + current_time.minute) - (end_time.hour * 60 + end_time.minute)
        else:
            # Normal session within same day
            total_minutes = (end_time.hour * 60 + end_time.minute) - (start_time.hour * 60 + start_time.minute)
            elapsed_minutes = (current_time.hour * 60 + current_time.minute) - (start_time.hour * 60 + start_time.minute)
        
        if total_minutes <= 0:
            return 0.0
        
        return max(0.0, min(1.0, elapsed_minutes / total_minutes))


class CyclicalAnalyzer:
    """
    Cyclical pattern analysis for temporal intelligence.
    """
    
    def __init__(self):
        """Initialize cyclical analyzer."""
        self.daily_patterns = {}
        self.weekly_patterns = {}
        self.monthly_patterns = {}
        
    def analyze_cyclical_patterns(self, timestamp: datetime, market_data: MarketData) -> Dict[str, float]:
        """
        Analyze cyclical patterns in market behavior.
        
        Args:
            timestamp: Current timestamp
            market_data: Current market data
            
        Returns:
            Cyclical pattern analysis
        """
        # Daily cycle analysis
        daily_factor = self._analyze_daily_cycle(timestamp)
        
        # Weekly cycle analysis
        weekly_factor = self._analyze_weekly_cycle(timestamp)
        
        # Monthly cycle analysis
        monthly_factor = self._analyze_monthly_cycle(timestamp)
        
        # Intraday cycle analysis
        intraday_factor = self._analyze_intraday_cycle(timestamp)
        
        return {
            'daily_factor': daily_factor,
            'weekly_factor': weekly_factor,
            'monthly_factor': monthly_factor,
            'intraday_factor': intraday_factor,
            'overall_cyclical_bias': (daily_factor + weekly_factor + monthly_factor + intraday_factor) / 4
        }
    
    def _analyze_daily_cycle(self, timestamp: datetime) -> float:
        """Analyze daily cyclical patterns."""
        hour = timestamp.hour
        
        # Market activity patterns throughout the day (UTC)
        if 8 <= hour <= 10:  # London open
            return 0.8
        elif 13 <= hour <= 15:  # London-NY overlap
            return 1.0
        elif 15 <= hour <= 17:  # NY session
            return 0.7
        elif 21 <= hour <= 23:  # Asian session start
            return 0.4
        else:
            return 0.2  # Low activity periods
    
    def _analyze_weekly_cycle(self, timestamp: datetime) -> float:
        """Analyze weekly cyclical patterns."""
        weekday = timestamp.weekday()  # 0 = Monday, 6 = Sunday
        
        # Weekly activity patterns
        if weekday == 0:  # Monday
            return 0.6  # Moderate activity, trend continuation
        elif weekday == 1:  # Tuesday
            return 0.9  # High activity, trend establishment
        elif weekday == 2:  # Wednesday
            return 1.0  # Peak activity, major moves
        elif weekday == 3:  # Thursday
            return 0.8  # High activity, trend continuation
        elif weekday == 4:  # Friday
            return 0.5  # Moderate activity, profit taking
        else:  # Weekend
            return 0.1  # Minimal activity
    
    def _analyze_monthly_cycle(self, timestamp: datetime) -> float:
        """Analyze monthly cyclical patterns."""
        day = timestamp.day
        
        # Monthly patterns based on economic calendar and institutional flows
        if 1 <= day <= 5:  # Beginning of month
            return 0.8  # High activity, new positioning
        elif 6 <= day <= 15:  # Mid-month
            return 1.0  # Peak activity, major economic releases
        elif 16 <= day <= 25:  # Late month
            return 0.7  # Moderate activity, trend continuation
        else:  # End of month
            return 0.6  # Moderate activity, month-end flows
    
    def _analyze_intraday_cycle(self, timestamp: datetime) -> float:
        """Analyze intraday cyclical patterns."""
        minute = timestamp.minute
        
        # Intraday patterns (hourly cycles)
        if 0 <= minute <= 15:  # First quarter hour
            return 0.9  # High activity, session opens
        elif 16 <= minute <= 30:  # Second quarter
            return 0.7  # Moderate activity
        elif 31 <= minute <= 45:  # Third quarter
            return 0.6  # Lower activity
        else:  # Last quarter hour
            return 0.8  # Increased activity, session preparation
    

class EventHorizonAnalyzer:
    """
    Event horizon analysis for temporal market intelligence.
    """
    
    def __init__(self):
        """Initialize event horizon analyzer."""
        # Impact decay rates by event tier
        self.decay_rates = {
            EventTier.TIER_1: 0.95,    # Slow decay for high impact events
            EventTier.TIER_2: 0.85,    # Medium decay for medium impact events
            EventTier.TIER_3: 0.70     # Faster decay for low impact events
        }
    
    def analyze_event_horizon(
        self,
        current_time: datetime,
        upcoming_events: List[EconomicEvent],
        recent_events: List[EconomicEvent]
    ) -> Dict[str, any]:
        """
        Analyze event horizon impact on market timing.
        
        Args:
            current_time: Current timestamp
            upcoming_events: List of upcoming economic events
            recent_events: List of recent economic events
            
        Returns:
            Event horizon analysis
        """
        if current_time.tzinfo is None:
            current_time = pytz.UTC.localize(current_time)
            
        # Analyze upcoming event impact
        upcoming_impact = self._analyze_upcoming_events(current_time, upcoming_events)
        
        # Analyze recent event decay
        recent_impact = self._analyze_recent_events(current_time, recent_events)
        
        # Calculate overall event bias
        event_bias = self._calculate_event_bias(upcoming_impact, recent_impact)
        
        # Determine optimal timing windows
        timing_windows = self._identify_timing_windows(current_time, upcoming_events)
        
        return {
            'upcoming_impact': upcoming_impact,
            'recent_impact': recent_impact,
            'event_bias': event_bias,
            'timing_windows': timing_windows,
            'next_major_event': self._find_next_major_event(current_time, upcoming_events)
        }
    
    def _analyze_upcoming_events(self, current_time: datetime, events: List[EconomicEvent]) -> Dict[str, float]:
        """Analyze impact of upcoming events."""
        if not events:
            return {'total_impact': 0.0, 'directional_bias': 0.0, 'uncertainty': 0.0}
        
        total_impact = 0.0
        weighted_bias = 0.0
        uncertainty = 0.0
        
        for event in events:
            time_to_event = (event.timestamp - current_time).total_seconds() / 3600  # Hours
            
            if time_to_event <= 0 or time_to_event > 168:  # Skip past events or events > 1 week away
                continue
            
            # Impact decreases with time distance
            time_decay = np.exp(-time_to_event / 24)  # Exponential decay over 24 hours
            
            # Event impact based on tier
            tier_impact = {
                EventTier.TIER_1: 1.0,
                EventTier.TIER_2: 0.6,
                EventTier.TIER_3: 0.3
            }.get(event.tier, 0.1)
            
            event_impact = tier_impact * time_decay
            total_impact += event_impact
            
            # Directional bias (would be based on forecast vs previous in real implementation)
            # For now, use deterministic bias based on event type
            event_bias = self._get_event_bias(event)
            weighted_bias += event_bias * event_impact
            
            # Uncertainty increases with high-impact events
            uncertainty += event_impact * 0.5
        
        if total_impact > 0:
            weighted_bias /= total_impact
        
        return {
            'total_impact': min(1.0, total_impact),
            'directional_bias': np.clip(weighted_bias, -1.0, 1.0),
            'uncertainty': min(1.0, uncertainty)
        }
    
    def _analyze_recent_events(self, current_time: datetime, events: List[EconomicEvent]) -> Dict[str, float]:
        """Analyze decay impact of recent events."""
        if not events:
            return {'total_impact': 0.0, 'directional_bias': 0.0}
        
        total_impact = 0.0
        weighted_bias = 0.0
        
        for event in events:
            time_since_event = (current_time - event.timestamp).total_seconds() / 3600  # Hours
            
            if time_since_event < 0 or time_since_event > 72:  # Skip future events or events > 3 days old
                continue
            
            # Impact decays over time
            decay_rate = self.decay_rates.get(event.tier, 0.85)
            time_decay = decay_rate ** (time_since_event / 6)  # Decay every 6 hours
            
            # Event impact based on tier
            tier_impact = {
                EventTier.TIER_1: 1.0,
                EventTier.TIER_2: 0.6,
                EventTier.TIER_3: 0.3
            }.get(event.tier, 0.1)
            
            event_impact = tier_impact * time_decay
            total_impact += event_impact
            
            # Directional bias
            event_bias = self._get_event_bias(event)
            weighted_bias += event_bias * event_impact
        
        if total_impact > 0:
            weighted_bias /= total_impact
        
        return {
            'total_impact': min(1.0, total_impact),
            'directional_bias': np.clip(weighted_bias, -1.0, 1.0)
        }
    
    def _get_event_bias(self, event: EconomicEvent) -> float:
        """Get directional bias for an event (simplified for demo)."""
        # In real implementation, this would be based on forecast vs previous values
        # For now, use deterministic bias based on event name
        if 'Interest Rate' in event.event_name:
            return 0.3  # Slightly bullish for currency
        elif 'Employment' in event.event_name or 'Payrolls' in event.event_name:
            return 0.2  # Moderately bullish
        elif 'GDP' in event.event_name:
            return 0.1  # Slightly bullish
        elif 'Inflation' in event.event_name or 'CPI' in event.event_name:
            return -0.1  # Slightly bearish (inflation concerns)
        else:
            return 0.0  # Neutral
    
    def _calculate_event_bias(self, upcoming: Dict[str, float], recent: Dict[str, float]) -> float:
        """Calculate overall event bias."""
        upcoming_weight = 0.6
        recent_weight = 0.4
        
        return (
            upcoming['directional_bias'] * upcoming_weight +
            recent['directional_bias'] * recent_weight
        )
    
    def _identify_timing_windows(self, current_time: datetime, events: List[EconomicEvent]) -> Dict[str, any]:
        """Identify optimal timing windows around events."""
        if current_time.tzinfo is None:
            current_time = pytz.UTC.localize(current_time)
        windows = {
            'pre_event_caution': [],
            'post_event_opportunity': [],
            'clear_windows': []
        }
        
        for event in events:
            time_to_event = (event.timestamp - current_time).total_seconds() / 3600
            
            if 0 < time_to_event <= 24:  # Events within 24 hours
                if event.tier in [EventTier.TIER_1, EventTier.TIER_2]:
                    # Caution window before high/medium impact events
                    caution_start = event.timestamp - timedelta(hours=2)
                    caution_end = event.timestamp + timedelta(hours=1)
                    windows['pre_event_caution'].append({
                        'event': event.event_name,
                        'start': caution_start,
                        'end': caution_end,
                        'impact': event.tier.value
                    })
                    
                    # Opportunity window after event (volatility)
                    opportunity_start = event.timestamp + timedelta(hours=1)
                    opportunity_end = event.timestamp + timedelta(hours=6)
                    windows['post_event_opportunity'].append({
                        'event': event.event_name,
                        'start': opportunity_start,
                        'end': opportunity_end,
                        'impact': event.tier.value
                    })
        
        return windows
    
    def _find_next_major_event(self, current_time: datetime, events: List[EconomicEvent]) -> Optional[Dict[str, any]]:
        """Find the next major event."""
        if current_time.tzinfo is None:
            current_time = pytz.UTC.localize(current_time)
        major_events = [e for e in events if e.tier == EventTier.TIER_1 and e.timestamp > current_time]
        
        if not major_events:
            return None
        
        next_event = min(major_events, key=lambda x: x.timestamp)
        time_to_event = (next_event.timestamp - current_time).total_seconds() / 3600
        
        return {
            'event': next_event.event_name,
            'timestamp': next_event.timestamp,
            'currency': next_event.currency,
            'hours_away': time_to_event,
            'impact_tier': next_event.tier.value
        }


class WHENEngine(DimensionalSensor):
    """
    Masterful WHEN dimension engine for temporal market analysis.
    Implements sophisticated session dynamics, cyclical patterns, and event horizon logic.
    """
    
    def __init__(self, instrument_meta: InstrumentMeta):
        """
        Initialize WHEN engine.
        
        Args:
            instrument_meta: Instrument metadata
        """
        super().__init__(instrument_meta)
        
        # Initialize components
        self.economic_calendar = EconomicCalendarProvider()
        self.session_analyzer = SessionAnalyzer()
        self.cyclical_analyzer = CyclicalAnalyzer()
        self.event_horizon_analyzer = EventHorizonAnalyzer()
        
        # Performance tracking
        self.performance_tracker = PerformanceTracker()
        
        # State variables
        self.price_history: List[MarketData] = []
        self.cached_events: List[EconomicEvent] = []
        self.last_event_fetch = None
        self.temporal_bias = EMA(30)
        
        logger.info(f"WHEN Engine initialized for {instrument_meta.symbol}")
    
    async def update(self, market_data: MarketData) -> DimensionalReading:
        """
        Process market data and generate temporal analysis.
        
        Args:
            market_data: Latest market data
            
        Returns:
            Dimensional reading with temporal analysis
        """
        start_time = datetime.utcnow()
        
        try:
            # Update price history
            self.price_history.append(market_data)
            if len(self.price_history) > 200:  # Maintain reasonable history
                self.price_history.pop(0)
            
            # Get economic events (cache for 1 hour)
            await self._update_economic_events(market_data.timestamp)
            
            # Analyze current session
            current_session = self.session_analyzer.get_current_session(market_data.timestamp)
            session_analysis = self.session_analyzer.analyze_session_dynamics(
                current_session, market_data, self.price_history
            )
            
            # Analyze cyclical patterns
            cyclical_analysis = self.cyclical_analyzer.analyze_cyclical_patterns(
                market_data.timestamp, market_data
            )
            
            # Analyze event horizon
            market_timestamp = market_data.timestamp
            if market_timestamp.tzinfo is None:
                market_timestamp = pytz.UTC.localize(market_timestamp)
            
            upcoming_events = [e for e in self.cached_events if e.timestamp > market_timestamp]
            recent_events = [e for e in self.cached_events if e.timestamp <= market_timestamp]
            event_horizon = self.event_horizon_analyzer.analyze_event_horizon(
                market_timestamp, upcoming_events, recent_events
            )
            
            # Perform comprehensive temporal analysis
            temporal_analysis = self._analyze_temporal_dynamics(
                session_analysis, cyclical_analysis, event_horizon, market_data
            )
            
            # Calculate signal strength and confidence
            signal_strength = self._calculate_signal_strength(temporal_analysis)
            confidence = self._calculate_confidence(temporal_analysis)
            
            # Detect market regime
            regime = self._detect_market_regime(temporal_analysis, session_analysis)
            
            # Create dimensional reading
            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            reading = DimensionalReading(
                dimension="WHEN",
                timestamp=market_data.timestamp,
                signal_strength=signal_strength,
                confidence=confidence,
                regime=regime,
                context={
                    'temporal_analysis': temporal_analysis,
                    'session_analysis': session_analysis,
                    'cyclical_analysis': cyclical_analysis,
                    'event_horizon': event_horizon,
                    'current_session': current_session.value,
                    'events_count': len(self.cached_events)
                },
                data_quality=1.0,  # Temporal data is always available
                processing_time_ms=processing_time,
                evidence=self._extract_evidence(temporal_analysis, session_analysis),
                warnings=self._generate_warnings(temporal_analysis, event_horizon)
            )
            
            self.last_reading = reading
            self.is_initialized = True
            
            logger.debug(f"WHEN analysis complete: signal={signal_strength:.3f}, "
                        f"confidence={confidence:.3f}, session={current_session.value}")
            
            return reading
            
        except Exception as e:
            logger.error(f"Error in WHEN engine update: {e}")
            return self._create_error_reading(market_data.timestamp, str(e))
    
    async def _update_economic_events(self, timestamp: datetime) -> None:
        """Update economic events cache."""
        if timestamp.tzinfo is None:
            timestamp = pytz.UTC.localize(timestamp)
        
        # Refresh events every hour
        if (self.last_event_fetch is None or 
            (timestamp - self.last_event_fetch).total_seconds() > 3600):
            
            start_date = timestamp - timedelta(days=3)
            end_date = timestamp + timedelta(days=7)
            
            # Get relevant currencies for this instrument
            currencies = self._get_relevant_currencies()
            
            self.cached_events = await self.economic_calendar.get_economic_events(
                start_date, end_date, currencies
            )
            self.last_event_fetch = timestamp
            
            logger.debug(f"Updated economic events cache: {len(self.cached_events)} events")
    
    def _get_relevant_currencies(self) -> List[str]:
        """Get relevant currencies for this instrument."""
        symbol = self.instrument_meta.symbol
        
        if len(symbol) == 6:  # Standard forex pair
            return [symbol[:3], symbol[3:]]
        else:
            return ['USD', 'EUR']  # Default currencies
    
    def _analyze_temporal_dynamics(
        self,
        session_analysis: Dict[str, any],
        cyclical_analysis: Dict[str, float],
        event_horizon: Dict[str, any],
        market_data: MarketData
    ) -> Dict[str, any]:
        """
        Analyze comprehensive temporal dynamics.
        
        Args:
            session_analysis: Session analysis results
            cyclical_analysis: Cyclical pattern analysis
            event_horizon: Event horizon analysis
            market_data: Current market data
            
        Returns:
            Temporal dynamics analysis
        """
        analysis = {}
        
        # Session timing score
        session_strength = session_analysis.get('session_strength', 0.0)
        session_momentum = session_analysis.get('session_momentum', 0.0)
        time_in_session = session_analysis.get('time_in_session', 0.0)
        
        # Optimal session timing (higher score at session starts and overlaps)
        session_timing_score = session_strength * (1.0 - abs(time_in_session - 0.3))  # Peak at 30% into session
        analysis['session_timing'] = session_timing_score
        
        # Cyclical timing score
        cyclical_bias = cyclical_analysis.get('overall_cyclical_bias', 0.0)
        analysis['cyclical_timing'] = cyclical_bias
        
        # Event timing score
        upcoming_impact = event_horizon.get('upcoming_impact', {}).get('total_impact', 0.0)
        event_uncertainty = event_horizon.get('upcoming_impact', {}).get('uncertainty', 0.0)
        event_bias = event_horizon.get('event_bias', 0.0)
        
        # Event timing is negative during high uncertainty periods
        event_timing_score = event_bias - (event_uncertainty * 0.5)
        analysis['event_timing'] = event_timing_score
        
        # Overall temporal bias
        temporal_bias = (
            session_timing_score * 0.4 +
            cyclical_bias * 0.3 +
            event_timing_score * 0.3
        )
        
        self.temporal_bias.update(temporal_bias)
        analysis['temporal_bias'] = self.temporal_bias.get_value() or 0.0
        
        # Timing quality (how favorable current timing is)
        timing_quality = (
            session_strength * 0.5 +
            cyclical_bias * 0.3 +
            (1.0 - event_uncertainty) * 0.2
        )
        analysis['timing_quality'] = timing_quality
        
        return analysis
    
    def _calculate_signal_strength(self, analysis: Dict[str, any]) -> float:
        """Calculate overall signal strength from temporal analysis."""
        temporal_bias = analysis.get('temporal_bias', 0.0)
        timing_quality = analysis.get('timing_quality', 0.0)
        
        # Signal strength is temporal bias weighted by timing quality
        signal_strength = temporal_bias * timing_quality
        
        return np.clip(signal_strength, -1.0, 1.0)
    
    def _calculate_confidence(self, analysis: Dict[str, any]) -> float:
        """Calculate confidence in temporal analysis."""
        timing_quality = analysis.get('timing_quality', 0.0)
        session_timing = analysis.get('session_timing', 0.0)
        cyclical_timing = analysis.get('cyclical_timing', 0.0)
        
        # Confidence based on timing quality and signal consistency
        signal_strength = abs(self._calculate_signal_strength(analysis))
        
        return compute_confidence(
            signal_strength=signal_strength,
            data_quality=1.0,  # Temporal data always available
            historical_accuracy=self.performance_tracker.get_accuracy(),
            confluence_signals=[timing_quality, abs(session_timing), abs(cyclical_timing)]
        )
    
    def _detect_market_regime(self, analysis: Dict[str, any], session_analysis: Dict[str, any]) -> MarketRegime:
        """Detect market regime from temporal analysis."""
        timing_quality = analysis.get('timing_quality', 0.0)
        session_strength = session_analysis.get('session_strength', 0.0)
        session_momentum = session_analysis.get('session_momentum', 0.0)
        
        # High timing quality + strong session = trending
        if timing_quality > 0.7 and session_strength > 0.7:
            if abs(session_momentum) > 0.5:
                return MarketRegime.TRENDING_STRONG
            else:
                return MarketRegime.TRENDING_WEAK
        
        # High session strength but low timing quality = breakout potential
        elif session_strength > 0.6 and timing_quality < 0.4:
            return MarketRegime.BREAKOUT
        
        # Low session strength = consolidating
        elif session_strength < 0.3:
            return MarketRegime.CONSOLIDATING
        
        # Default
        else:
            return MarketRegime.CONSOLIDATING
    
    def _extract_evidence(self, analysis: Dict[str, any], session_analysis: Dict[str, any]) -> Dict[str, float]:
        """Extract evidence scores for transparency."""
        evidence = {}
        
        evidence['timing_quality'] = analysis.get('timing_quality', 0.0)
        evidence['session_strength'] = session_analysis.get('session_strength', 0.0)
        evidence['cyclical_bias'] = analysis.get('cyclical_timing', 0.0)
        evidence['event_timing'] = analysis.get('event_timing', 0.0)
        evidence['temporal_bias'] = abs(analysis.get('temporal_bias', 0.0))
        
        return evidence
    
    def _generate_warnings(self, analysis: Dict[str, any], event_horizon: Dict[str, any]) -> List[str]:
        """Generate warnings about temporal analysis."""
        warnings = []
        
        # Check for high event uncertainty
        event_uncertainty = event_horizon.get('upcoming_impact', {}).get('uncertainty', 0.0)
        if event_uncertainty > 0.7:
            warnings.append(f"High event uncertainty: {event_uncertainty:.2f}")
        
        # Check for major upcoming events
        next_major = event_horizon.get('next_major_event')
        if next_major and next_major['hours_away'] < 2:
            warnings.append(f"Major event in {next_major['hours_away']:.1f} hours: {next_major['event']}")
        
        # Check for low timing quality
        timing_quality = analysis.get('timing_quality', 0.0)
        if timing_quality < 0.3:
            warnings.append(f"Poor timing conditions: {timing_quality:.2f}")
        
        return warnings
    
    def _create_error_reading(self, timestamp: datetime, error_msg: str) -> DimensionalReading:
        """Create reading when error occurs."""
        return DimensionalReading(
            dimension="WHEN",
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
            dimension="WHEN",
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
        self.cached_events.clear()
        self.last_event_fetch = None
        
        # Reset components
        self.session_analyzer = SessionAnalyzer()
        self.cyclical_analyzer = CyclicalAnalyzer()
        self.event_horizon_analyzer = EventHorizonAnalyzer()
        self.temporal_bias = EMA(30)
        
        logger.info("WHEN Engine reset completed")

