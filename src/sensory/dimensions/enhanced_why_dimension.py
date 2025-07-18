"""
Enhanced WHY Dimension - Fundamental Intelligence Engine

This module implements sophisticated fundamental analysis that goes beyond simple economic indicators
to understand the deep structural forces driving market behavior. It analyzes central bank policy,
economic cycles, geopolitical events, and currency dynamics to provide actionable fundamental insights.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple, NamedTuple
from collections import deque, defaultdict
from dataclasses import dataclass, field
from enum import Enum, auto
import math
import logging
import requests
import json
from scipy import stats

from ..core.base import DimensionalReading, MarketData, MarketRegime, DimensionalSensor, InstrumentMeta

logger = logging.getLogger(__name__)

class EconomicIndicatorType(Enum):
    """Types of economic indicators"""
    GROWTH = auto()           # GDP, Industrial Production
    INFLATION = auto()        # CPI, PPI, Core PCE
    EMPLOYMENT = auto()       # NFP, Unemployment Rate
    MONETARY = auto()         # Interest Rates, Money Supply
    TRADE = auto()           # Trade Balance, Current Account
    SENTIMENT = auto()        # Consumer Confidence, PMI
    FISCAL = auto()          # Government Spending, Debt

class CentralBankStance(Enum):
    """Central bank policy stance"""
    VERY_DOVISH = auto()     # Aggressive easing
    DOVISH = auto()          # Easing bias
    NEUTRAL = auto()         # No clear bias
    HAWKISH = auto()         # Tightening bias
    VERY_HAWKISH = auto()    # Aggressive tightening

class GeopoliticalRisk(Enum):
    """Geopolitical risk levels"""
    LOW = auto()             # Stable environment
    MODERATE = auto()        # Some tensions
    ELEVATED = auto()        # Significant concerns
    HIGH = auto()            # Crisis conditions
    EXTREME = auto()         # War/major crisis

class CurrencyStrength(Enum):
    """Currency strength classification"""
    VERY_WEAK = auto()
    WEAK = auto()
    NEUTRAL = auto()
    STRONG = auto()
    VERY_STRONG = auto()

@dataclass
class EconomicEvent:
    """Economic event data structure"""
    timestamp: datetime
    indicator: str
    country: str
    importance: str  # low, medium, high
    forecast: Optional[float]
    actual: Optional[float]
    previous: Optional[float]
    impact_score: float  # -1 to 1
    surprise_factor: float  # How much actual deviated from forecast

@dataclass
class CentralBankPolicy:
    """Central bank policy analysis"""
    bank: str
    current_rate: float
    rate_direction: str  # raising, lowering, holding
    stance: CentralBankStance
    next_meeting: datetime
    policy_divergence: float  # vs other major banks
    forward_guidance: str
    credibility_score: float

@dataclass
class GeopoliticalEvent:
    """Geopolitical event analysis"""
    event_type: str
    region: str
    risk_level: GeopoliticalRisk
    market_impact: float  # Expected impact on markets
    duration_estimate: timedelta
    affected_currencies: List[str]
    safe_haven_flow: float  # Expected safe haven demand

@dataclass
class CurrencyAnalysis:
    """Currency strength analysis"""
    currency: str
    strength_score: float  # -1 to 1
    strength_classification: CurrencyStrength
    drivers: List[str]  # Key fundamental drivers
    relative_yield: float  # Yield advantage vs others
    economic_momentum: float  # Economic growth momentum
    policy_support: float  # Central bank policy support

class EconomicDataProvider:
    """
    Economic data provider with real API integration
    """
    
    def __init__(self):
        # Economic calendar cache
        self.economic_calendar = deque(maxlen=1000)
        self.last_calendar_update = None
        
        # Central bank data
        self.central_bank_data = {}
        self.last_cb_update = None
        
        # Currency data
        self.currency_data = {}
        
        # Geopolitical events
        self.geopolitical_events = deque(maxlen=100)
        
    def get_economic_calendar(self, days_ahead: int = 7) -> List[EconomicEvent]:
        """Get economic calendar events"""
        
        # Check if we need to update calendar
        if (self.last_calendar_update is None or 
            datetime.now() - self.last_calendar_update > timedelta(hours=6)):
            self._update_economic_calendar()
        
        # Filter events for the requested period
        start_time = datetime.now()
        end_time = start_time + timedelta(days=days_ahead)
        
        relevant_events = [
            event for event in self.economic_calendar
            if start_time <= event.timestamp <= end_time
        ]
        
        return relevant_events
    
    def _update_economic_calendar(self):
        """Update economic calendar from data sources"""
        
        try:
            # Simulate economic calendar data (in production, use real API)
            self._generate_simulated_calendar()
            self.last_calendar_update = datetime.now()
            
        except Exception as e:
            logger.warning(f"Failed to update economic calendar: {e}")
            # Use cached data if available
    
    def _generate_simulated_calendar(self):
        """Generate simulated economic calendar for demonstration"""
        
        base_time = datetime.now()
        
        # Major economic indicators
        indicators = [
            ('Non-Farm Payrolls', 'USD', 'high', 0.3),
            ('CPI YoY', 'USD', 'high', 0.25),
            ('GDP QoQ', 'USD', 'high', 0.2),
            ('Federal Funds Rate', 'USD', 'high', 0.4),
            ('ECB Interest Rate', 'EUR', 'high', 0.35),
            ('BoE Interest Rate', 'GBP', 'high', 0.3),
            ('BoJ Interest Rate', 'JPY', 'high', 0.25),
            ('Unemployment Rate', 'USD', 'medium', 0.15),
            ('Retail Sales', 'USD', 'medium', 0.1),
            ('PMI Manufacturing', 'USD', 'medium', 0.12),
            ('Consumer Confidence', 'USD', 'low', 0.08)
        ]
        
        # Generate events for next 30 days
        for i in range(30):
            event_time = base_time + timedelta(days=i)
            
            # Random chance of events each day
            if np.random.random() < 0.3:  # 30% chance per day
                indicator, currency, importance, base_impact = np.random.choice(
                    indicators, p=[0.15, 0.15, 0.1, 0.1, 0.1, 0.08, 0.08, 0.08, 0.06, 0.06, 0.04]
                )
                
                # Generate forecast and actual values
                forecast = np.random.normal(0, 1)
                actual = forecast + np.random.normal(0, 0.5)  # Some surprise
                previous = forecast + np.random.normal(0, 0.3)
                
                # Calculate impact and surprise
                surprise_factor = abs(actual - forecast) if forecast != 0 else 0
                impact_direction = 1 if actual > forecast else -1
                impact_score = impact_direction * base_impact * (1 + surprise_factor)
                
                event = EconomicEvent(
                    timestamp=event_time,
                    indicator=indicator,
                    country=currency,
                    importance=importance,
                    forecast=forecast,
                    actual=actual if np.random.random() < 0.7 else None,  # 70% have actual
                    previous=previous,
                    impact_score=impact_score,
                    surprise_factor=surprise_factor
                )
                
                self.economic_calendar.append(event)
    
    def get_central_bank_policies(self) -> Dict[str, CentralBankPolicy]:
        """Get central bank policy analysis"""
        
        if (self.last_cb_update is None or 
            datetime.now() - self.last_cb_update > timedelta(hours=12)):
            self._update_central_bank_data()
        
        return self.central_bank_data.copy()
    
    def _update_central_bank_data(self):
        """Update central bank policy data"""
        
        try:
            # Simulate central bank data (in production, use real sources)
            self._generate_simulated_cb_data()
            self.last_cb_update = datetime.now()
            
        except Exception as e:
            logger.warning(f"Failed to update central bank data: {e}")
    
    def _generate_simulated_cb_data(self):
        """Generate simulated central bank data"""
        
        banks = {
            'FED': {
                'rate': 5.25,
                'direction': 'holding',
                'stance': CentralBankStance.HAWKISH,
                'next_meeting': datetime.now() + timedelta(days=21),
                'guidance': 'Data-dependent approach to future rate decisions'
            },
            'ECB': {
                'rate': 4.50,
                'direction': 'holding',
                'stance': CentralBankStance.NEUTRAL,
                'next_meeting': datetime.now() + timedelta(days=14),
                'guidance': 'Monitoring inflation developments closely'
            },
            'BOE': {
                'rate': 5.00,
                'direction': 'holding',
                'stance': CentralBankStance.HAWKISH,
                'next_meeting': datetime.now() + timedelta(days=28),
                'guidance': 'Prepared to act if inflation persists'
            },
            'BOJ': {
                'rate': -0.10,
                'direction': 'holding',
                'stance': CentralBankStance.DOVISH,
                'next_meeting': datetime.now() + timedelta(days=35),
                'guidance': 'Maintaining ultra-loose monetary policy'
            }
        }
        
        for bank, data in banks.items():
            # Calculate policy divergence
            other_rates = [info['rate'] for name, info in banks.items() if name != bank]
            avg_other_rate = np.mean(other_rates)
            divergence = data['rate'] - avg_other_rate
            
            # Calculate credibility score (simplified)
            credibility = 0.8 + np.random.normal(0, 0.1)
            credibility = max(0.3, min(credibility, 1.0))
            
            policy = CentralBankPolicy(
                bank=bank,
                current_rate=data['rate'],
                rate_direction=data['direction'],
                stance=data['stance'],
                next_meeting=data['next_meeting'],
                policy_divergence=divergence,
                forward_guidance=data['guidance'],
                credibility_score=credibility
            )
            
            self.central_bank_data[bank] = policy
    
    def get_geopolitical_events(self) -> List[GeopoliticalEvent]:
        """Get current geopolitical events"""
        
        # Return recent geopolitical events
        cutoff_time = datetime.now() - timedelta(days=30)
        return [event for event in self.geopolitical_events if event.timestamp > cutoff_time]
    
    def analyze_currency_strength(self, currency: str) -> CurrencyAnalysis:
        """Analyze currency strength fundamentals"""
        
        # Simulate currency analysis (in production, use real data)
        return self._generate_currency_analysis(currency)
    
    def _generate_currency_analysis(self, currency: str) -> CurrencyAnalysis:
        """Generate simulated currency analysis"""
        
        # Currency characteristics
        currency_profiles = {
            'USD': {
                'base_strength': 0.2,
                'yield_advantage': 0.15,
                'economic_momentum': 0.1,
                'policy_support': 0.25,
                'drivers': ['Federal Reserve policy', 'Economic growth', 'Safe haven demand']
            },
            'EUR': {
                'base_strength': -0.1,
                'yield_advantage': 0.05,
                'economic_momentum': -0.05,
                'policy_support': 0.0,
                'drivers': ['ECB policy', 'Economic recovery', 'Political stability']
            },
            'GBP': {
                'base_strength': 0.05,
                'yield_advantage': 0.1,
                'economic_momentum': 0.0,
                'policy_support': 0.15,
                'drivers': ['BoE policy', 'Brexit effects', 'Economic data']
            },
            'JPY': {
                'base_strength': -0.3,
                'yield_advantage': -0.4,
                'economic_momentum': -0.1,
                'policy_support': -0.2,
                'drivers': ['BoJ ultra-loose policy', 'Carry trade flows', 'Risk sentiment']
            }
        }
        
        profile = currency_profiles.get(currency, {
            'base_strength': 0.0,
            'yield_advantage': 0.0,
            'economic_momentum': 0.0,
            'policy_support': 0.0,
            'drivers': ['Economic fundamentals']
        })
        
        # Add some randomness
        strength_score = (
            profile['base_strength'] + 
            np.random.normal(0, 0.1)
        )
        
        # Classify strength
        if strength_score > 0.3:
            classification = CurrencyStrength.VERY_STRONG
        elif strength_score > 0.1:
            classification = CurrencyStrength.STRONG
        elif strength_score > -0.1:
            classification = CurrencyStrength.NEUTRAL
        elif strength_score > -0.3:
            classification = CurrencyStrength.WEAK
        else:
            classification = CurrencyStrength.VERY_WEAK
        
        return CurrencyAnalysis(
            currency=currency,
            strength_score=strength_score,
            strength_classification=classification,
            drivers=profile['drivers'],
            relative_yield=profile['yield_advantage'],
            economic_momentum=profile['economic_momentum'],
            policy_support=profile['policy_support']
        )

class FundamentalAnalyzer:
    """
    Advanced fundamental analysis engine
    """
    
    def __init__(self):
        self.data_provider = EconomicDataProvider()
        
        # Analysis state
        self.economic_momentum = 0.0
        self.policy_momentum = 0.0
        self.risk_sentiment = 0.0
        self.currency_flows = {}
        
        # Historical tracking
        self.momentum_history = deque(maxlen=50)
        self.sentiment_history = deque(maxlen=50)
        
    def analyze_economic_momentum(self, currency_pair: str) -> float:
        """Analyze economic momentum for currency pair"""
        
        # Extract base and quote currencies
        base_currency = currency_pair[:3]
        quote_currency = currency_pair[3:]
        
        # Get currency analyses
        base_analysis = self.data_provider.analyze_currency_strength(base_currency)
        quote_analysis = self.data_provider.analyze_currency_strength(quote_currency)
        
        # Calculate relative momentum
        momentum_differential = (
            base_analysis.economic_momentum - quote_analysis.economic_momentum
        )
        
        # Factor in policy support
        policy_differential = (
            base_analysis.policy_support - quote_analysis.policy_support
        )
        
        # Combine factors
        total_momentum = (momentum_differential * 0.6 + policy_differential * 0.4)
        
        # Normalize to -1 to 1 range
        self.economic_momentum = np.tanh(total_momentum)
        
        return self.economic_momentum
    
    def analyze_central_bank_divergence(self, currency_pair: str) -> float:
        """Analyze central bank policy divergence"""
        
        base_currency = currency_pair[:3]
        quote_currency = currency_pair[3:]
        
        # Map currencies to central banks
        currency_to_bank = {
            'USD': 'FED',
            'EUR': 'ECB', 
            'GBP': 'BOE',
            'JPY': 'BOJ'
        }
        
        base_bank = currency_to_bank.get(base_currency)
        quote_bank = currency_to_bank.get(quote_currency)
        
        if not base_bank or not quote_bank:
            return 0.0
        
        # Get central bank policies
        cb_policies = self.data_provider.get_central_bank_policies()
        
        base_policy = cb_policies.get(base_bank)
        quote_policy = cb_policies.get(quote_bank)
        
        if not base_policy or not quote_policy:
            return 0.0
        
        # Calculate policy divergence
        rate_differential = base_policy.current_rate - quote_policy.current_rate
        
        # Factor in policy stance
        stance_values = {
            CentralBankStance.VERY_DOVISH: -2,
            CentralBankStance.DOVISH: -1,
            CentralBankStance.NEUTRAL: 0,
            CentralBankStance.HAWKISH: 1,
            CentralBankStance.VERY_HAWKISH: 2
        }
        
        base_stance = stance_values.get(base_policy.stance, 0)
        quote_stance = stance_values.get(quote_policy.stance, 0)
        stance_differential = base_stance - quote_stance
        
        # Combine rate and stance differentials
        policy_divergence = (rate_differential * 0.7 + stance_differential * 0.3)
        
        # Normalize
        self.policy_momentum = np.tanh(policy_divergence / 3.0)
        
        return self.policy_momentum
    
    def analyze_economic_calendar_impact(self, currency_pair: str, 
                                       hours_ahead: int = 24) -> float:
        """Analyze upcoming economic calendar impact"""
        
        base_currency = currency_pair[:3]
        quote_currency = currency_pair[3:]
        
        # Get upcoming events
        events = self.data_provider.get_economic_calendar(days_ahead=hours_ahead//24 + 1)
        
        # Filter events for our currencies
        relevant_events = [
            event for event in events
            if event.country in [base_currency, quote_currency] and
            event.timestamp <= datetime.now() + timedelta(hours=hours_ahead)
        ]
        
        # Calculate net impact
        base_impact = 0.0
        quote_impact = 0.0
        
        for event in relevant_events:
            # Weight by importance and time proximity
            importance_weight = {
                'high': 1.0,
                'medium': 0.6,
                'low': 0.3
            }.get(event.importance, 0.5)
            
            # Time decay (closer events have more impact)
            time_to_event = (event.timestamp - datetime.now()).total_seconds() / 3600
            time_weight = math.exp(-time_to_event / 12)  # 12-hour half-life
            
            weighted_impact = event.impact_score * importance_weight * time_weight
            
            if event.country == base_currency:
                base_impact += weighted_impact
            elif event.country == quote_currency:
                quote_impact += weighted_impact
        
        # Calculate net impact for currency pair
        net_impact = base_impact - quote_impact
        
        # Normalize
        return np.tanh(net_impact)
    
    def analyze_risk_sentiment(self) -> float:
        """Analyze global risk sentiment"""
        
        # Factors affecting risk sentiment
        sentiment_factors = []
        
        # Geopolitical events
        geo_events = self.data_provider.get_geopolitical_events()
        
        if geo_events:
            # Calculate average geopolitical risk
            risk_values = {
                GeopoliticalRisk.LOW: 0.1,
                GeopoliticalRisk.MODERATE: 0.3,
                GeopoliticalRisk.ELEVATED: 0.6,
                GeopoliticalRisk.HIGH: 0.8,
                GeopoliticalRisk.EXTREME: 1.0
            }
            
            avg_risk = np.mean([
                risk_values.get(event.risk_level, 0.5) 
                for event in geo_events
            ])
            
            # High geopolitical risk = risk-off sentiment
            sentiment_factors.append(-avg_risk)
        
        # Central bank policy uncertainty
        cb_policies = self.data_provider.get_central_bank_policies()
        
        if cb_policies:
            # Calculate policy uncertainty
            credibility_scores = [policy.credibility_score for policy in cb_policies.values()]
            avg_credibility = np.mean(credibility_scores)
            
            # High credibility = risk-on sentiment
            sentiment_factors.append((avg_credibility - 0.5) * 0.5)
        
        # Economic momentum (simplified)
        if self.momentum_history:
            recent_momentum = np.mean(list(self.momentum_history)[-10:])
            sentiment_factors.append(recent_momentum * 0.3)
        
        # Combine factors
        if sentiment_factors:
            self.risk_sentiment = np.mean(sentiment_factors)
        else:
            self.risk_sentiment = 0.0
        
        # Store in history
        self.sentiment_history.append(self.risk_sentiment)
        
        return self.risk_sentiment
    
    def analyze_yield_differentials(self, currency_pair: str) -> float:
        """Analyze yield differentials and carry trade flows"""
        
        base_currency = currency_pair[:3]
        quote_currency = currency_pair[3:]
        
        # Get currency analyses
        base_analysis = self.data_provider.analyze_currency_strength(base_currency)
        quote_analysis = self.data_provider.analyze_currency_strength(quote_currency)
        
        # Calculate yield differential
        yield_differential = base_analysis.relative_yield - quote_analysis.relative_yield
        
        # Factor in risk sentiment (carry trades work better in risk-on environment)
        risk_adjustment = self.risk_sentiment * 0.5
        
        # Adjusted yield differential
        adjusted_differential = yield_differential * (1.0 + risk_adjustment)
        
        return np.tanh(adjusted_differential)

class EnhancedFundamentalIntelligenceEngine(DimensionalSensor):
    """
    Enhanced fundamental intelligence engine with sophisticated analysis
    """
    
    def __init__(self, instrument_meta: InstrumentMeta):
        super().__init__(instrument_meta)
        self.fundamental_analyzer = FundamentalAnalyzer()
        
        # Analysis weights (adaptive)
        self.weights = {
            'economic_momentum': 0.25,
            'policy_divergence': 0.30,
            'calendar_impact': 0.15,
            'risk_sentiment': 0.20,
            'yield_differential': 0.10
        }
        
        # Performance tracking
        self.analysis_history = deque(maxlen=100)
        self.confidence_history = deque(maxlen=50)
        
        # Adaptive parameters
        self.learning_rate = 0.05
        self.confidence_threshold = 0.6
        
    async def analyze_fundamental_intelligence(self, market_data: MarketData) -> DimensionalReading:
        """Perform comprehensive fundamental analysis"""
        
        try:
            # Extract currency pair from symbol
            currency_pair = market_data.symbol
            
            # Perform individual analyses
            economic_momentum = self.fundamental_analyzer.analyze_economic_momentum(currency_pair)
            policy_divergence = self.fundamental_analyzer.analyze_central_bank_divergence(currency_pair)
            calendar_impact = self.fundamental_analyzer.analyze_economic_calendar_impact(currency_pair)
            risk_sentiment = self.fundamental_analyzer.analyze_risk_sentiment()
            yield_differential = self.fundamental_analyzer.analyze_yield_differentials(currency_pair)
            
            # Calculate weighted fundamental score
            fundamental_score = (
                economic_momentum * self.weights['economic_momentum'] +
                policy_divergence * self.weights['policy_divergence'] +
                calendar_impact * self.weights['calendar_impact'] +
                risk_sentiment * self.weights['risk_sentiment'] +
                yield_differential * self.weights['yield_differential']
            )
            
            # Calculate confidence based on factor agreement
            factor_values = [economic_momentum, policy_divergence, calendar_impact, 
                           risk_sentiment, yield_differential]
            
            confidence = self._calculate_confidence(factor_values, fundamental_score)
            
            # Update adaptive weights based on recent performance
            self._update_adaptive_weights()
            
            # Generate context
            context = self._generate_context(
                economic_momentum, policy_divergence, calendar_impact,
                risk_sentiment, yield_differential, currency_pair
            )
            
            # Store analysis
            self.analysis_history.append({
                'timestamp': market_data.timestamp,
                'score': fundamental_score,
                'confidence': confidence,
                'factors': {
                    'economic_momentum': economic_momentum,
                    'policy_divergence': policy_divergence,
                    'calendar_impact': calendar_impact,
                    'risk_sentiment': risk_sentiment,
                    'yield_differential': yield_differential
                }
            })
            
            # Create reading
            reading = DimensionalReading(
                dimension='WHY',
                signal_strength=fundamental_score,
                confidence=confidence,
                regime=self._determine_regime(fundamental_score),
                context=context,
                timestamp=market_data.timestamp
            )
            
            # Store last reading and mark as initialized
            self.last_reading = reading
            self.is_initialized = True
            
            return reading
            
        except Exception as e:
            logger.error(f"Fundamental analysis failed: {e}")
            
            # Return neutral reading on error
            return DimensionalReading(
                dimension='WHY',
                value=0.0,
                confidence=0.3,
                context={'error': str(e), 'status': 'degraded'},
                timestamp=market_data.timestamp
            )
    
    async def update(self, market_data: MarketData) -> DimensionalReading:
        """Process new market data and return dimensional reading."""
        return await self.analyze_fundamental_intelligence(market_data)
    
    def snapshot(self) -> DimensionalReading:
        """Return current dimensional state without processing new data."""
        if self.last_reading:
            return self.last_reading
        else:
            # Return default reading
            return DimensionalReading(
                dimension='WHY',
                signal_strength=0.0,
                confidence=0.0,
                regime=MarketRegime.UNKNOWN,
                context={'status': 'not_initialized'}
            )
    
    def reset(self) -> None:
        """Reset sensor state for new trading session or instrument."""
        self.analysis_history.clear()
        self.confidence_history.clear()
        self.last_reading = None
        self.is_initialized = False
    
    def _calculate_confidence(self, factor_values: List[float], final_score: float) -> float:
        """Calculate confidence based on factor agreement and strength"""
        
        # Factor agreement (how much factors agree on direction)
        positive_factors = sum(1 for f in factor_values if f > 0.1)
        negative_factors = sum(1 for f in factor_values if f < -0.1)
        neutral_factors = len(factor_values) - positive_factors - negative_factors
        
        # Agreement score
        if positive_factors > negative_factors:
            agreement = positive_factors / len(factor_values)
        elif negative_factors > positive_factors:
            agreement = negative_factors / len(factor_values)
        else:
            agreement = 0.5  # Mixed signals
        
        # Signal strength
        avg_strength = np.mean([abs(f) for f in factor_values])
        
        # Data quality (how many factors we have data for)
        data_quality = len([f for f in factor_values if f != 0.0]) / len(factor_values)
        
        # Combine factors
        base_confidence = (agreement * 0.4 + avg_strength * 0.4 + data_quality * 0.2)
        
        # Adjust for historical performance
        if self.confidence_history:
            historical_accuracy = np.mean(list(self.confidence_history))
            base_confidence = base_confidence * 0.8 + historical_accuracy * 0.2
        
        # Ensure bounds
        confidence = max(0.1, min(base_confidence, 0.95))
        
        self.confidence_history.append(confidence)
        
        return confidence
    
    def _update_adaptive_weights(self):
        """Update adaptive weights based on recent performance"""
        
        if len(self.analysis_history) < 10:
            return
        
        # Analyze recent performance of each factor
        recent_analyses = list(self.analysis_history)[-10:]
        
        # Calculate factor performance (simplified)
        for factor_name in self.weights.keys():
            factor_values = [analysis['factors'].get(factor_name, 0) for analysis in recent_analyses]
            factor_accuracy = np.mean([abs(f) for f in factor_values])  # Simplified accuracy measure
            
            # Adjust weight based on performance
            current_weight = self.weights[factor_name]
            adjustment = (factor_accuracy - 0.5) * self.learning_rate
            new_weight = current_weight + adjustment
            
            # Keep weights within reasonable bounds
            self.weights[factor_name] = max(0.05, min(new_weight, 0.5))
        
        # Normalize weights to sum to 1.0
        total_weight = sum(self.weights.values())
        if total_weight > 0:
            for factor_name in self.weights:
                self.weights[factor_name] /= total_weight
    
    def _generate_context(self, economic_momentum: float, policy_divergence: float,
                         calendar_impact: float, risk_sentiment: float,
                         yield_differential: float, currency_pair: str) -> Dict[str, Any]:
        """Generate contextual information for the analysis"""
        
        context = {
            'currency_pair': currency_pair,
            'analysis_components': {
                'economic_momentum': economic_momentum,
                'policy_divergence': policy_divergence,
                'calendar_impact': calendar_impact,
                'risk_sentiment': risk_sentiment,
                'yield_differential': yield_differential
            },
            'adaptive_weights': self.weights.copy(),
            'dominant_factor': max(self.weights.items(), key=lambda x: x[1])[0]
        }
        
        # Add qualitative assessments
        if abs(economic_momentum) > 0.3:
            context['economic_assessment'] = 'strong_momentum' if economic_momentum > 0 else 'weak_momentum'
        
        if abs(policy_divergence) > 0.3:
            context['policy_assessment'] = 'hawkish_divergence' if policy_divergence > 0 else 'dovish_divergence'
        
        if abs(risk_sentiment) > 0.3:
            context['risk_assessment'] = 'risk_on' if risk_sentiment > 0 else 'risk_off'
        
        # Add upcoming events impact
        if abs(calendar_impact) > 0.2:
            context['calendar_assessment'] = 'positive_events' if calendar_impact > 0 else 'negative_events'
        
        # Add carry trade assessment
        if abs(yield_differential) > 0.2:
            context['carry_assessment'] = 'favorable_carry' if yield_differential > 0 else 'unfavorable_carry'
        
        return context
    
    def get_diagnostic_information(self) -> Dict[str, Any]:
        """Get diagnostic information about the fundamental analysis"""
        
        diagnostics = {
            'current_weights': self.weights.copy(),
            'analysis_count': len(self.analysis_history),
            'avg_confidence': np.mean(list(self.confidence_history)) if self.confidence_history else 0.5
        }
        
        if self.analysis_history:
            recent_scores = [analysis['score'] for analysis in list(self.analysis_history)[-10:]]
            diagnostics['recent_avg_score'] = np.mean(recent_scores)
            diagnostics['score_volatility'] = np.std(recent_scores)
        
        # Add data provider status
        diagnostics['data_provider_status'] = {
            'economic_calendar_events': len(self.fundamental_analyzer.data_provider.economic_calendar),
            'central_bank_policies': len(self.fundamental_analyzer.data_provider.central_bank_data),
            'last_calendar_update': self.fundamental_analyzer.data_provider.last_calendar_update,
            'last_cb_update': self.fundamental_analyzer.data_provider.last_cb_update
        }
        
        return diagnostics

    def _determine_regime(self, fundamental_score: float) -> MarketRegime:
        """Determine market regime from fundamental score."""
        if fundamental_score > 0.5:
            return MarketRegime.TRENDING_BULL
        elif fundamental_score > 0.2:
            return MarketRegime.TRENDING_WEAK
        elif fundamental_score < -0.5:
            return MarketRegime.TRENDING_BEAR
        elif fundamental_score < -0.2:
            return MarketRegime.TRENDING_WEAK
        else:
            return MarketRegime.CONSOLIDATING

# Example usage
async def main():
    """Example usage of the enhanced fundamental intelligence engine"""
    
    # Initialize engine
    engine = EnhancedFundamentalIntelligenceEngine()
    
    # Simulate market data
    for i in range(20):
        
        current_time = datetime.now() + timedelta(minutes=i * 30)
        
        market_data = MarketData(
            timestamp=current_time,
            symbol='EURUSD',
            bid=1.0950 + np.random.normal(0, 0.001),
            ask=1.0952 + np.random.normal(0, 0.001),
            volume=1000 + np.random.randint(-200, 200),
            spread=0.0002,
            volatility=0.005 + np.random.exponential(0.002)
        )
        
        # Perform fundamental analysis
        reading = await engine.analyze_fundamental_intelligence(market_data)
        
        if i % 5 == 0:  # Print every 5th analysis
            print(f"Fundamental Intelligence Reading (Period {i}):")
            print(f"  Value: {reading.value:.3f}")
            print(f"  Confidence: {reading.confidence:.3f}")
            print(f"  Dominant Factor: {reading.context.get('dominant_factor', 'unknown')}")
            
            # Show component breakdown
            components = reading.context.get('analysis_components', {})
            print(f"  Components:")
            for component, value in components.items():
                print(f"    {component}: {value:.3f}")
            
            print()

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())

