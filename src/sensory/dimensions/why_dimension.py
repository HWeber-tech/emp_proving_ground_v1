"""
WHY Dimension - Fundamental Market Forces

This dimension seeks to understand the fundamental forces driving market behavior:
- Economic momentum and cycles
- Central bank policy and monetary flows
- Risk sentiment and capital allocation
- Geopolitical events and structural changes

The WHY dimension provides context for all other dimensions, answering
"Why is the market behaving this way?"
"""

import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple
from collections import deque
from dataclasses import dataclass
import math

from ..core.base import (
    DimensionalSensor, DimensionalReading, MarketData, MarketRegime
)


@dataclass
class EconomicIndicator:
    """Economic indicator with trend analysis"""
    name: str
    value: float
    previous_value: float
    timestamp: datetime
    importance: float  # 0-1 scale
    
    @property
    def change_rate(self) -> float:
        if self.previous_value == 0:
            return 0.0
        return (self.value - self.previous_value) / abs(self.previous_value)
    
    @property
    def momentum(self) -> float:
        """Momentum score based on change rate and importance"""
        return self.change_rate * self.importance


@dataclass
class PolicySignal:
    """Central bank policy signal"""
    bank: str
    signal_type: str  # 'hawkish', 'dovish', 'neutral'
    strength: float  # 0-1 scale
    timestamp: datetime
    
    @property
    def directional_value(self) -> float:
        """Convert policy signal to directional value"""
        multiplier = {'hawkish': 1.0, 'dovish': -1.0, 'neutral': 0.0}.get(self.signal_type, 0.0)
        return multiplier * self.strength


class EconomicMomentumTracker:
    """Tracks economic momentum across multiple indicators"""
    
    def __init__(self):
        self.indicators: Dict[str, deque[EconomicIndicator]] = {}
        self.weights = {
            'gdp_growth': 0.25,
            'inflation': 0.20,
            'employment': 0.20,
            'manufacturing': 0.15,
            'consumer_confidence': 0.10,
            'trade_balance': 0.10
        }
        
    def update_indicator(self, indicator: EconomicIndicator) -> None:
        """Update economic indicator"""
        if indicator.name not in self.indicators:
            self.indicators[indicator.name] = deque(maxlen=50)
        
        self.indicators[indicator.name].append(indicator)
    
    def calculate_momentum(self) -> Tuple[float, float]:
        """Calculate overall economic momentum and confidence"""
        if not self.indicators:
            return 0.0, 0.0
        
        weighted_momentum = 0.0
        total_weight = 0.0
        confidence_scores = []
        
        for name, weight in self.weights.items():
            if name in self.indicators and self.indicators[name]:
                latest = self.indicators[name][-1]
                weighted_momentum += latest.momentum * weight
                total_weight += weight
                
                # Confidence based on data recency and consistency
                data_age_hours = (datetime.now() - latest.timestamp).total_seconds() / 3600
                recency_score = max(0.0, 1.0 - data_age_hours / 168)  # Decay over week
                
                # Consistency score from recent trend
                if len(self.indicators[name]) >= 3:
                    recent_changes = [ind.change_rate for ind in list(self.indicators[name])[-3:]]
                    consistency = 1.0 - np.std(recent_changes) if recent_changes else 0.5
                else:
                    consistency = 0.5
                
                confidence_scores.append(recency_score * consistency)
        
        if total_weight > 0:
            momentum = weighted_momentum / total_weight
            confidence = np.mean(confidence_scores) if confidence_scores else 0.0
        else:
            momentum, confidence = 0.0, 0.0
        
        return momentum, confidence


class MonetaryPolicyTracker:
    """Tracks central bank policy across major economies"""
    
    def __init__(self):
        self.policy_signals: Dict[str, deque[PolicySignal]] = {
            'fed': deque(maxlen=20),
            'ecb': deque(maxlen=20),
            'boe': deque(maxlen=20),
            'boj': deque(maxlen=20)
        }
        
        # Currency weights for EURUSD
        self.currency_weights = {
            'fed': 0.5,   # USD influence
            'ecb': 0.5,   # EUR influence
            'boe': 0.0,   # Minimal for EURUSD
            'boj': 0.0    # Minimal for EURUSD
        }
    
    def add_policy_signal(self, signal: PolicySignal) -> None:
        """Add policy signal"""
        if signal.bank in self.policy_signals:
            self.policy_signals[signal.bank].append(signal)
    
    def calculate_policy_divergence(self) -> Tuple[float, float]:
        """Calculate policy divergence and confidence"""
        
        # Get latest signals for each bank
        latest_signals = {}
        for bank, signals in self.policy_signals.items():
            if signals:
                latest_signals[bank] = signals[-1]
        
        if len(latest_signals) < 2:
            return 0.0, 0.0
        
        # Calculate weighted policy stance
        weighted_stance = 0.0
        total_weight = 0.0
        confidence_scores = []
        
        for bank, signal in latest_signals.items():
            weight = self.currency_weights.get(bank, 0.0)
            if weight > 0:
                weighted_stance += signal.directional_value * weight
                total_weight += weight
                
                # Confidence based on signal strength and recency
                signal_age_hours = (datetime.now() - signal.timestamp).total_seconds() / 3600
                recency_score = max(0.0, 1.0 - signal_age_hours / 72)  # 3-day decay
                confidence_scores.append(signal.strength * recency_score)
        
        if total_weight > 0:
            divergence = weighted_stance / total_weight
            confidence = np.mean(confidence_scores) if confidence_scores else 0.0
        else:
            divergence, confidence = 0.0, 0.0
        
        return divergence, confidence


class RiskSentimentAnalyzer:
    """Analyzes market risk sentiment and capital flows"""
    
    def __init__(self):
        self.sentiment_indicators = deque(maxlen=100)
        self.vix_history = deque(maxlen=50)
        self.yield_spreads = deque(maxlen=50)
        self.commodity_flows = deque(maxlen=50)
        
    def update_sentiment_data(self, vix: float, yield_spread: float, 
                            commodity_index: float, timestamp: datetime) -> None:
        """Update sentiment indicators"""
        self.vix_history.append((vix, timestamp))
        self.yield_spreads.append((yield_spread, timestamp))
        self.commodity_flows.append((commodity_index, timestamp))
    
    def calculate_risk_sentiment(self) -> Tuple[float, float]:
        """Calculate risk sentiment score"""
        if not (self.vix_history and self.yield_spreads and self.commodity_flows):
            return 0.0, 0.0
        
        # VIX analysis (fear gauge)
        current_vix = self.vix_history[-1][0]
        vix_values = [v[0] for v in self.vix_history]
        vix_percentile = np.percentile(vix_values, 50) if len(vix_values) > 10 else 20
        
        # Normalize VIX (lower VIX = risk-on, higher VIX = risk-off)
        vix_score = -(current_vix - vix_percentile) / vix_percentile
        vix_score = max(-1.0, min(1.0, vix_score))
        
        # Yield spread analysis (credit risk)
        current_spread = self.yield_spreads[-1][0]
        spread_values = [s[0] for s in self.yield_spreads]
        spread_percentile = np.percentile(spread_values, 50) if len(spread_values) > 10 else current_spread
        
        # Normalize spread (lower spread = risk-on)
        spread_score = -(current_spread - spread_percentile) / max(spread_percentile, 0.01)
        spread_score = max(-1.0, min(1.0, spread_score))
        
        # Commodity flows (risk appetite)
        current_commodity = self.commodity_flows[-1][0]
        commodity_values = [c[0] for c in self.commodity_flows]
        if len(commodity_values) > 1:
            commodity_change = (current_commodity - commodity_values[-2]) / abs(commodity_values[-2])
            commodity_score = max(-1.0, min(1.0, commodity_change * 10))  # Amplify signal
        else:
            commodity_score = 0.0
        
        # Weighted combination
        sentiment = (vix_score * 0.4 + spread_score * 0.4 + commodity_score * 0.2)
        
        # Confidence based on data consistency
        scores = [vix_score, spread_score, commodity_score]
        consistency = 1.0 - np.std(scores) if len(scores) > 1 else 0.5
        confidence = max(0.2, min(1.0, consistency))
        
        return sentiment, confidence


class GeopoliticalEventTracker:
    """Tracks geopolitical events and their market impact"""
    
    def __init__(self):
        self.events = deque(maxlen=50)
        self.impact_weights = {
            'election': 0.7,
            'trade_war': 0.9,
            'military_conflict': 1.0,
            'brexit': 0.8,
            'sanctions': 0.6,
            'summit': 0.4
        }
    
    def add_event(self, event_type: str, impact_score: float, timestamp: datetime) -> None:
        """Add geopolitical event"""
        weight = self.impact_weights.get(event_type, 0.5)
        weighted_impact = impact_score * weight
        self.events.append((event_type, weighted_impact, timestamp))
    
    def calculate_geopolitical_impact(self) -> Tuple[float, float]:
        """Calculate current geopolitical impact"""
        if not self.events:
            return 0.0, 0.0
        
        # Weight events by recency (exponential decay)
        current_time = datetime.now()
        weighted_impact = 0.0
        total_weight = 0.0
        
        for event_type, impact, timestamp in self.events:
            hours_ago = (current_time - timestamp).total_seconds() / 3600
            decay_factor = math.exp(-hours_ago / 168)  # Week half-life
            
            weight = decay_factor
            weighted_impact += impact * weight
            total_weight += weight
        
        if total_weight > 0:
            impact = weighted_impact / total_weight
            confidence = min(1.0, total_weight / 5.0)  # More events = higher confidence
        else:
            impact, confidence = 0.0, 0.0
        
        return impact, confidence


class WhyDimension(DimensionalSensor):
    """
    WHY Dimension - Understanding fundamental market forces
    
    Synthesizes multiple fundamental factors to understand why markets move:
    - Economic momentum from key indicators
    - Central bank policy divergence
    - Risk sentiment and capital flows
    - Geopolitical events and structural changes
    """
    
    def __init__(self):
        super().__init__("WHY")
        
        # Component analyzers
        self.economic_tracker = EconomicMomentumTracker()
        self.policy_tracker = MonetaryPolicyTracker()
        self.sentiment_analyzer = RiskSentimentAnalyzer()
        self.geopolitical_tracker = GeopoliticalEventTracker()
        
        # Synthesis weights
        self.component_weights = {
            'economic': 0.35,
            'policy': 0.30,
            'sentiment': 0.25,
            'geopolitical': 0.10
        }
        
        # Peer influence tracking
        self.peer_influences = {
            'how': 0.15,    # Institutional flows can indicate fundamental shifts
            'when': 0.10,   # Timing can amplify fundamental moves
            'what': 0.05,   # Technical levels can influence fundamental interpretation
            'anomaly': 0.20 # Anomalies can override fundamental analysis
        }
    
    def process(self, data: MarketData, peer_readings: Dict[str, DimensionalReading]) -> DimensionalReading:
        """Process market data to understand fundamental forces"""
        
        # Calculate component scores
        economic_score, economic_conf = self.economic_tracker.calculate_momentum()
        policy_score, policy_conf = self.policy_tracker.calculate_policy_divergence()
        sentiment_score, sentiment_conf = self.sentiment_analyzer.calculate_risk_sentiment()
        geopolitical_score, geopolitical_conf = self.geopolitical_tracker.calculate_geopolitical_impact()
        
        # Weighted synthesis
        components = {
            'economic': (economic_score, economic_conf),
            'policy': (policy_score, policy_conf),
            'sentiment': (sentiment_score, sentiment_conf),
            'geopolitical': (geopolitical_score, geopolitical_conf)
        }
        
        weighted_score = 0.0
        weighted_confidence = 0.0
        total_weight = 0.0
        
        for component, (score, conf) in components.items():
            weight = self.component_weights[component]
            weighted_score += score * weight * conf  # Weight by confidence
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
                
                # Peer influence on confidence (alignment increases confidence)
                alignment = 1.0 - abs(base_score - peer_reading.value) / 2.0
                peer_confidence_boost += alignment * influence_weight * peer_reading.confidence
        
        # Final score and confidence
        final_score = base_score + peer_adjustment * 0.3  # Moderate peer influence
        final_confidence = base_confidence + peer_confidence_boost * 0.2
        
        # Normalize
        final_score = max(-1.0, min(1.0, final_score))
        final_confidence = max(0.0, min(1.0, final_confidence))
        
        # Build context
        context = {
            'economic_momentum': economic_score,
            'policy_divergence': policy_score,
            'risk_sentiment': sentiment_score,
            'geopolitical_impact': geopolitical_score,
            'component_confidences': {k: v[1] for k, v in components.items()},
            'peer_adjustment': peer_adjustment,
            'regime': data.regime.name if data.regime else 'unknown'
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
    
    def update_economic_data(self, indicator_name: str, value: float, 
                           previous_value: float, importance: float = 1.0) -> None:
        """Update economic indicator data"""
        indicator = EconomicIndicator(
            name=indicator_name,
            value=value,
            previous_value=previous_value,
            timestamp=datetime.now(),
            importance=importance
        )
        self.economic_tracker.update_indicator(indicator)
    
    def update_policy_signal(self, bank: str, signal_type: str, strength: float) -> None:
        """Update central bank policy signal"""
        signal = PolicySignal(
            bank=bank,
            signal_type=signal_type,
            strength=strength,
            timestamp=datetime.now()
        )
        self.policy_tracker.add_policy_signal(signal)
    
    def update_sentiment_data(self, vix: float, yield_spread: float, commodity_index: float) -> None:
        """Update risk sentiment data"""
        self.sentiment_analyzer.update_sentiment_data(
            vix=vix,
            yield_spread=yield_spread,
            commodity_index=commodity_index,
            timestamp=datetime.now()
        )
    
    def add_geopolitical_event(self, event_type: str, impact_score: float) -> None:
        """Add geopolitical event"""
        self.geopolitical_tracker.add_event(event_type, impact_score, datetime.now())
    
    def get_component_breakdown(self) -> Dict[str, Dict[str, float]]:
        """Get detailed breakdown of component scores"""
        economic_score, economic_conf = self.economic_tracker.calculate_momentum()
        policy_score, policy_conf = self.policy_tracker.calculate_policy_divergence()
        sentiment_score, sentiment_conf = self.sentiment_analyzer.calculate_risk_sentiment()
        geopolitical_score, geopolitical_conf = self.geopolitical_tracker.calculate_geopolitical_impact()
        
        return {
            'economic': {'score': economic_score, 'confidence': economic_conf},
            'policy': {'score': policy_score, 'confidence': policy_conf},
            'sentiment': {'score': sentiment_score, 'confidence': sentiment_conf},
            'geopolitical': {'score': geopolitical_score, 'confidence': geopolitical_conf}
        }

