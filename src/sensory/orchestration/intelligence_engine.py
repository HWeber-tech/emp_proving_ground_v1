"""
Intelligence Engine - Orchestrated Dimensional Synthesis

This is the central orchestration engine that synthesizes all five dimensions
(WHY, HOW, WHAT, WHEN, ANOMALY) into unified market intelligence.

The engine doesn't just aggregate readings - it understands how dimensions
interact, influence each other, and collectively form market understanding.
"""

import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple, NamedTuple
from collections import deque, defaultdict
from dataclasses import dataclass
from enum import Enum, auto
import threading
import math
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

from ..core.base import (
    DimensionalSensor, DimensionalReading, MarketData, MarketRegime,
    MarketNarrative, MemoryBank, DimensionalCorrelationMatrix
)
from ..dimensions.why_dimension import WhyDimension
from ..dimensions.how_dimension import HowDimension
from ..dimensions.what_dimension import WhatDimension
from ..dimensions.when_dimension import WhenDimension
from ..dimensions.anomaly_dimension import AnomalyDimension


class IntelligenceLevel(Enum):
    CONFUSED = auto()      # Conflicting signals, low confidence
    UNCERTAIN = auto()     # Some clarity but missing pieces
    AWARE = auto()         # Good understanding of current state
    INSIGHTFUL = auto()    # Deep understanding with predictive power
    PRESCIENT = auto()     # Exceptional understanding across all dimensions


class MarketUnderstanding(NamedTuple):
    """Unified market understanding from all dimensions"""
    narrative: str
    confidence: float
    intelligence_level: IntelligenceLevel
    primary_drivers: List[str]
    risk_factors: List[str]
    dimensional_consensus: float  # How much dimensions agree
    predictive_power: float       # Ability to predict near-term moves
    regime: MarketRegime
    timestamp: datetime


@dataclass
class DimensionalConsensus:
    """Analysis of how well dimensions agree with each other"""
    agreement_score: float        # 0-1, how much dimensions agree
    conflicting_dimensions: List[str]
    supporting_dimensions: List[str]
    confidence_weighted_score: float
    narrative_coherence: float    # How well the story hangs together
    
    @property
    def is_coherent(self) -> bool:
        return self.agreement_score > 0.6 and self.narrative_coherence > 0.5


class AdaptiveWeightingSystem:
    """Dynamically adjusts dimensional weights based on performance and context"""
    
    def __init__(self):
        # Base weights for each dimension
        self.base_weights = {
            'why': 0.25,
            'how': 0.25,
            'what': 0.20,
            'when': 0.15,
            'anomaly': 0.15
        }
        
        # Performance tracking
        self.performance_history: Dict[str, deque] = {
            dim: deque(maxlen=100) for dim in self.base_weights
        }
        
        # Context-based adjustments
        self.regime_adjustments = {
            MarketRegime.TRENDING_BULL: {'why': 1.2, 'what': 1.1, 'when': 0.9},
            MarketRegime.TRENDING_BEAR: {'why': 1.2, 'what': 1.1, 'when': 0.9},
            MarketRegime.RANGING: {'how': 1.3, 'what': 1.2, 'when': 1.1},
            MarketRegime.VOLATILE: {'anomaly': 1.4, 'how': 1.2, 'what': 0.8},
            MarketRegime.UNKNOWN: {}  # No adjustments
        }
        
        # Volatility-based adjustments
        self.volatility_adjustments = {
            'low': {'why': 1.1, 'when': 1.2},      # Fundamentals matter more in calm markets
            'medium': {},                           # No adjustments
            'high': {'anomaly': 1.3, 'how': 1.2}   # Focus on manipulation and institutional activity
        }
        
        self._lock = threading.Lock()
    
    def update_performance(self, dimension: str, accuracy: float) -> None:
        """Update performance tracking for a dimension"""
        with self._lock:
            self.performance_history[dimension].append(accuracy)
    
    def calculate_weights(self, regime: MarketRegime, volatility_level: str, 
                         dimensional_readings: Dict[str, DimensionalReading]) -> Dict[str, float]:
        """Calculate adaptive weights based on context and performance"""
        
        weights = self.base_weights.copy()
        
        # Apply regime-based adjustments
        if regime in self.regime_adjustments:
            for dim, adjustment in self.regime_adjustments[regime].items():
                if dim in weights:
                    weights[dim] *= adjustment
        
        # Apply volatility-based adjustments
        if volatility_level in self.volatility_adjustments:
            for dim, adjustment in self.volatility_adjustments[volatility_level].items():
                if dim in weights:
                    weights[dim] *= adjustment
        
        # Apply performance-based adjustments
        with self._lock:
            for dim in weights:
                if len(self.performance_history[dim]) >= 10:
                    recent_performance = list(self.performance_history[dim])[-10:]
                    avg_performance = np.mean(recent_performance)
                    
                    # Boost weight for well-performing dimensions
                    if avg_performance > 0.7:
                        weights[dim] *= 1.1
                    elif avg_performance < 0.4:
                        weights[dim] *= 0.9
        
        # Apply confidence-based adjustments
        for dim, reading in dimensional_readings.items():
            if dim in weights:
                # Higher confidence dimensions get slightly more weight
                confidence_boost = 1.0 + (reading.confidence - 0.5) * 0.2
                weights[dim] *= confidence_boost
        
        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {dim: weight / total_weight for dim, weight in weights.items()}
        
        return weights


class NarrativeConstructor:
    """Constructs coherent market narratives from dimensional readings"""
    
    def __init__(self):
        self.narrative_templates = {
            'trending_bull': [
                "Strong {why_driver} fundamentals driving sustained upward momentum",
                "Institutional {how_pattern} supporting {what_pattern} breakout",
                "Bullish sentiment reinforced by {when_factor} timing dynamics"
            ],
            'trending_bear': [
                "Deteriorating {why_driver} fundamentals pressuring prices lower",
                "Institutional {how_pattern} confirming {what_pattern} breakdown",
                "Bearish sentiment amplified by {when_factor} timing factors"
            ],
            'ranging': [
                "Balanced {why_driver} fundamentals keeping price in range",
                "Institutional {how_pattern} maintaining {what_pattern} equilibrium",
                "Sideways action supported by {when_factor} timing neutrality"
            ],
            'volatile': [
                "Conflicting {why_driver} signals creating uncertainty",
                "Erratic {how_pattern} reflecting {what_pattern} instability",
                "High volatility driven by {when_factor} and {anomaly_factor}"
            ]
        }
    
    def construct_narrative(self, dimensional_readings: Dict[str, DimensionalReading],
                          regime: MarketRegime, consensus: DimensionalConsensus) -> str:
        """Construct coherent narrative from dimensional readings"""
        
        # Extract key drivers from each dimension
        drivers = {}
        
        for dim_name, reading in dimensional_readings.items():
            context = reading.context
            
            if dim_name == 'why':
                drivers['why_driver'] = self._extract_why_driver(context)
            elif dim_name == 'how':
                drivers['how_pattern'] = self._extract_how_pattern(context)
            elif dim_name == 'what':
                drivers['what_pattern'] = self._extract_what_pattern(context)
            elif dim_name == 'when':
                drivers['when_factor'] = self._extract_when_factor(context)
            elif dim_name == 'anomaly':
                drivers['anomaly_factor'] = self._extract_anomaly_factor(context)
        
        # Select appropriate template based on regime
        regime_key = regime.name.lower()
        if regime_key not in self.narrative_templates:
            regime_key = 'ranging'  # Default
        
        templates = self.narrative_templates[regime_key]
        
        # Choose template based on consensus
        if consensus.is_coherent:
            template_idx = 0  # Use primary template for coherent signals
        elif len(consensus.conflicting_dimensions) > 2:
            template_idx = min(2, len(templates) - 1)  # Use conflict template
        else:
            template_idx = 1  # Use secondary template
        
        template = templates[template_idx]
        
        # Fill in template with drivers
        try:
            narrative = template.format(**drivers)
        except KeyError:
            # Fallback if template formatting fails
            narrative = f"Market showing {regime_key} characteristics with mixed signals"
        
        # Add confidence qualifier
        if consensus.confidence_weighted_score > 0.8:
            narrative = "High confidence: " + narrative
        elif consensus.confidence_weighted_score < 0.4:
            narrative = "Low confidence: " + narrative
        
        # Add anomaly warning if significant
        if 'anomaly' in dimensional_readings:
            anomaly_reading = dimensional_readings['anomaly']
            if anomaly_reading.value > 0.6:
                narrative += f" (Anomaly alert: {drivers.get('anomaly_factor', 'unusual activity')})"
        
        return narrative
    
    def _extract_why_driver(self, context: Dict[str, Any]) -> str:
        """Extract primary fundamental driver"""
        if 'primary_driver' in context:
            return context['primary_driver']
        elif 'economic_momentum' in context and context['economic_momentum'] > 0.6:
            return "economic growth"
        elif 'risk_sentiment' in context and context['risk_sentiment'] > 0.6:
            return "risk-on sentiment"
        elif 'risk_sentiment' in context and context['risk_sentiment'] < -0.6:
            return "risk-off sentiment"
        else:
            return "mixed fundamental"
    
    def _extract_how_pattern(self, context: Dict[str, Any]) -> str:
        """Extract primary institutional pattern"""
        if 'dominant_pattern' in context:
            return context['dominant_pattern']
        elif 'order_flow_bias' in context:
            bias = context['order_flow_bias']
            if bias > 0.3:
                return "buying pressure"
            elif bias < -0.3:
                return "selling pressure"
        return "balanced flow"
    
    def _extract_what_pattern(self, context: Dict[str, Any]) -> str:
        """Extract primary technical pattern"""
        if 'primary_pattern' in context:
            return context['primary_pattern']
        elif 'trend_strength' in context and context['trend_strength'] > 0.6:
            return "strong trend"
        elif 'support_resistance_strength' in context and context['support_resistance_strength'] > 0.6:
            return "key level"
        else:
            return "consolidation"
    
    def _extract_when_factor(self, context: Dict[str, Any]) -> str:
        """Extract primary timing factor"""
        if 'primary_timing_factor' in context:
            return context['primary_timing_factor']
        elif 'session_bias' in context and abs(context['session_bias']) > 0.5:
            return "session dynamics"
        elif 'event_proximity' in context and context['event_proximity'] > 0.7:
            return "event anticipation"
        else:
            return "neutral timing"
    
    def _extract_anomaly_factor(self, context: Dict[str, Any]) -> str:
        """Extract primary anomaly factor"""
        if 'anomaly_types' in context and context['anomaly_types']:
            return f"{context['anomaly_types'][0].lower()} detected"
        elif 'manipulation_risk' in context and context['manipulation_risk'] > 0.6:
            return "manipulation risk"
        elif 'regime_change_detected' in context and context['regime_change_detected']:
            return "regime shift"
        else:
            return "market stress"


class IntelligenceEngine:
    """
    Central orchestration engine that synthesizes all dimensions into unified market intelligence.
    
    This engine:
    1. Coordinates all dimensional sensors
    2. Analyzes cross-dimensional relationships
    3. Constructs coherent market narratives
    4. Adapts to changing market conditions
    5. Provides unified market understanding
    """
    
    def __init__(self):
        # Initialize dimensional sensors
        self.dimensions = {
            'why': WhyDimension(),
            'how': HowDimension(),
            'what': WhatDimension(),
            'when': WhenDimension(),
            'anomaly': AnomalyDimension()
        }
        
        # Orchestration components
        self.weighting_system = AdaptiveWeightingSystem()
        self.narrative_constructor = NarrativeConstructor()
        self.memory_bank = MemoryBank()
        
        # Cross-dimensional analysis
        self.correlation_tracker = DimensionalCorrelationMatrix()
        
        # Intelligence history
        self.understanding_history: deque = deque(maxlen=1000)
        self.consensus_history: deque = deque(maxlen=100)
        
        # Current state
        self.current_regime = MarketRegime.UNKNOWN
        self.current_understanding: Optional[MarketUnderstanding] = None
        
        # Performance tracking
        self.prediction_accuracy: deque = deque(maxlen=50)
        
        self._lock = threading.Lock()
    
    def process_market_data(self, data: MarketData) -> MarketUnderstanding:
        """Process market data through all dimensions and synthesize understanding"""
        
        # Get readings from all dimensions
        dimensional_readings = {}
        
        # Process dimensions in order (some may depend on others)
        for dim_name, dimension in self.dimensions.items():
            try:
                reading = dimension.process(data, dimensional_readings)
                dimensional_readings[dim_name] = reading
            except Exception as e:
                # Graceful degradation - create minimal reading
                dimensional_readings[dim_name] = DimensionalReading(
                    dimension=dim_name,
                    value=0.0,
                    confidence=0.0,
                    timestamp=data.timestamp,
                    context={'error': str(e)},
                    influences={}
                )
        
        # Update cross-dimensional correlations
        self.correlation_tracker.update(dimensional_readings)
        
        # Detect current market regime
        self.current_regime = self._detect_regime(dimensional_readings, data)
        
        # Calculate dimensional consensus
        consensus = self._calculate_consensus(dimensional_readings)
        
        # Determine volatility level for adaptive weighting
        volatility_level = self._categorize_volatility(data)
        
        # Calculate adaptive weights
        weights = self.weighting_system.calculate_weights(
            self.current_regime, volatility_level, dimensional_readings
        )
        
        # Synthesize unified understanding
        understanding = self._synthesize_understanding(
            dimensional_readings, weights, consensus, data
        )
        
        # Store in memory bank
        self.memory_bank.store_episode(
            data, dimensional_readings, understanding, self.current_regime
        )
        
        # Update history
        with self._lock:
            self.understanding_history.append(understanding)
            self.consensus_history.append(consensus)
        
        self.current_understanding = understanding
        return understanding
    
    def _detect_regime(self, readings: Dict[str, DimensionalReading], data: MarketData) -> MarketRegime:
        """Detect current market regime from dimensional readings"""
        
        # Get regime indicators from each dimension
        regime_votes = defaultdict(float)
        
        for dim_name, reading in readings.items():
            context = reading.context
            confidence_weight = reading.confidence
            
            if dim_name == 'why':
                # Fundamental regime indicators
                if context.get('economic_momentum', 0) > 0.6:
                    regime_votes[MarketRegime.TRENDING_BULL] += confidence_weight * 0.3
                elif context.get('economic_momentum', 0) < -0.6:
                    regime_votes[MarketRegime.TRENDING_BEAR] += confidence_weight * 0.3
                
                risk_sentiment = context.get('risk_sentiment', 0)
                if abs(risk_sentiment) < 0.3:
                    regime_votes[MarketRegime.RANGING] += confidence_weight * 0.2
            
            elif dim_name == 'what':
                # Technical regime indicators
                trend_strength = context.get('trend_strength', 0)
                if trend_strength > 0.6:
                    if reading.value > 0.5:
                        regime_votes[MarketRegime.TRENDING_BULL] += confidence_weight * 0.4
                    else:
                        regime_votes[MarketRegime.TRENDING_BEAR] += confidence_weight * 0.4
                elif abs(trend_strength) < 0.3:
                    regime_votes[MarketRegime.RANGING] += confidence_weight * 0.3
            
            elif dim_name == 'anomaly':
                # Volatility regime indicators
                if reading.value > 0.7:
                    regime_votes[MarketRegime.VOLATILE] += confidence_weight * 0.5
        
        # Add volatility-based regime detection
        if hasattr(data, 'volatility') and data.volatility > 0:
            if data.volatility > 0.02:  # High volatility threshold
                regime_votes[MarketRegime.VOLATILE] += 0.3
        
        # Determine regime with highest vote
        if regime_votes:
            detected_regime = max(regime_votes.items(), key=lambda x: x[1])[0]
            
            # Require minimum confidence for regime change
            if regime_votes[detected_regime] > 0.5:
                return detected_regime
        
        # Default to previous regime or unknown
        return self.current_regime if self.current_regime != MarketRegime.UNKNOWN else MarketRegime.RANGING
    
    def _calculate_consensus(self, readings: Dict[str, DimensionalReading]) -> DimensionalConsensus:
        """Calculate how well dimensions agree with each other"""
        
        if len(readings) < 2:
            return DimensionalConsensus(
                agreement_score=0.0,
                conflicting_dimensions=[],
                supporting_dimensions=list(readings.keys()),
                confidence_weighted_score=0.0,
                narrative_coherence=0.0
            )
        
        # Calculate pairwise agreements
        agreements = []
        confidence_weights = []
        
        dim_names = list(readings.keys())
        for i in range(len(dim_names)):
            for j in range(i + 1, len(dim_names)):
                dim1, dim2 = dim_names[i], dim_names[j]
                reading1, reading2 = readings[dim1], readings[dim2]
                
                # Calculate agreement based on value similarity and influence
                value_agreement = 1.0 - abs(reading1.value - reading2.value)
                
                # Check for mutual influence
                influence_agreement = 0.0
                if dim2 in reading1.influences and dim1 in reading2.influences:
                    influence_agreement = 0.2  # Bonus for mutual influence
                
                total_agreement = value_agreement + influence_agreement
                agreements.append(total_agreement)
                
                # Weight by combined confidence
                combined_confidence = (reading1.confidence + reading2.confidence) / 2
                confidence_weights.append(combined_confidence)
        
        # Calculate weighted agreement score
        if agreements and confidence_weights:
            agreement_score = np.average(agreements, weights=confidence_weights)
        else:
            agreement_score = 0.0
        
        # Identify conflicting and supporting dimensions
        conflicting_dimensions = []
        supporting_dimensions = []
        
        for dim_name, reading in readings.items():
            # Check if this dimension conflicts with others
            conflicts = 0
            supports = 0
            
            for other_dim, other_reading in readings.items():
                if dim_name != other_dim:
                    value_diff = abs(reading.value - other_reading.value)
                    if value_diff > 0.5:  # Significant disagreement
                        conflicts += 1
                    elif value_diff < 0.2:  # Good agreement
                        supports += 1
            
            if conflicts > supports:
                conflicting_dimensions.append(dim_name)
            else:
                supporting_dimensions.append(dim_name)
        
        # Calculate confidence-weighted score
        total_confidence = sum(reading.confidence for reading in readings.values())
        if total_confidence > 0:
            confidence_weighted_score = sum(
                reading.value * reading.confidence for reading in readings.values()
            ) / total_confidence
        else:
            confidence_weighted_score = 0.0
        
        # Calculate narrative coherence
        narrative_coherence = self._calculate_narrative_coherence(readings)
        
        return DimensionalConsensus(
            agreement_score=agreement_score,
            conflicting_dimensions=conflicting_dimensions,
            supporting_dimensions=supporting_dimensions,
            confidence_weighted_score=confidence_weighted_score,
            narrative_coherence=narrative_coherence
        )
    
    def _calculate_narrative_coherence(self, readings: Dict[str, DimensionalReading]) -> float:
        """Calculate how coherent the overall narrative is"""
        
        # Check for logical consistency between dimensions
        coherence_score = 1.0
        
        # WHY-WHAT consistency: fundamentals should align with technicals
        if 'why' in readings and 'what' in readings:
            why_reading = readings['why']
            what_reading = readings['what']
            
            # Strong fundamentals should support strong technicals
            if why_reading.value > 0.7 and what_reading.value < 0.3:
                coherence_score -= 0.2  # Fundamental-technical divergence
            elif why_reading.value < 0.3 and what_reading.value > 0.7:
                coherence_score -= 0.2
        
        # HOW-WHAT consistency: institutional activity should align with technicals
        if 'how' in readings and 'what' in readings:
            how_reading = readings['how']
            what_reading = readings['what']
            
            # Institutional buying should support bullish technicals
            if how_reading.value > 0.7 and what_reading.value < 0.3:
                coherence_score -= 0.15
            elif how_reading.value < 0.3 and what_reading.value > 0.7:
                coherence_score -= 0.15
        
        # ANOMALY consistency: high anomalies should reduce other dimension confidence
        if 'anomaly' in readings:
            anomaly_reading = readings['anomaly']
            if anomaly_reading.value > 0.7:
                # High anomalies should make other dimensions less certain
                other_high_confidence = sum(
                    1 for dim, reading in readings.items()
                    if dim != 'anomaly' and reading.confidence > 0.8
                )
                if other_high_confidence > 2:
                    coherence_score -= 0.2  # Too confident despite anomalies
        
        return max(0.0, coherence_score)
    
    def _categorize_volatility(self, data: MarketData) -> str:
        """Categorize current volatility level"""
        if hasattr(data, 'volatility') and data.volatility > 0:
            if data.volatility < 0.005:
                return 'low'
            elif data.volatility > 0.02:
                return 'high'
        return 'medium'
    
    def _synthesize_understanding(self, readings: Dict[str, DimensionalReading],
                                weights: Dict[str, float], consensus: DimensionalConsensus,
                                data: MarketData) -> MarketUnderstanding:
        """Synthesize unified market understanding"""
        
        # Calculate weighted confidence
        total_confidence = sum(
            readings[dim].confidence * weight
            for dim, weight in weights.items()
            if dim in readings
        )
        
        # Determine intelligence level
        intelligence_level = self._determine_intelligence_level(consensus, total_confidence)
        
        # Extract primary drivers
        primary_drivers = self._extract_primary_drivers(readings, weights)
        
        # Identify risk factors
        risk_factors = self._identify_risk_factors(readings)
        
        # Calculate predictive power
        predictive_power = self._calculate_predictive_power(consensus, intelligence_level)
        
        # Construct narrative
        narrative = self.narrative_constructor.construct_narrative(
            readings, self.current_regime, consensus
        )
        
        return MarketUnderstanding(
            narrative=narrative,
            confidence=total_confidence,
            intelligence_level=intelligence_level,
            primary_drivers=primary_drivers,
            risk_factors=risk_factors,
            dimensional_consensus=consensus.agreement_score,
            predictive_power=predictive_power,
            regime=self.current_regime,
            timestamp=data.timestamp
        )
    
    def _determine_intelligence_level(self, consensus: DimensionalConsensus, 
                                    total_confidence: float) -> IntelligenceLevel:
        """Determine current intelligence level"""
        
        if consensus.agreement_score > 0.8 and total_confidence > 0.8:
            return IntelligenceLevel.PRESCIENT
        elif consensus.agreement_score > 0.6 and total_confidence > 0.7:
            return IntelligenceLevel.INSIGHTFUL
        elif consensus.agreement_score > 0.4 and total_confidence > 0.5:
            return IntelligenceLevel.AWARE
        elif total_confidence > 0.3:
            return IntelligenceLevel.UNCERTAIN
        else:
            return IntelligenceLevel.CONFUSED
    
    def _extract_primary_drivers(self, readings: Dict[str, DimensionalReading],
                               weights: Dict[str, float]) -> List[str]:
        """Extract primary market drivers"""
        drivers = []
        
        # Sort dimensions by weighted importance
        weighted_dims = sorted(
            [(dim, readings[dim].value * weights.get(dim, 0)) 
             for dim in readings if readings[dim].confidence > 0.5],
            key=lambda x: x[1], reverse=True
        )
        
        # Extract top drivers
        for dim, weighted_value in weighted_dims[:3]:
            if weighted_value > 0.3:
                context = readings[dim].context
                
                if dim == 'why':
                    if context.get('economic_momentum', 0) > 0.5:
                        drivers.append("Economic growth momentum")
                    elif context.get('risk_sentiment', 0) > 0.5:
                        drivers.append("Risk-on sentiment")
                    elif context.get('risk_sentiment', 0) < -0.5:
                        drivers.append("Risk-off sentiment")
                
                elif dim == 'how':
                    if context.get('order_flow_bias', 0) > 0.3:
                        drivers.append("Institutional buying pressure")
                    elif context.get('order_flow_bias', 0) < -0.3:
                        drivers.append("Institutional selling pressure")
                
                elif dim == 'what':
                    if context.get('trend_strength', 0) > 0.5:
                        drivers.append("Strong technical trend")
                    elif context.get('support_resistance_strength', 0) > 0.5:
                        drivers.append("Key technical levels")
                
                elif dim == 'when':
                    if context.get('session_bias', 0) != 0:
                        drivers.append("Session timing dynamics")
                
                elif dim == 'anomaly':
                    if readings[dim].value > 0.5:
                        drivers.append("Market anomalies/manipulation")
        
        return drivers[:3]  # Top 3 drivers
    
    def _identify_risk_factors(self, readings: Dict[str, DimensionalReading]) -> List[str]:
        """Identify current risk factors"""
        risks = []
        
        # Check each dimension for risk indicators
        for dim, reading in readings.items():
            context = reading.context
            
            if dim == 'anomaly' and reading.value > 0.6:
                risks.append("High anomaly/manipulation risk")
            
            elif dim == 'why':
                if context.get('policy_uncertainty', 0) > 0.6:
                    risks.append("Policy uncertainty")
                if context.get('geopolitical_tension', 0) > 0.6:
                    risks.append("Geopolitical tensions")
            
            elif dim == 'how':
                if context.get('liquidity_stress', 0) > 0.6:
                    risks.append("Liquidity stress")
            
            elif dim == 'what':
                if context.get('technical_breakdown_risk', 0) > 0.6:
                    risks.append("Technical breakdown risk")
            
            elif dim == 'when':
                if context.get('event_risk', 0) > 0.6:
                    risks.append("Event timing risk")
        
        # Add consensus-based risks
        if len(readings) > 0:
            avg_confidence = np.mean([r.confidence for r in readings.values()])
            if avg_confidence < 0.4:
                risks.append("Low confidence/unclear signals")
        
        return risks[:4]  # Top 4 risks
    
    def _calculate_predictive_power(self, consensus: DimensionalConsensus,
                                  intelligence_level: IntelligenceLevel) -> float:
        """Calculate predictive power of current understanding"""
        
        base_power = {
            IntelligenceLevel.CONFUSED: 0.1,
            IntelligenceLevel.UNCERTAIN: 0.3,
            IntelligenceLevel.AWARE: 0.5,
            IntelligenceLevel.INSIGHTFUL: 0.7,
            IntelligenceLevel.PRESCIENT: 0.9
        }.get(intelligence_level, 0.1)
        
        # Adjust based on consensus
        consensus_boost = consensus.agreement_score * 0.2
        
        # Adjust based on historical accuracy
        if len(self.prediction_accuracy) >= 10:
            historical_accuracy = np.mean(list(self.prediction_accuracy)[-10:])
            accuracy_adjustment = (historical_accuracy - 0.5) * 0.3
        else:
            accuracy_adjustment = 0.0
        
        predictive_power = base_power + consensus_boost + accuracy_adjustment
        return max(0.0, min(1.0, predictive_power))
    
    def get_current_understanding(self) -> Optional[MarketUnderstanding]:
        """Get current market understanding"""
        return self.current_understanding
    
    def get_dimensional_summary(self) -> Dict[str, Any]:
        """Get summary of all dimensional readings"""
        if not self.current_understanding:
            return {}
        
        summary = {
            'intelligence_level': self.current_understanding.intelligence_level.name,
            'confidence': self.current_understanding.confidence,
            'regime': self.current_understanding.regime.name,
            'narrative': self.current_understanding.narrative,
            'primary_drivers': self.current_understanding.primary_drivers,
            'risk_factors': self.current_understanding.risk_factors,
            'dimensional_consensus': self.current_understanding.dimensional_consensus,
            'predictive_power': self.current_understanding.predictive_power
        }
        
        # Add individual dimension summaries
        summary['dimensions'] = {}
        for dim_name, dimension in self.dimensions.items():
            if hasattr(dimension, 'get_latest_reading'):
                latest = dimension.get_latest_reading()
                if latest:
                    summary['dimensions'][dim_name] = {
                        'value': latest.value,
                        'confidence': latest.confidence,
                        'key_context': self._extract_key_context(latest.context)
                    }
        
        return summary
    
    def _extract_key_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key context information for summary"""
        key_fields = [
            'primary_driver', 'dominant_pattern', 'trend_strength',
            'session_bias', 'anomaly_types', 'confidence_score'
        ]
        
        return {
            field: context[field] for field in key_fields
            if field in context
        }
    
    def update_prediction_accuracy(self, accuracy: float) -> None:
        """Update prediction accuracy tracking"""
        self.prediction_accuracy.append(max(0.0, min(1.0, accuracy)))
        
        # Update individual dimension performance
        if self.current_understanding:
            for dim_name in self.dimensions:
                # Simplified: assume all dimensions contributed equally
                self.weighting_system.update_performance(dim_name, accuracy)

