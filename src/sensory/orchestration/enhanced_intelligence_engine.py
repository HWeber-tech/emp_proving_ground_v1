"""
Enhanced Intelligence Engine - Contextual Fusion and Cross-Dimensional Orchestration

This module implements sophisticated contextual fusion that goes beyond simple arithmetic
to create coherent market narratives and unified understanding. It orchestrates the
5 dimensions (WHY, HOW, WHAT, WHEN, ANOMALY) with cross-dimensional awareness and
adaptive intelligence synthesis.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple, NamedTuple
from collections import deque, defaultdict
from dataclasses import dataclass, field
from enum import Enum, auto
import math
from scipy import stats
from scipy.optimize import minimize
import logging

from src.sensory.core.base import DimensionalReading, MarketData, MarketRegime
from src.sensory.dimensions.enhanced_why_dimension import EnhancedFundamentalIntelligenceEngine
from src.sensory.dimensions.enhanced_how_dimension import InstitutionalMechanicsEngine
from src.sensory.dimensions.enhanced_what_dimension import TechnicalRealityEngine
from src.sensory.dimensions.enhanced_when_dimension import ChronalIntelligenceEngine
from src.sensory.dimensions.enhanced_anomaly_dimension import AnomalyIntelligenceEngine

logger = logging.getLogger(__name__)

class IntelligenceLevel(Enum):
    """Levels of market intelligence understanding"""
    CONFUSED = auto()           # Conflicting signals, low confidence
    UNCERTAIN = auto()          # Some clarity but significant uncertainty
    AWARE = auto()              # Good understanding with moderate confidence
    INSIGHTFUL = auto()         # Clear understanding with high confidence
    PRESCIENT = auto()          # Exceptional understanding, very high confidence

class NarrativeCoherence(Enum):
    """Levels of narrative coherence across dimensions"""
    CONTRADICTORY = auto()      # Dimensions tell conflicting stories
    FRAGMENTED = auto()         # Some alignment but gaps in story
    COHERENT = auto()           # Dimensions tell consistent story
    COMPELLING = auto()         # Strong, unified narrative
    PROPHETIC = auto()          # Extraordinary narrative clarity

class MarketNarrative(Enum):
    """Types of market narratives"""
    FUNDAMENTAL_DRIVEN = auto()     # WHY dimension dominates
    INSTITUTIONAL_FLOW = auto()     # HOW dimension dominates
    TECHNICAL_BREAKOUT = auto()     # WHAT dimension dominates
    TEMPORAL_CYCLE = auto()         # WHEN dimension dominates
    CHAOS_EMERGENCE = auto()        # ANOMALY dimension dominates
    CONFLUENCE_SETUP = auto()       # Multiple dimensions align
    REGIME_TRANSITION = auto()      # Market changing character
    MANIPULATION_ACTIVE = auto()    # Artificial market behavior

@dataclass
class DimensionalCorrelation:
    """Correlation between two dimensions"""
    dimension_a: str
    dimension_b: str
    correlation: float  # -1 to 1
    significance: float  # 0 to 1
    lag: int  # Time lag in periods
    stability: float  # How stable this correlation is

@dataclass
class CrossDimensionalPattern:
    """Pattern involving multiple dimensions"""
    pattern_name: str
    involved_dimensions: List[str]
    pattern_strength: float
    confidence: float
    expected_outcome: str
    historical_accuracy: float
    time_horizon: timedelta

@dataclass
class MarketSynthesis:
    """Unified market understanding synthesis"""
    intelligence_level: IntelligenceLevel
    narrative_coherence: NarrativeCoherence
    dominant_narrative: MarketNarrative
    unified_score: float  # -1 to 1 (bearish to bullish)
    confidence: float  # 0 to 1
    narrative_text: str
    supporting_evidence: List[str]
    contradicting_evidence: List[str]
    risk_factors: List[str]
    opportunity_factors: List[str]

@dataclass
class AdaptiveWeight:
    """Adaptive weight for dimensional fusion"""
    dimension: str
    base_weight: float
    current_weight: float
    performance_factor: float
    regime_factor: float
    correlation_factor: float
    confidence_factor: float

class CorrelationAnalyzer:
    """
    Analyzes correlations and interactions between dimensions
    """
    
    def __init__(self, lookback_periods: int = 200):
        self.lookback_periods = lookback_periods
        
        # Historical dimensional readings
        self.dimensional_history = {
            'WHY': deque(maxlen=lookback_periods),
            'HOW': deque(maxlen=lookback_periods),
            'WHAT': deque(maxlen=lookback_periods),
            'WHEN': deque(maxlen=lookback_periods),
            'ANOMALY': deque(maxlen=lookback_periods)
        }
        
        # Correlation tracking
        self.correlations: Dict[Tuple[str, str], DimensionalCorrelation] = {}
        self.correlation_history = deque(maxlen=50)
        
        # Cross-dimensional patterns
        self.detected_patterns: List[CrossDimensionalPattern] = []
        
    def update_dimensional_reading(self, reading: DimensionalReading) -> None:
        """Update with new dimensional reading"""
        
        if reading.dimension in self.dimensional_history:
            self.dimensional_history[reading.dimension].append({
                'value': reading.value,
                'confidence': reading.confidence,
                'timestamp': reading.timestamp,
                'context': reading.context
            })
        
        # Update correlations periodically
        if len(self.dimensional_history['WHY']) >= 20:
            self._update_correlations()
            self._detect_cross_dimensional_patterns()
    
    def _update_correlations(self) -> None:
        """Update correlation analysis between dimensions"""
        
        dimensions = ['WHY', 'HOW', 'WHAT', 'WHEN', 'ANOMALY']
        
        # Calculate pairwise correlations
        for i, dim_a in enumerate(dimensions):
            for j, dim_b in enumerate(dimensions):
                if i < j:  # Avoid duplicate pairs
                    correlation = self._calculate_dimensional_correlation(dim_a, dim_b)
                    if correlation:
                        self.correlations[(dim_a, dim_b)] = correlation
    
    def _calculate_dimensional_correlation(self, dim_a: str, dim_b: str) -> Optional[DimensionalCorrelation]:
        """Calculate correlation between two dimensions"""
        
        history_a = self.dimensional_history[dim_a]
        history_b = self.dimensional_history[dim_b]
        
        if len(history_a) < 20 or len(history_b) < 20:
            return None
        
        # Extract values and align timestamps
        values_a = []
        values_b = []
        
        # Get common time periods
        timestamps_a = [reading['timestamp'] for reading in history_a]
        timestamps_b = [reading['timestamp'] for reading in history_b]
        
        for reading_a in history_a:
            timestamp = reading_a['timestamp']
            
            # Find corresponding reading in dimension B
            for reading_b in history_b:
                if abs((reading_b['timestamp'] - timestamp).total_seconds()) < 300:  # Within 5 minutes
                    values_a.append(reading_a['value'])
                    values_b.append(reading_b['value'])
                    break
        
        if len(values_a) < 10:
            return None
        
        # Calculate correlation
        try:
            correlation, p_value = stats.pearsonr(values_a, values_b)
            significance = 1.0 - p_value if not np.isnan(p_value) else 0.0
            
            # Calculate lag correlation (simplified)
            lag = self._calculate_optimal_lag(values_a, values_b)
            
            # Calculate stability (how consistent correlation is over time)
            stability = self._calculate_correlation_stability(values_a, values_b)
            
            return DimensionalCorrelation(
                dimension_a=dim_a,
                dimension_b=dim_b,
                correlation=correlation if not np.isnan(correlation) else 0.0,
                significance=significance,
                lag=lag,
                stability=stability
            )
        
        except Exception as e:
            logger.warning(f"Failed to calculate correlation between {dim_a} and {dim_b}: {e}")
            return None
    
    def _calculate_optimal_lag(self, values_a: List[float], values_b: List[float]) -> int:
        """Calculate optimal lag between two time series"""
        
        if len(values_a) < 10:
            return 0
        
        max_lag = min(5, len(values_a) // 4)  # Maximum 5 periods or 25% of data
        best_correlation = 0.0
        best_lag = 0
        
        for lag in range(-max_lag, max_lag + 1):
            try:
                if lag == 0:
                    corr, _ = stats.pearsonr(values_a, values_b)
                elif lag > 0:
                    # B leads A
                    if len(values_a) > lag and len(values_b) > lag:
                        corr, _ = stats.pearsonr(values_a[lag:], values_b[:-lag])
                    else:
                        continue
                else:
                    # A leads B
                    lag_abs = abs(lag)
                    if len(values_a) > lag_abs and len(values_b) > lag_abs:
                        corr, _ = stats.pearsonr(values_a[:-lag_abs], values_b[lag_abs:])
                    else:
                        continue
                
                if not np.isnan(corr) and abs(corr) > abs(best_correlation):
                    best_correlation = corr
                    best_lag = lag
            
            except:
                continue
        
        return best_lag
    
    def _calculate_correlation_stability(self, values_a: List[float], values_b: List[float]) -> float:
        """Calculate how stable the correlation is over time"""
        
        if len(values_a) < 20:
            return 0.5
        
        # Calculate rolling correlations
        window_size = 10
        rolling_correlations = []
        
        for i in range(window_size, len(values_a)):
            window_a = values_a[i-window_size:i]
            window_b = values_b[i-window_size:i]
            
            try:
                corr, _ = stats.pearsonr(window_a, window_b)
                if not np.isnan(corr):
                    rolling_correlations.append(corr)
            except:
                continue
        
        if len(rolling_correlations) < 3:
            return 0.5
        
        # Stability = 1 - coefficient of variation
        mean_corr = np.mean(rolling_correlations)
        std_corr = np.std(rolling_correlations)
        
        if abs(mean_corr) > 0.1:
            stability = 1.0 - (std_corr / abs(mean_corr))
            return max(0.0, min(stability, 1.0))
        else:
            return 0.5
    
    def _detect_cross_dimensional_patterns(self) -> None:
        """Detect patterns involving multiple dimensions"""
        
        self.detected_patterns.clear()
        
        # Pattern 1: Fundamental-Technical Confluence
        confluence_pattern = self._detect_fundamental_technical_confluence()
        if confluence_pattern:
            self.detected_patterns.append(confluence_pattern)
        
        # Pattern 2: Institutional-Temporal Alignment
        institutional_temporal = self._detect_institutional_temporal_alignment()
        if institutional_temporal:
            self.detected_patterns.append(institutional_temporal)
        
        # Pattern 3: Anomaly-Driven Regime Change
        anomaly_regime = self._detect_anomaly_regime_change()
        if anomaly_regime:
            self.detected_patterns.append(anomaly_regime)
        
        # Pattern 4: Multi-Dimensional Breakout Setup
        breakout_setup = self._detect_multidimensional_breakout()
        if breakout_setup:
            self.detected_patterns.append(breakout_setup)
    
    def _detect_fundamental_technical_confluence(self) -> Optional[CrossDimensionalPattern]:
        """Detect when fundamental and technical analysis align"""
        
        why_history = self.dimensional_history['WHY']
        what_history = self.dimensional_history['WHAT']
        
        if len(why_history) < 10 or len(what_history) < 10:
            return None
        
        # Get recent readings
        recent_why = [reading['value'] for reading in list(why_history)[-10:]]
        recent_what = [reading['value'] for reading in list(what_history)[-10:]]
        
        # Check for alignment
        why_direction = 1 if np.mean(recent_why) > 0 else -1
        what_direction = 1 if np.mean(recent_what) > 0 else -1
        
        if why_direction == what_direction:
            # Calculate strength of alignment
            why_strength = abs(np.mean(recent_why))
            what_strength = abs(np.mean(recent_what))
            
            alignment_strength = min(why_strength, what_strength)
            
            if alignment_strength > 0.3:  # Significant alignment
                return CrossDimensionalPattern(
                    pattern_name='fundamental_technical_confluence',
                    involved_dimensions=['WHY', 'WHAT'],
                    pattern_strength=alignment_strength,
                    confidence=0.7,
                    expected_outcome='trend_continuation' if why_direction > 0 else 'trend_reversal',
                    historical_accuracy=0.65,
                    time_horizon=timedelta(hours=4)
                )
        
        return None
    
    def _detect_institutional_temporal_alignment(self) -> Optional[CrossDimensionalPattern]:
        """Detect when institutional flow aligns with temporal patterns"""
        
        how_history = self.dimensional_history['HOW']
        when_history = self.dimensional_history['WHEN']
        
        if len(how_history) < 5 or len(when_history) < 5:
            return None
        
        # Get recent readings
        recent_how = list(how_history)[-5:]
        recent_when = list(when_history)[-5:]
        
        # Check for high institutional activity during high temporal activity
        avg_how = np.mean([reading['value'] for reading in recent_how])
        avg_when = np.mean([reading['value'] for reading in recent_when])
        
        if avg_how > 0.5 and avg_when > 0.5:  # Both dimensions showing high activity
            
            # Check for session overlap or high-activity periods
            latest_when_context = recent_when[-1]['context']
            temporal_regime = latest_when_context.get('temporal_regime', '')
            
            if 'HIGH_ACTIVITY' in temporal_regime or 'OVERLAP' in temporal_regime:
                return CrossDimensionalPattern(
                    pattern_name='institutional_temporal_alignment',
                    involved_dimensions=['HOW', 'WHEN'],
                    pattern_strength=min(avg_how, avg_when),
                    confidence=0.6,
                    expected_outcome='volatility_spike',
                    historical_accuracy=0.55,
                    time_horizon=timedelta(hours=2)
                )
        
        return None
    
    def _detect_anomaly_regime_change(self) -> Optional[CrossDimensionalPattern]:
        """Detect when anomalies signal regime changes"""
        
        anomaly_history = self.dimensional_history['ANOMALY']
        
        if len(anomaly_history) < 10:
            return None
        
        recent_anomalies = [reading['value'] for reading in list(anomaly_history)[-10:]]
        
        # Check for sustained high anomaly levels
        if np.mean(recent_anomalies) > 0.4 and np.std(recent_anomalies) < 0.2:
            
            # Check if other dimensions are showing uncertainty
            other_dimensions = ['WHY', 'HOW', 'WHAT', 'WHEN']
            uncertainty_count = 0
            
            for dim in other_dimensions:
                if len(self.dimensional_history[dim]) >= 5:
                    recent_confidence = [
                        reading['confidence'] 
                        for reading in list(self.dimensional_history[dim])[-5:]
                    ]
                    if np.mean(recent_confidence) < 0.5:
                        uncertainty_count += 1
            
            if uncertainty_count >= 2:  # At least 2 dimensions showing uncertainty
                return CrossDimensionalPattern(
                    pattern_name='anomaly_regime_change',
                    involved_dimensions=['ANOMALY'] + other_dimensions,
                    pattern_strength=np.mean(recent_anomalies),
                    confidence=0.5,
                    expected_outcome='regime_transition',
                    historical_accuracy=0.45,
                    time_horizon=timedelta(hours=6)
                )
        
        return None
    
    def _detect_multidimensional_breakout(self) -> Optional[CrossDimensionalPattern]:
        """Detect when multiple dimensions suggest breakout conditions"""
        
        # Need at least 3 dimensions with recent data
        active_dimensions = []
        for dim in ['WHY', 'HOW', 'WHAT', 'WHEN']:
            if len(self.dimensional_history[dim]) >= 3:
                active_dimensions.append(dim)
        
        if len(active_dimensions) < 3:
            return None
        
        # Check for alignment across dimensions
        aligned_dimensions = []
        total_strength = 0.0
        
        for dim in active_dimensions:
            recent_readings = list(self.dimensional_history[dim])[-3:]
            avg_value = np.mean([reading['value'] for reading in recent_readings])
            avg_confidence = np.mean([reading['confidence'] for reading in recent_readings])
            
            if abs(avg_value) > 0.3 and avg_confidence > 0.5:  # Strong signal with confidence
                aligned_dimensions.append(dim)
                total_strength += abs(avg_value) * avg_confidence
        
        if len(aligned_dimensions) >= 3:  # At least 3 dimensions aligned
            pattern_strength = total_strength / len(aligned_dimensions)
            
            return CrossDimensionalPattern(
                pattern_name='multidimensional_breakout',
                involved_dimensions=aligned_dimensions,
                pattern_strength=pattern_strength,
                confidence=0.8,
                expected_outcome='significant_move',
                historical_accuracy=0.7,
                time_horizon=timedelta(hours=3)
            )
        
        return None
    
    def get_dimensional_correlations(self) -> Dict[Tuple[str, str], DimensionalCorrelation]:
        """Get current dimensional correlations"""
        return self.correlations.copy()
    
    def get_cross_dimensional_patterns(self) -> List[CrossDimensionalPattern]:
        """Get detected cross-dimensional patterns"""
        return self.detected_patterns.copy()

class AdaptiveWeightManager:
    """
    Manages adaptive weights for dimensional fusion based on performance and context
    """
    
    def __init__(self):
        # Initialize base weights (equal by default)
        self.weights = {
            'WHY': AdaptiveWeight('WHY', 0.2, 0.2, 1.0, 1.0, 1.0, 1.0),
            'HOW': AdaptiveWeight('HOW', 0.2, 0.2, 1.0, 1.0, 1.0, 1.0),
            'WHAT': AdaptiveWeight('WHAT', 0.2, 0.2, 1.0, 1.0, 1.0, 1.0),
            'WHEN': AdaptiveWeight('WHEN', 0.2, 0.2, 1.0, 1.0, 1.0, 1.0),
            'ANOMALY': AdaptiveWeight('ANOMALY', 0.2, 0.2, 1.0, 1.0, 1.0, 1.0)
        }
        
        # Performance tracking
        self.performance_history = defaultdict(list)
        self.prediction_accuracy = defaultdict(lambda: 0.5)
        
        # Market regime tracking
        self.current_regime = MarketRegime.UNKNOWN
        
    def update_performance(self, dimension: str, accuracy: float) -> None:
        """Update performance tracking for a dimension"""
        
        if dimension in self.weights:
            self.performance_history[dimension].append(accuracy)
            
            # Keep only recent performance data
            if len(self.performance_history[dimension]) > 20:
                self.performance_history[dimension] = self.performance_history[dimension][-20:]
            
            # Update exponential moving average
            alpha = 0.1
            self.prediction_accuracy[dimension] = (
                alpha * accuracy + (1 - alpha) * self.prediction_accuracy[dimension]
            )
            
            # Update performance factor
            self.weights[dimension].performance_factor = self.prediction_accuracy[dimension]
    
    def update_regime_factors(self, regime: MarketRegime) -> None:
        """Update regime-based weight factors"""
        
        self.current_regime = regime
        
        # Adjust weights based on market regime
        regime_adjustments = {
            MarketRegime.TRENDING_BULL: {
                'WHY': 1.2,    # Fundamentals matter more in trends
                'HOW': 1.1,    # Institutional flow important
                'WHAT': 1.3,   # Technical analysis crucial
                'WHEN': 0.9,   # Timing less critical
                'ANOMALY': 0.8 # Anomalies less relevant
            },
            MarketRegime.TRENDING_BEAR: {
                'WHY': 1.2,
                'HOW': 1.1,
                'WHAT': 1.3,
                'WHEN': 0.9,
                'ANOMALY': 0.8
            },
            MarketRegime.RANGING: {
                'WHY': 0.8,    # Fundamentals less important
                'HOW': 1.2,    # Institutional levels matter
                'WHAT': 1.1,   # Support/resistance key
                'WHEN': 1.3,   # Timing crucial in ranges
                'ANOMALY': 1.0 # Normal anomaly relevance
            },
            MarketRegime.VOLATILE: {
                'WHY': 0.9,
                'HOW': 1.1,
                'WHAT': 0.8,   # Technical less reliable
                'WHEN': 1.2,   # Session timing important
                'ANOMALY': 1.4 # Anomalies very relevant
            },
            MarketRegime.TRANSITIONAL: {
                'WHY': 1.1,
                'HOW': 1.0,
                'WHAT': 0.9,
                'WHEN': 1.0,
                'ANOMALY': 1.3 # Transitions often anomalous
            }
        }
        
        adjustments = regime_adjustments.get(regime, {})
        
        for dimension, factor in adjustments.items():
            if dimension in self.weights:
                self.weights[dimension].regime_factor = factor
    
    def update_correlation_factors(self, correlations: Dict[Tuple[str, str], DimensionalCorrelation]) -> None:
        """Update correlation-based weight factors"""
        
        # Calculate correlation strength for each dimension
        correlation_strengths = defaultdict(list)
        
        for (dim_a, dim_b), correlation in correlations.items():
            # High correlation with other dimensions = higher weight
            strength = abs(correlation.correlation) * correlation.significance * correlation.stability
            
            correlation_strengths[dim_a].append(strength)
            correlation_strengths[dim_b].append(strength)
        
        # Update correlation factors
        for dimension in self.weights:
            if dimension in correlation_strengths:
                avg_correlation = np.mean(correlation_strengths[dimension])
                # Higher correlation = higher weight (up to 1.3x)
                self.weights[dimension].correlation_factor = 1.0 + (avg_correlation * 0.3)
            else:
                self.weights[dimension].correlation_factor = 1.0
    
    def update_confidence_factors(self, readings: Dict[str, DimensionalReading]) -> None:
        """Update confidence-based weight factors"""
        
        for dimension, reading in readings.items():
            if dimension in self.weights:
                # Higher confidence = higher weight
                self.weights[dimension].confidence_factor = reading.confidence
    
    def calculate_current_weights(self) -> Dict[str, float]:
        """Calculate current adaptive weights"""
        
        # Calculate raw weights
        raw_weights = {}
        for dimension, weight_obj in self.weights.items():
            raw_weight = (
                weight_obj.base_weight *
                weight_obj.performance_factor *
                weight_obj.regime_factor *
                weight_obj.correlation_factor *
                weight_obj.confidence_factor
            )
            raw_weights[dimension] = raw_weight
        
        # Normalize weights to sum to 1.0
        total_weight = sum(raw_weights.values())
        
        if total_weight > 0:
            normalized_weights = {
                dimension: weight / total_weight
                for dimension, weight in raw_weights.items()
            }
        else:
            # Fallback to equal weights
            normalized_weights = {dimension: 0.2 for dimension in self.weights}
        
        # Update current weights
        for dimension, weight in normalized_weights.items():
            self.weights[dimension].current_weight = weight
        
        return normalized_weights
    
    def get_weight_diagnostics(self) -> Dict[str, Any]:
        """Get diagnostic information about weight calculations"""
        
        diagnostics = {}
        
        for dimension, weight_obj in self.weights.items():
            diagnostics[dimension] = {
                'base_weight': weight_obj.base_weight,
                'current_weight': weight_obj.current_weight,
                'performance_factor': weight_obj.performance_factor,
                'regime_factor': weight_obj.regime_factor,
                'correlation_factor': weight_obj.correlation_factor,
                'confidence_factor': weight_obj.confidence_factor,
                'prediction_accuracy': self.prediction_accuracy[dimension]
            }
        
        diagnostics['current_regime'] = self.current_regime.name
        
        return diagnostics

class NarrativeGenerator:
    """
    Generates coherent market narratives from dimensional analysis
    """
    
    def __init__(self):
        self.narrative_templates = self._initialize_narrative_templates()
        self.recent_narratives = deque(maxlen=10)
        
    def _initialize_narrative_templates(self) -> Dict[str, Dict]:
        """Initialize narrative templates for different market conditions"""
        
        return {
            'bullish_confluence': {
                'template': "Strong bullish confluence detected with {why_evidence} supporting {what_evidence} during {when_evidence}. {how_evidence} confirms institutional participation.",
                'required_dimensions': ['WHY', 'WHAT', 'WHEN', 'HOW'],
                'min_strength': 0.6
            },
            'bearish_confluence': {
                'template': "Bearish pressure building as {why_evidence} aligns with {what_evidence}. {when_evidence} suggests timing is appropriate, while {how_evidence} shows institutional positioning.",
                'required_dimensions': ['WHY', 'WHAT', 'WHEN', 'HOW'],
                'min_strength': 0.6
            },
            'technical_breakout': {
                'template': "Technical breakout in progress with {what_evidence}. {when_evidence} provides favorable timing, supported by {how_evidence}.",
                'required_dimensions': ['WHAT', 'WHEN', 'HOW'],
                'min_strength': 0.5
            },
            'institutional_flow': {
                'template': "Institutional flow dominates with {how_evidence}. {when_evidence} aligns with typical institutional behavior patterns.",
                'required_dimensions': ['HOW', 'WHEN'],
                'min_strength': 0.5
            },
            'fundamental_shift': {
                'template': "Fundamental shift detected: {why_evidence}. Market structure shows {what_evidence} with {when_evidence}.",
                'required_dimensions': ['WHY', 'WHAT', 'WHEN'],
                'min_strength': 0.5
            },
            'anomaly_warning': {
                'template': "Market anomalies detected: {anomaly_evidence}. This creates uncertainty in {affected_dimensions} and suggests {risk_assessment}.",
                'required_dimensions': ['ANOMALY'],
                'min_strength': 0.4
            },
            'regime_transition': {
                'template': "Market regime transition in progress. {anomaly_evidence} disrupts normal patterns while {why_evidence} suggests new fundamental backdrop.",
                'required_dimensions': ['ANOMALY', 'WHY'],
                'min_strength': 0.4
            },
            'consolidation': {
                'template': "Market consolidation phase with {what_evidence}. {when_evidence} suggests patience required, while {how_evidence} shows institutional positioning.",
                'required_dimensions': ['WHAT', 'WHEN', 'HOW'],
                'min_strength': 0.3
            },
            'uncertainty': {
                'template': "Market uncertainty prevails with conflicting signals. {conflicting_evidence} creates challenging environment requiring careful risk management.",
                'required_dimensions': [],
                'min_strength': 0.0
            }
        }
    
    def generate_narrative(self, readings: Dict[str, DimensionalReading], 
                          synthesis: MarketSynthesis,
                          correlations: Dict[Tuple[str, str], DimensionalCorrelation],
                          patterns: List[CrossDimensionalPattern]) -> str:
        """Generate coherent market narrative"""
        
        # Determine narrative type based on synthesis
        narrative_type = self._determine_narrative_type(readings, synthesis, patterns)
        
        # Extract evidence from dimensional readings
        evidence = self._extract_evidence(readings)
        
        # Generate narrative text
        narrative_text = self._construct_narrative(narrative_type, evidence, synthesis)
        
        # Add pattern information
        if patterns:
            pattern_text = self._add_pattern_context(patterns)
            narrative_text += f" {pattern_text}"
        
        # Add correlation insights
        correlation_text = self._add_correlation_insights(correlations)
        if correlation_text:
            narrative_text += f" {correlation_text}"
        
        # Store narrative for consistency tracking
        self.recent_narratives.append({
            'text': narrative_text,
            'type': narrative_type,
            'timestamp': datetime.now(),
            'synthesis': synthesis
        })
        
        return narrative_text
    
    def _determine_narrative_type(self, readings: Dict[str, DimensionalReading],
                                 synthesis: MarketSynthesis,
                                 patterns: List[CrossDimensionalPattern]) -> str:
        """Determine the most appropriate narrative type"""
        
        # Check for specific patterns first
        if patterns:
            for pattern in patterns:
                if pattern.pattern_name == 'multidimensional_breakout':
                    return 'technical_breakout'
                elif pattern.pattern_name == 'fundamental_technical_confluence':
                    if synthesis.unified_score > 0:
                        return 'bullish_confluence'
                    else:
                        return 'bearish_confluence'
                elif pattern.pattern_name == 'anomaly_regime_change':
                    return 'regime_transition'
                elif pattern.pattern_name == 'institutional_temporal_alignment':
                    return 'institutional_flow'
        
        # Check for anomaly dominance
        if 'ANOMALY' in readings and readings['ANOMALY'].value > 0.5:
            return 'anomaly_warning'
        
        # Check for strong directional bias
        if abs(synthesis.unified_score) > 0.6:
            if synthesis.unified_score > 0:
                return 'bullish_confluence'
            else:
                return 'bearish_confluence'
        
        # Check for fundamental dominance
        if 'WHY' in readings and readings['WHY'].value > 0.5:
            return 'fundamental_shift'
        
        # Check for technical dominance
        if 'WHAT' in readings and readings['WHAT'].value > 0.5:
            return 'technical_breakout'
        
        # Check for institutional dominance
        if 'HOW' in readings and readings['HOW'].value > 0.5:
            return 'institutional_flow'
        
        # Check for low volatility/consolidation
        if all(abs(reading.value) < 0.3 for reading in readings.values()):
            return 'consolidation'
        
        # Default to uncertainty
        return 'uncertainty'
    
    def _extract_evidence(self, readings: Dict[str, DimensionalReading]) -> Dict[str, str]:
        """Extract evidence text from dimensional readings"""
        
        evidence = {}
        
        for dimension, reading in readings.items():
            context = reading.context or {}
            
            if dimension == 'WHY':
                evidence['why_evidence'] = self._extract_why_evidence(reading, context)
            elif dimension == 'HOW':
                evidence['how_evidence'] = self._extract_how_evidence(reading, context)
            elif dimension == 'WHAT':
                evidence['what_evidence'] = self._extract_what_evidence(reading, context)
            elif dimension == 'WHEN':
                evidence['when_evidence'] = self._extract_when_evidence(reading, context)
            elif dimension == 'ANOMALY':
                evidence['anomaly_evidence'] = self._extract_anomaly_evidence(reading, context)
        
        return evidence
    
    def _extract_why_evidence(self, reading: DimensionalReading, context: Dict) -> str:
        """Extract WHY dimension evidence"""
        
        if reading.value > 0.5:
            return "positive fundamental backdrop with supportive economic conditions"
        elif reading.value < -0.5:
            return "negative fundamental pressures from economic headwinds"
        elif reading.value > 0.2:
            return "moderately positive fundamental environment"
        elif reading.value < -0.2:
            return "some fundamental concerns emerging"
        else:
            return "neutral fundamental conditions"
    
    def _extract_how_evidence(self, reading: DimensionalReading, context: Dict) -> str:
        """Extract HOW dimension evidence"""
        
        if reading.value > 0.5:
            return "strong institutional buying interest with significant order flow"
        elif reading.value < -0.5:
            return "institutional selling pressure evident in order flow patterns"
        elif reading.value > 0.2:
            return "moderate institutional participation"
        elif reading.value < -0.2:
            return "some institutional distribution detected"
        else:
            return "balanced institutional positioning"
    
    def _extract_what_evidence(self, reading: DimensionalReading, context: Dict) -> str:
        """Extract WHAT dimension evidence"""
        
        if reading.value > 0.5:
            return "strong technical momentum with clear directional bias"
        elif reading.value < -0.5:
            return "weak technical structure with bearish momentum"
        elif reading.value > 0.2:
            return "constructive technical setup developing"
        elif reading.value < -0.2:
            return "technical deterioration in progress"
        else:
            return "neutral technical conditions with range-bound action"
    
    def _extract_when_evidence(self, reading: DimensionalReading, context: Dict) -> str:
        """Extract WHEN dimension evidence"""
        
        session = context.get('current_session', 'unknown')
        regime = context.get('temporal_regime', 'unknown')
        
        if reading.value > 0.5:
            return f"favorable timing with high-activity {session} session"
        elif reading.value < -0.5:
            return f"poor timing during low-activity {session} period"
        elif 'OVERLAP' in session:
            return "session overlap providing increased volatility potential"
        elif 'HIGH_ACTIVITY' in regime:
            return "high-activity period supporting price movement"
        else:
            return "standard session timing with moderate activity"
    
    def _extract_anomaly_evidence(self, reading: DimensionalReading, context: Dict) -> str:
        """Extract ANOMALY dimension evidence"""
        
        anomaly_level = reading.value
        stress_level = context.get('system_stress_level', 0)
        
        if anomaly_level > 0.7:
            return "significant market anomalies creating high uncertainty"
        elif anomaly_level > 0.4:
            return "moderate anomalies detected requiring caution"
        elif stress_level > 0.5:
            return "elevated system stress with potential instability"
        else:
            return "normal market behavior with minimal anomalies"
    
    def _construct_narrative(self, narrative_type: str, evidence: Dict[str, str], 
                           synthesis: MarketSynthesis) -> str:
        """Construct narrative text using templates"""
        
        template_info = self.narrative_templates.get(narrative_type, self.narrative_templates['uncertainty'])
        template = template_info['template']
        
        # Handle special cases
        if narrative_type == 'uncertainty':
            conflicting_evidence = self._identify_conflicting_evidence(evidence)
            return template.format(conflicting_evidence=conflicting_evidence)
        
        # Handle missing evidence with robust error recovery
        try:
            narrative = template.format(**evidence)
        except KeyError as e:
            # Handle missing evidence gracefully by providing fallback text
            missing_key = str(e).strip("'")
            fallback_evidence = {
                'why_evidence': 'fundamental factors',
                'how_evidence': 'institutional activity',
                'what_evidence': 'technical patterns',
                'when_evidence': 'timing considerations',
                'anomaly_evidence': 'market anomalies'
            }
            evidence[missing_key] = fallback_evidence.get(missing_key, 'mixed signals')
            narrative = template.format(**evidence)
        
        return narrative
    
    def _identify_conflicting_evidence(self, evidence: Dict[str, str]) -> str:
        """Identify conflicting evidence for uncertainty narrative"""
        
        conflicts = []
        
        # Check for directional conflicts
        positive_indicators = []
        negative_indicators = []
        
        for key, text in evidence.items():
            if any(word in text.lower() for word in ['positive', 'strong', 'supportive', 'buying']):
                positive_indicators.append(key.replace('_evidence', ''))
            elif any(word in text.lower() for word in ['negative', 'weak', 'selling', 'pressure']):
                negative_indicators.append(key.replace('_evidence', ''))
        
        if positive_indicators and negative_indicators:
            conflicts.append(f"{', '.join(positive_indicators)} showing strength while {', '.join(negative_indicators)} showing weakness")
        
        if not conflicts:
            conflicts.append("mixed signals across multiple dimensions")
        
        return '; '.join(conflicts)
    
    def _add_pattern_context(self, patterns: List[CrossDimensionalPattern]) -> str:
        """Add cross-dimensional pattern context"""
        
        if not patterns:
            return ""
        
        pattern_descriptions = []
        
        for pattern in patterns:
            if pattern.confidence > 0.6:
                pattern_descriptions.append(
                    f"{pattern.pattern_name.replace('_', ' ')} pattern detected with {pattern.confidence:.0%} confidence"
                )
        
        if pattern_descriptions:
            return "Cross-dimensional analysis reveals " + ", ".join(pattern_descriptions) + "."
        
        return ""
    
    def _add_correlation_insights(self, correlations: Dict[Tuple[str, str], DimensionalCorrelation]) -> str:
        """Add correlation insights to narrative"""
        
        strong_correlations = [
            corr for corr in correlations.values()
            if abs(corr.correlation) > 0.6 and corr.significance > 0.7
        ]
        
        if not strong_correlations:
            return ""
        
        insights = []
        
        for corr in strong_correlations[:2]:  # Limit to top 2
            direction = "positive" if corr.correlation > 0 else "negative"
            insights.append(f"{corr.dimension_a}-{corr.dimension_b} showing strong {direction} correlation")
        
        if insights:
            return "Dimensional correlations show " + " and ".join(insights) + "."
        
        return ""

class ContextualFusionEngine:
    """
    Main contextual fusion engine that orchestrates all components
    """
    
    def __init__(self):
        # Initialize dimensional engines
        self.why_engine = EnhancedFundamentalIntelligenceEngine()
        self.how_engine = InstitutionalMechanicsEngine()
        self.what_engine = TechnicalRealityEngine()
        self.when_engine = ChronalIntelligenceEngine()
        self.anomaly_engine = AnomalyIntelligenceEngine()
        
        # Fusion components
        self.correlation_analyzer = CorrelationAnalyzer()
        self.weight_manager = AdaptiveWeightManager()
        self.narrative_generator = NarrativeGenerator()
        
        # Current state
        self.current_readings: Dict[str, DimensionalReading] = {}
        self.current_synthesis: Optional[MarketSynthesis] = None
        
        # Historical synthesis
        self.synthesis_history = deque(maxlen=100)
        
    async def analyze_market_intelligence(self, market_data: MarketData) -> MarketSynthesis:
        """Perform comprehensive market intelligence analysis"""
        
        # Get dimensional readings
        readings = await self._get_dimensional_readings(market_data)
        
        # Update correlation analysis
        for reading in readings.values():
            self.correlation_analyzer.update_dimensional_reading(reading)
        
        # Get correlations and patterns
        correlations = self.correlation_analyzer.get_dimensional_correlations()
        patterns = self.correlation_analyzer.get_cross_dimensional_patterns()
        
        # Update adaptive weights
        self._update_adaptive_weights(readings, correlations)
        
        # Calculate unified synthesis
        synthesis = self._calculate_unified_synthesis(readings, correlations, patterns)
        
        # Generate narrative
        narrative = self.narrative_generator.generate_narrative(
            readings, synthesis, correlations, patterns
        )
        synthesis.narrative_text = narrative
        
        # Store current state
        self.current_readings = readings
        self.current_synthesis = synthesis
        self.synthesis_history.append(synthesis)
        
        return synthesis
    
    async def _get_dimensional_readings(self, market_data: MarketData) -> Dict[str, DimensionalReading]:
        """Get readings from all dimensional engines"""
        
        readings = {}
        
        try:
            # Get WHY reading
            why_reading = await self.why_engine.analyze_fundamental_intelligence(market_data)
            readings['WHY'] = why_reading
        except Exception as e:
            logger.warning(f"Failed to get WHY reading: {e}")
        
        try:
            # Get HOW reading
            how_reading = await self.how_engine.analyze_institutional_intelligence(market_data)
            readings['HOW'] = how_reading
        except Exception as e:
            logger.warning(f"Failed to get HOW reading: {e}")
        
        try:
            # Get WHAT reading
            what_reading = await self.what_engine.analyze_technical_reality(market_data)
            readings['WHAT'] = what_reading
        except Exception as e:
            logger.warning(f"Failed to get WHAT reading: {e}")
        
        try:
            # Get WHEN reading
            when_reading = await self.when_engine.analyze_temporal_intelligence(market_data)
            readings['WHEN'] = when_reading
        except Exception as e:
            logger.warning(f"Failed to get WHEN reading: {e}")
        
        try:
            # Get ANOMALY reading
            anomaly_reading = await self.anomaly_engine.analyze_anomaly_intelligence(market_data)
            readings['ANOMALY'] = anomaly_reading
        except Exception as e:
            logger.warning(f"Failed to get ANOMALY reading: {e}")
        
        return readings
    
    def _update_adaptive_weights(self, readings: Dict[str, DimensionalReading],
                                correlations: Dict[Tuple[str, str], DimensionalCorrelation]) -> None:
        """Update adaptive weights based on current conditions"""
        
        # Update confidence factors
        self.weight_manager.update_confidence_factors(readings)
        
        # Update correlation factors
        self.weight_manager.update_correlation_factors(correlations)
        
        # Determine market regime for regime factors
        regime = self._determine_market_regime(readings)
        self.weight_manager.update_regime_factors(regime)
    
    def _determine_market_regime(self, readings: Dict[str, DimensionalReading]) -> MarketRegime:
        """Determine current market regime from dimensional readings"""
        
        # Simple regime detection based on dimensional characteristics
        if 'ANOMALY' in readings and readings['ANOMALY'].value > 0.6:
            return MarketRegime.VOLATILE
        
        if 'WHAT' in readings:
            what_context = readings['WHAT'].context or {}
            regime = what_context.get('market_regime', 'UNKNOWN')
            
            if 'TRENDING_BULL' in regime:
                return MarketRegime.TRENDING_BULL
            elif 'TRENDING_BEAR' in regime:
                return MarketRegime.TRENDING_BEAR
            elif 'RANGING' in regime:
                return MarketRegime.RANGING
            elif 'TRANSITIONAL' in regime:
                return MarketRegime.TRANSITIONAL
        
        # Check for volatility indicators
        volatility_indicators = 0
        for reading in readings.values():
            if reading.value > 0.7 or reading.confidence < 0.3:
                volatility_indicators += 1
        
        if volatility_indicators >= 2:
            return MarketRegime.VOLATILE
        
        return MarketRegime.UNKNOWN
    
    def _calculate_unified_synthesis(self, readings: Dict[str, DimensionalReading],
                                   correlations: Dict[Tuple[str, str], DimensionalCorrelation],
                                   patterns: List[CrossDimensionalPattern]) -> MarketSynthesis:
        """Calculate unified market synthesis"""
        
        # Get current adaptive weights
        weights = self.weight_manager.calculate_current_weights()
        
        # Calculate weighted unified score
        unified_score = 0.0
        total_weight = 0.0
        
        for dimension, reading in readings.items():
            if dimension in weights:
                weight = weights[dimension]
                weighted_value = reading.value * reading.confidence * weight
                unified_score += weighted_value
                total_weight += weight * reading.confidence
        
        if total_weight > 0:
            unified_score = unified_score / total_weight
        
        # Calculate overall confidence
        confidence = self._calculate_overall_confidence(readings, weights)
        
        # Determine intelligence level
        intelligence_level = self._determine_intelligence_level(readings, confidence, patterns)
        
        # Determine narrative coherence
        narrative_coherence = self._determine_narrative_coherence(readings, correlations)
        
        # Determine dominant narrative
        dominant_narrative = self._determine_dominant_narrative(readings, patterns, unified_score)
        
        # Extract supporting and contradicting evidence
        supporting_evidence, contradicting_evidence = self._extract_evidence_lists(readings, unified_score)
        
        # Identify risk and opportunity factors
        risk_factors, opportunity_factors = self._identify_risk_opportunity_factors(readings, patterns)
        
        return MarketSynthesis(
            intelligence_level=intelligence_level,
            narrative_coherence=narrative_coherence,
            dominant_narrative=dominant_narrative,
            unified_score=unified_score,
            confidence=confidence,
            narrative_text="",  # Will be filled by narrative generator
            supporting_evidence=supporting_evidence,
            contradicting_evidence=contradicting_evidence,
            risk_factors=risk_factors,
            opportunity_factors=opportunity_factors
        )
    
    def _calculate_overall_confidence(self, readings: Dict[str, DimensionalReading],
                                    weights: Dict[str, float]) -> float:
        """Calculate overall confidence in the synthesis"""
        
        weighted_confidence = 0.0
        total_weight = 0.0
        
        for dimension, reading in readings.items():
            if dimension in weights:
                weight = weights[dimension]
                weighted_confidence += reading.confidence * weight
                total_weight += weight
        
        if total_weight > 0:
            base_confidence = weighted_confidence / total_weight
        else:
            base_confidence = 0.5
        
        # Adjust for dimensional agreement
        agreement_bonus = self._calculate_dimensional_agreement(readings)
        
        # Adjust for data quality
        data_quality_factor = min(len(readings) / 5.0, 1.0)  # Prefer all 5 dimensions
        
        final_confidence = base_confidence * (1.0 + agreement_bonus * 0.2) * data_quality_factor
        
        return min(final_confidence, 1.0)
    
    def _calculate_dimensional_agreement(self, readings: Dict[str, DimensionalReading]) -> float:
        """Calculate how much dimensions agree with each other"""
        
        if len(readings) < 2:
            return 0.0
        
        values = [reading.value for reading in readings.values()]
        
        # Calculate pairwise agreement
        agreements = []
        for i in range(len(values)):
            for j in range(i + 1, len(values)):
                # Agreement based on sign and magnitude similarity
                sign_agreement = 1.0 if values[i] * values[j] >= 0 else 0.0
                magnitude_similarity = 1.0 - abs(abs(values[i]) - abs(values[j]))
                agreement = (sign_agreement + magnitude_similarity) / 2.0
                agreements.append(agreement)
        
        return np.mean(agreements) if agreements else 0.0
    
    def _determine_intelligence_level(self, readings: Dict[str, DimensionalReading],
                                    confidence: float,
                                    patterns: List[CrossDimensionalPattern]) -> IntelligenceLevel:
        """Determine the level of market intelligence"""
        
        # High confidence with multiple patterns = prescient
        if confidence > 0.8 and len(patterns) >= 2:
            return IntelligenceLevel.PRESCIENT
        
        # Good confidence with clear patterns = insightful
        if confidence > 0.7 and len(patterns) >= 1:
            return IntelligenceLevel.INSIGHTFUL
        
        # Moderate confidence with some clarity = aware
        if confidence > 0.5 and len(readings) >= 3:
            return IntelligenceLevel.AWARE
        
        # Low confidence or conflicting signals = uncertain
        if confidence > 0.3:
            return IntelligenceLevel.UNCERTAIN
        
        # Very low confidence = confused
        return IntelligenceLevel.CONFUSED
    
    def _determine_narrative_coherence(self, readings: Dict[str, DimensionalReading],
                                     correlations: Dict[Tuple[str, str], DimensionalCorrelation]) -> NarrativeCoherence:
        """Determine narrative coherence across dimensions"""
        
        # Check dimensional agreement
        agreement = self._calculate_dimensional_agreement(readings)
        
        # Check correlation strength
        strong_correlations = sum(
            1 for corr in correlations.values()
            if abs(corr.correlation) > 0.6 and corr.significance > 0.7
        )
        
        # Check confidence consistency
        confidences = [reading.confidence for reading in readings.values()]
        confidence_consistency = 1.0 - np.std(confidences) if len(confidences) > 1 else 1.0
        
        # Combine factors
        coherence_score = (agreement + confidence_consistency) / 2.0
        
        # Adjust for correlations
        if strong_correlations >= 3:
            coherence_score += 0.2
        elif strong_correlations >= 1:
            coherence_score += 0.1
        
        # Determine coherence level
        if coherence_score > 0.9:
            return NarrativeCoherence.PROPHETIC
        elif coherence_score > 0.7:
            return NarrativeCoherence.COMPELLING
        elif coherence_score > 0.5:
            return NarrativeCoherence.COHERENT
        elif coherence_score > 0.3:
            return NarrativeCoherence.FRAGMENTED
        else:
            return NarrativeCoherence.CONTRADICTORY
    
    def _determine_dominant_narrative(self, readings: Dict[str, DimensionalReading],
                                    patterns: List[CrossDimensionalPattern],
                                    unified_score: float) -> MarketNarrative:
        """Determine the dominant market narrative"""
        
        # Check for specific patterns first
        for pattern in patterns:
            if pattern.confidence > 0.6:
                if 'confluence' in pattern.pattern_name:
                    return MarketNarrative.CONFLUENCE_SETUP
                elif 'breakout' in pattern.pattern_name:
                    return MarketNarrative.TECHNICAL_BREAKOUT
                elif 'institutional' in pattern.pattern_name:
                    return MarketNarrative.INSTITUTIONAL_FLOW
                elif 'anomaly' in pattern.pattern_name or 'regime' in pattern.pattern_name:
                    return MarketNarrative.REGIME_TRANSITION
        
        # Check for anomaly dominance
        if 'ANOMALY' in readings and readings['ANOMALY'].value > 0.6:
            anomaly_context = readings['ANOMALY'].context or {}
            if anomaly_context.get('active_manipulations'):
                return MarketNarrative.MANIPULATION_ACTIVE
            else:
                return MarketNarrative.CHAOS_EMERGENCE
        
        # Check for dimensional dominance
        max_value = 0.0
        dominant_dimension = None
        
        for dimension, reading in readings.items():
            weighted_value = abs(reading.value) * reading.confidence
            if weighted_value > max_value:
                max_value = weighted_value
                dominant_dimension = dimension
        
        if dominant_dimension == 'WHY':
            return MarketNarrative.FUNDAMENTAL_DRIVEN
        elif dominant_dimension == 'HOW':
            return MarketNarrative.INSTITUTIONAL_FLOW
        elif dominant_dimension == 'WHAT':
            return MarketNarrative.TECHNICAL_BREAKOUT
        elif dominant_dimension == 'WHEN':
            return MarketNarrative.TEMPORAL_CYCLE
        
        # Check for regime transition indicators
        regime_indicators = 0
        for reading in readings.values():
            if reading.confidence < 0.4:  # Low confidence suggests transition
                regime_indicators += 1
        
        if regime_indicators >= 3:
            return MarketNarrative.REGIME_TRANSITION
        
        # Default based on unified score strength
        if abs(unified_score) > 0.5:
            return MarketNarrative.CONFLUENCE_SETUP
        else:
            return MarketNarrative.REGIME_TRANSITION
    
    def _extract_evidence_lists(self, readings: Dict[str, DimensionalReading],
                              unified_score: float) -> Tuple[List[str], List[str]]:
        """Extract supporting and contradicting evidence"""
        
        supporting_evidence = []
        contradicting_evidence = []
        
        for dimension, reading in readings.items():
            # Evidence supports unified score direction
            if (unified_score > 0 and reading.value > 0.2) or (unified_score < 0 and reading.value < -0.2):
                supporting_evidence.append(f"{dimension} dimension shows {reading.value:.2f} with {reading.confidence:.0%} confidence")
            
            # Evidence contradicts unified score direction
            elif (unified_score > 0 and reading.value < -0.2) or (unified_score < 0 and reading.value > 0.2):
                contradicting_evidence.append(f"{dimension} dimension shows opposing signal of {reading.value:.2f}")
        
        return supporting_evidence, contradicting_evidence
    
    def _identify_risk_opportunity_factors(self, readings: Dict[str, DimensionalReading],
                                         patterns: List[CrossDimensionalPattern]) -> Tuple[List[str], List[str]]:
        """Identify risk and opportunity factors"""
        
        risk_factors = []
        opportunity_factors = []
        
        # Anomaly-based risks
        if 'ANOMALY' in readings and readings['ANOMALY'].value > 0.4:
            risk_factors.append("Elevated anomaly levels increase uncertainty")
            
            anomaly_context = readings['ANOMALY'].context or {}
            if anomaly_context.get('active_manipulations'):
                risk_factors.append("Active market manipulation detected")
            
            if anomaly_context.get('system_stress_level', 0) > 0.5:
                risk_factors.append("High system stress indicates potential instability")
        
        # Low confidence risks
        low_confidence_dimensions = [
            dim for dim, reading in readings.items()
            if reading.confidence < 0.4
        ]
        
        if len(low_confidence_dimensions) >= 2:
            risk_factors.append(f"Low confidence in {', '.join(low_confidence_dimensions)} dimensions")
        
        # Pattern-based opportunities
        for pattern in patterns:
            if pattern.confidence > 0.6:
                opportunity_factors.append(f"{pattern.pattern_name.replace('_', ' ')} pattern with {pattern.confidence:.0%} confidence")
        
        # High confidence opportunities
        high_confidence_dimensions = [
            dim for dim, reading in readings.items()
            if reading.confidence > 0.7 and abs(reading.value) > 0.3
        ]
        
        if len(high_confidence_dimensions) >= 2:
            opportunity_factors.append(f"High confidence signals from {', '.join(high_confidence_dimensions)}")
        
        # Temporal opportunities
        if 'WHEN' in readings:
            when_context = readings['WHEN'].context or {}
            if 'OVERLAP' in when_context.get('current_session', ''):
                opportunity_factors.append("Session overlap provides increased volatility potential")
        
        return risk_factors, opportunity_factors
    
    def get_diagnostic_information(self) -> Dict[str, Any]:
        """Get comprehensive diagnostic information"""
        
        diagnostics = {
            'current_readings': {
                dim: {
                    'value': reading.value,
                    'confidence': reading.confidence,
                    'timestamp': reading.timestamp.isoformat()
                }
                for dim, reading in self.current_readings.items()
            },
            'adaptive_weights': self.weight_manager.get_weight_diagnostics(),
            'correlations': {
                f"{dim_a}-{dim_b}": {
                    'correlation': corr.correlation,
                    'significance': corr.significance,
                    'stability': corr.stability,
                    'lag': corr.lag
                }
                for (dim_a, dim_b), corr in self.correlation_analyzer.get_dimensional_correlations().items()
            },
            'patterns': [
                {
                    'name': pattern.pattern_name,
                    'dimensions': pattern.involved_dimensions,
                    'strength': pattern.pattern_strength,
                    'confidence': pattern.confidence
                }
                for pattern in self.correlation_analyzer.get_cross_dimensional_patterns()
            ]
        }
        
        if self.current_synthesis:
            diagnostics['current_synthesis'] = {
                'intelligence_level': self.current_synthesis.intelligence_level.name,
                'narrative_coherence': self.current_synthesis.narrative_coherence.name,
                'dominant_narrative': self.current_synthesis.dominant_narrative.name,
                'unified_score': self.current_synthesis.unified_score,
                'confidence': self.current_synthesis.confidence
            }
        
        return diagnostics

# Example usage
async def main():
    """Example usage of the enhanced contextual fusion engine"""
    
    # Initialize engine
    engine = ContextualFusionEngine()
    
    # Simulate market data with evolving conditions
    base_price = 1.0950
    
    for i in range(100):
        
        current_time = datetime.now() + timedelta(minutes=i * 5)
        
        # Create evolving market conditions
        if i < 30:  # Initial trending phase
            price_trend = 0.0002
            volatility_base = 0.008
        elif i < 60:  # Consolidation phase
            price_trend = 0.0
            volatility_base = 0.004
        else:  # Breakout phase
            price_trend = 0.0003
            volatility_base = 0.012
        
        # Add some randomness
        price_change = price_trend + np.random.normal(0, 0.0001)
        current_price = base_price + price_change * i
        
        volatility = volatility_base + np.random.exponential(0.002)
        volume = 1500 + np.random.exponential(800)
        
        market_data = MarketData(
            timestamp=current_time,
            bid=current_price - 0.0001,
            ask=current_price + 0.0001,
            volume=volume,
            volatility=volatility
        )
        
        # Analyze market intelligence
        synthesis = await engine.analyze_market_intelligence(market_data)
        
        if i % 20 == 0:  # Print every 20th analysis
            print(f"Market Intelligence Synthesis (Period {i}):")
            print(f"  Intelligence Level: {synthesis.intelligence_level.name}")
            print(f"  Narrative Coherence: {synthesis.narrative_coherence.name}")
            print(f"  Dominant Narrative: {synthesis.dominant_narrative.name}")
            print(f"  Unified Score: {synthesis.unified_score:.3f}")
            print(f"  Confidence: {synthesis.confidence:.3f}")
            print(f"  Narrative: {synthesis.narrative_text}")
            
            if synthesis.supporting_evidence:
                print(f"  Supporting Evidence: {len(synthesis.supporting_evidence)} factors")
            
            if synthesis.risk_factors:
                print(f"  Risk Factors: {len(synthesis.risk_factors)} identified")
            
            if synthesis.opportunity_factors:
                print(f"  Opportunities: {len(synthesis.opportunity_factors)} identified")
            
            print()

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())

