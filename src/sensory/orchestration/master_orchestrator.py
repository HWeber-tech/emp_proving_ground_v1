"""
Sensory Cortex v2.2 - Master Orchestrator

Masterful implementation of cross-dimensional intelligence synthesis.
Implements contextual weighting, graceful degradation, and coherent market awareness.
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import json

from ..core.base import (
    DimensionalSensor, DimensionalReading, MarketData, InstrumentMeta,
    MarketRegime, OrderBookSnapshot
)
from ..core.utils import (
    EMA, WelfordVar, compute_confidence, normalize_signal,
    calculate_momentum, PerformanceTracker
)
from ..dimensions.why_engine import WHYEngine
from ..dimensions.how_engine import HOWEngine
from ..dimensions.what_engine import WATEngine
from ..dimensions.when_engine import WHENEngine
from ..dimensions.anomaly_engine import ANOMALYEngine

logger = logging.getLogger(__name__)


class SynthesisMode(Enum):
    """Synthesis modes for dimensional integration."""
    CONSENSUS = "consensus"          # Require agreement across dimensions
    WEIGHTED_AVERAGE = "weighted"    # Weight by confidence and performance
    DOMINANT_SIGNAL = "dominant"     # Follow strongest signal
    ADAPTIVE = "adaptive"            # Adapt based on market conditions
    ANTIFRAGILE = "antifragile"     # Gain strength from disagreement


class ContextualWeight(Enum):
    """Contextual weighting factors."""
    MARKET_REGIME = "market_regime"
    VOLATILITY = "volatility"
    VOLUME = "volume"
    TIME_OF_DAY = "time_of_day"
    EVENT_PROXIMITY = "event_proximity"
    ANOMALY_LEVEL = "anomaly_level"


@dataclass
class DimensionalState:
    """
    State of a dimensional engine.
    """
    dimension: str
    reading: Optional[DimensionalReading]
    performance_score: float
    reliability_score: float
    last_update: datetime
    error_count: int = 0
    consecutive_errors: int = 0
    is_healthy: bool = True


@dataclass
class SynthesisResult:
    """
    Result of cross-dimensional synthesis.
    """
    timestamp: datetime
    signal_strength: float
    confidence: float
    regime: MarketRegime
    synthesis_mode: SynthesisMode
    dimensional_weights: Dict[str, float]
    dimensional_contributions: Dict[str, float]
    consensus_level: float
    narrative: str
    evidence: Dict[str, Any]
    warnings: List[str] = field(default_factory=list)
    processing_time_ms: float = 0.0


class ContextualWeightManager:
    """
    Manages contextual weighting of dimensional engines based on market conditions.
    """
    
    def __init__(self):
        """Initialize contextual weight manager."""
        self.base_weights = {
            'WHY': 0.25,    # Fundamental analysis
            'HOW': 0.25,    # Institutional mechanics
            'WHAT': 0.25,   # Technical analysis
            'WHEN': 0.15,   # Temporal analysis
            'ANOMALY': 0.10 # Anomaly detection
        }
        
        # Adaptive weight trackers
        self.weight_trackers = {dim: EMA(30) for dim in self.base_weights.keys()}
        
        # Context-specific weight adjustments
        self.regime_adjustments = {
            MarketRegime.TRENDING_STRONG: {'WHAT': 1.3, 'HOW': 1.2, 'WHY': 0.9},
            MarketRegime.TRENDING_WEAK: {'WHAT': 1.1, 'HOW': 1.1, 'WHY': 1.0},
            MarketRegime.CONSOLIDATING: {'WHY': 1.2, 'WHEN': 1.3, 'WHAT': 0.9},
            MarketRegime.BREAKOUT: {'HOW': 1.4, 'WHAT': 1.2, 'ANOMALY': 1.1},
            MarketRegime.EXHAUSTED: {'ANOMALY': 1.5, 'WHEN': 1.2, 'WHAT': 0.8}
        }
        
        self.volatility_adjustments = {
            'high': {'ANOMALY': 1.3, 'HOW': 1.2, 'WHAT': 1.1},
            'medium': {'WHY': 1.1, 'WHAT': 1.0, 'HOW': 1.0},
            'low': {'WHY': 1.2, 'WHEN': 1.1, 'WHAT': 0.9}
        }
        
    def calculate_contextual_weights(
        self,
        dimensional_states: Dict[str, DimensionalState],
        market_data: MarketData,
        current_regime: MarketRegime
    ) -> Dict[str, float]:
        """
        Calculate contextual weights for dimensional engines.
        
        Args:
            dimensional_states: Current state of all dimensional engines
            market_data: Current market data
            current_regime: Current market regime
            
        Returns:
            Contextual weights for each dimension
        """
        weights = self.base_weights.copy()
        
        # Apply regime-based adjustments
        regime_adj = self.regime_adjustments.get(current_regime, {})
        for dim, multiplier in regime_adj.items():
            if dim in weights:
                weights[dim] *= multiplier
        
        # Apply volatility-based adjustments
        volatility_level = self._assess_volatility_level(market_data)
        vol_adj = self.volatility_adjustments.get(volatility_level, {})
        for dim, multiplier in vol_adj.items():
            if dim in weights:
                weights[dim] *= multiplier
        
        # Apply performance-based adjustments
        for dim, state in dimensional_states.items():
            if dim in weights:
                performance_multiplier = 0.5 + state.performance_score  # 0.5 to 1.5 range
                reliability_multiplier = 0.7 + (state.reliability_score * 0.6)  # 0.7 to 1.3 range
                weights[dim] *= performance_multiplier * reliability_multiplier
        
        # Apply health-based adjustments (reduce weight for unhealthy engines)
        for dim, state in dimensional_states.items():
            if dim in weights and not state.is_healthy:
                weights[dim] *= 0.3  # Significantly reduce weight for unhealthy engines
        
        # Normalize weights to sum to 1.0
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {dim: w / total_weight for dim, w in weights.items()}
        
        # Update adaptive trackers
        for dim, weight in weights.items():
            if dim in self.weight_trackers:
                self.weight_trackers[dim].update(weight)
        
        return weights
    
    def _assess_volatility_level(self, market_data: MarketData) -> str:
        """Assess current volatility level."""
        # Simple volatility assessment based on spread and range
        spread_ratio = (market_data.ask - market_data.bid) / market_data.close
        range_ratio = (market_data.high - market_data.low) / market_data.close
        
        volatility_score = (spread_ratio + range_ratio) / 2
        
        if volatility_score > 0.01:  # 1%
            return 'high'
        elif volatility_score > 0.005:  # 0.5%
            return 'medium'
        else:
            return 'low'


class GracefulDegradationManager:
    """
    Manages graceful degradation when dimensional engines fail or become unreliable.
    """
    
    def __init__(self):
        """Initialize graceful degradation manager."""
        self.min_healthy_engines = 2
        self.error_threshold = 5
        self.consecutive_error_threshold = 3
        self.recovery_time_threshold = 300  # 5 minutes
        
    def assess_engine_health(self, state: DimensionalState) -> bool:
        """
        Assess health of a dimensional engine.
        
        Args:
            state: Dimensional engine state
            
        Returns:
            True if engine is healthy
        """
        # Check error counts
        if state.consecutive_errors >= self.consecutive_error_threshold:
            return False
        
        if state.error_count >= self.error_threshold:
            return False
        
        # Check last update time
        time_since_update = (datetime.utcnow() - state.last_update).total_seconds()
        if time_since_update > self.recovery_time_threshold:
            return False
        
        # Check reading quality
        if state.reading and state.reading.confidence < 0.1:
            return False
        
        return True
    
    def get_degradation_strategy(
        self,
        dimensional_states: Dict[str, DimensionalState]
    ) -> Dict[str, Any]:
        """
        Get degradation strategy based on engine health.
        
        Args:
            dimensional_states: Current state of all engines
            
        Returns:
            Degradation strategy
        """
        healthy_engines = [
            dim for dim, state in dimensional_states.items()
            if self.assess_engine_health(state)
        ]
        
        strategy = {
            'healthy_engines': healthy_engines,
            'degraded_engines': [dim for dim in dimensional_states.keys() if dim not in healthy_engines],
            'can_operate': len(healthy_engines) >= self.min_healthy_engines,
            'confidence_penalty': max(0.0, 1.0 - len(healthy_engines) / len(dimensional_states)),
            'fallback_mode': len(healthy_engines) < 3
        }
        
        return strategy


class NarrativeGenerator:
    """
    Generates coherent market narratives from dimensional analysis.
    """
    
    def __init__(self):
        """Initialize narrative generator."""
        self.narrative_templates = {
            MarketRegime.TRENDING_STRONG: [
                "Strong {direction} trend supported by {primary_factors}",
                "Institutional {direction} momentum with {confluence} confluence",
                "Clear {direction} structure with {timing} timing"
            ],
            MarketRegime.TRENDING_WEAK: [
                "Weak {direction} trend with {concerns} concerns",
                "Tentative {direction} bias amid {uncertainty}",
                "Developing {direction} structure with {caution} caution"
            ],
            MarketRegime.CONSOLIDATING: [
                "Consolidation phase with {balance} balance",
                "Range-bound action with {factors} factors",
                "Sideways movement pending {catalyst} catalyst"
            ],
            MarketRegime.BREAKOUT: [
                "Breakout potential with {setup} setup",
                "Structure break suggesting {direction} move",
                "Momentum building for {direction} breakout"
            ],
            MarketRegime.EXHAUSTED: [
                "Exhausted move with {reversal} reversal signs",
                "Overextended conditions suggesting {correction}",
                "Manipulation concerns amid {anomalies} anomalies"
            ]
        }
    
    def generate_narrative(
        self,
        synthesis_result: SynthesisResult,
        dimensional_readings: Dict[str, DimensionalReading]
    ) -> str:
        """
        Generate coherent market narrative.
        
        Args:
            synthesis_result: Synthesis result
            dimensional_readings: Individual dimensional readings
            
        Returns:
            Market narrative string
        """
        regime = synthesis_result.regime
        signal_strength = synthesis_result.signal_strength
        
        # Determine direction
        direction = "bullish" if signal_strength > 0 else "bearish" if signal_strength < 0 else "neutral"
        
        # Get primary contributing factors
        primary_factors = self._get_primary_factors(synthesis_result, dimensional_readings)
        
        # Get template for regime
        templates = self.narrative_templates.get(regime, ["Market showing {direction} bias"])
        template = templates[0]  # Use first template for simplicity
        
        # Fill template variables
        narrative = template.format(
            direction=direction,
            primary_factors=primary_factors,
            confluence=self._get_confluence_description(synthesis_result),
            timing=self._get_timing_description(dimensional_readings.get('WHEN')),
            concerns=self._get_concerns_description(dimensional_readings),
            uncertainty=self._get_uncertainty_description(synthesis_result),
            caution=self._get_caution_description(dimensional_readings),
            balance=self._get_balance_description(synthesis_result),
            factors=self._get_factors_description(dimensional_readings),
            catalyst=self._get_catalyst_description(dimensional_readings),
            setup=self._get_setup_description(dimensional_readings),
            reversal=self._get_reversal_description(dimensional_readings),
            correction=self._get_correction_description(dimensional_readings),
            anomalies=self._get_anomalies_description(dimensional_readings.get('ANOMALY'))
        )
        
        return narrative
    
    def _get_primary_factors(
        self,
        synthesis_result: SynthesisResult,
        dimensional_readings: Dict[str, DimensionalReading]
    ) -> str:
        """Get primary contributing factors."""
        contributions = synthesis_result.dimensional_contributions
        top_contributors = sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)[:2]
        
        factor_names = {
            'WHY': 'fundamentals',
            'HOW': 'institutional flow',
            'WHAT': 'technical structure',
            'WHEN': 'timing factors',
            'ANOMALY': 'anomaly detection'
        }
        
        factors = [factor_names.get(dim, dim.lower()) for dim, _ in top_contributors]
        return " and ".join(factors)
    
    def _get_confluence_description(self, synthesis_result: SynthesisResult) -> str:
        """Get confluence description."""
        consensus = synthesis_result.consensus_level
        if consensus > 0.8:
            return "strong"
        elif consensus > 0.6:
            return "moderate"
        else:
            return "weak"
    
    def _get_timing_description(self, when_reading: Optional[DimensionalReading]) -> str:
        """Get timing description."""
        if not when_reading:
            return "unclear"
        
        signal = when_reading.signal_strength
        if abs(signal) > 0.6:
            return "favorable" if signal > 0 else "unfavorable"
        else:
            return "neutral"
    
    def _get_concerns_description(self, dimensional_readings: Dict[str, DimensionalReading]) -> str:
        """Get concerns description."""
        concerns = []
        
        anomaly_reading = dimensional_readings.get('ANOMALY')
        if anomaly_reading and anomaly_reading.signal_strength < -0.5:
            concerns.append("manipulation")
        
        when_reading = dimensional_readings.get('WHEN')
        if when_reading and when_reading.signal_strength < -0.3:
            concerns.append("timing")
        
        return " and ".join(concerns) if concerns else "mixed signals"
    
    def _get_uncertainty_description(self, synthesis_result: SynthesisResult) -> str:
        """Get uncertainty description."""
        confidence = synthesis_result.confidence
        if confidence < 0.4:
            return "high uncertainty"
        elif confidence < 0.6:
            return "moderate uncertainty"
        else:
            return "low uncertainty"
    
    def _get_caution_description(self, dimensional_readings: Dict[str, DimensionalReading]) -> str:
        """Get caution description."""
        return "advised"  # Simplified
    
    def _get_balance_description(self, synthesis_result: SynthesisResult) -> str:
        """Get balance description."""
        consensus = synthesis_result.consensus_level
        if consensus > 0.7:
            return "clear directional"
        else:
            return "mixed signal"
    
    def _get_factors_description(self, dimensional_readings: Dict[str, DimensionalReading]) -> str:
        """Get factors description."""
        return "multiple"  # Simplified
    
    def _get_catalyst_description(self, dimensional_readings: Dict[str, DimensionalReading]) -> str:
        """Get catalyst description."""
        when_reading = dimensional_readings.get('WHEN')
        if when_reading and when_reading.context.get('next_major_event'):
            return "economic event"
        else:
            return "technical breakout"
    
    def _get_setup_description(self, dimensional_readings: Dict[str, DimensionalReading]) -> str:
        """Get setup description."""
        how_reading = dimensional_readings.get('HOW')
        if how_reading and how_reading.signal_strength != 0:
            return "institutional"
        else:
            return "technical"
    
    def _get_reversal_description(self, dimensional_readings: Dict[str, DimensionalReading]) -> str:
        """Get reversal description."""
        return "potential"  # Simplified
    
    def _get_correction_description(self, dimensional_readings: Dict[str, DimensionalReading]) -> str:
        """Get correction description."""
        return "pullback"  # Simplified
    
    def _get_anomalies_description(self, anomaly_reading: Optional[DimensionalReading]) -> str:
        """Get anomalies description."""
        if not anomaly_reading:
            return "none detected"
        
        if anomaly_reading.signal_strength < -0.7:
            return "significant"
        elif anomaly_reading.signal_strength < -0.4:
            return "moderate"
        else:
            return "minor"


class MasterOrchestrator:
    """
    Masterful orchestration engine for cross-dimensional intelligence synthesis.
    Implements contextual weighting, graceful degradation, and coherent market awareness.
    """
    
    def __init__(self, instrument_meta: InstrumentMeta):
        """
        Initialize master orchestrator.
        
        Args:
            instrument_meta: Instrument metadata
        """
        self.instrument_meta = instrument_meta
        
        # Initialize dimensional engines
        self.engines = {
            'WHY': WHYEngine(instrument_meta),
            'HOW': HOWEngine(instrument_meta),
            'WHAT': WATEngine(instrument_meta),
            'WHEN': WHENEngine(instrument_meta),
            'ANOMALY': ANOMALYEngine(instrument_meta)
        }
        
        # Initialize managers
        self.weight_manager = ContextualWeightManager()
        self.degradation_manager = GracefulDegradationManager()
        self.narrative_generator = NarrativeGenerator()
        
        # State tracking
        self.dimensional_states = {
            dim: DimensionalState(
                dimension=dim,
                reading=None,
                performance_score=0.5,
                reliability_score=0.5,
                last_update=datetime.utcnow(),
                error_count=0,
                consecutive_errors=0,
                is_healthy=True
            )
            for dim in self.engines.keys()
        }
        
        # Synthesis tracking
        self.synthesis_history: List[SynthesisResult] = []
        self.performance_tracker = PerformanceTracker()
        self.consensus_tracker = EMA(20)
        self.confidence_tracker = EMA(15)
        
        # Configuration
        self.synthesis_mode = SynthesisMode.ADAPTIVE
        self.min_confidence_threshold = 0.3
        self.consensus_threshold = 0.6
        
        logger.info(f"Master Orchestrator initialized for {instrument_meta.symbol}")
    
    async def update(
        self,
        market_data: MarketData,
        order_book: Optional[OrderBookSnapshot] = None
    ) -> SynthesisResult:
        """
        Process market data through all dimensional engines and synthesize results.
        
        Args:
            market_data: Latest market data
            order_book: Optional order book snapshot
            
        Returns:
            Synthesis result with cross-dimensional intelligence
        """
        start_time = datetime.utcnow()
        
        try:
            # Update all dimensional engines in parallel
            dimensional_readings = await self._update_dimensional_engines(market_data, order_book)
            
            # Update dimensional states
            self._update_dimensional_states(dimensional_readings)
            
            # Assess engine health and degradation strategy
            degradation_strategy = self.degradation_manager.get_degradation_strategy(self.dimensional_states)
            
            # Check if we can operate
            if not degradation_strategy['can_operate']:
                return self._create_degraded_synthesis(market_data.timestamp, degradation_strategy)
            
            # Calculate contextual weights
            current_regime = self._determine_consensus_regime(dimensional_readings)
            contextual_weights = self.weight_manager.calculate_contextual_weights(
                self.dimensional_states, market_data, current_regime
            )
            
            # Perform cross-dimensional synthesis
            synthesis_result = await self._synthesize_dimensional_intelligence(
                dimensional_readings, contextual_weights, market_data, degradation_strategy
            )
            
            # Generate narrative
            synthesis_result.narrative = self.narrative_generator.generate_narrative(
                synthesis_result, dimensional_readings
            )
            
            # Update tracking
            self.synthesis_history.append(synthesis_result)
            if len(self.synthesis_history) > 100:
                self.synthesis_history.pop(0)
            
            self.consensus_tracker.update(synthesis_result.consensus_level)
            self.confidence_tracker.update(synthesis_result.confidence)
            
            # Calculate processing time
            synthesis_result.processing_time_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            logger.debug(f"Synthesis complete: signal={synthesis_result.signal_strength:.3f}, "
                        f"confidence={synthesis_result.confidence:.3f}, "
                        f"consensus={synthesis_result.consensus_level:.3f}")
            
            return synthesis_result
            
        except Exception as e:
            logger.error(f"Error in master orchestrator update: {e}")
            return self._create_error_synthesis(market_data.timestamp, str(e))
    
    async def _update_dimensional_engines(
        self,
        market_data: MarketData,
        order_book: Optional[OrderBookSnapshot]
    ) -> Dict[str, DimensionalReading]:
        """Update all dimensional engines in parallel."""
        tasks = []
        
        for dim, engine in self.engines.items():
            if dim == 'ANOMALY':
                # ANOMALY engine takes order book
                task = engine.update(market_data, order_book)
            else:
                task = engine.update(market_data)
            tasks.append((dim, task))
        
        # Execute all updates in parallel
        results = {}
        for dim, task in tasks:
            try:
                reading = await task
                results[dim] = reading
            except Exception as e:
                logger.error(f"Error updating {dim} engine: {e}")
                # Create error reading
                results[dim] = self._create_error_reading(dim, market_data.timestamp, str(e))
        
        return results
    
    def _update_dimensional_states(self, dimensional_readings: Dict[str, DimensionalReading]) -> None:
        """Update dimensional engine states."""
        for dim, reading in dimensional_readings.items():
            state = self.dimensional_states[dim]
            
            # Update reading
            state.reading = reading
            state.last_update = datetime.utcnow()
            
            # Update error tracking
            if 'error' in reading.context:
                state.error_count += 1
                state.consecutive_errors += 1
            else:
                state.consecutive_errors = 0
            
            # Update performance score (based on confidence and data quality)
            if reading.confidence > 0 and reading.data_quality > 0:
                performance = (reading.confidence + reading.data_quality) / 2
                state.performance_score = state.performance_score * 0.9 + performance * 0.1
            
            # Update reliability score (based on error rate)
            error_rate = state.error_count / max(1, state.error_count + 10)  # Assume 10 successful updates
            state.reliability_score = 1.0 - error_rate
            
            # Update health status
            state.is_healthy = self.degradation_manager.assess_engine_health(state)
    
    def _determine_consensus_regime(self, dimensional_readings: Dict[str, DimensionalReading]) -> MarketRegime:
        """Determine consensus market regime from dimensional readings."""
        regime_votes = {}
        
        for reading in dimensional_readings.values():
            if reading.regime in regime_votes:
                regime_votes[reading.regime] += reading.confidence
            else:
                regime_votes[reading.regime] = reading.confidence
        
        if regime_votes:
            return max(regime_votes.items(), key=lambda x: x[1])[0]
        else:
            return MarketRegime.CONSOLIDATING
    
    async def _synthesize_dimensional_intelligence(
        self,
        dimensional_readings: Dict[str, DimensionalReading],
        contextual_weights: Dict[str, float],
        market_data: MarketData,
        degradation_strategy: Dict[str, Any]
    ) -> SynthesisResult:
        """
        Synthesize cross-dimensional intelligence.
        
        Args:
            dimensional_readings: Individual dimensional readings
            contextual_weights: Contextual weights for each dimension
            market_data: Current market data
            degradation_strategy: Degradation strategy
            
        Returns:
            Synthesis result
        """
        # Calculate weighted signal strength
        weighted_signals = []
        dimensional_contributions = {}
        
        for dim, reading in dimensional_readings.items():
            if dim in contextual_weights and 'error' not in reading.context:
                weight = contextual_weights[dim]
                contribution = reading.signal_strength * weight * reading.confidence
                weighted_signals.append(contribution)
                dimensional_contributions[dim] = contribution
            else:
                dimensional_contributions[dim] = 0.0
        
        # Overall signal strength
        signal_strength = sum(weighted_signals) if weighted_signals else 0.0
        
        # Calculate consensus level
        consensus_level = self._calculate_consensus_level(dimensional_readings)
        
        # Calculate overall confidence
        confidence = self._calculate_synthesis_confidence(
            dimensional_readings, contextual_weights, consensus_level, degradation_strategy
        )
        
        # Determine synthesis mode
        synthesis_mode = self._determine_synthesis_mode(dimensional_readings, consensus_level)
        
        # Determine regime
        regime = self._determine_consensus_regime(dimensional_readings)
        
        # Extract evidence
        evidence = self._extract_synthesis_evidence(dimensional_readings, contextual_weights)
        
        # Generate warnings
        warnings = self._generate_synthesis_warnings(dimensional_readings, degradation_strategy)
        
        return SynthesisResult(
            timestamp=market_data.timestamp,
            signal_strength=np.clip(signal_strength, -1.0, 1.0),
            confidence=confidence,
            regime=regime,
            synthesis_mode=synthesis_mode,
            dimensional_weights=contextual_weights,
            dimensional_contributions=dimensional_contributions,
            consensus_level=consensus_level,
            narrative="",  # Will be filled by narrative generator
            evidence=evidence,
            warnings=warnings
        )
    
    def _calculate_consensus_level(self, dimensional_readings: Dict[str, DimensionalReading]) -> float:
        """Calculate consensus level across dimensions."""
        signals = []
        confidences = []
        
        for reading in dimensional_readings.values():
            if 'error' not in reading.context:
                signals.append(reading.signal_strength)
                confidences.append(reading.confidence)
        
        if len(signals) < 2:
            return 0.0
        
        # Calculate signal agreement
        signal_std = np.std(signals) if len(signals) > 1 else 0.0
        signal_agreement = max(0.0, 1.0 - float(signal_std))
        
        # Weight by average confidence
        avg_confidence = np.mean(confidences) if confidences else 0.0
        
        consensus = signal_agreement * avg_confidence
        return min(1.0, float(consensus))
    
    def _calculate_synthesis_confidence(
        self,
        dimensional_readings: Dict[str, DimensionalReading],
        contextual_weights: Dict[str, float],
        consensus_level: float,
        degradation_strategy: Dict[str, Any]
    ) -> float:
        """Calculate overall synthesis confidence."""
        # Base confidence from dimensional readings
        weighted_confidences = []
        
        for dim, reading in dimensional_readings.items():
            if dim in contextual_weights and 'error' not in reading.context:
                weight = contextual_weights[dim]
                weighted_confidence = reading.confidence * weight
                weighted_confidences.append(weighted_confidence)
        
        base_confidence = sum(weighted_confidences) if weighted_confidences else 0.0
        
        # Adjust for consensus
        consensus_bonus = consensus_level * 0.2
        
        # Adjust for degradation
        degradation_penalty = degradation_strategy.get('confidence_penalty', 0.0)
        
        # Adjust for historical performance
        historical_accuracy = self.performance_tracker.get_accuracy()
        
        final_confidence = (
            base_confidence * 0.6 +
            consensus_bonus +
            historical_accuracy * 0.2 -
            degradation_penalty
        )
        
        return max(0.0, min(1.0, final_confidence))
    
    def _determine_synthesis_mode(
        self,
        dimensional_readings: Dict[str, DimensionalReading],
        consensus_level: float
    ) -> SynthesisMode:
        """Determine appropriate synthesis mode."""
        if consensus_level > 0.8:
            return SynthesisMode.CONSENSUS
        elif consensus_level > 0.6:
            return SynthesisMode.WEIGHTED_AVERAGE
        elif consensus_level < 0.3:
            return SynthesisMode.ANTIFRAGILE  # Gain from disagreement
        else:
            return SynthesisMode.ADAPTIVE
    
    def _extract_synthesis_evidence(
        self,
        dimensional_readings: Dict[str, DimensionalReading],
        contextual_weights: Dict[str, float]
    ) -> Dict[str, Any]:
        """Extract evidence from synthesis."""
        evidence = {}
        
        # Dimensional evidence
        for dim, reading in dimensional_readings.items():
            if 'error' not in reading.context:
                evidence[f'{dim.lower()}_signal'] = reading.signal_strength
                evidence[f'{dim.lower()}_confidence'] = reading.confidence
                evidence[f'{dim.lower()}_weight'] = contextual_weights.get(dim, 0.0)
        
        # Synthesis evidence
        evidence['consensus_level'] = self._calculate_consensus_level(dimensional_readings)
        evidence['healthy_engines'] = sum(1 for state in self.dimensional_states.values() if state.is_healthy)
        evidence['total_engines'] = len(self.dimensional_states)
        
        return evidence
    
    def _generate_synthesis_warnings(
        self,
        dimensional_readings: Dict[str, DimensionalReading],
        degradation_strategy: Dict[str, Any]
    ) -> List[str]:
        """Generate synthesis warnings."""
        warnings = []
        
        # Degradation warnings
        if degradation_strategy['fallback_mode']:
            warnings.append("Operating in fallback mode due to engine failures")
        
        degraded_engines = degradation_strategy.get('degraded_engines', [])
        if degraded_engines:
            warnings.append(f"Degraded engines: {', '.join(degraded_engines)}")
        
        # Low consensus warning
        consensus = self._calculate_consensus_level(dimensional_readings)
        if consensus < 0.3:
            warnings.append(f"Low dimensional consensus: {consensus:.2f}")
        
        # Individual engine warnings
        for dim, reading in dimensional_readings.items():
            if reading.warnings:
                for warning in reading.warnings[:2]:  # Limit to 2 warnings per engine
                    warnings.append(f"{dim}: {warning}")
        
        return warnings
    
    def _create_degraded_synthesis(
        self,
        timestamp: datetime,
        degradation_strategy: Dict[str, Any]
    ) -> SynthesisResult:
        """Create synthesis result when system is degraded."""
        return SynthesisResult(
            timestamp=timestamp,
            signal_strength=0.0,
            confidence=0.0,
            regime=MarketRegime.CONSOLIDATING,
            synthesis_mode=SynthesisMode.CONSENSUS,
            dimensional_weights={},
            dimensional_contributions={},
            consensus_level=0.0,
            narrative="System degraded - insufficient healthy engines for analysis",
            evidence={'degraded': True},
            warnings=[f"System degraded: {degradation_strategy}"]
        )
    
    def _create_error_synthesis(self, timestamp: datetime, error_msg: str) -> SynthesisResult:
        """Create synthesis result when error occurs."""
        return SynthesisResult(
            timestamp=timestamp,
            signal_strength=0.0,
            confidence=0.0,
            regime=MarketRegime.CONSOLIDATING,
            synthesis_mode=SynthesisMode.CONSENSUS,
            dimensional_weights={},
            dimensional_contributions={},
            consensus_level=0.0,
            narrative=f"Synthesis error: {error_msg}",
            evidence={'error': error_msg},
            warnings=[f'Synthesis error: {error_msg}']
        )
    
    def _create_error_reading(self, dimension: str, timestamp: datetime, error_msg: str) -> DimensionalReading:
        """Create error reading for failed dimensional engine."""
        return DimensionalReading(
            dimension=dimension,
            timestamp=timestamp,
            signal_strength=0.0,
            confidence=0.0,
            regime=MarketRegime.CONSOLIDATING,
            context={'error': error_msg},
            data_quality=0.0,
            processing_time_ms=0.0,
            evidence={},
            warnings=[f'Engine error: {error_msg}']
        )
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health status."""
        healthy_count = sum(1 for state in self.dimensional_states.values() if state.is_healthy)
        total_count = len(self.dimensional_states)
        
        return {
            'overall_health': healthy_count / total_count,
            'healthy_engines': healthy_count,
            'total_engines': total_count,
            'degraded_engines': [dim for dim, state in self.dimensional_states.items() if not state.is_healthy],
            'average_performance': np.mean([state.performance_score for state in self.dimensional_states.values()]),
            'average_reliability': np.mean([state.reliability_score for state in self.dimensional_states.values()]),
            'consensus_trend': self.consensus_tracker.get_value() or 0.0,
            'confidence_trend': self.confidence_tracker.get_value() or 0.0
        }
    
    def reset(self) -> None:
        """Reset orchestrator state."""
        # Reset all engines
        for engine in self.engines.values():
            engine.reset()
        
        # Reset states
        for state in self.dimensional_states.values():
            state.reading = None
            state.performance_score = 0.5
            state.reliability_score = 0.5
            state.last_update = datetime.utcnow()
            state.error_count = 0
            state.consecutive_errors = 0
            state.is_healthy = True
        
        # Reset tracking
        self.synthesis_history.clear()
        self.consensus_tracker = EMA(20)
        self.confidence_tracker = EMA(15)
        
        logger.info("Master Orchestrator reset completed")

