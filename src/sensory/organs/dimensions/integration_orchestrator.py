"""
Sensory Integration Orchestrator
==============================

Unified orchestrator for the 5D+1 sensory cortex integration.
Coordinates all sensory dimensions and provides unified market intelligence.

Author: EMP Development Team
Phase: 2 - Truth-First Completion
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Any, Mapping, Protocol, TypedDict, cast, runtime_checkable

import numpy as np
import pandas as pd

from src.core.types import JSONObject
from src.sensory.enhanced.anomaly.manipulation_detection import (
    ManipulationDetectionSystem,
)

# Import all sensory dimensions
from src.sensory.organs.dimensions.chaos_adaptation import ChaosAdaptationSystem
from src.sensory.organs.dimensions.institutional_tracker import (
    InstitutionalFootprint,
    InstitutionalFootprintHunter,
)
from src.sensory.organs.dimensions.macro_intelligence import (
    MacroEnvironmentState,
    MacroPredatorIntelligence,
)
from src.sensory.organs.dimensions.pattern_engine import (
    PatternSynthesis,
    PatternSynthesisEngine,
)
from src.sensory.organs.dimensions.temporal_system import (
    TemporalAdvantage,
    TemporalAdvantageSystem,
)

# Typed payloads and minimal Protocols for orchestrator I/O


class OrchestratorMarketData(TypedDict, total=False):
    price_data: pd.DataFrame
    meta: JSONObject


@runtime_checkable
class AnomalyDetectionLike(Protocol):
    confidence: float
    overall_risk_score: float


@runtime_checkable
class ChaosAdaptationLike(Protocol):
    confidence: float
    black_swan_probability: float


__all__ = ["UnifiedMarketIntelligence", "SensoryIntegrationOrchestrator", "OrchestratorMarketData"]

logger = logging.getLogger(__name__)


class UnifiedMarketIntelligence:
    """Unified market intelligence from all sensory dimensions"""

    def __init__(self, symbol: str | None = None) -> None:
        self.symbol: str | None = symbol
        self.macro_environment: MacroEnvironmentState | None = None
        self.institutional_footprint: InstitutionalFootprint | None = None
        self.pattern_synthesis: PatternSynthesis | None = None
        self.temporal_advantage: TemporalAdvantage | None = None
        self.anomaly_detection: AnomalyDetectionLike | None = None
        self.chaos_adaptation: ChaosAdaptationLike | None = None
        self.overall_confidence: float = 0.0  # Changed to match test
        self.signal_strength: float = 0.0
        self.risk_assessment: float = 0.0  # Changed to match test
        self.opportunity_score: float = 0.0  # Added to match test
        self.confluence_score: float = 0.0  # Added to match test
        self.recommended_action: str = "hold"
        self.timestamp: datetime = datetime.now()


class SensoryIntegrationOrchestrator:
    """
    Master orchestrator for the 5D+1 sensory cortex.
    """

    def __init__(self) -> None:
        # Initialize all sensory dimensions
        self.macro_intelligence: MacroPredatorIntelligence = MacroPredatorIntelligence()
        self.footprint_hunter: InstitutionalFootprintHunter = InstitutionalFootprintHunter()
        self.pattern_engine: PatternSynthesisEngine = PatternSynthesisEngine()
        self.temporal_analyzer: TemporalAdvantageSystem = TemporalAdvantageSystem()
        self.anomaly_detector: ManipulationDetectionSystem = ManipulationDetectionSystem()
        self.chaos_adapter: ChaosAdaptationSystem = ChaosAdaptationSystem()

        # Initialize adaptive weights
        self.adaptive_weights: dict[str, float] = {
            "macro": 0.20,
            "footprint": 0.25,
            "patterns": 0.20,
            "timing": 0.15,
            "anomalies": 0.10,
            "chaos": 0.10,
        }

        logger.info("Sensory Integration Orchestrator initialized")

    async def analyze_unified_intelligence(
        self, market_data: OrchestratorMarketData | Mapping[str, object], symbol: str | None = None
    ) -> UnifiedMarketIntelligence:
        """
        Analyze unified intelligence across all dimensions.

        Args:
            market_data: Market data mapping; may include 'price_data' as a pandas.DataFrame and optional JSON-like metadata.
            symbol: Trading symbol (optional)

        Returns:
            UnifiedMarketIntelligence: Complete market analysis
        """
        return await self.process_market_intelligence(market_data, symbol)

    async def process_market_intelligence(
        self, market_data: OrchestratorMarketData | Mapping[str, object], symbol: str | None = None
    ) -> UnifiedMarketIntelligence:
        """Process all sensory dimensions and create unified market intelligence."""
        try:
            logger.info("Starting unified market intelligence processing")

            # Extract data components
            price_data = cast(pd.DataFrame, market_data.get("price_data", pd.DataFrame()))

            # Process each dimension in parallel
            tasks = [
                self.macro_intelligence.analyze_macro_environment(),
                cast(Any, self.footprint_hunter).analyze_institutional_footprint(price_data),
                self.pattern_engine.synthesize_patterns(price_data),
                cast(Any, self.temporal_analyzer).analyze_timing(cast(dict[str, Any], dict(market_data))),
                self.anomaly_detector.detect_manipulation(market_data),
                self.chaos_adapter.assess_chaos_opportunities(price_data),
            ]

            # Execute all tasks concurrently
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Create unified intelligence
            intelligence = UnifiedMarketIntelligence(symbol=symbol)

            # Assign results
            res0 = results[0]
            if isinstance(res0, Exception):
                intelligence.macro_environment = self._get_fallback_macro()
            else:
                intelligence.macro_environment = cast(MacroEnvironmentState, res0)

            res1 = results[1]
            if isinstance(res1, Exception):
                intelligence.institutional_footprint = self._get_fallback_footprint()
            else:
                intelligence.institutional_footprint = cast(InstitutionalFootprint, res1)

            res2 = results[2]
            if isinstance(res2, Exception):
                intelligence.pattern_synthesis = self._get_fallback_patterns()
            else:
                intelligence.pattern_synthesis = cast(PatternSynthesis, res2)

            res3 = results[3]
            if isinstance(res3, Exception):
                intelligence.temporal_advantage = self._get_fallback_timing()
            else:
                intelligence.temporal_advantage = cast(TemporalAdvantage, res3)

            res4 = results[4]
            if isinstance(res4, Exception):
                intelligence.anomaly_detection = self._get_fallback_anomalies()
            else:
                intelligence.anomaly_detection = cast(AnomalyDetectionLike, res4)

            res5 = results[5]
            if isinstance(res5, Exception):
                intelligence.chaos_adaptation = self._get_fallback_chaos()
            else:
                intelligence.chaos_adaptation = cast(ChaosAdaptationLike, res5)

            # Calculate unified confidence
            intelligence.overall_confidence = self._calculate_overall_confidence(intelligence)
            intelligence.signal_strength = intelligence.overall_confidence

            # Calculate additional scores
            intelligence.opportunity_score = self._calculate_opportunity_score(intelligence)
            intelligence.confluence_score = self._calculate_confluence_score(intelligence)
            intelligence.risk_assessment = self._calculate_risk_assessment(intelligence)

            # Generate recommended action
            intelligence.recommended_action = self._generate_recommended_action(intelligence)

            logger.info("Unified market intelligence processing completed")
            return intelligence

        except Exception as e:
            logger.error(f"Unified market intelligence processing failed: {e}")
            return self._get_fallback_intelligence(symbol)

    def _calculate_overall_confidence(self, intelligence: UnifiedMarketIntelligence) -> float:
        """Calculate overall confidence score"""
        try:
            confidences: list[float] = []

            # Safely extract confidence scores
            if intelligence.macro_environment and hasattr(
                intelligence.macro_environment, "confidence_score"
            ):
                confidences.append(intelligence.macro_environment.confidence_score)
            else:
                confidences.append(0.1)

            if intelligence.institutional_footprint and hasattr(
                intelligence.institutional_footprint, "confidence_score"
            ):
                confidences.append(intelligence.institutional_footprint.confidence_score)
            else:
                confidences.append(0.1)

            if intelligence.pattern_synthesis and hasattr(
                intelligence.pattern_synthesis, "confidence_score"
            ):
                confidences.append(intelligence.pattern_synthesis.confidence_score)
            else:
                confidences.append(0.1)

            if intelligence.temporal_advantage and hasattr(
                intelligence.temporal_advantage, "confidence_score"
            ):
                confidences.append(intelligence.temporal_advantage.confidence_score)
            else:
                confidences.append(0.1)

            if intelligence.anomaly_detection and hasattr(
                intelligence.anomaly_detection, "confidence"
            ):
                confidences.append(intelligence.anomaly_detection.confidence)
            else:
                confidences.append(0.1)

            if intelligence.chaos_adaptation and hasattr(
                intelligence.chaos_adaptation, "confidence"
            ):
                confidences.append(intelligence.chaos_adaptation.confidence)
            else:
                confidences.append(0.1)

            weights: list[float] = list(self.adaptive_weights.values())
            weighted_confidence = float(np.average(confidences, weights=weights))

            return float(min(max(weighted_confidence, 0.0), 1.0))

        except Exception as e:
            logger.error(f"Failed to calculate overall confidence: {e}")
            return 0.5

    def _calculate_opportunity_score(self, intelligence: UnifiedMarketIntelligence) -> float:
        """Calculate opportunity score based on all dimensions"""
        try:
            scores: list[float] = []

            # Macro opportunity
            if intelligence.macro_environment and hasattr(
                intelligence.macro_environment, "central_bank_sentiment"
            ):
                scores.append(abs(intelligence.macro_environment.central_bank_sentiment))
            else:
                scores.append(0.0)

            # Institutional opportunity
            if intelligence.institutional_footprint and hasattr(
                intelligence.institutional_footprint, "smart_money_flow"
            ):
                scores.append(abs(intelligence.institutional_footprint.smart_money_flow))
            else:
                scores.append(0.0)

            # Pattern opportunity
            if intelligence.pattern_synthesis and hasattr(
                intelligence.pattern_synthesis, "pattern_strength"
            ):
                scores.append(intelligence.pattern_synthesis.pattern_strength)
            else:
                scores.append(0.0)

            # Timing opportunity
            if intelligence.temporal_advantage and hasattr(
                intelligence.temporal_advantage, "confidence_score"
            ):
                scores.append(intelligence.temporal_advantage.confidence_score)
            else:
                scores.append(0.0)

            return float(np.mean(scores))

        except Exception as e:
            logger.error(f"Failed to calculate opportunity score: {e}")
            return 0.0

    def _calculate_confluence_score(self, intelligence: UnifiedMarketIntelligence) -> float:
        """Calculate confluence score based on agreement between dimensions"""
        try:
            signals: list[float] = []

            # Collect directional signals
            if intelligence.macro_environment and hasattr(
                intelligence.macro_environment, "central_bank_sentiment"
            ):
                signals.append(
                    float(np.sign(intelligence.macro_environment.central_bank_sentiment))
                )

            if intelligence.institutional_footprint and hasattr(
                intelligence.institutional_footprint, "institutional_bias"
            ):
                bias = intelligence.institutional_footprint.institutional_bias
                signals.append(1.0 if bias == "bullish" else -1.0 if bias == "bearish" else 0.0)

            if intelligence.pattern_synthesis and hasattr(
                intelligence.pattern_synthesis, "pattern_strength"
            ):
                strength = intelligence.pattern_synthesis.pattern_strength
                # Assume positive for bullish patterns
                signals.append(strength if strength > 0.5 else -strength)

            if len(signals) > 1:
                # Calculate agreement
                agreement = float(np.std(signals))
                return max(0.0, 1.0 - agreement)  # Lower std = higher agreement
            else:
                return 0.0

        except Exception as e:
            logger.error(f"Failed to calculate confluence score: {e}")
            return 0.0

    def _calculate_risk_assessment(self, intelligence: UnifiedMarketIntelligence) -> float:
        """Calculate overall risk assessment score"""
        try:
            risk_factors: list[float] = []

            # Macro risk
            if intelligence.macro_environment and hasattr(
                intelligence.macro_environment, "geopolitical_risk"
            ):
                risk_factors.append(intelligence.macro_environment.geopolitical_risk)
            else:
                risk_factors.append(0.5)

            # Anomaly risk
            if intelligence.anomaly_detection and hasattr(
                intelligence.anomaly_detection, "overall_risk_score"
            ):
                risk_factors.append(intelligence.anomaly_detection.overall_risk_score)
            else:
                risk_factors.append(0.5)

            # Chaos risk
            if intelligence.chaos_adaptation and hasattr(
                intelligence.chaos_adaptation, "black_swan_probability"
            ):
                risk_factors.append(intelligence.chaos_adaptation.black_swan_probability)
            else:
                risk_factors.append(0.5)

            return float(np.mean(risk_factors))

        except Exception as e:
            logger.error(f"Failed to calculate risk assessment: {e}")
            return 0.5

    def _generate_recommended_action(self, intelligence: UnifiedMarketIntelligence) -> str:
        """Generate recommended action based on all dimensions"""
        try:
            # Simple decision logic based on scores
            if intelligence.overall_confidence > 0.7 and intelligence.opportunity_score > 0.6:
                return "strong_buy"
            elif intelligence.overall_confidence > 0.5 and intelligence.opportunity_score > 0.4:
                return "buy"
            elif intelligence.overall_confidence < 0.3 or intelligence.risk_assessment > 0.7:
                return "sell"
            else:
                return "hold"

        except Exception as e:
            logger.error(f"Failed to generate recommended action: {e}")
            return "hold"

    def _get_fallback_macro(self) -> MacroEnvironmentState:
        """Return fallback macro environment"""
        return MacroEnvironmentState(
            central_bank_sentiment=0.0,
            geopolitical_risk=0.5,
            economic_momentum=0.0,
            policy_outlook=0.0,
            confidence_score=0.1,
            timestamp=datetime.now(),
            key_events=[],
            risk_factors=[],
        )

    def _get_fallback_footprint(self) -> InstitutionalFootprint:
        """Return fallback institutional footprint"""
        return InstitutionalFootprint(
            order_blocks=[],
            fair_value_gaps=[],
            liquidity_sweeps=[],
            smart_money_flow=0.0,
            institutional_bias="neutral",
            confidence_score=0.1,
            market_structure="unknown",
            key_levels=[],
        )

    def _get_fallback_patterns(self) -> PatternSynthesis:
        """Return fallback pattern synthesis"""
        return PatternSynthesis([], [], {}, {}, 0.0, 0.1)

    def _get_fallback_timing(self) -> TemporalAdvantage:
        """Return fallback temporal advantage"""
        return TemporalAdvantage(0.0, {}, {}, "", (datetime.now(), datetime.now()), 0.1)

    def _get_fallback_anomalies(self) -> AnomalyDetectionLike:
        """Return fallback anomaly detection (minimal Protocol-compatible object)."""

        class _AnomalyFallback:
            def __init__(self, overall_risk_score: float, confidence: float) -> None:
                self.overall_risk_score = overall_risk_score
                self.confidence = confidence

        return _AnomalyFallback(0.0, 0.1)

    def _get_fallback_chaos(self) -> ChaosAdaptationLike:
        """Return fallback chaos adaptation (minimal Protocol-compatible object)."""

        class _ChaosFallback:
            def __init__(self, black_swan_probability: float, confidence: float) -> None:
                self.black_swan_probability = black_swan_probability
                self.confidence = confidence

        return _ChaosFallback(0.0, 0.1)

    def _get_fallback_intelligence(self, symbol: str | None = None) -> UnifiedMarketIntelligence:
        """Return fallback unified intelligence"""
        intelligence = UnifiedMarketIntelligence(symbol=symbol)
        intelligence.macro_environment = self._get_fallback_macro()
        intelligence.institutional_footprint = self._get_fallback_footprint()
        intelligence.pattern_synthesis = self._get_fallback_patterns()
        intelligence.temporal_advantage = self._get_fallback_timing()
        intelligence.anomaly_detection = self._get_fallback_anomalies()
        intelligence.chaos_adaptation = self._get_fallback_chaos()
        intelligence.overall_confidence = 0.1
        intelligence.signal_strength = 0.1
        intelligence.risk_assessment = 0.5
        intelligence.opportunity_score = 0.0
        intelligence.confluence_score = 0.0
        intelligence.recommended_action = "hold"
        return intelligence
