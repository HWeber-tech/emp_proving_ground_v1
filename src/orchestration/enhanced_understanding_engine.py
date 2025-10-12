"""Contextual fusion engine for the understanding loop.

This module exposes the :class:`ContextualFusionEngine` – a lightweight façade
used throughout the test-suite when exercising the understanding pipeline. The
engine coordinates the enhanced sensory organs (WHY/HOW/WHAT/WHEN/ANOMALY) and
aggregates their outputs into a :class:`Synthesis` object. Housing the
implementation under the ``enhanced_understanding_engine`` namespace reflects
the roadmap's move away from the historical *intelligence* terminology.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from enum import Enum
from typing import Any, Awaitable, Dict, List, Protocol, Tuple, cast

from src.core.base import DimensionalReading, MarketData
from src.sensory.enhanced.anomaly_dimension import AnomalyUnderstandingEngine
from src.sensory.enhanced.how_dimension import InstitutionalUnderstandingEngine
from src.sensory.enhanced.what_dimension import TechnicalRealityEngine
from src.sensory.enhanced.when_dimension import ChronalUnderstandingEngine
from src.sensory.enhanced.why_dimension import (
    EnhancedFundamentalUnderstandingEngine,
)


class UnderstandingLevel(Enum):
    """Discrete confidence categories for the fused market understanding."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


# Backwards compatible alias – legacy callers historically imported
# ``IntelligenceLevel`` from this module.  Keeping the alias avoids a breaking
# change while encouraging new code to use the understanding nomenclature.
IntelligenceLevel = UnderstandingLevel


class Narrative(Enum):
    NEUTRAL = "NEUTRAL"
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    VOLATILE = "VOLATILE"


@dataclass
class Correlation:
    correlation: float
    significance: float


@dataclass
class Pattern:
    pattern_name: str
    involved_dimensions: List[str]
    pattern_strength: float
    confidence: float


@dataclass
class Synthesis:
    unified_score: float
    confidence: float
    narrative_text: str
    narrative_coherence: float
    understanding_level: UnderstandingLevel
    dominant_narrative: Narrative
    supporting_evidence: List[str]
    risk_factors: List[str]
    opportunity_factors: List[str]

    # Legacy attribute maintained as a property to avoid breaking
    # ``synthesis.intelligence_level`` access in downstream callers.  The
    # understanding loop terminology is the canonical surface moving forward.
    @property
    def intelligence_level(self) -> UnderstandingLevel:
        return self.understanding_level

    @intelligence_level.setter
    def intelligence_level(self, value: UnderstandingLevel) -> None:
        self.understanding_level = value


class WeightManager:
    def __init__(self) -> None:
        self._weights: Dict[str, float] = {
            "WHY": 0.20,
            "HOW": 0.20,
            "WHAT": 0.20,
            "WHEN": 0.20,
            "ANOMALY": 0.20,
        }

    def update(self, readings: Dict[str, DimensionalReading]) -> None:
        # Increase weight for dimensions with higher absolute signal and confidence
        scores: Dict[str, float] = {}
        for dim, r in readings.items():
            try:
                s = float(getattr(r, "signal_strength"))
                c = float(getattr(r, "confidence"))
            except Exception:
                s = 0.0
                c = 0.0
            scores[dim] = max(0.0, abs(s)) * (0.5 + 0.5 * c)

        # Blend with existing weights for stability
        blended: Dict[str, float] = {}
        for dim in self._weights.keys():
            blended[dim] = 0.7 * self._weights.get(dim, 0.2) + 0.3 * scores.get(dim, 0.0)

        # Normalize
        total = sum(blended.values()) or 1.0
        self._weights = {k: v / total for k, v in blended.items()}

    def calculate_current_weights(self) -> Dict[str, float]:
        return dict(self._weights)


class CorrelationAnalyzer:
    def __init__(self) -> None:
        self._history: Dict[str, List[float]] = {
            "WHY": [],
            "HOW": [],
            "WHAT": [],
            "WHEN": [],
            "ANOMALY": [],
        }
        self._max_len = 200

    def update(self, readings: Dict[str, DimensionalReading]) -> None:
        for dim, r in readings.items():
            try:
                val = float(getattr(r, "signal_strength"))
            except Exception:
                val = 0.0
            arr = self._history.setdefault(dim, [])
            arr.append(val)
            if len(arr) > self._max_len:
                del arr[0 : len(arr) - self._max_len]

    def _corr(self, a: List[float], b: List[float]) -> float:
        if len(a) < 2 or len(b) < 2:
            return 0.0
        n = min(len(a), len(b))
        ax = a[-n:]
        bx = b[-n:]
        mean_a = sum(ax) / n
        mean_b = sum(bx) / n
        cov = sum((x - mean_a) * (y - mean_b) for x, y in zip(ax, bx)) / n
        var_a = sum((x - mean_a) ** 2 for x in ax) / n
        var_b = sum((y - mean_b) ** 2 for y in bx) / n
        if var_a <= 0 or var_b <= 0:
            return 0.0
        corr = cov / ((var_a**0.5) * (var_b**0.5))
        # clamp to [-1, 1]
        return float(max(-1.0, min(1.0, corr)))

    def get_dimensional_correlations(self) -> Dict[Tuple[str, str], Correlation]:
        # Provide at least one correlation pair
        why = self._history.get("WHY", [])
        what = self._history.get("WHAT", [])
        corr_w = self._corr(why, what)
        return {("WHY", "WHAT"): Correlation(correlation=float(corr_w), significance=0.5)}

    def get_cross_dimensional_patterns(self) -> List[Pattern]:
        # Simple heuristic: if recent WHAT and WHEN align, emit a "TREND_ALIGNMENT" pattern
        what = self._history.get("WHAT", [-0.0])
        when = self._history.get("WHEN", [-0.0])
        strength = float(
            min(1.0, max(0.0, abs((what[-1] if what else 0.0) + (when[-1] if when else 0.0)) / 2.0))
        )
        return [
            Pattern(
                pattern_name="TREND_ALIGNMENT",
                involved_dimensions=["WHAT", "WHEN"],
                pattern_strength=strength,
                confidence=0.6 + 0.4 * strength,
            )
        ]


# Typing-only protocol describing the minimal API we call on enhanced engines
class EnhancedEngineProto(Protocol):
    def analyze_fundamental_understanding(
        self, market_data: MarketData
    ) -> Awaitable[DimensionalReading]: ...
    def analyze_fundamental_intelligence(
        self, market_data: MarketData
    ) -> Awaitable[DimensionalReading]: ...
    def analyze_institutional_understanding(
        self, market_data: MarketData
    ) -> Awaitable[DimensionalReading]: ...
    def analyze_institutional_intelligence(
        self, market_data: MarketData
    ) -> Awaitable[DimensionalReading]: ...
    def analyze_technical_reality(
        self, market_data: MarketData
    ) -> Awaitable[DimensionalReading]: ...
    def analyze_temporal_understanding(
        self, market_data: MarketData
    ) -> Awaitable[DimensionalReading]: ...
    def analyze_temporal_intelligence(
        self, market_data: MarketData
    ) -> Awaitable[DimensionalReading]: ...
    def analyze_anomaly_understanding(
        self, market_data: MarketData
    ) -> Awaitable[DimensionalReading]: ...
    def analyze_anomaly_intelligence(
        self, market_data: MarketData
    ) -> Awaitable[DimensionalReading]: ...


class ContextualFusionEngine:
    def __init__(self) -> None:
        # Engines
        self._why: EnhancedEngineProto = cast(
            EnhancedEngineProto, cast(Any, EnhancedFundamentalUnderstandingEngine)()
        )
        self._how: EnhancedEngineProto = cast(
            EnhancedEngineProto, cast(Any, InstitutionalUnderstandingEngine)()
        )
        self._what: EnhancedEngineProto = cast(
            EnhancedEngineProto, cast(Any, TechnicalRealityEngine)()
        )
        self._when: EnhancedEngineProto = cast(
            EnhancedEngineProto, cast(Any, ChronalUnderstandingEngine)()
        )
        self._anomaly: EnhancedEngineProto = cast(
            EnhancedEngineProto, cast(Any, AnomalyUnderstandingEngine)()
        )

        # Public state used in tests
        self.current_readings: Dict[str, DimensionalReading] = {}
        self.weight_manager = WeightManager()
        self.correlation_analyzer = CorrelationAnalyzer()

    async def analyze_market_understanding(self, market_data: MarketData) -> Synthesis:
        # Run all engines concurrently
        why_t = self._why.analyze_fundamental_understanding(market_data)
        how_t = self._how.analyze_institutional_understanding(market_data)
        what_t = self._what.analyze_technical_reality(market_data)
        when_t = self._when.analyze_temporal_understanding(market_data)
        anom_t = self._anomaly.analyze_anomaly_understanding(market_data)

        why, how, what, when, anomaly = await asyncio.gather(why_t, how_t, what_t, when_t, anom_t)

        readings: Dict[str, DimensionalReading] = {
            "WHY": why,
            "HOW": how,
            "WHAT": what,
            "WHEN": when,
            "ANOMALY": anomaly,
        }
        self.current_readings = readings

        # Update weights and correlations
        self.weight_manager.update(readings)
        self.correlation_analyzer.update(readings)

        weights = self.weight_manager.calculate_current_weights()

        # Compute unified score and confidence
        def _safe(f: Any, default: float = 0.0) -> float:
            try:
                return float(f)
            except Exception:
                return default

        unified = sum(
            weights[dim] * _safe(getattr(r, "signal_strength", 0.0)) for dim, r in readings.items()
        )
        conf = sum(_safe(getattr(r, "confidence", 0.0)) for r in readings.values()) / 5.0

        # Determine narratives
        dominant = Narrative.NEUTRAL
        if abs(unified) > 0.15:
            dominant = Narrative.BULLISH if unified > 0 else Narrative.BEARISH
        # Override to VOLATILE if anomaly is high
        if _safe(getattr(anomaly, "signal_strength", 0.0)) > 0.7:
            dominant = Narrative.VOLATILE

        understanding_level = (
            UnderstandingLevel.HIGH
            if conf > 0.7
            else UnderstandingLevel.MEDIUM
            if conf > 0.4
            else UnderstandingLevel.LOW
        )

        # Compose simple narrative text
        narrative_text = (
            f"Unified score {unified:+.3f} with confidence {conf:.2f}. "
            f"Dominant narrative: {dominant.name}."
        )

        # Supporting evidence, risk/opportunity factors (placeholders with signal-derived hints)
        supporting = [
            f"WHY:{_safe(getattr(why, 'signal_strength', 0.0)):+.2f}",
            f"HOW:{_safe(getattr(how, 'signal_strength', 0.0)):+.2f}",
            f"WHAT:{_safe(getattr(what, 'signal_strength', 0.0)):+.2f}",
            f"WHEN:{_safe(getattr(when, 'signal_strength', 0.0)):+.2f}",
            f"ANOMALY:{_safe(getattr(anomaly, 'signal_strength', 0.0)):+.2f}",
        ]
        risk_factors = (
            ["elevated volatility"] if dominant is Narrative.VOLATILE else ["standard risk"]
        )
        opportunity_factors = ["trend continuation"] if abs(unified) > 0.2 else ["range strategies"]

        # Coherence: simple function of agreement among dims (1 - variance proxy)
        sigs = [abs(_safe(getattr(r, "signal_strength", 0.0))) for r in readings.values()]
        variance_proxy = (max(sigs) - min(sigs)) if sigs else 0.0
        narrative_coherence = max(0.0, min(1.0, 1.0 - variance_proxy))

        return Synthesis(
            unified_score=float(unified),
            confidence=float(conf),
            narrative_text=narrative_text,
            narrative_coherence=float(narrative_coherence),
            understanding_level=understanding_level,
            dominant_narrative=dominant,
            supporting_evidence=supporting,
            risk_factors=risk_factors,
            opportunity_factors=opportunity_factors,
        )

    def get_diagnostic_information(self) -> Dict[str, Any]:
        return {
            "current_readings": self.current_readings,
            "adaptive_weights": self.weight_manager.calculate_current_weights(),
            "correlations": self.correlation_analyzer.get_dimensional_correlations(),
            "patterns": self.correlation_analyzer.get_cross_dimensional_patterns(),
        }
