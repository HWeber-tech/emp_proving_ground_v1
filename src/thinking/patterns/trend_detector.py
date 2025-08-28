"""
EMP Trend Detector v1.1

Analyzes sensory signals to detect and classify market trends.
Provides trend direction, strength, and confidence metrics.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Protocol, TypeAlias, cast

import numpy as np

if TYPE_CHECKING:
    from src.core.interfaces import AnalysisResult as AnalysisResult
    from src.core.interfaces import SensorySignal as SensorySignal
    from src.core.interfaces import ThinkingPattern as ThinkingPattern
else:
    ThinkingPattern: TypeAlias = Any
    SensorySignal: TypeAlias = Any
    AnalysisResult: TypeAlias = Dict[str, object]
from src.core.exceptions import TradingException

logger = logging.getLogger(__name__)


@dataclass
class TrendAnalysis:
    """Trend analysis result."""

    direction: str  # 'BULLISH', 'BEARISH', 'NEUTRAL'
    strength: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0
    duration: int  # Trend duration in periods
    metadata: dict[str, object]


class _SignalProto(Protocol):
    signal_type: str
    value: float
    confidence: float


class TrendDetector(ThinkingPattern):
    """Detects and analyzes market trends from sensory signals."""

    def __init__(self, config: Optional[dict[str, object]] = None):
        self.config = config or {}
        self.min_confidence = float(cast(Any, self.config.get("min_confidence", 0.3)))
        self.min_strength = float(cast(Any, self.config.get("min_strength", 0.2)))
        self.lookback_periods = int(cast(Any, self.config.get("lookback_periods", 20)))
        self._signal_history: List[SensorySignal] = []

    def analyze(self, signals: List[SensorySignal]) -> AnalysisResult:
        """Analyze sensory signals to detect trends."""
        try:
            signals_t = cast("List[_SignalProto]", signals)
            # Update signal history
            self._update_signal_history(signals)  # history keeps original objects

            # Extract trend-relevant signals
            trend_signals = self._extract_trend_signals(cast(List[SensorySignal], signals_t))

            if not trend_signals:
                return self._create_neutral_result()

            # Analyze trend components
            price_trend = self._analyze_price_trend(trend_signals)
            momentum_trend = self._analyze_momentum_trend(trend_signals)
            volume_trend = self._analyze_volume_trend(trend_signals)

            # Combine trend analysis
            composite_trend = self._combine_trend_analysis(
                price_trend, momentum_trend, volume_trend
            )

            # Create analysis result
            return cast(
                AnalysisResult,
                {
                    "timestamp": datetime.now(),
                    "analysis_type": "trend_detection",
                    "result": {
                        "trend_direction": composite_trend.direction,
                        "trend_strength": composite_trend.strength,
                        "trend_confidence": composite_trend.confidence,
                        "trend_duration": composite_trend.duration,
                        "price_trend": price_trend.__dict__,
                        "momentum_trend": momentum_trend.__dict__,
                        "volume_trend": volume_trend.__dict__,
                        "metadata": composite_trend.metadata,
                    },
                    "confidence": composite_trend.confidence,
                    "metadata": {
                        "signal_count": len(trend_signals),
                        "analysis_method": "composite_trend",
                    },
                },
            )

        except Exception as e:
            raise TradingException(f"Error in trend detection: {e}")

    def learn(self, feedback: Mapping[str, object]) -> bool:
        """Learn from feedback to improve trend detection."""
        try:
            # Extract learning data from feedback
            if "trend_accuracy" in feedback:
                accuracy = float(cast(Any, feedback["trend_accuracy"]))
                # Adjust confidence thresholds based on accuracy
                if accuracy < 0.5:
                    self.min_confidence *= 0.9  # Lower threshold
                elif accuracy > 0.8:
                    self.min_confidence *= 1.1  # Raise threshold

            if "trend_strength_error" in feedback:
                strength_error = float(cast(Any, feedback["trend_strength_error"]))
                # Adjust strength thresholds
                if strength_error > 0.3:
                    self.min_strength *= 0.9
                elif strength_error < 0.1:
                    self.min_strength *= 1.1

            logger.info("Trend detector learned from feedback")
            return True

        except Exception as e:
            logger.error(f"Error in trend detector learning: {e}")
            return False

    def _update_signal_history(self, signals: List[SensorySignal]) -> None:
        """Update the signal history."""
        self._signal_history.extend(signals)

        # Keep only recent signals
        if len(self._signal_history) > self.lookback_periods * 10:
            self._signal_history = self._signal_history[-self.lookback_periods * 10 :]

    def _extract_trend_signals(self, signals: List[SensorySignal]) -> List[SensorySignal]:
        """Extract signals relevant to trend detection."""
        trend_signals: list[SensorySignal] = []

        for signal in signals:
            st = getattr(signal, "signal_type", None)
            if isinstance(st, str) and st in ["price_composite", "volume_composite", "momentum"]:
                trend_signals.append(signal)

        return trend_signals

    def _analyze_price_trend(self, signals: List[SensorySignal]) -> TrendAnalysis:
        """Analyze price trend from signals."""
        sigs = cast(List[Any], signals)
        price_signals = [s for s in sigs if s.signal_type == "price_composite"]

        if not price_signals:
            return TrendAnalysis(
                direction="NEUTRAL", strength=0.0, confidence=0.0, duration=0, metadata={}
            )

        # Calculate trend direction
        recent_signals = price_signals[-5:]  # Last 5 signals
        avg_value = float(np.mean([float(s.value) for s in recent_signals]))

        if avg_value > 0.1:
            direction = "BULLISH"
        elif avg_value < -0.1:
            direction = "BEARISH"
        else:
            direction = "NEUTRAL"

        # Calculate trend strength
        strength = min(float(abs(avg_value)), 1.0)

        # Calculate confidence
        confidences = [float(s.confidence) for s in recent_signals]
        confidence = float(np.mean(confidences)) if confidences else 0.0

        # Calculate duration
        duration = self._calculate_trend_duration(price_signals, direction)

        return TrendAnalysis(
            direction=direction,
            strength=strength,
            confidence=confidence,
            duration=duration,
            metadata={"avg_value": avg_value, "signal_count": len(price_signals)},
        )

    def _analyze_momentum_trend(self, signals: List[SensorySignal]) -> TrendAnalysis:
        """Analyze momentum trend from signals."""
        sigs = cast(List[Any], signals)
        momentum_signals = [s for s in sigs if s.signal_type == "momentum"]

        if not momentum_signals:
            return TrendAnalysis(
                direction="NEUTRAL", strength=0.0, confidence=0.0, duration=0, metadata={}
            )

        # Calculate momentum direction
        recent_signals = momentum_signals[-3:]  # Last 3 signals
        avg_momentum = float(np.mean([float(s.value) for s in recent_signals]))

        if avg_momentum > 0.2:
            direction = "BULLISH"
        elif avg_momentum < -0.2:
            direction = "BEARISH"
        else:
            direction = "NEUTRAL"

        # Calculate momentum strength
        strength = min(float(abs(avg_momentum)), 1.0)

        # Calculate confidence
        confidences = [float(s.confidence) for s in recent_signals]
        confidence = float(np.mean(confidences)) if confidences else 0.0

        return TrendAnalysis(
            direction=direction,
            strength=strength,
            confidence=confidence,
            duration=len(momentum_signals),
            metadata={"avg_momentum": avg_momentum, "signal_count": len(momentum_signals)},
        )

    def _analyze_volume_trend(self, signals: List[SensorySignal]) -> TrendAnalysis:
        """Analyze volume trend from signals."""
        sigs = cast(List[Any], signals)
        volume_signals = [s for s in sigs if s.signal_type == "volume_composite"]

        if not volume_signals:
            return TrendAnalysis(
                direction="NEUTRAL", strength=0.0, confidence=0.0, duration=0, metadata={}
            )

        # Calculate volume trend
        recent_signals = volume_signals[-3:]
        avg_volume = float(np.mean([float(s.value) for s in recent_signals]))

        if avg_volume > 0.1:
            direction = "BULLISH"  # High volume supports trend
        elif avg_volume < -0.1:
            direction = "BEARISH"  # Low volume suggests weakness
        else:
            direction = "NEUTRAL"

        # Calculate volume strength
        strength = min(float(abs(avg_volume)), 1.0)

        # Calculate confidence
        confidences = [float(s.confidence) for s in recent_signals]
        confidence = float(np.mean(confidences)) if confidences else 0.0

        return TrendAnalysis(
            direction=direction,
            strength=strength,
            confidence=confidence,
            duration=len(volume_signals),
            metadata={"avg_volume": avg_volume, "signal_count": len(volume_signals)},
        )

    def _combine_trend_analysis(
        self, price_trend: TrendAnalysis, momentum_trend: TrendAnalysis, volume_trend: TrendAnalysis
    ) -> TrendAnalysis:
        """Combine multiple trend analyses into a composite trend."""
        # Weight the different trend components
        price_weight = 0.5
        momentum_weight = 0.3
        volume_weight = 0.2

        # Calculate weighted direction scores
        direction_scores = {"BULLISH": 0.0, "BEARISH": 0.0, "NEUTRAL": 0.0}

        # Price trend contribution
        if price_trend.direction == "BULLISH":
            direction_scores["BULLISH"] += price_trend.strength * price_weight
        elif price_trend.direction == "BEARISH":
            direction_scores["BEARISH"] += price_trend.strength * price_weight
        else:
            direction_scores["NEUTRAL"] += price_trend.strength * price_weight

        # Momentum trend contribution
        if momentum_trend.direction == "BULLISH":
            direction_scores["BULLISH"] += momentum_trend.strength * momentum_weight
        elif momentum_trend.direction == "BEARISH":
            direction_scores["BEARISH"] += momentum_trend.strength * momentum_weight
        else:
            direction_scores["NEUTRAL"] += momentum_trend.strength * momentum_weight

        # Volume trend contribution
        if volume_trend.direction == "BULLISH":
            direction_scores["BULLISH"] += volume_trend.strength * volume_weight
        elif volume_trend.direction == "BEARISH":
            direction_scores["BEARISH"] += volume_trend.strength * volume_weight
        else:
            direction_scores["NEUTRAL"] += volume_trend.strength * volume_weight

        # Determine composite direction
        max_score = max(direction_scores.values())
        composite_direction = max(direction_scores, key=lambda k: direction_scores[k])

        # Calculate composite strength and confidence
        composite_strength = max_score
        composite_confidence = float(
            np.mean([price_trend.confidence, momentum_trend.confidence, volume_trend.confidence])
        )

        # Calculate composite duration
        composite_duration = max(
            price_trend.duration, momentum_trend.duration, volume_trend.duration
        )

        return TrendAnalysis(
            direction=composite_direction,
            strength=composite_strength,
            confidence=composite_confidence,
            duration=composite_duration,
            metadata={
                "price_trend": price_trend.__dict__,
                "momentum_trend": momentum_trend.__dict__,
                "volume_trend": volume_trend.__dict__,
                "direction_scores": direction_scores,
            },
        )

    def _calculate_trend_duration(self, signals: List[SensorySignal], direction: str) -> int:
        """Calculate how long the current trend has been active."""
        duration = 0

        for signal in reversed(signals):
            val = float(cast(Any, signal).value)
            if direction == "BULLISH" and val > 0.0:
                duration += 1
            elif direction == "BEARISH" and val < 0.0:
                duration += 1
            elif direction == "NEUTRAL" and abs(val) < 0.1:
                duration += 1
            else:
                break

        return duration

    def _create_neutral_result(self) -> AnalysisResult:
        """Create a neutral analysis result when no signals are available."""
        return cast(
            AnalysisResult,
            {
                "timestamp": datetime.now(),
                "analysis_type": "trend_detection",
                "result": {
                    "trend_direction": "NEUTRAL",
                    "trend_strength": 0.0,
                    "trend_confidence": 0.0,
                    "trend_duration": 0,
                    "metadata": {},
                },
                "confidence": 0.0,
                "metadata": {"signal_count": 0, "analysis_method": "neutral_fallback"},
            },
        )
