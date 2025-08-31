"""
EMP Cycle Detector v1.1

Detects market cycles and cyclical patterns in sensory signals.
Provides cycle identification and phase analysis capabilities.
"""

import logging
from dataclasses import dataclass
from datetime import datetime

# Add Mapping and cast for typing-safe minimal fixes
from typing import Any, Dict, List, Mapping, Optional, cast

import numpy as np

from src.core.interfaces import AnalysisResult, SensorySignal, ThinkingPattern
# Guard the exception import: alias TradingException if ThinkingException absent
try:
    from src.core.exceptions import TradingException as ThinkingException
except Exception:  # pragma: no cover
    from src.core.exceptions import TradingException as ThinkingException

logger = logging.getLogger(__name__)


@dataclass
class CycleAnalysis:
    """Cycle analysis result."""

    cycle_type: str  # 'bull_market', 'bear_market', 'consolidation', 'unknown'
    phase: str  # 'early', 'middle', 'late', 'transition'
    cycle_length: int  # Estimated cycle length in periods
    confidence: float  # 0.0 to 1.0
    metadata: dict[str, object]


class CycleDetector(ThinkingPattern):
    """Detects market cycles and cyclical patterns."""

    def __init__(self, config: Optional[dict[str, object]] = None):
        self.config = config or {}
        _v = self.config.get("min_cycle_length", 10)
        self.min_cycle_length = int(_v) if isinstance(_v, (int, float, str)) else 10
        _v = self.config.get("max_cycle_length", 200)
        self.max_cycle_length = int(_v) if isinstance(_v, (int, float, str)) else 200
        _v = self.config.get("lookback_periods", 100)
        self.lookback_periods = int(_v) if isinstance(_v, (int, float, str)) else 100
        self._signal_history: List[SensorySignal] = []

    def analyze(self, signals: List[SensorySignal]) -> AnalysisResult:
        """Analyze sensory signals to detect cycles."""
        try:
            # Update signal history
            self._update_signal_history(signals)

            # Extract cycle-relevant signals
            cycle_signals = self._extract_cycle_signals(signals)

            if not cycle_signals:
                return self._create_neutral_result()

            # Detect cycles
            cycle_analysis = self._detect_cycles(cycle_signals)

            # Create analysis result
            return cast(
                AnalysisResult,
                {
                    "timestamp": datetime.now(),
                    "analysis_type": "cycle_detection",
                    "result": {
                        "cycle_type": cycle_analysis.cycle_type,
                        "phase": cycle_analysis.phase,
                        "cycle_length": cycle_analysis.cycle_length,
                        "confidence": cycle_analysis.confidence,
                        "metadata": cycle_analysis.metadata,
                    },
                    "confidence": cycle_analysis.confidence,
                    "metadata": {
                        "signal_count": len(cycle_signals),
                        "analysis_method": "fourier_cycle_detection",
                    },
                },
            )

        except Exception as e:
            raise ThinkingException(f"Error in cycle detection: {e}")

    def learn(self, feedback: Mapping[str, object]) -> bool:
        """Learn from feedback to improve cycle detection."""
        try:
            # Extract learning data from feedback
            acc = feedback.get("cycle_accuracy")
            if isinstance(acc, (int, float)):
                accuracy = float(acc)
                # Adjust parameters based on accuracy
                if accuracy < 0.5:
                    self.min_cycle_length = max(5, self.min_cycle_length - 2)
                elif accuracy > 0.8:
                    self.min_cycle_length = min(50, self.min_cycle_length + 2)
            logger.info("Cycle detector learned from feedback")
            return True

        except Exception as e:
            logger.error(f"Error in cycle detector learning: {e}")
            return False

    def _update_signal_history(self, signals: List[SensorySignal]) -> None:
        """Update the signal history."""
        self._signal_history.extend(signals)

        # Keep only recent signals
        if len(self._signal_history) > self.lookback_periods * 10:
            self._signal_history = self._signal_history[-self.lookback_periods * 10 :]

    def _extract_cycle_signals(self, signals: List[SensorySignal]) -> List[SensorySignal]:
        """Extract signals relevant to cycle detection."""
        cycle_signals = []

        for signal in signals:
            if signal.signal_type in ["price_composite", "momentum", "volatility"]:
                cycle_signals.append(signal)

        return cycle_signals

    def _detect_cycles(self, signals: List[SensorySignal]) -> CycleAnalysis:
        """Detect cycles in the signal data."""
        if len(signals) < self.min_cycle_length:
            return CycleAnalysis(
                cycle_type="unknown",
                phase="unknown",
                cycle_length=0,
                confidence=0.0,
                metadata={"reason": "insufficient_data"},
            )

        # Extract signal values
        values = [float(cast(Any, s).value) for s in signals]

        # Simple cycle detection using peak finding
        peaks = self._find_peaks(values)
        troughs = self._find_troughs(values)

        if len(peaks) < 2 or len(troughs) < 2:
            return CycleAnalysis(
                cycle_type="unknown",
                phase="unknown",
                cycle_length=0,
                confidence=0.0,
                metadata={"reason": "no_clear_cycles"},
            )

        # Calculate cycle characteristics
        cycle_length = self._calculate_cycle_length(peaks, troughs)
        cycle_type = self._classify_cycle_type(values, peaks, troughs)
        phase = self._determine_cycle_phase(values, peaks, troughs)
        confidence = self._calculate_cycle_confidence(values, peaks, troughs)

        return CycleAnalysis(
            cycle_type=cycle_type,
            phase=phase,
            cycle_length=cycle_length,
            confidence=confidence,
            metadata={
                "peak_count": len(peaks),
                "trough_count": len(troughs),
                "avg_cycle_length": cycle_length,
            },
        )

    def _find_peaks(self, values: List[float]) -> List[int]:
        """Find peaks in the signal values."""
        peaks = []

        for i in range(1, len(values) - 1):
            if values[i] > values[i - 1] and values[i] > values[i + 1]:
                peaks.append(i)

        return peaks

    def _find_troughs(self, values: List[float]) -> List[int]:
        """Find troughs in the signal values."""
        troughs = []

        for i in range(1, len(values) - 1):
            if values[i] < values[i - 1] and values[i] < values[i + 1]:
                troughs.append(i)

        return troughs

    def _calculate_cycle_length(self, peaks: List[int], troughs: List[int]) -> int:
        """Calculate average cycle length."""
        all_extrema = sorted(peaks + troughs)

        if len(all_extrema) < 2:
            return 0

        # Calculate distances between consecutive extrema
        distances = []
        for i in range(1, len(all_extrema)):
            distance = all_extrema[i] - all_extrema[i - 1]
            if distance >= self.min_cycle_length:
                distances.append(distance)

        if not distances:
            return 0

        return int(np.mean(distances))

    def _classify_cycle_type(
        self, values: List[float], peaks: List[int], troughs: List[int]
    ) -> str:
        """Classify the type of cycle."""
        if not peaks or not troughs:
            return "unknown"

        # Calculate trend
        if len(values) >= 20:
            recent_values = values[-20:]
            trend = np.polyfit(range(len(recent_values)), recent_values, 1)[0]
        else:
            trend = 0

        # Classify based on trend and volatility
        volatility = np.std(values) if len(values) > 1 else 0

        if trend > 0.01:
            return "bull_market"
        elif trend < -0.01:
            return "bear_market"
        elif volatility < 0.1:
            return "consolidation"
        else:
            return "unknown"

    def _determine_cycle_phase(
        self, values: List[float], peaks: List[int], troughs: List[int]
    ) -> str:
        """Determine the current phase of the cycle."""
        if not peaks or not troughs:
            return "unknown"

        current_index = len(values) - 1

        # Find the most recent peak and trough
        recent_peaks = [p for p in peaks if p < current_index]
        recent_troughs = [t for t in troughs if t < current_index]

        if not recent_peaks and not recent_troughs:
            return "unknown"

        # Determine if we're closer to a peak or trough
        if recent_peaks and recent_troughs:
            last_peak = max(recent_peaks)
            last_trough = max(recent_troughs)

            if last_peak > last_trough:
                # We're in a peak-to-trough phase
                phase_position = (
                    (current_index - last_peak) / (last_trough - last_peak)
                    if last_trough > last_peak
                    else 0.5
                )
            else:
                # We're in a trough-to-peak phase
                phase_position = (
                    (current_index - last_trough) / (last_peak - last_trough)
                    if last_peak > last_trough
                    else 0.5
                )

            if phase_position < 0.33:
                return "early"
            elif phase_position < 0.67:
                return "middle"
            else:
                return "late"
        else:
            return "transition"

    def _calculate_cycle_confidence(
        self, values: List[float], peaks: List[int], troughs: List[int]
    ) -> float:
        """Calculate confidence in cycle detection."""
        if not peaks or not troughs:
            return 0.0

        # Factors affecting confidence:
        # 1. Number of cycles detected
        cycle_count = min(len(peaks), len(troughs))
        cycle_confidence = min(cycle_count / 3.0, 1.0)

        # 2. Regularity of cycles
        if len(peaks) >= 2 and len(troughs) >= 2:
            peak_distances = [peaks[i] - peaks[i - 1] for i in range(1, len(peaks))]
            trough_distances = [troughs[i] - troughs[i - 1] for i in range(1, len(troughs))]

            if peak_distances and trough_distances:
                peak_regularity = (
                    1.0 - (np.std(peak_distances) / np.mean(peak_distances))
                    if np.mean(peak_distances) > 0
                    else 0.0
                )
                trough_regularity = (
                    1.0 - (np.std(trough_distances) / np.mean(trough_distances))
                    if np.mean(trough_distances) > 0
                    else 0.0
                )
                regularity_confidence = (peak_regularity + trough_regularity) / 2.0
            else:
                regularity_confidence = 0.0
        else:
            regularity_confidence = 0.0

        # 3. Signal strength
        signal_strength = np.std(values) if len(values) > 1 else 0
        strength_confidence = min(float(signal_strength) * 10.0, 1.0)

        # Combine confidence factors
        overall_confidence = (cycle_confidence + regularity_confidence + strength_confidence) / 3.0
        return float(max(0.0, min(1.0, float(overall_confidence))))

    def _create_neutral_result(self) -> AnalysisResult:
        """Create neutral result when no cycles are detected."""
        return cast(
            AnalysisResult,
            {
                "timestamp": datetime.now(),
                "analysis_type": "cycle_detection",
                "result": {
                    "cycle_type": "unknown",
                    "phase": "unknown",
                    "cycle_length": 0,
                    "confidence": 0.0,
                    "metadata": {"reason": "no_cycles_detected"},
                },
                "confidence": 0.0,
                "metadata": {"signal_count": 0, "analysis_method": "no_data"},
            },
        )
