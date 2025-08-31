"""
EMP Anomaly Detector v1.1

Detects anomalies in market data and sensory signals.
Provides anomaly detection and classification capabilities.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Mapping, Optional, cast

import numpy as np

from src.core.interfaces import AnalysisResult, SensorySignal, ThinkingPattern
from src.core.exceptions import TradingException

logger = logging.getLogger(__name__)


@dataclass
class AnomalyDetectionResult:
    """Anomaly detection result (canonicalized name to avoid duplicate class across modules)."""

    is_anomaly: bool
    anomaly_type: str  # 'price_spike', 'volume_surge', 'pattern_break', etc.
    severity: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0
    description: str
    metadata: dict[str, object]


# Backward-compat alias (structural unification without behavior change)
AnomalyDetection = AnomalyDetectionResult


class AnomalyDetector(ThinkingPattern):
    """Detects anomalies in market data and sensory signals."""

    def __init__(self, config: Optional[dict[str, object]] = None):
        self.config = config or {}
        val = self.config.get("sensitivity", 0.7)
        self.sensitivity = float(val) if isinstance(val, (int, float, str)) else 0.7
        lb = self.config.get("lookback_periods", 50)
        self.lookback_periods = int(lb) if isinstance(lb, (int, float, str)) else 50
        self._signal_history: List[SensorySignal] = []

    def analyze(self, signals: List[SensorySignal]) -> AnalysisResult:
        """Analyze sensory signals to detect anomalies."""
        try:
            # Update signal history
            self._update_signal_history(signals)

            # Detect different types of anomalies
            price_anomalies = self._detect_price_anomalies(signals)
            volume_anomalies = self._detect_volume_anomalies(signals)
            pattern_anomalies = self._detect_pattern_anomalies(signals)

            # Combine anomaly detections
            combined_anomaly = self._combine_anomaly_detections(
                price_anomalies, volume_anomalies, pattern_anomalies
            )

            # Create analysis result
            return cast(
                AnalysisResult,
                {
                    "timestamp": datetime.now(),
                    "analysis_type": "anomaly_detection",
                    "result": {
                        "is_anomaly": combined_anomaly.is_anomaly,
                        "anomaly_type": combined_anomaly.anomaly_type,
                        "severity": combined_anomaly.severity,
                        "confidence": combined_anomaly.confidence,
                        "description": combined_anomaly.description,
                        "price_anomalies": price_anomalies.__dict__,
                        "volume_anomalies": volume_anomalies.__dict__,
                        "pattern_anomalies": pattern_anomalies.__dict__,
                        "metadata": combined_anomaly.metadata,
                    },
                    "confidence": combined_anomaly.confidence,
                    "metadata": {
                        "signal_count": len(signals),
                        "analysis_method": "multi_dimensional_anomaly",
                    },
                },
            )

        except Exception as e:
            raise TradingException(f"Error in anomaly detection: {e}")

    def learn(self, feedback: Mapping[str, object]) -> bool:
        """Learn from feedback to improve anomaly detection."""
        try:
            # Extract learning data from feedback
            acc = feedback.get("anomaly_accuracy")
            if isinstance(acc, (int, float)):
                accuracy = float(acc)
                # Adjust sensitivity based on accuracy
                if accuracy < 0.5:
                    self.sensitivity *= 0.9  # Lower sensitivity
                elif accuracy > 0.8:
                    self.sensitivity *= 1.1  # Raise sensitivity

            logger.info("Anomaly detector learned from feedback")
            return True

        except Exception as e:
            logger.error(f"Error in anomaly detector learning: {e}")
            return False

    def _update_signal_history(self, signals: List[SensorySignal]) -> None:
        """Update the signal history."""
        self._signal_history.extend(signals)

        # Keep only recent signals
        if len(self._signal_history) > self.lookback_periods * 10:
            self._signal_history = self._signal_history[-self.lookback_periods * 10 :]

    def _detect_price_anomalies(self, signals: List[SensorySignal]) -> AnomalyDetection:
        """Detect price-related anomalies."""
        price_signals = [s for s in signals if getattr(s, "signal_type", None) == "price_composite"]

        if not price_signals:
            return AnomalyDetection(
                is_anomaly=False,
                anomaly_type="none",
                severity=0.0,
                confidence=0.0,
                description="No price signals available",
                metadata={},
            )

        # Calculate price volatility
        values = [float(getattr(s, "value", 0.0)) for s in price_signals[-10:]]  # Last 10 signals
        if len(values) < 2:
            return AnomalyDetection(
                is_anomaly=False,
                anomaly_type="none",
                severity=0.0,
                confidence=0.0,
                description="Insufficient price data",
                metadata={},
            )

        volatility = np.std(values)
        mean_value = np.mean(values)

        # Detect price spikes
        latest_value = values[-1]
        price_change = abs(latest_value - mean_value)

        # Check if change is anomalous
        threshold = volatility * 2.5 * self.sensitivity
        is_anomaly = price_change > threshold

        if is_anomaly:
            anomaly_type = "price_spike"
            severity = float(min(float(price_change) / float(threshold), 1.0))
            confidence = float(
                min(float(volatility) / 0.1, 1.0)
            )  # Higher volatility = higher confidence
            description = f"Price spike detected: {price_change:.4f} vs threshold {threshold:.4f}"
        else:
            anomaly_type = "none"
            severity = 0.0
            confidence = 1.0
            description = "No price anomalies detected"

        # Normalize primitive types with explicit variables for mypy
        is_anomaly_b: bool = bool(is_anomaly)
        severity_f: float = float(severity)
        confidence_f: float = float(confidence)

        return AnomalyDetection(
            is_anomaly=is_anomaly_b,
            anomaly_type=anomaly_type,
            severity=severity_f,
            confidence=confidence_f,
            description=description,
            metadata={
                "volatility": volatility,
                "mean_value": mean_value,
                "latest_value": latest_value,
                "price_change": price_change,
                "threshold": threshold,
            },
        )

    def _detect_volume_anomalies(self, signals: List[SensorySignal]) -> AnomalyDetection:
        """Detect volume-related anomalies."""
        volume_signals = [
            s for s in signals if getattr(s, "signal_type", None) == "volume_composite"
        ]

        if not volume_signals:
            return AnomalyDetection(
                is_anomaly=False,
                anomaly_type="none",
                severity=0.0,
                confidence=0.0,
                description="No volume signals available",
                metadata={},
            )

        # Calculate volume statistics
        values = [float(getattr(s, "value", 0.0)) for s in volume_signals[-10:]]
        if len(values) < 2:
            return AnomalyDetection(
                is_anomaly=False,
                anomaly_type="none",
                severity=0.0,
                confidence=0.0,
                description="Insufficient volume data",
                metadata={},
            )

        mean_volume = np.mean(values)
        volume_std = np.std(values)
        latest_volume = values[-1]

        # Detect volume surges
        volume_ratio = latest_volume / mean_volume if mean_volume > 0 else 1.0
        threshold = 2.0 * self.sensitivity

        is_anomaly = volume_ratio > threshold

        if is_anomaly:
            anomaly_type = "volume_surge"
            severity = float(min(float((volume_ratio - 1.0)) / float((threshold - 1.0)), 1.0))
            confidence = float(min(float(volume_std) / 0.1, 1.0))
            description = f"Volume surge detected: {volume_ratio:.2f}x normal volume"
        else:
            anomaly_type = "none"
            severity = 0.0
            confidence = 1.0
            description = "No volume anomalies detected"

        # Normalize primitive types with explicit variables for mypy
        is_anomaly_b: bool = bool(is_anomaly)
        severity_f: float = float(severity)
        confidence_f: float = float(confidence)

        return AnomalyDetection(
            is_anomaly=is_anomaly_b,
            anomaly_type=anomaly_type,
            severity=severity_f,
            confidence=confidence_f,
            description=description,
            metadata={
                "mean_volume": mean_volume,
                "latest_volume": latest_volume,
                "volume_ratio": volume_ratio,
                "threshold": threshold,
            },
        )

    def _detect_pattern_anomalies(self, signals: List[SensorySignal]) -> AnomalyDetection:
        """Detect pattern-related anomalies."""
        # This is a simplified pattern anomaly detection
        # In a real implementation, this would use more sophisticated pattern recognition

        if len(signals) < 5:
            return AnomalyDetection(
                is_anomaly=False,
                anomaly_type="none",
                severity=0.0,
                confidence=0.0,
                description="Insufficient signals for pattern analysis",
                metadata={},
            )

        # Check for unusual signal patterns
        recent_signals = signals[-5:]
        signal_types = [str(getattr(s, "signal_type", "")) for s in recent_signals]

        # Detect if all signals are of the same type (unusual)
        unique_types = set(signal_types)
        if len(unique_types) == 1:
            anomaly_type = "pattern_uniformity"
            severity = 0.5
            confidence = 0.7
            description = f"Unusual signal uniformity: all {len(recent_signals)} signals are {list(unique_types)[0]}"
            is_anomaly = True
        else:
            anomaly_type = "none"
            severity = 0.0
            confidence = 1.0
            description = "No pattern anomalies detected"
            is_anomaly = False

        return AnomalyDetection(
            is_anomaly=is_anomaly,
            anomaly_type=anomaly_type,
            severity=severity,
            confidence=confidence,
            description=description,
            metadata={"signal_types": signal_types, "unique_types": len(unique_types)},
        )

    def _combine_anomaly_detections(
        self,
        price_anomaly: AnomalyDetection,
        volume_anomaly: AnomalyDetection,
        pattern_anomaly: AnomalyDetection,
    ) -> AnomalyDetection:
        """Combine multiple anomaly detections into a single result."""

        # Find the most severe anomaly
        anomalies = [price_anomaly, volume_anomaly, pattern_anomaly]
        severe_anomalies = [a for a in anomalies if a.is_anomaly]

        if not severe_anomalies:
            return AnomalyDetection(
                is_anomaly=False,
                anomaly_type="none",
                severity=0.0,
                confidence=1.0,
                description="No anomalies detected across all dimensions",
                metadata={
                    "price_anomaly": price_anomaly.anomaly_type,
                    "volume_anomaly": volume_anomaly.anomaly_type,
                    "pattern_anomaly": pattern_anomaly.anomaly_type,
                },
            )

        # Select the most severe anomaly
        most_severe = max(severe_anomalies, key=lambda a: a.severity)

        # Calculate combined confidence
        confidences = [a.confidence for a in severe_anomalies]
        combined_confidence = float(np.mean(confidences))

        # Create combined description
        descriptions = [a.description for a in severe_anomalies]
        combined_description = f"Multiple anomalies detected: {'; '.join(descriptions)}"

        return AnomalyDetection(
            is_anomaly=True,
            anomaly_type=most_severe.anomaly_type,
            severity=most_severe.severity,
            confidence=combined_confidence,
            description=combined_description,
            metadata={
                "anomaly_count": len(severe_anomalies),
                "price_anomaly": price_anomaly.anomaly_type,
                "volume_anomaly": volume_anomaly.anomaly_type,
                "pattern_anomaly": pattern_anomaly.anomaly_type,
                "most_severe_type": most_severe.anomaly_type,
            },
        )
