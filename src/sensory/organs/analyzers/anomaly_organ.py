"""
Anomaly Detection Module - Anomaly Sense

This module handles statistical anomaly detection and chaos analysis
for the "anomaly" sense.

Author: EMP Development Team
Date: July 18, 2024
Phase: 2 - Missing Function Implementation
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Mapping, NotRequired, Required, Sequence, TypedDict, cast

import numpy as np
import pandas as pd

from src.core.types import JSONObject

logger = logging.getLogger(__name__)

# TypedDicts for structured payloads used in this module


class DetectionConfig(TypedDict, total=False):
    threshold: float
    min_points: int
    smoothing: NotRequired[float]


AnomalyRecord = TypedDict(
    "AnomalyRecord",
    {
        "type": Required[str],
        "subtype": NotRequired[str],
        "index": NotRequired[int],
        "z_score": NotRequired[float],
        "return": NotRequired[float],
        "confidence": NotRequired[float],
        "timestamp": NotRequired[datetime | pd.Timestamp | None],
        "volume": NotRequired[float],
        "volatility": NotRequired[float],
        "gap_size": NotRequired[float],
    },
    total=False,
)

ChaosPattern = TypedDict(
    "ChaosPattern",
    {
        "type": Required[str],
        "subtype": NotRequired[str],
        "hurst_exponent": NotRequired[float],
        "entropy": NotRequired[float],
        "sensitivity": NotRequired[float],
        "confidence": NotRequired[float],
        "description": NotRequired[str],
    },
    total=False,
)

ManipulationPattern = TypedDict(
    "ManipulationPattern",
    {
        "type": Required[str],
        "subtype": NotRequired[str],
        "index": NotRequired[int],
        "pump_size": NotRequired[float],
        "dump_size": NotRequired[float],
        "volume_spike": NotRequired[float],
        "price_reversal": NotRequired[float],
        "price_stability": NotRequired[float],
        "volume_consistency": NotRequired[float],
        "confidence": NotRequired[float],
        "description": NotRequired[str],
    },
    total=False,
)


class AnomalyWindowSummary(TypedDict):
    anomalies_detected: int
    anomaly_types: list[str]
    anomalies: list[AnomalyRecord]
    anomaly_score: float
    timestamp: datetime


class ChaosWindowSummary(TypedDict):
    chaos_patterns_detected: int
    chaos_types: list[str]
    patterns: list[ChaosPattern]
    chaos_score: float
    timestamp: datetime


class ManipulationWindowSummary(TypedDict):
    manipulation_patterns_detected: int
    manipulation_types: list[str]
    patterns: list[ManipulationPattern]
    manipulation_score: float
    timestamp: datetime


__all__ = [
    "StatisticalAnomalyDetector",
    "ChaosDetector",
    "ManipulationDetector",
    "DetectionConfig",
    "AnomalyRecord",
    "ChaosPattern",
    "ManipulationPattern",
    "AnomalyWindowSummary",
    "ChaosWindowSummary",
    "ManipulationWindowSummary",
]


class StatisticalAnomalyDetector:
    """
    Statistical Anomaly Detector

    Detects statistical anomalies using various methods.
    """

    def __init__(self, config: DetectionConfig | None = None) -> None:
        """Initialize statistical anomaly detector"""
        self.config = config or {}
        self.anomalies_detected: list[AnomalyRecord] = []
        logger.info("StatisticalAnomalyDetector initialized")

    def detect_statistical_anomalies(self, df: pd.DataFrame) -> Sequence[AnomalyRecord]:
        """
        Detect statistical anomalies in market data.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            List of detected anomalies
        """
        if df.empty:
            return []

        try:
            anomalies: list[AnomalyRecord] = []

            # Detect various types of anomalies
            anomalies.extend(self._detect_price_anomalies(df))
            anomalies.extend(self._detect_volume_anomalies(df))
            anomalies.extend(self._detect_volatility_anomalies(df))
            anomalies.extend(self._detect_pattern_anomalies(df))

            self.anomalies_detected.extend(anomalies)
            return anomalies

        except Exception as e:
            logger.error(f"Error detecting statistical anomalies: {e}")
            return []

    def update_data(self, df: pd.DataFrame) -> Mapping[str, object]:
        """
        Update data and detect anomalies.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            Anomaly detection results
        """
        if df.empty:
            return {}

        try:
            anomalies = self.detect_statistical_anomalies(df)

            analysis = {
                "anomalies_detected": len(anomalies),
                "anomaly_types": list(set(a["type"] for a in anomalies)),
                "anomalies": anomalies,
                "anomaly_score": self._calculate_anomaly_score(anomalies),
                "timestamp": datetime.now(),
            }

            return analysis

        except Exception as e:
            logger.error(f"Error updating anomaly data: {e}")
            return {}

    def _detect_price_anomalies(self, df: pd.DataFrame) -> list[AnomalyRecord]:
        """Detect price-based anomalies"""
        try:
            anomalies: list[AnomalyRecord] = []

            if len(df) < 20:
                return anomalies

            # Calculate price statistics
            returns = df["close"].pct_change().dropna()
            mean_return = float(returns.mean())
            std_return = float(returns.std())

            # Detect outliers using z-score as a numpy array to ensure indexable typing
            if std_return > 0:
                z_scores = ((returns - mean_return) / std_return).to_numpy(dtype=float)
            else:
                z_scores = np.zeros(len(returns), dtype=float)

            # Find extreme values (z-score > 3 or < -3)
            extreme_indices = np.where(np.abs(z_scores) > 3)[0]

            for idx in extreme_indices:
                idx_int = int(idx)
                if idx_int < len(df) - 1:  # Ensure we have the data point
                    anomalies.append(
                        {
                            "type": "price_anomaly",
                            "subtype": "extreme_return",
                            "index": idx_int,
                            "z_score": float(z_scores[idx_int]),
                            "return": float(returns.iloc[idx_int]),
                            "confidence": float(min(abs(z_scores[idx_int]) / 5.0, 1.0)),
                            "timestamp": cast(
                                datetime | pd.Timestamp | None,
                                (
                                    df.index[idx_int]
                                    if isinstance(df.index[idx_int], (datetime, pd.Timestamp))
                                    else None
                                ),
                            ),
                        }
                    )

            return anomalies

        except Exception as e:
            logger.error(f"Error detecting price anomalies: {e}")
            return []

    def _detect_volume_anomalies(self, df: pd.DataFrame) -> list[AnomalyRecord]:
        """Detect volume-based anomalies"""
        try:
            anomalies: list[AnomalyRecord] = []

            if len(df) < 20:
                return anomalies

            # Calculate volume statistics
            volume = df["volume"]
            mean_volume = float(volume.mean())
            std_volume = float(volume.std())

            # Detect volume spikes (ensure numpy array for indexing)
            if std_volume > 0:
                volume_z_scores = ((volume - mean_volume) / std_volume).to_numpy(dtype=float)
            else:
                volume_z_scores = np.zeros(len(volume), dtype=float)

            # Find extreme volume (z-score > 2.5)
            extreme_volume_indices = np.where(volume_z_scores > 2.5)[0]

            for idx in extreme_volume_indices:
                idx_int = int(idx)
                anomalies.append(
                    {
                        "type": "volume_anomaly",
                        "subtype": "volume_spike",
                        "index": idx_int,
                        "z_score": float(volume_z_scores[idx_int]),
                        "volume": float(volume.iloc[idx_int]),
                        "confidence": float(min(volume_z_scores[idx_int] / 5.0, 1.0)),
                        "timestamp": cast(
                            datetime | pd.Timestamp | None,
                            (
                                df.index[idx_int]
                                if isinstance(df.index[idx_int], (datetime, pd.Timestamp))
                                else None
                            ),
                        ),
                    }
                )

            return anomalies

        except Exception as e:
            logger.error(f"Error detecting volume anomalies: {e}")
            return []

    def _detect_volatility_anomalies(self, df: pd.DataFrame) -> list[AnomalyRecord]:
        """Detect volatility-based anomalies"""
        try:
            anomalies: list[AnomalyRecord] = []

            if len(df) < 20:
                return anomalies

            # Calculate rolling volatility
            returns = df["close"].pct_change().dropna()
            rolling_vol = returns.rolling(10).std()

            # Detect volatility spikes
            mean_vol = float(rolling_vol.mean())
            std_vol = float(rolling_vol.std())

            if std_vol > 0:
                vol_z_scores = ((rolling_vol - mean_vol) / std_vol).to_numpy(dtype=float)
            else:
                vol_z_scores = np.zeros(len(rolling_vol), dtype=float)

            # Find extreme volatility (z-score > 2)
            extreme_vol_indices = np.where(vol_z_scores > 2)[0]

            for idx in extreme_vol_indices:
                idx_int = int(idx)
                if idx_int < len(df) - 1:
                    anomalies.append(
                        {
                            "type": "volatility_anomaly",
                            "subtype": "volatility_spike",
                            "index": idx_int,
                            "z_score": float(vol_z_scores[idx_int]),
                            "volatility": float(rolling_vol.iloc[idx_int]),
                            "confidence": float(min(vol_z_scores[idx_int] / 4.0, 1.0)),
                            "timestamp": cast(
                                datetime | pd.Timestamp | None,
                                (
                                    df.index[idx_int]
                                    if isinstance(df.index[idx_int], (datetime, pd.Timestamp))
                                    else None
                                ),
                            ),
                        }
                    )

            return anomalies

        except Exception as e:
            logger.error(f"Error detecting volatility anomalies: {e}")
            return []

    def _detect_pattern_anomalies(self, df: pd.DataFrame) -> list[AnomalyRecord]:
        """Detect pattern-based anomalies"""
        try:
            anomalies: list[AnomalyRecord] = []

            if len(df) < 10:
                return anomalies

            # Detect unusual price patterns
            for i in range(2, len(df) - 2):
                current = df.iloc[i]
                prev = df.iloc[i - 1]
                next_candle = df.iloc[i + 1]

                # Detect unusual gaps
                gap_up = current["low"] > prev["high"]
                gap_down = current["high"] < prev["low"]

                if gap_up:
                    anomalies.append(
                        {
                            "type": "pattern_anomaly",
                            "subtype": "gap_up",
                            "index": i,
                            "gap_size": float(current["low"] - prev["high"]),
                            "confidence": 0.8,
                            "timestamp": df.index[i] if hasattr(df.index[i], "timestamp") else None,
                        }
                    )

                elif gap_down:
                    anomalies.append(
                        {
                            "type": "pattern_anomaly",
                            "subtype": "gap_down",
                            "index": i,
                            "gap_size": float(prev["low"] - current["high"]),
                            "confidence": 0.8,
                            "timestamp": df.index[i] if hasattr(df.index[i], "timestamp") else None,
                        }
                    )

            return anomalies

        except Exception as e:
            logger.error(f"Error detecting pattern anomalies: {e}")
            return []

    def _calculate_anomaly_score(self, anomalies: Sequence[AnomalyRecord]) -> float:
        """Calculate overall anomaly score"""
        try:
            if not anomalies:
                return 0.0

            # Weighted average of anomaly confidences
            total_weighted_confidence = 0.0
            total_weight = 0.0

            for anomaly in anomalies:
                confidence = float(anomaly.get("confidence", 0.0))
                weight = 1.0

                # Weight by anomaly type
                if anomaly.get("type") == "price_anomaly":
                    weight = 1.5
                elif anomaly.get("type") == "volume_anomaly":
                    weight = 1.2

                total_weighted_confidence += confidence * weight
                total_weight += weight

            anomaly_score = total_weighted_confidence / total_weight if total_weight > 0 else 0.0
            return min(max(anomaly_score, 0.0), 1.0)

        except Exception as e:
            logger.error(f"Error calculating anomaly score: {e}")
            return 0.0


class ChaosDetector:
    """
    Chaos Detector

    Detects chaos patterns and market disorder.
    """

    def __init__(self, config: DetectionConfig | None = None) -> None:
        """Initialize chaos detector"""
        self.config = config or {}
        logger.info("ChaosDetector initialized")

    def detect_chaos_patterns(self, df: pd.DataFrame) -> Sequence[ChaosPattern]:
        """
        Detect chaos patterns in market data.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            List of detected chaos patterns
        """
        if df.empty:
            return []

        try:
            patterns: list[ChaosPattern] = []

            # Detect various chaos patterns
            patterns.extend(self._detect_fractal_chaos(df))
            patterns.extend(self._detect_entropy_chaos(df))
            patterns.extend(self._detect_butterfly_effect(df))

            return patterns

        except Exception as e:
            logger.error(f"Error detecting chaos patterns: {e}")
            return []

    def update_data(self, df: pd.DataFrame) -> Mapping[str, object]:
        """
        Update data and detect chaos patterns.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            Chaos detection results
        """
        if df.empty:
            return {}

        try:
            patterns = self.detect_chaos_patterns(df)

            analysis = {
                "chaos_patterns_detected": len(patterns),
                "chaos_types": list(set(p["type"] for p in patterns)),
                "patterns": patterns,
                "chaos_score": self._calculate_chaos_score(patterns),
                "timestamp": datetime.now(),
            }

            return analysis

        except Exception as e:
            logger.error(f"Error updating chaos data: {e}")
            return {}

    def _detect_fractal_chaos(self, df: pd.DataFrame) -> list[ChaosPattern]:
        """Detect fractal chaos patterns"""
        try:
            patterns: list[ChaosPattern] = []

            if len(df) < 20:
                return patterns

            # Calculate fractal dimension (simplified)
            returns = df["close"].pct_change().dropna()

            # Calculate Hurst exponent approximation
            hurst = self._calculate_hurst_exponent(returns)

            if hurst < 0.4:  # Anti-persistent (chaotic)
                patterns.append(
                    {
                        "type": "fractal_chaos",
                        "subtype": "anti_persistent",
                        "hurst_exponent": hurst,
                        "confidence": 0.7,
                        "description": "Anti-persistent price movement indicating chaos",
                    }
                )

            return patterns

        except Exception as e:
            logger.error(f"Error detecting fractal chaos: {e}")
            return []

    def _detect_entropy_chaos(self, df: pd.DataFrame) -> list[ChaosPattern]:
        """Detect entropy-based chaos"""
        try:
            patterns: list[ChaosPattern] = []

            if len(df) < 20:
                return patterns

            # Calculate entropy of returns
            returns = df["close"].pct_change().dropna()
            entropy = self._calculate_entropy(returns)

            # High entropy indicates chaos
            if entropy > 0.8:
                patterns.append(
                    {
                        "type": "entropy_chaos",
                        "subtype": "high_entropy",
                        "entropy": entropy,
                        "confidence": 0.6,
                        "description": "High entropy indicating market disorder",
                    }
                )

            return patterns

        except Exception as e:
            logger.error(f"Error detecting entropy chaos: {e}")
            return []

    def _detect_butterfly_effect(self, df: pd.DataFrame) -> list[ChaosPattern]:
        """Detect butterfly effect patterns"""
        try:
            patterns: list[ChaosPattern] = []

            if len(df) < 30:
                return patterns

            # Look for small changes leading to large effects
            returns = df["close"].pct_change().dropna()

            # Calculate sensitivity to initial conditions
            sensitivity = self._calculate_sensitivity(returns)

            if sensitivity > 0.7:
                patterns.append(
                    {
                        "type": "butterfly_effect",
                        "subtype": "high_sensitivity",
                        "sensitivity": sensitivity,
                        "confidence": 0.5,
                        "description": "High sensitivity to initial conditions",
                    }
                )

            return patterns

        except Exception as e:
            logger.error(f"Error detecting butterfly effect: {e}")
            return []

    def _calculate_hurst_exponent(self, series: pd.Series) -> float:
        """Calculate Hurst exponent (simplified)"""
        try:
            if len(series) < 10:
                return 0.5

            # Simplified Hurst calculation
            # In practice, this would be more complex
            lags = range(2, min(10, len(series) // 2))
            tau = [np.sqrt(np.std(np.subtract(series[lag:], series[:-lag]))) for lag in lags]

            # Linear fit to double-log graph
            if len(tau) > 1:
                hurst = float(np.polyfit(np.log(lags), np.log(tau), 1)[0])
                return min(max(hurst, 0.0), 1.0)

            return 0.5

        except Exception as e:
            logger.error(f"Error calculating Hurst exponent: {e}")
            return 0.5

    def _calculate_entropy(self, series: pd.Series) -> float:
        """Calculate entropy of a series"""
        try:
            if len(series) < 10:
                return 0.0

            # Discretize the series into bins
            bins = pd.cut(series, bins=10, labels=False)

            # Calculate entropy
            value_counts = bins.value_counts()
            probabilities = (value_counts / len(bins)).to_numpy(dtype=float)

            # Shannon entropy
            entropy_raw = float(np.sum(probabilities * np.log2(probabilities + 1e-10)))
            entropy = float(0.0 - entropy_raw)

            # Normalize to [0, 1]
            max_entropy = np.log2(10)  # For 10 bins
            normalized_entropy = entropy / max_entropy

            return float(min(max(float(normalized_entropy), 0.0), 1.0))

        except Exception as e:
            logger.error(f"Error calculating entropy: {e}")
            return 0.0

    def _calculate_sensitivity(self, series: pd.Series) -> float:
        """Calculate sensitivity to initial conditions"""
        try:
            if len(series) < 20:
                return 0.0

            # Calculate how small changes propagate
            # Simplified measure based on volatility clustering
            volatility = series.rolling(5).std()
            volatility_autocorr = volatility.autocorr()

            # High autocorrelation indicates sensitivity
            sensitivity = (
                float(abs(volatility_autocorr)) if not pd.isna(volatility_autocorr) else 0.0
            )

            return min(max(sensitivity, 0.0), 1.0)

        except Exception as e:
            logger.error(f"Error calculating sensitivity: {e}")
            return 0.0

    def _calculate_chaos_score(self, patterns: Sequence[ChaosPattern]) -> float:
        """Calculate overall chaos score"""
        try:
            if not patterns:
                return 0.0

            # Average confidence of chaos patterns
            total_confidence = sum(p.get("confidence", 0.0) for p in patterns)
            chaos_score = total_confidence / len(patterns)

            return min(max(chaos_score, 0.0), 1.0)

        except Exception as e:
            logger.error(f"Error calculating chaos score: {e}")
            return 0.0


class ManipulationDetector:
    """
    Manipulation Detector

    Detects potential market manipulation patterns.
    """

    def __init__(self, config: DetectionConfig | None = None) -> None:
        """Initialize manipulation detector"""
        self.config = config or {}
        logger.info("ManipulationDetector initialized")

    def detect_manipulation_patterns(self, df: pd.DataFrame) -> Sequence[ManipulationPattern]:
        """
        Detect potential manipulation patterns.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            List of detected manipulation patterns
        """
        if df.empty:
            return []

        try:
            patterns: list[ManipulationPattern] = []

            # Detect various manipulation patterns
            patterns.extend(self._detect_pump_and_dump(df))
            patterns.extend(self._detect_spoofing(df))
            patterns.extend(self._detect_layering(df))

            return patterns

        except Exception as e:
            logger.error(f"Error detecting manipulation patterns: {e}")
            return []

    def update_data(self, df: pd.DataFrame) -> Mapping[str, object]:
        """
        Update data and detect manipulation patterns.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            Manipulation detection results
        """
        if df.empty:
            return {}

        try:
            patterns = self.detect_manipulation_patterns(df)

            analysis = {
                "manipulation_patterns_detected": len(patterns),
                "manipulation_types": list(set(p["type"] for p in patterns)),
                "patterns": patterns,
                "manipulation_score": self._calculate_manipulation_score(patterns),
                "timestamp": datetime.now(),
            }

            return analysis

        except Exception as e:
            logger.error(f"Error updating manipulation data: {e}")
            return {}

    def _detect_pump_and_dump(self, df: pd.DataFrame) -> list[ManipulationPattern]:
        """Detect pump and dump patterns"""
        try:
            patterns: list[ManipulationPattern] = []

            if len(df) < 20:
                return patterns

            # Look for rapid price increase followed by sharp decline
            returns = df["close"].pct_change().dropna()

            # Find rapid price increases
            for i in range(5, len(returns) - 5):
                recent_returns = returns.iloc[i - 5 : i + 1]
                future_returns = returns.iloc[i + 1 : i + 6]

                # Pump: rapid increase
                if recent_returns.sum() > 0.05:  # 5% increase
                    # Dump: sharp decline
                    if future_returns.sum() < -0.03:  # 3% decline
                        patterns.append(
                            {
                                "type": "manipulation",
                                "subtype": "pump_and_dump",
                                "index": i,
                                "pump_size": float(recent_returns.sum()),
                                "dump_size": float(future_returns.sum()),
                                "confidence": 0.6,
                                "description": "Potential pump and dump pattern",
                            }
                        )

            return patterns

        except Exception as e:
            logger.error(f"Error detecting pump and dump: {e}")
            return []

    def _detect_spoofing(self, df: pd.DataFrame) -> list[ManipulationPattern]:
        """Detect spoofing patterns"""
        try:
            patterns: list[ManipulationPattern] = []

            if len(df) < 10:
                return patterns

            # Look for large orders followed by cancellations
            # Simplified detection based on volume patterns
            volume = df["volume"]
            price = df["close"]

            for i in range(2, len(df) - 2):
                # Large volume spike
                if volume.iloc[i] > volume.iloc[i - 1] * 3:
                    # Price reversal
                    if price.iloc[i] > price.iloc[i - 1] and price.iloc[i + 1] < price.iloc[i]:
                        patterns.append(
                            {
                                "type": "manipulation",
                                "subtype": "spoofing",
                                "index": i,
                                "volume_spike": float(volume.iloc[i] / volume.iloc[i - 1]),
                                "price_reversal": float(
                                    (price.iloc[i] - price.iloc[i + 1]) / price.iloc[i]
                                ),
                                "confidence": 0.5,
                                "description": "Potential spoofing pattern",
                            }
                        )

            return patterns

        except Exception as e:
            logger.error(f"Error detecting spoofing: {e}")
            return []

    def _detect_layering(self, df: pd.DataFrame) -> list[ManipulationPattern]:
        """Detect layering patterns"""
        try:
            patterns: list[ManipulationPattern] = []

            if len(df) < 15:
                return patterns

            # Look for repeated small orders at same price level
            # Simplified detection
            price = df["close"]
            volume = df["volume"]

            for i in range(5, len(df) - 5):
                # Check for price stability with consistent volume
                recent_prices = price.iloc[i - 5 : i + 1]
                recent_volumes = volume.iloc[i - 5 : i + 1]

                price_std = recent_prices.std()
                volume_mean = recent_volumes.mean()
                volume_std = recent_volumes.std()

                # Layering: stable price with consistent small volumes
                if (
                    price_std < price.iloc[i] * 0.001  # Very stable price
                    and volume_std < volume_mean * 0.3
                ):  # Consistent volume
                    patterns.append(
                        {
                            "type": "manipulation",
                            "subtype": "layering",
                            "index": i,
                            "price_stability": float(1.0 - (price_std / price.iloc[i])),
                            "volume_consistency": float(1.0 - (volume_std / volume_mean)),
                            "confidence": 0.4,
                            "description": "Potential layering pattern",
                        }
                    )

            return patterns

        except Exception as e:
            logger.error(f"Error detecting layering: {e}")
            return []

    def _calculate_manipulation_score(self, patterns: Sequence[ManipulationPattern]) -> float:
        """Calculate overall manipulation score"""
        try:
            if not patterns:
                return 0.0

            # Weighted average of manipulation pattern confidences
            total_weighted_confidence = 0.0
            total_weight = 0.0

            for pattern in patterns:
                confidence = float(pattern.get("confidence", 0.0))
                weight = 1.0

                # Weight by manipulation type
                if pattern.get("subtype") == "pump_and_dump":
                    weight = 1.5
                elif pattern.get("subtype") == "spoofing":
                    weight = 1.3

                total_weighted_confidence += confidence * weight
                total_weight += weight

            manipulation_score = (
                total_weighted_confidence / total_weight if total_weight > 0 else 0.0
            )
            return min(max(manipulation_score, 0.0), 1.0)

        except Exception as e:
            logger.error(f"Error calculating manipulation score: {e}")
            return 0.0


# Example usage
if __name__ == "__main__":
    # Test anomaly detection modules
    detector = StatisticalAnomalyDetector()
    chaos = ChaosDetector()
    manipulation = ManipulationDetector()
    print("Anomaly detection modules initialized successfully")
