"""
EMP Volume Sensory Organ v1.1

Processes volume data and extracts volume-related sensory signals
including volume trends, volume momentum, and volume-based indicators.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, cast

import numpy as np

from src.core.base import MarketData, SensoryOrgan
from src.core.exceptions import ResourceException as SensoryException
from src.sensory.signals import SensorSignal as SensorySignal

logger = logging.getLogger(__name__)


@dataclass
class VolumeSignal:
    """Volume-related sensory signal."""

    timestamp: datetime
    signal_type: str
    value: float
    confidence: float
    metadata: Dict[str, Any]


class VolumeOrgan(SensoryOrgan):
    """Sensory organ for processing volume data."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.calibrated = False
        self._volume_history: List[float] = []
        self._price_history: List[float] = []
        self._max_history = self.config.get("max_history", 1000)

    def perceive(self, data: MarketData) -> SensorySignal:
        """Process raw volume data into sensory signals."""
        try:
            # Add to history
            self._volume_history.append(data.volume)
            self._price_history.append(data.close)

            if len(self._volume_history) > self._max_history:
                self._volume_history.pop(0)
                self._price_history.pop(0)

            # Calculate volume signals
            signals = []

            # Volume trend signal
            if len(self._volume_history) >= 20:
                volume_trend_signal = self._calculate_volume_trend_signal()
                signals.append(volume_trend_signal)

            # Volume momentum signal
            if len(self._volume_history) >= 14:
                volume_momentum_signal = self._calculate_volume_momentum_signal()
                signals.append(volume_momentum_signal)

            # Volume-price relationship signal
            if len(self._volume_history) >= 20:
                volume_price_signal = self._calculate_volume_price_signal()
                signals.append(volume_price_signal)

            # OBV signal
            if len(self._volume_history) >= 10:
                obv_signal = self._calculate_obv_signal()
                signals.append(obv_signal)

            # VWAP signal
            if len(self._volume_history) >= 5:
                vwap_signal = self._calculate_vwap_signal(data)
                signals.append(vwap_signal)

            # Combine signals into a single sensory reading
            combined_signal = self._combine_signals(signals)

            return SensorySignal(
                timestamp=data.timestamp,
                signal_type="volume_composite",
                value=combined_signal.value,
                confidence=combined_signal.confidence,
                metadata=cast(
                    dict[str, object],
                    {
                        "signals": [s.__dict__ for s in signals],
                        "volume_history_length": len(self._volume_history),
                        "organ_id": "volume_organ",
                    },
                ),
            )

        except Exception as e:
            raise SensoryException(f"Error in volume perception: {e}")

    def calibrate(self) -> bool:
        """Calibrate the volume organ."""
        try:
            # Reset calibration state
            self.calibrated = False

            # Perform calibration checks
            if len(self._volume_history) < 20:
                logger.warning("Insufficient volume history for calibration")
                return False

            # Calculate calibration metrics
            volume_array = np.array(self._volume_history)

            # Check for reasonable volume values
            if np.any(volume_array < 0):
                raise SensoryException("Invalid volume values detected")

            # Check for reasonable volume variation
            volume_std = np.std(volume_array)
            if volume_std == 0:
                raise SensoryException("Zero volume variation detected")

            # Mark as calibrated
            self.calibrated = True
            logger.info("Volume organ calibrated successfully")
            return True

        except Exception as e:
            logger.error(f"Volume organ calibration failed: {e}")
            return False

    def _calculate_volume_trend_signal(self) -> VolumeSignal:
        """Calculate volume trend signal using moving averages."""
        volumes = np.array(self._volume_history)

        # Calculate short and long volume moving averages
        short_vol_ma = np.mean(volumes[-10:])  # 10-period MA
        long_vol_ma = np.mean(volumes[-20:])  # 20-period MA

        # Calculate volume trend strength
        volume_trend_strength = (short_vol_ma - long_vol_ma) / long_vol_ma

        # Determine volume trend direction
        if volume_trend_strength > 0.1:
            trend_direction = 1.0  # Increasing volume
        elif volume_trend_strength < -0.1:
            trend_direction = -1.0  # Decreasing volume
        else:
            trend_direction = 0.0  # Stable volume

        # Calculate confidence based on trend strength
        confidence = min(float(abs(volume_trend_strength) * 5), 1.0)

        return VolumeSignal(
            timestamp=datetime.now(),
            signal_type="volume_trend",
            value=trend_direction,
            confidence=confidence,
            metadata={
                "short_vol_ma": short_vol_ma,
                "long_vol_ma": long_vol_ma,
                "volume_trend_strength": volume_trend_strength,
            },
        )

    def _calculate_volume_momentum_signal(self) -> VolumeSignal:
        """Calculate volume momentum signal using rate of change."""
        volumes = np.array(self._volume_history)

        # Calculate volume rate of change over 14 periods
        vol_roc = (volumes[-1] - volumes[-14]) / volumes[-14]

        # Normalize to [-1, 1] range
        momentum = np.tanh(vol_roc * 2)  # Scale and bound

        # Calculate confidence based on magnitude
        confidence = min(float(abs(vol_roc) * 2), 1.0)

        return VolumeSignal(
            timestamp=datetime.now(),
            signal_type="volume_momentum",
            value=float(momentum),
            confidence=confidence,
            metadata={"volume_roc": vol_roc, "period": 14},
        )

    def _calculate_volume_price_signal(self) -> VolumeSignal:
        """Calculate volume-price relationship signal."""
        volumes = np.array(self._volume_history)
        prices = np.array(self._price_history)

        # Calculate price and volume changes
        price_changes = np.diff(prices[-20:])
        volume_changes = np.diff(volumes[-20:])

        # Calculate correlation between price and volume changes
        if len(price_changes) > 1 and len(volume_changes) > 1:
            correlation = np.corrcoef(price_changes, volume_changes)[0, 1]
            if np.isnan(correlation):
                correlation = 0.0
        else:
            correlation = 0.0

        # Normalize correlation to [-1, 1]
        volume_price_signal = correlation

        # Calculate confidence based on data quality
        confidence = min(len(price_changes) / 20, 1.0)

        return VolumeSignal(
            timestamp=datetime.now(),
            signal_type="volume_price_relationship",
            value=float(volume_price_signal),
            confidence=confidence,
            metadata={"correlation": correlation, "period": 20},
        )

    def _calculate_obv_signal(self) -> VolumeSignal:
        """Calculate On-Balance Volume (OBV) signal."""
        volumes = np.array(self._volume_history)
        prices = np.array(self._price_history)

        # Calculate OBV
        obv = 0
        for i in range(1, len(prices)):
            if prices[i] > prices[i - 1]:
                obv += volumes[i]  # Price up, add volume
            elif prices[i] < prices[i - 1]:
                obv -= volumes[i]  # Price down, subtract volume

        # Normalize OBV to [-1, 1] range
        if len(volumes) > 0:
            avg_volume = float(np.mean(volumes))
            normalized_obv = float(np.tanh(obv / (avg_volume * 10)))  # Scale appropriately
        else:
            normalized_obv = 0.0

        # Calculate confidence based on data quality
        confidence = min(len(prices) / 20, 1.0)

        return VolumeSignal(
            timestamp=datetime.now(),
            signal_type="obv",
            value=normalized_obv,
            confidence=confidence,
            metadata={
                "obv": obv,
                "avg_volume": float(np.mean(volumes)) if len(volumes) > 0 else 0.0,
            },
        )

    def _calculate_vwap_signal(self, data: MarketData) -> VolumeSignal:
        """Calculate Volume Weighted Average Price (VWAP) signal."""
        if len(self._volume_history) < 2 or len(self._price_history) < 2:
            return VolumeSignal(
                timestamp=data.timestamp, signal_type="vwap", value=0.0, confidence=0.0, metadata={}
            )

        # Calculate VWAP
        volumes = np.array(self._volume_history[-20:])  # Last 20 periods
        prices = np.array(self._price_history[-20:])

        # Calculate typical price (HLC/3) - using close as approximation
        typical_prices = prices

        # Calculate VWAP
        volume_price_sum: float = float(np.sum(volumes * typical_prices))
        volume_sum: float = float(np.sum(volumes))

        if volume_sum > 0:
            vwap = volume_price_sum / volume_sum
        else:
            vwap = data.close

        # Calculate VWAP signal (current price vs VWAP)
        vwap_signal = (data.close - vwap) / vwap

        # Normalize to [-1, 1] range
        normalized_signal = float(np.tanh(vwap_signal * 10))

        # Calculate confidence based on data quality
        confidence = min(len(volumes) / 20, 1.0)

        return VolumeSignal(
            timestamp=data.timestamp,
            signal_type="vwap",
            value=normalized_signal,
            confidence=confidence,
            metadata={"vwap": vwap, "current_price": data.close, "vwap_signal": vwap_signal},
        )

    def _combine_signals(self, signals: List[VolumeSignal]) -> VolumeSignal:
        """Combine multiple volume signals into a composite signal."""
        if not signals:
            return VolumeSignal(
                timestamp=datetime.now(),
                signal_type="volume_composite",
                value=0.0,
                confidence=0.0,
                metadata={},
            )

        # Weighted average of signal values
        total_weight: float = 0.0
        weighted_sum: float = 0.0

        for signal in signals:
            weight = signal.confidence
            total_weight += weight
            weighted_sum += signal.value * weight

        if total_weight > 0:
            composite_value = weighted_sum / total_weight
            composite_confidence = float(np.mean([s.confidence for s in signals]))
        else:
            composite_value = 0.0
            composite_confidence = 0.0

        return VolumeSignal(
            timestamp=datetime.now(),
            signal_type="volume_composite",
            value=composite_value,
            confidence=composite_confidence,
            metadata={
                "signal_count": len(signals),
                "individual_signals": [s.signal_type for s in signals],
            },
        )
