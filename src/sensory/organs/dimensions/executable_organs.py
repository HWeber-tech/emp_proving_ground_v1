from __future__ import annotations

"""Executable HOW and ANOMALY organs backed by the canonical sensors."""

import math
from collections import deque
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any, cast

import pandas as pd

from .base_organ import MarketData, SensoryOrgan, SensoryReading
from src.sensory.anomaly import AnomalySensor
from src.sensory.how import HowSensor
from src.sensory.signals import SensorSignal

__all__ = ["HowSensoryOrgan", "AnomalySensoryOrgan"]


def _normalise_frame(payload: Any) -> pd.DataFrame:
    """Coerce *payload* into a DataFrame understood by the sensors."""

    if payload is None:
        return pd.DataFrame()

    if isinstance(payload, pd.DataFrame):
        if "timestamp" in payload:
            payload = payload.copy()
            payload["timestamp"] = pd.to_datetime(payload["timestamp"], utc=True, errors="coerce")
        return payload

    if isinstance(payload, MarketData):
        data = payload.dict()
        data.setdefault("timestamp", payload.timestamp)
        return pd.DataFrame([data])

    if isinstance(payload, Mapping):
        data = dict(payload)
        if "timestamp" in data:
            data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True, errors="coerce")
        return pd.DataFrame([data])

    return pd.DataFrame()


def _extract_signal_strength(signal: SensorSignal) -> float:
    value = signal.value
    if isinstance(value, Mapping):
        raw = value.get("strength")
        try:
            return float(raw)
        except (TypeError, ValueError):
            return 0.0
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0


def _extract_confidence(signal: SensorSignal) -> float:
    try:
        return float(signal.confidence)
    except (TypeError, ValueError):
        return 0.0


def _extract_numeric_telemetry(source: Mapping[str, Any] | None) -> dict[str, float]:
    telemetry: dict[str, float] = {}
    if not isinstance(source, Mapping):
        return telemetry
    for key, value in source.items():
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if math.isnan(numeric) or math.isinf(numeric):
            continue
        telemetry[str(key)] = numeric
    return telemetry


def _clone_metadata(metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(metadata, Mapping):
        return {}
    cloned: dict[str, Any] = {}
    for key, value in metadata.items():
        if isinstance(value, Mapping):
            cloned[key] = dict(value)
        elif isinstance(value, (list, tuple)):
            cloned[key] = list(value)
        else:
            cloned[key] = value
    return cloned


def _build_sensory_reading(
    *,
    organ_name: str,
    dimension: str,
    signal: SensorSignal,
) -> SensoryReading:
    """Translate a :class:`SensorSignal` into a :class:`SensoryReading`."""

    strength = _extract_signal_strength(signal)
    confidence = _extract_confidence(signal)

    data_payload: dict[str, Any] = {
        "dimension": dimension,
        "signal_strength": strength,
        "confidence": confidence,
        "value": signal.value,
    }

    metadata = _clone_metadata(signal.metadata)
    metadata["dimension"] = dimension
    telemetry = _extract_numeric_telemetry(metadata.get("audit"))
    if telemetry:
        metadata.setdefault("telemetry", {}).update(telemetry)

    threshold_state: str | None = None
    thresholds = metadata.get("threshold_assessment")
    if isinstance(thresholds, Mapping):
        candidate = thresholds.get("state")
        if isinstance(candidate, str):
            threshold_state = candidate
    if threshold_state is not None:
        metadata["threshold_state"] = threshold_state

    lineage = metadata.get("lineage")
    if isinstance(lineage, Mapping):
        metadata["lineage"] = dict(lineage)

    timestamp = signal.timestamp if isinstance(signal.timestamp, datetime) else datetime.utcnow()

    return SensoryReading(
        organ_name=organ_name,
        timestamp=timestamp,
        data=data_payload,
        metadata=metadata,
    )


class HowSensoryOrgan(SensoryOrgan):
    """Executable HOW organ that wraps :class:`HowSensor`."""

    def __init__(
        self,
        *,
        name: str = "how_organ",
        config: Mapping[str, Any] | None = None,
        sensor: HowSensor | None = None,
    ) -> None:
        super().__init__(name, dict(config or {}))
        self._sensor = sensor or HowSensor()

    async def process(
        self,
        market_data: MarketData | Mapping[str, Any] | pd.DataFrame | None,
    ) -> SensoryReading:
        frame = _normalise_frame(market_data)
        signals = self._sensor.process(frame)
        if not signals:
            signals = [SensorSignal(signal_type="HOW", value={}, confidence=0.0)]
        return _build_sensory_reading(
            organ_name=self.name,
            dimension="HOW",
            signal=signals[0],
        )


class AnomalySensoryOrgan(SensoryOrgan):
    """Executable ANOMALY organ that wraps :class:`AnomalySensor`."""

    def __init__(
        self,
        *,
        name: str = "anomaly_organ",
        config: Mapping[str, Any] | None = None,
        sensor: AnomalySensor | None = None,
    ) -> None:
        super().__init__(name, dict(config or {}))
        self._sensor = sensor or AnomalySensor()
        sensor_config = cast(Any, getattr(self._sensor, "_config", None))
        window = int(getattr(sensor_config, "window", 32))
        self._sequence_min_length = int(getattr(sensor_config, "sequence_min_length", 8))
        self._window: deque[float] = deque(maxlen=window if window > 0 else 32)

    async def process(
        self,
        market_data: MarketData | Mapping[str, Any] | pd.DataFrame | Sequence[float] | None,
    ) -> SensoryReading:
        if isinstance(market_data, Sequence) and not isinstance(market_data, (str, bytes, bytearray)):
            sequence = [float(value) for value in market_data]
            signals = self._sensor.process(sequence)
            if not signals:
                signals = [SensorSignal(signal_type="ANOMALY", value={}, confidence=0.0)]
            return _build_sensory_reading(
                organ_name=self.name,
                dimension="ANOMALY",
                signal=signals[0],
            )

        frame = _normalise_frame(market_data)
        if not frame.empty and "close" in frame:
            try:
                closes = frame["close"].astype(float)
            except (TypeError, ValueError):
                closes = pd.Series(dtype=float)
            for value in closes:
                self._window.append(float(value))

        if len(self._window) >= self._sequence_min_length:
            signals = self._sensor.process(list(self._window))
        else:
            signals = self._sensor.process(frame)

        if not signals:
            signals = [SensorSignal(signal_type="ANOMALY", value={}, confidence=0.0)]

        return _build_sensory_reading(
            organ_name=self.name,
            dimension="ANOMALY",
            signal=signals[0],
        )

