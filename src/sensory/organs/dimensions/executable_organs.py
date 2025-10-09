"""Executable organs that bridge canonical sensors into the cortex."""

import math
from collections import deque
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timezone
from typing import Any, cast

import pandas as pd

from .base_organ import MarketData, SensoryOrgan, SensoryReading
from src.sensory.anomaly import AnomalySensor
from src.sensory.how import HowSensor
from src.sensory.signals import SensorSignal
from src.sensory.what.what_sensor import WhatSensor
from src.sensory.when.when_sensor import WhenSensor
from src.sensory.why.why_sensor import WhySensor
from src.sensory.why.narrative_hooks import NarrativeEvent

__all__ = [
    "HowSensoryOrgan",
    "AnomalySensoryOrgan",
    "WhatSensoryOrgan",
    "WhenSensoryOrgan",
    "WhySensoryOrgan",
]


def _normalise_frame(
    payload: Any,
    *,
    drop_keys: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Coerce *payload* into a DataFrame understood by the sensors."""

    drop_set = {str(key) for key in drop_keys or ()}

    if payload is None:
        return pd.DataFrame()

    if isinstance(payload, pd.DataFrame):
        if drop_set:
            keep_cols = [col for col in payload.columns if col not in drop_set]
            payload = payload[keep_cols]
        if "timestamp" in payload:
            payload = payload.copy()
            payload["timestamp"] = pd.to_datetime(payload["timestamp"], utc=True, errors="coerce")
        return payload

    if isinstance(payload, MarketData):
        data = payload.dict()
        for key in drop_set:
            data.pop(key, None)
        data.setdefault("timestamp", payload.timestamp)
        return pd.DataFrame([data])

    if isinstance(payload, Mapping):
        data = dict(payload)
        for key in drop_set:
            data.pop(key, None)
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


def _ensure_utc(timestamp: datetime) -> datetime:
    if timestamp.tzinfo is None:
        return timestamp.replace(tzinfo=timezone.utc)
    return timestamp.astimezone(timezone.utc)


def _parse_datetime_sequence(source: Any) -> list[datetime]:
    if source is None:
        return []
    if isinstance(source, datetime):
        return [_ensure_utc(source)]

    values: Iterable[Any]
    if isinstance(source, Iterable) and not isinstance(source, (str, bytes, bytearray)):
        values = source
    else:
        values = [source]

    parsed: list[datetime] = []
    for value in values:
        if isinstance(value, datetime):
            parsed.append(_ensure_utc(value))
            continue
        try:
            ts = pd.to_datetime(value, utc=True, errors="coerce")
        except Exception:
            continue
        if ts is pd.NaT:
            continue
        parsed.append(ts.to_pydatetime())
    parsed.sort()
    return parsed


def _merge_datetime_sequences(*sequences: Iterable[datetime]) -> list[datetime]:
    merged: dict[str, datetime] = {}
    for sequence in sequences:
        for ts in sequence:
            key = _ensure_utc(ts).isoformat()
            merged[key] = _ensure_utc(ts)
    return sorted(merged.values())


def _normalise_option_positions(payload: Any) -> pd.DataFrame | None:
    if payload is None:
        return None
    if isinstance(payload, pd.DataFrame):
        return payload.copy()
    if isinstance(payload, Mapping):
        return pd.DataFrame([dict(payload)])
    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
        rows = [dict(row) for row in payload if isinstance(row, Mapping)]
        if rows:
            return pd.DataFrame(rows)
    return None


def _parse_macro_flags(source: Any) -> dict[str, float]:
    if not isinstance(source, Mapping):
        return {}
    result: dict[str, float] = {}
    for key, value in source.items():
        try:
            result[str(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return result


def _parse_narrative_events(source: Any) -> list[NarrativeEvent]:
    if source is None:
        return []
    events: list[NarrativeEvent] = []
    iterable: Iterable[Any]
    if isinstance(source, Iterable) and not isinstance(source, (str, bytes, bytearray)):
        iterable = source
    else:
        iterable = [source]

    for item in iterable:
        if isinstance(item, NarrativeEvent):
            events.append(item)
            continue
        if not isinstance(item, Mapping):
            continue
        timestamp_values = _parse_datetime_sequence(item.get("timestamp"))
        if not timestamp_values:
            continue
        ts = timestamp_values[0]
        try:
            sentiment = float(item.get("sentiment", 0.0))
        except (TypeError, ValueError):
            sentiment = 0.0
        try:
            importance = float(item.get("importance", 1.0))
        except (TypeError, ValueError):
            importance = 1.0
        description = item.get("description")
        events.append(
            NarrativeEvent(
                timestamp=ts,
                sentiment=sentiment,
                importance=importance,
                description=str(description) if description is not None else None,
            )
        )
    events.sort(key=lambda event: event.timestamp)
    return events


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


class WhatSensoryOrgan(SensoryOrgan):
    """Executable WHAT organ wrapping :class:`WhatSensor`."""

    def __init__(
        self,
        *,
        name: str = "what_organ",
        config: Mapping[str, Any] | None = None,
        sensor: WhatSensor | None = None,
    ) -> None:
        super().__init__(name, dict(config or {}))
        self._sensor = sensor or WhatSensor()

    async def process(
        self,
        market_data: MarketData | Mapping[str, Any] | pd.DataFrame | None,
    ) -> SensoryReading:
        frame = _normalise_frame(market_data)
        signals = self._sensor.process(frame)
        if not signals:
            signals = [SensorSignal(signal_type="WHAT", value={}, confidence=0.0)]
        return _build_sensory_reading(
            organ_name=self.name,
            dimension="WHAT",
            signal=signals[0],
        )


class WhenSensoryOrgan(SensoryOrgan):
    """Executable WHEN organ that marshals contextual inputs for :class:`WhenSensor`."""

    def __init__(
        self,
        *,
        name: str = "when_organ",
        config: Mapping[str, Any] | None = None,
        sensor: WhenSensor | None = None,
    ) -> None:
        base_config = dict(config or {})
        super().__init__(name, base_config)
        self._sensor = sensor or WhenSensor()
        self._config_macro_events = _parse_datetime_sequence(base_config.get("macro_events"))
        self._config_option_positions = _normalise_option_positions(
            base_config.get("option_positions")
        )

    async def process(
        self,
        market_data: MarketData | Mapping[str, Any] | pd.DataFrame | None,
    ) -> SensoryReading:
        drop_keys = ("macro_events", "option_positions")
        frame = _normalise_frame(market_data, drop_keys=drop_keys)

        macro_events = list(self._config_macro_events)
        option_positions = self._config_option_positions

        if isinstance(market_data, Mapping):
            payload_events = _parse_datetime_sequence(market_data.get("macro_events"))
            macro_events = _merge_datetime_sequences(macro_events, payload_events)
            payload_option_positions = _normalise_option_positions(
                market_data.get("option_positions")
            )
            if payload_option_positions is not None:
                option_positions = payload_option_positions

        signals = self._sensor.process(
            frame,
            option_positions=option_positions,
            macro_events=macro_events,
        )
        if not signals:
            signals = [SensorSignal(signal_type="WHEN", value={}, confidence=0.0)]
        return _build_sensory_reading(
            organ_name=self.name,
            dimension="WHEN",
            signal=signals[0],
        )


class WhySensoryOrgan(SensoryOrgan):
    """Executable WHY organ wrapping :class:`WhySensor`."""

    def __init__(
        self,
        *,
        name: str = "why_organ",
        config: Mapping[str, Any] | None = None,
        sensor: WhySensor | None = None,
    ) -> None:
        base_config = dict(config or {})
        super().__init__(name, base_config)
        self._sensor = sensor or WhySensor()
        self._config_narrative_events = _parse_narrative_events(base_config.get("narrative_events"))
        self._config_macro_flags = _parse_macro_flags(base_config.get("macro_regime_flags"))

    async def process(
        self,
        market_data: MarketData | Mapping[str, Any] | pd.DataFrame | None,
    ) -> SensoryReading:
        drop_keys = ("narrative_events", "macro_regime_flags", "as_of")
        frame = _normalise_frame(market_data, drop_keys=drop_keys)

        narrative_events = list(self._config_narrative_events)
        macro_flags = dict(self._config_macro_flags)
        as_of: datetime | None = None

        if isinstance(market_data, Mapping):
            payload_events = _parse_narrative_events(market_data.get("narrative_events"))
            if payload_events:
                narrative_events = _parse_narrative_events(narrative_events + payload_events)
            payload_flags = _parse_macro_flags(market_data.get("macro_regime_flags"))
            macro_flags.update(payload_flags)
            as_of_values = _parse_datetime_sequence(market_data.get("as_of"))
            if as_of_values:
                as_of = as_of_values[0]

        signals = self._sensor.process(
            frame,
            narrative_events=narrative_events,
            macro_regime_flags=macro_flags,
            as_of=as_of,
        )
        if not signals:
            signals = [SensorSignal(signal_type="WHY", value={}, confidence=0.0)]
        return _build_sensory_reading(
            organ_name=self.name,
            dimension="WHY",
            signal=signals[0],
        )
