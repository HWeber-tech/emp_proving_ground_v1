from __future__ import annotations

"""Summarise sensory cortex state for runtime dashboards and telemetry feeds."""

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Iterable, Mapping, MutableMapping, Sequence

import pandas as pd
from pandas.errors import OutOfBoundsDatetime, ParserError

from src.core.event_bus import Event, EventBus, TopicBus
from src.operations.event_bus_failover import publish_event_with_failover

__all__ = [
    "SensoryDimensionSummary",
    "SensorySummary",
    "build_sensory_summary",
    "publish_sensory_summary",
]


logger = logging.getLogger(__name__)


def _coerce_float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if value is None:
        return None
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def _normalise_mapping(mapping: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if not isinstance(mapping, Mapping):
        return {}
    return {str(key): value for key, value in mapping.items()}


_TIMESTAMP_EXCEPTIONS: tuple[type[BaseException], ...] = (
    TypeError,
    ValueError,
    OverflowError,
    ParserError,
    OutOfBoundsDatetime,
)


def _parse_timestamp(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value
    try:
        converted = pd.to_datetime(value, utc=True, errors="coerce")
    except _TIMESTAMP_EXCEPTIONS as exc:
        logger.debug("Failed to parse sensory timestamp", extra={"value": value}, exc_info=exc)
        return None
    if converted is pd.NaT:
        return None
    if hasattr(converted, "to_pydatetime"):
        return converted.to_pydatetime()
    return None


def _serialise_datetime(value: datetime | None) -> str | None:
    if value is None:
        return None
    if value.tzinfo is not None:
        return value.isoformat()
    return value.replace(tzinfo=timezone.utc).isoformat()


def _normalise_sequence(entries: Iterable[Any]) -> tuple[Mapping[str, Any], ...]:
    normalised: list[Mapping[str, Any]] = []
    for entry in entries:
        if isinstance(entry, Mapping):
            normalised.append({str(key): value for key, value in entry.items()})
    return tuple(normalised)


@dataclass(frozen=True)
class SensoryDimensionSummary:
    """Summarised telemetry for a single sensory dimension."""

    name: str
    signal: float | None
    confidence: float | None
    state: str | None
    threshold_state: str | None
    metadata: Mapping[str, Any]

    def as_dict(self) -> Mapping[str, Any]:
        payload: MutableMapping[str, Any] = {
            "name": self.name,
            "metadata": dict(self.metadata),
        }
        if self.signal is not None:
            payload["signal"] = self.signal
        if self.confidence is not None:
            payload["confidence"] = self.confidence
        if self.state is not None:
            payload["state"] = self.state
        if self.threshold_state is not None:
            payload["threshold_state"] = self.threshold_state
        return payload


@dataclass(frozen=True)
class SensorySummary:
    """Aggregated sensory cortex snapshot for dashboards and telemetry."""

    symbol: str | None
    generated_at: datetime | None
    samples: int
    integrated_strength: float | None
    integrated_confidence: float | None
    integrated_direction: float | None
    contributing: tuple[str, ...]
    dimensions: tuple[SensoryDimensionSummary, ...]
    drift_summary: Mapping[str, Any] | None
    audit_entries: tuple[Mapping[str, Any], ...]

    def as_dict(self) -> Mapping[str, Any]:
        return {
            "symbol": self.symbol,
            "generated_at": _serialise_datetime(self.generated_at),
            "samples": self.samples,
            "integrated_strength": self.integrated_strength,
            "integrated_confidence": self.integrated_confidence,
            "integrated_direction": self.integrated_direction,
            "contributing": list(self.contributing),
            "dimensions": [dimension.as_dict() for dimension in self.dimensions],
            "drift_summary": dict(self.drift_summary) if self.drift_summary else None,
            "audit_entries": [dict(entry) for entry in self.audit_entries],
        }

    def top_dimensions(self, limit: int = 3) -> tuple[SensoryDimensionSummary, ...]:
        if limit <= 0:
            return ()
        ranked = sorted(
            self.dimensions,
            key=lambda dimension: abs(dimension.signal or 0.0),
            reverse=True,
        )
        return tuple(ranked[:limit])

    def to_markdown(self, *, limit: int = 5) -> str:
        header: list[str] = []
        symbol = self.symbol or "UNKNOWN"
        integrated_strength = self.integrated_strength
        if integrated_strength is not None:
            header.append(f"**Integrated strength:** {integrated_strength:+.3f}")
        if self.integrated_confidence is not None:
            header.append(f"confidence={self.integrated_confidence:.2f}")
        if self.integrated_direction is not None:
            header.append(f"direction={self.integrated_direction:+.0f}")

        summary_line = f"**Symbol:** {symbol} | **Samples:** {self.samples}"
        if header:
            summary_line = summary_line + " | " + ", ".join(header)

        lines = [summary_line, ""]
        lines.append("| Dimension | Signal | Confidence | State | Threshold |")
        lines.append("| --- | --- | --- | --- | --- |")

        for dimension in self.top_dimensions(limit):
            signal = dimension.signal if dimension.signal is not None else 0.0
            confidence = dimension.confidence if dimension.confidence is not None else 0.0
            state = dimension.state or "-"
            threshold = dimension.threshold_state or "-"
            lines.append(
                f"| {dimension.name} | {signal:+.3f} | {confidence:.2f} | {state} | {threshold} |"
            )

        if self.drift_summary:
            exceeded = self.drift_summary.get("exceeded")
            if exceeded:
                exceeded_names = ", ".join(
                    str(entry.get("sensor")) for entry in exceeded if isinstance(entry, Mapping)
                )
                lines.append("")
                lines.append(f"Drift alerts: {exceeded_names or 'None'}")

        return "\n".join(lines)


def _build_dimension(name: str, payload: Mapping[str, Any]) -> SensoryDimensionSummary:
    signal = _coerce_float(payload.get("signal"))
    confidence = _coerce_float(payload.get("confidence"))
    metadata = _normalise_mapping(payload.get("metadata"))
    state = metadata.get("state") if isinstance(metadata.get("state"), str) else None

    threshold_state: str | None = None
    thresholds = metadata.get("threshold_assessment")
    if isinstance(thresholds, Mapping):
        value = thresholds.get("state")
        if isinstance(value, str):
            threshold_state = value

    return SensoryDimensionSummary(
        name=name,
        signal=signal,
        confidence=confidence,
        state=state,
        threshold_state=threshold_state,
        metadata=metadata,
    )


def build_sensory_summary(status: Mapping[str, Any] | None) -> SensorySummary:
    mapping = status if isinstance(status, Mapping) else {}
    samples = int(mapping.get("samples") or 0)

    latest = mapping.get("latest") if isinstance(mapping.get("latest"), Mapping) else {}
    generated_at = _parse_timestamp(latest.get("generated_at"))
    symbol = latest.get("symbol") if isinstance(latest.get("symbol"), str) else None

    integrated = latest.get("integrated_signal")
    if not isinstance(integrated, Mapping):
        integrated = {}

    integrated_strength = _coerce_float(integrated.get("strength"))
    integrated_confidence = _coerce_float(integrated.get("confidence"))
    integrated_direction = _coerce_float(integrated.get("direction"))
    contributions = tuple(
        str(entry)
        for entry in integrated.get("contributing", [])
        if isinstance(entry, str)
    )

    dimensions_payload = latest.get("dimensions")
    dimension_summaries: list[SensoryDimensionSummary] = []
    if isinstance(dimensions_payload, Mapping):
        for name, payload in dimensions_payload.items():
            if not isinstance(name, str) or not isinstance(payload, Mapping):
                continue
            dimension_summaries.append(_build_dimension(name, payload))

    dimension_summaries.sort(key=lambda dimension: dimension.name)

    drift_summary_raw = mapping.get("drift_summary")
    drift_summary = (
        {
            key: value if not isinstance(value, Sequence) else list(value)
            for key, value in drift_summary_raw.items()
        }
        if isinstance(drift_summary_raw, Mapping)
        else None
    )

    audit_entries_raw = mapping.get("sensor_audit")
    audit_entries = _normalise_sequence(audit_entries_raw or [])

    return SensorySummary(
        symbol=symbol,
        generated_at=generated_at,
        samples=samples,
        integrated_strength=integrated_strength,
        integrated_confidence=integrated_confidence,
        integrated_direction=integrated_direction,
        contributing=contributions,
        dimensions=tuple(dimension_summaries),
        drift_summary=drift_summary,
        audit_entries=audit_entries,
    )


def publish_sensory_summary(
    summary: SensorySummary,
    *,
    event_bus: EventBus,
    event_type: str = "telemetry.sensory.summary",
    global_bus_factory: Callable[[], TopicBus] | None = None,
) -> None:
    """Publish the sensory summary via the runtime event bus with failover."""

    payload = summary.as_dict()
    payload["markdown"] = summary.to_markdown()

    event = Event(
        type=event_type,
        payload=payload,
        source="operations.sensory_summary",
    )

    publish_event_with_failover(
        event_bus,
        event,
        logger=logger,
        runtime_fallback_message="Runtime bus rejected sensory summary; falling back to global bus",
        runtime_unexpected_message="Unexpected error publishing sensory summary via runtime bus",
        runtime_none_message="Runtime bus returned no result while publishing sensory summary",
        global_not_running_message="Global event bus not running while publishing sensory summary",
        global_unexpected_message="Unexpected error publishing sensory summary via global bus",
        global_bus_factory=global_bus_factory,  # type: ignore[arg-type]
    )
