"""Publish sensory lineage payloads to telemetry sinks.

This module plugs the lineage records emitted by the HOW and ANOMALY sensory
organs into the wider telemetry surface.  The modernization roadmap calls for
"executable organs with lineage telemetry", which means we need a reusable
bridge that can publish lineage snapshots to the event bus while retaining a
small inspection buffer for runtime diagnostics.
"""

from __future__ import annotations

import logging
from collections import deque
from datetime import datetime, timezone
from threading import Lock
from typing import Any, Callable, Mapping, MutableMapping

from src.core.event_bus import Event, TopicBus, get_global_bus
from src.operations.event_bus_failover import EventPublishError, publish_event_with_failover
from src.sensory.lineage import SensorLineageRecord

__all__ = ["SensoryLineagePublisher"]


logger = logging.getLogger(__name__)


def _as_utc_timestamp(value: datetime | None) -> datetime:
    if value is None:
        return datetime.now(timezone.utc)
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _serialise_mapping(mapping: Mapping[str, Any] | None) -> dict[str, Any]:
    if not mapping:
        return {}
    serialised: MutableMapping[str, Any] = {}
    for key, value in mapping.items():
        serialised[str(key)] = _serialise_value(value)
    return dict(serialised)


def _serialise_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _serialise_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialise_value(item) for item in value]
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc).isoformat()
    return value


def _normalise_lineage(lineage: Any) -> Mapping[str, Any] | None:
    if lineage is None:
        return None
    if isinstance(lineage, SensorLineageRecord):
        return lineage.as_dict()
    if hasattr(lineage, "as_dict") and callable(lineage.as_dict):  # type: ignore[attr-defined]
        try:
            return lineage.as_dict()  # type: ignore[no-any-return]
        except Exception:
            return None
    if isinstance(lineage, Mapping):
        return dict(lineage)
    return None


class SensoryLineagePublisher:
    """Capture and publish sensory lineage telemetry."""

    def __init__(
        self,
        *,
        event_bus: Any | None = None,
        event_type: str = "telemetry.sensory.lineage",
        max_records: int = 256,
        global_bus_factory: Callable[[], TopicBus] | None = None,
    ) -> None:
        if max_records <= 0:
            raise ValueError("max_records must be positive")
        self._event_bus = event_bus
        self._event_type = event_type
        self._records: deque[dict[str, Any]] = deque(maxlen=max_records)
        self._lock = Lock()
        self._global_bus_factory = global_bus_factory or get_global_bus

    # ------------------------------------------------------------------
    def record(
        self,
        dimension: str,
        lineage: Any,
        *,
        symbol: str | None = None,
        generated_at: datetime | None = None,
        strength: Any | None = None,
        confidence: Any | None = None,
        state: str | None = None,
        threshold_state: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        payload = _normalise_lineage(lineage)
        if payload is None:
            return

        timestamp_value = _parse_timestamp(payload.get("generated_at")) if isinstance(payload, Mapping) else None
        if timestamp_value is None:
            timestamp_value = generated_at
        timestamp = _as_utc_timestamp(timestamp_value)
        iso_timestamp = timestamp.isoformat()

        record = {
            "dimension": str(dimension),
            "symbol": symbol,
            "generated_at": iso_timestamp,
            "strength": _to_float_or_none(strength),
            "confidence": _to_float_or_none(confidence),
            "state": state,
            "threshold_state": threshold_state,
            "lineage": payload,
            "metadata": _serialise_mapping(metadata),
        }

        with self._lock:
            self._records.append(record)

        self._publish(record)

    # ------------------------------------------------------------------
    def history(self, limit: int | None = None) -> list[dict[str, Any]]:
        with self._lock:
            items: list[dict[str, Any]] = list(self._records)

        if limit is not None:
            if limit <= 0:
                return []
            items = items[-limit:]

        items = list(reversed(items))
        return [dict(entry) for entry in items]

    def latest(self) -> dict[str, Any] | None:
        with self._lock:
            if not self._records:
                return None
            latest = self._records[-1]
        return dict(latest)

    # ------------------------------------------------------------------
    def _publish(self, record: Mapping[str, Any]) -> None:
        event_bus = self._event_bus
        if event_bus is None:
            return
        event_payload = dict(record)
        event = Event(
            type=self._event_type,
            payload=event_payload,
            source="sensory.lineage_publisher",
        )

        publish_from_sync = getattr(event_bus, "publish_from_sync", None)
        if not callable(publish_from_sync) or not hasattr(event_bus, "is_running"):
            self._legacy_publish(event_bus, event, event_payload)
            return

        try:
            publish_event_with_failover(
                event_bus,
                event,
                logger=logger,
                runtime_fallback_message=(
                    "Runtime bus rejected sensory lineage; falling back to global bus"
                ),
                runtime_unexpected_message=(
                    "Unexpected error publishing sensory lineage via runtime bus"
                ),
                runtime_none_message=(
                    "Runtime bus returned no result while publishing sensory lineage"
                ),
                global_not_running_message=(
                    "Global event bus not running while publishing sensory lineage"
                ),
                global_unexpected_message=(
                    "Unexpected error publishing sensory lineage via global bus"
                ),
                global_bus_factory=self._global_bus_factory,
            )
            return
        except EventPublishError:
            logger.debug(
                "Failed to publish sensory lineage via runtime and global buses",
                exc_info=True,
            )
            return
        except (AttributeError, TypeError):  # pragma: no cover - compatibility path
            self._legacy_publish(event_bus, event, event_payload)

    def _legacy_publish(self, event_bus: Any, event: Event, payload: Mapping[str, Any]) -> None:
        publish_from_sync = getattr(event_bus, "publish_from_sync", None)
        if callable(publish_from_sync):
            try:
                publish_from_sync(event)
                return
            except Exception:
                return

        publish_sync = getattr(event_bus, "publish_sync", None)
        if callable(publish_sync):
            try:
                publish_sync(self._event_type, payload, source=event.source)
            except Exception:
                return


def _parse_timestamp(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError:
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed
    return None


def _to_float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
