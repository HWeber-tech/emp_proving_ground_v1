"""Ingest telemetry publishers that integrate Timescale runs with the event bus."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Mapping, Sequence

from src.core.event_bus import Event, EventBus, get_global_bus

from ..persist.timescale import TimescaleIngestResult
from .timescale_pipeline import IngestResultPublisher


def _normalise_metadata(metadata: Mapping[str, object]) -> dict[str, object]:
    """Coerce metadata payloads into event-bus-safe primitives."""

    def _coerce(value: object) -> object:
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, datetime):
            ts = value if value.tzinfo else value.replace(tzinfo=UTC)
            return ts.astimezone(UTC).isoformat()
        if isinstance(value, Mapping):
            return {str(k): _coerce(v) for k, v in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [_coerce(v) for v in value]
        return str(value)

    return {str(key): _coerce(val) for key, val in metadata.items()}


class EventBusIngestPublisher(IngestResultPublisher):
    """Emit ingest outcomes to an :class:`EventBus` for runtime consumers."""

    def __init__(
        self,
        bus: EventBus,
        *,
        topic: str = "telemetry.ingest",
        source: str = "timescale_ingest",
    ) -> None:
        self._bus = bus
        self._topic = topic
        self._source = source
        self._logger = logging.getLogger(f"{__name__}.EventBusIngestPublisher")

    def publish(
        self,
        result: TimescaleIngestResult,
        *,
        metadata: Mapping[str, object] | None = None,
    ) -> None:
        payload: dict[str, object] = {"result": result.as_dict()}
        if metadata:
            payload["metadata"] = _normalise_metadata(metadata)

        event = Event(type=self._topic, payload=payload, source=self._source)

        publish_from_sync = getattr(self._bus, "publish_from_sync", None)
        is_running = getattr(self._bus, "is_running", lambda: False)
        if callable(publish_from_sync) and callable(is_running) and is_running():
            try:
                publish_from_sync(event)
                return
            except Exception:
                self._logger.exception("Failed to publish ingest telemetry on local event bus")

        try:
            topic_bus = get_global_bus()
            topic_bus.publish_sync(self._topic, payload, source=self._source)
        except Exception:
            self._logger.debug(
                "Ingest telemetry publish skipped – no global bus available",
                exc_info=True,
            )


class CompositeIngestPublisher(IngestResultPublisher):
    """Dispatch ingest results to multiple downstream publishers."""

    def __init__(self, publishers: Sequence[IngestResultPublisher]) -> None:
        self._publishers = tuple(publishers)
        self._logger = logging.getLogger(f"{__name__}.CompositeIngestPublisher")

    def publish(
        self,
        result: TimescaleIngestResult,
        *,
        metadata: Mapping[str, object] | None = None,
    ) -> None:
        for publisher in self._publishers:
            try:
                publisher.publish(result, metadata=metadata)
            except Exception:
                self._logger.exception("Ingest publisher %s failed", publisher.__class__.__name__)


def combine_ingest_publishers(
    *publishers: IngestResultPublisher | None,
) -> IngestResultPublisher | None:
    """Filter ``None`` entries and compose ingest publishers when required."""

    resolved: list[IngestResultPublisher] = [
        publisher for publisher in publishers if publisher is not None
    ]
    if not resolved:
        return None
    if len(resolved) == 1:
        return resolved[0]
    return CompositeIngestPublisher(resolved)


__all__ = [
    "CompositeIngestPublisher",
    "EventBusIngestPublisher",
    "combine_ingest_publishers",
]
