"""Recorded replay telemetry publication helpers.

Connects recorded sensory replay evaluations to the operational telemetry
channel so dashboards inherit lineage-rich payloads even when the runtime bus
is degraded.
"""

from __future__ import annotations

import logging

from src.core.event_bus import Event, EventBus
from src.evolution.evaluation.telemetry import RecordedReplayTelemetrySnapshot
from src.operations.event_bus_failover import publish_event_with_failover

logger = logging.getLogger(__name__)

EVENT_TYPE_RECORDED_REPLAY = "telemetry.evolution.recorded_replay"
EVENT_SOURCE_RECORDED_REPLAY = "professional_runtime"


def build_recorded_replay_event(
    snapshot: RecordedReplayTelemetrySnapshot,
) -> Event:
    """Build an event payload for recorded replay telemetry."""

    return Event(
        type=EVENT_TYPE_RECORDED_REPLAY,
        payload=snapshot.as_dict(),
        source=EVENT_SOURCE_RECORDED_REPLAY,
    )


def publish_recorded_replay_snapshot(
    event_bus: EventBus, snapshot: RecordedReplayTelemetrySnapshot
) -> None:
    """Publish the recorded replay telemetry snapshot using failover semantics."""

    event = build_recorded_replay_event(snapshot)
    publish_event_with_failover(
        event_bus,
        event,
        logger=logger,
        runtime_fallback_message=(
            "Primary event bus publish_from_sync failed; falling back to global bus"
        ),
        runtime_unexpected_message=(
            "Unexpected error publishing recorded replay telemetry via runtime event bus"
        ),
        runtime_none_message=(
            "Primary event bus publish_from_sync returned None; falling back to global bus"
        ),
        global_not_running_message=(
            "Global event bus not running while publishing recorded replay telemetry"
        ),
        global_unexpected_message=(
            "Unexpected error publishing recorded replay telemetry via global bus"
        ),
    )


def format_recorded_replay_markdown(
    snapshot: RecordedReplayTelemetrySnapshot,
) -> str:
    """Render the snapshot as Markdown for dashboards and runbooks."""

    return snapshot.to_markdown()


__all__ = [
    "EVENT_TYPE_RECORDED_REPLAY",
    "EVENT_SOURCE_RECORDED_REPLAY",
    "build_recorded_replay_event",
    "publish_recorded_replay_snapshot",
    "format_recorded_replay_markdown",
]
