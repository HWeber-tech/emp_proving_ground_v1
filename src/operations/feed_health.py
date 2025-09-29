"""Operational helpers for data feed anomaly detection."""

from __future__ import annotations

from datetime import datetime
from typing import Iterable, Sequence

from src.core.event_bus import Event, EventBus, get_global_bus
from src.data_foundation.monitoring.feed_anomaly import (
    FeedAnomalyConfig,
    FeedAnomalyReport,
    FeedHealthStatus,
    Tick,
    analyse_feed,
)

__all__ = [
    "FeedAnomalyConfig",
    "FeedAnomalyReport",
    "FeedHealthStatus",
    "Tick",
    "evaluate_feed_health",
    "publish_feed_health",
]


def evaluate_feed_health(
    symbol: str,
    ticks: Sequence[Tick] | Iterable[Tick],
    *,
    config: FeedAnomalyConfig | None = None,
    now: datetime | None = None,
) -> FeedAnomalyReport:
    """Run the feed anomaly detector for operational reporting."""

    return analyse_feed(symbol, ticks, config=config, now=now)


def publish_feed_health(
    report: FeedAnomalyReport,
    bus: EventBus | None = None,
) -> int | None:
    """Publish feed health telemetry to the event bus."""

    resolved_bus = bus or get_global_bus()
    event = Event("telemetry.data_feed.health", report.as_dict())
    if hasattr(resolved_bus, "publish_from_sync"):
        return resolved_bus.publish_from_sync(event)
    return resolved_bus.publish(event)

