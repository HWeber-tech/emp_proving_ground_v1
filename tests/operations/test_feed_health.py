from datetime import UTC, datetime, timedelta

from src.core.event_bus import Event
from src.data_foundation.monitoring.feed_anomaly import Tick
from src.operations.feed_health import evaluate_feed_health, publish_feed_health


def _ticks() -> list[Tick]:
    start = datetime(2024, 1, 1, tzinfo=UTC)
    return [
        Tick(timestamp=start + timedelta(minutes=index), price=100.0 + index)
        for index in range(6)
    ]


def test_evaluate_feed_health_passes_through_report() -> None:
    report = evaluate_feed_health("EURUSD", _ticks(), now=datetime(2024, 1, 1, 0, 10, tzinfo=UTC))

    assert report.symbol == "EURUSD"
    assert report.sample_count == 6


def test_publish_feed_health_uses_event_bus() -> None:
    report = evaluate_feed_health("EURUSD", _ticks(), now=datetime(2024, 1, 1, 0, 10, tzinfo=UTC))

    class _StubBus:
        def __init__(self) -> None:
            self.events: list[Event] = []

        def publish_from_sync(self, event: Event) -> int:  # pragma: no cover - trivial
            self.events.append(event)
            return 1

    bus = _StubBus()
    publish_feed_health(report, bus)

    assert bus.events
    event = bus.events[0]
    assert event.type == "telemetry.data_feed.health"
    assert event.payload["symbol"] == "EURUSD"

