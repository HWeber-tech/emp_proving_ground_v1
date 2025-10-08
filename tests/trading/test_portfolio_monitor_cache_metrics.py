from __future__ import annotations

from dataclasses import dataclass

import pytest

from src.core.event_bus import Event
from src.data_foundation.cache import InMemoryRedis, ManagedRedisCache, RedisCachePolicy
from src.trading.monitoring.portfolio_monitor import PortfolioMonitor


@dataclass
class _RecordedEvent:
    event: Event
    published_via: str


class _RecordingBus:
    """Minimal EventBus stub that records synchronous publishes."""

    def __init__(self) -> None:
        self._events: list[_RecordedEvent] = []
        self._subscriptions: dict[str, list[object]] = {}
        self._running = True

    def subscribe(self, topic: str, handler: object) -> None:  # pragma: no cover - trivial
        self._subscriptions.setdefault(topic, []).append(handler)

    def publish_from_sync(self, event: Event) -> int:
        self._events.append(_RecordedEvent(event=event, published_via="local"))
        return 1

    def is_running(self) -> bool:
        return self._running

    @property
    def events(self) -> list[_RecordedEvent]:
        return self._events

    def clear(self) -> None:
        self._events.clear()


def _build_monitor(policy: RedisCachePolicy) -> tuple[PortfolioMonitor, _RecordingBus]:
    cache = ManagedRedisCache(InMemoryRedis(), policy)
    bus = _RecordingBus()
    monitor = PortfolioMonitor(bus, cache)
    return monitor, bus


def test_portfolio_monitor_emits_cache_metrics_on_initialisation() -> None:
    policy = RedisCachePolicy(ttl_seconds=900, max_keys=128, namespace="emp:test")
    _, bus = _build_monitor(policy)

    assert bus.events, "expected cache telemetry event on initial load"
    recorded = bus.events[-1]
    assert recorded.event.type == "telemetry.cache"

    payload = recorded.event.payload
    assert payload["reason"] == "initial_load"
    assert payload["policy"]["ttl_seconds"] == 900
    assert payload["policy"]["max_keys"] == 128
    assert payload["namespace"] == "emp:test"
    assert payload["configured"] is False
    assert payload["hits"] == 0
    assert payload["misses"] >= 1
    assert payload["hit_rate"] == pytest.approx(0.0)


@pytest.mark.parametrize("hit_expected, miss_expected", [(2, 1), (1, 2)])
def test_portfolio_monitor_cache_metrics_capture_activity(hit_expected: int, miss_expected: int) -> None:
    policy = RedisCachePolicy(ttl_seconds=120, max_keys=32, namespace="emp:metrics")
    monitor, bus = _build_monitor(policy)
    bus.clear()

    # Seed cache activity: set baseline state, generate hits and misses.
    for _ in range(hit_expected):
        monitor.redis_client.set("alpha", "1")
        assert monitor.redis_client.get("alpha") == "1"
    for _ in range(miss_expected):
        monitor.redis_client.get("missing")

    monitor._publish_cache_metrics(reason="unit_test")

    assert bus.events, "expected telemetry event after manual publish"
    payload = bus.events[-1].event.payload

    assert payload["reason"] == "unit_test"
    assert payload["hits"] >= hit_expected
    assert payload["misses"] >= miss_expected

    total = payload["hits"] + payload["misses"]
    assert total > 0
    assert payload["hit_rate"] == pytest.approx(payload["hits"] / total)
    assert payload["configured"] is False
    assert payload["backing"] == "InMemoryRedis"
    assert payload["policy"]["namespace"] == "emp:metrics"
