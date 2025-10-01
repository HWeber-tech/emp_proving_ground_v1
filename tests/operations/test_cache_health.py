from datetime import UTC, datetime
from typing import Any

import pytest

from src.operations.cache_health import (
    CacheHealthSnapshot,
    CacheHealthStatus,
    evaluate_cache_health,
    publish_cache_health,
)
from src.operations.event_bus_failover import EventPublishError


def test_cache_health_fails_when_expected_but_missing() -> None:
    snapshot = evaluate_cache_health(
        configured=False,
        expected=True,
        namespace="emp:cache",
        backing=None,
    )

    assert snapshot.status is CacheHealthStatus.fail
    assert any("expected" in issue for issue in snapshot.issues)


def test_cache_health_warns_on_zero_traffic() -> None:
    snapshot = evaluate_cache_health(
        configured=True,
        expected=True,
        namespace="emp:cache",
        backing="RedisClient",
        metrics={"hits": 0, "misses": 0},
    )

    assert snapshot.status is CacheHealthStatus.warn
    assert any("No cache traffic" in issue for issue in snapshot.issues)


def test_cache_health_ok_with_high_hit_rate() -> None:
    snapshot = evaluate_cache_health(
        configured=True,
        expected=True,
        namespace="emp:cache",
        backing="RedisClient",
        metrics={"hits": 90, "misses": 5},
    )

    assert snapshot.status is CacheHealthStatus.ok
    assert snapshot.hit_rate is not None and snapshot.hit_rate > 0.9


def test_cache_health_warns_on_evictions_and_includes_policy_metadata() -> None:
    snapshot = evaluate_cache_health(
        configured=True,
        expected=True,
        namespace="emp:cache",
        backing="RedisClient",
        metrics={"hits": 50, "misses": 10, "evictions": 3},
        policy={"ttl_seconds": 900, "max_keys": 256},
    )

    assert snapshot.status is CacheHealthStatus.warn
    assert any("evicted" in issue for issue in snapshot.issues)
    markdown = snapshot.to_markdown()
    assert "TTL seconds" in markdown
    assert "256" in markdown


def _snapshot() -> CacheHealthSnapshot:
    return CacheHealthSnapshot(
        service="redis_cache",
        generated_at=datetime.now(tz=UTC),
        status=CacheHealthStatus.ok,
        configured=True,
        expected=True,
        namespace="emp:cache",
        backing="RedisClient",
        hit_rate=0.9,
        hits=90,
        misses=10,
        evictions=0,
        expirations=0,
        invalidations=0,
        metadata={},
        issues=(),
    )


def test_publish_cache_health_falls_back_on_runtime_error(monkeypatch: pytest.MonkeyPatch) -> None:
    published: list[dict[str, Any]] = []

    class DummyEventBus:
        def __init__(self) -> None:
            self._running = True
            self.invocations = 0

        def is_running(self) -> bool:
            return self._running

        def publish_from_sync(self, event: Any) -> None:
            self.invocations += 1
            raise RuntimeError("primary bus unavailable")

    class DummyTopicBus:
        def publish_sync(self, topic: str, payload: dict[str, Any], *, source: str | None = None) -> None:
            published.append({"topic": topic, "payload": payload, "source": source})

    topic_bus = DummyTopicBus()
    monkeypatch.setattr(
        "src.operations.event_bus_failover.get_global_bus", lambda: topic_bus
    )

    publish_cache_health(DummyEventBus(), _snapshot())

    assert published and published[0]["topic"] == "telemetry.cache.health"


def test_publish_cache_health_raises_on_unexpected_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class DummyEventBus:
        def is_running(self) -> bool:
            return True

        def publish_from_sync(self, event: Any) -> None:
            raise ValueError("unexpected")

    def _fail() -> None:
        raise AssertionError("Global bus should not be called on unexpected errors")

    monkeypatch.setattr("src.operations.event_bus_failover.get_global_bus", _fail)

    with pytest.raises(EventPublishError) as excinfo:
        publish_cache_health(DummyEventBus(), _snapshot())

    assert excinfo.value.stage == "runtime"
    assert isinstance(excinfo.value.__cause__, ValueError)
