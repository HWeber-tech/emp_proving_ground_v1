from src.operations.cache_health import CacheHealthStatus, evaluate_cache_health


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
