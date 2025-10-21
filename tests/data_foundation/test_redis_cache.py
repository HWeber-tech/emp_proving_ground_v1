from __future__ import annotations

import logging

import pytest

fakeredis = pytest.importorskip("fakeredis")

from src.data_foundation.cache.redis_cache import (
    InMemoryRedis,
    ManagedRedisCache,
    RedisCachePolicy,
    RedisConnectionSettings,
    configure_redis_client,
    wrap_managed_cache,
)


def test_redis_connection_settings_from_mapping_url() -> None:
    settings = RedisConnectionSettings.from_mapping(
        {
            "REDIS_URL": "rediss://user:secret@example.cache:6380/5",
            "REDIS_CLIENT_NAME": "emp-test-cache",
            "REDIS_SOCKET_TIMEOUT": "1.5",
            "REDIS_HEALTH_CHECK_INTERVAL": "0.75",
            "REDIS_RETRY_ON_TIMEOUT": "false",
        }
    )

    assert settings.configured is True
    assert settings.url == "rediss://user:secret@example.cache:6380/5"
    assert settings.host == "example.cache"
    assert settings.port == 6380
    assert settings.db == 5
    assert settings.username == "user"
    assert settings.password == "secret"
    assert settings.ssl is True
    assert settings.client_name == "emp-test-cache"
    assert settings.socket_timeout == 1.5
    assert settings.health_check_interval == 0.75
    assert settings.retry_on_timeout is False
    assert settings.summary().startswith("Redis endpoint rediss://user:***@example.cache:6380/5")


def test_configure_redis_client_uses_factory() -> None:
    settings = RedisConnectionSettings.from_mapping(
        {
            "REDIS_HOST": "localhost",
            "REDIS_PORT": "6379",
            "REDIS_DB": "12",
        }
    )

    clients: list[fakeredis.FakeRedis] = []

    def factory(cfg: RedisConnectionSettings) -> fakeredis.FakeRedis:
        client = fakeredis.FakeRedis.from_url(cfg.connection_url())
        clients.append(client)
        return client

    client = configure_redis_client(settings, factory=factory)
    assert isinstance(client, fakeredis.FakeRedis)
    assert clients and clients[0] is client
    assert client.ping() is True


def test_configure_redis_client_skips_when_unconfigured(caplog: pytest.LogCaptureFixture) -> None:
    settings = RedisConnectionSettings.from_mapping({})

    with caplog.at_level(logging.DEBUG):
        client = configure_redis_client(settings)

    assert client is None
    assert any("skipping client creation" in message for message in caplog.messages)


def test_redis_cache_policy_from_mapping() -> None:
    policy = RedisCachePolicy.from_mapping(
        {
            "REDIS_CACHE_TTL_SECONDS": "120",
            "REDIS_CACHE_MAX_KEYS": "256",
            "REDIS_CACHE_NAMESPACE": "emp:test",
            "REDIS_CACHE_INVALIDATE_PREFIXES": "alpha,beta",
        }
    )

    assert policy.ttl_seconds == 120
    assert policy.max_keys == 256
    assert policy.namespace == "emp:test"
    assert policy.invalidate_prefixes == ("alpha", "beta")
    assert policy.strategy == "institutional"


def test_redis_cache_policy_named_strategy_extended() -> None:
    policy = RedisCachePolicy.from_mapping({"REDIS_CACHE_STRATEGY": "extended"})

    assert policy.strategy == "extended"
    assert policy.ttl_seconds == 43_200
    assert policy.max_keys == 4_096
    assert policy.invalidate_prefixes == ()


def test_redis_cache_policy_strategy_alias_and_overrides() -> None:
    policy = RedisCachePolicy.from_mapping(
        {
            "REDIS_CACHE_STRATEGY": "12H",
            "REDIS_CACHE_TTL_SECONDS": "60",
            "REDIS_CACHE_MAX_KEYS": "32",
        }
    )

    assert policy.strategy == "extended"
    assert policy.ttl_seconds == 60
    assert policy.max_keys == 32


def test_redis_cache_policy_unknown_strategy_falls_back() -> None:
    fallback = RedisCachePolicy(ttl_seconds=123, max_keys=45, namespace="emp:fallback")
    policy = RedisCachePolicy.from_mapping(
        {"REDIS_CACHE_STRATEGY": "mystery"}, fallback=fallback
    )

    assert policy.ttl_seconds == 123
    assert policy.max_keys == 45
    assert policy.strategy == fallback.strategy


def test_managed_redis_cache_ttl_and_eviction() -> None:
    policy = RedisCachePolicy(ttl_seconds=5, max_keys=2, namespace="emp:test")
    current_time = {"now": 1_000.0}

    def fake_time() -> float:
        return current_time["now"]

    base = InMemoryRedis()
    cache = ManagedRedisCache(base, policy, time_fn=fake_time)

    cache.set("alpha", "1")
    cache.set("beta", "2")
    assert cache.get("alpha") == "1"

    current_time["now"] += 6.0
    assert cache.get("alpha") is None  # expired

    metrics = cache.metrics()
    assert metrics["expirations"] >= 1
    assert metrics["misses"] >= 1
    assert metrics["sets"] == 2
    assert metrics["keys"] <= 2
    assert metrics["ttl_seconds"] == 5
    assert metrics["max_keys"] == 2

    # Reinsert beta to ensure capacity pressure before triggering eviction
    cache.set("beta", "2b")
    cache.set("gamma", "3")
    cache.set("delta", "4")  # triggers eviction because max_keys=2

    metrics = cache.metrics()
    assert metrics["evictions"] >= 1
    assert cache.get("beta") is None or cache.get("gamma") is not None


def test_managed_redis_cache_metrics_reset() -> None:
    policy = RedisCachePolicy(ttl_seconds=5, max_keys=2, namespace="emp:test")
    cache = ManagedRedisCache(InMemoryRedis(), policy)

    cache.set("alpha", "1")
    snapshot = cache.metrics(reset=True)
    assert snapshot["sets"] == 1
    assert snapshot["hits"] == 0
    assert "strategy" in snapshot

    after_reset = cache.metrics()
    assert after_reset["sets"] == 0
    assert after_reset["hits"] == 0


def test_wrap_managed_cache_returns_existing_instance() -> None:
    policy = RedisCachePolicy.institutional_defaults()
    managed = wrap_managed_cache(InMemoryRedis(), policy=policy)
    wrapped_again = wrap_managed_cache(managed, policy=policy)

    assert wrapped_again is managed
