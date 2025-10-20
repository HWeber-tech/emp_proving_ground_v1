from __future__ import annotations

import time

import pytest

from src.core.performance.market_data_cache import MarketDataCache, _InMemoryCache


def test_in_memory_cache_without_expiry(monkeypatch) -> None:
    cache = _InMemoryCache()
    cache.set("symbol", {"foo": "bar"}, ttl_seconds=None)

    future_time = time.time() + 10_000
    monkeypatch.setattr(
        "src.core.performance.market_data_cache.time.time", lambda: future_time
    )

    assert cache.get("symbol") == {"foo": "bar"}


def test_market_data_cache_without_expiry(monkeypatch) -> None:
    cache = MarketDataCache()
    cache.set("key", "value", ttl_seconds=None)

    future_time = time.time() + 10_000
    monkeypatch.setattr(
        "src.core.performance.market_data_cache.time.time", lambda: future_time
    )

    assert cache.get("key") == "value"


def test_in_memory_cache_zero_ttl_expires_immediately(monkeypatch) -> None:
    cache = _InMemoryCache()
    now = 1_000.0
    monkeypatch.setattr(
        "src.core.performance.market_data_cache.time.time", lambda: now
    )

    cache.set("key", "value", ttl_seconds=0)

    assert cache.get("key") is None


def test_market_data_cache_zero_ttl_expires_immediately(monkeypatch) -> None:
    cache = MarketDataCache()
    now = 2_000.0
    monkeypatch.setattr(
        "src.core.performance.market_data_cache.time.time", lambda: now
    )

    cache.set("key", "value", ttl_seconds=0)

    assert cache.get("key") is None


def test_in_memory_cache_rejects_nan_ttl() -> None:
    cache = _InMemoryCache()

    with pytest.raises(ValueError, match="ttl_seconds"):
        cache.set("key", "value", ttl_seconds=float("nan"))


def test_market_data_cache_rejects_nan_ttl() -> None:
    cache = MarketDataCache()

    with pytest.raises(ValueError, match="ttl_seconds"):
        cache.set("key", "value", ttl_seconds=float("nan"))
