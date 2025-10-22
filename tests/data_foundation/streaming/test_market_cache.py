from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from src.data_foundation.monitoring.feed_anomaly import Tick
from src.data_foundation.streaming.market_cache import InMemoryRedis, MarketDataCache


def _sample_tick(offset_seconds: int) -> Tick:
    base = datetime(2024, 1, 1, tzinfo=UTC)
    return Tick(timestamp=base + timedelta(seconds=offset_seconds), price=100.0 + offset_seconds)


def test_store_and_fetch_latest_tick() -> None:
    cache = MarketDataCache(redis_client=InMemoryRedis(), window_size=5)
    tick = _sample_tick(1)
    cache.store_tick("AAPL", tick)

    latest = cache.get_latest_tick("AAPL")
    assert latest is not None
    assert latest.price == pytest.approx(tick.price)
    assert latest.timestamp == tick.timestamp


def test_sliding_window_retains_most_recent_entries() -> None:
    cache = MarketDataCache(redis_client=InMemoryRedis(), window_size=3)
    ticks = [_sample_tick(i) for i in range(5)]

    cache.store_ticks("MSFT", ticks)

    recent = cache.get_recent_ticks("MSFT")
    assert len(recent) == 3
    assert [tick.price for tick in recent] == [104.0, 103.0, 102.0]


def test_accepts_mapping_payloads() -> None:
    cache = MarketDataCache(redis_client=InMemoryRedis(), window_size=2)
    cache.store_tick(
        "EURUSD",
        {
            "timestamp": datetime(2024, 1, 2, tzinfo=UTC).isoformat(),
            "price": 1.2345,
            "volume": 10.0,
            "seqno": 42,
        },
    )

    latest = cache.get_latest_tick("EURUSD")
    assert latest is not None
    assert latest.price == pytest.approx(1.2345)
    assert latest.volume == pytest.approx(10.0)
    assert latest.seqno == 42


def test_cache_warming_respects_window_size() -> None:
    warm_data = {
        "BTCUSD": tuple(_sample_tick(i) for i in range(6)),
        "ETHUSD": tuple(
            {
                "timestamp": (datetime(2024, 2, 1, tzinfo=UTC) + timedelta(seconds=i)).isoformat(),
                "price": 2000 + i,
            }
            for i in range(4)
        ),
    }

    cache = MarketDataCache(redis_client=InMemoryRedis(), window_size=3, warm_start=warm_data)

    btc_ticks = cache.get_recent_ticks("BTCUSD")
    assert len(btc_ticks) == 3
    assert [tick.price for tick in btc_ticks] == [105.0, 104.0, 103.0]

    eth_ticks = cache.get_recent_ticks("ETHUSD")
    assert len(eth_ticks) == 3
    assert [tick.price for tick in eth_ticks] == [2003, 2002, 2001]


def test_cache_warm_loader_callable() -> None:
    def loader() -> dict[str, tuple[Tick, ...]]:
        return {"AAPL": (_sample_tick(0),)}

    cache = MarketDataCache(redis_client=InMemoryRedis(), window_size=5, warm_start=loader)

    latest = cache.get_latest_tick("AAPL")
    assert latest is not None
    assert latest.price == pytest.approx(100.0)


def test_ttl_eviction_removes_expired_keys() -> None:
    now = datetime(2024, 3, 1, tzinfo=UTC)
    clock_holder = {"now": now}

    def clock() -> datetime:
        return clock_holder["now"]

    redis = InMemoryRedis(clock=clock)
    cache = MarketDataCache(redis_client=redis, window_size=5, ttl_seconds=60)

    cache.store_tick("ETHUSD", _sample_tick(1))
    assert cache.get_latest_tick("ETHUSD") is not None

    clock_holder["now"] = now + timedelta(seconds=30)
    assert cache.get_latest_tick("ETHUSD") is not None

    clock_holder["now"] = now + timedelta(seconds=61)
    assert cache.get_latest_tick("ETHUSD") is None
    assert redis.lrange("market:ETHUSD", 0, -1) == []


def test_zero_ttl_behaves_like_ephemeral_cache() -> None:
    cache = MarketDataCache(redis_client=InMemoryRedis(), window_size=5, ttl_seconds=0)

    cache.store_tick("BTCUSD", _sample_tick(0))
    assert cache.get_recent_ticks("BTCUSD") == ()
