from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from src.data_foundation import CallableConnector, MarketDataFabric


@pytest.mark.asyncio
async def test_fabric_prefers_higher_priority_and_caches() -> None:
    calls: list[str] = []

    async def primary(symbol: str, _: datetime | None) -> dict[str, Any]:
        calls.append("primary")
        return {
            "symbol": symbol,
            "bid": 99.5,
            "ask": 100.5,
            "volume": 10_000,
            "timestamp": datetime.now(timezone.utc),
        }

    async def secondary(symbol: str, _: datetime | None) -> dict[str, Any]:
        calls.append("secondary")
        return {
            "symbol": symbol,
            "bid": 10.0,
            "ask": 11.0,
            "timestamp": datetime.now(timezone.utc),
        }

    fabric = MarketDataFabric(
        {
            "primary": CallableConnector(name="primary", func=primary, priority=0),
            "secondary": CallableConnector(name="secondary", func=secondary, priority=10),
        },
        cache_ttl=timedelta(seconds=60),
    )

    reading = await fabric.fetch_latest("AAPL")
    assert reading.symbol == "AAPL"
    assert calls == ["primary"]

    # Cache hit should avoid additional connector calls
    cached = await fabric.fetch_latest("AAPL")
    assert cached is reading
    assert calls == ["primary"]

    diagnostics = fabric.get_diagnostics()
    assert diagnostics["telemetry"]["total_requests"] == 2
    assert diagnostics["telemetry"]["cache_hits"] == 1
    assert diagnostics["telemetry"]["success"]["primary"] == 1


@pytest.mark.asyncio
async def test_fabric_falls_back_on_connector_failure() -> None:
    async def failing(_: str, __: datetime | None) -> None:
        raise RuntimeError("boom")

    async def fallback(symbol: str, _: datetime | None) -> dict[str, Any]:
        return {
            "symbol": symbol,
            "price": 123.0,
            "timestamp": datetime.now(timezone.utc),
        }

    fabric = MarketDataFabric(
        {
            "failing": CallableConnector(name="failing", func=failing, priority=0),
            "fallback": CallableConnector(name="fallback", func=fallback, priority=1),
        },
        cache_ttl=timedelta(seconds=1),
    )

    reading = await fabric.fetch_latest("MSFT")
    assert reading.symbol == "MSFT"
    diagnostics = fabric.get_diagnostics()
    assert diagnostics["telemetry"]["success"]["fallback"] == 1
    assert diagnostics["telemetry"]["failures"]["failing"] == 1


@pytest.mark.asyncio
async def test_fabric_can_return_stale_data_when_all_connectors_fail() -> None:
    emitted = {
        "symbol": "NVDA",
        "price": 456.7,
        "timestamp": datetime.now(timezone.utc),
    }

    async def good_once(symbol: str, _: datetime | None) -> dict[str, Any]:
        if emitted.get("used"):
            raise RuntimeError("exhausted")
        emitted["used"] = True
        return emitted

    fabric = MarketDataFabric(
        {"degenerate": CallableConnector(name="degenerate", func=good_once, priority=0)},
        cache_ttl=timedelta(seconds=0),
    )

    first = await fabric.fetch_latest("NVDA")
    assert first.symbol == "NVDA"

    # Force cache expiry
    await asyncio.sleep(0.01)

    stale = await fabric.fetch_latest("NVDA")
    assert stale.symbol == "NVDA"
    assert stale is first
    diagnostics = fabric.get_diagnostics()
    assert diagnostics["telemetry"]["total_requests"] == 2
    assert diagnostics["telemetry"]["failures"]["degenerate"] >= 1
    assert diagnostics["cache"]["NVDA"]["source"] == "degenerate"


@pytest.mark.asyncio
async def test_fetch_latest_can_bypass_cache_and_invalidate() -> None:
    call_count = 0

    async def connector(symbol: str, _: datetime | None) -> dict[str, Any]:
        nonlocal call_count
        call_count += 1
        return {
            "symbol": symbol,
            "price": 100.0 + call_count,
            "timestamp": datetime.now(timezone.utc),
        }

    fabric = MarketDataFabric(
        {"primary": CallableConnector(name="primary", func=connector, priority=0)},
        cache_ttl=timedelta(seconds=60),
    )

    first = await fabric.fetch_latest("ETHUSD")
    assert call_count == 1

    second = await fabric.fetch_latest("ETHUSD", use_cache=False)
    assert call_count == 2
    assert second.close != first.close

    fabric.invalidate_cache("ETHUSD")
    third = await fabric.fetch_latest("ETHUSD")
    assert call_count == 3
    assert third.symbol == "ETHUSD"
