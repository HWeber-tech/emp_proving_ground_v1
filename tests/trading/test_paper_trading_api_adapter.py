import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

import pytest
from aiohttp import web

from src.trading.integration.paper_trading_api import (
    PaperTradingApiAdapter,
    PaperTradingApiError,
    PaperTradingApiSettings,
)


async def _start_test_server(handler: Callable[[web.Request], Any]) -> tuple[str, Callable[[], Awaitable[None]]]:
    app = web.Application()
    app.router.add_post("/orders", handler)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    sockets = site._server.sockets  # type: ignore[attr-defined]
    assert sockets, "Server failed to expose sockets"
    port = sockets[0].getsockname()[1]
    base_url = f"http://127.0.0.1:{port}"

    async def _shutdown() -> None:
        await runner.cleanup()

    return base_url, _shutdown


@pytest.mark.asyncio
async def test_paper_trading_api_adapter_places_market_order() -> None:
    captured: list[dict[str, Any]] = []

    async def handler(request: web.Request) -> web.Response:
        payload = await request.json()
        captured.append(payload)
        return web.json_response({"order_id": "ORD-123"})

    base_url, shutdown = await _start_test_server(handler)

    settings = PaperTradingApiSettings(
        base_url=base_url,
        api_key="demo-key",
        api_secret="demo-secret",
        api_key_header="X-KEY",
        api_secret_header="X-SECRET",
        order_endpoint="/orders",
        order_id_field="order_id",
        time_in_force="gtc",
        verify_ssl=False,
        request_timeout=2.0,
        client_order_prefix="alpha-demo",
    )
    adapter = PaperTradingApiAdapter(settings=settings)

    try:
        order_id = await adapter.place_market_order("eurusd", "buy", 1.5)
        assert order_id == "ORD-123"
        assert captured, "Adapter did not submit the order"
        submitted = captured[0]
        assert submitted["symbol"] == "EURUSD"
        assert submitted["side"] == "buy"
        assert submitted["type"] == "market"
        assert "client_order_id" in submitted
        snapshot = adapter.describe_last_submission()
        assert snapshot is not None
        assert snapshot.get("attempt") == 1
        assert snapshot.get("response", {}).get("order_id") == "ORD-123"
        assert snapshot.get("request", {}).get("url")
    finally:
        await adapter.close()
        await shutdown()


@pytest.mark.asyncio
async def test_paper_trading_api_adapter_raises_on_http_error() -> None:
    async def handler(_request: web.Request) -> web.Response:
        return web.Response(status=503, text="Service unavailable")

    base_url, shutdown = await _start_test_server(handler)

    settings = PaperTradingApiSettings(
        base_url=base_url,
        order_endpoint="/orders",
        order_id_field="order_id",
        verify_ssl=False,
        request_timeout=1.0,
    )
    adapter = PaperTradingApiAdapter(settings=settings)

    try:
        with pytest.raises(PaperTradingApiError):
            await adapter.place_market_order("GBPUSD", "sell", 2.0)
        snapshot = adapter.describe_last_submission()
        assert snapshot is not None
        assert snapshot.get("attempt") == settings.retry_attempts
        response = snapshot.get("response", {})
        assert response.get("status") == 503
        assert "Service unavailable" in response.get("body_text", "")
    finally:
        await adapter.close()
        await shutdown()


@pytest.mark.asyncio
async def test_paper_trading_api_adapter_retries_and_succeeds() -> None:
    call_counter = {"count": 0}

    async def handler(request: web.Request) -> web.Response:
        call_counter["count"] += 1
        if call_counter["count"] == 1:
            return web.Response(status=502, text="temporary failure")
        payload = await request.json()
        assert payload["side"] == "buy"
        return web.json_response({"order_id": "ORD-RETRY"})

    base_url, shutdown = await _start_test_server(handler)

    settings = PaperTradingApiSettings(
        base_url=base_url,
        order_endpoint="/orders",
        order_id_field="order_id",
        verify_ssl=False,
        request_timeout=1.0,
        retry_attempts=3,
        retry_backoff_seconds=0.0,
    )
    adapter = PaperTradingApiAdapter(settings=settings)
    order_id: str | None = None

    try:
        order_id = await adapter.place_market_order("EURUSD", "buy", 1.0)
    finally:
        await adapter.close()
        await shutdown()

    assert order_id == "ORD-RETRY"
    assert call_counter["count"] == 2
    snapshot = adapter.describe_last_submission()
    assert snapshot is not None
    assert snapshot.get("attempt") == 2
    response = snapshot.get("response", {})
    assert response.get("status") == 200
    assert response.get("order_id") == "ORD-RETRY"
    request = snapshot.get("request", {})
    assert request.get("payload", {}).get("symbol") == "EURUSD"
    assert "headers" in request


@pytest.mark.asyncio
async def test_paper_trading_api_adapter_retries_and_raises_after_limit() -> None:
    call_counter = {"count": 0}

    async def handler(_request: web.Request) -> web.Response:
        call_counter["count"] += 1
        return web.Response(status=500, text="boom")

    base_url, shutdown = await _start_test_server(handler)

    settings = PaperTradingApiSettings(
        base_url=base_url,
        order_endpoint="/orders",
        order_id_field="order_id",
        verify_ssl=False,
        request_timeout=1.0,
        retry_attempts=2,
        retry_backoff_seconds=0.0,
    )
    adapter = PaperTradingApiAdapter(settings=settings)

    try:
        with pytest.raises(PaperTradingApiError) as excinfo:
            await adapter.place_market_order("EURUSD", "sell", 1.0)
    finally:
        await adapter.close()
        await shutdown()

    assert call_counter["count"] == 2
    assert "2 attempts" in str(excinfo.value)
