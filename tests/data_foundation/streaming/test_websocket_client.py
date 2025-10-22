from __future__ import annotations

import asyncio
import socket
import time

import pytest
from aiohttp import WSMessage, WSMsgType, web

from src.data_foundation.streaming.websocket_client import (
    WebSocketClient,
    WebSocketClientSettings,
    WebSocketSubscription,
)


def _allocate_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return sock.getsockname()[1]


async def _start_server(app: web.Application, port: int) -> web.AppRunner:
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", port)
    await site.start()
    return runner


@pytest.mark.asyncio
async def test_websocket_client_reconnects() -> None:
    app = web.Application()
    received: list[tuple[int, str, float]] = []

    async def handler(request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        connection_id = request.app.setdefault("connection_id", 0)
        request.app["connection_id"] = connection_id + 1

        async for message in ws:
            if message.type is WSMsgType.TEXT:
                received.append((connection_id, message.data, time.monotonic()))
                await ws.send_str(f"ack:{message.data}")
                if message.data == "trigger-close":
                    await ws.close()
                    break
        return ws

    app.router.add_get("/ws", handler)
    port = _allocate_port()
    runner = await _start_server(app, port)
    url = f"http://127.0.0.1:{port}/ws"

    settings = WebSocketClientSettings(
        url=url,
        heartbeat_interval=0.2,
        heartbeat_timeout=0.4,
        reconnect_backoff=(0.05, 0.1, 0.2),
        max_retries=5,
        rate_limit_per_second=20.0,
    )
    client = WebSocketClient(settings)

    stop_event = asyncio.Event()
    responses: list[str] = []

    async def on_connect(instance: WebSocketClient, is_reconnect: bool) -> None:
        if not is_reconnect:
            await instance.send_text("trigger-close")
        else:
            await instance.send_text("second")

    async def on_message(message: WSMessage) -> None:
        responses.append(message.data)
        if len([item for item in responses if item.startswith("ack:")]) >= 2:
            stop_event.set()

    try:
        await asyncio.wait_for(
            client.run(on_message, on_connect=on_connect, stop_event=stop_event),
            timeout=5.0,
        )
    finally:
        await runner.cleanup()

    assert [payload for _, payload, _ in received] == ["trigger-close", "second"]
    assert responses.count("ack:trigger-close") == 1
    assert responses.count("ack:second") == 1


@pytest.mark.asyncio
async def test_websocket_client_detects_heartbeat_timeouts() -> None:
    app = web.Application()
    received: list[str] = []

    async def handler(request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)

        async for message in ws:
            if message.type is WSMsgType.TEXT:
                if message.data == "__heartbeat__":
                    continue
                received.append(message.data)
                await ws.send_str(f"ack:{message.data}")
        return ws

    app.router.add_get("/ws", handler)
    port = _allocate_port()
    runner = await _start_server(app, port)
    url = f"http://127.0.0.1:{port}/ws"

    settings = WebSocketClientSettings(
        url=url,
        heartbeat_interval=0.2,
        heartbeat_timeout=0.3,
        reconnect_backoff=(0.05, 0.1, 0.2),
        max_retries=5,
        rate_limit_per_second=20.0,
    )
    client = WebSocketClient(settings)

    stop_event = asyncio.Event()
    responses: list[str] = []

    async def on_connect(instance: WebSocketClient, is_reconnect: bool) -> None:
        await instance.send_text("hello" if not is_reconnect else "hello-again")

    async def on_message(message: WSMessage) -> None:
        responses.append(message.data)
        if responses.count("ack:hello") and responses.count("ack:hello-again"):
            stop_event.set()

    try:
        await asyncio.wait_for(
            client.run(on_message, on_connect=on_connect, stop_event=stop_event),
            timeout=6.0,
        )
    finally:
        await runner.cleanup()

    assert received.count("hello") >= 1
    assert received.count("hello-again") >= 1
    assert responses.count("ack:hello") == 1
    assert responses.count("ack:hello-again") == 1


@pytest.mark.asyncio
async def test_websocket_client_rate_limiting() -> None:
    app = web.Application()
    received: list[tuple[str, float]] = []

    async def handler(request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)

        async for message in ws:
            if message.type is WSMsgType.TEXT:
                timestamp = time.monotonic()
                received.append((message.data, timestamp))
                await ws.send_str(f"ack:{message.data}")
                if message.data == "third":
                    await ws.close()
                    break
        return ws

    app.router.add_get("/ws", handler)
    port = _allocate_port()
    runner = await _start_server(app, port)
    url = f"http://127.0.0.1:{port}/ws"

    settings = WebSocketClientSettings(
        url=url,
        heartbeat_interval=1.0,
        heartbeat_timeout=2.0,
        reconnect_backoff=(0.05,),
        max_retries=1,
        rate_limit_per_second=2.0,
    )
    client = WebSocketClient(settings)

    stop_event = asyncio.Event()

    async def on_connect(instance: WebSocketClient, is_reconnect: bool) -> None:
        if not is_reconnect:
            await instance.send_text("first")
            await instance.send_text("second")
            await instance.send_text("third")

    async def on_message(message: WSMessage) -> None:
        if message.data == "ack:third":
            stop_event.set()

    try:
        await asyncio.wait_for(
            client.run(on_message, on_connect=on_connect, stop_event=stop_event),
            timeout=6.0,
        )
    finally:
        await runner.cleanup()

    assert [payload for payload, _ in received] == ["first", "second", "third"]
    assert len(received) == 3
    timestamps = [ts for _, ts in received]
    deltas = [later - earlier for earlier, later in zip(timestamps, timestamps[1:])]
    assert len(deltas) == 2
    assert deltas[1] >= 0.45


@pytest.mark.asyncio
async def test_websocket_client_replays_subscriptions_on_reconnect() -> None:
    app = web.Application()
    received_messages: list[tuple[int, str]] = []
    responses: list[str] = []

    async def handler(request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)

        connection_id = request.app.setdefault("connection_id", 0)
        request.app["connection_id"] = connection_id + 1
        subscribed = False

        async for message in ws:
            if message.type is WSMsgType.TEXT:
                received_messages.append((connection_id, message.data))
                if message.data == "subscribe:alpha":
                    subscribed = True
                    await ws.send_str(f"ack:{connection_id}")
                    if connection_id == 0:
                        await ws.close()
                        break
                elif message.data == "trigger" and subscribed:
                    await ws.send_str("data:alpha")
                elif message.data == "stop":
                    await ws.close()
                    break
        return ws

    app.router.add_get("/ws", handler)
    port = _allocate_port()
    runner = await _start_server(app, port)
    url = f"http://127.0.0.1:{port}/ws"

    settings = WebSocketClientSettings(
        url=url,
        heartbeat_interval=0.2,
        heartbeat_timeout=0.4,
        reconnect_backoff=(0.05, 0.1),
        max_retries=5,
    )
    client = WebSocketClient(settings)
    await client.register_subscription(WebSocketSubscription.text("alpha", "subscribe:alpha"))

    stop_event = asyncio.Event()

    connect_states: list[bool] = []

    async def on_connect(instance: WebSocketClient, is_reconnect: bool) -> None:
        connect_states.append(is_reconnect)

    async def on_message(message: WSMessage) -> None:
        responses.append(message.data)
        if message.data == "data:alpha":
            await client.send_text("stop")
            stop_event.set()
        elif message.data == "ack:1":
            await client.send_text("trigger")

    try:
        await asyncio.wait_for(
            client.run(on_message, on_connect=on_connect, stop_event=stop_event),
            timeout=6.0,
        )
    finally:
        await runner.cleanup()

    connection_ids = [cid for cid, payload in received_messages if payload == "subscribe:alpha"]
    assert connection_ids == [0, 1]
    assert responses.count("ack:0") == 1
    assert responses.count("ack:1") == 1
    assert "data:alpha" in responses
    assert connect_states == [False, True]

