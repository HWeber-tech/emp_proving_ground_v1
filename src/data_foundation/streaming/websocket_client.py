"""High-resilience WebSocket client with reconnection, heartbeats and throttling."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from dataclasses import dataclass
from typing import Awaitable, Callable, Mapping, Sequence

import aiohttp
from aiohttp import ClientSession, ClientWebSocketResponse, WSMessage, WSMsgType

__all__ = [
    "WebSocketClient",
    "WebSocketClientError",
    "WebSocketClientSettings",
    "WebSocketConnectionError",
    "WebSocketHeartbeatError",
]

logger = logging.getLogger(__name__)

MessageHandler = Callable[[WSMessage], Awaitable[None]]
ConnectCallback = Callable[["WebSocketClient", bool], Awaitable[None] | None]


class WebSocketClientError(RuntimeError):
    """Base class for WebSocket client exceptions."""


class WebSocketConnectionError(WebSocketClientError):
    """Raised when the client cannot establish or maintain a connection."""


class WebSocketHeartbeatError(WebSocketClientError):
    """Raised when the server stops acknowledging heartbeats."""


@dataclass(frozen=True, slots=True)
class WebSocketClientSettings:
    """Configuration container for :class:`WebSocketClient`."""

    url: str
    headers: Mapping[str, str] | None = None
    params: Mapping[str, str] | None = None
    reconnect_backoff: Sequence[float] = (1.0, 2.0, 5.0)
    max_retries: int | None = None
    heartbeat_interval: float = 30.0
    heartbeat_timeout: float = 10.0
    heartbeat_payload: str = "__heartbeat__"
    receive_timeout: float | None = 30.0
    rate_limit_per_second: float | None = 50.0
    rate_limit_capacity: float | None = None
    connect_timeout: float = 10.0


class _AsyncRateLimiter:
    """Simple token bucket rate limiter for async send operations."""

    def __init__(self, rate: float, capacity: float | None = None) -> None:
        if rate <= 0:
            raise ValueError("rate must be positive")
        self._rate = float(rate)
        self._capacity = float(capacity) if capacity and capacity > 0 else float(rate)
        self._tokens = self._capacity
        self._updated = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        while True:
            async with self._lock:
                now = time.monotonic()
                elapsed = now - self._updated
                if elapsed > 0:
                    self._tokens = min(
                        self._capacity, self._tokens + elapsed * self._rate
                    )
                    self._updated = now
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return
                deficit = 1.0 - self._tokens
                wait_time = deficit / self._rate
            await asyncio.sleep(wait_time)


class WebSocketClient:
    """Robust asynchronous WebSocket client with reconnection support."""

    def __init__(
        self,
        settings: WebSocketClientSettings,
        *,
        session_factory: Callable[[], ClientSession] | None = None,
    ) -> None:
        self._settings = settings
        self._session_factory = session_factory
        self._session: ClientSession | None = None
        self._ws: ClientWebSocketResponse | None = None
        self._rate_limiter = (
            _AsyncRateLimiter(settings.rate_limit_per_second, settings.rate_limit_capacity)
            if settings.rate_limit_per_second
            else None
        )
        self._last_received = time.monotonic()
        self._heartbeat_task: asyncio.Task[None] | None = None
        self._closed = False
        self._lock = asyncio.Lock()

    @property
    def closed(self) -> bool:
        return self._closed

    async def run(
        self,
        handler: MessageHandler,
        *,
        on_connect: ConnectCallback | None = None,
        stop_event: asyncio.Event | None = None,
    ) -> None:
        """Run the client loop until ``stop_event`` is set."""

        if self._closed:
            raise RuntimeError("WebSocketClient is closed")

        reconnect_attempts = 0
        is_reconnect = False

        while True:
            if stop_event and stop_event.is_set():
                break
            try:
                ws = await self._connect()
            except WebSocketClientError:
                reconnect_attempts += 1
                await self._handle_retry(reconnect_attempts)
                continue

            reconnect_attempts = 0
            try:
                if on_connect is not None:
                    await _maybe_await(on_connect(self, is_reconnect))
                is_reconnect = True
                await self._consume(ws, handler, stop_event)
                break
            except asyncio.CancelledError:
                raise
            except WebSocketClientError as exc:
                logger.warning("WebSocket connection interrupted: %s", exc)
                reconnect_attempts += 1
                await self._handle_retry(reconnect_attempts)
                continue

        await self.close()

    async def send_text(self, payload: str) -> None:
        await self._send(lambda ws: ws.send_str(payload))

    async def send_json(self, payload: object) -> None:
        await self._send(lambda ws: ws.send_json(payload))

    async def close(self) -> None:
        async with self._lock:
            if self._closed:
                return
            self._closed = True
        await self._stop_heartbeat()
        await self._close_ws()
        if self._session is not None:
            await self._session.close()
            self._session = None

    async def _ensure_session(self) -> ClientSession:
        if self._session is not None:
            return self._session
        if self._session_factory is not None:
            session = self._session_factory()
        else:
            timeout = aiohttp.ClientTimeout(total=self._settings.connect_timeout)
            session = aiohttp.ClientSession(timeout=timeout)
        self._session = session
        return session

    async def _connect(self) -> ClientWebSocketResponse:
        session = await self._ensure_session()
        try:
            ws = await session.ws_connect(
                self._settings.url,
                headers=self._settings.headers,
                params=self._settings.params,
                autoping=True,
                heartbeat=None,
            )
        except aiohttp.ClientError as exc:
            raise WebSocketConnectionError("Failed to open websocket") from exc
        self._ws = ws
        self._last_received = time.monotonic()
        return ws

    async def _consume(
        self,
        ws: ClientWebSocketResponse,
        handler: MessageHandler,
        stop_event: asyncio.Event | None,
    ) -> None:
        self._start_heartbeat(ws)
        try:
            while True:
                if stop_event and stop_event.is_set():
                    return
                try:
                    message = (
                        await ws.receive(timeout=self._settings.receive_timeout)
                        if self._settings.receive_timeout is not None
                        else await ws.receive()
                    )
                except asyncio.TimeoutError as exc:
                    raise WebSocketConnectionError("WebSocket receive timed out") from exc

                if message.type is WSMsgType.TEXT or message.type is WSMsgType.BINARY:
                    self._last_received = time.monotonic()
                    await handler(message)
                elif message.type is WSMsgType.PING:
                    self._last_received = time.monotonic()
                    await ws.pong(message.data)
                elif message.type is WSMsgType.PONG:
                    self._last_received = time.monotonic()
                elif message.type in {WSMsgType.CLOSE, WSMsgType.CLOSING, WSMsgType.CLOSED}:
                    raise WebSocketConnectionError("WebSocket closed")
                elif message.type is WSMsgType.ERROR:
                    raise WebSocketConnectionError("WebSocket errored") from ws.exception()

                if self._heartbeat_task and self._heartbeat_task.done():
                    exc = self._heartbeat_task.exception()
                    if exc is not None:
                        if isinstance(exc, WebSocketClientError):
                            raise exc
                        raise WebSocketClientError("Heartbeat task failed") from exc
        finally:
            await self._stop_heartbeat()
            await self._close_ws()

    async def _send(
        self, sender: Callable[[ClientWebSocketResponse], Awaitable[None]]
    ) -> None:
        ws = self._ws
        if ws is None:
            raise WebSocketConnectionError("WebSocket not connected")
        if self._rate_limiter is not None:
            await self._rate_limiter.acquire()
        try:
            await sender(ws)
        except aiohttp.ClientError as exc:
            raise WebSocketConnectionError("Failed to send on websocket") from exc

    def _start_heartbeat(self, ws: ClientWebSocketResponse) -> None:
        if self._settings.heartbeat_interval <= 0:
            return
        if self._heartbeat_task is not None and not self._heartbeat_task.done():
            self._heartbeat_task.cancel()
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop(ws))

    async def _stop_heartbeat(self) -> None:
        if self._heartbeat_task is None:
            return
        self._heartbeat_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._heartbeat_task
        self._heartbeat_task = None

    async def _heartbeat_loop(self, ws: ClientWebSocketResponse) -> None:
        try:
            while True:
                await asyncio.sleep(self._settings.heartbeat_interval)
                if ws.closed:
                    return
                await self._send_heartbeat(ws)
                if self._settings.heartbeat_timeout <= 0:
                    continue
                await asyncio.sleep(self._settings.heartbeat_timeout)
                if time.monotonic() - self._last_received > self._settings.heartbeat_timeout:
                    raise WebSocketHeartbeatError("Heartbeat acknowledgement timeout")
        except asyncio.CancelledError:
            raise

    async def _send_heartbeat(self, ws: ClientWebSocketResponse) -> None:
        payload = self._settings.heartbeat_payload
        if payload:
            await self._send(lambda inner_ws: inner_ws.send_str(payload))
        else:
            await self._send(lambda inner_ws: inner_ws.ping())

    async def _close_ws(self) -> None:
        ws = self._ws
        self._ws = None
        if ws is None:
            return
        await ws.close()

    async def _handle_retry(self, attempt: int) -> None:
        max_retries = self._settings.max_retries
        if max_retries is not None and attempt > max_retries:
            raise WebSocketConnectionError("Maximum reconnection attempts exceeded")
        delay = self._backoff_delay(attempt)
        if delay > 0:
            await asyncio.sleep(delay)

    def _backoff_delay(self, attempt: int) -> float:
        backoff = self._settings.reconnect_backoff
        if not backoff:
            return 0.0
        index = min(max(attempt - 1, 0), len(backoff) - 1)
        return max(backoff[index], 0.0)


async def _maybe_await(result: Awaitable[None] | None) -> None:
    if result is None:
        return
    if asyncio.iscoroutine(result):
        await result
    else:
        # Support callables returning ``asyncio.Future``
        try:
            await asyncio.ensure_future(result)
        except TypeError:
            return

