"""
FIX Connection Manager Adapter
==============================

Provides a compatibility layer expected by the main system, wrapping the
genuine IC Markets FIX implementation in `src.operational.icmarkets_api`.

Responsibilities:
- Start/stop price and trade sessions
- Bridge market data and order updates into asyncio queues expected by
  sensory and broker components
- Expose `get_application(<price|trade>)` with `set_message_queue(queue)`
- Expose `get_initiator("trade")` with `send_message(msg)` for order flow
"""

from __future__ import annotations

import asyncio
import importlib
import logging
import os
from collections.abc import Callable, Sequence
from typing import Optional, Protocol, SupportsFloat, TypedDict, cast

from src.operational.mock_fix import MockFIXManager


class FIXMarketDataEntry(TypedDict):
    """Normalized FIX-style market data entry."""

    type: bytes
    px: float
    size: float


FIXMessage = dict[int | bytes, bytes | list[FIXMarketDataEntry]]


class OrderExecutionRecord(Protocol):
    """Minimal mapping interface used when inspecting executions."""

    def get(self, key: str, default: str | bytes | None = None) -> str | bytes | None: ...


class OrderInfoProtocol(Protocol):
    """Subset of ``OrderInfo`` attributes consumed by the connection manager."""

    cl_ord_id: str
    executions: Sequence[OrderExecutionRecord]


class OrderBookLevelProtocol(Protocol):
    """Expose price/size pairs from FIX managers."""

    price: SupportsFloat
    size: SupportsFloat


class OrderBookProtocol(Protocol):
    """Container for bid/ask ladders."""

    bids: Sequence[OrderBookLevelProtocol]
    asks: Sequence[OrderBookLevelProtocol]


class FIXTradeConnectionProtocol(Protocol):
    """Trade connection contract used for sending messages."""

    def send_message_and_track(self, msg: object) -> bool: ...


class FIXManagerProtocol(Protocol):
    """Protocol implemented by both genuine and mock FIX managers."""

    trade_connection: FIXTradeConnectionProtocol | None

    def add_market_data_callback(self, cb: Callable[[str, OrderBookProtocol], None]) -> None: ...

    def add_order_callback(self, cb: Callable[[OrderInfoProtocol], None]) -> None: ...

    def start(self) -> bool: ...

    def stop(self) -> None: ...


class ICMarketsConfigLike(Protocol):
    """Configuration interface required to construct the genuine FIX manager."""

    def __init__(self, *, environment: str, account_number: str) -> None: ...

    environment: str
    account_number: str
    password: str | None


class FIXManagerFactory(Protocol):
    def __call__(self, config: ICMarketsConfigLike) -> FIXManagerProtocol: ...


class SystemConfigProtocol(Protocol):
    """Subset of system config attributes referenced in this adapter."""

    environment: str
    account_number: str
    password: str | None

    def __getattr__(self, item: str) -> object: ...


def _load_genuine_components() -> tuple[type[object] | None, type[object] | None]:
    """Attempt to import the genuine FIX manager and config classes."""

    try:
        api_module = importlib.import_module("src.operational.icmarkets_api")
        config_module = importlib.import_module("src.operational.icmarkets_config")
    except Exception:  # pragma: no cover - optional dependency
        return (None, None)

    manager_cls = getattr(api_module, "GenuineFIXManager", None)
    config_cls = getattr(config_module, "ICMarketsConfig", None)
    if not isinstance(manager_cls, type) or not isinstance(config_cls, type):
        return (None, None)
    return manager_cls, config_cls


_GENUINE_MANAGER_CLASS, _IC_CONFIG_CLASS = _load_genuine_components()

logger = logging.getLogger(__name__)


class _FIXApplicationAdapter:
    """Adapter that forwards incoming messages into an asyncio.Queue."""

    def __init__(self, session_type: str) -> None:
        self._session_type = session_type
        self._queue: asyncio.Queue[FIXMessage] | None = None

    def set_message_queue(self, queue: asyncio.Queue[FIXMessage]) -> None:
        self._queue = queue

    def dispatch(self, message: FIXMessage) -> None:
        queue = self._queue
        if queue is None:
            return

        try:
            queue.put_nowait(message)
        except asyncio.QueueFull:
            logger.warning(
                "FIX %s queue is full; dropping message", self._session_type
            )


class _FIXInitiatorAdapter:
    """Adapter exposing a ``send_message`` API to send trade messages."""

    def __init__(self, manager: FIXManagerProtocol) -> None:
        self._manager = manager

    def send_message(self, msg: object) -> bool:
        trade_connection = self._manager.trade_connection
        if trade_connection is None:
            logger.error("Trade connection not initialized")
            return False
        return trade_connection.send_message_and_track(msg)


class FIXConnectionManager:
    """Compatibility wrapper expected by ``main.py``."""

    def __init__(self, system_config: SystemConfigProtocol) -> None:
        self._system_config = system_config
        self._manager: FIXManagerProtocol | None = None
        self._price_app = _FIXApplicationAdapter(session_type="quote")
        self._trade_app = _FIXApplicationAdapter(session_type="trade")
        self._initiator: _FIXInitiatorAdapter | None = None

    def start_sessions(self) -> bool:
        """Create and start genuine FIX sessions."""
        try:
            # Prefer real FIX if credentials exist and genuine manager is available,
            # unless explicitly forced to mock via env.
            force_mock = os.environ.get("EMP_USE_MOCK_FIX", "0") in ("1", "true", "True")
            creds_present = all(
                bool(os.environ.get(k) or getattr(self._system_config, k.lower(), None))
                for k in (
                    "FIX_PRICE_SENDER_COMP_ID",
                    "FIX_PRICE_USERNAME",
                    "FIX_PRICE_PASSWORD",
                    "FIX_TRADE_SENDER_COMP_ID",
                    "FIX_TRADE_USERNAME",
                    "FIX_TRADE_PASSWORD",
                )
            )
            use_mock = bool(
                force_mock
                or not creds_present
                or _GENUINE_MANAGER_CLASS is None
                or _IC_CONFIG_CLASS is None
            )

            manager: FIXManagerProtocol
            if use_mock:
                manager = cast(FIXManagerProtocol, MockFIXManager())
            else:
                config_factory = cast(type[ICMarketsConfigLike], _IC_CONFIG_CLASS)
                ic_cfg = config_factory(
                    environment=self._system_config.environment,
                    account_number=self._system_config.account_number,
                )
                ic_cfg.password = getattr(self._system_config, "password", None)
                manager_factory = cast(FIXManagerFactory, _GENUINE_MANAGER_CLASS)
                manager = manager_factory(ic_cfg)

            # Bridge market data: convert order book updates to queue-friendly messages
            def on_market_data(symbol: str, order_book: OrderBookProtocol) -> None:
                try:
                    entries: list[FIXMarketDataEntry] = []
                    for bid in getattr(order_book, "bids", [])[:10]:
                        try:
                            entries.append(
                                FIXMarketDataEntry(
                                    type=b"0",
                                    px=float(bid.price),
                                    size=float(bid.size),
                                )
                            )
                        except (TypeError, ValueError, AttributeError):
                            continue
                    for ask in getattr(order_book, "asks", [])[:10]:
                        try:
                            entries.append(
                                FIXMarketDataEntry(
                                    type=b"1",
                                    px=float(ask.price),
                                    size=float(ask.size),
                                )
                            )
                        except (TypeError, ValueError, AttributeError):
                            continue

                    msg: FIXMessage = {
                        35: b"W",  # Snapshot semantics for each full update
                        55: str(symbol).encode("utf-8"),
                        b"entries": entries,
                    }

                    self._price_app.dispatch(msg)
                except Exception as exc:
                    logger.error("Error bridging market data: %s", exc)

            # Bridge order updates: push ExecutionReport-like messages
            def on_order_update(order_info: OrderInfoProtocol) -> None:
                try:
                    exec_type_value: str | bytes | None = None
                    if order_info.executions:
                        exec_type_value = order_info.executions[-1].get("exec_type")

                    exec_type_bytes: bytes | None
                    if isinstance(exec_type_value, bytes):
                        exec_type_bytes = exec_type_value
                    elif isinstance(exec_type_value, str):
                        exec_type_bytes = exec_type_value.encode("utf-8")
                    else:
                        exec_type_bytes = None

                    msg: FIXMessage = {
                        35: b"8",  # ExecutionReport
                        11: order_info.cl_ord_id.encode("utf-8"),
                    }
                    if exec_type_bytes:
                        msg[150] = exec_type_bytes

                    self._trade_app.dispatch(msg)
                except Exception as exc:
                    logger.error("Error bridging order update: %s", exc)

            manager.add_market_data_callback(on_market_data)
            manager.add_order_callback(on_order_update)

            if not manager.start():
                logger.error("Failed to start FIX Manager sessions")
                return False

            self._manager = manager
            self._initiator = _FIXInitiatorAdapter(manager)
            logger.info("FIXConnectionManager sessions started")
            return True
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.error("Error starting FIX sessions: %s", exc)
            return False

    def stop_sessions(self) -> None:
        if self._manager:
            self._manager.stop()
            logger.info("FIXConnectionManager sessions stopped")

    def get_application(self, session: str) -> Optional[_FIXApplicationAdapter]:
        if session in ("price", "quote"):
            return self._price_app
        if session in ("trade",):
            return self._trade_app
        return None

    def get_initiator(self, session: str) -> Optional[_FIXInitiatorAdapter]:
        if session == "trade":
            return self._initiator
        return None
