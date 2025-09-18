"""
Mock FIX Manager for tests and offline development.

Implements a minimal interface used by ``FIXConnectionManager``:

* ``add_market_data_callback(cb)``
* ``add_order_callback(cb)``
* ``start()/stop()``
* ``trade_connection.send_message_and_track(msg)``
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Callable, Protocol, Sequence, SupportsFloat, TypedDict, cast


class OrderBookLevelProtocol(Protocol):
    price: SupportsFloat
    size: SupportsFloat


class OrderBookProtocol(Protocol):
    bids: Sequence[OrderBookLevelProtocol]
    asks: Sequence[OrderBookLevelProtocol]


class OrderInfoProtocol(Protocol):
    cl_ord_id: str
    executions: Sequence["ExecutionRecord"]


class FIXTradeConnectionProtocol(Protocol):
    def send_message_and_track(self, msg: object) -> bool: ...


class ExecutionRecord(TypedDict):
    """Shape of execution payloads emitted to callbacks."""

    exec_type: str


@dataclass(slots=True)
class MockOrderInfo:
    """Lightweight order status container for callbacks."""

    cl_ord_id: str
    executions: Sequence[ExecutionRecord]
    symbol: str = "TEST"
    side: str = "1"
    last_qty: float = 0.0
    last_px: float = 0.0


@dataclass(slots=True)
class MockOrderBookLevel:
    """Single side of the synthetic order book."""

    price: float
    size: float


@dataclass(slots=True)
class MockOrderBook:
    """Synthetic order book broadcast to market data callbacks."""

    bids: Sequence[OrderBookLevelProtocol]
    asks: Sequence[OrderBookLevelProtocol]


def _build_order_info(msg: object, exec_type: str) -> MockOrderInfo:
    """Create a fully populated ``MockOrderInfo`` for a given execution type."""

    def _coerce_float(value: SupportsFloat | str | None, default: float = 0.0) -> float:
        try:
            if value is None:
                return default
            return float(value)
        except (TypeError, ValueError):
            return default

    cl_ord_id = str(getattr(msg, "cl_ord_id", "TEST"))
    symbol = str(getattr(msg, "symbol", "TEST"))
    side = str(getattr(msg, "side", "1"))
    qty = _coerce_float(getattr(msg, "quantity", getattr(msg, "last_qty", 0.0)))
    px = _coerce_float(getattr(msg, "price", getattr(msg, "last_px", 0.0)))
    execution = ExecutionRecord(exec_type=exec_type)
    return MockOrderInfo(
        cl_ord_id=cl_ord_id,
        executions=(execution,),
        symbol=symbol,
        side=side,
        last_qty=qty,
        last_px=px,
    )


class _MockTradeConnection:
    def __init__(self, order_cbs: list[Callable[[OrderInfoProtocol], None]]) -> None:
        self._order_cbs = order_cbs

    def send_message_and_track(self, msg: object) -> bool:
        # Respect optional flags on msg for reject/cancel flows
        if getattr(msg, "reject", False):

            def _emit_reject() -> None:
                info = _build_order_info(msg, "8")  # Reject
                for cb in self._order_cbs:
                    try:
                        cb(info)
                    except Exception:
                        pass

            threading.Thread(target=_emit_reject, daemon=True).start()
            return True

        if getattr(msg, "cancel", False):

            def _emit_cancel() -> None:
                info = _build_order_info(msg, "4")  # Canceled
                for cb in self._order_cbs:
                    try:
                        cb(info)
                    except Exception:
                        pass

            threading.Thread(target=_emit_cancel, daemon=True).start()
            return True

        # Emit a New, Partial, then Fill execution on background threads
        def _emit_new() -> None:
            info = _build_order_info(msg, "0")  # New
            for cb in self._order_cbs:
                try:
                    cb(info)
                except Exception:
                    pass

        def _emit_partial() -> None:
            time.sleep(0.05)
            info = _build_order_info(msg, "1")  # Partial Fill
            for cb in self._order_cbs:
                try:
                    cb(info)
                except Exception:
                    pass

        def _emit_fill() -> None:
            time.sleep(0.1)
            info = _build_order_info(msg, "F")  # Fill
            for cb in self._order_cbs:
                try:
                    cb(info)
                except Exception:
                    pass

        threading.Thread(target=_emit_new, daemon=True).start()
        threading.Thread(target=_emit_partial, daemon=True).start()
        threading.Thread(target=_emit_fill, daemon=True).start()
        return True


class MockFIXManager:
    trade_connection: FIXTradeConnectionProtocol

    def __init__(self) -> None:
        self._md_cbs: list[Callable[[str, OrderBookProtocol], None]] = []
        self._order_cbs: list[Callable[[OrderInfoProtocol], None]] = []
        self._running = False
        self.trade_connection = _MockTradeConnection(self._order_cbs)

    def add_market_data_callback(self, cb: Callable[[str, OrderBookProtocol], None]) -> None:
        self._md_cbs.append(cb)

    def add_order_callback(self, cb: Callable[[OrderInfoProtocol], None]) -> None:
        self._order_cbs.append(cb)

    def start(self) -> bool:
        self._running = True

        # Emit a tiny market data tick in background repeatedly for a short period
        def _emit_md_loop() -> None:
            t0 = time.time()
            while self._running and (time.time() - t0) < 2.0:
                book = MockOrderBook(
                    bids=cast(
                        Sequence[OrderBookLevelProtocol],
                        (MockOrderBookLevel(price=1.1, size=1000.0),),
                    ),
                    asks=cast(
                        Sequence[OrderBookLevelProtocol],
                        (MockOrderBookLevel(price=1.1002, size=1000.0),),
                    ),
                )
                for cb in self._md_cbs:
                    try:
                        cb("EURUSD", book)
                    except Exception:
                        pass
                time.sleep(0.05)

        threading.Thread(target=_emit_md_loop, daemon=True).start()
        return True

    def stop(self) -> None:
        self._running = False
