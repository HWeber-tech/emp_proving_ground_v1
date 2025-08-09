from __future__ import annotations

"""
Mock FIX Manager for tests and offline development.

Implements a minimal interface used by FIXConnectionManager:
- add_market_data_callback(cb)
- add_order_callback(cb)
- start()/stop()
- trade_connection.send_message_and_track(msg)
"""

import threading
import time
from typing import Callable, List, Any


class _MockTradeConnection:
    def __init__(self, order_cbs: List[Callable[[Any], None]]):
        self._order_cbs = order_cbs

    def send_message_and_track(self, msg: Any) -> bool:
        # Immediately echo an execution report callback on another thread
        def _emit():
            info = type("OrderInfo", (), {})()
            info.cl_ord_id = str(getattr(msg, "cl_ord_id", "TEST"))
            info.executions = [{"exec_type": "0"}]  # New
            for cb in self._order_cbs:
                try:
                    cb(info)
                except Exception:
                    pass
        threading.Thread(target=_emit, daemon=True).start()
        return True


class MockFIXManager:
    def __init__(self):
        self._md_cbs: List[Callable[[str, Any], None]] = []
        self._order_cbs: List[Callable[[Any], None]] = []
        self._running = False
        self.trade_connection = _MockTradeConnection(self._order_cbs)

    def add_market_data_callback(self, cb: Callable[[str, Any], None]) -> None:
        self._md_cbs.append(cb)

    def add_order_callback(self, cb: Callable[[Any], None]) -> None:
        self._order_cbs.append(cb)

    def start(self) -> bool:
        self._running = True
        # Emit a tiny market data tick in background repeatedly for a short period
        def _emit_md_loop():
            t0 = time.time()
            while self._running and (time.time() - t0) < 2.0:
                book = type("Book", (), {})()
                book.bids = [type("L", (), {"price": 1.1, "size": 1000})()]
                book.asks = [type("L", (), {"price": 1.1002, "size": 1000})()]
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


