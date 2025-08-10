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
import os
import logging
from typing import Optional, Any, Dict, Callable, List

"""If genuine manager is available in the environment (installed as plugin or present
in the source tree), import it lazily. Otherwise, we will fall back to mock when
explicitly requested or when credentials are missing.
"""
try:
    from src.operational.icmarkets_api import GenuineFIXManager  # type: ignore
    from src.operational.icmarkets_config import ICMarketsConfig  # type: ignore
except Exception:  # pragma: no cover
    GenuineFIXManager = None  # type: ignore
    ICMarketsConfig = None  # type: ignore

try:
    from src.operational.mock_fix import MockFIXManager  # type: ignore
except Exception:
    MockFIXManager = None  # type: ignore


logger = logging.getLogger(__name__)


class _FIXApplicationAdapter:
    """Adapter that forwards incoming messages into an asyncio.Queue."""

    def __init__(self, session_type: str):
        self._session_type = session_type
        self._queue: Optional[asyncio.Queue] = None

    def set_message_queue(self, queue: asyncio.Queue) -> None:
        self._queue = queue

    async def _put(self, message: Dict[str, Any]) -> None:
        if self._queue is not None:
            await self._queue.put(message)


class _FIXInitiatorAdapter:
    """Adapter exposing a `send_message` API to send trade messages."""

    def __init__(self, manager: Any):
        self._manager = manager

    def send_message(self, msg: Any) -> bool:
        # Delegate to trade connection; `send_message_and_track` preserves headers
        if not self._manager or not self._manager.trade_connection:
            logger.error("Trade connection not initialized")
            return False
        return self._manager.trade_connection.send_message_and_track(msg)


class FIXConnectionManager:
    """Compatibility wrapper expected by `main.py`."""

    def __init__(self, system_config) -> None:
        self._system_config = system_config
        self._manager: Optional[GenuineFIXManager] = None
        self._price_app = _FIXApplicationAdapter(session_type="quote")
        self._trade_app = _FIXApplicationAdapter(session_type="trade")
        self._initiator: Optional[_FIXInitiatorAdapter] = None

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
                force_mock or
                (GenuineFIXManager is None) or
                (not creds_present)
            )

            if use_mock:
                if MockFIXManager is None:
                    logger.error("MockFIXManager not available")
                    return False
                manager = MockFIXManager()
            else:
                if ICMarketsConfig is None or GenuineFIXManager is None:
                    logger.error("Genuine FIX components not available")
                    return False
                ic_cfg = ICMarketsConfig(
                    environment=self._system_config.environment,
                    account_number=self._system_config.account_number,
                )
                # Inject password if present
                ic_cfg.password = self._system_config.password
                manager = GenuineFIXManager(ic_cfg)

            # Bridge market data: convert order book updates to queue-friendly messages
            def on_market_data(symbol: str, order_book) -> None:
                try:
                    # Build a synthetic FIX-like message that the sensory organ can process
                    entries = []
                    for bid in getattr(order_book, "bids", [])[:10]:
                        entries.append({"type": b"0", "px": bid.price, "size": bid.size})
                    for ask in getattr(order_book, "asks", [])[:10]:
                        entries.append({"type": b"1", "px": ask.price, "size": ask.size})

                    msg: Dict[Any, Any] = {
                        35: b"W",  # Snapshot semantics for each full update
                        55: symbol.encode("utf-8"),
                        b"entries": entries,
                    }

                    # Schedule put without blocking the FIX threads
                    if self._price_app._queue is not None:
                        try:
                            loop = asyncio.get_running_loop()
                        except RuntimeError:
                            loop = None
                        if loop and loop.is_running():
                            loop.call_soon_threadsafe(lambda: asyncio.create_task(self._price_app._put(msg)))
                        else:
                            # Fallback: create a temporary loop to enqueue
                            _loop = asyncio.new_event_loop()
                            try:
                                _loop.run_until_complete(self._price_app._put(msg))
                            finally:
                                _loop.close()
                except Exception as e:
                    logger.error(f"Error bridging market data: {e}")

            # Bridge order updates: push ExecutionReport-like messages
            def on_order_update(order_info) -> None:
                try:
                    exec_type = None
                    if order_info.executions:
                        exec_type = order_info.executions[-1].get("exec_type")

                    msg: Dict[Any, Any] = {
                        35: b"8",  # ExecutionReport
                        11: order_info.cl_ord_id.encode("utf-8"),
                    }
                    if exec_type:
                        msg[150] = exec_type.encode("utf-8") if isinstance(exec_type, str) else exec_type

                    if self._trade_app._queue is not None:
                        try:
                            loop = asyncio.get_running_loop()
                        except RuntimeError:
                            loop = None
                        if loop and loop.is_running():
                            loop.call_soon_threadsafe(lambda: asyncio.create_task(self._trade_app._put(msg)))
                        else:
                            _loop = asyncio.new_event_loop()
                            try:
                                _loop.run_until_complete(self._trade_app._put(msg))
                            finally:
                                _loop.close()
                except Exception as e:
                    logger.error(f"Error bridging order update: {e}")

            manager.add_market_data_callback(on_market_data)
            manager.add_order_callback(on_order_update)

            if not manager.start():
                logger.error("Failed to start FIX Manager sessions")
                return False

            self._manager = manager
            self._initiator = _FIXInitiatorAdapter(manager)
            logger.info("FIXConnectionManager sessions started")
            return True
        except Exception as e:
            logger.error(f"Error starting FIX sessions: {e}")
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


