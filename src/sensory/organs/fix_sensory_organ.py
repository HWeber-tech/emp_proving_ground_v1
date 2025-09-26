"""
FIX Sensory Organ for IC Markets
Processes market data from FIX protocol
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import suppress
from datetime import datetime
from typing import Any, Callable, Coroutine, Optional

logger = logging.getLogger(__name__)


TaskFactory = Callable[[Coroutine[Any, Any, Any], Optional[str]], asyncio.Task[Any]]


class FIXSensoryOrgan:
    """Processes market data from FIX protocol."""

    def __init__(
        self,
        event_bus: Any,
        price_queue: Any,
        config: dict[str, Any],
        task_factory: TaskFactory | None = None,
    ) -> None:
        """
        Initialize FIX sensory organ.

        Args:
            event_bus: Event bus for system communication
            price_queue: Queue for price messages
            config: System configuration
        """
        self.event_bus = event_bus
        self.price_queue = price_queue
        self.config = config
        self.running = False
        self.market_data: dict[str, dict[str, Any]] = {}
        self.symbols: list[str] = []
        self._price_task: asyncio.Task[Any] | None = None
        self._task_factory = task_factory

    async def start(self) -> None:
        """Start the sensory organ."""
        if self.running:
            return

        self.running = True
        logger.info("FIX sensory organ started")

        # Start message processing
        self._price_task = self._spawn_task(
            self._process_price_messages(),
            name="fix-sensory-price-feed",
        )

    async def stop(self) -> None:
        """Stop the sensory organ."""
        if not self.running:
            return

        self.running = False
        task = self._price_task
        if task is not None:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
        self._price_task = None
        logger.info("FIX sensory organ stopped")

    async def _process_price_messages(self) -> None:
        """Process price messages from the queue."""
        try:
            while True:
                try:
                    message = await self.price_queue.get()
                except asyncio.CancelledError:
                    break

                if not self.running:
                    if message is None:
                        break
                    continue

                if message is None:
                    continue

                # Process based on message type
                msg_type = message.get(35)

                if msg_type == b"W":  # MarketDataSnapshotFullRefresh
                    await self._handle_market_data_snapshot(message)
                elif msg_type == b"X":  # MarketDataIncrementalRefresh
                    await self._handle_market_data_incremental(message)

        except asyncio.CancelledError:
            logger.debug("FIX sensory organ price task cancelled")
        except Exception as e:
            logger.error(f"Error processing price message: {e}")

    def _spawn_task(self, coro: Coroutine[Any, Any, Any], *, name: str | None = None) -> asyncio.Task[Any]:
        if self._task_factory is not None:
            return self._task_factory(coro, name)
        return asyncio.create_task(coro, name=name)

    async def _handle_market_data_snapshot(self, message: Any) -> None:
        """Handle market data snapshot messages."""
        try:
            symbol = message.get(55).decode() if message.get(55) else None
            if not symbol:
                return

            # Extract market data
            market_data = self._extract_market_data(message)

            if market_data:
                # Update local cache
                self.market_data[symbol] = market_data

                # Emit event for system
                await self.event_bus.emit(
                    "market_data_update",
                    {"symbol": symbol, "data": market_data, "timestamp": datetime.utcnow()},
                )

                logger.debug(f"Market data updated for {symbol}: {market_data}")

        except Exception as e:
            logger.error(f"Error handling market data snapshot: {e}")

    async def _handle_market_data_incremental(self, message: Any) -> None:
        """Handle incremental market data updates."""
        try:
            symbol = message.get(55).decode() if message.get(55) else None
            if not symbol or symbol not in self.market_data:
                return

            # Update existing market data
            updates = self._extract_market_data(message)
            if updates:
                self.market_data[symbol].update(updates)

                # Emit event for system
                await self.event_bus.emit(
                    "market_data_incremental",
                    {"symbol": symbol, "updates": updates, "timestamp": datetime.utcnow()},
                )

                logger.debug(f"Market data incremental update for {symbol}: {updates}")

        except Exception as e:
            logger.error(f"Error handling market data incremental: {e}")

    def _extract_market_data(self, message: Any) -> dict[str, Any]:
        """Extract market data from FIX message."""
        data: dict[str, Any] = {}

        try:
            # Preferred path: adapter supplies parsed entries as b"entries"
            entries = message.get(b"entries")
            if entries:
                best_bid = None
                best_ask = None
                bid_size = None
                ask_size = None

                for entry in entries:
                    et = entry.get("type")
                    px = entry.get("px")
                    sz = entry.get("size")
                    if et == b"0":  # Bid
                        if best_bid is None or px > best_bid:
                            best_bid = px
                            bid_size = sz
                    elif et == b"1":  # Ask
                        if best_ask is None or px < best_ask:
                            best_ask = px
                            ask_size = sz

                if best_bid is not None or best_ask is not None:
                    data = {
                        "bid": best_bid,
                        "ask": best_ask,
                        "bid_size": bid_size,
                        "ask_size": ask_size,
                        "timestamp": datetime.utcnow(),
                    }
                    return data

            # Fallback: extremely simplified single-entry parsing
            if message.get(270):
                entry_type = message.get(269)
                entry_px = message.get(270)
                entry_sz = message.get(271)
                if entry_type == b"0":
                    data["bid"] = float(entry_px)
                    data["bid_size"] = float(entry_sz) if entry_sz else None
                elif entry_type == b"1":
                    data["ask"] = float(entry_px)
                    data["ask_size"] = float(entry_sz) if entry_sz else None
                data["timestamp"] = datetime.utcnow()

        except Exception as e:
            logger.error(f"Error extracting market data: {e}")

        return data

    def subscribe_to_symbols(self, symbols: list[str]) -> None:
        """Subscribe to market data for symbols."""
        self.symbols = symbols
        logger.info(f"Subscribed to symbols: {symbols}")

    def get_market_data(self, symbol: str) -> dict[str, Any]:
        """Get market data for a symbol."""
        return self.market_data.get(symbol, {})

    def get_all_market_data(self) -> dict[str, dict[str, Any]]:
        """Get all market data."""
        return self.market_data.copy()

    def get_subscribed_symbols(self) -> list[str]:
        """Get list of subscribed symbols."""
        return self.symbols.copy()
