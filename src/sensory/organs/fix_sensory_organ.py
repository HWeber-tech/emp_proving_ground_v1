"""
FIX Sensory Organ for IC Markets
Processes market data from FIX protocol
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import suppress
from datetime import datetime
from typing import Any, Callable, Coroutine, Mapping, Optional, Sequence, Protocol

from src.runtime.task_supervisor import TaskSupervisor

logger = logging.getLogger(__name__)


TaskFactory = Callable[[Coroutine[Any, Any, Any], Optional[str]], asyncio.Task[Any]]


class MarketDataSubscriptionClient(Protocol):
    """Interface for components capable of sending FIX market data requests."""

    def subscribe_market_data(self, symbols: Sequence[str], *, depth: int = 1) -> bool:
        ...

    def unsubscribe_market_data(self, symbols: Sequence[str] | None = None) -> bool:
        ...


class FIXSensoryOrgan:
    """Processes market data from FIX protocol."""

    def __init__(
        self,
        event_bus: Any,
        price_queue: Any,
        config: dict[str, Any],
        task_factory: TaskFactory | None = None,
        market_data_client: MarketDataSubscriptionClient | None = None,
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
        self._fallback_supervisor: TaskSupervisor | None = None
        self._market_data_client = market_data_client
        self._md_subscription_symbols: list[str] = []
        self._md_subscription_active = False
        if task_factory is None:
            self._fallback_supervisor = TaskSupervisor(namespace="fix-sensory-organ")

    async def start(self) -> None:
        """Start the sensory organ."""
        if self.running:
            return

        self.running = True
        logger.info("FIX sensory organ started")

        self._subscribe_market_data(self._resolve_subscription_symbols())

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
        self._unsubscribe_market_data()
        task = self._price_task
        if task is not None:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
        self._price_task = None
        if self._fallback_supervisor is not None:
            await self._fallback_supervisor.cancel_all()
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
        if self._fallback_supervisor is None:  # pragma: no cover - defensive guard
            self._fallback_supervisor = TaskSupervisor(namespace="fix-sensory-organ")
        return self._fallback_supervisor.create(coro, name=name)

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

    def _subscribe_market_data(self, symbols: Sequence[str]) -> None:
        if not symbols or self._market_data_client is None:
            return
        try:
            subscribed = bool(self._market_data_client.subscribe_market_data(symbols))
        except Exception:
            logger.exception("Failed to send MarketDataRequest subscribe", extra={"symbols": symbols})
            return
        if subscribed:
            self._md_subscription_active = True
            self._md_subscription_symbols = list(symbols)
            logger.info("Sent MarketDataRequest subscribe", extra={"symbols": symbols})
        else:
            logger.warning("MarketDataRequest subscribe was rejected", extra={"symbols": symbols})

    def _unsubscribe_market_data(self) -> None:
        if not self._md_subscription_active or self._market_data_client is None:
            return
        symbols = list(self._md_subscription_symbols)
        try:
            ok = bool(self._market_data_client.unsubscribe_market_data(symbols))
        except Exception:
            logger.exception("Failed to send MarketDataRequest unsubscribe", extra={"symbols": symbols})
            ok = False
        if not ok:
            logger.warning("MarketDataRequest unsubscribe may have failed", extra={"symbols": symbols})
        else:
            logger.info("Sent MarketDataRequest unsubscribe", extra={"symbols": symbols})
        self._md_subscription_active = False
        self._md_subscription_symbols = []

    def _resolve_subscription_symbols(self) -> list[str]:
        if self.symbols:
            resolved = self._normalise_symbols(self.symbols)
            if resolved:
                return resolved

        extras = self.config.get("extras") if isinstance(self.config, Mapping) else None
        if isinstance(extras, Mapping):
            for key in (
                "FIX_MARKET_DATA_SYMBOLS",
                "FIX_SYMBOLS",
                "MARKET_DATA_SYMBOLS",
            ):
                raw = extras.get(key)
                symbols = self._parse_symbol_source(raw)
                if symbols:
                    return symbols

        fallback = self.config.get("symbols") if isinstance(self.config, Mapping) else None
        symbols = self._parse_symbol_source(fallback)
        if symbols:
            return symbols

        instruments = self.config.get("instruments") if isinstance(self.config, Mapping) else None
        symbols = self._parse_symbol_source(instruments)
        if symbols:
            return symbols
        return []

    @staticmethod
    def _parse_symbol_source(raw: object) -> list[str]:
        if raw is None:
            return []
        if isinstance(raw, (list, tuple, set)):
            return FIXSensoryOrgan._normalise_symbols(list(raw))
        if isinstance(raw, str):
            cleaned = raw.replace(";", ",")
            parts = [part.strip() for part in cleaned.split(",") if part.strip()]
            return FIXSensoryOrgan._normalise_symbols(parts)
        return FIXSensoryOrgan._normalise_symbols([raw])

    @staticmethod
    def _normalise_symbols(symbols: Sequence[object]) -> list[str]:
        ordered: list[str] = []
        seen: set[str] = set()
        for symbol in symbols:
            text = str(symbol).strip()
            if not text or text in seen:
                continue
            ordered.append(text)
            seen.add(text)
        return ordered

    def subscribe_to_symbols(self, symbols: list[str]) -> None:
        """Subscribe to market data for symbols."""
        self.symbols = self._normalise_symbols(symbols)
        logger.info(f"Subscribed to symbols: {self.symbols}")
        if self.running:
            self._unsubscribe_market_data()
            self._subscribe_market_data(self.symbols)

    def get_market_data(self, symbol: str) -> dict[str, Any]:
        """Get market data for a symbol."""
        return self.market_data.get(symbol, {})

    def get_all_market_data(self) -> dict[str, dict[str, Any]]:
        """Get all market data."""
        return self.market_data.copy()

    def get_subscribed_symbols(self) -> list[str]:
        """Get list of subscribed symbols."""
        return self.symbols.copy()
