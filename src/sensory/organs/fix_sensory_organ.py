"""
FIX Sensory Organ for IC Markets
Processes market data from FIX protocol
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import suppress
from datetime import datetime, timezone
from typing import Any, Callable, Coroutine, Mapping, Optional, Protocol, Sequence

from src.runtime.task_supervisor import TaskSupervisor

logger = logging.getLogger(__name__)


TaskFactory = Callable[[Coroutine[Any, Any, Any], Optional[str]], asyncio.Task[Any]]


class _OrderBookState:
    """Maintains order book depth for a single symbol."""

    def __init__(self) -> None:
        self.bids: dict[float, float] = {}
        self.asks: dict[float, float] = {}

    def apply_snapshot(self, entries: Sequence[dict[str, Any]]) -> None:
        self.bids.clear()
        self.asks.clear()
        self.apply_incremental(entries)

    def apply_incremental(self, entries: Sequence[dict[str, Any]]) -> None:
        for entry in entries:
            side = entry.get("side")
            price = _coerce_float(entry.get("price"))
            size = entry.get("size")
            action = _normalise_action(entry.get("action"))

            if side not in {"bid", "ask"} or price is None:
                continue

            target = self.bids if side == "bid" else self.asks
            size_value = _coerce_float(size)

            if action == "delete" or size_value is None or size_value <= 0.0:
                target.pop(price, None)
                continue

            target[price] = size_value

    def build_depth_snapshot(self, *, levels: int = 5) -> dict[str, dict[str, float | None]]:
        depth: dict[str, dict[str, float | None]] = {}
        ordered_bids = sorted(self.bids.items(), key=lambda item: item[0], reverse=True)
        ordered_asks = sorted(self.asks.items(), key=lambda item: item[0])

        for idx in range(levels):
            level_key = f"L{idx + 1}"
            bid_price, bid_size = ordered_bids[idx] if idx < len(ordered_bids) else (None, None)
            ask_price, ask_size = ordered_asks[idx] if idx < len(ordered_asks) else (None, None)
            depth[level_key] = {
                "bid": bid_price,
                "bid_sz": bid_size,
                "ask": ask_price,
                "ask_sz": ask_size,
            }
        return depth


def _normalise_entry(raw_entry: Any) -> dict[str, Any] | None:
    if not isinstance(raw_entry, Mapping):
        return None

    side = _resolve_side(raw_entry.get("side"), raw_entry.get("type"), raw_entry.get(269))
    if side is None:
        return None

    price = raw_entry.get("px")
    if price is None:
        price = raw_entry.get("price")
    if price is None:
        price = raw_entry.get(270)

    size = raw_entry.get("size")
    if size is None:
        size = raw_entry.get(271)

    action = raw_entry.get("action")
    if action is None:
        action = raw_entry.get(279)

    return {
        "side": side,
        "price": price,
        "size": size,
        "action": action,
    }


def _resolve_side(*candidates: Any) -> str | None:
    for candidate in candidates:
        if candidate is None:
            continue
        if isinstance(candidate, bytes):
            try:
                candidate = candidate.decode("utf-8")
            except Exception:
                continue
        if isinstance(candidate, (int, float)):
            candidate = str(int(candidate))
        text = str(candidate).strip().lower()
        if text in {"0", "bid", "b"}:
            return "bid"
        if text in {"1", "ask", "a"}:
            return "ask"
    return None


def _normalise_action(action: Any) -> str | None:
    if action is None:
        return None
    if isinstance(action, bytes):
        try:
            action = action.decode("utf-8")
        except Exception:
            return None
    if isinstance(action, (int, float)):
        action = str(int(action))
    action_text = str(action).strip().lower()
    if action_text in {"2", "delete", "remove"}:
        return "delete"
    if action_text in {"1", "change", "update"}:
        return "change"
    if action_text in {"0", "add", "new"}:
        return "new"
    return None


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, bytes):
        try:
            value = value.decode("utf-8")
        except Exception:
            return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return None
    return None


def _coerce_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, bytes):
        try:
            value = value.decode("utf-8")
        except Exception:
            return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return int(text)
        except ValueError:
            return None
    return None


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
        self._order_books: dict[str, _OrderBookState] = {}
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

            raw_entries = self._extract_entries(message)
            if not raw_entries:
                return

            book = self._order_books.setdefault(symbol, _OrderBookState())
            book.apply_snapshot(raw_entries)

            payload = self._build_market_data_payload(symbol, book, message)
            self.market_data[symbol] = payload

            await self.event_bus.emit("market_data_update", payload)

            logger.debug("Market data snapshot applied", extra={"symbol": symbol, "seq": payload["seq"]})

        except Exception as e:
            logger.error(f"Error handling market data snapshot: {e}")

    async def _handle_market_data_incremental(self, message: Any) -> None:
        """Handle incremental market data updates."""
        try:
            symbol = message.get(55).decode() if message.get(55) else None
            if not symbol:
                return

            raw_entries = self._extract_entries(message)
            if not raw_entries:
                return

            book = self._order_books.setdefault(symbol, _OrderBookState())
            book.apply_incremental(raw_entries)

            payload = self._build_market_data_payload(symbol, book, message)
            self.market_data[symbol] = payload

            await self.event_bus.emit("market_data_update", payload)

            logger.debug("Market data incremental applied", extra={"symbol": symbol, "seq": payload["seq"]})

        except Exception as e:
            logger.error(f"Error handling market data incremental: {e}")

    def _build_market_data_payload(self, symbol: str, book: "_OrderBookState", message: Any) -> dict[str, Any]:
        depth = book.build_depth_snapshot(levels=5)
        level_one = depth.get("L1", {"bid": None, "ask": None, "bid_sz": None, "ask_sz": None})
        payload = {
            "symbol": symbol,
            "bid": level_one.get("bid"),
            "ask": level_one.get("ask"),
            "bid_sz": level_one.get("bid_sz"),
            "ask_sz": level_one.get("ask_sz"),
            "depth": depth,
            "ts": datetime.now(tz=timezone.utc),
            "seq": _coerce_int(message.get(34)),
        }
        return payload

    def _extract_entries(self, message: Any) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []

        if not isinstance(message, Mapping):
            return entries

        candidate_keys: tuple[object, ...] = (
            b"entries",
            "entries",
            268,
            b"268",
            "NoMDEntries",
        )

        for key in candidate_keys:
            try:
                raw_entries = message.get(key)  # type: ignore[call-arg]
            except AttributeError:  # pragma: no cover - defensive guard
                raw_entries = None
            if raw_entries is None:
                continue

            containers: Sequence[Any]
            if isinstance(raw_entries, Mapping):
                items: list[Any]
                try:
                    items = [raw_entries[idx] for idx in sorted(raw_entries)]
                except TypeError:
                    items = list(raw_entries.values())
                containers = items
            elif isinstance(raw_entries, Sequence) and not isinstance(
                raw_entries,
                (bytes, bytearray, str),
            ):
                containers = raw_entries
            else:
                continue

            for entry in containers:
                normalised = _normalise_entry(entry)
                if normalised is not None:
                    entries.append(normalised)

            if entries:
                return entries

        fallback_entry = _normalise_entry(message)
        if fallback_entry is not None:
            entries.append(fallback_entry)

        return entries

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
