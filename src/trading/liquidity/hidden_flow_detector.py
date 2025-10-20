from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Deque, Dict, Mapping, MutableMapping

__all__ = ["HiddenFlowDetector", "HiddenFlowSignal"]


@dataclass(frozen=True)
class QuoteEvent:
    timestamp: datetime
    bid_price: float
    bid_size: float
    ask_price: float
    ask_size: float


@dataclass(frozen=True)
class FillEvent:
    timestamp: datetime
    price: float
    quantity: float


@dataclass(frozen=True)
class HiddenFlowSignal:
    iceberg_score: float
    block_trade_score: float
    dark_pool_score: float
    iceberg_events: int
    block_trade_events: int
    dark_pool_events: int
    flicker_intensity: float
    iceberg_detected: bool
    block_trade_detected: bool
    dark_pool_detected: bool
    metadata: Mapping[str, float | int]


@dataclass
class _SymbolState:
    quotes: Deque[QuoteEvent] = field(default_factory=deque)
    fills: Deque[FillEvent] = field(default_factory=deque)
    fill_sizes: Deque[float] = field(default_factory=deque)
    sum_fill_sizes: float = 0.0
    last_quote: QuoteEvent | None = None
    last_bid_depletion: datetime | None = None
    last_ask_depletion: datetime | None = None
    bid_refills: Deque[datetime] = field(default_factory=lambda: deque(maxlen=128))
    ask_refills: Deque[datetime] = field(default_factory=lambda: deque(maxlen=128))
    flicker_events: int = 0
    total_quotes: int = 0
    iceberg_events: int = 0
    block_trade_events: int = 0
    dark_pool_events: int = 0


class HiddenFlowDetector:
    """Detects iceberg orders, block trades, and dark pool activity from microstructure signals."""

    def __init__(
        self,
        *,
        quote_window: int = 256,
        fill_window: int = 256,
        flicker_window_seconds: float = 1.5,
        iceberg_min_refills: int = 2,
        iceberg_partial_fill_threshold: float = 0.2,
        block_trade_multiplier: float = 3.5,
        block_trade_min_quantity: float = 0.0,
        dark_pool_quiet_seconds: float = 1.0,
        iceberg_score_threshold: float = 0.35,
        block_trade_score_threshold: float = 0.45,
        dark_pool_score_threshold: float = 0.35,
    ) -> None:
        if quote_window <= 0:
            raise ValueError("quote_window must be positive")
        if fill_window <= 0:
            raise ValueError("fill_window must be positive")
        if flicker_window_seconds <= 0:
            raise ValueError("flicker_window_seconds must be positive")
        if iceberg_partial_fill_threshold <= 0:
            raise ValueError("iceberg_partial_fill_threshold must be positive")
        if block_trade_multiplier <= 0:
            raise ValueError("block_trade_multiplier must be positive")
        if dark_pool_quiet_seconds <= 0:
            raise ValueError("dark_pool_quiet_seconds must be positive")

        self.quote_window = int(quote_window)
        self.fill_window = int(fill_window)
        self.flicker_window = timedelta(seconds=float(flicker_window_seconds))
        self.iceberg_min_refills = int(max(1, iceberg_min_refills))
        self.iceberg_partial_fill_threshold = float(iceberg_partial_fill_threshold)
        self.block_trade_multiplier = float(block_trade_multiplier)
        self.block_trade_min_quantity = float(max(0.0, block_trade_min_quantity))
        self.dark_pool_quiet = timedelta(seconds=float(dark_pool_quiet_seconds))
        self.iceberg_score_threshold = float(iceberg_score_threshold)
        self.block_trade_score_threshold = float(block_trade_score_threshold)
        self.dark_pool_score_threshold = float(dark_pool_score_threshold)

        self._state: Dict[str, _SymbolState] = {}

    def record_quote(
        self,
        symbol: str,
        timestamp: datetime,
        *,
        bid_price: float,
        bid_size: float,
        ask_price: float,
        ask_size: float,
    ) -> None:
        state = self._state.setdefault(symbol, _SymbolState())
        quote = QuoteEvent(
            timestamp=timestamp,
            bid_price=float(bid_price),
            bid_size=max(0.0, float(bid_size)),
            ask_price=float(ask_price),
            ask_size=max(0.0, float(ask_size)),
        )

        state.quotes.append(quote)
        while len(state.quotes) > self.quote_window:
            state.quotes.popleft()
        state.total_quotes += 1

        last = state.last_quote
        if last is not None:
            self._update_flicker_state(symbol, state, last, quote)

        state.last_quote = quote
        self._prune_refills(state, quote.timestamp)

    def record_fill(
        self,
        symbol: str,
        timestamp: datetime,
        *,
        price: float,
        quantity: float,
    ) -> None:
        quantity = abs(float(quantity))
        if quantity <= 0.0:
            return

        state = self._state.setdefault(symbol, _SymbolState())
        fill = FillEvent(timestamp=timestamp, price=float(price), quantity=quantity)

        state.fills.append(fill)
        state.fill_sizes.append(quantity)
        state.sum_fill_sizes += quantity
        while len(state.fills) > self.fill_window:
            state.fills.popleft()
        while len(state.fill_sizes) > self.fill_window:
            removed_qty = state.fill_sizes.popleft()
            state.sum_fill_sizes -= removed_qty
        if state.sum_fill_sizes < 0.0:
            state.sum_fill_sizes = 0.0

        self._check_for_iceberg(symbol, state, fill)
        self._check_for_block_trade(state, fill)
        self._check_for_dark_pool(state, fill)

    def evaluate_symbol(self, symbol: str) -> HiddenFlowSignal:
        state = self._state.get(symbol)
        if state is None:
            return HiddenFlowSignal(
                iceberg_score=0.0,
                block_trade_score=0.0,
                dark_pool_score=0.0,
                iceberg_events=0,
                block_trade_events=0,
                dark_pool_events=0,
                flicker_intensity=0.0,
                iceberg_detected=False,
                block_trade_detected=False,
                dark_pool_detected=False,
                metadata={"fills_observed": 0, "quotes_observed": 0},
            )

        fills_observed = len(state.fills)
        quotes_observed = state.total_quotes
        iceberg_score = self._bounded_ratio(state.iceberg_events, fills_observed)
        block_trade_score = self._bounded_ratio(state.block_trade_events, fills_observed)
        dark_pool_score = self._bounded_ratio(state.dark_pool_events, fills_observed)
        flicker_intensity = self._bounded_ratio(state.flicker_events, max(1, quotes_observed))

        return HiddenFlowSignal(
            iceberg_score=iceberg_score,
            block_trade_score=block_trade_score,
            dark_pool_score=dark_pool_score,
            iceberg_events=state.iceberg_events,
            block_trade_events=state.block_trade_events,
            dark_pool_events=state.dark_pool_events,
            flicker_intensity=flicker_intensity,
            iceberg_detected=iceberg_score >= self.iceberg_score_threshold,
            block_trade_detected=block_trade_score >= self.block_trade_score_threshold,
            dark_pool_detected=dark_pool_score >= self.dark_pool_score_threshold,
            metadata={
                "fills_observed": fills_observed,
                "quotes_observed": quotes_observed,
                "avg_fill_size": (
                    state.sum_fill_sizes / fills_observed if fills_observed else 0.0
                ),
            },
        )

    def evaluate_all(self) -> MutableMapping[str, HiddenFlowSignal]:
        return {symbol: self.evaluate_symbol(symbol) for symbol in self._state}

    def _update_flicker_state(
        self,
        symbol: str,
        state: _SymbolState,
        last: QuoteEvent,
        current: QuoteEvent,
    ) -> None:
        if current.bid_price == last.bid_price:
            if current.bid_size < last.bid_size:
                state.last_bid_depletion = current.timestamp
            elif (
                current.bid_size > last.bid_size
                and state.last_bid_depletion is not None
                and current.timestamp - state.last_bid_depletion <= self.flicker_window
            ):
                state.bid_refills.append(current.timestamp)
                state.flicker_events += 1
                state.last_bid_depletion = None

        if current.ask_price == last.ask_price:
            if current.ask_size < last.ask_size:
                state.last_ask_depletion = current.timestamp
            elif (
                current.ask_size > last.ask_size
                and state.last_ask_depletion is not None
                and current.timestamp - state.last_ask_depletion <= self.flicker_window
            ):
                state.ask_refills.append(current.timestamp)
                state.flicker_events += 1
                state.last_ask_depletion = None

        if current.bid_price != last.bid_price:
            state.last_bid_depletion = None
        if current.ask_price != last.ask_price:
            state.last_ask_depletion = None

    def _prune_refills(self, state: _SymbolState, now: datetime) -> None:
        while state.bid_refills and now - state.bid_refills[0] > self.flicker_window:
            state.bid_refills.popleft()
        while state.ask_refills and now - state.ask_refills[0] > self.flicker_window:
            state.ask_refills.popleft()

    def _check_for_iceberg(self, symbol: str, state: _SymbolState, fill: FillEvent) -> None:
        quote = state.last_quote
        if quote is None:
            return

        if quote.timestamp > fill.timestamp:
            return

        side_refills: Deque[datetime] | None = None
        if abs(fill.price - quote.bid_price) < 1e-9:
            side_refills = state.bid_refills
        elif abs(fill.price - quote.ask_price) < 1e-9:
            side_refills = state.ask_refills

        if side_refills is None:
            return

        window_refills = [ts for ts in side_refills if fill.timestamp - ts <= self.flicker_window]
        if len(window_refills) < self.iceberg_min_refills:
            return

        display_size = quote.bid_size if side_refills is state.bid_refills else quote.ask_size
        fill_threshold = max(self.iceberg_partial_fill_threshold * max(display_size, 1.0), 1e-9)
        if fill.quantity <= fill_threshold:
            state.iceberg_events += 1

    def _check_for_block_trade(self, state: _SymbolState, fill: FillEvent) -> None:
        fills_count = len(state.fill_sizes)
        if fills_count <= 1:
            return

        prev_count = fills_count - 1
        prev_sum = state.sum_fill_sizes - fill.quantity
        if prev_count <= 0 or prev_sum <= 0.0:
            baseline = self.block_trade_min_quantity
        else:
            baseline = max(prev_sum / prev_count, self.block_trade_min_quantity)
        if baseline <= 0.0:
            baseline = fill.quantity
        if baseline <= 0.0:
            return
        if fill.quantity >= baseline * self.block_trade_multiplier:
            state.block_trade_events += 1

    def _check_for_dark_pool(self, state: _SymbolState, fill: FillEvent) -> None:
        quote = state.last_quote
        if quote is None:
            return

        if quote.timestamp > fill.timestamp:
            return

        if not (quote.bid_price <= fill.price <= quote.ask_price):
            return

        if abs(fill.price - quote.bid_price) < 1e-9 or abs(fill.price - quote.ask_price) < 1e-9:
            return

        if fill.timestamp - quote.timestamp >= self.dark_pool_quiet:
            state.dark_pool_events += 1

    @staticmethod
    def _bounded_ratio(numerator: int, denominator: int) -> float:
        if denominator <= 0:
            return 0.0
        return min(1.0, max(0.0, float(numerator) / float(denominator)))
