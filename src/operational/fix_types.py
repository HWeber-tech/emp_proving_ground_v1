"""Shared protocols and typed payloads for FIX integration layers."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import NotRequired, Protocol, Required, SupportsFloat, TypedDict, runtime_checkable


class FIXMarketDataEntry(TypedDict, total=False):
    """Normalized FIX market data entry used by application adapters."""

    type: Required[bytes]
    px: Required[float]
    size: Required[float]
    venue: NotRequired[str]
    level: NotRequired[int]


FIXMessage = dict[int | bytes, bytes | list[FIXMarketDataEntry]]


class ExecutionRecordPayload(TypedDict, total=False):
    """Shape of execution payloads emitted by FIX order callbacks."""

    exec_type: str | bytes
    ord_status: str | bytes
    last_px: float
    last_qty: float
    cum_qty: float
    leaves_qty: float
    avg_px: float
    text: str | bytes
    ord_rej_reason: str | bytes
    cancel_reason: str | bytes
    order_id: str | bytes
    exec_id: str | bytes
    transact_time: str | bytes
    sending_time: str | bytes
    account: str | bytes
    order_type: str | bytes
    time_in_force: str | bytes
    commission: float
    cum_commission: float
    comm_type: str | bytes
    currency: str | bytes
    settle_type: str | bytes
    settle_date: str | bytes
    trade_date: str | bytes
    order_capacity: str | bytes
    customer_or_firm: str | bytes


@runtime_checkable
class OrderBookLevelProtocol(Protocol):
    """Expose price/size pairs from FIX managers."""

    price: SupportsFloat
    size: SupportsFloat


@runtime_checkable
class OrderBookProtocol(Protocol):
    """Container for bid/ask ladders."""

    bids: Sequence[OrderBookLevelProtocol]
    asks: Sequence[OrderBookLevelProtocol]


class OrderExecutionRecord(Protocol):
    """Minimal mapping interface when inspecting execution payloads."""

    def get(
        self,
        key: str,
        default: str | bytes | float | None = None,
    ) -> str | bytes | float | None: ...


class OrderInfoProtocol(Protocol):
    """Subset of OrderInfo attributes consumed by connection managers."""

    cl_ord_id: str
    executions: Sequence[OrderExecutionRecord]


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
