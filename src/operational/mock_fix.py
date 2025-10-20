"""Mock FIX Manager for tests and offline development."""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field, replace
from datetime import date, datetime, timezone
from itertools import count
from typing import Callable, Mapping, Sequence, SupportsFloat, cast

from src.operational.fix_types import (
    ExecutionRecordPayload,
    FIXTradeConnectionProtocol,
    OrderExecutionRecord,
    OrderBookLevelProtocol,
    OrderBookProtocol,
    OrderInfoProtocol,
)

logger = logging.getLogger(__name__)


_MISSING = object()


@dataclass(slots=True)
class TelemetryEvent:
    """Lightweight record of activity within the mock FIX manager."""

    timestamp: float
    event: str
    details: dict[str, object]


def _clone_execution(record: ExecutionRecordPayload) -> ExecutionRecordPayload:
    """Create a shallow copy of an execution payload for safe reuse."""

    return cast(ExecutionRecordPayload, dict(record))


def _clone_order_info(info: "MockOrderInfo") -> "MockOrderInfo":
    """Return a detached copy of the provided order info container."""

    executions = tuple(
        _clone_execution(cast(ExecutionRecordPayload, execution)) for execution in info.executions
    )
    return replace(
        info,
        executions=cast(Sequence[OrderExecutionRecord], executions),
    )


def _noop_sleep(_: float) -> None:
    """Sleep stub used when executing synchronous mock order flows."""

    return None


def _default_timestamp() -> str:
    """Return a UTC timestamp formatted like common FIX fields."""

    now = datetime.now(timezone.utc)
    # FIX standard: YYYYMMDD-HH:MM:SS.sss - trim microseconds to milliseconds
    return now.strftime("%Y%m%d-%H:%M:%S.%f")[:-3]


@dataclass(slots=True)
class _OrderState:
    """Mutable order state tracked while generating execution lifecycles."""

    cl_ord_id: str
    symbol: str
    side: str
    orig_qty: float
    price: float
    cum_qty: float = 0.0
    partial_fill_ratio: float | None = None
    fill_price_override: float | None = None
    auto_complete: bool = True
    order_id: str | None = None
    exec_sequence: int = 0
    exec_id_prefix: str | None = None
    reject_reason: str | None = None
    cancel_reason: str | None = None
    reject_text: str | None = None
    cancel_text: str | None = None
    transact_time_override: str | None = None
    sending_time_override: str | None = None
    filled_notional: float = 0.0
    history: list[ExecutionRecordPayload] = field(default_factory=list)
    last_info: "MockOrderInfo" | None = None
    account: str | None = None
    order_type: str | None = None
    time_in_force: str | None = None
    commission_default: float | None = None
    commission_total: float = 0.0
    last_commission: float = 0.0
    comm_type: str | None = None
    currency: str | None = None
    settle_type: str | None = None
    settle_date: str | None = None
    trade_date: str | None = None
    order_capacity: str | None = None
    customer_or_firm: str | None = None

    @property
    def remaining_qty(self) -> float:
        return max(self.orig_qty - self.cum_qty, 0.0)


class _TelemetryRecorder:
    """Thread-safe bounded buffer of telemetry events."""

    def __init__(self, maxlen: int = 256) -> None:
        self._events: deque[TelemetryEvent] = deque(maxlen=maxlen)
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)

    def record(self, event: str, **details: object) -> None:
        with self._condition:
            self._events.append(TelemetryEvent(timestamp=time.time(), event=event, details=details))
            self._condition.notify_all()

    def snapshot(self) -> list[TelemetryEvent]:
        with self._condition:
            return list(self._events)

    def wait_for(
        self,
        predicate: Callable[[TelemetryEvent], bool],
        *,
        count: int = 1,
        timeout: float | None = None,
    ) -> bool:
        """Block until ``count`` telemetry events satisfy ``predicate``."""

        deadline = time.monotonic() + timeout if timeout is not None else 0.0
        with self._condition:
            while True:
                matches = sum(1 for event in self._events if predicate(event))
                if matches >= count:
                    return True
                if timeout is None:
                    self._condition.wait()
                    continue
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
        self._condition.wait(timeout=remaining)


class _MockPriceConnection:
    """Minimal price session stub that records outbound market data requests."""

    def __init__(self, telemetry: _TelemetryRecorder) -> None:
        self._telemetry = telemetry
        self._requests: deque[tuple[str, object]] = deque(maxlen=128)

    def send_message_and_track(self, msg: object, req_id: str) -> bool:
        """Accept outbound FIX messages and record them for inspection."""

        self._requests.append((req_id, msg))
        self._telemetry.record("market_data_request", req_id=req_id)
        return True

    def snapshot_requests(self) -> list[tuple[str, object]]:
        return list(self._requests)


@dataclass(slots=True)
class MockOrderInfo:
    """Lightweight order status container for callbacks."""

    cl_ord_id: str
    executions: Sequence[OrderExecutionRecord]
    symbol: str = "TEST"
    side: str = "1"
    last_qty: float = 0.0
    last_px: float = 0.0
    cum_qty: float = 0.0
    leaves_qty: float = 0.0
    ord_status: str = "0"
    avg_px: float = 0.0
    order_id: str = ""
    exec_id: str = ""
    orig_qty: float = 0.0
    order_px: float = 0.0
    text: str | None = None
    ord_rej_reason: str | None = None
    cancel_reason: str | None = None
    transact_time: str | None = None
    sending_time: str | None = None
    account: str | None = None
    order_type: str | None = None
    time_in_force: str | None = None
    last_commission: float = 0.0
    cum_commission: float = 0.0
    comm_type: str | None = None
    currency: str | None = None
    settle_type: str | None = None
    settle_date: str | None = None
    trade_date: str | None = None
    order_capacity: str | None = None
    customer_or_firm: str | None = None


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


@dataclass(slots=True)
class MockMarketDataStep:
    """Declarative market data snapshot emitted by the mock manager."""

    bids: Sequence[object] = ()
    asks: Sequence[object] = ()
    delay: float | None = None


@dataclass(slots=True)
class _ResolvedMarketDataStep:
    """Internal representation of a normalized market data snapshot."""

    bids: tuple[MockOrderBookLevel, ...]
    asks: tuple[MockOrderBookLevel, ...]
    delay: float


def _coerce_float(value: SupportsFloat | str | None, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _coerce_optional_float(
    value: SupportsFloat | str | bytes | None,
    *,
    default: float | None = None,
) -> float | None:
    if value is None:
        return default
    if isinstance(value, (bytes, bytearray)):
        value = value.decode("ascii", "ignore")
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _coerce_optional_str(value: object | None, *, default: str | None = None) -> str | None:
    if value is None:
        return default
    if isinstance(value, (bytes, bytearray)):
        value = value.decode("utf-8", "ignore")
    text = str(value).strip()
    return text or default


def _coerce_optional_fix_date(value: object | None) -> str | None:
    """Best-effort coercion for FIX-style settlement dates."""

    if value is None:
        return None
    if isinstance(value, (bytes, bytearray)):
        value = value.decode("utf-8", "ignore")
    if isinstance(value, datetime):
        return value.strftime("%Y%m%d")
    if isinstance(value, date):
        return value.strftime("%Y%m%d")
    if isinstance(value, (int, float)):
        try:
            return f"{int(value):08d}"
        except (OverflowError, TypeError, ValueError):
            return None
    text = str(value).strip()
    if not text:
        return None
    digits = "".join(ch for ch in text if ch.isdigit())
    if len(digits) == 8:
        return digits
    return text


def _coerce_optional_bool(value: object | None) -> bool | None:
    """Best-effort coercion of optional truthy/falsey inputs."""

    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, (bytes, bytearray)):
        value = value.decode("utf-8", "ignore")
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y"}:
        return True
    if text in {"0", "false", "no", "n"}:
        return False
    return None


def _coerce_positive_int(
    value: SupportsFloat | str | bytes | None,
    *,
    minimum: int = 1,
) -> int | None:
    """Coerce ``value`` to an integer greater than or equal to ``minimum``."""

    if value is None:
        return None
    if isinstance(value, (bytes, bytearray)):
        try:
            value = value.decode("ascii")
        except UnicodeDecodeError as exc:
            logger.debug(
                "Failed to decode positive int from bytes payload: %s",
                exc,
                exc_info=exc,
            )
            return None
    try:
        number = int(float(value))
    except (TypeError, ValueError):
        return None
    if number < minimum:
        number = minimum
    return number


def _coerce_ratio(value: SupportsFloat | str | bytes | None) -> float | None:
    ratio = _coerce_optional_float(value)
    if ratio is None:
        return None
    return max(min(ratio, 1.0), 0.0)


def _coerce_repetition(value: SupportsFloat | str | bytes | None) -> int | None:
    """Coerce optional repetition counters to positive integers."""

    count = _coerce_optional_float(value)
    if count is None:
        return None
    try:
        repetitions = int(count)
    except (TypeError, ValueError):
        return None
    if repetitions <= 0:
        return None
    return repetitions


def _coerce_order_book_level(value: object) -> MockOrderBookLevel | None:
    try:
        if isinstance(value, OrderBookLevelProtocol):
            return MockOrderBookLevel(
                price=_coerce_float(getattr(value, "price", 0.0)),
                size=_coerce_float(getattr(value, "size", 0.0)),
            )
    except (AttributeError, TypeError, ValueError) as exc:
        logger.debug(
            "Failed to coerce order book protocol instance %s: %s",
            value,
            exc,
            exc_info=exc,
        )
        return None
    if isinstance(value, Mapping):
        price = _coerce_float(
            cast(
                SupportsFloat | str | None,
                value.get("price") or value.get("px") or value.get("bid") or value.get("ask"),
            ),
            default=0.0,
        )
        size = _coerce_float(
            cast(
                SupportsFloat | str | None,
                value.get("size") or value.get("qty") or value.get("quantity"),
            ),
            default=0.0,
        )
        return MockOrderBookLevel(price=price, size=size)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        if len(value) >= 2:
            return MockOrderBookLevel(
                price=_coerce_float(cast(SupportsFloat | str | None, value[0])),
                size=_coerce_float(cast(SupportsFloat | str | None, value[1])),
            )
        return None
    if hasattr(value, "price") and hasattr(value, "size"):
        try:
            return MockOrderBookLevel(
                price=_coerce_float(getattr(value, "price")),
                size=_coerce_float(getattr(value, "size")),
            )
        except (AttributeError, TypeError, ValueError) as exc:
            logger.debug(
                "Failed to coerce order book level from %s: %s",
                value,
                exc,
                exc_info=exc,
            )
            return None
    return None


def _normalize_market_data_side(levels: object) -> tuple[MockOrderBookLevel, ...]:
    if levels is None:
        return ()
    if isinstance(levels, (str, bytes, bytearray)):
        return ()
    if isinstance(levels, Mapping):
        level = _coerce_order_book_level(levels)
        return (level,) if level else ()
    if isinstance(levels, Sequence):
        result: list[MockOrderBookLevel] = []
        for entry in levels:
            level = _coerce_order_book_level(entry)
            if level is not None:
                result.append(level)
        return tuple(result)
    level = _coerce_order_book_level(levels)
    return (level,) if level else ()


def _levels_to_telemetry(levels: Sequence[MockOrderBookLevel]) -> list[dict[str, float]]:
    return [{"price": level.price, "size": level.size} for level in levels]


@dataclass(slots=True)
class MockExecutionStep:
    """Instruction describing a synthetic execution to emit."""

    exec_type: str
    quantity: float | None = None
    delay: float | None = None
    price: float | None = None
    order_id: str | None = None
    exec_id: str | None = None
    account: str | None = None
    order_type: str | None = None
    time_in_force: str | None = None
    text: str | None = None
    ord_rej_reason: str | None = None
    cancel_reason: str | None = None
    ratio: float | None = None
    quantity_ratio: float | None = None
    remaining_ratio: float | None = None
    repeat: int | None = None
    transact_time: str | None = None
    sending_time: str | None = None
    commission: float | None = None
    comm_type: str | None = None
    currency: str | None = None
    settle_type: str | None = None
    settle_date: str | None = None
    trade_date: str | None = None
    order_capacity: str | None = None
    customer_or_firm: str | None = None


@dataclass(slots=True)
class _ResolvedExecutionStep:
    """Internal representation of a normalized execution step."""

    exec_type: str
    quantity: float | None
    delay: float
    price: float | None
    order_id: str | None
    exec_id: str | None
    account: str | None
    order_type: str | None
    time_in_force: str | None
    text: str | None
    ord_rej_reason: str | None
    cancel_reason: str | None
    quantity_ratio: float | None
    remaining_ratio: float | None
    repeat: int
    transact_time: str | None
    sending_time: str | None
    commission: float | None
    comm_type: str | None
    currency: str | None
    settle_type: str | None
    settle_date: str | None
    trade_date: str | None
    order_capacity: str | None
    customer_or_firm: str | None


class _MockTradeConnection(FIXTradeConnectionProtocol):
    def __init__(
        self,
        order_cbs: list[Callable[[OrderInfoProtocol], None]],
        telemetry: _TelemetryRecorder,
        *,
        partial_fill_ratio: float,
        execution_interval: float,
        synchronous: bool = False,
        timestamp_factory: Callable[[], str] | None = None,
        default_account: str | None = None,
        default_order_type: str | None = None,
        default_time_in_force: str | None = None,
        order_id_prefix: str | None = None,
        order_id_start: int = 1,
        order_id_padding: int = 6,
        exec_id_start: int = 1,
        default_exec_id_prefix: str | None = None,
        default_commission: float | None = None,
        default_commission_type: str | None = None,
        default_commission_currency: str | None = None,
        default_settle_type: str | None = None,
        default_settle_date: str | int | float | date | datetime | None = None,
        default_trade_date: str | int | float | date | datetime | None = None,
        default_order_capacity: str | None = None,
        default_customer_or_firm: str | None = None,
    ) -> None:
        self._order_cbs = order_cbs
        self._telemetry = telemetry
        self._partial_fill_ratio = partial_fill_ratio
        self._execution_interval = max(execution_interval, 0.0)
        self._synchronous = bool(synchronous)
        self._timestamp_factory = timestamp_factory or _default_timestamp
        self._default_account = _coerce_optional_str(default_account)
        self._default_order_type = _coerce_optional_str(default_order_type)
        self._default_time_in_force = _coerce_optional_str(default_time_in_force)
        commission_default = _coerce_optional_float(
            cast(SupportsFloat | str | bytes | None, default_commission)
        )
        if commission_default is not None and commission_default < 0.0:
            commission_default = 0.0
        self._default_commission = commission_default
        self._default_commission_type = _coerce_optional_str(default_commission_type)
        self._default_commission_currency = _coerce_optional_str(default_commission_currency)
        self._default_settle_type = _coerce_optional_str(default_settle_type)
        self._default_settle_date = _coerce_optional_fix_date(
            cast(object | None, default_settle_date)
        )
        self._default_trade_date = _coerce_optional_fix_date(
            cast(object | None, default_trade_date)
        )
        self._default_order_capacity = _coerce_optional_str(default_order_capacity)
        self._default_customer_or_firm = _coerce_optional_str(default_customer_or_firm)
        self._order_id_prefix = _coerce_optional_str(order_id_prefix) or "MOCK-ORD"
        padding = _coerce_positive_int(order_id_padding, minimum=1)
        self._order_id_padding = padding if padding is not None else 6
        start_value = _coerce_positive_int(order_id_start, minimum=1)
        start_counter = start_value if start_value is not None else 1
        self._order_id_sequence = count(start_counter)
        exec_start_value = _coerce_positive_int(exec_id_start, minimum=1)
        exec_start = exec_start_value if exec_start_value is not None else 1
        self._exec_sequence_start = exec_start - 1
        self._default_exec_id_prefix = _coerce_optional_str(default_exec_id_prefix)
        self._sleep: Callable[[float], None]
        if self._synchronous:
            self._sleep = _noop_sleep
        else:
            self._sleep = time.sleep
        self._orders: dict[str, _OrderState] = {}
        self._completed_history: dict[str, tuple[ExecutionRecordPayload, ...]] = {}
        self._completed_info: dict[str, MockOrderInfo] = {}
        self._lock = threading.Lock()
        self._active_threads: set[threading.Thread] = set()
        self._idle_event = threading.Event()
        self._idle_event.set()

    def _spawn(self, target: Callable[[], None]) -> None:
        """Execute ``target`` on a background thread tracked for idleness."""

        if self._synchronous:
            self._idle_event.clear()
            try:
                target()
            finally:
                self._idle_event.set()
            return

        thread: threading.Thread | None = None

        def runner() -> None:
            nonlocal thread
            try:
                target()
            finally:
                with self._lock:
                    if thread is not None:
                        self._active_threads.discard(thread)
                    if not self._active_threads:
                        self._idle_event.set()

        thread = threading.Thread(target=runner, daemon=True)
        with self._lock:
            self._active_threads.add(thread)
            self._idle_event.clear()
        thread.start()

    def wait_for_idle(self, timeout: float | None = None) -> bool:
        """Block until all outstanding order threads have finished."""

        return self._idle_event.wait(timeout=timeout)

    def list_active_order_ids(self) -> list[str]:
        """Return a snapshot of currently active client order identifiers."""

        with self._lock:
            return list(self._orders.keys())

    def snapshot_active_orders(self) -> list[MockOrderInfo]:
        """Return clones of the most recent info for each active order."""

        with self._lock:
            snapshots: list[MockOrderInfo] = []
            for state in self._orders.values():
                info = state.last_info
                if info is not None:
                    snapshots.append(_clone_order_info(info))
                    continue
                avg_px = state.filled_notional / state.cum_qty if state.cum_qty > 0.0 else 0.0
                snapshots.append(
                    MockOrderInfo(
                        cl_ord_id=state.cl_ord_id,
                        executions=(),
                        symbol=state.symbol,
                        side=state.side,
                        last_qty=0.0,
                        last_px=state.price,
                        cum_qty=state.cum_qty,
                        leaves_qty=state.remaining_qty,
                        ord_status="0",
                        avg_px=avg_px,
                        order_id=state.order_id or "",
                        exec_id="",
                        orig_qty=state.orig_qty,
                        order_px=state.price,
                        transact_time=state.transact_time_override,
                        sending_time=state.sending_time_override,
                        account=state.account,
                        order_type=state.order_type,
                        time_in_force=state.time_in_force,
                        last_commission=state.last_commission,
                        cum_commission=state.commission_total,
                        comm_type=state.comm_type,
                        currency=state.currency,
                        settle_type=state.settle_type,
                        settle_date=state.settle_date,
                        trade_date=state.trade_date,
                        order_capacity=state.order_capacity,
                        customer_or_firm=state.customer_or_firm,
                    )
                )
            return snapshots

    def configure_defaults(
        self,
        *,
        account: object = _MISSING,
        order_type: object = _MISSING,
        time_in_force: object = _MISSING,
        order_capacity: object = _MISSING,
        customer_or_firm: object = _MISSING,
        commission: object = _MISSING,
        comm_type: object = _MISSING,
        currency: object = _MISSING,
        settle_type: object = _MISSING,
        settle_date: object = _MISSING,
        trade_date: object = _MISSING,
    ) -> None:
        """Adjust default order metadata applied to newly tracked orders."""

        with self._lock:
            old_commission_default = self._default_commission
            old_commission_type = self._default_commission_type
            old_commission_currency = self._default_commission_currency
            old_settle_type = self._default_settle_type
            old_settle_date = self._default_settle_date
            old_trade_date = self._default_trade_date
            old_order_capacity = self._default_order_capacity
            old_customer_or_firm = self._default_customer_or_firm
            if account is not _MISSING:
                account_value = _coerce_optional_str(cast(object | None, account))
                self._default_account = account_value
                if account_value is not None:
                    for state in self._orders.values():
                        if state.account is None:
                            state.account = account_value
            if order_type is not _MISSING:
                order_type_value = _coerce_optional_str(cast(object | None, order_type))
                self._default_order_type = order_type_value
                if order_type_value is not None:
                    for state in self._orders.values():
                        if state.order_type is None:
                            state.order_type = order_type_value
            if time_in_force is not _MISSING:
                tif_value = _coerce_optional_str(cast(object | None, time_in_force))
                self._default_time_in_force = tif_value
                if tif_value is not None:
                    for state in self._orders.values():
                        if state.time_in_force is None:
                            state.time_in_force = tif_value
            if order_capacity is not _MISSING:
                capacity_value = _coerce_optional_str(cast(object | None, order_capacity))
                self._default_order_capacity = capacity_value
                for state in self._orders.values():
                    if capacity_value is None:
                        if state.order_capacity == old_order_capacity:
                            state.order_capacity = None
                    elif state.order_capacity is None:
                        state.order_capacity = capacity_value
            if customer_or_firm is not _MISSING:
                cust_value = _coerce_optional_str(cast(object | None, customer_or_firm))
                self._default_customer_or_firm = cust_value
                for state in self._orders.values():
                    if cust_value is None:
                        if state.customer_or_firm == old_customer_or_firm:
                            state.customer_or_firm = None
                    elif state.customer_or_firm is None:
                        state.customer_or_firm = cust_value
            if commission is not _MISSING:
                commission_value = _coerce_optional_float(
                    cast(SupportsFloat | str | bytes | None, commission)
                )
                if commission_value is not None and commission_value < 0.0:
                    commission_value = 0.0
                self._default_commission = commission_value
                for state in self._orders.values():
                    if commission_value is None:
                        if state.commission_default == old_commission_default:
                            state.commission_default = None
                    elif state.commission_default is None:
                        state.commission_default = commission_value
            if comm_type is not _MISSING:
                comm_type_value = _coerce_optional_str(cast(object | None, comm_type))
                self._default_commission_type = comm_type_value
                for state in self._orders.values():
                    if comm_type_value is None:
                        if state.comm_type == old_commission_type:
                            state.comm_type = None
                    elif state.comm_type is None:
                        state.comm_type = comm_type_value
            if currency is not _MISSING:
                currency_value = _coerce_optional_str(cast(object | None, currency))
                self._default_commission_currency = currency_value
                for state in self._orders.values():
                    if currency_value is None:
                        if state.currency == old_commission_currency:
                            state.currency = None
                    elif state.currency is None:
                        state.currency = currency_value
            if settle_type is not _MISSING:
                settle_type_value = _coerce_optional_str(cast(object | None, settle_type))
                self._default_settle_type = settle_type_value
                for state in self._orders.values():
                    if settle_type_value is None:
                        if state.settle_type == old_settle_type:
                            state.settle_type = None
                    elif state.settle_type is None:
                        state.settle_type = settle_type_value
            if settle_date is not _MISSING:
                settle_date_value = _coerce_optional_fix_date(cast(object | None, settle_date))
                self._default_settle_date = settle_date_value
                for state in self._orders.values():
                    if settle_date_value is None:
                        if state.settle_date == old_settle_date:
                            state.settle_date = None
                    elif state.settle_date is None:
                        state.settle_date = settle_date_value
            if trade_date is not _MISSING:
                trade_date_value = _coerce_optional_fix_date(cast(object | None, trade_date))
                self._default_trade_date = trade_date_value
                for state in self._orders.values():
                    if trade_date_value is None:
                        if state.trade_date == old_trade_date:
                            state.trade_date = None
                    elif state.trade_date is None:
                        state.trade_date = trade_date_value

    def configure_id_generation(
        self,
        *,
        order_id_prefix: object = _MISSING,
        order_id_start: object = _MISSING,
        order_id_padding: object = _MISSING,
        exec_id_start: object = _MISSING,
        exec_id_prefix: object = _MISSING,
    ) -> None:
        """Adjust order/execution identifier sequencing preferences."""

        with self._lock:
            if order_id_prefix is not _MISSING:
                prefix_value = _coerce_optional_str(cast(object | None, order_id_prefix))
                self._order_id_prefix = prefix_value or "MOCK-ORD"
            if order_id_padding is not _MISSING:
                padding_value = _coerce_positive_int(
                    cast(SupportsFloat | str | bytes | None, order_id_padding),
                    minimum=1,
                )
                if padding_value is not None:
                    self._order_id_padding = padding_value
            if order_id_start is not _MISSING:
                start_value = _coerce_positive_int(
                    cast(SupportsFloat | str | bytes | None, order_id_start),
                    minimum=1,
                )
                if start_value is not None:
                    self._order_id_sequence = count(start_value)
            if exec_id_start is not _MISSING:
                exec_start_value = _coerce_positive_int(
                    cast(SupportsFloat | str | bytes | None, exec_id_start),
                    minimum=1,
                )
                if exec_start_value is not None:
                    self._exec_sequence_start = exec_start_value - 1
                    for state in self._orders.values():
                        if state.exec_sequence <= self._exec_sequence_start:
                            state.exec_sequence = self._exec_sequence_start
            if exec_id_prefix is not _MISSING:
                prefix_value = _coerce_optional_str(cast(object | None, exec_id_prefix))
                self._default_exec_id_prefix = prefix_value

    def emit_order_update(
        self,
        cl_ord_id: str,
        exec_type: str,
        *,
        quantity: float | None = None,
        price: float | None = None,
        text: str | None = None,
        ord_rej_reason: str | None = None,
        cancel_reason: str | None = None,
        order_id: str | None = None,
        exec_id: str | None = None,
        account: str | None = None,
        order_type: str | None = None,
        time_in_force: str | None = None,
        transact_time: str | None = None,
        sending_time: str | None = None,
        commission: SupportsFloat | str | bytes | None = None,
        comm_type: str | None = None,
        currency: str | None = None,
        settle_type: str | None = None,
        settle_date: str | int | float | date | datetime | None = None,
        trade_date: str | int | float | date | datetime | None = None,
        order_capacity: str | None = None,
        customer_or_firm: str | None = None,
    ) -> bool:
        """Manually emit an execution update for an active order."""

        with self._lock:
            state = self._orders.get(cl_ord_id)
        if state is None:
            return False

        commission_value = _coerce_optional_float(
            cast(SupportsFloat | str | bytes | None, commission)
        )
        if commission_value is not None and commission_value < 0.0:
            commission_value = 0.0
        account_value = _coerce_optional_str(account)
        order_type_value = _coerce_optional_str(order_type)
        time_in_force_value = _coerce_optional_str(time_in_force)
        comm_type_value = _coerce_optional_str(comm_type)
        currency_value = _coerce_optional_str(currency)
        settle_type_value = _coerce_optional_str(settle_type)
        settle_date_value = _coerce_optional_fix_date(cast(object | None, settle_date))
        trade_date_value = _coerce_optional_fix_date(cast(object | None, trade_date))
        order_capacity_value = _coerce_optional_str(order_capacity)
        customer_or_firm_value = _coerce_optional_str(customer_or_firm)

        info = self._build_order_info(
            state,
            exec_type,
            fill_qty=quantity,
            fill_px=price,
            order_id_override=order_id,
            exec_id_override=exec_id,
            account_override=account_value,
            order_type_override=order_type_value,
            time_in_force_override=time_in_force_value,
            text_override=text,
            ord_rej_reason_override=ord_rej_reason,
            cancel_reason_override=cancel_reason,
            transact_time_override=transact_time,
            sending_time_override=sending_time,
            commission_override=commission_value,
            comm_type_override=comm_type_value,
            currency_override=currency_value,
            settle_type_override=settle_type_value,
            settle_date_override=settle_date_value,
            trade_date_override=trade_date_value,
            order_capacity_override=order_capacity_value,
            customer_or_firm_override=customer_or_firm_value,
        )
        self._dispatch(info, exec_type)
        if exec_type in {"F", "4", "8"}:
            self._clear_state(cl_ord_id)
        return True

    def _resolve_order_id(self, state: _OrderState, override: str | None = None) -> str:
        if override:
            if state.order_id != override:
                state.order_id = override
                state.exec_sequence = self._exec_sequence_start
        if not state.order_id:
            sequence = next(self._order_id_sequence)
            state.order_id = f"{self._order_id_prefix}-{sequence:0{self._order_id_padding}d}"
            state.exec_sequence = self._exec_sequence_start
        return state.order_id

    def _next_exec_id(self, state: _OrderState, *, override: str | None = None) -> str:
        state.exec_sequence += 1
        if override:
            return override
        explicit_prefix = state.exec_id_prefix or self._default_exec_id_prefix
        if explicit_prefix:
            prefix = explicit_prefix
        else:
            prefix = state.order_id or state.cl_ord_id or "MOCK"
        suffix = f"{state.exec_sequence:03d}"
        if explicit_prefix:
            return f"{prefix}-{suffix}"
        return f"{prefix}-EXEC-{suffix}"

    def _dispatch(self, info: MockOrderInfo, exec_type: str) -> None:
        self._telemetry.record(
            "order_execution",
            cl_ord_id=info.cl_ord_id,
            exec_type=exec_type,
            sequence=len(info.executions),
            ord_status=info.ord_status,
            last_qty=info.last_qty,
            last_px=info.last_px,
            cum_qty=info.cum_qty,
            leaves_qty=info.leaves_qty,
            avg_px=info.avg_px,
            order_id=info.order_id,
            exec_id=info.exec_id,
            text=info.text,
            ord_rej_reason=info.ord_rej_reason,
            cancel_reason=info.cancel_reason,
            transact_time=info.transact_time,
            sending_time=info.sending_time,
            account=info.account,
            order_type=info.order_type,
            time_in_force=info.time_in_force,
            last_commission=info.last_commission,
            cum_commission=info.cum_commission,
            comm_type=info.comm_type,
            currency=info.currency,
            settle_type=info.settle_type,
            settle_date=info.settle_date,
            trade_date=info.trade_date,
            order_capacity=info.order_capacity,
            customer_or_firm=info.customer_or_firm,
        )
        for cb in self._order_cbs:
            try:
                cb(info)
            except Exception:  # pragma: no cover - defensive guard for tests
                logger.exception("Mock order callback raised")

    def _ensure_state(self, msg: object, *, force_update: bool = False) -> _OrderState:
        sentinel = object()

        def _get_field(*names: str) -> object | None:
            if isinstance(msg, Mapping):
                mapping = cast(Mapping[str, object], msg)
                for name in names:
                    if name in mapping:
                        return mapping[name]
            for name in names:
                value = getattr(msg, name, sentinel)
                if value is not sentinel:
                    return value
            return None

        cl_ord_id = _coerce_optional_str(_get_field("cl_ord_id"), default="TEST") or "TEST"
        original_cl_ord_id = _coerce_optional_str(
            _get_field(
                "original_cl_ord_id",
                "orig_cl_ord_id",
                "origClOrdID",
                "orig_cl_ord_id",
            )
        )
        symbol = _coerce_optional_str(_get_field("symbol"), default="TEST") or "TEST"
        side = _coerce_optional_str(_get_field("side"), default="1") or "1"
        qty = _coerce_float(cast(SupportsFloat | str | None, _get_field("quantity", "last_qty")))
        px = _coerce_float(cast(SupportsFloat | str | None, _get_field("price", "last_px")))
        ratio_override = _coerce_ratio(
            cast(
                SupportsFloat | str | bytes | None,
                _get_field("mock_partial_fill_ratio", "partial_fill_ratio"),
            )
        )
        price_override = _coerce_optional_float(
            cast(
                SupportsFloat | str | bytes | None,
                _get_field("mock_fill_price", "fill_price", "last_px"),
            )
        )
        order_id_override = _coerce_optional_str(_get_field("mock_order_id", "order_id"))
        exec_prefix_override = _coerce_optional_str(
            _get_field("mock_exec_id_prefix", "exec_id_prefix")
        )
        reject_reason_override = _coerce_optional_str(
            _get_field("mock_reject_reason", "reject_reason")
        )
        cancel_reason_override = _coerce_optional_str(
            _get_field("mock_cancel_reason", "cancel_reason")
        )
        reject_text_override = _coerce_optional_str(_get_field("mock_reject_text", "reject_text"))
        cancel_text_override = _coerce_optional_str(_get_field("mock_cancel_text", "cancel_text"))
        generic_text_override = _coerce_optional_str(_get_field("mock_text", "text"))
        if reject_text_override is None:
            reject_text_override = generic_text_override
        if cancel_text_override is None:
            cancel_text_override = generic_text_override
        transact_time_override = _coerce_optional_str(
            _get_field("mock_transact_time", "transact_time", "transactTime")
        )
        sending_time_override = _coerce_optional_str(
            _get_field("mock_sending_time", "sending_time", "sendingTime")
        )
        if sending_time_override is None:
            sending_time_override = transact_time_override
        account_override = _coerce_optional_str(_get_field("mock_account", "account", "acct"))
        order_type_override = _coerce_optional_str(
            _get_field("mock_order_type", "order_type", "ord_type", "orderType")
        )
        time_in_force_override = _coerce_optional_str(
            _get_field(
                "mock_time_in_force",
                "time_in_force",
                "timeInForce",
                "tif",
            )
        )
        order_capacity_override = _coerce_optional_str(
            _get_field("mock_order_capacity", "order_capacity", "capacity")
        )
        customer_or_firm_override = _coerce_optional_str(
            _get_field(
                "mock_customer_or_firm",
                "customer_or_firm",
                "cust_or_firm",
                "customerOrFirm",
            )
        )
        commission_override = _coerce_optional_float(
            cast(
                SupportsFloat | str | bytes | None,
                _get_field("mock_commission", "commission"),
            )
        )
        if commission_override is not None and commission_override < 0.0:
            commission_override = 0.0
        commission_total_override = _coerce_optional_float(
            cast(
                SupportsFloat | str | bytes | None,
                _get_field("mock_cum_commission", "cum_commission"),
            )
        )
        if commission_total_override is not None and commission_total_override < 0.0:
            commission_total_override = 0.0
        comm_type_override = _coerce_optional_str(
            _get_field(
                "mock_commission_type",
                "commission_type",
                "comm_type",
            )
        )
        currency_override = _coerce_optional_str(_get_field("mock_currency", "currency"))
        settle_type_override = _coerce_optional_str(_get_field("mock_settle_type", "settle_type"))
        settle_date_override = _coerce_optional_fix_date(
            _get_field("mock_settle_date", "settle_date", "settleDate")
        )
        trade_date_override = _coerce_optional_fix_date(
            _get_field("mock_trade_date", "trade_date", "tradeDate")
        )
        auto_complete_override = _coerce_optional_bool(
            _get_field("mock_auto_complete", "auto_complete")
        )

        with self._lock:
            state: _OrderState | None
            if (
                original_cl_ord_id
                and original_cl_ord_id != cl_ord_id
                and original_cl_ord_id in self._orders
            ):
                state = self._orders.pop(original_cl_ord_id)
                state.cl_ord_id = cl_ord_id
                self._orders[cl_ord_id] = state
            else:
                state = self._orders.get(cl_ord_id)
            if state is None:
                self._completed_history.pop(cl_ord_id, None)
                self._completed_info.pop(cl_ord_id, None)
                account_value = (
                    account_override if account_override is not None else self._default_account
                )
                order_type_value = (
                    order_type_override
                    if order_type_override is not None
                    else self._default_order_type
                )
                time_in_force_value = (
                    time_in_force_override
                    if time_in_force_override is not None
                    else self._default_time_in_force
                )
                commission_default = (
                    commission_override
                    if commission_override is not None
                    else self._default_commission
                )
                comm_type_value = (
                    comm_type_override
                    if comm_type_override is not None
                    else self._default_commission_type
                )
                currency_value = (
                    currency_override
                    if currency_override is not None
                    else self._default_commission_currency
                )
                settle_type_value = (
                    settle_type_override
                    if settle_type_override is not None
                    else self._default_settle_type
                )
                settle_date_value = (
                    settle_date_override
                    if settle_date_override is not None
                    else self._default_settle_date
                )
                trade_date_value = (
                    trade_date_override
                    if trade_date_override is not None
                    else self._default_trade_date
                )
                order_capacity_value = (
                    order_capacity_override
                    if order_capacity_override is not None
                    else self._default_order_capacity
                )
                customer_or_firm_value = (
                    customer_or_firm_override
                    if customer_or_firm_override is not None
                    else self._default_customer_or_firm
                )
                state = _OrderState(
                    cl_ord_id=cl_ord_id,
                    symbol=symbol,
                    side=side,
                    orig_qty=qty,
                    price=px,
                    partial_fill_ratio=ratio_override,
                    fill_price_override=price_override,
                    auto_complete=(
                        auto_complete_override if auto_complete_override is not None else True
                    ),
                    order_id=order_id_override,
                    exec_id_prefix=exec_prefix_override,
                    reject_reason=reject_reason_override,
                    cancel_reason=cancel_reason_override,
                    reject_text=reject_text_override,
                    cancel_text=cancel_text_override,
                    transact_time_override=transact_time_override,
                    sending_time_override=sending_time_override,
                    account=account_value,
                    order_type=order_type_value,
                    time_in_force=time_in_force_value,
                    commission_default=commission_default,
                    commission_total=(
                        commission_total_override if commission_total_override is not None else 0.0
                    ),
                    comm_type=comm_type_value,
                    currency=currency_value,
                    settle_type=settle_type_value,
                    settle_date=settle_date_value,
                    trade_date=trade_date_value,
                    order_capacity=order_capacity_value,
                    customer_or_firm=customer_or_firm_value,
                )
                self._orders[cl_ord_id] = state
                if commission_total_override is not None:
                    state.last_commission = commission_total_override
            else:
                if qty > 0.0 and (force_update or state.orig_qty <= 0.0):
                    state.orig_qty = qty
                if px > 0.0 and (force_update or state.price <= 0.0):
                    state.price = px
                if symbol and state.symbol == "TEST":
                    state.symbol = symbol
                if side and state.side == "1":
                    state.side = side
                if ratio_override is not None:
                    state.partial_fill_ratio = ratio_override
                if price_override is not None:
                    state.fill_price_override = price_override
                if order_id_override is not None:
                    state.order_id = order_id_override
                    state.exec_sequence = 0
                if exec_prefix_override is not None:
                    state.exec_id_prefix = exec_prefix_override
                    state.exec_sequence = 0
                if reject_reason_override is not None:
                    state.reject_reason = reject_reason_override
                if cancel_reason_override is not None:
                    state.cancel_reason = cancel_reason_override
                if reject_text_override is not None:
                    state.reject_text = reject_text_override
                if cancel_text_override is not None:
                    state.cancel_text = cancel_text_override
                if transact_time_override is not None:
                    state.transact_time_override = transact_time_override
                if sending_time_override is not None:
                    state.sending_time_override = sending_time_override
                if account_override is not None:
                    state.account = account_override
                elif state.account is None and self._default_account is not None:
                    state.account = self._default_account
                if order_type_override is not None:
                    state.order_type = order_type_override
                elif state.order_type is None and self._default_order_type is not None:
                    state.order_type = self._default_order_type
                if time_in_force_override is not None:
                    state.time_in_force = time_in_force_override
                elif state.time_in_force is None and self._default_time_in_force is not None:
                    state.time_in_force = self._default_time_in_force
                if commission_override is not None:
                    state.commission_default = commission_override
                elif state.commission_default is None and self._default_commission is not None:
                    state.commission_default = self._default_commission
                if commission_total_override is not None:
                    state.commission_total = commission_total_override
                    state.last_commission = commission_total_override
                if comm_type_override is not None:
                    state.comm_type = comm_type_override
                elif state.comm_type is None and self._default_commission_type is not None:
                    state.comm_type = self._default_commission_type
                if currency_override is not None:
                    state.currency = currency_override
                elif state.currency is None and self._default_commission_currency is not None:
                    state.currency = self._default_commission_currency
                if settle_type_override is not None:
                    state.settle_type = settle_type_override
                elif state.settle_type is None and self._default_settle_type is not None:
                    state.settle_type = self._default_settle_type
                if settle_date_override is not None:
                    state.settle_date = settle_date_override
                elif state.settle_date is None and self._default_settle_date is not None:
                    state.settle_date = self._default_settle_date
                if trade_date_override is not None:
                    state.trade_date = trade_date_override
                elif state.trade_date is None and self._default_trade_date is not None:
                    state.trade_date = self._default_trade_date
                if order_capacity_override is not None:
                    state.order_capacity = order_capacity_override
                elif state.order_capacity is None and self._default_order_capacity is not None:
                    state.order_capacity = self._default_order_capacity
                if customer_or_firm_override is not None:
                    state.customer_or_firm = customer_or_firm_override
                elif state.customer_or_firm is None and self._default_customer_or_firm is not None:
                    state.customer_or_firm = self._default_customer_or_firm
                if auto_complete_override is not None:
                    state.auto_complete = auto_complete_override
        assert state is not None
        return state

    def _estimate_partial_qty(self, state: _OrderState) -> float:
        with self._lock:
            remaining = state.remaining_qty
            if remaining <= 0.0:
                return 0.0
            ratio = (
                state.partial_fill_ratio
                if state.partial_fill_ratio is not None
                else self._partial_fill_ratio
            )
            target = max(remaining * ratio, 0.0)
            return min(target, remaining)

    def _estimate_remaining_qty(self, state: _OrderState) -> float:
        with self._lock:
            return state.remaining_qty

    def _calculate_ratio_qty(
        self,
        state: _OrderState,
        *,
        quantity_ratio: float | None = None,
        remaining_ratio: float | None = None,
    ) -> float | None:
        with self._lock:
            remaining = state.remaining_qty
            orig_qty = state.orig_qty

        ratio_qty: float | None = None
        if remaining_ratio is not None:
            ratio_qty = remaining * remaining_ratio
        elif quantity_ratio is not None:
            ratio_qty = orig_qty * quantity_ratio

        if ratio_qty is None:
            return None

        if ratio_qty > remaining:
            ratio_qty = remaining
        if ratio_qty <= 0.0:
            return 0.0
        return ratio_qty

    def _build_order_info(
        self,
        state: _OrderState,
        exec_type: str,
        *,
        fill_qty: float | None = None,
        fill_px: float | None = None,
        order_id_override: str | None = None,
        exec_id_override: str | None = None,
        account_override: str | None = None,
        order_type_override: str | None = None,
        time_in_force_override: str | None = None,
        text_override: str | None = None,
        ord_rej_reason_override: str | None = None,
        cancel_reason_override: str | None = None,
        transact_time_override: str | None = None,
        sending_time_override: str | None = None,
        commission_override: float | None = None,
        comm_type_override: str | None = None,
        currency_override: str | None = None,
        settle_type_override: str | None = None,
        settle_date_override: str | None = None,
        trade_date_override: str | None = None,
        order_capacity_override: str | None = None,
        customer_or_firm_override: str | None = None,
    ) -> MockOrderInfo:
        commission_total: float = 0.0
        last_commission: float = 0.0
        comm_type_value: str | None = None
        currency_value: str | None = None
        settle_type_value: str | None = None
        settle_date_value: str | None = None
        trade_date_value: str | None = None
        order_capacity_value: str | None = None
        customer_or_firm_value: str | None = None
        with self._lock:
            orig_qty = state.orig_qty
            cum_qty = state.cum_qty
            price = state.price
            order_px_value = state.price
            symbol = state.symbol
            side = state.side
            account_value = state.account
            order_type_value = state.order_type
            time_in_force_value = state.time_in_force
            if account_override is not None:
                account_value = account_override
                state.account = account_override
            elif account_value is None and self._default_account is not None:
                account_value = self._default_account
                state.account = account_value
            if order_type_override is not None:
                order_type_value = order_type_override
                state.order_type = order_type_override
            elif order_type_value is None and self._default_order_type is not None:
                order_type_value = self._default_order_type
                state.order_type = order_type_value
            if time_in_force_override is not None:
                time_in_force_value = time_in_force_override
                state.time_in_force = time_in_force_override
            elif time_in_force_value is None and self._default_time_in_force is not None:
                time_in_force_value = self._default_time_in_force
                state.time_in_force = time_in_force_value
            settle_type_value = state.settle_type
            settle_date_value = state.settle_date
            trade_date_value = state.trade_date
            order_capacity_value = state.order_capacity
            customer_or_firm_value = state.customer_or_firm
            ratio = (
                state.partial_fill_ratio
                if state.partial_fill_ratio is not None
                else self._partial_fill_ratio
            )
            order_id = self._resolve_order_id(state, order_id_override)
            exec_id = self._next_exec_id(state, override=exec_id_override)
            transact_time_value = transact_time_override if transact_time_override else None
            if transact_time_value is None:
                transact_time_value = state.transact_time_override
            sending_time_value = sending_time_override if sending_time_override else None
            if sending_time_value is None:
                sending_time_value = state.sending_time_override
            generated_time: str | None = None
            if transact_time_value is None or sending_time_value is None:
                generated_time = self._timestamp_factory()
            if transact_time_value is None:
                transact_time_value = sending_time_value or generated_time
            if sending_time_value is None:
                sending_time_value = transact_time_value or generated_time
            if exec_type in {"1", "F"} and fill_px is None:
                fill_px = state.fill_price_override
            if fill_px is not None:
                price = fill_px

            text_value = text_override
            if text_value is None:
                if exec_type == "8":
                    text_value = state.reject_text
                elif exec_type == "4":
                    text_value = state.cancel_text

            ord_rej_reason_value = ord_rej_reason_override
            if ord_rej_reason_value is None and exec_type == "8":
                ord_rej_reason_value = state.reject_reason

            cancel_reason_value = cancel_reason_override
            if cancel_reason_value is None and exec_type == "4":
                cancel_reason_value = state.cancel_reason

            if exec_type == "1":
                remaining = max(orig_qty - cum_qty, 0.0)
                requested = fill_qty if fill_qty is not None else remaining * ratio
                fill = max(min(requested, remaining), 0.0)
                if fill > 0.0:
                    cum_qty = min(cum_qty + fill, orig_qty)
                    state.cum_qty = cum_qty
                leaves = max(orig_qty - cum_qty, 0.0)
                ord_status = "1" if leaves > 0.0 else "2"
            elif exec_type == "F":
                remaining = max(orig_qty - cum_qty, 0.0)
                requested = fill_qty if fill_qty is not None and fill_qty > 0.0 else remaining
                fill = max(min(requested, remaining), 0.0)
                if fill > 0.0:
                    cum_qty = min(cum_qty + fill, orig_qty)
                    state.cum_qty = cum_qty
                leaves = max(orig_qty - cum_qty, 0.0)
                ord_status = "2" if leaves <= 0.0 else "1"
            elif exec_type == "4":
                fill = 0.0
                leaves = 0.0
                ord_status = "4"
            elif exec_type == "8":
                fill = 0.0
                leaves = 0.0
                cum_qty = 0.0
                state.cum_qty = cum_qty
                ord_status = "8"
            else:
                fill = 0.0
                leaves = max(orig_qty - cum_qty, 0.0)
                ord_status = "0"

            if fill > 0.0:
                notional_price = price if price is not None else state.price
                state.filled_notional += max(notional_price, 0.0) * fill
            elif exec_type == "8":
                state.filled_notional = 0.0

            avg_px = state.filled_notional / cum_qty if cum_qty > 0.0 else 0.0

            comm_type_result = state.comm_type
            if comm_type_override is not None:
                comm_type_result = comm_type_override
                state.comm_type = comm_type_override
            elif comm_type_result is None and self._default_commission_type is not None:
                comm_type_result = self._default_commission_type
                state.comm_type = comm_type_result

            currency_result = state.currency
            if currency_override is not None:
                currency_result = currency_override
                state.currency = currency_override
            elif currency_result is None and self._default_commission_currency is not None:
                currency_result = self._default_commission_currency
                state.currency = currency_result

            if settle_type_override is not None:
                settle_type_value = settle_type_override
                state.settle_type = settle_type_override
            elif settle_type_value is None and self._default_settle_type is not None:
                settle_type_value = self._default_settle_type
                state.settle_type = settle_type_value

            if settle_date_override is not None:
                settle_date_value = settle_date_override
                state.settle_date = settle_date_override
            elif settle_date_value is None and self._default_settle_date is not None:
                settle_date_value = self._default_settle_date
                state.settle_date = settle_date_value
            if trade_date_override is not None:
                trade_date_value = trade_date_override
                state.trade_date = trade_date_override
            elif trade_date_value is None and self._default_trade_date is not None:
                trade_date_value = self._default_trade_date
                state.trade_date = trade_date_value

            if order_capacity_override is not None:
                order_capacity_value = order_capacity_override
                state.order_capacity = order_capacity_override
            elif order_capacity_value is None and self._default_order_capacity is not None:
                order_capacity_value = self._default_order_capacity
                state.order_capacity = order_capacity_value

            if customer_or_firm_override is not None:
                customer_or_firm_value = customer_or_firm_override
                state.customer_or_firm = customer_or_firm_override
            elif customer_or_firm_value is None and self._default_customer_or_firm is not None:
                customer_or_firm_value = self._default_customer_or_firm
                state.customer_or_firm = customer_or_firm_value

            commission_increment = 0.0
            if commission_override is not None:
                commission_increment = max(commission_override, 0.0)
                state.commission_default = commission_override
            elif fill > 0.0 and state.commission_default is not None:
                commission_increment = max(state.commission_default, 0.0)

            if commission_increment != 0.0 or commission_override is not None:
                state.commission_total = max(state.commission_total + commission_increment, 0.0)
                state.last_commission = commission_increment
            else:
                state.last_commission = 0.0

            commission_total = state.commission_total
            last_commission = state.last_commission
            comm_type_value = comm_type_result
            currency_value = currency_result

        execution: ExecutionRecordPayload = ExecutionRecordPayload(
            exec_type=exec_type,
            ord_status=ord_status,
            last_px=price,
            last_qty=fill,
            cum_qty=cum_qty,
            leaves_qty=leaves,
            order_id=order_id,
            exec_id=exec_id,
        )
        if avg_px > 0.0:
            execution["avg_px"] = avg_px
        if text_value is not None:
            execution["text"] = text_value
        if ord_rej_reason_value is not None:
            execution["ord_rej_reason"] = ord_rej_reason_value
        if cancel_reason_value is not None:
            execution["cancel_reason"] = cancel_reason_value
        if transact_time_value is not None:
            execution["transact_time"] = transact_time_value
        if sending_time_value is not None:
            execution["sending_time"] = sending_time_value
        if account_value is not None:
            execution["account"] = account_value
        if order_type_value is not None:
            execution["order_type"] = order_type_value
        if time_in_force_value is not None:
            execution["time_in_force"] = time_in_force_value
        if last_commission != 0.0 or commission_override is not None:
            execution["commission"] = last_commission
        if commission_total != 0.0 or commission_override is not None:
            execution["cum_commission"] = commission_total
        if comm_type_value is not None:
            execution["comm_type"] = comm_type_value
        if currency_value is not None:
            execution["currency"] = currency_value
        if settle_type_value is not None:
            execution["settle_type"] = settle_type_value
        if settle_date_value is not None:
            execution["settle_date"] = settle_date_value
        if trade_date_value is not None:
            execution["trade_date"] = trade_date_value
        if order_capacity_value is not None:
            execution["order_capacity"] = order_capacity_value
        if customer_or_firm_value is not None:
            execution["customer_or_firm"] = customer_or_firm_value
        state.history.append(execution)
        executions_tuple = tuple(_clone_execution(payload) for payload in state.history)
        info = MockOrderInfo(
            cl_ord_id=state.cl_ord_id,
            executions=cast(Sequence[OrderExecutionRecord], executions_tuple),
            symbol=symbol,
            side=side,
            last_qty=fill,
            last_px=price,
            cum_qty=cum_qty,
            leaves_qty=leaves,
            ord_status=ord_status,
            avg_px=avg_px,
            order_id=order_id,
            exec_id=exec_id,
            orig_qty=orig_qty,
            order_px=order_px_value,
            text=text_value,
            ord_rej_reason=ord_rej_reason_value,
            cancel_reason=cancel_reason_value,
            transact_time=transact_time_value,
            sending_time=sending_time_value,
            account=account_value,
            order_type=order_type_value,
            time_in_force=time_in_force_value,
            last_commission=last_commission,
            cum_commission=commission_total,
            comm_type=comm_type_value,
            currency=currency_value,
            settle_type=settle_type_value,
            settle_date=settle_date_value,
            trade_date=trade_date_value,
            order_capacity=order_capacity_value,
            customer_or_firm=customer_or_firm_value,
        )
        state.last_info = _clone_order_info(info)

        return info

    def _clear_state(self, cl_ord_id: str) -> None:
        with self._lock:
            state = self._orders.pop(cl_ord_id, None)
            if state is None:
                return
            history_snapshot = tuple(_clone_execution(payload) for payload in state.history)
            self._completed_history[cl_ord_id] = history_snapshot
            if state.last_info is not None:
                self._completed_info[cl_ord_id] = _clone_order_info(state.last_info)
            else:
                self._completed_info.pop(cl_ord_id, None)

        last_info = state.last_info
        last_execution = state.history[-1] if state.history else None
        account_value = state.account
        order_type_value = state.order_type
        time_in_force_value = state.time_in_force
        settle_type_value = state.settle_type
        settle_date_value = state.settle_date
        trade_date_value = state.trade_date
        if last_info is not None:
            order_id_value = last_info.order_id or state.order_id or ""
            final_status = last_info.ord_status
            cum_qty = last_info.cum_qty
            leaves_qty = last_info.leaves_qty
            avg_px = last_info.avg_px
            text_value = last_info.text
            ord_rej_reason_value = last_info.ord_rej_reason
            cancel_reason_value = last_info.cancel_reason
            transact_time_value = last_info.transact_time
            sending_time_value = last_info.sending_time
            account_value = last_info.account
            order_type_value = last_info.order_type
            time_in_force_value = last_info.time_in_force
            last_commission_value = last_info.last_commission
            cum_commission_value = last_info.cum_commission
            comm_type_value = last_info.comm_type
            currency_value = last_info.currency
            settle_type_value = last_info.settle_type
            settle_date_value = last_info.settle_date
            trade_date_value = last_info.trade_date
        else:
            order_id_value = state.order_id or ""
            final_status = "0"
            cum_qty = state.cum_qty
            leaves_qty = max(state.orig_qty - state.cum_qty, 0.0)
            avg_px = state.filled_notional / cum_qty if cum_qty > 0.0 else 0.0
            text_value = state.reject_text or state.cancel_text
            ord_rej_reason_value = state.reject_reason
            cancel_reason_value = state.cancel_reason
            transact_time_value = state.transact_time_override
            sending_time_value = state.sending_time_override
            last_commission_value = state.last_commission
            cum_commission_value = state.commission_total
            comm_type_value = state.comm_type
            currency_value = state.currency
            settle_type_value = state.settle_type
            settle_date_value = state.settle_date
            trade_date_value = state.trade_date

        last_exec_type: str | None = None
        if last_execution is not None:
            exec_type_value = last_execution.get("exec_type")
            if isinstance(exec_type_value, bytes):
                last_exec_type = exec_type_value.decode("ascii", "ignore")
            elif exec_type_value is not None:
                last_exec_type = str(exec_type_value)

        details: dict[str, object] = {
            "cl_ord_id": cl_ord_id,
            "order_id": order_id_value,
            "final_status": final_status,
            "cum_qty": cum_qty,
            "leaves_qty": leaves_qty,
            "avg_px": avg_px,
        }
        if last_exec_type is not None:
            details["last_exec_type"] = last_exec_type
        if text_value is not None:
            details["text"] = text_value
        if ord_rej_reason_value is not None:
            details["ord_rej_reason"] = ord_rej_reason_value
        if cancel_reason_value is not None:
            details["cancel_reason"] = cancel_reason_value
        if transact_time_value is not None:
            details["transact_time"] = transact_time_value
        if sending_time_value is not None:
            details["sending_time"] = sending_time_value
        if account_value is not None:
            details["account"] = account_value
        if order_type_value is not None:
            details["order_type"] = order_type_value
        if time_in_force_value is not None:
            details["time_in_force"] = time_in_force_value
        if last_commission_value:
            details["last_commission"] = last_commission_value
        if cum_commission_value:
            details["cum_commission"] = cum_commission_value
        if comm_type_value is not None:
            details["comm_type"] = comm_type_value
        if currency_value is not None:
            details["currency"] = currency_value
        if settle_type_value is not None:
            details["settle_type"] = settle_type_value
        if settle_date_value is not None:
            details["settle_date"] = settle_date_value
        if trade_date_value is not None:
            details["trade_date"] = trade_date_value

        self._telemetry.record("order_complete", **details)

    def get_order_history(self, cl_ord_id: str) -> list[ExecutionRecordPayload]:
        with self._lock:
            state = self._orders.get(cl_ord_id)
            if state is not None:
                return [_clone_execution(payload) for payload in state.history]
            history = self._completed_history.get(cl_ord_id)
        if history is None:
            return []
        return [_clone_execution(payload) for payload in history]

    def get_last_order_info(self, cl_ord_id: str) -> MockOrderInfo | None:
        with self._lock:
            state = self._orders.get(cl_ord_id)
            if state is not None and state.last_info is not None:
                return _clone_order_info(state.last_info)
            info = self._completed_info.get(cl_ord_id)
        if info is None:
            return None
        return _clone_order_info(info)

    def _coerce_execution_step(self, raw: object) -> _ResolvedExecutionStep | None:
        exec_type: object | None = None
        quantity: object | None = None
        delay: object | None = None
        price: object | None = None
        order_id: object | None = None
        exec_id: object | None = None
        account: object | None = None
        order_type: object | None = None
        time_in_force: object | None = None
        text: object | None = None
        ord_rej_reason: object | None = None
        cancel_reason: object | None = None
        quantity_ratio: object | None = None
        remaining_ratio: object | None = None
        repeat: object | None = None
        transact_time: object | None = None
        sending_time: object | None = None
        commission: object | None = None
        comm_type: object | None = None
        currency: object | None = None
        settle_type: object | None = None
        settle_date: object | None = None
        trade_date: object | None = None
        order_capacity: object | None = None
        customer_or_firm: object | None = None

        if isinstance(raw, MockExecutionStep):
            exec_type = raw.exec_type
            quantity = raw.quantity
            delay = raw.delay
            price = raw.price
            order_id = raw.order_id
            exec_id = raw.exec_id
            account = raw.account
            order_type = raw.order_type
            time_in_force = raw.time_in_force
            text = raw.text
            ord_rej_reason = raw.ord_rej_reason
            cancel_reason = raw.cancel_reason
            if raw.quantity_ratio is not None:
                quantity_ratio = raw.quantity_ratio
            else:
                quantity_ratio = raw.ratio
            remaining_ratio = raw.remaining_ratio
            repeat = raw.repeat
            transact_time = raw.transact_time
            sending_time = raw.sending_time
            commission = raw.commission
            comm_type = raw.comm_type
            currency = raw.currency
            settle_type = raw.settle_type
            settle_date = raw.settle_date
            trade_date = raw.trade_date
            order_capacity = raw.order_capacity
            customer_or_firm = raw.customer_or_firm
        elif isinstance(raw, tuple):
            if raw:
                exec_type = raw[0]
            if len(raw) > 1:
                quantity = raw[1]
            if len(raw) > 2:
                delay = raw[2]
            if len(raw) > 3:
                price = raw[3]
            if len(raw) > 4:
                exec_id = raw[4]
            if len(raw) > 5:
                order_id = raw[5]
            if len(raw) > 6:
                account = raw[6]
            if len(raw) > 7:
                order_type = raw[7]
            if len(raw) > 8:
                time_in_force = raw[8]
            if len(raw) > 9:
                text = raw[9]
            if len(raw) > 10:
                ord_rej_reason = raw[10]
            if len(raw) > 11:
                cancel_reason = raw[11]
            if len(raw) > 12:
                commission = raw[12]
            if len(raw) > 13:
                comm_type = raw[13]
            if len(raw) > 14:
                currency = raw[14]
            if len(raw) > 15:
                settle_type = raw[15]
            if len(raw) > 16:
                settle_date = raw[16]
            if len(raw) > 17:
                trade_date = raw[17]
            if len(raw) > 18:
                order_capacity = raw[18]
            if len(raw) > 19:
                customer_or_firm = raw[19]
        elif isinstance(raw, dict):
            exec_type = raw.get("exec_type") or raw.get("type")
            quantity = raw.get("quantity", raw.get("qty"))
            delay = raw.get("delay", raw.get("sleep"))
            price = raw.get("price", raw.get("px"))
            order_id = raw.get("order_id") or raw.get("mock_order_id")
            exec_id = raw.get("exec_id") or raw.get("mock_exec_id")
            account = raw.get("account") or raw.get("mock_account")
            order_type = raw.get("order_type") or raw.get("mock_order_type") or raw.get("ord_type")
            time_in_force = (
                raw.get("time_in_force")
                or raw.get("mock_time_in_force")
                or raw.get("timeInForce")
                or raw.get("tif")
            )
            text = raw.get("text") or raw.get("message")
            ord_rej_reason = raw.get("ord_rej_reason") or raw.get("reject_reason")
            cancel_reason = raw.get("cancel_reason")
            transact_time = (
                raw.get("transact_time") or raw.get("transactTime") or raw.get("mock_transact_time")
            )
            sending_time = (
                raw.get("sending_time") or raw.get("sendingTime") or raw.get("mock_sending_time")
            )
            commission = raw.get("commission") or raw.get("fee") or raw.get("last_commission")
            comm_type = raw.get("comm_type") or raw.get("commission_type")
            currency = raw.get("currency")
            settle_type = raw.get("settle_type") or raw.get("mock_settle_type")
            settle_date = (
                raw.get("settle_date") or raw.get("settleDate") or raw.get("mock_settle_date")
            )
            trade_date = raw.get("trade_date") or raw.get("tradeDate") or raw.get("mock_trade_date")
            order_capacity = (
                raw.get("order_capacity") or raw.get("mock_order_capacity") or raw.get("capacity")
            )
            customer_or_firm = (
                raw.get("customer_or_firm")
                or raw.get("mock_customer_or_firm")
                or raw.get("cust_or_firm")
                or raw.get("customerOrFirm")
            )
            if "quantity_ratio" in raw:
                quantity_ratio = raw.get("quantity_ratio")
            elif "ratio" in raw:
                quantity_ratio = raw.get("ratio")
            elif "proportion" in raw:
                quantity_ratio = raw.get("proportion")
            if "remaining_ratio" in raw:
                remaining_ratio = raw.get("remaining_ratio")
            elif "remaining_proportion" in raw:
                remaining_ratio = raw.get("remaining_proportion")
            repeat_sentinel = object()
            repeat_candidate = raw.get("repeat", repeat_sentinel)
            if repeat_candidate is repeat_sentinel:
                repeat_candidate = raw.get("repetitions", repeat_sentinel)
            if repeat_candidate is repeat_sentinel:
                repeat_candidate = raw.get("times", repeat_sentinel)
            if repeat_candidate is repeat_sentinel:
                repeat_candidate = raw.get("count", None)
            repeat = None if repeat_candidate is repeat_sentinel else repeat_candidate
        else:
            exec_type = getattr(raw, "exec_type", None) or getattr(raw, "type", None)
            quantity = getattr(raw, "quantity", getattr(raw, "qty", None))
            delay = getattr(raw, "delay", getattr(raw, "sleep", None))
            price = getattr(raw, "price", getattr(raw, "px", None))
            order_id = getattr(raw, "order_id", getattr(raw, "mock_order_id", None))
            exec_id = getattr(raw, "exec_id", getattr(raw, "mock_exec_id", None))
            account = getattr(raw, "account", getattr(raw, "mock_account", None))
            order_type = getattr(
                raw,
                "order_type",
                getattr(raw, "mock_order_type", getattr(raw, "ord_type", None)),
            )
            time_in_force = getattr(
                raw,
                "time_in_force",
                getattr(
                    raw,
                    "mock_time_in_force",
                    getattr(raw, "timeInForce", getattr(raw, "tif", None)),
                ),
            )
            text = getattr(raw, "text", getattr(raw, "message", None))
            ord_rej_reason = getattr(
                raw,
                "ord_rej_reason",
                getattr(raw, "reject_reason", None),
            )
            cancel_reason = getattr(raw, "cancel_reason", None)
            transact_time = getattr(
                raw,
                "transact_time",
                getattr(
                    raw,
                    "transactTime",
                    getattr(raw, "mock_transact_time", None),
                ),
            )
            sending_time = getattr(
                raw,
                "sending_time",
                getattr(
                    raw,
                    "sendingTime",
                    getattr(raw, "mock_sending_time", None),
                ),
            )
            commission = getattr(
                raw,
                "commission",
                getattr(raw, "fee", getattr(raw, "last_commission", None)),
            )
            comm_type = getattr(
                raw,
                "comm_type",
                getattr(raw, "commission_type", None),
            )
            currency = getattr(raw, "currency", None)
            settle_type = getattr(raw, "settle_type", getattr(raw, "mock_settle_type", None))
            settle_date = getattr(
                raw,
                "settle_date",
                getattr(raw, "settleDate", getattr(raw, "mock_settle_date", None)),
            )
            trade_date = getattr(
                raw,
                "trade_date",
                getattr(raw, "tradeDate", getattr(raw, "mock_trade_date", None)),
            )
            order_capacity = getattr(
                raw,
                "order_capacity",
                getattr(raw, "mock_order_capacity", getattr(raw, "capacity", None)),
            )
            customer_or_firm = getattr(
                raw,
                "customer_or_firm",
                getattr(
                    raw,
                    "mock_customer_or_firm",
                    getattr(
                        raw,
                        "cust_or_firm",
                        getattr(raw, "customerOrFirm", None),
                    ),
                ),
            )
            sentinel = object()
            quantity_ratio_attr = getattr(raw, "quantity_ratio", sentinel)
            if quantity_ratio_attr is sentinel:
                quantity_ratio_attr = getattr(raw, "ratio", sentinel)
            if quantity_ratio_attr is sentinel:
                quantity_ratio_attr = getattr(raw, "proportion", None)
            quantity_ratio = None if quantity_ratio_attr is sentinel else quantity_ratio_attr
            remaining_ratio_attr = getattr(raw, "remaining_ratio", sentinel)
            if remaining_ratio_attr is sentinel:
                remaining_ratio_attr = getattr(raw, "remaining_proportion", None)
            remaining_ratio = None if remaining_ratio_attr is sentinel else remaining_ratio_attr
            repeat_attr = getattr(raw, "repeat", sentinel)
            if repeat_attr is sentinel:
                repeat_attr = getattr(raw, "repetitions", sentinel)
            if repeat_attr is sentinel:
                repeat_attr = getattr(raw, "times", sentinel)
            if repeat_attr is sentinel:
                repeat_attr = getattr(raw, "count", None)
            repeat = None if repeat_attr is sentinel else repeat_attr

        if not exec_type:
            return None

        exec_text = str(exec_type).upper()
        qty_value = _coerce_optional_float(cast(SupportsFloat | str | bytes | None, quantity))
        if qty_value is not None:
            qty_value = max(qty_value, 0.0)
        delay_value = _coerce_optional_float(
            cast(SupportsFloat | str | bytes | None, delay),
            default=self._execution_interval,
        )
        if delay_value is None:
            delay_value = self._execution_interval
        delay_value = max(delay_value, 0.0)
        price_value = _coerce_optional_float(cast(SupportsFloat | str | bytes | None, price))
        order_id_value = _coerce_optional_str(order_id)
        exec_id_value = _coerce_optional_str(exec_id)
        account_value = _coerce_optional_str(account)
        order_type_value = _coerce_optional_str(order_type)
        time_in_force_value = _coerce_optional_str(time_in_force)
        text_value = _coerce_optional_str(text)
        ord_rej_reason_value = _coerce_optional_str(ord_rej_reason)
        cancel_reason_value = _coerce_optional_str(cancel_reason)
        quantity_ratio_value = _coerce_ratio(
            cast(SupportsFloat | str | bytes | None, quantity_ratio)
        )
        remaining_ratio_value = _coerce_ratio(
            cast(SupportsFloat | str | bytes | None, remaining_ratio)
        )
        transact_time_value = _coerce_optional_str(transact_time)
        sending_time_value = _coerce_optional_str(sending_time)
        if sending_time_value is None:
            sending_time_value = transact_time_value
        repeat_value = _coerce_repetition(cast(SupportsFloat | str | bytes | None, repeat))
        if repeat_value is None:
            repeat_value = 1

        commission_value = _coerce_optional_float(
            cast(SupportsFloat | str | bytes | None, commission)
        )
        if commission_value is not None and commission_value < 0.0:
            commission_value = 0.0
        comm_type_value = _coerce_optional_str(comm_type)
        currency_value = _coerce_optional_str(currency)
        settle_type_value = _coerce_optional_str(settle_type)
        settle_date_value = _coerce_optional_fix_date(settle_date)
        trade_date_value = _coerce_optional_fix_date(trade_date)
        order_capacity_value = _coerce_optional_str(order_capacity)
        customer_or_firm_value = _coerce_optional_str(customer_or_firm)

        return _ResolvedExecutionStep(
            exec_type=exec_text,
            quantity=qty_value,
            delay=delay_value,
            price=price_value,
            order_id=order_id_value,
            exec_id=exec_id_value,
            account=account_value,
            order_type=order_type_value,
            time_in_force=time_in_force_value,
            text=text_value,
            ord_rej_reason=ord_rej_reason_value,
            cancel_reason=cancel_reason_value,
            quantity_ratio=quantity_ratio_value,
            remaining_ratio=remaining_ratio_value,
            repeat=repeat_value,
            transact_time=transact_time_value,
            sending_time=sending_time_value,
            commission=commission_value,
            comm_type=comm_type_value,
            currency=currency_value,
            settle_type=settle_type_value,
            settle_date=settle_date_value,
            trade_date=trade_date_value,
            order_capacity=order_capacity_value,
            customer_or_firm=customer_or_firm_value,
        )

    def _resolve_execution_plan(self, msg: object) -> list[_ResolvedExecutionStep]:
        plan_attr = getattr(msg, "mock_execution_plan", None)
        if plan_attr is None:
            plan_attr = getattr(msg, "execution_plan", None)
        if plan_attr is None or isinstance(plan_attr, (str, bytes)):
            return []

        try:
            iterator = iter(plan_attr)
        except TypeError:
            return []

        steps: list[_ResolvedExecutionStep] = []
        for raw_step in iterator:
            step = self._coerce_execution_step(raw_step)
            if step is not None:
                steps.append(step)
        return steps

    def send_message_and_track(self, msg: object) -> bool:
        replace_flag = False
        original_cl_ord_id: str | None = None

        if isinstance(msg, Mapping):
            mapping = cast(Mapping[str, object], msg)
            for key in ("replace", "cancel_replace", "amend"):
                if key in mapping and _coerce_optional_bool(mapping[key]):
                    replace_flag = True
                    break
            if not replace_flag:
                replace_flag = bool(mapping.get("orig_cl_ord_id"))
            for name in (
                "original_cl_ord_id",
                "orig_cl_ord_id",
                "origClOrdID",
                "orig_cl_ord_id",
            ):
                if name in mapping:
                    text = _coerce_optional_str(mapping[name])
                    if text:
                        original_cl_ord_id = text
                        break
        else:
            for key in ("replace", "cancel_replace", "amend"):
                if _coerce_optional_bool(getattr(msg, key, None)):
                    replace_flag = True
                    break
            if not replace_flag:
                replace_flag = bool(getattr(msg, "orig_cl_ord_id", None))
            for name in (
                "original_cl_ord_id",
                "orig_cl_ord_id",
                "origClOrdID",
                "orig_cl_ord_id",
            ):
                value = getattr(msg, name, None)
                text = _coerce_optional_str(value)
                if text:
                    original_cl_ord_id = text
                    break

        state = self._ensure_state(msg, force_update=replace_flag)
        with self._lock:
            order_id = self._resolve_order_id(state)
            self._telemetry.record(
                "order_received",
                cl_ord_id=state.cl_ord_id,
                quantity=state.orig_qty,
                price=state.price,
                partial_fill_ratio=(
                    state.partial_fill_ratio
                    if state.partial_fill_ratio is not None
                    else self._partial_fill_ratio
                ),
                fill_price=state.fill_price_override,
                order_id=order_id,
                replace=replace_flag,
                original_cl_ord_id=original_cl_ord_id,
                auto_complete=state.auto_complete,
                account=state.account,
                order_type=state.order_type,
                time_in_force=state.time_in_force,
                order_capacity=state.order_capacity,
                customer_or_firm=state.customer_or_firm,
            )

        # Respect optional flags on msg for reject/cancel flows
        if getattr(msg, "reject", False):

            def _emit_reject() -> None:
                info = self._build_order_info(state, "8")  # Reject
                self._dispatch(info, "8")
                self._clear_state(state.cl_ord_id)

            self._spawn(_emit_reject)
            return True

        if getattr(msg, "cancel", False):

            def _emit_cancel() -> None:
                info = self._build_order_info(state, "4")  # Canceled
                self._dispatch(info, "4")
                self._clear_state(state.cl_ord_id)

            self._spawn(_emit_cancel)
            return True

        execution_plan = self._resolve_execution_plan(msg)
        initial_exec_type: str | None
        run_default_flow = False
        if replace_flag:
            initial_exec_type = "5"
        else:
            initial_exec_type = None if execution_plan else "0"
            run_default_flow = not execution_plan and state.auto_complete

        # Emit scripted execution plan with optional default fallbacks
        def _emit_flow() -> None:
            completed = False

            def _dispatch_step(
                exec_type: str,
                *,
                fill_qty: float | None = None,
                fill_px: float | None = None,
                order_id_override: str | None = None,
                exec_id_override: str | None = None,
                account_override: str | None = None,
                order_type_override: str | None = None,
                time_in_force_override: str | None = None,
                text_override: str | None = None,
                ord_rej_reason_override: str | None = None,
                cancel_reason_override: str | None = None,
                transact_time_override: str | None = None,
                sending_time_override: str | None = None,
                commission_override: float | None = None,
                comm_type_override: str | None = None,
                currency_override: str | None = None,
                settle_type_override: str | None = None,
                settle_date_override: str | None = None,
                trade_date_override: str | None = None,
                order_capacity_override: str | None = None,
                customer_or_firm_override: str | None = None,
            ) -> None:
                nonlocal completed
                info = self._build_order_info(
                    state,
                    exec_type,
                    fill_qty=fill_qty,
                    fill_px=fill_px,
                    order_id_override=order_id_override,
                    exec_id_override=exec_id_override,
                    account_override=account_override,
                    order_type_override=order_type_override,
                    time_in_force_override=time_in_force_override,
                    text_override=text_override,
                    ord_rej_reason_override=ord_rej_reason_override,
                    cancel_reason_override=cancel_reason_override,
                    transact_time_override=transact_time_override,
                    sending_time_override=sending_time_override,
                    commission_override=commission_override,
                    comm_type_override=comm_type_override,
                    currency_override=currency_override,
                    settle_type_override=settle_type_override,
                    settle_date_override=settle_date_override,
                    trade_date_override=trade_date_override,
                    order_capacity_override=order_capacity_override,
                    customer_or_firm_override=customer_or_firm_override,
                )
                self._dispatch(info, exec_type)
                if exec_type in {"F", "4", "8"}:
                    completed = True

            if initial_exec_type is not None:
                _dispatch_step(initial_exec_type)

            if execution_plan:
                last_index = len(execution_plan) - 1
                for idx, step in enumerate(execution_plan):
                    repetitions = step.repeat if step.repeat > 0 else 1
                    for iteration in range(repetitions):
                        fill_qty = step.quantity
                        if fill_qty is None:
                            ratio_qty = self._calculate_ratio_qty(
                                state,
                                quantity_ratio=step.quantity_ratio,
                                remaining_ratio=step.remaining_ratio,
                            )
                            if ratio_qty is not None:
                                fill_qty = ratio_qty
                        _dispatch_step(
                            step.exec_type,
                            fill_qty=fill_qty,
                            fill_px=step.price,
                            order_id_override=step.order_id,
                            exec_id_override=step.exec_id,
                            account_override=step.account,
                            order_type_override=step.order_type,
                            time_in_force_override=step.time_in_force,
                            text_override=step.text,
                            ord_rej_reason_override=step.ord_rej_reason,
                            cancel_reason_override=step.cancel_reason,
                            transact_time_override=step.transact_time,
                            sending_time_override=step.sending_time,
                            commission_override=step.commission,
                            comm_type_override=step.comm_type,
                            currency_override=step.currency,
                            settle_type_override=step.settle_type,
                            settle_date_override=step.settle_date,
                            trade_date_override=step.trade_date,
                            order_capacity_override=step.order_capacity,
                            customer_or_firm_override=step.customer_or_firm,
                        )
                        if completed:
                            break
                        if step.delay > 0.0 and (iteration < repetitions - 1 or idx < last_index):
                            self._sleep(step.delay)
                    if completed:
                        break

                if (
                    not completed
                    and state.auto_complete
                    and not any(step.exec_type in {"F", "4", "8"} for step in execution_plan)
                ):
                    remaining_qty = self._estimate_remaining_qty(state)
                    if remaining_qty > 0.0:
                        _dispatch_step("F", fill_qty=remaining_qty)
                    else:
                        _dispatch_step("F")
            elif run_default_flow:
                self._sleep(self._execution_interval)
                partial_qty = self._estimate_partial_qty(state)
                if partial_qty > 0.0:
                    _dispatch_step("1", fill_qty=partial_qty)
                    if not completed and self._execution_interval > 0.0:
                        self._sleep(self._execution_interval)
                elif self._execution_interval > 0.0:
                    self._sleep(self._execution_interval)
                remaining_qty = self._estimate_remaining_qty(state)
                if remaining_qty > 0.0:
                    _dispatch_step("F", fill_qty=remaining_qty)
                else:
                    _dispatch_step("F")

            if completed:
                self._clear_state(state.cl_ord_id)

        self._spawn(_emit_flow)
        return True


class MockFIXManager:
    trade_connection: FIXTradeConnectionProtocol

    def __init__(
        self,
        *,
        symbol: str = "EURUSD",
        market_data_interval: float = 0.05,
        market_data_duration: float = 2.0,
        market_data_plan: Sequence[object] | None = None,
        market_data_loop: bool = False,
        partial_fill_ratio: float = 0.5,
        execution_interval: float = 0.05,
        synchronous_order_flows: bool = False,
        timestamp_factory: Callable[[], str] | None = None,
        default_account: str | None = None,
        default_order_type: str | None = None,
        default_time_in_force: str | None = None,
        default_commission: float | None = None,
        default_commission_type: str | None = None,
        default_commission_currency: str | None = None,
        default_settle_type: str | None = None,
        default_settle_date: str | int | float | date | datetime | None = None,
        default_trade_date: str | int | float | date | datetime | None = None,
        default_order_capacity: str | None = None,
        default_customer_or_firm: str | None = None,
        order_id_prefix: str | None = None,
        order_id_start: int = 1,
        order_id_padding: int = 6,
        exec_id_start: int = 1,
        exec_id_prefix: str | None = None,
    ) -> None:
        self._md_cbs: list[Callable[[str, OrderBookProtocol], None]] = []
        self._order_cbs: list[Callable[[OrderInfoProtocol], None]] = []
        self._running = False
        self._symbol = symbol
        self._market_data_interval = market_data_interval
        self._market_data_duration = market_data_duration
        self._telemetry = _TelemetryRecorder()
        self._partial_fill_ratio = max(min(partial_fill_ratio, 1.0), 0.0)
        self._execution_interval = max(execution_interval, 0.0)
        self._market_data_thread: threading.Thread | None = None
        self._plan_lock = threading.Lock()
        self._market_data_plan = self._resolve_market_data_plan(market_data_plan)
        self._market_data_loop = bool(market_data_loop)
        self._timestamp_factory = timestamp_factory or _default_timestamp
        self._trade_connection = _MockTradeConnection(
            self._order_cbs,
            self._telemetry,
            partial_fill_ratio=self._partial_fill_ratio,
            execution_interval=self._execution_interval,
            synchronous=bool(synchronous_order_flows),
            timestamp_factory=self._timestamp_factory,
            default_account=default_account,
            default_order_type=default_order_type,
            default_time_in_force=default_time_in_force,
            order_id_prefix=order_id_prefix,
            order_id_start=order_id_start,
            order_id_padding=order_id_padding,
            exec_id_start=exec_id_start,
            default_exec_id_prefix=exec_id_prefix,
            default_commission=default_commission,
            default_commission_type=default_commission_type,
            default_commission_currency=default_commission_currency,
            default_settle_type=default_settle_type,
            default_settle_date=default_settle_date,
            default_trade_date=default_trade_date,
            default_order_capacity=default_order_capacity,
            default_customer_or_firm=default_customer_or_firm,
        )
        self.trade_connection = self._trade_connection
        self.price_connection = _MockPriceConnection(self._telemetry)

    def _coerce_market_data_step(self, raw: object) -> _ResolvedMarketDataStep | None:
        if isinstance(raw, _ResolvedMarketDataStep):
            return raw
        bids_obj: object | None = None
        asks_obj: object | None = None
        delay_obj: object | None = None

        if isinstance(raw, MockMarketDataStep):
            bids_obj = raw.bids
            asks_obj = raw.asks
            delay_obj = raw.delay
        elif isinstance(raw, Mapping):
            bids_obj = (
                raw.get("bids")
                or raw.get("bid_levels")
                or raw.get("book_bids")
                or raw.get("levels")
            )
            asks_obj = (
                raw.get("asks")
                or raw.get("ask_levels")
                or raw.get("book_asks")
                or raw.get("levels_ask")
            )
            delay_obj = raw.get("delay") or raw.get("sleep") or raw.get("interval")
        elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
            if raw:
                bids_obj = raw[0]
            if len(raw) > 1:
                asks_obj = raw[1]
            if len(raw) > 2:
                delay_obj = raw[2]
        else:
            bids_obj = getattr(raw, "bids", getattr(raw, "bid_levels", None))
            asks_obj = getattr(raw, "asks", getattr(raw, "ask_levels", None))
            delay_obj = getattr(raw, "delay", getattr(raw, "sleep", None))

        bids = _normalize_market_data_side(bids_obj)
        asks = _normalize_market_data_side(asks_obj)
        delay_value = _coerce_optional_float(
            cast(SupportsFloat | str | bytes | None, delay_obj),
            default=self._market_data_interval,
        )
        if delay_value is None:
            delay_value = self._market_data_interval
        delay_value = max(delay_value, 0.0)
        return _ResolvedMarketDataStep(bids=bids, asks=asks, delay=delay_value)

    def _resolve_market_data_plan(
        self,
        plan: Sequence[object] | MockMarketDataStep | Mapping[str, object] | None,
    ) -> list[_ResolvedMarketDataStep]:
        if plan is None:
            return []
        if isinstance(plan, _ResolvedMarketDataStep):
            return [plan]
        if isinstance(plan, Mapping):
            step = self._coerce_market_data_step(plan)
            return [step] if step else []
        if isinstance(plan, (str, bytes, bytearray)):
            return []
        if isinstance(plan, Sequence) and not isinstance(plan, (str, bytes, bytearray)):
            steps: list[_ResolvedMarketDataStep] = []
            for raw in plan:
                step = self._coerce_market_data_step(raw)
                if step is not None:
                    steps.append(step)
            return steps
        step = self._coerce_market_data_step(plan)
        return [step] if step else []

    def configure_market_data_plan(
        self,
        plan: Sequence[object] | MockMarketDataStep | Mapping[str, object] | None,
        *,
        loop: bool = False,
    ) -> None:
        resolved = self._resolve_market_data_plan(plan)
        with self._plan_lock:
            self._market_data_plan = resolved
            self._market_data_loop = bool(loop)

    def configure_order_defaults(
        self,
        *,
        account: object = _MISSING,
        order_type: object = _MISSING,
        time_in_force: object = _MISSING,
        order_capacity: object = _MISSING,
        customer_or_firm: object = _MISSING,
        commission: object = _MISSING,
        commission_type: object = _MISSING,
        commission_currency: object = _MISSING,
        settle_type: object = _MISSING,
        settle_date: object = _MISSING,
        trade_date: object = _MISSING,
    ) -> None:
        """Update the default order metadata applied to new flows."""

        self._trade_connection.configure_defaults(
            account=account,
            order_type=order_type,
            time_in_force=time_in_force,
            order_capacity=order_capacity,
            customer_or_firm=customer_or_firm,
            commission=commission,
            comm_type=commission_type,
            currency=commission_currency,
            settle_type=settle_type,
            settle_date=settle_date,
            trade_date=trade_date,
        )

    def configure_id_generation(
        self,
        *,
        order_id_prefix: object = _MISSING,
        order_id_start: int | None = None,
        order_id_padding: int | None = None,
        exec_id_start: int | None = None,
        exec_id_prefix: object = _MISSING,
    ) -> None:
        """Adjust identifier sequencing for subsequent mock orders."""

        updates: dict[str, object] = {}
        if order_id_prefix is not _MISSING:
            updates["order_id_prefix"] = order_id_prefix
        if order_id_start is not None:
            updates["order_id_start"] = order_id_start
        if order_id_padding is not None:
            updates["order_id_padding"] = order_id_padding
        if exec_id_start is not None:
            updates["exec_id_start"] = exec_id_start
        if exec_id_prefix is not _MISSING:
            updates["exec_id_prefix"] = exec_id_prefix
        if updates:
            self._trade_connection.configure_id_generation(**updates)

    def add_market_data_callback(self, cb: Callable[[str, OrderBookProtocol], None]) -> None:
        self._md_cbs.append(cb)

    def add_order_callback(self, cb: Callable[[OrderInfoProtocol], None]) -> None:
        self._order_cbs.append(cb)

    def start(self) -> bool:
        if self._running:
            return True

        self._running = True
        self._telemetry.record("start", symbol=self._symbol)

        # Emit market data snapshots according to the configured plan or defaults
        def _emit_md_loop() -> None:
            snapshot_index = 0
            while self._running:
                with self._plan_lock:
                    plan = list(self._market_data_plan)
                    loop_plan = self._market_data_loop
                if plan:
                    while self._running:
                        for plan_index, step in enumerate(plan):
                            book = MockOrderBook(
                                bids=cast(Sequence[OrderBookLevelProtocol], step.bids),
                                asks=cast(Sequence[OrderBookLevelProtocol], step.asks),
                            )
                            bid_details = _levels_to_telemetry(step.bids)
                            ask_details = _levels_to_telemetry(step.asks)
                            best_bid = bid_details[0]["price"] if bid_details else None
                            best_ask = ask_details[0]["price"] if ask_details else None
                            self._telemetry.record(
                                "market_data_snapshot",
                                symbol=self._symbol,
                                bid=best_bid,
                                ask=best_ask,
                                bids=bid_details,
                                asks=ask_details,
                                snapshot_index=snapshot_index,
                                plan_index=plan_index,
                            )
                            snapshot_index += 1
                            for cb in self._md_cbs:
                                try:
                                    cb(self._symbol, book)
                                except Exception:  # pragma: no cover - defensive guard for tests
                                    logger.exception("Mock market data callback raised")
                            if not self._running:
                                break
                            if step.delay > 0.0:
                                time.sleep(step.delay)
                        self._telemetry.record(
                            "market_data_complete",
                            symbol=self._symbol,
                            steps=len(plan),
                            loop=loop_plan,
                        )
                        if not loop_plan or not self._running:
                            break
                    if loop_plan and self._running:
                        continue
                    break
                else:
                    t0 = time.time()
                    while self._running and (time.time() - t0) < self._market_data_duration:
                        bids = (MockOrderBookLevel(price=1.1, size=1000.0),)
                        asks = (MockOrderBookLevel(price=1.1002, size=1000.0),)
                        book = MockOrderBook(
                            bids=cast(Sequence[OrderBookLevelProtocol], bids),
                            asks=cast(Sequence[OrderBookLevelProtocol], asks),
                        )
                        bid_details = _levels_to_telemetry(bids)
                        ask_details = _levels_to_telemetry(asks)
                        best_bid = bid_details[0]["price"] if bid_details else None
                        best_ask = ask_details[0]["price"] if ask_details else None
                        self._telemetry.record(
                            "market_data_snapshot",
                            symbol=self._symbol,
                            bid=best_bid,
                            ask=best_ask,
                            bids=bid_details,
                            asks=ask_details,
                            snapshot_index=snapshot_index,
                            plan_index=None,
                        )
                        snapshot_index += 1
                        for cb in self._md_cbs:
                            try:
                                cb(self._symbol, book)
                            except Exception:  # pragma: no cover - defensive guard for tests
                                logger.exception("Mock market data callback raised")
                        if not self._running:
                            break
                        time.sleep(self._market_data_interval)
                    break

        self._market_data_thread = threading.Thread(
            target=_emit_md_loop,
            daemon=True,
        )
        self._market_data_thread.start()
        return True

    def stop(self) -> None:
        if not self._running:
            return

        self._running = False
        thread = self._market_data_thread
        if thread and thread.is_alive():
            thread.join(timeout=self._market_data_interval * 2)
        self._market_data_thread = None
        self._telemetry.record("stop", symbol=self._symbol)

    def snapshot_telemetry(self) -> list[TelemetryEvent]:
        """Return a copy of recent telemetry events for assertions/tests."""

        return self._telemetry.snapshot()

    def get_order_history(self, cl_ord_id: str) -> list[ExecutionRecordPayload]:
        """Retrieve the execution history for ``cl_ord_id``."""

        return self._trade_connection.get_order_history(cl_ord_id)

    def get_last_order_info(self, cl_ord_id: str) -> MockOrderInfo | None:
        """Return the most recent order info emitted for ``cl_ord_id``."""

        return self._trade_connection.get_last_order_info(cl_ord_id)

    def list_active_order_ids(self) -> list[str]:
        """Return client order identifiers currently in-flight."""

        return self._trade_connection.list_active_order_ids()

    def snapshot_active_orders(self) -> list[MockOrderInfo]:
        """Return clones of the latest info for each active order."""

        return self._trade_connection.snapshot_active_orders()

    def emit_order_update(
        self,
        cl_ord_id: str,
        exec_type: str,
        *,
        quantity: float | None = None,
        price: float | None = None,
        text: str | None = None,
        ord_rej_reason: str | None = None,
        cancel_reason: str | None = None,
        order_id: str | None = None,
        exec_id: str | None = None,
        account: str | None = None,
        order_type: str | None = None,
        time_in_force: str | None = None,
        transact_time: str | None = None,
        sending_time: str | None = None,
        commission: float | None = None,
        comm_type: str | None = None,
        currency: str | None = None,
        settle_type: str | None = None,
        settle_date: str | int | float | date | datetime | None = None,
        trade_date: str | int | float | date | datetime | None = None,
        order_capacity: str | None = None,
        customer_or_firm: str | None = None,
    ) -> bool:
        """Manually inject an execution update for an in-flight order."""

        exec_text = str(exec_type).upper()
        account_value = _coerce_optional_str(account)
        order_type_value = _coerce_optional_str(order_type)
        time_in_force_value = _coerce_optional_str(time_in_force)
        settle_type_value = _coerce_optional_str(settle_type)
        settle_date_value = _coerce_optional_fix_date(cast(object | None, settle_date))
        trade_date_value = _coerce_optional_fix_date(cast(object | None, trade_date))
        order_capacity_value = _coerce_optional_str(order_capacity)
        customer_or_firm_value = _coerce_optional_str(customer_or_firm)
        return self._trade_connection.emit_order_update(
            cl_ord_id,
            exec_text,
            quantity=quantity,
            price=price,
            text=text,
            ord_rej_reason=ord_rej_reason,
            cancel_reason=cancel_reason,
            order_id=order_id,
            exec_id=exec_id,
            account=account_value,
            order_type=order_type_value,
            time_in_force=time_in_force_value,
            transact_time=_coerce_optional_str(transact_time),
            sending_time=_coerce_optional_str(sending_time),
            commission=commission,
            comm_type=comm_type,
            currency=currency,
            settle_type=settle_type_value,
            settle_date=settle_date_value,
            trade_date=trade_date_value,
            order_capacity=order_capacity_value,
            customer_or_firm=customer_or_firm_value,
        )

    def complete_order(
        self,
        cl_ord_id: str,
        *,
        quantity: float | None = None,
        price: float | None = None,
        text: str | None = None,
        transact_time: str | None = None,
        sending_time: str | None = None,
        commission: float | None = None,
        comm_type: str | None = None,
        currency: str | None = None,
        settle_type: str | None = None,
        settle_date: str | int | float | date | datetime | None = None,
        trade_date: str | int | float | date | datetime | None = None,
        account: str | None = None,
        order_type: str | None = None,
        time_in_force: str | None = None,
        order_capacity: str | None = None,
        customer_or_firm: str | None = None,
    ) -> bool:
        """Emit a final fill for ``cl_ord_id`` when auto-complete is disabled."""

        return self.emit_order_update(
            cl_ord_id,
            "F",
            quantity=quantity,
            price=price,
            text=text,
            transact_time=transact_time,
            sending_time=sending_time,
            commission=commission,
            comm_type=comm_type,
            currency=currency,
            settle_type=settle_type,
            settle_date=settle_date,
            trade_date=trade_date,
            account=account,
            order_type=order_type,
            time_in_force=time_in_force,
            order_capacity=order_capacity,
            customer_or_firm=customer_or_firm,
        )

    def wait_for_idle(self, timeout: float | None = None) -> bool:
        """Block until all mock order flows have completed."""

        return self._trade_connection.wait_for_idle(timeout=timeout)

    def wait_for_telemetry(
        self,
        event: str,
        *,
        count: int = 1,
        timeout: float | None = None,
        predicate: Callable[[TelemetryEvent], bool] | None = None,
    ) -> bool:
        """Block until telemetry records ``count`` entries matching ``event``.

        Parameters
        ----------
        event:
            Telemetry event name to wait for.
        count:
            Number of matching records required before returning. Defaults to 1.
        timeout:
            Maximum time to wait. ``None`` waits indefinitely.
        predicate:
            Optional additional filter applied to the telemetry event.
        """

        def _matches(entry: TelemetryEvent) -> bool:
            if entry.event != event:
                return False
            if predicate is not None and not predicate(entry):
                return False
            return True

        return self._telemetry.wait_for(_matches, count=count, timeout=timeout)

    def wait_for_order_completion(self, cl_ord_id: str, timeout: float | None = None) -> bool:
        """Block until ``cl_ord_id`` emits an ``order_complete`` telemetry event."""

        return self.wait_for_telemetry(
            "order_complete",
            timeout=timeout,
            predicate=lambda event: event.details.get("cl_ord_id") == cl_ord_id,
        )
