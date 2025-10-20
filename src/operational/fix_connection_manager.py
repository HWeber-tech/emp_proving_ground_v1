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
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from itertools import islice
from typing import Optional, Protocol, SupportsFloat, cast
from uuid import uuid4

import simplefix
from src.operational.fix_types import (
    FIXManagerProtocol,
    FIXMarketDataEntry,
    FIXMessage,
    FIXTradeConnectionProtocol,
    OrderBookLevelProtocol,
    OrderBookProtocol,
    OrderInfoProtocol,
)
from src.operational.live_broker_secrets import (
    BrokerCredentialProfile,
    LiveBrokerSecrets,
    load_live_broker_secrets,
)
from src.operational.mock_fix import MockFIXManager


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
    account_number: str | None
    password: str | None
    extras: Mapping[str, object] | None

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


def _encode_text(value: object | None) -> bytes | None:
    """Encode textual FIX fields to UTF-8 if present."""

    if value is None:
        return None
    if isinstance(value, bytes):
        return value
    if isinstance(value, str):
        return value.encode("utf-8")
    return str(value).encode("utf-8")


def _encode_numeric(value: SupportsFloat | str | bytes | None) -> bytes | None:
    """Encode numeric FIX fields using general format trimming."""

    if value is None:
        return None
    if isinstance(value, (bytes, bytearray)):
        try:
            value = value.decode("ascii", "ignore")
        except Exception:
            return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    text = f"{number:.10g}"
    return text.encode("ascii")


class _FIXApplicationAdapter:
    """Adapter that forwards incoming messages into an asyncio.Queue."""

    def __init__(self, session_type: str) -> None:
        self._session_type = session_type
        self._queue: asyncio.Queue[FIXMessage] | None = None
        self._delivered = 0
        self._dropped = 0

    def set_message_queue(self, queue: asyncio.Queue[FIXMessage]) -> None:
        self._queue = queue

    def dispatch(self, message: FIXMessage) -> None:
        queue = self._queue
        if queue is None:
            return

        try:
            queue.put_nowait(message)
        except asyncio.QueueFull:
            self._dropped += 1
            logger.warning("FIX %s queue is full; dropping message", self._session_type)
        else:
            self._delivered += 1

    def reset_metrics(self) -> None:
        self._delivered = 0
        self._dropped = 0

    def get_queue_metrics(self) -> dict[str, int]:
        return {"delivered": self._delivered, "dropped": self._dropped}


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
        self._live_broker_secrets: LiveBrokerSecrets | None = None
        self._md_sequence_counter = 1
        self._active_market_data_request: dict[str, object] | None = None

    def start_sessions(self) -> bool:
        """Create and start genuine FIX sessions."""
        try:
            # Prefer real FIX if credentials exist and genuine manager is available,
            # unless explicitly forced to mock via env.
            force_mock = os.environ.get("EMP_USE_MOCK_FIX", "0") in ("1", "true", "True")
            secrets = self._resolve_live_broker_secrets()
            active_profile = secrets.active_profile if secrets else None
            profile_complete = bool(active_profile and active_profile.is_complete())
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
            creds_present = creds_present or profile_complete
            use_mock = bool(
                force_mock
                or not creds_present
                or _GENUINE_MANAGER_CLASS is None
                or _IC_CONFIG_CLASS is None
            )

            if self._manager is not None:
                self.stop_sessions()

            manager: FIXManagerProtocol
            if use_mock:
                default_account = cast(
                    str | None,
                    getattr(self._system_config, "default_account", None),
                )
                if not default_account and active_profile and active_profile.trade:
                    default_account = active_profile.trade.username
                default_order_type = cast(
                    str | None,
                    getattr(self._system_config, "default_order_type", None),
                )
                default_time_in_force = cast(
                    str | None,
                    getattr(self._system_config, "default_time_in_force", None),
                )
                default_commission = cast(
                    float | None,
                    getattr(self._system_config, "default_commission", None),
                )
                default_commission_type = cast(
                    str | None,
                    getattr(self._system_config, "default_commission_type", None),
                )
                default_commission_currency = cast(
                    str | None,
                    getattr(
                        self._system_config,
                        "default_commission_currency",
                        None,
                    ),
                )
                default_settle_type = cast(
                    str | None,
                    getattr(self._system_config, "default_settle_type", None),
                )
                default_settle_date = cast(
                    str | None,
                    getattr(self._system_config, "default_settle_date", None),
                )
                default_trade_date = cast(
                    str | None,
                    getattr(self._system_config, "default_trade_date", None),
                )
                default_order_capacity = cast(
                    str | None,
                    getattr(self._system_config, "default_order_capacity", None),
                )
                default_customer_or_firm = cast(
                    str | None,
                    getattr(self._system_config, "default_customer_or_firm", None),
                )
                manager = cast(
                    FIXManagerProtocol,
                    MockFIXManager(
                        default_account=default_account,
                        default_order_type=default_order_type,
                        default_time_in_force=default_time_in_force,
                        default_commission=default_commission,
                        default_commission_type=default_commission_type,
                        default_commission_currency=default_commission_currency,
                        default_settle_type=default_settle_type,
                        default_settle_date=default_settle_date,
                        default_trade_date=default_trade_date,
                        default_order_capacity=default_order_capacity,
                        default_customer_or_firm=default_customer_or_firm,
                    ),
                )
            else:
                config_factory = cast(type[ICMarketsConfigLike], _IC_CONFIG_CLASS)
                if active_profile is None or not active_profile.is_complete():
                    raise RuntimeError("active live broker profile missing required credentials")

                environment_name = self._map_environment_hint(secrets)
                account_number = self._resolve_account_number(active_profile)
                ic_cfg = config_factory(
                    environment=environment_name,
                    account_number=account_number,
                )
                trade_password = active_profile.trade.password if active_profile.trade else None
                fallback_password = getattr(self._system_config, "password", None)
                ic_cfg.password = trade_password or fallback_password
                self._apply_profile_to_config(ic_cfg, active_profile)
                manager_factory = cast(FIXManagerFactory, _GENUINE_MANAGER_CLASS)
                manager = manager_factory(ic_cfg)
                self._apply_profile_to_manager(manager, active_profile)

            # Bridge market data: convert order book updates to queue-friendly messages
            def on_market_data(symbol: str, order_book: OrderBookProtocol) -> None:
                try:
                    entries: list[FIXMarketDataEntry] = []
                    bids = cast(
                        Sequence[OrderBookLevelProtocol],
                        getattr(order_book, "bids", ()),
                    )
                    asks = cast(
                        Sequence[OrderBookLevelProtocol],
                        getattr(order_book, "asks", ()),
                    )
                    for bid in islice(bids, 10):
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
                    for ask in islice(asks, 10):
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
                    exec_record = order_info.executions[-1] if order_info.executions else None

                    msg: FIXMessage = {
                        35: b"8",  # ExecutionReport
                        11: order_info.cl_ord_id.encode("utf-8"),
                    }

                    exec_type_value = exec_record.get("exec_type") if exec_record else None
                    is_reject = exec_type_value in ("8", b"8")
                    exec_type_bytes = _encode_text(exec_type_value)
                    if exec_type_bytes:
                        msg[150] = exec_type_bytes

                    order_id_value = getattr(order_info, "order_id", None)
                    if (order_id_value in (None, "")) and exec_record:
                        candidate_order = exec_record.get("order_id")
                        if isinstance(candidate_order, (str, bytes)):
                            order_id_value = candidate_order
                    order_id_bytes = _encode_text(order_id_value)
                    if order_id_bytes:
                        msg[37] = order_id_bytes

                    exec_id_value = getattr(order_info, "exec_id", None)
                    if (exec_id_value in (None, "")) and exec_record:
                        candidate_exec = exec_record.get("exec_id")
                        if isinstance(candidate_exec, (str, bytes)):
                            exec_id_value = candidate_exec
                    exec_id_bytes = _encode_text(exec_id_value)
                    if exec_id_bytes:
                        msg[17] = exec_id_bytes

                    symbol_bytes = _encode_text(getattr(order_info, "symbol", None))
                    if symbol_bytes:
                        msg[55] = symbol_bytes

                    side_bytes = _encode_text(getattr(order_info, "side", None))
                    if side_bytes:
                        msg[54] = side_bytes

                    account_value = getattr(order_info, "account", None)
                    if (account_value in (None, "")) and exec_record:
                        candidate_account = exec_record.get("account") if exec_record else None
                        if isinstance(candidate_account, (str, bytes)):
                            account_value = candidate_account
                    account_bytes = _encode_text(account_value)
                    if account_bytes:
                        msg[1] = account_bytes

                    order_type_value = getattr(order_info, "order_type", None)
                    if (order_type_value in (None, "")) and exec_record:
                        candidate_ord_type = exec_record.get("order_type") if exec_record else None
                        if isinstance(candidate_ord_type, (str, bytes)):
                            order_type_value = candidate_ord_type
                    order_type_bytes = _encode_text(order_type_value)
                    if order_type_bytes:
                        msg[40] = order_type_bytes

                    time_in_force_value = getattr(order_info, "time_in_force", None)
                    if (time_in_force_value in (None, "")) and exec_record:
                        candidate_tif = exec_record.get("time_in_force") if exec_record else None
                        if isinstance(candidate_tif, (str, bytes)):
                            time_in_force_value = candidate_tif
                    time_in_force_bytes = _encode_text(time_in_force_value)
                    if time_in_force_bytes:
                        msg[59] = time_in_force_bytes

                    ord_status_value = getattr(order_info, "ord_status", None)
                    if ord_status_value is None and exec_record:
                        ord_status_value = exec_record.get("ord_status")
                    ord_status_bytes = _encode_text(ord_status_value)
                    if ord_status_bytes:
                        msg[39] = ord_status_bytes

                    last_qty_value = getattr(order_info, "last_qty", None)
                    if last_qty_value in (None, 0.0) and exec_record:
                        candidate = exec_record.get("last_qty")
                        if isinstance(candidate, (int, float, str, bytes)):
                            last_qty_value = candidate
                    last_qty_bytes = _encode_numeric(last_qty_value)
                    if last_qty_bytes:
                        msg[32] = last_qty_bytes

                    last_px_value = getattr(order_info, "last_px", None)
                    if last_px_value in (None, 0.0) and exec_record:
                        candidate_px = exec_record.get("last_px")
                        if isinstance(candidate_px, (int, float, str, bytes)):
                            last_px_value = candidate_px
                    last_px_bytes = _encode_numeric(last_px_value)
                    if last_px_bytes:
                        msg[31] = last_px_bytes

                    avg_px_value = getattr(order_info, "avg_px", None)
                    if avg_px_value in (None, 0.0) and exec_record:
                        candidate_avg = exec_record.get("avg_px")
                        if isinstance(candidate_avg, (int, float, str, bytes)):
                            avg_px_value = candidate_avg
                    avg_px_bytes = _encode_numeric(avg_px_value)
                    if avg_px_bytes:
                        msg[6] = avg_px_bytes

                    cum_qty_value = getattr(order_info, "cum_qty", None)
                    if cum_qty_value is None and exec_record:
                        candidate_cum = exec_record.get("cum_qty")
                        if isinstance(candidate_cum, (int, float, str, bytes)):
                            cum_qty_value = candidate_cum
                    cum_qty_bytes = _encode_numeric(cum_qty_value)
                    if cum_qty_bytes:
                        msg[14] = cum_qty_bytes

                    leaves_qty_value = getattr(order_info, "leaves_qty", None)
                    if leaves_qty_value is None and exec_record:
                        candidate_leaves = exec_record.get("leaves_qty")
                        if isinstance(candidate_leaves, (int, float, str, bytes)):
                            leaves_qty_value = candidate_leaves
                    leaves_qty_bytes = _encode_numeric(leaves_qty_value)
                    if leaves_qty_bytes:
                        msg[151] = leaves_qty_bytes

                    commission_value = getattr(order_info, "cum_commission", None)
                    if (commission_value in (None, 0.0)) and exec_record:
                        candidate_cum_commission = exec_record.get("cum_commission")
                        if isinstance(
                            candidate_cum_commission,
                            (int, float, str, bytes),
                        ):
                            commission_value = candidate_cum_commission
                        else:
                            candidate_commission = exec_record.get("commission")
                            if isinstance(
                                candidate_commission,
                                (int, float, str, bytes),
                            ):
                                commission_value = candidate_commission
                    commission_bytes = _encode_numeric(commission_value)
                    if commission_bytes:
                        msg[12] = commission_bytes

                    comm_type_value = getattr(order_info, "comm_type", None)
                    if (comm_type_value in (None, "")) and exec_record:
                        candidate_comm_type = exec_record.get("comm_type") if exec_record else None
                        if isinstance(candidate_comm_type, (str, bytes)):
                            comm_type_value = candidate_comm_type
                    comm_type_bytes = _encode_text(comm_type_value)
                    if comm_type_bytes:
                        msg[13] = comm_type_bytes

                    currency_value = getattr(order_info, "currency", None)
                    if (currency_value in (None, "")) and exec_record:
                        candidate_currency = exec_record.get("currency") if exec_record else None
                        if isinstance(candidate_currency, (str, bytes)):
                            currency_value = candidate_currency
                    currency_bytes = _encode_text(currency_value)
                    if currency_bytes:
                        msg[15] = currency_bytes

                    settle_type_value = getattr(order_info, "settle_type", None)
                    if (settle_type_value in (None, "")) and exec_record:
                        candidate_settle_type = (
                            exec_record.get("settle_type") if exec_record else None
                        )
                        if isinstance(candidate_settle_type, (str, bytes)):
                            settle_type_value = candidate_settle_type
                    settle_type_bytes = _encode_text(settle_type_value)
                    if settle_type_bytes:
                        msg[63] = settle_type_bytes

                    settle_date_value = getattr(order_info, "settle_date", None)
                    if (settle_date_value in (None, "")) and exec_record:
                        candidate_settle_date = (
                            exec_record.get("settle_date") if exec_record else None
                        )
                        if isinstance(candidate_settle_date, (str, bytes)):
                            settle_date_value = candidate_settle_date
                    settle_date_bytes = _encode_text(settle_date_value)
                    if settle_date_bytes:
                        msg[64] = settle_date_bytes

                    trade_date_value = getattr(order_info, "trade_date", None)
                    if (trade_date_value in (None, "")) and exec_record:
                        candidate_trade_date = (
                            exec_record.get("trade_date") if exec_record else None
                        )
                        if isinstance(candidate_trade_date, (str, bytes)):
                            trade_date_value = candidate_trade_date
                    trade_date_bytes = _encode_text(trade_date_value)
                    if trade_date_bytes:
                        msg[75] = trade_date_bytes

                    order_capacity_value = getattr(order_info, "order_capacity", None)
                    if (order_capacity_value in (None, "")) and exec_record:
                        candidate_capacity = (
                            exec_record.get("order_capacity") if exec_record else None
                        )
                        if isinstance(candidate_capacity, (str, bytes)):
                            order_capacity_value = candidate_capacity
                    order_capacity_bytes = _encode_text(order_capacity_value)
                    if order_capacity_bytes:
                        msg[528] = order_capacity_bytes

                    customer_or_firm_value = getattr(order_info, "customer_or_firm", None)
                    if (customer_or_firm_value in (None, "")) and exec_record:
                        candidate_customer = (
                            exec_record.get("customer_or_firm") if exec_record else None
                        )
                        if isinstance(candidate_customer, (str, bytes)):
                            customer_or_firm_value = candidate_customer
                    customer_or_firm_bytes = _encode_text(customer_or_firm_value)
                    if customer_or_firm_bytes:
                        msg[204] = customer_or_firm_bytes

                    text_value = getattr(order_info, "text", None)
                    if (text_value in (None, "")) and exec_record:
                        candidate_text = exec_record.get("text")
                        if isinstance(candidate_text, (str, bytes)):
                            text_value = candidate_text
                    text_bytes = _encode_text(text_value)
                    if text_bytes:
                        msg[58] = text_bytes

                    ord_rej_reason_value = getattr(order_info, "ord_rej_reason", None)
                    if (ord_rej_reason_value in (None, "")) and exec_record:
                        candidate_reason = (
                            exec_record.get("ord_rej_reason") if exec_record else None
                        )
                        if isinstance(candidate_reason, (str, bytes)):
                            ord_rej_reason_value = candidate_reason
                    ord_rej_reason_bytes = _encode_text(ord_rej_reason_value)
                    if ord_rej_reason_bytes and is_reject:
                        msg[103] = ord_rej_reason_bytes

                    transact_time_value = getattr(order_info, "transact_time", None)
                    if (transact_time_value in (None, "")) and exec_record:
                        candidate_transact = exec_record.get("transact_time")
                        if isinstance(candidate_transact, (str, bytes)):
                            transact_time_value = candidate_transact
                    transact_time_bytes = _encode_text(transact_time_value)
                    if transact_time_bytes:
                        msg[60] = transact_time_bytes

                    sending_time_value = getattr(order_info, "sending_time", None)
                    if (sending_time_value in (None, "")) and exec_record:
                        candidate_sending = exec_record.get("sending_time")
                        if isinstance(candidate_sending, (str, bytes)):
                            sending_time_value = candidate_sending
                    sending_time_bytes = _encode_text(sending_time_value)
                    if sending_time_bytes:
                        msg[52] = sending_time_bytes

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

    def describe_live_broker_secrets(self, *, masked: bool = True) -> dict[str, object] | None:
        """Expose a redacted summary of the resolved live broker credentials."""

        secrets = self._live_broker_secrets
        if secrets is None:
            return None
        return secrets.describe(masked=masked)

    def get_active_broker_profile(self) -> BrokerCredentialProfile | None:
        """Return the active broker credential profile if available."""

        secrets = self._live_broker_secrets
        if secrets is None:
            return None
        return secrets.active_profile

    def _dispatch_market_data_request(
        self,
        symbols: Sequence[str] | None,
        *,
        subscription_type: str,
        depth: int = 1,
    ) -> bool:
        manager = self._manager
        if manager is None:
            logger.warning("Cannot send MarketDataRequest; FIX sessions not running")
            return False

        price_connection = getattr(manager, "price_connection", None)
        if price_connection is None:
            logger.warning("FIX manager missing price_connection; cannot send MarketDataRequest")
            return False

        resolved_symbols = self._normalise_symbol_list(symbols)
        request_meta = self._active_market_data_request

        if subscription_type == "1":
            if not resolved_symbols:
                logger.warning("MarketDataRequest subscribe requires at least one symbol")
                return False
            md_req_id = self._generate_md_request_id()
            depth_value = max(int(depth), 0)
        else:
            if request_meta is None:
                logger.debug("No active market data subscription to cancel")
                return True
            md_req_id = str(request_meta.get("id"))
            if not resolved_symbols:
                cached = request_meta.get("symbols")
                resolved_symbols = list(cached) if isinstance(cached, Sequence) else []
            depth_value = int(request_meta.get("depth", depth))
            if not resolved_symbols:
                logger.debug("Active subscription metadata missing symbols; skipping cancel request")
                return True

        try:
            msg = self._build_market_data_request_message(
                md_req_id=md_req_id,
                subscription_type=subscription_type,
                symbols=resolved_symbols,
                depth=depth_value,
            )
        except Exception:
            logger.exception("Failed to build MarketDataRequest message")
            return False

        try:
            ok = bool(price_connection.send_message_and_track(msg, md_req_id))
        except Exception:
            logger.exception("Failed to dispatch MarketDataRequest")
            return False

        if not ok:
            logger.warning("MarketDataRequest send not acknowledged", extra={"md_req_id": md_req_id})
            return False

        if subscription_type == "1":
            self._active_market_data_request = {
                "id": md_req_id,
                "symbols": tuple(resolved_symbols),
                "depth": depth_value,
                "timestamp": time.time(),
            }
        else:
            self._active_market_data_request = None
        return True

    def _generate_md_request_id(self) -> str:
        return f"MDREQ_{uuid4().hex[:12].upper()}_{self._next_md_sequence()}"

    def _next_md_sequence(self) -> int:
        current = self._md_sequence_counter
        self._md_sequence_counter += 1
        return current

    @staticmethod
    def _normalise_symbol_list(symbols: Sequence[str] | None) -> list[str]:
        if not symbols:
            return []
        ordered: dict[str, None] = {}
        for symbol in symbols:
            text = str(symbol).strip()
            if not text:
                continue
            ordered.setdefault(text, None)
        return list(ordered.keys())

    def _build_market_data_request_message(
        self,
        *,
        md_req_id: str,
        subscription_type: str,
        symbols: Sequence[str],
        depth: int,
        entry_types: Sequence[str] | None = None,
    ) -> simplefix.FixMessage:
        msg = simplefix.FixMessage()
        msg.append_pair(8, "FIX.4.4")
        msg.append_pair(35, "V")

        sender_comp_id = self._resolve_price_identifier(
            attr="price_sender_comp_id",
            extras_key="FIX_PRICE_SENDER_COMP_ID",
            default=self._default_sender_comp_id(),
        )
        if sender_comp_id:
            msg.append_pair(49, sender_comp_id)

        target_comp_id = self._resolve_price_identifier(
            attr="price_target_comp_id",
            extras_key="FIX_PRICE_TARGET_COMP_ID",
            default="cServer",
        )
        if target_comp_id:
            msg.append_pair(56, target_comp_id)

        target_sub_id = self._resolve_price_identifier(
            attr="price_target_sub_id",
            extras_key="FIX_PRICE_TARGET_SUB_ID",
            default="QUOTE",
        )
        if target_sub_id:
            msg.append_pair(57, target_sub_id)

        sender_sub_id = self._resolve_price_identifier(
            attr="price_sender_sub_id",
            extras_key="FIX_PRICE_SENDER_SUB_ID",
            default="QUOTE",
        )
        if sender_sub_id:
            msg.append_pair(50, sender_sub_id)

        msg.append_pair(34, str(self._next_md_sequence()))
        msg.append_pair(52, datetime.now(timezone.utc).strftime("%Y%m%d-%H:%M:%S.%f")[:-3])
        msg.append_pair(262, md_req_id)
        msg.append_pair(263, subscription_type)
        msg.append_pair(264, str(max(depth, 0)))

        md_entry_types = tuple(entry_types) if entry_types else ("0", "1")
        msg.append_pair(267, str(len(md_entry_types)))
        for entry_type in md_entry_types:
            msg.append_pair(269, str(entry_type))

        msg.append_pair(146, str(len(symbols)))
        for symbol in symbols:
            msg.append_pair(55, str(symbol))

        return msg

    def _resolve_price_identifier(
        self,
        *,
        attr: str,
        extras_key: str,
        default: str | None = None,
    ) -> str | None:
        manager = self._manager
        if manager is not None:
            value = getattr(manager, attr, None)
            if isinstance(value, str) and value.strip():
                return value.strip()
        extras = getattr(self._system_config, "extras", None)
        if isinstance(extras, Mapping):
            raw = extras.get(extras_key)
            if raw is not None:
                text = str(raw).strip()
                if text:
                    return text
        fallback = getattr(self._system_config, attr, None)
        if isinstance(fallback, str) and fallback.strip():
            return fallback.strip()
        return default

    def _default_sender_comp_id(self) -> str | None:
        account = getattr(self._system_config, "account_number", None)
        if not account:
            return None
        environment = getattr(self._system_config, "environment", "demo")
        return f"{environment}.icmarkets.{account}"

    # Internal helpers -------------------------------------------------

    def _resolve_live_broker_secrets(self) -> LiveBrokerSecrets:
        extras = getattr(self._system_config, "extras", None)
        fallback: dict[str, object] = {}
        attr_keys = (
            "FIX_PRICE_SENDER_COMP_ID",
            "FIX_PRICE_USERNAME",
            "FIX_PRICE_PASSWORD",
            "FIX_TRADE_SENDER_COMP_ID",
            "FIX_TRADE_USERNAME",
            "FIX_TRADE_PASSWORD",
        )
        for key in attr_keys:
            value = getattr(self._system_config, key.lower(), None)
            if value not in (None, ""):
                fallback[key] = value
        if isinstance(extras, Mapping):
            fallback.update(extras)

        secrets = load_live_broker_secrets(
            mapping=dict(os.environ),
            environment=getattr(self._system_config, "environment", None),
            fallback=fallback,
        )
        self._live_broker_secrets = secrets
        return secrets

    def _map_environment_hint(self, secrets: LiveBrokerSecrets | None) -> str:
        if secrets is None or secrets.active_key is None:
            return str(getattr(self._system_config, "environment", "demo"))
        if secrets.active_key.upper() == "SANDBOX":
            return "demo"
        if secrets.active_key.upper() == "PROD":
            return "production"
        return str(getattr(self._system_config, "environment", "demo"))

    def _resolve_account_number(self, profile: BrokerCredentialProfile) -> str:
        candidates = (
            getattr(profile.trade, "username", None),
            getattr(profile.price, "username", None),
            getattr(self._system_config, "account_number", None),
        )
        for candidate in candidates:
            if candidate:
                return str(candidate)
        raise RuntimeError("live broker account number is not configured")

    def _apply_profile_to_config(
        self, config: ICMarketsConfigLike, profile: BrokerCredentialProfile
    ) -> None:
        if profile.price is not None:
            self._maybe_setattr(config, "price_sender_comp_id", profile.price.sender_comp_id)
            self._maybe_setattr(config, "price_username", profile.price.username)
            self._maybe_setattr(config, "price_password", profile.price.password)
        if profile.trade is not None:
            self._maybe_setattr(config, "trade_sender_comp_id", profile.trade.sender_comp_id)
            self._maybe_setattr(config, "trade_username", profile.trade.username)
            self._maybe_setattr(config, "trade_password", profile.trade.password)

    def _apply_profile_to_manager(
        self, manager: FIXManagerProtocol, profile: BrokerCredentialProfile
    ) -> None:
        if profile.price is not None:
            self._maybe_setattr(manager, "price_sender_comp_id", profile.price.sender_comp_id)
            self._maybe_setattr(manager, "price_username", profile.price.username)
        if profile.trade is not None:
            self._maybe_setattr(manager, "trade_sender_comp_id", profile.trade.sender_comp_id)
            self._maybe_setattr(manager, "trade_username", profile.trade.username)
        configure = getattr(manager, "configure_credentials", None)
        if callable(configure) and profile.trade and profile.price:
            try:
                configure(
                    price_sender_comp_id=profile.price.sender_comp_id,
                    price_username=profile.price.username,
                    price_password=profile.price.password,
                    trade_sender_comp_id=profile.trade.sender_comp_id,
                    trade_username=profile.trade.username,
                    trade_password=profile.trade.password,
                )
            except Exception:  # pragma: no cover - defensive
                logger.debug("configure_credentials failed", exc_info=True)

    @staticmethod
    def _maybe_setattr(target: object, attr: str, value: object) -> None:
        if value in (None, ""):
            return
        if not hasattr(target, attr):
            return
        try:
            setattr(target, attr, value)
        except Exception:  # pragma: no cover - defensive guard
            logger.debug(
                "Failed to set attribute %s on %s", attr, type(target).__name__, exc_info=True
            )

    def stop_sessions(self) -> None:
        if self._manager:
            self._manager.stop()
            self._manager = None
            self._initiator = None
            self._price_app.reset_metrics()
            self._trade_app.reset_metrics()
            self._active_market_data_request = None
            logger.info("FIXConnectionManager sessions stopped")

    def subscribe_market_data(self, symbols: Sequence[str], *, depth: int = 1) -> bool:
        """Issue a MarketDataRequest subscribe for the provided symbols."""

        return self._dispatch_market_data_request(symbols, subscription_type="1", depth=depth)

    def unsubscribe_market_data(self, symbols: Sequence[str] | None = None) -> bool:
        """Disable an existing MarketDataRequest subscription."""

        return self._dispatch_market_data_request(symbols, subscription_type="2")

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
