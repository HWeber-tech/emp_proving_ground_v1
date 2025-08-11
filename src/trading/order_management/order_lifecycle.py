from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Callable
import asyncio
import logging

from src.operational.event_bus import Event, EventBus
from src.trading.integration.fix_broker_interface import FIXBrokerInterface
from src.trading.models.order import Order, OrderStatus, OrderType

try:
    # Optional execution config and estimators
    from src.data_foundation.config.execution_config import ExecutionConfig  # type: ignore
    from src.trading.execution.execution_model import ExecContext, estimate_slippage_bps, estimate_commission_bps  # type: ignore
except Exception:  # pragma: no cover
    ExecutionConfig = None  # type: ignore
    ExecContext = None  # type: ignore
    def estimate_slippage_bps(*args, **kwargs):  # type: ignore
        return 0.0
    def estimate_commission_bps(*args, **kwargs):  # type: ignore
        return 0.0

# Ensure simplefix availability for type references inside broker interface usage
try:
    import simplefix  # type: ignore
except Exception:  # pragma: no cover
    class _FixMessageStub:
        def __init__(self):
            self._pairs = []
        def append_pair(self, tag, value):
            self._pairs.append((tag, value))
        def to_dict(self):
            return dict(self._pairs)
        def __repr__(self):
            return f"FixMessageStub({self._pairs!r})"
    class _SimpleFixNamespace:
        FixMessage = _FixMessageStub
    simplefix = _SimpleFixNamespace()  # type: ignore


logger = logging.getLogger(__name__)


@dataclass
class _OrderMeta:
    order: Order
    # Last known bid/ask to infer fill price if missing
    last_bid: Optional[float] = None
    last_ask: Optional[float] = None


class OrderLifecycle:
    """Lightweight order lifecycle state machine over FIXBrokerInterface."""

    def __init__(
        self,
        event_bus: EventBus,
        broker: FIXBrokerInterface,
        execution_config: Optional["ExecutionConfig"] = None,
    ) -> None:
        self._event_bus = event_bus
        self._broker = broker
        self._execution_config = execution_config
        self._orders: Dict[str, _OrderMeta] = {}
        self._subscribers_installed = False

    async def start(self) -> None:
        if self._subscribers_installed:
            return
        # Subscribe to order and market data events
        self._event_bus.subscribe("order_update", self._on_order_update)
        self._event_bus.subscribe("order_cancel_rejected", self._on_order_cancel_rejected)
        self._event_bus.subscribe("market_data_update", self._on_market_data_update)
        self._event_bus.subscribe("market_data_incremental", self._on_market_data_update)
        self._subscribers_installed = True
        logger.info("OrderLifecycle started: subscribers registered")

    async def stop(self) -> None:
        if not self._subscribers_installed:
            return
        # Best-effort unsubscribe; EventBus ignores missing
        self._event_bus.unsubscribe("order_update", self._on_order_update)
        self._event_bus.unsubscribe("order_cancel_rejected", self._on_order_cancel_rejected)
        self._event_bus.unsubscribe("market_data_update", self._on_market_data_update)
        self._event_bus.unsubscribe("market_data_incremental", self._on_market_data_update)
        self._subscribers_installed = False
        logger.info("OrderLifecycle stopped: subscribers removed")

    def _infer_fill_price(self, order_meta: _OrderMeta) -> Optional[float]:
        if order_meta.order.side.upper() == "BUY":
            return order_meta.last_ask or order_meta.last_bid
        return order_meta.last_bid or order_meta.last_ask

    def _update_order_market_snap(self, symbol: str, bid: Optional[float], ask: Optional[float]) -> None:
        # Update snapshot for any active orders of this symbol
        for meta in self._orders.values():
            if meta.order.symbol == symbol and meta.order.is_active:
                if bid is not None:
                    meta.last_bid = bid
                if ask is not None:
                    meta.last_ask = ask

    def _on_market_data_update(self, event: Event) -> None:
        data = event.data or {}
        symbol = data.get("symbol")
        if not symbol:
            return
        md = data.get("data") or data.get("updates") or {}
        bid = md.get("bid")
        ask = md.get("ask")
        self._update_order_market_snap(symbol, bid, ask)

    def _on_order_update(self, event: Event) -> None:
        # Schedule async handling to avoid blocking synchronous EventBus
        asyncio.create_task(self._handle_order_update_async(event.data or {}))

    def _on_order_cancel_rejected(self, event: Event) -> None:
        payload = event.data or {}
        order_id = payload.get("order_id")
        if not order_id:
            return
        meta = self._orders.get(order_id)
        if not meta:
            return
        # Keep status unchanged but update timestamp via a no-op update
        meta.order.update_status(meta.order.status)

    async def _handle_order_update_async(self, payload: Dict) -> None:
        try:
            order_id: Optional[str] = payload.get("order_id")
            exec_type: Optional[str] = payload.get("exec_type")
            last_qty: Optional[float] = payload.get("last_qty")
            last_px: Optional[float] = payload.get("last_px")
            if not order_id or not exec_type:
                return
            meta = self._orders.get(order_id)
            if not meta:
                # Could be an order we didn't originate; ignore for now
                return

            # Map FIX ExecType to internal OrderStatus and fill handling
            if exec_type in ("0", "A", "B"):  # New/Partial types
                meta.order.update_status(OrderStatus.PARTIALLY_FILLED)
                if last_qty and last_px:
                    meta.order.add_fill(last_qty, last_px)
            elif exec_type in ("F",):  # Fill
                # If broker did not include price, infer from last known bid/ask
                if not last_px:
                    last_px = self._infer_fill_price(meta) or meta.order.price or 0.0
                if not last_qty:
                    # Assume full remaining
                    last_qty = max(0.0, meta.order.quantity - meta.order.filled_quantity)
                if last_qty > 0:
                    meta.order.add_fill(last_qty, last_px)
                meta.order.update_status(OrderStatus.FILLED)
            elif exec_type in ("4",):  # Canceled
                meta.order.update_status(OrderStatus.CANCELLED)
            elif exec_type in ("8",):  # Rejected
                meta.order.update_status(OrderStatus.REJECTED)
            else:
                # Unknown exec type; leave as is but update timestamp
                meta.order.update_status(meta.order.status)
        except Exception as e:
            logger.error(f"Error handling order update: {e}")

    async def place_market_order(self, symbol: str, side: str, quantity: float) -> Optional[str]:
        # Optional: pre-trade check using risk manager on event bus
        try:
            risk_manager = getattr(self._event_bus, 'risk_manager', None)
            if risk_manager and hasattr(risk_manager, 'check_risk_thresholds'):
                if not risk_manager.check_risk_thresholds():
                    logger.warning("Order blocked by risk thresholds (VaR/ES limits)")
                    return None
        except Exception:
            pass

        order_id = await self._broker.place_market_order(symbol=symbol, side=side, quantity=quantity)
        if not order_id:
            return None

        ord_obj = Order(
            order_id=order_id,
            symbol=symbol,
            side=side,
            quantity=quantity,
            order_type=OrderType.MARKET.value,
        )
        self._orders[order_id] = _OrderMeta(order=ord_obj)
        return order_id

    async def cancel_order(self, order_id: str) -> bool:
        return await self._broker.cancel_order(order_id)

    def get_order(self, order_id: str) -> Optional[Order]:
        meta = self._orders.get(order_id)
        return meta.order if meta else None

    def get_all_orders(self) -> Dict[str, Order]:
        return {oid: meta.order for oid, meta in self._orders.items()}