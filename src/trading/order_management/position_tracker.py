from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional
import logging
from datetime import datetime

from src.operational.event_bus import Event, EventBus
from src.trading.models.position import Position
from src.trading.models.order import Order

logger = logging.getLogger(__name__)


@dataclass
class PortfolioSummary:
    total_unrealized_pnl: float
    total_realized_pnl: float
    total_pnl: float


class PositionTracker:
    """Tracks positions and P&L per symbol and at portfolio level."""

    def __init__(self, event_bus: EventBus):
        self._event_bus = event_bus
        self._positions: Dict[str, Position] = {}
        self._subscribers_installed = False

    async def start(self) -> None:
        if self._subscribers_installed:
            return
        self._event_bus.subscribe("order_update", self._on_order_update)
        self._event_bus.subscribe("market_data_update", self._on_market_data)
        self._event_bus.subscribe("market_data_incremental", self._on_market_data)
        self._subscribers_installed = True
        logger.info("PositionTracker started: subscribers registered")

    async def stop(self) -> None:
        if not self._subscribers_installed:
            return
        self._event_bus.unsubscribe("order_update", self._on_order_update)
        self._event_bus.unsubscribe("market_data_update", self._on_market_data)
        self._event_bus.unsubscribe("market_data_incremental", self._on_market_data)
        self._subscribers_installed = False
        logger.info("PositionTracker stopped: subscribers removed")

    def _on_market_data(self, event: Event) -> None:
        payload = event.data or {}
        symbol = payload.get("symbol")
        md = payload.get("data") or payload.get("updates") or {}
        if not symbol or symbol not in self._positions:
            return
        pos = self._positions[symbol]
        bid = md.get("bid")
        ask = md.get("ask")
        # Use mid if both available, else whichever exists
        mid = None
        if bid is not None and ask is not None:
            mid = (bid + ask) / 2.0
        elif bid is not None:
            mid = bid
        elif ask is not None:
            mid = ask
        if mid is not None:
            pos.update_market_price(mid)

    def _on_order_update(self, event: Event) -> None:
        payload = event.data or {}
        order_id = payload.get("order_id")
        exec_type = payload.get("exec_type")
        last_qty = payload.get("last_qty")
        last_px = payload.get("last_px")
        symbol = payload.get("symbol")
        side = payload.get("side")

        if not exec_type or exec_type not in ("F",):
            # Only impact positions on fills
            return
        if not symbol or not side:
            # Best-effort: cannot update position without symbol/side
            return
        if last_qty is None or last_px is None:
            # Without qty/px, cannot compute realized component; defer
            return

        signed_qty = float(last_qty) if side.upper() == "BUY" else -float(last_qty)
        fill_price = float(last_px)

        pos = self._positions.get(symbol)
        if pos is None:
            # Opening new position
            self._positions[symbol] = Position(symbol=symbol, quantity=signed_qty, average_price=fill_price)
            return

        # Realized P&L occurs when trade direction reduces absolute exposure
        old_qty = pos.quantity
        new_qty = old_qty + signed_qty
        if old_qty == 0:
            # Just opening new position
            pos.update_quantity(new_qty, fill_price)
            return

        if (old_qty > 0 and signed_qty < 0) or (old_qty < 0 and signed_qty > 0):
            # Closing or reducing: realized P&L on the closed portion
            closed_qty = min(abs(old_qty), abs(signed_qty))
            direction = 1.0 if old_qty > 0 else -1.0
            realized = (fill_price - pos.average_price) * (closed_qty * direction)
            pos.add_realized_pnl(realized)

        # Update remaining quantity and possibly average price for any remaining/opening part
        remaining_qty = new_qty
        if (old_qty > 0 and new_qty > 0 and signed_qty > 0) or (old_qty < 0 and new_qty < 0 and signed_qty < 0):
            # Adding to existing exposure: recompute average price
            total_cost = pos.average_price * abs(old_qty) + fill_price * abs(signed_qty)
            new_avg = total_cost / max(1e-12, abs(remaining_qty))
            pos.update_quantity(remaining_qty, new_avg)
        else:
            # Reduced or flipped exposure
            if remaining_qty == 0:
                pos.update_quantity(0, pos.average_price)
            else:
                # Flipped direction: set new average at fill price
                pos.update_quantity(remaining_qty, fill_price)

    def get_position(self, symbol: str) -> Optional[Position]:
        return self._positions.get(symbol)

    def get_all_positions(self) -> Dict[str, Position]:
        return dict(self._positions)

    def portfolio_summary(self) -> PortfolioSummary:
        total_unrealized = sum(p.unrealized_pnl for p in self._positions.values())
        total_realized = sum(p.realized_pnl for p in self._positions.values())
        return PortfolioSummary(
            total_unrealized_pnl=total_unrealized,
            total_realized_pnl=total_realized,
            total_pnl=total_unrealized + total_realized,
        )