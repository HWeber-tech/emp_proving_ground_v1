from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from typing import Dict, Optional
from datetime import datetime
import logging

from src.operational.persistence import RedisStateStore, JSONStateStore

logger = logging.getLogger(__name__)


@dataclass
class PositionState:
    symbol: str
    quantity: float = 0.0
    avg_price: float = 0.0
    realized_pnl: float = 0.0


class PortfolioTracker:
    """Lightweight portfolio tracker consuming OrderInfo updates from FIX manager."""

    def __init__(self) -> None:
        # Select store (Redis preferred)
        try:
            host = os.environ.get("REDIS_HOST", "localhost")
            port = int(os.environ.get("REDIS_PORT", "6379"))
            db = int(os.environ.get("REDIS_DB", "0"))
            pwd = os.environ.get("REDIS_PASSWORD")
            self._store = RedisStateStore(host, port, db, pwd, fallback_dir="data/portfolio")
        except Exception:
            self._store = JSONStateStore(base_dir="data/portfolio")
        self.cash: float = 0.0
        self.positions: Dict[str, PositionState] = {}
        self._load()

    def _key(self) -> str:
        return "portfolio_state"

    def _load(self) -> None:
        try:
            data = {}
            # Use JSONStateStore interface
            # It stores by name; reuse its API to load generic dict
            if isinstance(self._store, JSONStateStore):
                path = os.path.join(self._store.base_dir, f"{self._key()}.json")
                if os.path.exists(path):
                    with open(path, "r", encoding="utf-8") as f:
                        data = json.load(f)
            else:
                # RedisStateStore; use its client directly
                if getattr(self._store, "client", None):
                    raw = self._store.client.get("emp:portfolio")
                    data = json.loads(raw) if raw else {}
            self.cash = float(data.get("cash", 0.0))
            self.positions = {s: PositionState(**p) for s, p in data.get("positions", {}).items()}
        except Exception:
            self.cash = 0.0
            self.positions = {}

    def _save(self) -> None:
        state = {
            "cash": self.cash,
            "positions": {s: asdict(p) for s, p in self.positions.items()},
            "timestamp": datetime.utcnow().isoformat(),
        }
        try:
            if isinstance(self._store, JSONStateStore):
                path = os.path.join(self._store.base_dir, f"{self._key()}.json")
                os.makedirs(self._store.base_dir, exist_ok=True)
                tmp = path + ".tmp"
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump(state, f)
                os.replace(tmp, path)
            else:
                if getattr(self._store, "client", None):
                    self._store.client.set("emp:portfolio", json.dumps(state))
        except Exception as e:
            logger.warning(f"Failed to persist portfolio: {e}")

    def attach_to_manager(self, fix_manager) -> None:
        # Subscribe to order updates
        def _on_order(order_info):
            try:
                self._handle_order_info(order_info)
            except Exception as e:
                logger.error(f"Portfolio update failed: {e}")
        fix_manager.add_order_callback(_on_order)
        logger.info("PortfolioTracker attached to FIX manager")

    def _handle_order_info(self, order) -> None:
        # Use the last execution to detect fills
        if not getattr(order, "executions", None):
            return
        last = order.executions[-1]
        exec_type = last.get("exec_type")
        if exec_type not in ("1", "2"):
            return
        symbol = order.symbol
        side = order.side
        qty = float(order.last_qty)
        px = float(order.last_px)
        if qty <= 0:
            return
        pos = self.positions.get(symbol) or PositionState(symbol=symbol)
        if side == "1":  # BUY
            new_qty = pos.quantity + qty
            if new_qty > 0:
                pos.avg_price = (pos.avg_price * pos.quantity + px * qty) / new_qty if pos.quantity > 0 else px
            pos.quantity = new_qty
        else:  # SELL
            # Realized PnL for the portion closed
            closing_qty = min(qty, max(0.0, pos.quantity))
            pos.realized_pnl += (px - pos.avg_price) * closing_qty
            pos.quantity -= qty
            if pos.quantity <= 0:
                pos.avg_price = 0.0
        self.positions[symbol] = pos
        self._save()


