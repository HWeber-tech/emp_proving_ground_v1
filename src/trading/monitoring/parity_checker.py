from __future__ import annotations

import logging
from typing import Dict, Any

from src.operational.metrics import fix_parity_mismatched_orders, fix_parity_mismatched_positions

logger = logging.getLogger(__name__)


class ParityChecker:
    """Compares local FIX-tracked state with broker-reported snapshots to flag mismatches."""

    def __init__(self, fix_manager) -> None:
        self.fix_manager = fix_manager

    def check_orders(self, broker_orders: Dict[str, Any]) -> int:
        """broker_orders: map clOrdID -> status/fields."""
        local = self.fix_manager.get_all_orders()
        mismatches = 0
        for cl_id, order in local.items():
            b = broker_orders.get(cl_id)
            if not b:
                # Potential mismatch (local exists, broker missing)
                mismatches += 1
                continue
            try:
                if str(b.get("order_id")) != str(order.order_id):
                    mismatches += 1
                    continue
                if str(b.get("status")) != str(getattr(order.status, "value", order.status)):
                    mismatches += 1
                    continue
            except Exception:
                mismatches += 1
                continue
        try:
            fix_parity_mismatched_orders.set(mismatches)
        except Exception:
            pass
        logger.info(f"Order parity mismatches: {mismatches}")
        return mismatches

    def check_positions(self, broker_positions: Dict[str, Any]) -> int:
        """broker_positions: map symbol -> quantity/avg_price."""
        mismatches = 0
        try:
            from src.trading.monitoring.portfolio_tracker import PortfolioTracker
            pt = PortfolioTracker()
            local_positions = pt.positions
        except Exception:
            local_positions = {}
        for sym, pos in local_positions.items():
            b = broker_positions.get(sym)
            try:
                if not b:
                    mismatches += 1
                    continue
                if abs(float(b.get("quantity", 0.0)) - float(pos.quantity)) > 1e-6:
                    mismatches += 1
                    continue
            except Exception:
                mismatches += 1
                continue
        try:
            fix_parity_mismatched_positions.set(mismatches)
        except Exception:
            pass
        logger.info(f"Position parity mismatches: {mismatches}")
        return mismatches


