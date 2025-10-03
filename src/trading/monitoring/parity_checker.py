from __future__ import annotations

import logging
from typing import Any, Mapping, Protocol

from src.core.telemetry import get_metrics_sink

logger = logging.getLogger(__name__)


class OrderLike(Protocol):
    status: Any
    leaves_qty: Any
    cum_qty: Any
    avg_px: Any
    order_id: Any


class PositionLike(Protocol):
    quantity: Any


class FixManagerLike(Protocol):
    def get_all_orders(self) -> Mapping[str, "OrderLike"]: ...


class ParityChecker:
    """Compares local FIX-tracked state with broker-reported snapshots to flag mismatches."""

    def __init__(self, fix_manager: "FixManagerLike") -> None:
        self.fix_manager = fix_manager

    def check_orders(self, broker_orders: Mapping[str, Mapping[str, Any]]) -> int:
        """broker_orders: map clOrdID -> status/fields."""
        local = self.fix_manager.get_all_orders()
        mismatches = 0
        for cl_id, order in local.items():
            b = broker_orders.get(cl_id)
            if b is None:
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
        self._record_gauge(
            "fix_parity_mismatched_orders",
            float(mismatches),
            context="order parity",
        )
        logger.info(f"Order parity mismatches: {mismatches}")
        return mismatches

    def compare_order_fields(
        self, local_order: "OrderLike", broker_order: Mapping[str, Any]
    ) -> dict[str, object]:
        """Return a dict of mismatched fields between local OrderInfo and broker snapshot.
        Compares: status, leaves_qty, cum_qty, avg_px, order_id.
        """
        diffs: dict[str, object] = {}
        try:

            def norm(v: object) -> object:
                return (
                    float(v)
                    if isinstance(v, (int, float, str)) and str(v).replace(".", "", 1).isdigit()
                    else v
                )

            fields = (
                (
                    "status",
                    getattr(local_order.status, "value", local_order.status),
                    broker_order.get("status"),
                ),
                (
                    "leaves_qty",
                    getattr(local_order, "leaves_qty", None),
                    broker_order.get("leaves_qty"),
                ),
                ("cum_qty", getattr(local_order, "cum_qty", None), broker_order.get("cum_qty")),
                ("avg_px", getattr(local_order, "avg_px", None), broker_order.get("avg_px")),
                ("order_id", getattr(local_order, "order_id", None), broker_order.get("order_id")),
            )
            for name, lv, bv in fields:
                lvv = norm(lv)
                bvv = norm(bv)
                if lvv is None and bvv is None:
                    continue
                if str(lvv) != str(bvv):
                    diffs[name] = {"local": lv, "broker": bv}
        except Exception:
            diffs["error"] = "compare_failed"
        return diffs

    def check_positions(self, broker_positions: Mapping[str, Mapping[str, Any]]) -> int:
        """broker_positions: map symbol -> quantity/avg_price."""
        mismatches = 0
        try:
            from src.trading.monitoring.portfolio_tracker import PortfolioTracker  # isort: skip

            pt = PortfolioTracker()
            local_positions = pt.positions
        except Exception:
            local_positions = {}
        for sym, pos in local_positions.items():
            b = broker_positions.get(sym)
            try:
                if b is None:
                    mismatches += 1
                    continue
                if abs(float(b.get("quantity", 0.0)) - float(pos.quantity)) > 1e-6:
                    mismatches += 1
                    continue
            except Exception:
                mismatches += 1
                continue
        self._record_gauge(
            "fix_parity_mismatched_positions",
            float(mismatches),
            context="position parity",
        )
        logger.info(f"Position parity mismatches: {mismatches}")
        return mismatches

    def _record_gauge(self, name: str, value: float, *, context: str) -> None:
        """Safely publish parity telemetry without masking failures."""

        try:
            sink = get_metrics_sink()
        except Exception as exc:  # pragma: no cover - defensive guardrail
            logger.warning(
                "Failed to resolve metrics sink for %s gauge: %s",
                context,
                exc,
                exc_info=True,
            )
            return

        try:
            sink.set_gauge(name, value)
        except Exception as exc:  # pragma: no cover - defensive guardrail
            logger.warning(
                "Failed to publish %s gauge (%s): %s",
                name,
                context,
                exc,
                exc_info=True,
            )
