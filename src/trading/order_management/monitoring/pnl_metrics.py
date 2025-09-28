"""Metrics publisher for position and PnL observability."""

from __future__ import annotations

from typing import Optional

from src.operational.metrics_registry import MetricsRegistry, get_registry

from ..position_tracker import PositionSnapshot

__all__ = ["PositionMetricsPublisher"]


class PositionMetricsPublisher:
    """Publish position and PnL metrics to a Prometheus-compatible registry."""

    def __init__(self, *, registry: MetricsRegistry | None = None) -> None:
        self._registry = registry or get_registry()

        self._net_qty = self._registry.get_gauge(
            "emp_position_net_quantity",
            "Net position size per account and symbol",
            ["account", "symbol"],
        )
        self._gross_long = self._registry.get_gauge(
            "emp_position_gross_long",
            "Gross long quantity per account and symbol",
            ["account", "symbol"],
        )
        self._gross_short = self._registry.get_gauge(
            "emp_position_gross_short",
            "Gross short quantity per account and symbol",
            ["account", "symbol"],
        )
        self._realized_pnl = self._registry.get_gauge(
            "emp_position_realized_pnl",
            "Realized PnL per account and symbol",
            ["account", "symbol"],
        )
        self._unrealized_pnl = self._registry.get_gauge(
            "emp_position_unrealized_pnl",
            "Unrealized PnL per account and symbol",
            ["account", "symbol"],
        )
        self._exposure = self._registry.get_gauge(
            "emp_position_notional_exposure",
            "Notional exposure per account and symbol",
            ["account", "symbol"],
        )
        self._market_value = self._registry.get_gauge(
            "emp_position_market_value",
            "Market value per account and symbol",
            ["account", "symbol"],
        )
        self._account_exposure = self._registry.get_gauge(
            "emp_account_total_exposure",
            "Aggregated notional exposure per account",
            ["account"],
        )
        self._account_realized = self._registry.get_gauge(
            "emp_account_total_realized_pnl",
            "Aggregated realized PnL per account",
            ["account"],
        )
        self._account_unrealized = self._registry.get_gauge(
            "emp_account_total_unrealized_pnl",
            "Aggregated unrealized PnL per account",
            ["account"],
        )

    # ------------------------------------------------------------------
    def publish(self, snapshot: PositionSnapshot) -> None:
        """Publish metrics for a single position snapshot."""

        labels = {"account": snapshot.account, "symbol": snapshot.symbol}
        self._net_qty.labels(**labels).set(float(snapshot.net_quantity))
        self._gross_long.labels(**labels).set(float(snapshot.long_quantity))
        self._gross_short.labels(**labels).set(float(snapshot.short_quantity))
        self._realized_pnl.labels(**labels).set(float(snapshot.realized_pnl))

        unrealized = float(snapshot.unrealized_pnl) if snapshot.unrealized_pnl is not None else 0.0
        self._unrealized_pnl.labels(**labels).set(unrealized)

        exposure = float(snapshot.exposure) if snapshot.exposure is not None else 0.0
        self._exposure.labels(**labels).set(exposure)

        market_value = snapshot.market_value or 0.0
        self._market_value.labels(**labels).set(float(market_value))

    # ------------------------------------------------------------------
    def publish_account_totals(
        self,
        *,
        account: str,
        total_exposure: float,
        total_realized_pnl: float,
        total_unrealized_pnl: Optional[float] = None,
    ) -> None:
        """Publish aggregated metrics for an account."""

        labels = {"account": account}
        self._account_exposure.labels(**labels).set(float(total_exposure))
        self._account_realized.labels(**labels).set(float(total_realized_pnl))

        unrealized = float(total_unrealized_pnl) if total_unrealized_pnl is not None else 0.0
        self._account_unrealized.labels(**labels).set(unrealized)

