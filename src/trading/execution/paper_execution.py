"""Execution adapter for the bootstrap paper-trading stack."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Mapping, MutableMapping

from src.trading.monitoring.portfolio_monitor import PortfolioMonitor


def _extract(mapping: Any, *names: str, default: Any = None) -> Any:
    if isinstance(mapping, Mapping):
        for name in names:
            if name in mapping:
                return mapping[name]
    for name in names:
        if hasattr(mapping, name):
            return getattr(mapping, name)
    return default


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


@dataclass
class ImmediateFillExecutionAdapter:
    """Process intents by immediately filling them against the portfolio monitor."""

    portfolio_monitor: PortfolioMonitor
    fills: list[dict[str, Any]] = field(default_factory=list)
    _order_seq: int = 0

    async def process_order(self, intent: Any) -> str:
        symbol = str(_extract(intent, "symbol", default="UNKNOWN"))
        side = str(_extract(intent, "side", "direction", default="BUY")).upper()
        quantity = _to_float(_extract(intent, "quantity", "size", "volume", default=0.0), 0.0)
        price = _to_float(_extract(intent, "price", "limit_price", "entry_price", default=0.0), 0.0)

        self._order_seq += 1
        order_id = f"PAPER-{self._order_seq:05d}"

        # Remove optimistic reservation prior to issuing the execution report
        if quantity:
            self.portfolio_monitor.release_position(symbol, quantity)

        report = SimpleNamespace(
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=price,
            order_id=order_id,
        )
        await self.portfolio_monitor.on_execution_report(report)

        metadata: MutableMapping[str, Any]
        meta_value = _extract(intent, "metadata", default={})
        if isinstance(meta_value, MutableMapping):
            metadata = meta_value
        else:
            metadata = {}

        self.fills.append(
            {
                "order_id": order_id,
                "symbol": symbol,
                "side": side,
                "quantity": quantity,
                "price": price,
                "metadata": dict(metadata),
            }
        )
        return order_id


__all__ = ["ImmediateFillExecutionAdapter"]
