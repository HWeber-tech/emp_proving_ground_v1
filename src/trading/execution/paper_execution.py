"""Execution adapter for the bootstrap paper-trading stack."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Mapping, MutableMapping

from src.trading.monitoring.portfolio_monitor import PortfolioMonitor
from ._risk_context import (
    RiskContextProvider,
    capture_risk_context,
    describe_risk_context,
)


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
    risk_context_provider: RiskContextProvider | None = field(default=None, repr=False)
    _last_risk_metadata: dict[str, object] | None = field(default=None, init=False, repr=False)
    _last_risk_error: dict[str, object] | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self._last_risk_metadata = None
        self._last_risk_error = None

    def set_risk_context_provider(
        self, provider: RiskContextProvider | None
    ) -> None:
        """Install or replace the callable that resolves trading risk metadata."""

        self.risk_context_provider = provider

    def _update_risk_context(self) -> None:
        metadata, error = capture_risk_context(self.risk_context_provider)
        self._last_risk_metadata = metadata
        self._last_risk_error = error

    def describe_risk_context(self) -> dict[str, object]:
        """Expose the most recent deterministic risk context snapshot."""

        return describe_risk_context(self._last_risk_metadata, self._last_risk_error)

    async def process_order(self, intent: Any) -> str:
        symbol = str(_extract(intent, "symbol", default="UNKNOWN"))
        side = str(_extract(intent, "side", "direction", default="BUY")).upper()
        quantity = _to_float(_extract(intent, "quantity", "size", "volume", default=0.0), 0.0)
        price = _to_float(_extract(intent, "price", "limit_price", "entry_price", default=0.0), 0.0)

        self._update_risk_context()

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
