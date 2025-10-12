"""Paper trading broker adapter bridging FIX integrations with the release router.

This module fulfils the roadmap deliverable that wires AlphaTrade's execution
stack into a paper broker endpoint.  The adapter translates validated trade
intents into FIX broker invocations while preserving the deterministic risk
context telemetry expected by governance tooling.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Mapping, MutableMapping

from src.trading.execution._risk_context import (
    RiskContextProvider,
    capture_risk_context,
    describe_risk_context,
)
from src.trading.monitoring.portfolio_monitor import PortfolioMonitor

logger = logging.getLogger(__name__)

__all__ = ["PaperBrokerError", "PaperBrokerExecutionAdapter"]


class PaperBrokerError(RuntimeError):
    """Raised when the paper broker fails to accept or acknowledge an order."""


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
class PaperBrokerExecutionAdapter:
    """Adapter that forwards validated intents to a paper FIX broker interface."""

    broker_interface: Any
    portfolio_monitor: PortfolioMonitor
    order_timeout: float | None = 5.0
    risk_context_provider: RiskContextProvider | None = field(default=None, repr=False)
    _last_order: dict[str, object] | None = field(default=None, init=False, repr=False)
    _last_risk_metadata: dict[str, object] | None = field(default=None, init=False, repr=False)
    _last_risk_error: dict[str, object] | None = field(default=None, init=False, repr=False)
    _last_error: dict[str, object] | None = field(default=None, init=False, repr=False)
    _total_orders: int = field(default=0, init=False, repr=False)
    _successful_orders: int = field(default=0, init=False, repr=False)
    _failed_orders: int = field(default=0, init=False, repr=False)
    _latency_samples: int = field(default=0, init=False, repr=False)
    _total_latency: float = field(default=0.0, init=False, repr=False)
    _last_latency: float | None = field(default=None, init=False, repr=False)

    def set_risk_context_provider(
        self, provider: RiskContextProvider | None
    ) -> None:
        """Install or clear the callable that surfaces deterministic risk metadata."""

        if provider is not None and not callable(provider):
            raise TypeError("risk_context_provider must be callable or None")
        self.risk_context_provider = provider

    def describe_risk_context(self) -> dict[str, object]:
        """Expose the most recent deterministic risk snapshot."""

        return describe_risk_context(self._last_risk_metadata, self._last_risk_error)

    def describe_last_order(self) -> Mapping[str, object] | None:
        """Return the most recently dispatched order metadata, if any."""

        if self._last_order is None:
            return None
        return dict(self._last_order)

    async def process_order(self, intent: Any) -> str:
        """Submit a validated intent to the paper broker via FIX."""

        self._last_error = None
        latency: float | None = None
        symbol = str(_extract(intent, "symbol", "instrument", "asset", default="UNKNOWN"))
        side = str(_extract(intent, "side", "direction", default="BUY")).upper()
        quantity = _to_float(_extract(intent, "quantity", "size", "volume"), 0.0)
        order_type = str(
            _extract(intent, "order_type", "type", "kind", default="market")
        ).lower()

        if side not in {"BUY", "SELL"}:
            self._record_failure(
                stage="validation",
                message=f"Unsupported order side '{side}'",
                intent=intent,
                symbol=symbol,
                side=side,
                quantity=quantity,
                order_type=order_type,
            )
            raise PaperBrokerError(f"Unsupported order side '{side}'")
        if quantity <= 0:
            self._record_failure(
                stage="validation",
                message="Order quantity must be positive for paper broker execution",
                intent=intent,
                symbol=symbol,
                side=side,
                quantity=quantity,
                order_type=order_type,
            )
            raise PaperBrokerError(
                "Order quantity must be positive for paper broker execution"
            )
        if order_type not in {"market", "market_order"}:
            self._record_failure(
                stage="validation",
                message=f"Order type '{order_type}' is not supported by the paper broker",
                intent=intent,
                symbol=symbol,
                side=side,
                quantity=quantity,
                order_type=order_type,
            )
            raise PaperBrokerError(
                f"Order type '{order_type}' is not supported by the paper broker"
            )

        self._capture_risk_context()
        start_time = perf_counter()

        try:
            broker_call = self.broker_interface.place_market_order(symbol, side, quantity)
        except Exception as exc:  # pragma: no cover - defensive guard for broker wrappers
            self._record_failure(
                stage="broker_invocation",
                message="Broker rejected order upfront",
                intent=intent,
                symbol=symbol,
                side=side,
                quantity=quantity,
                order_type=order_type,
                exception=exc,
            )
            raise PaperBrokerError(f"Broker rejected order upfront: {exc}") from exc

        if not asyncio.iscoroutine(broker_call):
            message = (
                "Broker interface returned a non-awaitable response for place_market_order"
            )
            self._record_failure(
                stage="broker_invocation",
                message=message,
                intent=intent,
                symbol=symbol,
                side=side,
                quantity=quantity,
                order_type=order_type,
            )
            raise PaperBrokerError(message)

        try:
            if self.order_timeout is not None:
                order_id = await asyncio.wait_for(broker_call, timeout=self.order_timeout)
            else:
                order_id = await broker_call
        except Exception as exc:
            self._record_failure(
                stage="broker_submission",
                message="Paper broker failed to submit order",
                intent=intent,
                symbol=symbol,
                side=side,
                quantity=quantity,
                order_type=order_type,
                exception=exc,
            )
            raise PaperBrokerError(f"Paper broker failed to submit order: {exc}") from exc

        if not order_id:
            self._record_failure(
                stage="broker_submission",
                message="Paper broker returned an empty order identifier",
                intent=intent,
                symbol=symbol,
                side=side,
                quantity=quantity,
                order_type=order_type,
            )
            raise PaperBrokerError("Paper broker returned an empty order identifier")

        latency = perf_counter() - start_time
        self._register_success(latency)

        # Release the optimistic reservation now that the broker has accepted the order.
        try:
            if quantity:
                self.portfolio_monitor.release_position(symbol, quantity)
        except Exception:  # pragma: no cover - reservation release is best effort
            logger.debug("Failed to release reserved position after broker submission", exc_info=True)

        self._record_last_order(
            order_id=str(order_id),
            symbol=symbol,
            side=side,
            quantity=float(quantity),
            order_type="market",
            intent=intent,
            latency=latency,
        )
        self._last_error = None
        return str(order_id)

    def _capture_risk_context(self) -> None:
        metadata, error = capture_risk_context(self.risk_context_provider)
        self._last_risk_metadata = metadata
        self._last_risk_error = error

    def _record_last_order(
        self,
        *,
        order_id: str,
        symbol: str,
        side: str,
        quantity: float,
        order_type: str,
        intent: Any,
        latency: float | None = None,
    ) -> None:
        metadata: MutableMapping[str, Any] = {}
        raw_metadata = _extract(intent, "metadata", default={})
        if isinstance(raw_metadata, Mapping):
            metadata.update({str(key): value for key, value in raw_metadata.items()})

        self._last_order = {
            "order_id": order_id,
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "order_type": order_type,
            "metadata": dict(metadata),
        }
        if latency is not None:
            self._last_order["latency_s"] = latency

        if self._last_risk_metadata is not None:
            self._last_order["risk_context"] = dict(self._last_risk_metadata)
        if self._last_risk_error is not None:
            self._last_order["risk_error"] = dict(self._last_risk_error)

    def describe_last_error(self) -> Mapping[str, object] | None:
        """Expose the most recent execution failure metadata, if any."""

        if self._last_error is None:
            return None
        return dict(self._last_error)

    def describe_metrics(self) -> Mapping[str, object]:
        """Return aggregated execution metrics for paper broker submissions."""

        avg_latency: float | None = None
        if self._latency_samples > 0:
            avg_latency = self._total_latency / self._latency_samples
        return {
            "total_orders": self._total_orders,
            "successful_orders": self._successful_orders,
            "failed_orders": self._failed_orders,
            "latency_samples": self._latency_samples,
            "avg_latency_s": avg_latency,
            "last_latency_s": self._last_latency,
        }

    def _record_failure(
        self,
        *,
        stage: str,
        message: str,
        intent: Any,
        symbol: str | None,
        side: str | None,
        quantity: float | None,
        order_type: str | None,
        exception: Exception | None = None,
    ) -> None:
        payload: MutableMapping[str, Any] = {
            "stage": stage,
            "message": message,
        }
        if symbol:
            payload["symbol"] = symbol
        if side:
            payload["side"] = side
        if quantity is not None:
            payload["quantity"] = float(quantity)
        if order_type:
            payload["order_type"] = order_type
        if exception is not None:
            payload["exception_type"] = exception.__class__.__name__
            payload["exception"] = str(exception)

        metadata = _extract(intent, "metadata", default=None)
        if isinstance(metadata, Mapping):
            payload["metadata"] = {str(key): value for key, value in metadata.items()}

        if self._last_risk_metadata is not None:
            payload["risk_context"] = dict(self._last_risk_metadata)
        if self._last_risk_error is not None:
            payload["risk_error"] = dict(self._last_risk_error)

        self._register_failure()
        self._last_error = dict(payload)

    def _register_success(self, latency: float | None) -> None:
        self._total_orders += 1
        self._successful_orders += 1
        if latency is not None:
            self._latency_samples += 1
            self._total_latency += latency
        self._last_latency = latency

    def _register_failure(self) -> None:
        self._total_orders += 1
        self._failed_orders += 1
        self._last_latency = None
