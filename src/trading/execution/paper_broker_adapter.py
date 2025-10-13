"""Paper trading broker adapter bridging FIX integrations with the release router.

This module fulfils the roadmap deliverable that wires AlphaTrade's execution
stack into a paper broker endpoint.  The adapter translates validated trade
intents into FIX broker invocations while preserving the deterministic risk
context telemetry expected by governance tooling.
"""

from __future__ import annotations

import asyncio
import logging
from collections import deque
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
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


def _percentile(sorted_values: list[float], percentile: float) -> float | None:
    """Compute a percentile for an already sorted latency sample window."""

    if not sorted_values:
        return None
    percentile = max(0.0, min(100.0, float(percentile)))
    if not math.isfinite(percentile):
        return None
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    rank = (percentile / 100.0) * (len(sorted_values) - 1)
    lower_index = math.floor(rank)
    upper_index = math.ceil(rank)
    lower_value = float(sorted_values[lower_index])
    upper_value = float(sorted_values[upper_index])
    if lower_index == upper_index:
        return lower_value
    weight = rank - lower_index
    return lower_value + (upper_value - lower_value) * weight


@dataclass
class PaperBrokerExecutionAdapter:
    """Adapter that forwards validated intents to a paper FIX broker interface."""

    broker_interface: Any
    portfolio_monitor: PortfolioMonitor
    order_timeout: float | None = 5.0
    failover_threshold: int = 3
    failover_cooldown_seconds: float = 30.0
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
    _latency_history: deque[float] = field(
        default_factory=lambda: deque(maxlen=2048), init=False, repr=False
    )
    _order_history: deque[dict[str, object]] = field(
        default_factory=lambda: deque(maxlen=512), init=False, repr=False
    )
    _error_history: deque[dict[str, object]] = field(
        default_factory=lambda: deque(maxlen=512), init=False, repr=False
    )
    _first_order_time: datetime | None = field(default=None, init=False, repr=False)
    _last_order_time: datetime | None = field(default=None, init=False, repr=False)
    _last_error_time: datetime | None = field(default=None, init=False, repr=False)
    _consecutive_failures: int = field(default=0, init=False, repr=False)
    _failover_until: datetime | None = field(default=None, init=False, repr=False)

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
        self._reset_failover_state()

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
        now = datetime.now(timezone.utc)
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
            "placed_at": now.isoformat(),
        }
        if latency is not None:
            self._last_order["latency_s"] = latency

        if self._last_risk_metadata is not None:
            self._last_order["risk_context"] = dict(self._last_risk_metadata)
        if self._last_risk_error is not None:
            self._last_order["risk_error"] = dict(self._last_risk_error)

        broker_submission = self._fetch_broker_submission()
        if broker_submission:
            self._last_order["broker_submission"] = broker_submission

        self._order_history.append(dict(self._last_order))
        self._last_order_time = now
        if self._first_order_time is None:
            self._first_order_time = now

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

        if self._total_orders > 0:
            total = float(self._total_orders)
            success_ratio = self._successful_orders / total
            failure_ratio = self._failed_orders / total
        else:
            success_ratio = 0.0
            failure_ratio = 0.0

        payload: MutableMapping[str, object | None] = {
            "total_orders": self._total_orders,
            "successful_orders": self._successful_orders,
            "failed_orders": self._failed_orders,
            "latency_samples": self._latency_samples,
            "avg_latency_s": avg_latency,
            "last_latency_s": self._last_latency,
            "success_ratio": success_ratio,
            "failure_ratio": failure_ratio,
        }

        latency_window = list(self._latency_history)
        if latency_window:
            sorted_latencies = sorted(latency_window)
            payload["p50_latency_s"] = _percentile(sorted_latencies, 50.0)
            payload["p90_latency_s"] = _percentile(sorted_latencies, 90.0)
            payload["p99_latency_s"] = _percentile(sorted_latencies, 99.0)
        else:
            payload["p50_latency_s"] = None
            payload["p90_latency_s"] = None
            payload["p99_latency_s"] = None
        payload["latency_history_samples"] = len(latency_window)

        if self._first_order_time is not None:
            payload["first_order_at"] = self._first_order_time.isoformat()
        if self._last_order_time is not None:
            payload["last_order_at"] = self._last_order_time.isoformat()
        if self._last_error_time is not None:
            payload["last_error_at"] = self._last_error_time.isoformat()

        payload["consecutive_failures"] = self._consecutive_failures
        payload["failover_threshold"] = max(0, int(self.failover_threshold))
        failover_snapshot = self.describe_failover()
        payload["failover_active"] = failover_snapshot.get("active", False)
        payload["failover"] = failover_snapshot

        return dict(payload)

    def describe_broker(self) -> Mapping[str, Any] | None:
        """Expose broker interface metadata when available."""

        summary: dict[str, Any] = {}

        describe = getattr(self.broker_interface, "describe", None)
        if callable(describe):
            try:
                snapshot = describe()
            except Exception:  # pragma: no cover - diagnostics only
                logger.debug(
                    "Failed to describe broker interface via adapter",
                    exc_info=True,
                )
            else:
                if isinstance(snapshot, Mapping):
                    summary.update({str(key): value for key, value in snapshot.items()})

        settings = getattr(self.broker_interface, "settings", None)

        attrs = (
            "base_url",
            "order_endpoint",
            "order_id_field",
            "time_in_force",
            "verify_ssl",
            "request_timeout",
            "retry_attempts",
            "retry_backoff_seconds",
            "account_id",
        )

        for attr in attrs:
            if attr in summary:
                continue
            value: Any | None = None
            if settings is not None:
                if isinstance(settings, Mapping):
                    value = settings.get(attr)
                else:
                    value = getattr(settings, attr, None)
            if value is None:
                value = getattr(self.broker_interface, attr, None)
            if value is not None:
                summary[attr] = value

        if not summary:
            return None
        return dict(summary)

    def consume_order_history(self) -> list[Mapping[str, object]]:
        """Return and clear the buffered successfully submitted orders."""

        if not self._order_history:
            return []
        payload = [dict(entry) for entry in self._order_history]
        self._order_history.clear()
        return payload

    def consume_error_history(self) -> list[Mapping[str, object]]:
        """Return and clear the buffered broker submission failures."""

        if not self._error_history:
            return []
        payload = [dict(entry) for entry in self._error_history]
        self._error_history.clear()
        return payload

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

        broker_submission = self._fetch_broker_submission()
        if broker_submission:
            payload["broker_submission"] = broker_submission

        now = datetime.now(timezone.utc)
        payload["recorded_at"] = now.isoformat()

        self._register_failure()
        self._update_failover_state(stage)
        failover_snapshot = self.describe_failover()
        if failover_snapshot.get("active"):
            payload["failover"] = dict(failover_snapshot)
        self._last_error = dict(payload)
        self._error_history.append(dict(self._last_error))
        self._last_error_time = now

    def _register_success(self, latency: float | None) -> None:
        self._total_orders += 1
        self._successful_orders += 1
        if latency is not None:
            self._latency_samples += 1
            self._total_latency += latency
            self._latency_history.append(float(latency))
        self._last_latency = latency

    def _register_failure(self) -> None:
        self._total_orders += 1
        self._failed_orders += 1
        self._last_latency = None

    def _fetch_broker_submission(self) -> Mapping[str, Any] | None:
        """Capture request/response telemetry exposed by the broker interface."""

        interface = self.broker_interface
        if interface is None:
            return None

        accessor_names = (
            "describe_last_submission",
            "get_last_submission",
            "snapshot_last_submission",
            "last_submission",
        )

        for name in accessor_names:
            candidate = getattr(interface, name, None)
            if candidate is None:
                continue
            try:
                if callable(candidate):
                    submission = candidate()
                else:
                    submission = candidate
            except Exception:  # pragma: no cover - diagnostics only
                logger.debug("Broker submission accessor '%s' failed", name, exc_info=True)
                continue
            if isinstance(submission, Mapping):
                return {str(key): value for key, value in submission.items()}
        return None

    def _reset_failover_state(self) -> None:
        self._consecutive_failures = 0
        self._failover_until = None

    def _update_failover_state(self, stage: str) -> None:
        if stage not in {"broker_invocation", "broker_submission"}:
            self._reset_failover_state()
            return

        if self.failover_threshold <= 0:
            return

        self._consecutive_failures += 1
        cooldown = max(0.0, float(self.failover_cooldown_seconds))
        if cooldown <= 0.0:
            return
        if self._consecutive_failures < self.failover_threshold:
            return
        self._failover_until = datetime.now(timezone.utc) + timedelta(seconds=cooldown)

    def should_block_orders(self, _intent: Any | None = None) -> Mapping[str, Any] | None:
        if self._failover_until is None:
            return None
        now = datetime.now(timezone.utc)
        if now >= self._failover_until:
            self._reset_failover_state()
            return None
        remaining = max(0.0, (self._failover_until - now).total_seconds())
        payload: MutableMapping[str, Any] = {
            "reason": "paper_broker_failover",
            "retry_in_seconds": remaining,
            "consecutive_failures": self._consecutive_failures,
            "failover_resumes_at": self._failover_until.isoformat(),
        }
        last_error = self.describe_last_error()
        if last_error:
            payload["last_error"] = dict(last_error)
        submission = self._fetch_broker_submission()
        if submission:
            payload["last_submission"] = submission
        payload["failover"] = self.describe_failover()
        return dict(payload)

    def describe_failover(self) -> Mapping[str, Any]:
        payload: MutableMapping[str, Any] = {
            "threshold": max(0, int(self.failover_threshold)),
            "consecutive_failures": self._consecutive_failures,
        }
        if self._failover_until is None:
            payload["active"] = False
        else:
            now = datetime.now(timezone.utc)
            remaining = max(0.0, (self._failover_until - now).total_seconds())
            payload.update(
                {
                    "active": remaining > 0.0,
                    "resumes_at": self._failover_until.isoformat(),
                    "retry_in_seconds": remaining,
                }
            )
        return dict(payload)
