"""cTrader bridge that emits structured order events on the EMP event bus.

The historical OpenAPI integration shipped opaque domain event objects that were
removed alongside the `src.domain` package.  Hidden call sites still expect a
thin adapter capable of turning cTrader execution callbacks into the canonical
order lifecycle payloads consumed by downstream processors (order state
machines, telemetry, reconciliation, etc.).

This module provides a minimal async-friendly adapter that:

* normalises raw cTrader execution payloads into typed dictionaries;
* publishes lifecycle events to ``trading.order.lifecycle`` and generic
  ``order_update`` topics; and
* safely interoperates with both the async :class:`~src.core.event_bus.EventBus`
  and synchronous ``TopicBus`` shims by awaiting coroutine results when
  necessary.
"""

from __future__ import annotations

import inspect
from datetime import datetime, timezone
from typing import Any, Awaitable, Iterable, Mapping, MutableMapping, TypedDict, cast

from src.core.event_bus import Event, publish_event
from src.operational.structured_logging import get_logger

logger = get_logger(__name__)

__all__ = [
    "CTraderAdapter",
    "CTraderOrderUpdatePayload",
    "CTraderLifecycleEventPayload",
]


_OrderLifecycleEventType = {
    "acknowledged",
    "partial_fill",
    "filled",
    "cancelled",
    "rejected",
    "cancel_rejected",
}

_EXEC_TYPE_MAP = {
    "0": "acknowledged",
    "1": "partial_fill",
    "2": "filled",
    "4": "cancelled",
    "8": "rejected",
}

_EVENT_ALIASES = {
    "ack": "acknowledged",
    "acknowledge": "acknowledged",
    "order_acknowledged": "acknowledged",
    "partial": "partial_fill",
    "partialfill": "partial_fill",
    "partial-fill": "partial_fill",
    "fill": "filled",
    "filled": "filled",
    "cancel": "cancelled",
    "cancelled": "cancelled",
    "canceled": "cancelled",
    "reject": "rejected",
    "rejected": "rejected",
    "cancel_rejected": "cancel_rejected",
    "cancel-rejected": "cancel_rejected",
}

_KEY_ALIASES: Mapping[str, tuple[str, ...]] = {
    "order_id": (
        "order_id",
        "orderId",
        "OrderID",
        "cl_ord_id",
        "clOrdId",
        "ClOrdID",
        "id",
        "order",
    ),
    "symbol": ("symbol", "Symbol", "instrument", "Instrument"),
    "side": ("side", "Side", "direction", "Direction"),
    "status": (
        "status",
        "Status",
        "order_status",
        "orderStatus",
        "state",
        "State",
    ),
    "event": (
        "event",
        "event_type",
        "eventType",
        "exec_event",
        "ExecEvent",
        "lifecycle_event",
    ),
    "exec_type": ("exec_type", "ExecType", "execType", "type", "Type"),
    "exec_id": ("exec_id", "ExecID", "execID", "execId"),
    "last_qty": (
        "last_qty",
        "lastQty",
        "LastQty",
        "last_quantity",
        "lastQuantity",
    ),
    "last_px": (
        "last_px",
        "lastPx",
        "LastPx",
        "last_price",
        "lastPrice",
    ),
    "cum_qty": (
        "cum_qty",
        "cumQty",
        "CumQty",
        "cumulative_qty",
        "cumulativeQty",
        "filled_qty",
        "filledQty",
        "FilledQty",
    ),
    "leaves_qty": (
        "leaves_qty",
        "leavesQty",
        "LeavesQty",
        "remaining_qty",
        "remainingQty",
    ),
    "order_quantity": (
        "order_quantity",
        "quantity",
        "Quantity",
        "orderQty",
        "OrderQty",
        "qty",
        "Qty",
    ),
    "avg_px": (
        "avg_px",
        "avgPx",
        "average_price",
        "averagePrice",
        "avg_price",
        "avgPrice",
    ),
    "timestamp": (
        "timestamp",
        "Timestamp",
        "event_time",
        "eventTime",
        "time",
        "Time",
        "created_at",
        "createdAt",
        "update_time",
        "updateTime",
    ),
    "reason": ("reason", "Reason", "text", "Text", "message", "Message"),
    "metadata": ("metadata", "Metadata"),
    "risk_decision": ("risk_decision", "riskDecision", "decision", "Decision"),
    "policy_snapshot": (
        "policy_snapshot",
        "policySnapshot",
        "snapshot",
        "Snapshot",
    ),
    "risk_context": ("risk_context", "riskContext"),
    "risk_reference": ("risk_reference", "riskReference"),
    "runbook": ("runbook", "Runbook"),
    "violations": ("violations", "Violations"),
    "severity": ("severity", "Severity"),
    "policy_violation": ("policy_violation", "policyViolation"),
}


class CTraderOrderUpdatePayload(TypedDict, total=False):
    """Structured order update shared across event topics."""

    order_id: str
    event: str
    status: str
    symbol: str
    side: str
    order_quantity: float
    filled_qty: float
    avg_px: float
    exec_type: str
    exec_id: str
    last_qty: float
    last_px: float
    cum_qty: float
    leaves_qty: float
    reason: str
    timestamp: str
    metadata: dict[str, Any]
    risk_decision: dict[str, Any]
    policy_snapshot: dict[str, Any]
    risk_context: dict[str, Any]
    risk_reference: dict[str, Any]
    runbook: str
    violations: list[str]
    policy_violation: bool
    severity: str
    source: str


class CTraderLifecycleEventPayload(CTraderOrderUpdatePayload, total=False):
    """Lifecycle event payload with canonical event classification."""

    event: str


class CTraderAdapter:
    """Normalise cTrader execution reports and publish lifecycle events."""

    lifecycle_topic = "trading.order.lifecycle"
    order_update_topic = "order_update"

    def __init__(
        self,
        event_bus: Any | None = None,
        *,
        source: str = "ctrader.adapter",
    ) -> None:
        self.event_bus = event_bus
        self.source = source

    # ------------------------------------------------------------------
    async def publish_order_update(
        self,
        order_id: str,
        update: Mapping[str, Any] | None = None,
    ) -> None:
        """Publish a plain order update event to the bus."""

        payload = self._build_order_update(order_id, update or {})
        await self._emit_topic_event(self.order_update_topic, payload)

    async def publish_lifecycle_event(
        self,
        event_type: str,
        order_id: str,
        update: Mapping[str, Any] | None = None,
    ) -> None:
        """Publish a lifecycle event to the canonical topic."""

        canonical = self._normalise_event_type(event_type)
        if canonical not in _OrderLifecycleEventType:
            raise ValueError(f"Unsupported lifecycle event type: {event_type}")
        payload = self._build_lifecycle_event(order_id, canonical, update or {})
        await self._emit_topic_event(self.lifecycle_topic, payload)

    async def process_execution_report(self, payload: Mapping[str, Any]) -> None:
        """Handle a raw cTrader execution payload and broadcast events."""

        order_id = self._lookup(payload, "order_id")
        if not order_id:
            raise ValueError("Execution payload missing order identifier")

        update_payload = self._build_order_update(order_id, payload)
        await self._emit_topic_event(self.order_update_topic, update_payload)

        event_type = self._resolve_event_type(payload)
        if event_type is None:
            return
        lifecycle_payload = self._build_lifecycle_event(
            order_id,
            event_type,
            payload,
            base_update=update_payload,
        )
        await self._emit_topic_event(self.lifecycle_topic, lifecycle_payload)

    async def publish_plain_event(self, topic: str, payload: Mapping[str, Any]) -> None:
        """Publish an arbitrary event to ``topic`` without additional shaping."""

        shaped = dict(payload)
        shaped.setdefault("source", self.source)
        await self._emit_topic_event(topic, shaped)

    # ------------------------------------------------------------------
    def _build_order_update(
        self,
        order_id: str,
        update: Mapping[str, Any],
    ) -> CTraderOrderUpdatePayload:
        payload: dict[str, Any] = {"order_id": order_id, "source": self.source}

        self._update_if_not_none(payload, "event", self._lookup(update, "event"))
        self._update_if_not_none(payload, "status", self._lookup(update, "status"))
        self._update_if_not_none(payload, "symbol", self._lookup(update, "symbol"))
        side = self._lookup(update, "side")
        if isinstance(side, str) and side:
            payload["side"] = side.upper()
        quantity = self._coerce_float(self._lookup(update, "order_quantity"))
        if quantity is not None:
            payload["order_quantity"] = quantity
        filled = self._coerce_float(self._lookup(update, "cum_qty"))
        if filled is not None:
            payload["filled_qty"] = filled
            payload["cum_qty"] = filled
        avg_px = self._coerce_float(self._lookup(update, "avg_px"))
        if avg_px is not None:
            payload["avg_px"] = avg_px
        exec_type = self._lookup(update, "exec_type")
        if exec_type is not None:
            payload["exec_type"] = str(exec_type)
        exec_id = self._lookup(update, "exec_id")
        if exec_id is not None:
            payload["exec_id"] = str(exec_id)
        last_qty = self._coerce_float(self._lookup(update, "last_qty"))
        if last_qty is not None:
            payload["last_qty"] = last_qty
        last_px = self._coerce_float(self._lookup(update, "last_px"))
        if last_px is not None:
            payload["last_px"] = last_px
        leaves_qty = self._coerce_float(self._lookup(update, "leaves_qty"))
        if leaves_qty is not None:
            payload["leaves_qty"] = leaves_qty
        reason = self._lookup(update, "reason")
        if isinstance(reason, str) and reason:
            payload["reason"] = reason
        timestamp = self._serialise_timestamp(self._lookup(update, "timestamp"))
        payload["timestamp"] = timestamp

        for key in (
            "metadata",
            "risk_decision",
            "policy_snapshot",
            "risk_context",
            "risk_reference",
        ):
            mapping = self._ensure_mapping(self._lookup(update, key))
            if mapping:
                payload[key] = mapping

        violations = self._ensure_sequence(self._lookup(update, "violations"))
        if violations:
            payload["violations"] = violations

        severity = self._lookup(update, "severity")
        if isinstance(severity, str) and severity:
            payload["severity"] = severity
        policy_violation = self._lookup(update, "policy_violation")
        if isinstance(policy_violation, bool):
            payload["policy_violation"] = policy_violation
        runbook = self._lookup(update, "runbook")
        if isinstance(runbook, str) and runbook:
            payload["runbook"] = runbook

        return cast(CTraderOrderUpdatePayload, payload)

    def _build_lifecycle_event(
        self,
        order_id: str,
        event_type: str,
        update: Mapping[str, Any],
        *,
        base_update: Mapping[str, Any] | None = None,
    ) -> CTraderLifecycleEventPayload:
        payload = dict(base_update or self._build_order_update(order_id, update))
        payload["order_id"] = order_id
        payload["event"] = event_type
        if "timestamp" not in payload:
            payload["timestamp"] = self._serialise_timestamp(self._lookup(update, "timestamp"))
        if "filled_qty" not in payload:
            filled = self._coerce_float(self._lookup(update, "cum_qty"))
            if filled is not None:
                payload["filled_qty"] = filled
        if "last_qty" not in payload:
            last_qty = self._coerce_float(self._lookup(update, "last_qty"))
            if last_qty is not None:
                payload["last_qty"] = last_qty
        if "last_px" not in payload:
            last_px = self._coerce_float(self._lookup(update, "last_px"))
            if last_px is not None:
                payload["last_px"] = last_px
        if "cum_qty" not in payload:
            cum_qty = self._coerce_float(self._lookup(update, "cum_qty"))
            if cum_qty is not None:
                payload["cum_qty"] = cum_qty
        if "leaves_qty" not in payload:
            leaves_qty = self._coerce_float(self._lookup(update, "leaves_qty"))
            if leaves_qty is not None:
                payload["leaves_qty"] = leaves_qty

        return cast(CTraderLifecycleEventPayload, payload)

    # ------------------------------------------------------------------
    def _resolve_event_type(self, payload: Mapping[str, Any]) -> str | None:
        explicit = self._lookup(payload, "event")
        if isinstance(explicit, str):
            resolved = self._normalise_event_type(explicit)
            if resolved in _OrderLifecycleEventType:
                return resolved

        exec_type = self._lookup(payload, "exec_type")
        if exec_type is not None:
            resolved = _EXEC_TYPE_MAP.get(str(exec_type))
            if resolved is not None:
                return resolved
        return None

    @staticmethod
    def _normalise_event_type(value: Any) -> str:
        if isinstance(value, str):
            key = value.strip().lower()
            if key in _OrderLifecycleEventType:
                return key
            return _EVENT_ALIASES.get(key, key)
        return str(value)

    def _lookup(self, payload: Mapping[str, Any], field: str) -> Any:
        aliases = _KEY_ALIASES.get(field, (field,))
        for key in aliases:
            if key in payload:
                return payload[key]
        if "_" in field:
            camel = field.split("_")
            if camel:
                candidate = camel[0] + "".join(part.capitalize() for part in camel[1:])
                if candidate in payload:
                    return payload[candidate]
        return None

    @staticmethod
    def _coerce_float(value: Any) -> float | None:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _serialise_timestamp(self, value: Any) -> str:
        if isinstance(value, datetime):
            ts = value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
            return ts.astimezone(timezone.utc).isoformat()
        if isinstance(value, (int, float)):
            return datetime.fromtimestamp(float(value), tz=timezone.utc).isoformat()
        if isinstance(value, str) and value.strip():
            return value
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _ensure_mapping(value: Any) -> dict[str, Any] | None:
        if isinstance(value, Mapping):
            return {str(key): val for key, val in value.items()}
        if isinstance(value, MutableMapping):  # pragma: no cover - defensive
            return dict(value)
        return None

    @staticmethod
    def _ensure_sequence(value: Any) -> list[str] | None:
        if isinstance(value, Iterable) and not isinstance(value, (str, bytes, bytearray)):
            result = [str(item) for item in value if item is not None]
            if result:
                return result
        return None

    @staticmethod
    def _update_if_not_none(target: dict[str, Any], key: str, value: Any) -> None:
        if value is None:
            return
        if isinstance(value, str):
            trimmed = value.strip()
            if not trimmed:
                return
            target[key] = trimmed
            return
        target[key] = value

    async def _emit_topic_event(self, topic: str, payload: Mapping[str, Any]) -> None:
        event_payload = dict(payload)
        event_payload.setdefault("source", self.source)
        bus = self.event_bus

        if bus is not None:
            publish_callable = getattr(bus, "publish", None)
            if callable(publish_callable):
                attempts = [
                    ((Event(type=topic, payload=event_payload, source=self.source),), {}),
                    ((topic, event_payload, self.source), {}),
                    ((topic, event_payload), {"source": self.source}),
                    ((topic, event_payload), {}),
                ]
                for args, kwargs in attempts:
                    try:
                        result = publish_callable(*args, **kwargs)
                    except TypeError:
                        continue
                    except Exception as exc:
                        logger.debug(
                            "ctrader_event_publish_failed",
                            topic=topic,
                            error=str(exc),
                        )
                        break
                    else:
                        if inspect.isawaitable(result):
                            await cast(Awaitable[Any], result)
                        return

            emit_callable = getattr(bus, "emit", None)
            if callable(emit_callable):
                attempts = [
                    ((topic, event_payload, self.source), {}),
                    ((topic, event_payload), {"source": self.source}),
                    ((topic, event_payload), {}),
                ]
                for args, kwargs in attempts:
                    try:
                        result = emit_callable(*args, **kwargs)
                    except TypeError:
                        continue
                    except Exception as exc:
                        logger.debug(
                            "ctrader_event_emit_failed",
                            topic=topic,
                            error=str(exc),
                        )
                        break
                    else:
                        if inspect.isawaitable(result):
                            await cast(Awaitable[Any], result)
                        return

            publish_sync = getattr(bus, "publish_sync", None)
            if callable(publish_sync):
                try:
                    publish_sync(topic, event_payload, source=self.source)
                except Exception as exc:
                    logger.debug(
                        "ctrader_event_publish_sync_failed",
                        topic=topic,
                        error=str(exc),
                    )
                else:
                    return

        try:
            await publish_event(Event(type=topic, payload=event_payload, source=self.source))
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.debug(
                "ctrader_event_global_publish_failed",
                topic=topic,
                error=str(exc),
            )
