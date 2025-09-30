"""Drop-copy ingestion and reconciliation helpers for the FIX pilot."""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Callable, Coroutine, Mapping, MutableMapping, Optional

from src.core.event_bus import Event, EventBus, TopicBus, get_global_bus

logger = logging.getLogger(__name__)


def _decode_value(value: object) -> object | None:
    """Decode FIX field values into JSON-friendly representations."""

    if value is None:
        return None
    if isinstance(value, (int, float, bool)):
        return value
    if isinstance(value, (bytes, bytearray)):
        try:
            return value.decode()
        except Exception:  # pragma: no cover - best effort only
            return value.hex()
    return str(value)


def _coerce_float(value: object | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return None


def _sanitize_mapping(mapping: Mapping[str, object]) -> Mapping[str, object]:
    return {key: value for key, value in mapping.items() if value is not None}


@dataclass(frozen=True)
class DropcopyEvent:
    """Normalised drop-copy execution report."""

    order_id: str | None
    exec_id: str | None
    exec_type: str | None
    order_status: str | None
    last_qty: float | None
    last_px: float | None
    leaves_qty: float | None
    received_at: datetime
    raw: Mapping[str, object]

    def as_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "order_id": self.order_id,
            "exec_id": self.exec_id,
            "exec_type": self.exec_type,
            "order_status": self.order_status,
            "last_qty": self.last_qty,
            "last_px": self.last_px,
            "leaves_qty": self.leaves_qty,
            "received_at": self.received_at.isoformat(),
            "raw": dict(self.raw),
        }
        return {key: value for key, value in payload.items() if value is not None}


TaskFactory = Callable[[Coroutine[Any, Any, Any], Optional[str]], asyncio.Task[Any]]


class FixDropcopyReconciler:
    """Consume drop-copy messages and reconcile them with broker state."""

    def __init__(
        self,
        *,
        event_bus: EventBus | None = None,
        broker_order_lookup: Callable[[], Mapping[str, Mapping[str, object]]] | None = None,
        channel: str = "telemetry.execution.dropcopy",
        queue_maxsize: int = 2048,
        history: int = 256,
        logger: logging.Logger | None = None,
        task_factory: TaskFactory | None = None,
    ) -> None:
        bus = event_bus or get_global_bus()
        self.event_bus: EventBus | TopicBus = bus
        self._broker_order_lookup = broker_order_lookup
        self._channel = channel
        self._logger = logger or logging.getLogger(__name__)
        self.dropcopy_queue: asyncio.Queue[Any] = asyncio.Queue(maxsize=queue_maxsize)
        self.running = False
        self._dropcopy_task: asyncio.Task[Any] | None = None
        self._events: deque[DropcopyEvent] = deque(maxlen=max(8, history))
        self._order_events: MutableMapping[str, DropcopyEvent] = {}
        self._unmatched_reports: deque[Mapping[str, object]] = deque(maxlen=64)
        self._task_factory = task_factory

    async def start(self) -> None:
        if self.running:
            return
        self.running = True
        self._dropcopy_task = self._spawn_task(
            self._process_queue(), name="fix-dropcopy-reconciler"
        )

    async def stop(self) -> None:
        if not self.running:
            return
        self.running = False
        task = self._dropcopy_task
        if task is not None:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
        self._dropcopy_task = None

    async def _process_queue(self) -> None:
        while True:
            try:
                message = await self.dropcopy_queue.get()
            except asyncio.CancelledError:
                break

            if not self.running:
                if message is None:
                    break
                continue

            if message is None:
                continue

            event = self._normalise_message(message)
            if event is None:
                continue

            if event.order_id:
                self._order_events[event.order_id] = event
            else:
                self._unmatched_reports.append(event.as_dict())
            self._events.append(event)

            payload = event.as_dict()
            try:
                if isinstance(self.event_bus, TopicBus):
                    self.event_bus.publish_sync(self._channel, payload)
                    continue

                publish = getattr(self.event_bus, "publish", None)
                if publish is not None:
                    await publish(Event(self._channel, payload))
                    continue

                emit_nowait = getattr(self.event_bus, "emit_nowait", None)
                if callable(emit_nowait):
                    emit_nowait(Event(self._channel, payload))
                    continue

                self._logger.debug(
                    "Event bus object does not expose publish/emit interfaces: %s",
                    type(self.event_bus),
                )
            except Exception:  # pragma: no cover - diagnostics only
                self._logger.debug("Failed to publish drop-copy event", exc_info=True)

    def _spawn_task(
        self, coro: Coroutine[Any, Any, Any], *, name: str | None = None
    ) -> asyncio.Task[Any]:
        if self._task_factory is not None:
            return self._task_factory(coro, name)
        return asyncio.create_task(coro, name=name)

    # ------------------------------------------------------------------
    def get_backlog(self) -> int:
        return self.dropcopy_queue.qsize()

    def get_last_event(self) -> Mapping[str, object] | None:
        if not self._events:
            return None
        return self._events[-1].as_dict()

    def recent_events(self, limit: int = 5) -> list[Mapping[str, object]]:
        if limit <= 0:
            return []
        return [event.as_dict() for event in list(self._events)[-limit:]]

    def reconciliation_summary(self) -> Mapping[str, object]:
        summary: dict[str, object] = {
            "observed_orders": len(self._order_events),
            "backlog": self.get_backlog(),
            "recent_events": self.recent_events(5),
        }

        last_event = self.get_last_event()
        if last_event is not None:
            summary["last_event"] = last_event

        if self._unmatched_reports:
            summary["unmatched_reports"] = list(self._unmatched_reports)

        if self._broker_order_lookup is None:
            return summary

        try:
            broker_orders = self._broker_order_lookup()
        except Exception:  # pragma: no cover - diagnostics only
            self._logger.debug("Failed to collect broker orders", exc_info=True)
            return summary

        normalized_orders: dict[str, Mapping[str, object]] = {}
        if isinstance(broker_orders, Mapping):
            normalized_orders = {
                str(order_id): dict(order_details)
                if isinstance(order_details, Mapping)
                else {"details": order_details}
                for order_id, order_details in broker_orders.items()
            }

        summary["total_orders"] = len(normalized_orders)

        unmatched = sorted(
            order_id for order_id in normalized_orders if order_id not in self._order_events
        )
        if unmatched:
            summary["orders_without_dropcopy"] = unmatched

        mismatches: list[str] = []
        for order_id, order_payload in normalized_orders.items():
            drop_event = self._order_events.get(order_id)
            if drop_event is None:
                continue
            broker_status = str(order_payload.get("status", "")).upper()
            exec_type = (drop_event.exec_type or "").upper()
            if broker_status and exec_type and exec_type not in broker_status:
                mismatches.append(order_id)
        if mismatches:
            summary["status_mismatches"] = sorted(mismatches)

        return summary

    # ------------------------------------------------------------------
    def _normalise_message(self, message: Any) -> DropcopyEvent | None:
        payload: MutableMapping[int, object] = {}
        if isinstance(message, Mapping):
            for raw_key, value in message.items():
                key: int | None
                if isinstance(raw_key, int):
                    key = raw_key
                else:
                    try:
                        key = int(str(raw_key))
                    except (TypeError, ValueError):
                        key = None
                if key is None:
                    continue
                payload[key] = value
        elif hasattr(message, "get"):
            for tag in (11, 17, 31, 32, 39, 150, 151, 14):
                try:
                    payload[tag] = message.get(tag)
                except Exception:  # pragma: no cover - diagnostics only
                    continue
        else:  # pragma: no cover - unexpected payload types
            self._logger.debug("Unsupported drop-copy message: %s", type(message))
            return None

        raw = {
            str(tag): _decode_value(value)
            for tag, value in payload.items()
            if _decode_value(value) is not None
        }

        order_id = _decode_value(payload.get(11))
        exec_id = _decode_value(payload.get(17))
        exec_type = _decode_value(payload.get(150))
        order_status = _decode_value(payload.get(39))
        last_qty = _coerce_float(_decode_value(payload.get(32)))
        last_px = _coerce_float(_decode_value(payload.get(31)))
        leaves_qty = _coerce_float(_decode_value(payload.get(151)))

        if order_id is None and exec_id is None and exec_type is None and order_status is None:
            # Nothing meaningful in the payload
            return None

        return DropcopyEvent(
            order_id=str(order_id) if order_id is not None else None,
            exec_id=str(exec_id) if exec_id is not None else None,
            exec_type=str(exec_type) if exec_type is not None else None,
            order_status=str(order_status) if order_status is not None else None,
            last_qty=last_qty,
            last_px=last_px,
            leaves_qty=leaves_qty,
            received_at=datetime.now(tz=UTC),
            raw=_sanitize_mapping(raw),
        )


__all__ = ["FixDropcopyReconciler", "DropcopyEvent"]
