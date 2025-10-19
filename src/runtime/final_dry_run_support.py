"""Helpers for wiring final dry run evidence sinks into the runtime."""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, TYPE_CHECKING

from src.core.event_bus import Event, EventBus, SubscriptionHandle

if TYPE_CHECKING:  # pragma: no cover - imported for type checking only
    from src.runtime.predator_app import ProfessionalPredatorApp


logger = logging.getLogger(__name__)


class FinalDryRunPerformanceWriter:
    """Persist strategy performance telemetry snapshots for final dry runs."""

    def __init__(self, path: Path, *, run_label: str | None = None) -> None:
        self._path = path
        self._run_label = (run_label or "").strip() or None
        self._queue: asyncio.Queue[
            tuple[dict[str, Any], asyncio.Future[None]] | None
        ] = asyncio.Queue(maxsize=32)
        self._task: asyncio.Task[Any] | None = None
        self._subscription: SubscriptionHandle | None = None
        self._bus: EventBus | None = None
        self._sequence = 0
        self._stopping = False

    def install(self, app: "ProfessionalPredatorApp") -> None:
        """Attach the writer to the runtime event bus and schedule the worker."""

        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._write_placeholder()

        bus = app.event_bus
        self._subscription = bus.subscribe(
            "telemetry.strategy.performance", self._handle_event
        )
        self._bus = bus

        self._task = app.create_background_task(
            self._drain_queue(),
            name="final-dry-run.performance-writer",
            metadata={"output_path": self._path.as_posix()},
        )
        app.add_cleanup_callback(self.close)

    async def close(self) -> None:
        """Flush outstanding events and detach the writer from the bus."""

        if self._subscription is not None and self._bus is not None:
            try:
                self._bus.unsubscribe(self._subscription)
            except Exception:  # pragma: no cover - defensive shutdown
                logger.debug(
                    "Failed to unsubscribe final dry run performance writer",
                    exc_info=True,
                )
            self._subscription = None

        if not self._stopping:
            self._stopping = True
            await self._queue.put(None)

        if self._task is not None:
            try:
                await self._task
            except asyncio.CancelledError:  # pragma: no cover - supervisor cancellations
                pass
            self._task = None

    def _write_placeholder(self) -> None:
        payload: dict[str, Any] = {
            "status": "waiting",
            "message": "Awaiting strategy performance telemetry.",
            "updated_at": datetime.now(tz=UTC).isoformat(),
        }
        if self._run_label:
            payload["run_label"] = self._run_label
        self._atomic_write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    async def _handle_event(self, event: Event) -> None:
        record = self._normalise_event(event)
        loop = asyncio.get_running_loop()
        future: asyncio.Future[None] = loop.create_future()
        await self._queue.put((record, future))
        await future

    def _normalise_event(self, event: Event) -> dict[str, Any]:
        snapshot: Mapping[str, Any]
        payload = event.payload
        if isinstance(payload, Mapping):
            snapshot = dict(payload)
        else:
            snapshot = {}

        record: dict[str, Any] = {
            "status": "ok",
            "captured_at": datetime.now(tz=UTC).isoformat(),
            "event_type": event.type,
            "snapshot": snapshot,
        }
        if self._run_label:
            record["run_label"] = self._run_label
        if event.source:
            record["event_source"] = event.source

        generated_at = snapshot.get("generated_at")
        if isinstance(generated_at, str):
            record["snapshot_generated_at"] = generated_at
        snapshot_status = snapshot.get("status")
        if isinstance(snapshot_status, str):
            record["snapshot_status"] = snapshot_status

        return record

    async def _drain_queue(self) -> None:
        while True:
            item = await self._queue.get()
            if item is None:
                break
            record, future = item
            self._sequence += 1
            record["sequence"] = self._sequence
            record.setdefault("updated_at", datetime.now(tz=UTC).isoformat())
            try:
                await self._write_payload(record)
            except Exception as exc:
                if not future.done():
                    future.set_exception(exc)
                raise
            else:
                if not future.done():
                    future.set_result(None)

    async def _write_payload(self, payload: Mapping[str, Any]) -> None:
        text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
        await asyncio.to_thread(self._atomic_write_text, text)

    def _atomic_write_text(self, text: str) -> None:
        temp_path = self._path.with_name(f".{self._path.name}.tmp")
        temp_path.write_text(text, encoding="utf-8")
        temp_path.replace(self._path)


def configure_final_dry_run_support(app: "ProfessionalPredatorApp") -> None:
    """Enable final dry run evidence sinks when the config provides paths."""

    extras = app.config.extras or {}
    raw_path = extras.get("FINAL_DRY_RUN_PERFORMANCE_PATH") or extras.get(
        "PERFORMANCE_METRICS_PATH"
    )
    if not raw_path:
        logger.debug("Final dry run performance path not configured; skipping writer")
        return

    path = Path(str(raw_path)).expanduser()
    run_label = (extras.get("FINAL_DRY_RUN_LABEL") or "").strip() or None

    writer = FinalDryRunPerformanceWriter(path, run_label=run_label)
    try:
        writer.install(app)
    except Exception:  # pragma: no cover - defensive guard
        logger.exception(
            "Failed to initialise final dry run performance writer",
            extra={"performance_path": path.as_posix()},
        )


__all__ = [
    "FinalDryRunPerformanceWriter",
    "configure_final_dry_run_support",
]
