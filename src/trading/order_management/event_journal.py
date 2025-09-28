"""Utilities for persisting order lifecycle events for replay/audit."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional

from .order_state_machine import OrderExecutionEvent, OrderLifecycleSnapshot

__all__ = ["OrderEventJournal", "InMemoryOrderEventJournal"]

logger = logging.getLogger(__name__)


class OrderEventJournal:
    """Append-only event journal backed by Parquet with JSON fallback."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._json_fallback_path = self._path.with_suffix(self._path.suffix + ".jsonl")

        self._pd = None
        self._parquet_available = False
        try:  # pragma: no cover - optional dependency discovery
            import pandas as pd  # type: ignore

            self._pd = pd
            # pandas requires a parquet engine; pyarrow is the default recommendation
            try:
                import pyarrow  # noqa: F401  # type: ignore

                self._parquet_available = True
            except Exception:  # pragma: no cover - best effort detection
                self._parquet_available = False
        except Exception:  # pragma: no cover - optional dependency discovery
            self._pd = None
            self._parquet_available = False

    # ------------------------------------------------------------------
    def append(
        self,
        event: OrderExecutionEvent,
        snapshot: OrderLifecycleSnapshot,
        *,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Persist an execution event and the resulting snapshot."""

        record = {
            "order_id": event.order_id,
            "event_type": event.event_type,
            "exec_type": event.exec_type,
            "last_quantity": event.last_quantity,
            "last_price": event.last_price,
            "cumulative_quantity": event.cumulative_quantity,
            "leaves_quantity": event.leaves_quantity,
            "event_timestamp": event.timestamp.isoformat(),
            "symbol": snapshot.symbol,
            "side": snapshot.side,
            "order_quantity": snapshot.order_quantity,
            "account": snapshot.account,
            "status": snapshot.status.value,
            "filled_quantity": snapshot.filled_quantity,
            "remaining_quantity": snapshot.remaining_quantity,
            "average_fill_price": snapshot.average_fill_price,
            "last_event": snapshot.last_event,
            "last_update": snapshot.last_update.isoformat(),
        }
        if extra:
            record.update(extra)

        if self._parquet_available and self._pd is not None:
            self._append_parquet(record)
        else:
            self._append_json(record)

    # ------------------------------------------------------------------
    def append_error(self, payload: Dict[str, Any], *, reason: str) -> None:
        """Persist malformed events into a dead-letter log for later inspection."""

        record = {
            "reason": reason,
            "payload": payload,
        }
        with self._dead_letter_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str) + "\n")

    @property
    def path(self) -> Path:
        """Path to the primary journal artefact."""

        return self._path

    # ------------------------------------------------------------------
    def _append_parquet(self, record: Dict[str, Any]) -> None:
        assert self._pd is not None
        try:
            df = self._pd.DataFrame([record])
            if self._path.exists():
                existing = self._pd.read_parquet(self._path)
                df = self._pd.concat([existing, df], ignore_index=True)
            df.to_parquet(self._path, index=False)
        except Exception as exc:  # pragma: no cover - fallback path
            logger.warning(
                "Falling back to JSON journaling due to parquet error: %s", exc
            )
            self._append_json(record)

    def _append_json(self, record: Dict[str, Any]) -> None:
        with self._json_fallback_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str) + "\n")

    @property
    def _dead_letter_path(self) -> Path:
        return self._path.with_suffix(self._path.suffix + ".deadletter.jsonl")


class InMemoryOrderEventJournal(OrderEventJournal):
    """Testing helper that stores events in memory instead of disk."""

    def __init__(self) -> None:
        self.records: list[Dict[str, Any]] = []
        self.errors: list[Dict[str, Any]] = []
        self._path = Path("/dev/null")
        self._json_fallback_path = Path("/dev/null")
        self._pd = None
        self._parquet_available = False

    def append(
        self,
        event: OrderExecutionEvent,
        snapshot: OrderLifecycleSnapshot,
        *,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        record = {
            "event": asdict(event),
            "snapshot": {
                "order_id": snapshot.order_id,
                "symbol": snapshot.symbol,
                "side": snapshot.side,
                "order_quantity": snapshot.order_quantity,
                "account": snapshot.account,
                "status": snapshot.status.value,
                "filled_quantity": snapshot.filled_quantity,
                "remaining_quantity": snapshot.remaining_quantity,
                "average_fill_price": snapshot.average_fill_price,
                "last_event": snapshot.last_event,
            },
        }
        if extra:
            record["extra"] = extra
        self.records.append(record)

    def append_error(self, payload: Dict[str, Any], *, reason: str) -> None:
        self.errors.append({"payload": payload, "reason": reason})
