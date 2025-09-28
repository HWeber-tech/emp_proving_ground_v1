"""Utilities for loading and replaying order event journals.

The order management roadmap emphasises reproducible dry-runs and analytics
backed by the canonical journal stored in
``data_foundation/events/order_events.parquet`` (with an automatic JSONL
fallback).  Multiple operational scripts needed to parse those artefacts and
replay fills into :class:`~src.trading.order_management.position_tracker.PositionTracker`.

This module centralises that logic so new tooling – including the capital
efficiency memo generator – can share a single, well-tested implementation
without copy/paste.  The helpers are defensive: they accept partially
populated records, tolerate missing optional dependencies (``pandas`` /
``pyarrow``), and normalise timestamps.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

from .position_tracker import PositionTracker

__all__ = [
    "ProcessedFill",
    "load_order_journal",
    "replay_journal_into_tracker",
]


@dataclass(slots=True)
class ProcessedFill:
    """Normalised representation of a fill extracted from the journal."""

    timestamp: datetime
    symbol: str
    quantity: float
    price: float
    side: str
    account: str | None
    order_id: str

    @property
    def notional(self) -> float:
        """Return the absolute notional traded for the fill."""

        return abs(self.quantity * self.price)


def load_order_journal(path: Path) -> list[dict[str, Any]]:
    """Load raw order events from ``path``.

    The function understands both Parquet and JSONL representations produced by
    :class:`src.trading.order_management.event_journal.OrderEventJournal`.  If
    Parquet support is unavailable at runtime it transparently falls back to
    the JSONL mirror.
    """

    if path.suffix == ".parquet":
        try:  # pragma: no cover - optional dependency discovery
            import pandas as pd  # type: ignore
        except Exception:
            fallback = path.with_suffix(path.suffix + ".jsonl")
            if fallback.exists():
                path = fallback
            else:
                raise RuntimeError(
                    "Parquet support requires pandas/pyarrow; provide the JSONL fallback",
                ) from None
        else:  # pragma: no branch - thin wrapper
            df = pd.read_parquet(path)  # type: ignore[attr-defined]
            return df.to_dict(orient="records")

    if path.suffix.endswith(".jsonl") or path.name.endswith(".jsonl"):
        records: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                payload = line.strip()
                if not payload:
                    continue
                records.append(json.loads(payload))
        return records

    raise RuntimeError(f"Unsupported journal format for {path}")


def replay_journal_into_tracker(
    records: Iterable[dict[str, Any]],
    tracker: PositionTracker,
    *,
    on_fill: Callable[[ProcessedFill, PositionTracker], None] | None = None,
) -> list[ProcessedFill]:
    """Replay *records* into ``tracker`` and return the processed fills.

    Args:
        records: Iterable of raw event dictionaries.
        tracker: Position tracker instance mutated in-place.
        on_fill: Optional callback executed after each processed fill.
    """

    processed: list[ProcessedFill] = []
    prior_filled = defaultdict(float)

    for record in _iter_sorted_records(records):
        fill = _extract_fill(record, prior_filled)
        if fill is None:
            continue

        tracker.record_fill(
            symbol=fill.symbol,
            quantity=fill.quantity,
            price=fill.price,
            account=fill.account,
        )
        tracker.update_mark_price(fill.symbol, fill.price)

        processed.append(fill)
        if on_fill is not None:
            on_fill(fill, tracker)

    return processed


# ---------------------------------------------------------------------------
def _iter_sorted_records(records: Iterable[dict[str, Any]]) -> Iterator[dict[str, Any]]:
    return iter(sorted(records, key=_resolve_timestamp))


def _resolve_timestamp(record: dict[str, Any]) -> datetime:
    raw = record.get("event_timestamp")
    timestamp = _parse_timestamp(raw)
    if timestamp is not None:
        return timestamp

    snapshot = record.get("snapshot")
    if isinstance(snapshot, dict):
        ts = _parse_timestamp(snapshot.get("last_update"))
        if ts is not None:
            return ts

    return datetime.now(timezone.utc)


def _parse_timestamp(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, str) and value:
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError:
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed
    return None


def _extract_fill(
    record: dict[str, Any],
    prior_filled: dict[str, float],
) -> ProcessedFill | None:
    event_type = str(record.get("event_type", "")).lower()
    if event_type not in {"partial_fill", "filled"}:
        return None

    order_id = str(record.get("order_id") or "").strip()
    if not order_id:
        return None

    symbol = record.get("symbol")
    if symbol is None and isinstance(record.get("snapshot"), dict):
        symbol = record["snapshot"].get("symbol")
    if not symbol:
        return None

    side = str(
        record.get("side")
        or record.get("snapshot", {}).get("side")
        or "",
    ).upper()
    if side not in {"BUY", "SELL"}:
        return None

    account = record.get("account")
    if account is None and isinstance(record.get("snapshot"), dict):
        account = record["snapshot"].get("account")

    filled_quantity = _coerce_float(
        record.get("filled_quantity")
        or record.get("snapshot", {}).get("filled_quantity"),
    )
    prev_filled = prior_filled[order_id]
    delta = 0.0
    if filled_quantity is not None:
        delta = max(filled_quantity - prev_filled, 0.0)

    if delta <= 0:
        last_qty = _coerce_float(record.get("last_quantity"))
        if last_qty is not None:
            delta = max(last_qty, 0.0)

    if delta <= 0:
        return None

    price = _coerce_float(
        record.get("last_price")
        or record.get("average_fill_price")
        or record.get("snapshot", {}).get("average_fill_price"),
    )
    if price is None:
        return None

    signed_qty = delta if side == "BUY" else -delta
    prior_filled[order_id] = prev_filled + delta

    timestamp = _resolve_timestamp(record)

    return ProcessedFill(
        timestamp=timestamp,
        symbol=str(symbol),
        quantity=signed_qty,
        price=price,
        side=side,
        account=str(account) if account is not None else None,
        order_id=order_id,
    )


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

