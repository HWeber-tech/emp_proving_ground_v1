"""Utilities for replaying order journals and generating reconciliation reports."""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping

from .position_tracker import PositionTracker, ReconciliationReport

__all__ = [
    "load_order_journal_records",
    "replay_order_events",
    "load_broker_positions",
    "report_to_dict",
]


JournalRecord = Mapping[str, Any]


def load_order_journal_records(path: Path) -> List[Dict[str, Any]]:
    """Load order lifecycle records from the append-only journal.

    The journal primarily writes to Parquet but falls back to ``.jsonl`` when
    the optional dependencies are unavailable.  This helper mirrors that
    behaviour, preferring the Parquet payload when dependencies are present and
    otherwise reading the JSONL companion file.  Results are ordered by the
    event timestamp to guarantee deterministic playback.
    """

    path = Path(path)
    records: List[Dict[str, Any]] = []

    if path.exists() and path.suffix == ".parquet":
        try:
            import pandas as pd  # type: ignore

            df = pd.read_parquet(path)
            records.extend(df.to_dict(orient="records"))
        except Exception:  # pragma: no cover - optional dependency
            records.clear()

    if not records:
        json_path = path if path.suffix.endswith(".jsonl") else path.with_suffix(path.suffix + ".jsonl")
        if json_path.exists():
            with json_path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:  # pragma: no cover - log files can be messy
                        continue

    records.sort(key=_record_sort_key)
    return records


def replay_order_events(records: Iterable[JournalRecord], tracker: PositionTracker) -> None:
    """Replay journal records into a :class:`PositionTracker` instance."""

    cumulative_by_order: MutableMapping[str, float] = {}

    for record in records:
        event_type = str(record.get("event_type", "")).lower()
        symbol = str(record.get("symbol") or "").strip()
        if not symbol:
            continue

        side = str(record.get("side", "")).upper()
        account = record.get("account")
        price = _coerce_float(record.get("last_price"))
        if price is None:
            price = _coerce_float(record.get("average_fill_price"))

        if event_type in {"partial_fill", "filled"}:
            quantity = _coerce_float(record.get("last_quantity"))
            cumulative = _coerce_float(record.get("cumulative_quantity"))
            order_id = str(record.get("order_id") or symbol)
            if cumulative is not None:
                previous = cumulative_by_order.get(order_id, 0.0)
                quantity = max(cumulative - previous, quantity or 0.0)
                cumulative_by_order[order_id] = cumulative

            if quantity is None or quantity <= 0.0:
                continue
            if price is None:
                continue

            signed_quantity = quantity if side == "BUY" else -quantity
            tracker.record_fill(symbol, signed_quantity, price, account=account)
            tracker.update_mark_price(symbol, price)
        else:
            if price is not None:
                tracker.update_mark_price(symbol, price)


def load_broker_positions(path: Path) -> Dict[str, float]:
    """Load broker reported positions from JSON or CSV."""

    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in {".json", ".jsonl"}:
        data = json.loads(path.read_text(encoding="utf-8"))
        return _positions_from_json(data)

    if suffix == ".csv":
        return _positions_from_csv(path)

    raise ValueError(f"Unsupported broker statement format: {path.suffix}")


def report_to_dict(report: ReconciliationReport) -> Dict[str, Any]:
    """Serialise a :class:`ReconciliationReport` to primitive types."""

    return {
        "timestamp": report.timestamp.isoformat(),
        "account": report.account,
        "differences": [
            {
                "symbol": diff.symbol,
                "tracker_quantity": diff.tracker_quantity,
                "broker_quantity": diff.broker_quantity,
                "difference": diff.difference,
            }
            for diff in report.differences
        ],
    }


def _positions_from_json(data: Any) -> Dict[str, float]:
    positions: Dict[str, float] = {}
    if isinstance(data, dict):
        for symbol, quantity in data.items():
            positions[str(symbol)] = _coerce_float(quantity, default=0.0)
        return positions

    if isinstance(data, list):
        for entry in data:
            if not isinstance(entry, Mapping):
                continue
            symbol = entry.get("symbol")
            if symbol is None:
                continue
            quantity = _coerce_float(entry.get("quantity") or entry.get("qty"), default=0.0)
            positions[str(symbol)] = quantity
        return positions

    raise ValueError("JSON broker statement must be an object or list of mappings")


def _positions_from_csv(path: Path) -> Dict[str, float]:
    positions: Dict[str, float] = {}
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            symbol = row.get("symbol") or row.get("Symbol")
            if not symbol:
                continue
            quantity = row.get("quantity") or row.get("qty") or row.get("Quantity")
            positions[str(symbol)] = _coerce_float(quantity, default=0.0)
    return positions


def _record_sort_key(record: Mapping[str, Any]) -> tuple[datetime, str]:
    timestamp = _parse_timestamp(record.get("event_timestamp") or record.get("last_update"))
    order_id = str(record.get("order_id") or record.get("symbol") or "")
    return (timestamp, order_id)


def _parse_timestamp(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        candidate = value.replace("Z", "+00:00")
        try:
            return datetime.fromisoformat(candidate)
        except ValueError:
            return datetime.min
    return datetime.min


def _coerce_float(value: Any, *, default: float | None = None) -> float | None:
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return default
    return default

