"""Replay FIX execution logs and validate order lifecycle behaviour.

This CLI utility ingests execution events (JSON, JSONL, CSV, or raw FIX)
and replays them through the canonical :class:`OrderLifecycleProcessor`.
It mirrors the Phase 1 roadmap requirement for an end-to-end dry run tool
that asserts state transitions, records discrepancies, and summarises
resulting order and position state.  The implementation intentionally keeps
the heavy lifting in pure Python so it can be unit tested and embedded in
operational automation.
"""

from __future__ import annotations

import argparse
import csv
import json
import copy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import simplefix

from src.trading.order_management import (
    InMemoryOrderEventJournal,
    OrderEventJournal,
    OrderLifecycleProcessor,
    OrderMetadata,
    OrderStateError,
    PositionSnapshot,
    PositionTracker,
)


_EPOCH = datetime.min.replace(tzinfo=timezone.utc)


@dataclass(slots=True)
class OrderSeed:
    """Incrementally collected metadata required to register an order."""

    order_id: str
    symbol: str | None = None
    side: str | None = None
    quantity: float | None = None
    account: str | None = None
    created_at: datetime | None = None

    def merge(self, other: "OrderSeed") -> None:
        if other.symbol:
            self.symbol = other.symbol
        if other.side:
            self.side = other.side
        if other.quantity is not None:
            self.quantity = other.quantity
        if other.account:
            self.account = other.account
        if self.created_at is None and other.created_at is not None:
            self.created_at = other.created_at

    def to_metadata(self, *, default_account: str) -> OrderMetadata:
        missing: list[str] = []
        if not self.symbol:
            missing.append("symbol")
        if not self.side:
            missing.append("side")
        if self.quantity is None:
            missing.append("quantity")
        if missing:
            raise ValueError(
                f"Missing required order metadata for {self.order_id}: {', '.join(missing)}"
            )

        side = _normalise_side(self.side)
        if side is None:
            raise ValueError(f"Unsupported side value for {self.order_id}: {self.side!r}")

        created = self.created_at
        if created is not None and created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)

        return OrderMetadata(
            order_id=self.order_id,
            symbol=self.symbol,
            side=side,
            quantity=float(self.quantity),
            account=self.account or default_account,
            created_at=created,
        )


@dataclass(slots=True)
class ParsedEvent:
    """Representation of a normalised execution report ready for replay."""

    order_id: str
    payload: Dict[str, object]
    seed: OrderSeed
    source_index: int


@dataclass(slots=True)
class OrderSummary:
    order_id: str
    status: str
    filled_quantity: float
    remaining_quantity: float
    average_fill_price: float | None
    last_event: str | None
    created_at: str
    acknowledged_at: str | None
    final_fill_at: str | None
    cancelled_at: str | None
    rejected_at: str | None


@dataclass(slots=True)
class DryRunResult:
    """Container returned by :func:`run_dry_run` for reporting & testing."""

    events_processed: int
    order_summaries: List[OrderSummary]
    position_snapshots: List[Dict[str, Any]]
    errors: List[str]
    journal_path: Path | None


def run_dry_run(
    log_path: Path,
    *,
    journal_path: Path | None = None,
    summary_path: Path | None = None,
    pnl_mode: str = "fifo",
    default_account: str = "PRIMARY",
    fail_fast: bool = False,
    max_events: int | None = None,
    verbose: bool = False,
) -> DryRunResult:
    """Replay execution events from ``log_path`` and validate transitions."""

    events, parse_errors = _load_events(log_path, limit=max_events)
    if parse_errors and fail_fast:
        raise SystemExit("; ".join(parse_errors))

    journal: OrderEventJournal | InMemoryOrderEventJournal
    if journal_path is None:
        journal = InMemoryOrderEventJournal()
    else:
        journal = OrderEventJournal(journal_path)

    tracker = PositionTracker(pnl_mode=pnl_mode, default_account=default_account)
    processor = OrderLifecycleProcessor(journal=journal, position_tracker=tracker)

    seeds: Dict[str, OrderSeed] = {}
    errors: List[str] = list(parse_errors)
    events_processed = 0

    for event in events:
        seed = seeds.get(event.order_id)
        if seed is None:
            seed = OrderSeed(order_id=event.order_id)
            seeds[event.order_id] = seed
        seed.merge(event.seed)

        if not processor.state_machine.has_order(event.order_id):
            try:
                metadata = seeds[event.order_id].to_metadata(default_account=default_account)
            except ValueError as exc:
                message = f"order {event.order_id}: {exc}"
                errors.append(message)
                if fail_fast:
                    raise SystemExit(message)
                if verbose:
                    print(f"[WARN] {message}")
                continue
            processor.register_order(metadata)

        state_map = getattr(processor.state_machine, "_orders", None)
        previous_state = None
        if isinstance(state_map, dict) and event.order_id in state_map:
            previous_state = copy.deepcopy(state_map[event.order_id])

        try:
            processor.handle_broker_payload(event.order_id, event.payload)
        except OrderStateError as exc:
            message = f"order {event.order_id}: {exc}"
            if previous_state is not None and isinstance(state_map, dict):
                state_map[event.order_id] = previous_state
            errors.append(message)
            if fail_fast:
                raise SystemExit(message)
            if verbose:
                print(f"[WARN] {message}")
            continue

        events_processed += 1

    order_ids = sorted(seeds)
    order_summaries = [
        _snapshot_to_summary(processor.get_snapshot(order_id))
        for order_id in order_ids
        if processor.state_machine.has_order(order_id)
    ]

    position_snapshots = [
        _position_snapshot_to_dict(snapshot)
        for snapshot in tracker.iter_positions()
    ]

    result = DryRunResult(
        events_processed=events_processed,
        order_summaries=order_summaries,
        position_snapshots=position_snapshots,
        errors=errors,
        journal_path=journal_path,
    )

    if summary_path is not None:
        summary_payload = _result_to_dict(result)
        summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    if verbose:
        _print_summary(result)

    return result


def _print_summary(result: DryRunResult) -> None:
    print(f"Processed {result.events_processed} events across {len(result.order_summaries)} orders")
    for summary in result.order_summaries:
        print(
            "  {order_id}: {status} filled {filled:.4f}/{quantity:.4f}".format(
                order_id=summary.order_id,
                status=summary.status,
                filled=summary.filled_quantity,
                quantity=summary.filled_quantity + summary.remaining_quantity,
            )
        )
    if result.errors:
        print("Errors:")
        for error in result.errors:
            print(f"  - {error}")


def _load_events(path: Path, *, limit: int | None = None) -> tuple[list[ParsedEvent], list[str]]:
    suffix = path.suffix.lower()
    if suffix in {".json", ".jsonl"}:
        records = _load_json_records(path)
    elif suffix == ".csv":
        records = _load_csv_records(path)
    else:
        records = _load_fix_records(path)

    events: list[ParsedEvent] = []
    errors: list[str] = []

    for idx, record in enumerate(records):
        if limit is not None and len(events) >= limit:
            break

        try:
            parsed = _normalise_record(record, source_index=idx)
        except ValueError as exc:
            errors.append(str(exc))
            continue

        if parsed is None:
            continue

        events.append(parsed)

    events.sort(key=lambda evt: (evt.seed.created_at or _EPOCH, evt.source_index))
    return events, errors


def _load_json_records(path: Path) -> list[Mapping[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        records: list[Mapping[str, Any]] = []
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
        return records

    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [record for record in payload if isinstance(record, Mapping)]
    if isinstance(payload, Mapping):
        if "events" in payload and isinstance(payload["events"], list):
            return [record for record in payload["events"] if isinstance(record, Mapping)]
        return [payload]
    raise ValueError(f"Unsupported JSON structure in {path}")


def _load_csv_records(path: Path) -> list[Mapping[str, Any]]:
    records: list[Mapping[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            records.append(row)
    return records


def _load_fix_records(path: Path) -> list[Mapping[str, Any]]:
    parser = simplefix.FixParser()
    records: list[Mapping[str, Any]] = []

    with path.open("rb") as fh:
        for raw_line in fh:
            raw_line = raw_line.strip()
            if not raw_line or b"=" not in raw_line:
                continue

            buffer = raw_line.replace(b"|", b"\x01")
            if not buffer.endswith(b"\x01"):
                buffer += b"\x01"
            parser.append_buffer(buffer)

            while True:
                message = parser.get_message()
                if message is None:
                    break

                record: Dict[str, Any] = {}
                for tag, value in message:
                    decoded = value.decode(errors="ignore") if isinstance(value, (bytes, bytearray)) else str(value)
                    record[str(tag)] = decoded

                for tag, key in _FIX_TAG_MAPPING.items():
                    value = record.get(str(tag))
                    if value is not None:
                        record[key] = value

                records.append(record)

    return records


_FIX_TAG_MAPPING: Dict[int, str] = {
    11: "order_id",
    37: "order_id_secondary",
    38: "order_qty",
    55: "symbol",
    54: "side",
    32: "last_qty",
    31: "last_price",
    14: "cum_qty",
    151: "leaves_qty",
    1: "account",
    60: "timestamp",
    150: "exec_type",
}


_EXEC_TYPE_ALIASES: Dict[str, str] = {
    "0": "0",
    "NEW": "0",
    "ACK": "0",
    "ACKNOWLEDGED": "0",
    "1": "1",
    "PARTIAL_FILL": "1",
    "PARTIALLY_FILLED": "1",
    "2": "2",
    "FILL": "2",
    "FILLED": "2",
    "4": "4",
    "CANCEL": "4",
    "CANCELLED": "4",
    "CANCELED": "4",
    "8": "8",
    "REJECT": "8",
    "REJECTED": "8",
}


def _normalise_record(record: Mapping[str, Any], *, source_index: int) -> ParsedEvent | None:
    order_id = _coerce_str(
        record.get("order_id")
        or record.get("ClOrdID")
        or record.get("cl_ord_id")
        or record.get("11")
        or record.get("37")
    )
    if not order_id:
        return None

    exec_raw = _coerce_str(record.get("exec_type") or record.get("ExecType") or record.get("150"))
    if exec_raw is None:
        raise ValueError(f"order {order_id}: missing exec_type field")

    exec_type = _EXEC_TYPE_ALIASES.get(exec_raw.upper())
    if exec_type is None:
        raise ValueError(f"order {order_id}: unsupported exec_type {exec_raw!r}")

    last_qty = _coerce_float(record.get("last_qty") or record.get("LastQty") or record.get("32"))
    last_price = _coerce_float(record.get("last_price") or record.get("LastPx") or record.get("31"))
    cumulative = _coerce_float(record.get("cum_qty") or record.get("CumQty") or record.get("14"))
    leaves = _coerce_float(record.get("leaves_qty") or record.get("LeavesQty") or record.get("151"))
    timestamp = _parse_timestamp(record.get("timestamp") or record.get("TransactTime") or record.get("60"))

    payload: Dict[str, object] = {"exec_type": exec_type}
    if last_qty is not None:
        payload["last_qty"] = last_qty
    if last_price is not None:
        payload["last_px"] = last_price
    if cumulative is not None:
        payload["cum_qty"] = cumulative
    if leaves is not None:
        payload["leaves_qty"] = leaves
    if timestamp is not None:
        payload["timestamp"] = timestamp

    seed = OrderSeed(
        order_id=order_id,
        symbol=_coerce_str(record.get("symbol") or record.get("Symbol") or record.get("55")),
        side=_coerce_str(record.get("side") or record.get("Side") or record.get("54")),
        quantity=_coerce_float(record.get("order_qty") or record.get("OrderQty") or record.get("38")),
        account=_coerce_str(record.get("account") or record.get("Account") or record.get("1")),
        created_at=timestamp,
    )

    return ParsedEvent(order_id=order_id, payload=payload, seed=seed, source_index=source_index)


def _snapshot_to_summary(snapshot) -> OrderSummary:
    def _iso(value: datetime | None) -> str | None:
        if value is None:
            return None
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.isoformat()

    return OrderSummary(
        order_id=snapshot.order_id,
        status=snapshot.status.value,
        filled_quantity=snapshot.filled_quantity,
        remaining_quantity=snapshot.remaining_quantity,
        average_fill_price=snapshot.average_fill_price,
        last_event=snapshot.last_event,
        created_at=_iso(snapshot.created_at),
        acknowledged_at=_iso(snapshot.acknowledged_at),
        final_fill_at=_iso(snapshot.final_fill_at),
        cancelled_at=_iso(snapshot.cancelled_at),
        rejected_at=_iso(snapshot.rejected_at),
    )


def _position_snapshot_to_dict(snapshot: PositionSnapshot) -> Dict[str, Any]:
    return {
        "symbol": snapshot.symbol,
        "account": snapshot.account,
        "net_quantity": snapshot.net_quantity,
        "long_quantity": snapshot.long_quantity,
        "short_quantity": snapshot.short_quantity,
        "market_price": snapshot.market_price,
        "average_long_price": snapshot.average_long_price,
        "average_short_price": snapshot.average_short_price,
        "realized_pnl": snapshot.realized_pnl,
        "unrealized_pnl": snapshot.unrealized_pnl,
        "exposure": snapshot.exposure,
    }


def _result_to_dict(result: DryRunResult) -> Dict[str, Any]:
    return {
        "events_processed": result.events_processed,
        "orders": [asdict(summary) for summary in result.order_summaries],
        "positions": result.position_snapshots,
        "errors": result.errors,
        "journal_path": str(result.journal_path) if result.journal_path else None,
    }


def _coerce_str(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        return value or None
    return str(value)


def _coerce_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def _normalise_side(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().upper()
    mapping = {
        "1": "BUY",
        "BUY": "BUY",
        "B": "BUY",
        "LONG": "BUY",
        "2": "SELL",
        "SELL": "SELL",
        "S": "SELL",
        "SHORT": "SELL",
    }
    return mapping.get(text)


def _parse_timestamp(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(float(value), tz=timezone.utc)

    text = str(value).strip()
    if not text:
        return None

    iso_candidate = text.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(iso_candidate)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed
    except ValueError:
        pass

    for fmt in ("%Y%m%d-%H:%M:%S.%f", "%Y%m%d-%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(text, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path, help="Path to execution event log (FIX/JSON/CSV)")
    parser.add_argument("--journal", type=Path, help="Optional path to append a replayed journal")
    parser.add_argument(
        "--summary",
        type=Path,
        help="Optional path to write a JSON summary of replay results",
    )
    parser.add_argument(
        "--pnl-mode",
        choices=("fifo", "lifo"),
        default="fifo",
        help="Inventory accounting mode used for PnL computation",
    )
    parser.add_argument(
        "--account",
        default="PRIMARY",
        help="Default account identifier applied when events omit one",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Abort on the first parsing or lifecycle error",
    )
    parser.add_argument(
        "--max-events",
        type=int,
        help="Optional cap on the number of events to replay",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print a textual summary after processing",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    result = run_dry_run(
        args.log,
        journal_path=args.journal,
        summary_path=args.summary,
        pnl_mode=args.pnl_mode,
        default_account=args.account,
        fail_fast=args.fail_fast,
        max_events=args.max_events,
        verbose=args.verbose,
    )

    return 1 if result.errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
