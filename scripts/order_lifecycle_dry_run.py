#!/usr/bin/env python3
"""Replay FIX execution events against the order lifecycle processor."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

from src.trading.order_management import (
    OrderEventJournal,
    OrderLifecycleProcessor,
    OrderMetadata,
)


def _load_orders(path: Path) -> Iterable[OrderMetadata]:
    data = json.loads(path.read_text())
    for entry in data:
        yield OrderMetadata(
            order_id=str(entry["order_id"]),
            symbol=str(entry["symbol"]),
            side=str(entry["side"]).upper(),
            quantity=float(entry["quantity"]),
        )


def _load_events(path: Path) -> Iterable[tuple[str, dict]]:
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        order_id = str(payload.get("order_id"))
        if not order_id:
            raise ValueError("Event missing order_id")
        yield order_id, payload


def run_dry_run(order_file: Path, events_file: Path, journal_path: Path) -> int:
    processor = OrderLifecycleProcessor(journal=OrderEventJournal(journal_path))

    for metadata in _load_orders(order_file):
        processor.register_order(metadata)

    failures = 0
    for order_id, payload in _load_events(events_file):
        try:
            snapshot = processor.handle_broker_payload(order_id, payload)
        except Exception as exc:  # pragma: no cover - script level logging
            failures += 1
            print(f"[ERROR] {order_id}: {exc}", file=sys.stderr)
            continue

        print(
            f"{order_id}: {snapshot.status.value} "
            f"filled={snapshot.filled_quantity} remaining={snapshot.remaining_quantity}"
        )

    if failures:
        print(f"Encountered {failures} failures", file=sys.stderr)
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("orders", type=Path, help="JSON list of order definitions")
    parser.add_argument(
        "events", type=Path, help="Newline-delimited JSON events from FIX execution logs"
    )
    parser.add_argument(
        "--journal",
        type=Path,
        default=Path("data_foundation/events/order_events.parquet"),
        help="Target Parquet path for the append-only journal",
    )
    args = parser.parse_args(argv)

    return run_dry_run(args.orders, args.events, args.journal)


if __name__ == "__main__":
    raise SystemExit(main())
