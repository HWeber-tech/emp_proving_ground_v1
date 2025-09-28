"""PnL and exposure dashboard derived from the order event journal."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

from src.trading.order_management import PositionSnapshot, PositionTracker
from src.trading.order_management.journal_loader import (
    load_order_journal,
    replay_journal_into_tracker,
)


def _format_number(value: float | None, *, precision: int = 2) -> str:
    if value is None:
        return "-"
    return f"{value:,.{precision}f}"


def _render_table(snapshots: Iterable[PositionSnapshot]) -> str:
    headers = [
        ("Symbol", 12),
        ("Account", 12),
        ("Net Qty", 14),
        ("Long Qty", 14),
        ("Short Qty", 14),
        ("Avg Long", 12),
        ("Avg Short", 12),
        ("Realized", 14),
        ("Unrealized", 14),
        ("Exposure", 14),
    ]
    header_row = " ".join(title.ljust(width) for title, width in headers)
    divider = "-" * len(header_row)
    lines = [header_row, divider]

    totals: dict[str, float] = defaultdict(float)

    for snapshot in snapshots:
        totals["realized"] += snapshot.realized_pnl
        totals["unrealized"] += snapshot.unrealized_pnl or 0.0
        totals["exposure"] += snapshot.exposure or 0.0
        lines.append(
            " ".join(
                [
                    snapshot.symbol.ljust(12),
                    (snapshot.account or "DEFAULT").ljust(12),
                    _format_number(snapshot.net_quantity, precision=4).rjust(14),
                    _format_number(snapshot.long_quantity, precision=4).rjust(14),
                    _format_number(snapshot.short_quantity, precision=4).rjust(14),
                    _format_number(snapshot.average_long_price).rjust(12),
                    _format_number(snapshot.average_short_price).rjust(12),
                    _format_number(snapshot.realized_pnl).rjust(14),
                    _format_number(snapshot.unrealized_pnl).rjust(14),
                    _format_number(snapshot.exposure).rjust(14),
                ]
            )
        )

    lines.append(divider)
    lines.append(
        f"Totals{'':<20} Realized={_format_number(totals['realized'])} "
        f"Unrealized={_format_number(totals['unrealized'])} "
        f"Exposure={_format_number(totals['exposure'])}"
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "journal",
        type=Path,
        nargs="?",
        default=Path("data_foundation/events/order_events.parquet"),
        help="Path to the order event journal (Parquet or JSONL)",
    )
    args = parser.parse_args(argv)

    path = args.journal
    if not path.exists():
        jsonl = path.with_suffix(path.suffix + ".jsonl")
        if jsonl.exists():
            path = jsonl
        else:
            print(f"Journal path {path} does not exist", file=sys.stderr)
            return 1

    records = load_order_journal(path)
    tracker = PositionTracker()
    replay_journal_into_tracker(records, tracker)

    snapshots = sorted(
        tracker.iter_positions(),
        key=lambda snap: (snap.account or "", snap.symbol),
    )

    print(_render_table(snapshots))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
