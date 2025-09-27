"""PnL and exposure dashboard derived from the order event journal."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from src.trading.order_management import PositionSnapshot, PositionTracker


def _load_event_records(path: Path) -> Iterable[dict[str, Any]]:
    if path.suffix == ".parquet":
        try:
            import pandas as pd  # type: ignore
        except Exception:  # pragma: no cover - optional dependency
            fallback = path.with_suffix(path.suffix + ".jsonl")
            if fallback.exists():
                path = fallback
            else:
                raise RuntimeError(
                    "Parquet support requires pandas/pyarrow; provide the JSONL fallback"
                )
        else:
            df = pd.read_parquet(path)  # type: ignore[attr-defined]
            return df.to_dict(orient="records")

    if path.suffix.endswith(".jsonl") or path.name.endswith(".jsonl"):
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)
        return

    raise RuntimeError(f"Unsupported journal format for {path}")


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


def _process_events(records: Iterable[dict[str, Any]], tracker: PositionTracker) -> None:
    prior_filled: dict[str, float] = defaultdict(float)

    for record in records:
        event_type = str(record.get("event_type", ""))
        if event_type not in {"partial_fill", "filled"}:
            continue

        order_id = str(record.get("order_id"))
        if not order_id:
            continue

        symbol = record.get("symbol") or record.get("snapshot", {}).get("symbol")
        if not symbol:
            continue

        side = str(record.get("side") or record.get("snapshot", {}).get("side") or "").upper()
        if side not in {"BUY", "SELL"}:
            continue

        account = record.get("account") or record.get("snapshot", {}).get("account")

        filled_quantity = float(record.get("filled_quantity") or record.get("snapshot", {}).get("filled_quantity") or 0.0)
        prev_filled = prior_filled[order_id]
        delta = max(filled_quantity - prev_filled, 0.0)
        last_qty = record.get("last_quantity")
        if delta <= 0 and last_qty is not None:
            try:
                delta = max(float(last_qty), 0.0)
            except (TypeError, ValueError):  # pragma: no cover - defensive
                delta = 0.0

        if delta <= 0:
            continue

        price = record.get("last_price")
        if price is None:
            price = record.get("average_fill_price") or record.get("snapshot", {}).get("average_fill_price")
        if price is None:
            continue

        signed_quantity = delta if side == "BUY" else -delta
        tracker.record_fill(symbol=str(symbol), quantity=signed_quantity, price=float(price), account=account)
        tracker.update_mark_price(str(symbol), float(price))
        prior_filled[order_id] = filled_quantity if filled_quantity > prev_filled else prev_filled + delta


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

    records = list(_load_event_records(path))
    tracker = PositionTracker()
    _process_events(records, tracker)

    snapshots = sorted(
        tracker.iter_positions(),
        key=lambda snap: (snap.account or "", snap.symbol),
    )

    print(_render_table(snapshots))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
