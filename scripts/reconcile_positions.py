"""Replay the order event journal and compare against broker positions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

from src.trading.order_management import (
    PositionTracker,
    load_broker_positions,
    load_order_journal_records,
    replay_order_events,
    report_to_dict,
)


def run_reconciliation(
    *,
    journal_path: Path,
    broker_path: Path | None,
    account: str,
    pnl_mode: str,
    tolerance: float,
    output_path: Path | None,
    fail_on_discrepancy: bool,
) -> int:
    tracker = PositionTracker(pnl_mode=pnl_mode, default_account=account)
    records = load_order_journal_records(journal_path)
    if not records:
        print(f"No order events found in {journal_path}")
    else:
        replay_order_events(records, tracker)

    broker_positions = {}
    if broker_path is not None:
        broker_positions = load_broker_positions(broker_path)

    report = tracker.generate_reconciliation_report(
        broker_positions,
        account=account,
        tolerance=tolerance,
    )

    if report.differences:
        print("Discrepancies detected:")
        for diff in report.differences:
            print(
                f"  {diff.symbol}: tracker={diff.tracker_quantity:.6f} "
                f"broker={diff.broker_quantity:.6f} Δ={diff.difference:.6f}"
            )
    else:
        print("Tracker positions match broker balances within tolerance.")

    if output_path is not None:
        payload = report_to_dict(report)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote reconciliation report to {output_path}")

    if report.differences and fail_on_discrepancy:
        return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--journal",
        type=Path,
        default=Path("data_foundation/events/order_events.parquet"),
        help="Path to the order event journal (Parquet or JSONL)",
    )
    parser.add_argument(
        "--broker",
        type=Path,
        help="Path to broker statement (JSON/CSV) for nightly reconciliation",
    )
    parser.add_argument(
        "--account",
        default="PRIMARY",
        help="Account identifier to reconcile (defaults to tracker default)",
    )
    parser.add_argument(
        "--pnl-mode",
        choices=("fifo", "lifo"),
        default="fifo",
        help="Inventory accounting mode for the position tracker",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-6,
        help="Tolerance for quantity differences before flagging discrepancies",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path to write the reconciliation report as JSON",
    )
    parser.add_argument(
        "--fail-on-discrepancy",
        action="store_true",
        help="Exit with status 1 if discrepancies are detected",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    return run_reconciliation(
        journal_path=args.journal,
        broker_path=args.broker,
        account=args.account,
        pnl_mode=args.pnl_mode,
        tolerance=args.tolerance,
        output_path=args.output,
        fail_on_discrepancy=args.fail_on_discrepancy,
    )


if __name__ == "__main__":
    raise SystemExit(main())
