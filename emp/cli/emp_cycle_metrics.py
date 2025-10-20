"""CLI for reporting experimentation cycle time-to-candidate KPIs."""
from __future__ import annotations

import argparse
import sys
from typing import Iterable, Optional

from emp.core import findings_memory


def _format_hours(value: Optional[float]) -> str:
    if value is None:
        return "-"
    return f"{value:.2f}h"


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Report idea-to-candidate turnaround KPIs")
    parser.add_argument(
        "--db-path",
        default=str(findings_memory.DEFAULT_DB_PATH),
        help="Path to the experimentation findings SQLite database",
    )
    parser.add_argument(
        "--threshold-hours",
        type=float,
        default=24.0,
        help="SLA threshold in hours for time-to-candidate",
    )
    parser.add_argument(
        "--window-hours",
        type=float,
        default=None,
        help="Optional trailing window (hours) to scope the analysis",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    conn = findings_memory.connect(args.db_path)
    stats = findings_memory.time_to_candidate_stats(
        conn,
        threshold_hours=float(args.threshold_hours),
        window_hours=args.window_hours,
    )

    print(f"Findings evaluated: {stats.count}")
    if not stats.count:
        print("No candidates have completed replay scoring yet.")
        return 0

    print(f"Average turnaround: {_format_hours(stats.average_hours)}")
    print(f"Median turnaround: {_format_hours(stats.median_hours)}")
    print(f"P90 turnaround: {_format_hours(stats.p90_hours)}")
    print(f"Max turnaround: {_format_hours(stats.max_hours)}")

    status = "PASS" if stats.sla_met else "FAIL"
    print(f"SLA (<= {args.threshold_hours:.2f}h): {status}")

    if stats.breaches:
        print("Breaches (id, stage, hours, created_at, completed_at):")
        for breach in stats.breaches:
            print(
                f"  {breach.id} | {breach.stage} | "
                f"{breach.hours:.2f}h | {breach.created_at} | {breach.completed_at}"
            )

    return 0 if stats.sla_met else 1


if __name__ == "__main__":
    sys.exit(main())
