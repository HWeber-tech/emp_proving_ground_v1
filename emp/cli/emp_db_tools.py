"""Utility CLI for maintaining the experimentation findings database."""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from typing import Iterable, List

from emp.core import findings_memory


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EMP findings DB maintenance tools")
    parser.add_argument("--db-path", default=str(findings_memory.DEFAULT_DB_PATH))
    subparsers = parser.add_subparsers(dest="command", required=True)

    prune_parser = subparsers.add_parser("prune", help="Delete stale rows by stage and age")
    prune_parser.add_argument("--keep-days", type=int, default=90)
    prune_parser.add_argument(
        "--stages",
        default="idea,screened",
        help="Comma separated list of stages eligible for pruning",
    )

    subparsers.add_parser("vacuum", help="Run VACUUM to reclaim unused space")
    return parser.parse_args(list(argv) if argv is not None else None)


def _prune(conn, keep_days: int, stages: List[str]) -> int:
    cutoff = (datetime.now(timezone.utc) - timedelta(days=int(keep_days))).isoformat(timespec="seconds")
    stage_clause = ",".join("?" for _ in stages)
    sql = f"DELETE FROM findings WHERE stage IN ({stage_clause}) AND created_at < ?"
    with conn:
        cursor = conn.execute(sql, (*stages, cutoff))
    return int(cursor.rowcount or 0)


def _vacuum(conn) -> None:
    with conn:
        conn.execute("VACUUM")


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(argv)
    conn = findings_memory.connect(args.db_path)

    if args.command == "prune":
        stages = [stage.strip() for stage in str(args.stages).split(",") if stage.strip()]
        if not stages:
            raise ValueError("At least one stage must be provided for pruning")
        removed = _prune(conn, args.keep_days, stages)
        print(f"Pruned {removed} rows older than {args.keep_days} days")
    elif args.command == "vacuum":
        _vacuum(conn)
        print("VACUUM complete")
    else:  # pragma: no cover - defensive fallback
        raise ValueError(f"Unknown command '{args.command}'")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
