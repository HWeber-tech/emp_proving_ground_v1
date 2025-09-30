"""CLI for archiving microstructure datasets into tiered storage."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Iterable

from src.data_foundation.storage import MicrostructureTieredArchive, RetentionPolicy


def _iter_source_files(source: Path, pattern: str) -> Iterable[Path]:
    if source.is_file():
        yield source
        return
    if not source.exists():
        raise FileNotFoundError(source)
    yield from sorted(source.rglob(pattern))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", help="Dataset name used to partition the archive")
    parser.add_argument("source", type=Path, help="Path to a microstructure file or directory")
    parser.add_argument("--pattern", default="*.parquet", help="Glob pattern when archiving directories")
    parser.add_argument(
        "--hot-dir",
        type=Path,
        default=Path("artifacts/microstructure/hot"),
        help="Destination directory for the hot tier",
    )
    parser.add_argument(
        "--cold-dir",
        type=Path,
        default=Path("artifacts/microstructure/cold"),
        help="Destination directory for the cold tier",
    )
    parser.add_argument("--hot-days", type=int, default=7, help="Retention window (days) for the hot tier")
    parser.add_argument("--cold-days", type=int, default=90, help="Retention window (days) for the cold tier")
    parser.add_argument(
        "--as-of",
        default=None,
        help="Timestamp (ISO 8601) applied to all files; defaults to current UTC time",
    )
    parser.add_argument(
        "--enforce-retention",
        action="store_true",
        help="Run retention enforcement after archiving",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    try:
        retention = RetentionPolicy.from_days(hot_days=args.hot_days, cold_days=args.cold_days)
    except ValueError as exc:  # pragma: no cover - argparse handles messaging
        parser.error(str(exc))

    if args.as_of is not None:
        try:
            as_of = datetime.fromisoformat(args.as_of)
        except ValueError as exc:  # pragma: no cover - argparse handles messaging
            parser.error(f"invalid --as-of timestamp: {exc}")
    else:
        as_of = None

    archive = MicrostructureTieredArchive(args.hot_dir, args.cold_dir, retention_policy=retention)

    sources = list(_iter_source_files(args.source, args.pattern))
    if not sources:
        parser.error("no files found to archive")

    for file_path in sources:
        archive.archive_file(file_path, args.dataset, as_of=as_of)

    if args.enforce_retention:
        archive.enforce_retention()


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()

