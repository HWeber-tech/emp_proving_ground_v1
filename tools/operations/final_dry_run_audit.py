#!/usr/bin/env python3
"""Generate a final dry run audit summary for AlphaTrade."""

from __future__ import annotations

import argparse
import json
from datetime import timedelta
from pathlib import Path
from typing import Sequence

from src.operations.dry_run_audit import (
    DEFAULT_FAIL_GAP,
    DEFAULT_WARN_GAP,
    DryRunStatus,
    evaluate_dry_run,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compile structured log, decision diary, and performance telemetry "
            "into a final dry run summary."
        )
    )
    parser.add_argument(
        "--log",
        dest="logs",
        action="append",
        required=True,
        metavar="PATH",
        help="Path to a structured JSON log file (can be supplied multiple times)",
    )
    parser.add_argument(
        "--diary",
        type=Path,
        help="Optional path to a decision diary JSON file",
    )
    parser.add_argument(
        "--performance",
        type=Path,
        help="Optional path to a strategy performance JSON report",
    )
    parser.add_argument(
        "--format",
        choices=("json", "markdown"),
        default="json",
        help="Output format for the final summary (default: json)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Write the summary to the provided path instead of stdout",
    )
    parser.add_argument(
        "--treat-warn-as-error",
        action="store_true",
        help="Exit with status 1 when the summary status is WARN",
    )
    parser.add_argument(
        "--warn-gap-minutes",
        type=float,
        default=None,
        help=(
            "Warn when consecutive log entries are separated by more than this many minutes "
            f"(default: {DEFAULT_WARN_GAP.total_seconds() / 60:g})."
        ),
    )
    parser.add_argument(
        "--fail-gap-minutes",
        type=float,
        default=None,
        help=(
            "Fail when consecutive log entries are separated by more than this many minutes "
            f"(default: {DEFAULT_FAIL_GAP.total_seconds() / 60:g})."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    log_paths = [Path(path) for path in args.logs]
    warn_gap = (
        timedelta(minutes=args.warn_gap_minutes)
        if args.warn_gap_minutes is not None
        else None
    )
    fail_gap = (
        timedelta(minutes=args.fail_gap_minutes)
        if args.fail_gap_minutes is not None
        else None
    )

    summary = evaluate_dry_run(
        log_paths=log_paths,
        diary_path=args.diary,
        performance_path=args.performance,
        log_gap_warn=warn_gap,
        log_gap_fail=fail_gap,
    )

    if args.format == "json":
        text = json.dumps(summary.as_dict(), indent=2)
    else:
        text = summary.to_markdown()

    if args.output is not None:
        args.output.write_text(text if text.endswith("\n") else text + "\n", encoding="utf-8")
    else:
        print(text)

    if summary.status is DryRunStatus.fail:
        return 1
    if summary.status is DryRunStatus.warn and args.treat_warn_as_error:
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
