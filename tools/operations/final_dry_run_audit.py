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
    assess_sign_off_readiness,
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
    parser.add_argument(
        "--sign-off",
        action="store_true",
        help=(
            "Evaluate default sign-off criteria (72h duration, uptime ratio ≥0.98, "
            "and require diary/performance evidence)."
        ),
    )
    parser.add_argument(
        "--sign-off-min-duration-hours",
        type=float,
        default=None,
        help="Override the minimum dry run duration (in hours) required for sign-off.",
    )
    parser.add_argument(
        "--sign-off-min-uptime-ratio",
        type=float,
        default=None,
        help="Override the minimum uptime ratio required for sign-off (0.0–1.0).",
    )
    parser.add_argument(
        "--sign-off-min-sharpe",
        type=float,
        default=None,
        help="Require the performance report to include a Sharpe ratio at or above this value.",
    )
    parser.add_argument(
        "--sign-off-allow-warnings",
        action="store_true",
        help="Allow WARN severities to pass sign-off instead of treating them as failures.",
    )
    parser.add_argument(
        "--sign-off-optional-diary",
        action="store_true",
        help="Do not require decision diary evidence for sign-off evaluation.",
    )
    parser.add_argument(
        "--sign-off-optional-performance",
        action="store_true",
        help="Do not require performance telemetry for sign-off evaluation.",
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

    sign_off_report = None
    sign_off_requested = (
        args.sign_off
        or args.sign_off_min_duration_hours is not None
        or args.sign_off_min_uptime_ratio is not None
        or args.sign_off_min_sharpe is not None
        or args.sign_off_allow_warnings
        or args.sign_off_optional_diary
        or args.sign_off_optional_performance
    )

    if sign_off_requested:
        min_duration = None
        if args.sign_off_min_duration_hours is not None:
            min_duration = timedelta(hours=args.sign_off_min_duration_hours)
        elif args.sign_off:
            min_duration = timedelta(hours=72)

        min_uptime = args.sign_off_min_uptime_ratio
        if min_uptime is None and args.sign_off:
            min_uptime = 0.98

        sign_off_report = assess_sign_off_readiness(
            summary,
            minimum_duration=min_duration,
            minimum_uptime_ratio=min_uptime,
            require_diary=not args.sign_off_optional_diary,
            require_performance=not args.sign_off_optional_performance,
            allow_warnings=args.sign_off_allow_warnings,
            minimum_sharpe_ratio=args.sign_off_min_sharpe,
        )

    if args.format == "json":
        payload = summary.as_dict()
        if sign_off_report is not None:
            payload["sign_off"] = sign_off_report.as_dict()
        text = json.dumps(payload, indent=2)
    else:
        text_parts = [summary.to_markdown().rstrip("\n")]
        if sign_off_report is not None:
            text_parts.extend(["", sign_off_report.to_markdown().rstrip("\n")])
        text = "\n".join(text_parts) + "\n"

    if args.output is not None:
        args.output.write_text(text if text.endswith("\n") else text + "\n", encoding="utf-8")
    else:
        print(text)

    exit_code = 0
    if summary.status is DryRunStatus.fail:
        exit_code = 1
    elif summary.status is DryRunStatus.warn and args.treat_warn_as_error:
        exit_code = 1

    if sign_off_report is not None:
        if sign_off_report.status is DryRunStatus.fail:
            exit_code = 1
        elif sign_off_report.status is DryRunStatus.warn and args.treat_warn_as_error:
            exit_code = 1

    return exit_code


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
