#!/usr/bin/env python3
"""Compile a final dry run review brief for AlphaTrade."""

from __future__ import annotations

import argparse
import json
from datetime import timedelta
from pathlib import Path
from typing import Iterable, Sequence

from src.operations.dry_run_audit import (
    DEFAULT_FAIL_GAP,
    DEFAULT_WARN_GAP,
    DryRunStatus,
    assess_sign_off_readiness,
    evaluate_dry_run,
)
from src.operations.dry_run_packet import write_dry_run_packet
from src.operations.final_dry_run_review import build_review, parse_objective_spec


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a meeting-ready review brief from final dry run evidence."
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
        choices=("markdown", "json"),
        default="markdown",
        help="Output format for the review brief (default: markdown)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Write the review brief to the provided path instead of stdout",
    )
    parser.add_argument(
        "--run-label",
        help="Optional display name for the dry run (e.g. 'Paper Trail #7')",
    )
    parser.add_argument(
        "--attendee",
        action="append",
        dest="attendees",
        default=None,
        metavar="NAME",
        help="Add a reviewer or stakeholder to the attendee list.",
    )
    parser.add_argument(
        "--objective",
        action="append",
        dest="objectives",
        default=None,
        metavar="NAME=STATUS[:NOTE]",
        help=(
            "Record an acceptance objective outcome (pass/warn/fail). "
            "Example: --objective data-backbone=pass:Ingestion stable"
        ),
    )
    parser.add_argument(
        "--note",
        action="append",
        dest="notes",
        default=None,
        metavar="TEXT",
        help="Add a free-form note to the review brief.",
    )
    parser.add_argument(
        "--notes-file",
        action="append",
        dest="notes_files",
        default=None,
        metavar="PATH",
        help="Load additional notes from a text file (one item per line).",
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
        "--treat-warn-as-error",
        action="store_true",
        help="Exit with status 1 when the review status is WARN.",
    )
    parser.add_argument(
        "--include-summary",
        dest="include_summary",
        action="store_true",
        default=True,
        help="Include the full dry run summary in the output (default).",
    )
    parser.add_argument(
        "--skip-summary",
        dest="include_summary",
        action="store_false",
        help="Do not embed the dry run summary in the output.",
    )
    parser.add_argument(
        "--include-sign-off",
        dest="include_sign_off",
        action="store_true",
        default=True,
        help="Include the sign-off assessment in the output (default).",
    )
    parser.add_argument(
        "--skip-sign-off",
        dest="include_sign_off",
        action="store_false",
        help="Do not embed the sign-off assessment in the output.",
    )
    parser.add_argument(
        "--packet-dir",
        type=Path,
        help="Optional directory where an evidence packet should be written.",
    )
    parser.add_argument(
        "--packet-archive",
        type=Path,
        help="Optional path to a .tar.gz archive for the evidence packet.",
    )
    parser.add_argument(
        "--packet-skip-raw",
        dest="packet_include_raw",
        action="store_false",
        help="Do not copy raw artefacts into the evidence packet directory.",
    )
    parser.set_defaults(packet_include_raw=True)

    # Sign-off evaluation flags
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


def _load_notes(notes_files: Iterable[Path]) -> list[str]:
    collected: list[str] = []
    for path in notes_files:
        text = path.read_text(encoding="utf-8") if path.exists() else ""
        for line in text.splitlines():
            line = line.strip()
            if line:
                collected.append(line)
    return collected


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

    packet_paths = None
    if args.packet_dir is not None:
        packet_paths = write_dry_run_packet(
            summary=summary,
            output_dir=args.packet_dir,
            sign_off_report=sign_off_report,
            log_paths=log_paths,
            diary_path=args.diary,
            performance_path=args.performance,
            include_raw_artifacts=args.packet_include_raw,
            archive_path=args.packet_archive,
        )

    notes: list[str] = list(args.notes or [])
    notes_files = [Path(path) for path in (args.notes_files or [])]
    if notes_files:
        notes.extend(_load_notes(notes_files))

    attendees: Iterable[str] = args.attendees or []
    objective_specs: list[object] = []
    if args.objectives:
        for spec in args.objectives:
            try:
                objective_specs.append(parse_objective_spec(spec))
            except ValueError as exc:
                parser.error(str(exc))

    review = build_review(
        summary,
        sign_off_report,
        run_label=args.run_label,
        attendees=attendees,
        notes=notes,
        evidence_packet=packet_paths,
        objectives=tuple(objective_specs),
    )

    if args.format == "json":
        payload = review.as_dict()
        payload["summary"] = summary.as_dict()
        if sign_off_report is not None:
            payload["sign_off"] = sign_off_report.as_dict()
        if packet_paths is not None:
            payload["packet"] = packet_paths.as_dict()
        text = json.dumps(payload, indent=2)
    else:
        text = review.to_markdown(
            include_summary=args.include_summary,
            include_sign_off=args.include_sign_off,
        )

    if args.output is not None:
        args.output.write_text(text if text.endswith("\n") else text + "\n", encoding="utf-8")
    else:
        print(text)

    exit_code = 0
    if review.status is DryRunStatus.fail:
        exit_code = 1
    elif review.status is DryRunStatus.warn and args.treat_warn_as_error:
        exit_code = 1

    return exit_code


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
