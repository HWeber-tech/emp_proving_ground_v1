#!/usr/bin/env python3
"""Launch a short final dry run smoke workflow using the simulated runtime."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Mapping

from src.operations.dry_run_audit import DryRunStatus
from src.operations.final_dry_run_smoke import SmokeRunOptions, run_smoke_workflow


def _parse_key_value(text: str) -> tuple[str, str]:
    if "=" not in text:
        raise argparse.ArgumentTypeError("Expected KEY=VALUE format")
    key, value = text.split("=", 1)
    key = key.strip()
    if not key:
        raise argparse.ArgumentTypeError("Key cannot be empty")
    return key, value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the final dry run harness against the bundled simulated runtime "
            "to validate configuration, reporting, and evidence generation."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory where smoke artefacts should be written. Defaults to "
            "artifacts/final_dry_run_smoke/<timestamp>."
        ),
    )
    parser.add_argument(
        "--duration-seconds",
        type=float,
        default=60.0,
        help="How long the smoke run should execute (default: 60).",
    )
    parser.add_argument(
        "--tick-interval",
        type=float,
        default=2.0,
        help="Refresh interval (seconds) for log/diary/performance artefacts.",
    )
    parser.add_argument(
        "--progress-interval",
        type=float,
        default=5.0,
        help="Interval (seconds) between progress snapshots (default: 5).",
    )
    parser.add_argument(
        "--minimum-uptime",
        type=float,
        default=0.95,
        help="Minimum uptime ratio required for the smoke run to pass (default: 0.95).",
    )
    parser.add_argument(
        "--allow-warnings",
        action="store_true",
        help="Treat WARN severity outcomes as success (exit code 0).",
    )
    parser.add_argument(
        "--packet-dir",
        type=Path,
        help="Optional evidence packet directory to materialise after the run.",
    )
    parser.add_argument(
        "--packet-archive",
        type=Path,
        help="Optional .tar.gz archive path for the evidence packet.",
    )
    parser.add_argument(
        "--packet-skip-raw",
        dest="packet_include_raw",
        action="store_false",
        help="Skip copying raw artefacts into the packet directory.",
    )
    parser.set_defaults(packet_include_raw=True)
    parser.add_argument(
        "--review-markdown",
        type=Path,
        help="Optional path to write the meeting-ready review brief (Markdown).",
    )
    parser.add_argument(
        "--review-json",
        type=Path,
        help="Optional path to write the review brief as JSON.",
    )
    parser.add_argument(
        "--review-note",
        action="append",
        dest="review_notes",
        default=None,
        metavar="TEXT",
        help="Add a note to the generated review brief (can be repeated).",
    )
    parser.add_argument(
        "--metadata",
        action="append",
        default=None,
        metavar="KEY=VALUE",
        help="Attach additional metadata to the dry run summary.",
    )
    parser.add_argument(
        "--env",
        action="append",
        default=None,
        metavar="KEY=VALUE",
        help="Environment variables to forward to the simulated runtime.",
    )
    parser.add_argument(
        "--treat-warn-as-error",
        action="store_true",
        help="Exit with status 1 when the run status is WARN.",
    )
    return parser


def _collect_mapping(pairs: list[tuple[str, str]] | None) -> Mapping[str, str]:
    mapping: dict[str, str] = {}
    if not pairs:
        return mapping
    for key, value in pairs:
        mapping[key] = value
    return mapping


def _determine_output_dir(explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit
    timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    return Path("artifacts/final_dry_run_smoke") / timestamp


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.duration_seconds <= 0:
        parser.error("--duration-seconds must be positive")
    if args.tick_interval <= 0:
        parser.error("--tick-interval must be positive")
    if args.progress_interval <= 0:
        parser.error("--progress-interval must be positive")
    if not (0.0 < args.minimum_uptime <= 1.0):
        parser.error("--minimum-uptime must be between 0 and 1")

    metadata_pairs = [_parse_key_value(item) for item in args.metadata or []]
    env_pairs = [_parse_key_value(item) for item in args.env or []]

    output_dir = _determine_output_dir(args.output_dir)

    options = SmokeRunOptions(
        output_dir=output_dir,
        duration=timedelta(seconds=args.duration_seconds),
        tick_interval=timedelta(seconds=args.tick_interval),
        progress_interval=timedelta(seconds=args.progress_interval),
        minimum_uptime_ratio=args.minimum_uptime,
        allow_warnings=args.allow_warnings,
        metadata=_collect_mapping(metadata_pairs),
        environment=_collect_mapping(env_pairs),
        packet_dir=args.packet_dir,
        packet_archive=args.packet_archive,
        packet_include_raw=args.packet_include_raw,
        review_markdown=args.review_markdown,
        review_json=args.review_json,
        review_notes=tuple(args.review_notes or ()),
    )

    result = run_smoke_workflow(options)
    status = result.run_result.status

    log_path = result.run_result.log_path
    print(
        f"Smoke run completed with status {status.value.upper()} — logs: {log_path.as_posix()}",
        file=sys.stdout,
    )

    if result.review is not None and args.review_markdown:
        print(
            f"Review brief written to {args.review_markdown.as_posix()}",
            file=sys.stdout,
        )

    if status is DryRunStatus.fail:
        return 1
    if status is DryRunStatus.warn and (args.treat_warn_as_error and not args.allow_warnings):
        return 1
    if status is DryRunStatus.warn and not args.allow_warnings:
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    sys.exit(main())
