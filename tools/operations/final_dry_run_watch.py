#!/usr/bin/env python3
"""Monitor a final dry run progress file in real time."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Tuple

from src.operations.dry_run_audit import DryRunStatus
from src.operations.final_dry_run_progress import (
    DryRunProgressSnapshot,
    format_progress_snapshot,
    load_progress_snapshot,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Tail a progress snapshot JSON emitted by the final dry run harness "
            "and render a compact status report."
        )
    )
    parser.add_argument(
        "--progress",
        type=Path,
        required=True,
        help="Path to the progress JSON file produced by the harness.",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=30.0,
        help="Polling interval in seconds when watching for updates (default: 30).",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Render the current snapshot once and exit without watching.",
    )
    parser.add_argument(
        "--max-incidents",
        type=int,
        default=5,
        help="Maximum number of incidents to display when rendering snapshots.",
    )
    parser.add_argument(
        "--no-incidents",
        action="store_true",
        help="Suppress incident details in the rendered output.",
    )
    parser.add_argument(
        "--skip-summary",
        action="store_true",
        help="Do not include the embedded dry run summary status in the output.",
    )
    parser.add_argument(
        "--skip-sign-off",
        action="store_true",
        help="Do not include the sign-off status in the output.",
    )
    parser.add_argument(
        "--exit-when-complete",
        action="store_true",
        help="Exit once the snapshot reports PASS/WARN/FAIL (respecting warn handling).",
    )
    parser.add_argument(
        "--treat-warn-as-error",
        action="store_true",
        help="Exit with status 1 when the terminal state is WARN.",
    )
    return parser


def _attempt_load(path: Path) -> Tuple[DryRunProgressSnapshot | None, str | None]:
    try:
        snapshot = load_progress_snapshot(path)
        return snapshot, None
    except FileNotFoundError:
        return None, f"Progress file not found: {path.as_posix()}"
    except json.JSONDecodeError as exc:
        return None, f"Failed to parse progress file: {exc}"
    except ValueError as exc:
        return None, f"Invalid progress payload: {exc}"
    except Exception as exc:  # pragma: no cover - defensive guard
        return None, f"Unexpected error loading progress file: {exc}"


def _render_snapshot(
    snapshot: DryRunProgressSnapshot,
    *,
    max_incidents: int,
    include_incidents: bool,
    include_summary: bool,
    include_sign_off: bool,
) -> str:
    return format_progress_snapshot(
        snapshot,
        include_incidents=include_incidents,
        max_incidents=max_incidents,
        include_summary=include_summary,
        include_sign_off=include_sign_off,
    )


def _exit_code_for_status(
    status: DryRunStatus | None,
    *,
    treat_warn_as_error: bool,
) -> int:
    if status is DryRunStatus.fail:
        return 1
    if status is DryRunStatus.warn:
        return 1 if treat_warn_as_error else 2
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.interval <= 0 and not args.once:
        parser.error("--interval must be positive when watching for updates")
    if args.max_incidents <= 0:
        parser.error("--max-incidents must be positive")

    progress_path: Path = args.progress
    include_incidents = not args.no_incidents
    include_summary = not args.skip_summary
    include_sign_off = not args.skip_sign_off

    last_render: str | None = None
    last_error: str | None = None

    try:
        while True:
            snapshot, error = _attempt_load(progress_path)
            if snapshot is not None:
                output = _render_snapshot(
                    snapshot,
                    max_incidents=args.max_incidents,
                    include_incidents=include_incidents,
                    include_summary=include_summary,
                    include_sign_off=include_sign_off,
                )
                if output != last_render:
                    if last_render is not None:
                        print()
                    print(output, flush=True)
                    last_render = output
                    last_error = None
                if args.exit_when_complete and snapshot.is_terminal:
                    return _exit_code_for_status(
                        snapshot.status_severity,
                        treat_warn_as_error=args.treat_warn_as_error,
                    )
            else:
                if error and error != last_error:
                    print(error, file=sys.stderr, flush=True)
                    last_error = error
                if args.once:
                    return 1

            if args.once:
                return 0 if snapshot is not None else 1

            time.sleep(args.interval)
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
