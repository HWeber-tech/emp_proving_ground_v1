#!/usr/bin/env python3
"""CLI wrapper for executing the AlphaTrade final dry run harness."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import timedelta
from pathlib import Path
from typing import Sequence

from src.operations.dry_run_audit import DryRunStatus
from src.operations.final_dry_run import (
    FinalDryRunConfig,
    run_final_dry_run,
)


def _parse_key_value(text: str) -> tuple[str, str]:
    if "=" not in text:
        raise argparse.ArgumentTypeError(
            "Expected KEY=VALUE format (received %r)" % (text,)
        )
    key, value = text.split("=", 1)
    key = key.strip()
    if not key:
        raise argparse.ArgumentTypeError("Key cannot be empty")
    return key, value


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the AlphaTrade runtime for the final dry run window and compile "
            "a sign-off readiness summary."
        ),
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        required=True,
        help="Directory where harness logs (raw + JSONL) and reports will be stored.",
    )
    parser.add_argument(
        "--duration-hours",
        type=float,
        default=72.0,
        help="Target runtime duration in hours (default: 72).",
    )
    parser.add_argument(
        "--required-duration-hours",
        type=float,
        default=None,
        help=(
            "Minimum duration (hours) required for sign-off; defaults to the target "
            "duration when omitted."
        ),
    )
    parser.add_argument(
        "--minimum-uptime-ratio",
        type=float,
        default=0.98,
        help=(
            "Minimum acceptable uptime ratio (0-1 range) before sign-off fails "
            "(default: 0.98)."
        ),
    )
    parser.add_argument(
        "--shutdown-grace-seconds",
        type=float,
        default=60.0,
        help="Grace period (seconds) allowed for graceful shutdown after timeout.",
    )
    parser.add_argument(
        "--diary",
        type=Path,
        help="Optional path to the decision diary JSONL backing store for the run.",
    )
    parser.add_argument(
        "--performance",
        type=Path,
        help="Optional path to a performance telemetry JSON report.",
    )
    parser.add_argument(
        "--allow-warnings",
        action="store_true",
        help="Allow WARN severities to pass sign-off (otherwise treated as failure).",
    )
    parser.add_argument(
        "--no-diary-required",
        action="store_true",
        help="Do not require decision diary evidence for sign-off evaluation.",
    )
    parser.add_argument(
        "--no-performance-required",
        action="store_true",
        help="Do not require performance telemetry for sign-off evaluation.",
    )
    parser.add_argument(
        "--minimum-sharpe-ratio",
        type=float,
        default=None,
        help="Minimum Sharpe ratio required for sign-off (optional).",
    )
    parser.add_argument(
        "--metadata",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Additional metadata entries to attach to the summary payload.",
    )
    parser.add_argument(
        "--env",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Environment variables to inject when launching the runtime.",
    )
    parser.add_argument(
        "--json-report",
        type=Path,
        help="Optional path to write the final summary bundle as JSON.",
    )
    parser.add_argument(
        "--markdown-report",
        type=Path,
        help="Optional path to write the final summary bundle as Markdown.",
    )
    parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help=(
            "Command to execute as the runtime. Specify after '--', e.g. "
            "final_dry_run.py --log-dir logs -- python3 main.py"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        parser.error("No runtime command supplied; provide it after '--'.")

    metadata_pairs = dict(_parse_key_value(item) for item in args.metadata)
    env_pairs = dict(_parse_key_value(item) for item in args.env)

    duration = timedelta(hours=args.duration_hours)
    required_duration = (
        timedelta(hours=args.required_duration_hours)
        if args.required_duration_hours is not None
        else None
    )
    shutdown_grace = timedelta(seconds=max(args.shutdown_grace_seconds, 0.0))

    config = FinalDryRunConfig(
        command=command,
        duration=duration,
        log_directory=args.log_dir,
        diary_path=args.diary,
        performance_path=args.performance,
        minimum_uptime_ratio=args.minimum_uptime_ratio,
        required_duration=required_duration,
        shutdown_grace=shutdown_grace,
        require_diary_evidence=not args.no_diary_required,
        require_performance_evidence=not args.no_performance_required,
        allow_warnings=args.allow_warnings,
        minimum_sharpe_ratio=args.minimum_sharpe_ratio,
        metadata=metadata_pairs,
        environment=env_pairs or None,
    )

    result = run_final_dry_run(config)

    print(f"Final dry run status: {result.status.value.upper()}")
    print(f"  Runtime command: {' '.join(result.config.command)}")
    print(f"  Exit code: {result.exit_code}")
    print(f"  Duration: {result.duration}")
    print(f"  Logs: {result.log_path}")
    print(f"  Raw logs: {result.raw_log_path}")
    if result.incidents:
        print("  Harness incidents:")
        for incident in result.incidents:
            print(
                f"    - {incident.occurred_at.astimezone().isoformat()} — "
                f"{incident.severity.value.upper()}: {incident.message}"
            )
    print(f"  Log summary status: {result.summary.log_summary.status.value if result.summary.log_summary else 'n/a'}")
    if result.sign_off is not None:
        print(f"  Sign-off status: {result.sign_off.status.value.upper()}")

    if args.json_report is not None:
        payload = {
            "status": result.status.value,
            "summary": result.summary.as_dict(),
        }
        if result.sign_off is not None:
            payload["sign_off"] = result.sign_off.as_dict()
        if result.incidents:
            payload["incidents"] = [incident.as_dict() for incident in result.incidents]
        args.json_report.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if args.markdown_report is not None:
        parts = [result.summary.to_markdown()]
        if result.sign_off is not None:
            parts.append(result.sign_off.to_markdown())
        args.markdown_report.write_text("\n".join(parts), encoding="utf-8")

    if result.status is DryRunStatus.fail:
        return 2
    if result.status is DryRunStatus.warn:
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
