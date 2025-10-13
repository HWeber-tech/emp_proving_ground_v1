#!/usr/bin/env python3
"""CLI wrapper for executing the AlphaTrade final dry run harness."""

from __future__ import annotations

import argparse
import json
from datetime import timedelta
from pathlib import Path
from typing import Sequence

from src.operations.dry_run_audit import DryRunStatus
from src.operations.final_dry_run import FinalDryRunConfig
from src.operations.final_dry_run_review import parse_objective_spec
from src.operations.final_dry_run_workflow import run_final_dry_run_workflow


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


def _load_notes(paths: Sequence[Path]) -> list[str]:
    notes: list[str] = []
    for path in paths:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        for line in text.splitlines():
            line = line.strip()
            if line:
                notes.append(line)
    return notes


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
        "--progress-path",
        type=Path,
        help="Write periodic progress updates to this JSON file.",
    )
    parser.add_argument(
        "--progress-interval-minutes",
        type=float,
        default=5.0,
        help=(
            "Interval in minutes between progress snapshots (set to 0 to disable)."
        ),
    )
    parser.add_argument(
        "--progress-timeline-dir",
        type=Path,
        help="Directory for immutable progress timeline snapshots (default: disabled).",
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
        "--log-rotate-hours",
        type=float,
        default=None,
        help="Rotate structured/raw logs after this many hours (default: disabled).",
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
        "--warn-gap-minutes",
        type=float,
        default=None,
        help=(
            "Warn when structured logs contain gaps longer than this many minutes "
            "during the post-run audit (default: 60 when omitted)."
        ),
    )
    parser.add_argument(
        "--fail-gap-minutes",
        type=float,
        default=None,
        help=(
            "Fail when structured logs contain gaps longer than this many minutes "
            "during the post-run audit (default: 360 when omitted)."
        ),
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
        "--packet-dir",
        type=Path,
        help="Write an evidence packet (summaries + artefacts) to this directory.",
    )
    parser.add_argument(
        "--packet-archive",
        type=Path,
        help="Optional path to a .tar.gz archive of the evidence packet.",
    )
    parser.add_argument(
        "--packet-skip-raw",
        action="store_true",
        help="Skip copying raw logs/diary/performance artefacts into the packet.",
    )
    parser.add_argument(
        "--no-log-monitor",
        action="store_true",
        help=(
            "Disable real-time log level monitoring during the run (post-run audit still "
            "captures issues)."
        ),
    )
    parser.add_argument(
        "--diary-stale-warn-minutes",
        type=float,
        default=None,
        help=(
            "Warn when the decision diary is missing updates for more than this many minutes during"
            " the run (requires --diary)."
        ),
    )
    parser.add_argument(
        "--diary-stale-fail-minutes",
        type=float,
        default=None,
        help=(
            "Fail when the decision diary is missing updates for more than this many minutes during"
            " the run (requires --diary)."
        ),
    )
    parser.add_argument(
        "--performance-stale-warn-minutes",
        type=float,
        default=None,
        help=(
            "Warn when the performance telemetry file is missing updates for more than this many"
            " minutes during the run (requires --performance)."
        ),
    )
    parser.add_argument(
        "--performance-stale-fail-minutes",
        type=float,
        default=None,
        help=(
            "Fail when the performance telemetry file is missing updates for more than this many"
            " minutes during the run (requires --performance)."
        ),
    )
    parser.add_argument(
        "--evidence-check-interval-minutes",
        type=float,
        default=None,
        help=(
            "Override the polling interval (minutes) for evidence freshness monitors."
        ),
    )
    parser.add_argument(
        "--evidence-initial-grace-minutes",
        type=float,
        default=15.0,
        help=(
            "Grace period (minutes) after startup before evidence freshness is evaluated."
        ),
    )
    parser.add_argument(
        "--live-gap-alert-minutes",
        type=float,
        default=None,
        help=(
            "Emit a harness incident when no runtime logs are observed for this many minutes."
        ),
    )
    parser.add_argument(
        "--live-gap-alert-severity",
        choices=(DryRunStatus.warn.value, DryRunStatus.fail.value),
        default=DryRunStatus.warn.value,
        help="Severity assigned to live log gap incidents (default: warn).",
    )
    parser.add_argument(
        "--compress-logs",
        action="store_true",
        help="Compress structured and raw log outputs (writes .gz files).",
    )
    parser.add_argument(
        "--resource-sample-interval-minutes",
        type=float,
        default=1.0,
        help=(
            "Interval in minutes between resource usage samples (set to 0 to "
            "disable monitoring)."
        ),
    )
    parser.add_argument(
        "--resource-max-cpu-percent",
        type=float,
        default=None,
        help="Trigger a resource incident when CPU utilisation exceeds this percent.",
    )
    parser.add_argument(
        "--resource-max-memory-mb",
        type=float,
        default=None,
        help="Trigger a resource incident when RSS memory exceeds this many MiB.",
    )
    parser.add_argument(
        "--resource-max-memory-percent",
        type=float,
        default=None,
        help=(
            "Trigger a resource incident when memory percentage exceeds this threshold."
        ),
    )
    parser.add_argument(
        "--resource-violation-severity",
        choices=(DryRunStatus.warn.value, DryRunStatus.fail.value),
        default=DryRunStatus.fail.value,
        help="Severity recorded when resource thresholds are breached.",
    )
    parser.add_argument(
        "--no-resource-monitor",
        action="store_true",
        help="Disable resource usage monitoring during the run.",
    )
    parser.add_argument(
        "--review-output",
        type=Path,
        help="Write the review brief to this path (use '-' for stdout).",
    )
    parser.add_argument(
        "--review-format",
        choices=("markdown", "json"),
        default="markdown",
        help="Format for the review brief output (default: markdown).",
    )
    parser.add_argument(
        "--review-run-label",
        help="Optional display name for the dry run in the review brief.",
    )
    parser.add_argument(
        "--attendee",
        action="append",
        dest="attendees",
        default=None,
        metavar="NAME",
        help="Add an attendee to the review brief (can be supplied multiple times).",
    )
    parser.add_argument(
        "--objective",
        action="append",
        dest="objectives",
        default=None,
        metavar="NAME=STATUS[:NOTE]",
        help=(
            "Record an acceptance objective outcome for the review (pass/warn/fail). "
            "Example: --objective data-backbone=pass:Timescale ingestion live"
        ),
    )
    parser.add_argument(
        "--note",
        action="append",
        dest="notes",
        default=None,
        metavar="TEXT",
        help="Attach a free-form note to the review brief (can be supplied multiple times).",
    )
    parser.add_argument(
        "--notes-file",
        action="append",
        dest="notes_files",
        default=None,
        metavar="PATH",
        help="Load notes from a text file (one entry per line).",
    )
    parser.add_argument(
        "--no-review",
        action="store_true",
        help="Skip generating the review brief.",
    )
    parser.add_argument(
        "--review-include-summary",
        dest="review_include_summary",
        action="store_true",
        default=True,
        help="Include the full dry run summary in the review brief (default).",
    )
    parser.add_argument(
        "--review-skip-summary",
        dest="review_include_summary",
        action="store_false",
        help="Do not embed the dry run summary in the review brief.",
    )
    parser.add_argument(
        "--review-include-sign-off",
        dest="review_include_sign_off",
        action="store_true",
        default=True,
        help="Include the sign-off assessment in the review brief (default).",
    )
    parser.add_argument(
        "--review-skip-sign-off",
        dest="review_include_sign_off",
        action="store_false",
        help="Do not embed the sign-off assessment in the review brief.",
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

    if args.packet_archive is not None and args.packet_dir is None:
        parser.error("--packet-archive requires --packet-dir")
    if args.packet_skip_raw and args.packet_dir is None:
        parser.error("--packet-skip-raw requires --packet-dir")
    if args.no_review and args.review_output is not None:
        parser.error("--no-review cannot be combined with --review-output")

    metadata_pairs = dict(_parse_key_value(item) for item in args.metadata)
    env_pairs = dict(_parse_key_value(item) for item in args.env)

    attendees = tuple(args.attendees or ())
    notes_list = list(args.notes or ())
    notes_files = [Path(path) for path in (args.notes_files or ())]
    notes_list.extend(_load_notes(notes_files))
    review_notes = tuple(notes_list)

    objective_specs: tuple[object, ...] = tuple()
    if args.objectives:
        parsed_objectives = []
        for spec in args.objectives:
            try:
                parsed_objectives.append(parse_objective_spec(spec))
            except ValueError as exc:
                parser.error(str(exc))
        objective_specs = tuple(parsed_objectives)

    duration = timedelta(hours=args.duration_hours)
    required_duration = (
        timedelta(hours=args.required_duration_hours)
        if args.required_duration_hours is not None
        else None
    )
    shutdown_grace = timedelta(seconds=max(args.shutdown_grace_seconds, 0.0))
    progress_interval = (
        timedelta(minutes=args.progress_interval_minutes)
        if args.progress_interval_minutes > 0
        else None
    )
    progress_timeline = (
        args.progress_timeline_dir if progress_interval is not None else None
    )
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
    live_gap_alert = (
        timedelta(minutes=args.live_gap_alert_minutes)
        if args.live_gap_alert_minutes is not None
        else None
    )
    live_gap_severity = DryRunStatus(args.live_gap_alert_severity)
    diary_stale_warn = (
        timedelta(minutes=args.diary_stale_warn_minutes)
        if args.diary_stale_warn_minutes is not None
        else None
    )
    diary_stale_fail = (
        timedelta(minutes=args.diary_stale_fail_minutes)
        if args.diary_stale_fail_minutes is not None
        else None
    )
    performance_stale_warn = (
        timedelta(minutes=args.performance_stale_warn_minutes)
        if args.performance_stale_warn_minutes is not None
        else None
    )
    performance_stale_fail = (
        timedelta(minutes=args.performance_stale_fail_minutes)
        if args.performance_stale_fail_minutes is not None
        else None
    )
    evidence_check_interval = (
        timedelta(minutes=args.evidence_check_interval_minutes)
        if args.evidence_check_interval_minutes is not None
        else None
    )
    evidence_initial_grace = timedelta(
        minutes=max(args.evidence_initial_grace_minutes, 0.0)
    )
    log_rotate_interval = (
        timedelta(hours=args.log_rotate_hours)
        if args.log_rotate_hours is not None and args.log_rotate_hours > 0.0
        else None
    )

    resource_sample_interval = None
    if not args.no_resource_monitor and args.resource_sample_interval_minutes > 0:
        resource_sample_interval = timedelta(
            minutes=args.resource_sample_interval_minutes
        )
    resource_severity = DryRunStatus(args.resource_violation_severity)

    config = FinalDryRunConfig(
        command=command,
        duration=duration,
        log_directory=args.log_dir,
        progress_path=args.progress_path,
        progress_interval=progress_interval,
        progress_timeline_dir=progress_timeline,
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
        monitor_log_levels=not args.no_log_monitor,
        log_gap_warn=warn_gap,
        log_gap_fail=fail_gap,
        live_gap_alert=live_gap_alert,
        live_gap_severity=live_gap_severity,
        diary_stale_warn=diary_stale_warn,
        diary_stale_fail=diary_stale_fail,
        performance_stale_warn=performance_stale_warn,
        performance_stale_fail=performance_stale_fail,
        evidence_check_interval=evidence_check_interval,
        evidence_initial_grace=evidence_initial_grace,
        compress_logs=args.compress_logs,
        log_rotate_interval=log_rotate_interval,
        resource_sample_interval=resource_sample_interval,
        resource_max_cpu_percent=args.resource_max_cpu_percent,
        resource_max_memory_mb=args.resource_max_memory_mb,
        resource_max_memory_percent=args.resource_max_memory_percent,
        resource_violation_severity=resource_severity,
    )

    workflow = run_final_dry_run_workflow(
        config,
        evidence_dir=args.packet_dir,
        evidence_archive=args.packet_archive,
        include_raw_artifacts=not args.packet_skip_raw,
        review_run_label=args.review_run_label,
        review_attendees=attendees,
        review_notes=review_notes,
        review_objectives=objective_specs,
        create_review=not args.no_review,
    )
    result = workflow.run_result

    print(f"Final dry run status: {result.status.value.upper()}")
    print(f"  Runtime command: {' '.join(result.config.command)}")
    print(f"  Exit code: {result.exit_code}")
    print(f"  Duration: {result.duration}")
    log_display = str(result.log_path)
    if len(result.log_paths) > 1:
        log_display += f" (+{len(result.log_paths) - 1} rotated)"
    print(f"  Logs: {log_display}")
    raw_display = str(result.raw_log_path)
    if len(result.raw_log_paths) > 1:
        raw_display += f" (+{len(result.raw_log_paths) - 1} rotated)"
    print(f"  Raw logs: {raw_display}")
    if result.progress_path is not None:
        print(f"  Progress: {result.progress_path}")
    if result.progress_timeline_dir is not None:
        print(f"  Progress timeline: {result.progress_timeline_dir}")
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
    if workflow.evidence_packet is not None:
        print(f"  Evidence packet directory: {workflow.evidence_packet.output_dir}")
        if workflow.evidence_packet.archive_path is not None:
            print(f"  Evidence packet archive: {workflow.evidence_packet.archive_path}")
    if workflow.review is not None:
        print(f"  Review status: {workflow.review.status.value.upper()}")
        if args.review_output is not None:
            review_text = (
                json.dumps(workflow.review.as_dict(), indent=2)
                if args.review_format == "json"
                else workflow.review.to_markdown(
                    include_summary=args.review_include_summary,
                    include_sign_off=args.review_include_sign_off,
                )
            )
            if args.review_output.as_posix() == "-":
                print(review_text)
            else:
                args.review_output.write_text(review_text, encoding="utf-8")
                print(f"  Review written to: {args.review_output}")
        else:
            print("  Review brief not written (pass --review-output or '--review-output -' to emit).")

    if args.json_report is not None:
        payload = {
            "status": result.status.value,
            "summary": result.summary.as_dict(),
        }
        if result.progress_path is not None:
            payload["progress_path"] = str(result.progress_path)
        if result.progress_timeline_dir is not None:
            payload["progress_timeline_dir"] = str(result.progress_timeline_dir)
        if result.sign_off is not None:
            payload["sign_off"] = result.sign_off.as_dict()
        if result.incidents:
            payload["incidents"] = [incident.as_dict() for incident in result.incidents]
        if workflow.review is not None:
            payload["review"] = workflow.review.as_dict()
        args.json_report.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if args.markdown_report is not None:
        parts = [result.summary.to_markdown()]
        if result.sign_off is not None:
            parts.append(result.sign_off.to_markdown())
        if workflow.review is not None:
            parts.append(
                workflow.review.to_markdown(
                    include_summary=args.review_include_summary,
                    include_sign_off=args.review_include_sign_off,
                )
            )
        args.markdown_report.write_text("\n".join(parts), encoding="utf-8")

    if result.status is DryRunStatus.fail:
        return 2
    if result.status is DryRunStatus.warn:
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
