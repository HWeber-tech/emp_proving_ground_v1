#!/usr/bin/env python3
"""Turnkey runner for the AlphaTrade final dry run workflow."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Iterable, MutableMapping, Sequence

from src.operations.dry_run_audit import DryRunStatus
from src.operations.final_dry_run import FinalDryRunConfig
from src.operations.final_dry_run_review import parse_objective_spec
from src.operations.final_dry_run_workflow import run_final_dry_run_workflow


_DEFAULT_ROOT = Path("artifacts/final_dry_run")
_DIARY_ENV = "FINAL_DRY_RUN_DIARY_PATH"
_PERFORMANCE_ENV = "FINAL_DRY_RUN_PERFORMANCE_PATH"
_LOG_DIR_ENV = "FINAL_DRY_RUN_LOG_DIR"


def _parse_pairs(label: str, values: Sequence[str]) -> dict[str, str]:
    pairs: dict[str, str] = {}
    for item in values:
        if "=" not in item:
            raise argparse.ArgumentTypeError(
                f"{label} expects KEY=VALUE entries (received '{item}')"
            )
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise argparse.ArgumentTypeError(f"{label} key cannot be empty")
        pairs[key] = value
    return pairs


def _load_notes(paths: Iterable[Path]) -> list[str]:
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
            "Launch the AlphaTrade runtime under the final dry run harness, "
            "capture evidence, and materialise review artefacts in a timestamped directory."
        )
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=_DEFAULT_ROOT,
        help=f"Base directory for dry run artefacts (default: {_DEFAULT_ROOT}).",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        help="Optional directory name under --output-root (default: timestamp slug).",
    )
    parser.add_argument(
        "--run-label",
        type=str,
        help="Human-friendly label recorded in summaries and review briefs.",
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
        help="Minimum runtime duration (hours) required for sign-off.",
    )
    parser.add_argument(
        "--minimum-uptime-ratio",
        type=float,
        default=0.98,
        help="Minimum acceptable uptime ratio (0-1 range).",
    )
    parser.add_argument(
        "--allow-warnings",
        action="store_true",
        help="Allow WARN severities to pass sign-off checks.",
    )
    parser.add_argument(
        "--minimum-sharpe-ratio",
        type=float,
        help="Minimum Sharpe ratio required for sign-off (optional).",
    )
    parser.add_argument(
        "--progress-interval-minutes",
        type=float,
        default=5.0,
        help="Interval in minutes between progress snapshots (0 disables snapshots).",
    )
    parser.add_argument(
        "--shutdown-grace-seconds",
        type=float,
        default=60.0,
        help="Grace period in seconds before force-terminating the runtime (default: 60).",
    )
    parser.add_argument(
        "--warn-gap-minutes",
        type=float,
        help="Warn when structured logs contain gaps longer than this many minutes.",
    )
    parser.add_argument(
        "--fail-gap-minutes",
        type=float,
        help="Fail when structured logs contain gaps longer than this many minutes.",
    )
    parser.add_argument(
        "--diary-stale-warn-minutes",
        type=float,
        help="Warn when the decision diary stalls for this many minutes.",
    )
    parser.add_argument(
        "--diary-stale-fail-minutes",
        type=float,
        help="Fail when the decision diary stalls for this many minutes.",
    )
    parser.add_argument(
        "--performance-stale-warn-minutes",
        type=float,
        help="Warn when performance telemetry stalls for this many minutes.",
    )
    parser.add_argument(
        "--performance-stale-fail-minutes",
        type=float,
        help="Fail when performance telemetry stalls for this many minutes.",
    )
    parser.add_argument(
        "--evidence-check-interval-minutes",
        type=float,
        help="Polling interval (minutes) for evidence freshness checks (default auto).",
    )
    parser.add_argument(
        "--evidence-initial-grace-minutes",
        type=float,
        default=15.0,
        help="Minutes to wait before enforcing evidence freshness thresholds (default: 15).",
    )
    parser.add_argument(
        "--live-gap-alert-minutes",
        type=float,
        help="Emit live incidents when no logs arrive for this many minutes (optional).",
    )
    parser.add_argument(
        "--live-gap-alert-severity",
        choices=[DryRunStatus.warn.value, DryRunStatus.fail.value],
        default=DryRunStatus.warn.value,
        help="Severity for live log gap incidents (default: warn).",
    )
    parser.add_argument(
        "--log-rotate-hours",
        type=float,
        help="Rotate logs after this many hours (default: disabled).",
    )
    parser.add_argument(
        "--compress-logs",
        action="store_true",
        help="Compress structured and raw logs (writes .gz files).",
    )
    parser.add_argument(
        "--resource-sample-interval-minutes",
        type=float,
        default=1.0,
        help=(
            "Interval in minutes between resource usage samples (set to 0 to disable)."
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
        choices=[DryRunStatus.warn.value, DryRunStatus.fail.value],
        default=DryRunStatus.fail.value,
        help="Severity recorded when resource thresholds are breached.",
    )
    parser.add_argument(
        "--no-resource-monitor",
        action="store_true",
        help="Disable resource usage monitoring during the run.",
    )
    parser.add_argument(
        "--skip-diary-evidence",
        action="store_true",
        help="Do not require decision diary evidence for sign-off.",
    )
    parser.add_argument(
        "--skip-performance-evidence",
        action="store_true",
        help="Do not require performance telemetry evidence for sign-off.",
    )
    parser.add_argument(
        "--diary-path",
        type=Path,
        help="Explicit path for the decision diary JSONL file (default: run_dir/decision_diary.jsonl).",
    )
    parser.add_argument(
        "--performance-path",
        type=Path,
        help="Explicit path for the performance metrics JSON (default: run_dir/performance_metrics.json).",
    )
    parser.add_argument(
        "--metadata",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Attach additional metadata entries to the dry run summary.",
    )
    parser.add_argument(
        "--env",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Environment variables to inject into the runtime command.",
    )
    parser.add_argument(
        "--objective",
        action="append",
        default=[],
        metavar="NAME=STATUS[:NOTE]",
        help="Record review objectives (pass/warn/fail).",
    )
    parser.add_argument(
        "--attendee",
        action="append",
        default=[],
        metavar="NAME",
        help="Add an attendee to the review brief.",
    )
    parser.add_argument(
        "--note",
        action="append",
        default=[],
        metavar="TEXT",
        help="Attach a free-form note to the review brief.",
    )
    parser.add_argument(
        "--notes-file",
        action="append",
        default=[],
        metavar="PATH",
        help="Load additional notes from a text file (one per line).",
    )
    parser.add_argument(
        "--no-review",
        action="store_true",
        help="Skip generating the review brief.",
    )
    parser.add_argument(
        "--review-include-summary",
        "--review-skip-summary",
        dest="review_include_summary",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Toggle inclusion of the dry run summary in the review brief (default: include).",
    )
    parser.add_argument(
        "--review-include-sign-off",
        "--review-skip-sign-off",
        dest="review_include_sign_off",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Toggle inclusion of the sign-off assessment in the review brief (default: include).",
    )
    parser.add_argument(
        "--review-json",
        type=Path,
        help="Optional path for the review brief in JSON format (default: run_dir/review.json).",
    )
    parser.add_argument(
        "--review-markdown",
        type=Path,
        help="Optional path for the review brief in Markdown format (default: run_dir/review.md).",
    )
    parser.add_argument(
        "--no-packet",
        action="store_true",
        help="Skip building an evidence packet directory/archive.",
    )
    parser.add_argument(
        "--packet-dir",
        type=Path,
        help="Explicit directory for the evidence packet (default: run_dir/packet).",
    )
    parser.add_argument(
        "--packet-archive",
        type=Path,
        help="Optional .tar.gz path for the evidence packet (default: run_dir/packet.tar.gz).",
    )
    parser.add_argument(
        "--packet-skip-raw",
        action="store_true",
        help="Do not copy raw artefacts into the evidence packet.",
    )
    parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="Runtime command to execute (provide after '--').",
    )
    return parser


def _timedelta_from_minutes(
    value: float | None,
    *,
    allow_zero: bool = False,
) -> timedelta | None:
    if value is None:
        return None
    if value < 0:
        raise ValueError("Interval values must be non-negative when provided")
    if value == 0 and not allow_zero:
        raise ValueError("Interval values must be positive when provided")
    return timedelta(minutes=value)


def _timedelta_from_hours(value: float | None) -> timedelta | None:
    if value is None:
        return None
    if value <= 0:
        raise ValueError("Hour values must be positive when provided")
    return timedelta(hours=value)


def _timedelta_from_seconds(value: float) -> timedelta:
    if value < 0:
        raise ValueError("Seconds value must be non-negative")
    return timedelta(seconds=value)


def _normalise_command(values: Sequence[str]) -> list[str]:
    command = list(values)
    if command and command[0] == "--":
        command = command[1:]
    return command


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    command = _normalise_command(args.command)
    if not command:
        parser.error(
            "Provide the runtime command after '--', e.g. final_dry_run_orchestrator.py -- python main.py"
        )

    if args.no_review and (args.review_markdown or args.review_json):
        parser.error("--no-review cannot be combined with --review-markdown or --review-json")
    if args.no_packet and (args.packet_dir or args.packet_archive or args.packet_skip_raw):
        parser.error("--no-packet cannot be combined with --packet-dir, --packet-archive, or --packet-skip-raw")

    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    slug = args.run_name or datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = output_root / slug
    try:
        run_dir.mkdir(parents=False, exist_ok=False)
    except FileExistsError as exc:
        raise SystemExit(f"Run directory already exists: {run_dir}") from exc

    log_dir = run_dir / "logs"
    log_dir.mkdir()
    progress_path = run_dir / "progress.json"
    summary_json = run_dir / "summary.json"
    summary_markdown = run_dir / "summary.md"

    default_review_md = run_dir / "review.md"
    default_review_json = run_dir / "review.json"

    review_markdown_path = args.review_markdown or default_review_md
    review_json_path = args.review_json or default_review_json

    packet_dir = None
    packet_archive = None
    if not args.no_packet:
        packet_dir = args.packet_dir or (run_dir / "packet")
        packet_archive = args.packet_archive or (run_dir / "packet.tar.gz")

    diary_path: Path | None
    if args.skip_diary_evidence:
        diary_path = None
    else:
        diary_path = args.diary_path or (run_dir / "decision_diary.jsonl")
        diary_path.parent.mkdir(parents=True, exist_ok=True)

    performance_path: Path | None
    if args.skip_performance_evidence:
        performance_path = None
    else:
        performance_path = args.performance_path or (run_dir / "performance_metrics.json")
        performance_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        metadata = _parse_pairs("--metadata", args.metadata)
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))
    metadata.setdefault("run_dir", run_dir.as_posix())
    metadata.setdefault("run_name", slug)
    if args.run_label:
        metadata.setdefault("run_label", args.run_label)

    try:
        env_pairs = _parse_pairs("--env", args.env)
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))
    auto_env: dict[str, str] = {
        _LOG_DIR_ENV: log_dir.as_posix(),
    }
    if diary_path is not None and _DIARY_ENV not in env_pairs:
        auto_env[_DIARY_ENV] = diary_path.as_posix()
    if performance_path is not None and _PERFORMANCE_ENV not in env_pairs:
        auto_env[_PERFORMANCE_ENV] = performance_path.as_posix()
    if args.run_label and "FINAL_DRY_RUN_LABEL" not in env_pairs:
        auto_env["FINAL_DRY_RUN_LABEL"] = args.run_label
    merged_env: dict[str, str] = {**auto_env, **env_pairs}

    note_lines = list(args.note)
    note_lines.extend(_load_notes(Path(path) for path in args.notes_file))
    review_notes = tuple(line for line in note_lines if line.strip())

    objectives = []
    for item in args.objective:
        try:
            objectives.append(parse_objective_spec(item))
        except ValueError as exc:
            parser.error(str(exc))
    objectives_tuple = tuple(objectives)

    try:
        duration = _timedelta_from_hours(args.duration_hours)
        required_duration = (
            _timedelta_from_hours(args.required_duration_hours)
            if args.required_duration_hours is not None
            else None
        )
        progress_interval = (
            _timedelta_from_minutes(args.progress_interval_minutes)
            if args.progress_interval_minutes > 0
            else None
        )
        shutdown_grace = _timedelta_from_seconds(args.shutdown_grace_seconds)
        warn_gap = _timedelta_from_minutes(args.warn_gap_minutes)
        fail_gap = _timedelta_from_minutes(args.fail_gap_minutes)
        diary_stale_warn = _timedelta_from_minutes(args.diary_stale_warn_minutes)
        diary_stale_fail = _timedelta_from_minutes(args.diary_stale_fail_minutes)
        performance_stale_warn = _timedelta_from_minutes(args.performance_stale_warn_minutes)
        performance_stale_fail = _timedelta_from_minutes(args.performance_stale_fail_minutes)
        evidence_interval = _timedelta_from_minutes(args.evidence_check_interval_minutes)
        evidence_grace = _timedelta_from_minutes(
            args.evidence_initial_grace_minutes,
            allow_zero=True,
        )
        live_gap_alert = _timedelta_from_minutes(args.live_gap_alert_minutes)
        live_gap_severity = DryRunStatus(args.live_gap_alert_severity)
        log_rotate_interval = _timedelta_from_hours(args.log_rotate_hours)
        resource_interval = None
        if (
            not args.no_resource_monitor
            and args.resource_sample_interval_minutes > 0
        ):
            resource_interval = _timedelta_from_minutes(
                args.resource_sample_interval_minutes
            )
        resource_severity = DryRunStatus(args.resource_violation_severity)
    except ValueError as exc:
        parser.error(str(exc))

    config = FinalDryRunConfig(
        command=tuple(command),
        duration=duration,
        required_duration=required_duration,
        log_directory=log_dir,
        progress_path=progress_path,
        progress_interval=progress_interval,
        diary_path=diary_path,
        performance_path=performance_path,
        minimum_uptime_ratio=args.minimum_uptime_ratio,
        shutdown_grace=shutdown_grace,
        require_diary_evidence=not args.skip_diary_evidence,
        require_performance_evidence=not args.skip_performance_evidence,
        allow_warnings=args.allow_warnings,
        minimum_sharpe_ratio=args.minimum_sharpe_ratio,
        metadata=metadata,
        environment=merged_env,
        log_gap_warn=warn_gap,
        log_gap_fail=fail_gap,
        diary_stale_warn=diary_stale_warn,
        diary_stale_fail=diary_stale_fail,
        performance_stale_warn=performance_stale_warn,
        performance_stale_fail=performance_stale_fail,
        evidence_check_interval=evidence_interval,
        evidence_initial_grace=evidence_grace or timedelta(minutes=0),
        live_gap_alert=live_gap_alert,
        live_gap_severity=live_gap_severity,
        compress_logs=args.compress_logs,
        log_rotate_interval=log_rotate_interval,
        resource_sample_interval=resource_interval,
        resource_max_cpu_percent=args.resource_max_cpu_percent,
        resource_max_memory_mb=args.resource_max_memory_mb,
        resource_max_memory_percent=args.resource_max_memory_percent,
        resource_violation_severity=resource_severity,
    )

    workflow = run_final_dry_run_workflow(
        config,
        evidence_dir=packet_dir,
        evidence_archive=packet_archive,
        include_raw_artifacts=not args.packet_skip_raw,
        review_run_label=args.run_label or slug,
        review_attendees=tuple(args.attendee),
        review_notes=review_notes,
        review_objectives=objectives_tuple,
        create_review=not args.no_review,
    )
    result = workflow.run_result

    summary_payload: MutableMapping[str, object] = {
        "status": result.status.value,
        "summary": result.summary.as_dict(),
        "command": list(result.config.command),
        "run_directory": run_dir.as_posix(),
        "log_directory": log_dir.as_posix(),
        "progress_path": str(result.progress_path) if result.progress_path else None,
        "incidents": [incident.as_dict() for incident in result.incidents],
    }
    if result.sign_off is not None:
        summary_payload["sign_off"] = result.sign_off.as_dict()
    if workflow.review is not None:
        summary_payload["review"] = workflow.review.as_dict()
    if workflow.evidence_packet is not None:
        summary_payload["evidence_packet"] = workflow.evidence_packet.as_dict()

    summary_json.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    markdown_sections = [result.summary.to_markdown().rstrip("\n")]
    if result.sign_off is not None:
        markdown_sections.append(result.sign_off.to_markdown().rstrip("\n"))
    if workflow.review is not None:
        markdown_sections.append(
            workflow.review.to_markdown(
                include_summary=args.review_include_summary,
                include_sign_off=args.review_include_sign_off,
            ).rstrip("\n")
        )
    summary_markdown.write_text("\n\n".join(markdown_sections) + "\n", encoding="utf-8")

    if workflow.review is not None:
        review_json_path.write_text(
            json.dumps(workflow.review.as_dict(), indent=2),
            encoding="utf-8",
        )
        review_markdown_path.write_text(
            workflow.review.to_markdown(
                include_summary=args.review_include_summary,
                include_sign_off=args.review_include_sign_off,
            ),
            encoding="utf-8",
        )

    print(f"Final dry run status: {result.status.value.upper()}")
    print(f"  Runtime command: {' '.join(result.config.command)}")
    print(f"  Exit code: {result.exit_code}")
    print(f"  Logs directory: {log_dir}")
    if result.progress_path is not None:
        print(f"  Progress: {result.progress_path}")
    if result.incidents:
        print("  Harness incidents:")
        for incident in result.incidents:
            print(
                f"    - {incident.occurred_at.astimezone().isoformat()} — "
                f"{incident.severity.value.upper()}: {incident.message}"
            )
    print(f"  Summary JSON: {summary_json}")
    print(f"  Summary Markdown: {summary_markdown}")
    if workflow.review is not None:
        print(f"  Review status: {workflow.review.status.value.upper()}")
        print(f"  Review markdown: {review_markdown_path}")
        print(f"  Review JSON: {review_json_path}")
    else:
        print("  Review generation disabled.")
    if workflow.evidence_packet is not None:
        print(f"  Evidence packet directory: {workflow.evidence_packet.output_dir}")
        if workflow.evidence_packet.archive_path is not None:
            print(f"  Evidence packet archive: {workflow.evidence_packet.archive_path}")
    else:
        print("  Evidence packet generation skipped.")

    if result.status is DryRunStatus.fail:
        return 2
    if result.status is DryRunStatus.warn:
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
