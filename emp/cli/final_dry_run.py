"""Click command for orchestrating the AlphaTrade final dry run."""

from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import click

from src.operations.dry_run_audit import DryRunStatus
from src.operations.final_dry_run import FinalDryRunConfig
from src.operations.final_dry_run_workflow import run_final_dry_run_workflow


def _parse_pairs(option: str, values: Sequence[str]) -> Mapping[str, str]:
    pairs: dict[str, str] = {}
    for item in values:
        if "=" not in item:
            raise click.BadParameter(
                f"{option} expects KEY=VALUE entries (received '{item}')"
            )
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise click.BadParameter(f"{option} key cannot be empty")
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


@click.command("final-dry-run")
@click.option(
    "--log-dir",
    "log_dir",
    type=click.Path(path_type=Path),
    required=True,
    help="Directory for harness logs, structured output, and reports.",
)
@click.option(
    "--duration-hours",
    "duration_hours",
    type=float,
    default=72.0,
    show_default=True,
    help="Requested runtime duration in hours.",
)
@click.option(
    "--progress-path",
    "progress_path",
    type=click.Path(path_type=Path),
    help="Optional JSON file for periodic progress snapshots.",
)
@click.option(
    "--progress-interval-minutes",
    "progress_interval_minutes",
    type=float,
    default=5.0,
    show_default=True,
    help="Minutes between progress snapshots (0 to disable).",
)
@click.option(
    "--required-duration-hours",
    "required_duration_hours",
    type=float,
    help="Minimum runtime duration (hours) required for sign-off.",
)
@click.option(
    "--minimum-uptime-ratio",
    "minimum_uptime_ratio",
    type=float,
    default=0.98,
    show_default=True,
    help="Minimum acceptable uptime ratio for log coverage.",
)
@click.option(
    "--shutdown-grace-seconds",
    "shutdown_grace_seconds",
    type=float,
    default=60.0,
    show_default=True,
    help="Grace period in seconds before force-terminating the runtime.",
)
@click.option(
    "--diary",
    type=click.Path(path_type=Path),
    help="Optional decision diary JSONL path.",
)
@click.option(
    "--performance",
    type=click.Path(path_type=Path),
    help="Optional performance telemetry JSON path.",
)
@click.option(
    "--allow-warnings",
    is_flag=True,
    help="Treat WARN severities as acceptable for sign-off.",
)
@click.option(
    "--no-diary-required",
    is_flag=True,
    help="Do not require decision diary evidence for sign-off.",
)
@click.option(
    "--no-performance-required",
    is_flag=True,
    help="Do not require performance telemetry for sign-off.",
)
@click.option(
    "--minimum-sharpe-ratio",
    type=float,
    help="Minimum Sharpe ratio required for sign-off.",
)
@click.option(
    "--warn-gap-minutes",
    type=float,
    help="Warn when structured logs contain gaps longer than this many minutes.",
)
@click.option(
    "--fail-gap-minutes",
    type=float,
    help="Fail when structured logs contain gaps longer than this many minutes.",
)
@click.option(
    "--metadata",
    multiple=True,
    help="Attach KEY=VALUE metadata entries to the summary payload.",
)
@click.option(
    "--env",
    multiple=True,
    help="Inject KEY=VALUE environment variables into the runtime command.",
)
@click.option(
    "--json-report",
    type=click.Path(path_type=Path),
    help="Optional path for a JSON bundle containing summary/sign-off/review data.",
)
@click.option(
    "--markdown-report",
    type=click.Path(path_type=Path),
    help="Optional path for a Markdown bundle containing summary and sign-off details.",
)
@click.option(
    "--packet-dir",
    type=click.Path(path_type=Path),
    help="Write an evidence packet (logs + summaries) to this directory.",
)
@click.option(
    "--packet-archive",
    type=click.Path(path_type=Path),
    help="Optional .tar.gz archive path for the evidence packet.",
)
@click.option(
    "--packet-skip-raw",
    is_flag=True,
    help="Skip copying raw logs/diary/performance artefacts into the evidence packet.",
)
@click.option(
    "--no-log-monitor",
    is_flag=True,
    help="Disable live log level monitoring during the run.",
)
@click.option(
    "--live-gap-alert-minutes",
    type=float,
    help="Emit a harness incident when no logs arrive for this many minutes.",
)
@click.option(
    "--live-gap-alert-severity",
    type=click.Choice([DryRunStatus.warn.value, DryRunStatus.fail.value]),
    default=DryRunStatus.warn.value,
    show_default=True,
    help="Severity assigned to live log gap incidents.",
)
@click.option(
    "--review-output",
    type=str,
    help="Path for the review brief (use '-' for stdout).",
)
@click.option(
    "--review-format",
    type=click.Choice(["markdown", "json"]),
    default="markdown",
    show_default=True,
    help="Output format for the review brief when --review-output is supplied.",
)
@click.option(
    "--review-run-label",
    type=str,
    help="Optional descriptor for the run in the review brief.",
)
@click.option(
    "--attendee",
    "attendees",
    multiple=True,
    help="Add an attendee to the review brief.",
)
@click.option(
    "--note",
    "notes",
    multiple=True,
    help="Attach a free-form note to the review brief.",
)
@click.option(
    "--notes-file",
    type=click.Path(path_type=Path),
    multiple=True,
    help="Load notes from a text file (one per line).",
)
@click.option(
    "--no-review",
    is_flag=True,
    help="Skip generating the review brief.",
)
@click.option(
    "--review-include-summary/--review-skip-summary",
    default=True,
    help="Toggle inclusion of the dry run summary in the review output.",
)
@click.option(
    "--review-include-sign-off/--review-skip-sign-off",
    default=True,
    help="Toggle inclusion of the sign-off assessment in the review output.",
)
@click.argument("command", nargs=-1)
def final_dry_run(  # noqa: PLR0913 - CLI fan-out handled by click
    *,
    log_dir: Path,
    duration_hours: float,
    progress_path: Path | None,
    progress_interval_minutes: float,
    required_duration_hours: float | None,
    minimum_uptime_ratio: float,
    shutdown_grace_seconds: float,
    diary: Path | None,
    performance: Path | None,
    allow_warnings: bool,
    no_diary_required: bool,
    no_performance_required: bool,
    minimum_sharpe_ratio: float | None,
    warn_gap_minutes: float | None,
    fail_gap_minutes: float | None,
    metadata: Sequence[str],
    env: Sequence[str],
    json_report: Path | None,
    markdown_report: Path | None,
    packet_dir: Path | None,
    packet_archive: Path | None,
    packet_skip_raw: bool,
    no_log_monitor: bool,
    live_gap_alert_minutes: float | None,
    live_gap_alert_severity: str,
    review_output: str | None,
    review_format: str,
    review_run_label: str | None,
    attendees: Sequence[str],
    notes: Sequence[str],
    notes_file: Sequence[Path],
    no_review: bool,
    review_include_summary: bool,
    review_include_sign_off: bool,
    command: Sequence[str],
) -> None:
    """Run the AlphaTrade runtime under the final dry run harness."""

    if not command:
        raise click.UsageError(
            "Provide the runtime command after '--', e.g. emp final-dry-run -- python main.py"
        )
    if packet_archive is not None and packet_dir is None:
        raise click.UsageError("--packet-archive requires --packet-dir")
    if packet_skip_raw and packet_dir is None:
        raise click.UsageError("--packet-skip-raw requires --packet-dir")
    if no_review and review_output is not None:
        raise click.UsageError("--no-review cannot be combined with --review-output")

    metadata_pairs = _parse_pairs("--metadata", metadata)
    env_pairs = _parse_pairs("--env", env)

    note_lines = list(notes)
    note_lines.extend(_load_notes(notes_file))
    review_notes = tuple(line for line in note_lines if line.strip())

    duration = timedelta(hours=duration_hours)
    required_duration = (
        timedelta(hours=required_duration_hours)
        if required_duration_hours is not None
        else None
    )
    shutdown_grace = timedelta(seconds=max(shutdown_grace_seconds, 0.0))
    progress_interval = (
        timedelta(minutes=progress_interval_minutes)
        if progress_interval_minutes > 0.0
        else None
    )
    warn_gap = (
        timedelta(minutes=warn_gap_minutes)
        if warn_gap_minutes is not None
        else None
    )
    fail_gap = (
        timedelta(minutes=fail_gap_minutes)
        if fail_gap_minutes is not None
        else None
    )
    live_gap = (
        timedelta(minutes=live_gap_alert_minutes)
        if live_gap_alert_minutes is not None
        else None
    )
    live_severity = DryRunStatus(live_gap_alert_severity)

    config = FinalDryRunConfig(
        command=command,
        duration=duration,
        log_directory=log_dir,
        progress_path=progress_path,
        progress_interval=progress_interval,
        diary_path=diary,
        performance_path=performance,
        minimum_uptime_ratio=minimum_uptime_ratio,
        required_duration=required_duration,
        shutdown_grace=shutdown_grace,
        require_diary_evidence=not no_diary_required,
        require_performance_evidence=not no_performance_required,
        allow_warnings=allow_warnings,
        minimum_sharpe_ratio=minimum_sharpe_ratio,
        metadata=metadata_pairs,
        environment=env_pairs or None,
        monitor_log_levels=not no_log_monitor,
        log_gap_warn=warn_gap,
        log_gap_fail=fail_gap,
        live_gap_alert=live_gap,
        live_gap_severity=live_severity,
    )

    workflow = run_final_dry_run_workflow(
        config,
        evidence_dir=packet_dir,
        evidence_archive=packet_archive,
        include_raw_artifacts=not packet_skip_raw,
        review_run_label=review_run_label,
        review_attendees=attendees,
        review_notes=review_notes,
        create_review=not no_review,
    )
    result = workflow.run_result

    click.echo(f"Final dry run status: {result.status.value.upper()}")
    click.echo(f"  Runtime command: {' '.join(result.config.command)}")
    click.echo(f"  Exit code: {result.exit_code}")
    click.echo(f"  Duration: {result.duration}")
    click.echo(f"  Logs: {result.log_path}")
    click.echo(f"  Raw logs: {result.raw_log_path}")
    if result.progress_path is not None:
        click.echo(f"  Progress: {result.progress_path}")
    if result.incidents:
        click.echo("  Harness incidents:")
        for incident in result.incidents:
            occurred = incident.occurred_at.astimezone().isoformat()
            click.echo(
                f"    - {occurred} — {incident.severity.value.upper()}: {incident.message}"
            )
    log_summary_status = (
        result.summary.log_summary.status.value
        if result.summary.log_summary is not None
        else "n/a"
    )
    click.echo(f"  Log summary status: {log_summary_status}")
    if result.sign_off is not None:
        click.echo(f"  Sign-off status: {result.sign_off.status.value.upper()}")
    if workflow.evidence_packet is not None:
        click.echo(
            f"  Evidence packet directory: {workflow.evidence_packet.output_dir}"
        )
        if workflow.evidence_packet.archive_path is not None:
            click.echo(
                f"  Evidence packet archive: {workflow.evidence_packet.archive_path}"
            )

    if workflow.review is not None:
        if review_output is not None:
            review_text = (
                json.dumps(workflow.review.as_dict(), indent=2)
                if review_format == "json"
                else workflow.review.to_markdown(
                    include_summary=review_include_summary,
                    include_sign_off=review_include_sign_off,
                )
            )
            if review_output == "-":
                click.echo(review_text)
            else:
                review_path = Path(review_output)
                review_path.parent.mkdir(parents=True, exist_ok=True)
                review_path.write_text(review_text, encoding="utf-8")
                click.echo(f"  Review written to: {review_path}")
        else:
            click.echo(
                "  Review brief not written (pass --review-output or '--review-output -' to emit)."
            )

    if json_report is not None:
        payload: dict[str, object] = {
            "status": result.status.value,
            "summary": result.summary.as_dict(),
        }
        if result.sign_off is not None:
            payload["sign_off"] = result.sign_off.as_dict()
        if result.incidents:
            payload["incidents"] = [incident.as_dict() for incident in result.incidents]
        if workflow.review is not None:
            payload["review"] = workflow.review.as_dict()
        json_report.parent.mkdir(parents=True, exist_ok=True)
        json_report.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if markdown_report is not None:
        parts = [result.summary.to_markdown()]
        if result.sign_off is not None:
            parts.append(result.sign_off.to_markdown())
        if workflow.review is not None:
            parts.append(
                workflow.review.to_markdown(
                    include_summary=review_include_summary,
                    include_sign_off=review_include_sign_off,
                )
            )
        markdown_report.parent.mkdir(parents=True, exist_ok=True)
        markdown_report.write_text("\n".join(parts), encoding="utf-8")

    exit_code = 0
    if result.status is DryRunStatus.fail:
        exit_code = 2
    elif result.status is DryRunStatus.warn:
        exit_code = 1
    raise click.exceptions.Exit(exit_code)
