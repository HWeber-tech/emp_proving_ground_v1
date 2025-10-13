"""Utilities for exercising the final dry run harness via a smoke workflow."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Mapping, MutableMapping

from src.operations.dry_run_audit import DryRunStatus
from src.operations.final_dry_run import FinalDryRunConfig
from src.operations.final_dry_run_workflow import (
    FinalDryRunWorkflowResult,
    run_final_dry_run_workflow,
)


@dataclass(slots=True)
class SmokeRunOptions:
    """Configuration envelope for running a final dry run smoke test."""

    output_dir: Path
    duration: timedelta = timedelta(seconds=60)
    tick_interval: timedelta = timedelta(seconds=2)
    progress_interval: timedelta = timedelta(seconds=5)
    minimum_uptime_ratio: float = 0.95
    allow_warnings: bool = False
    metadata: Mapping[str, object] = field(default_factory=dict)
    environment: Mapping[str, str] | None = None
    packet_dir: Path | None = None
    packet_archive: Path | None = None
    packet_include_raw: bool = True
    review_markdown: Path | None = None
    review_json: Path | None = None
    review_notes: tuple[str, ...] = tuple()


def run_smoke_workflow(options: SmokeRunOptions) -> FinalDryRunWorkflowResult:
    """Run the final dry run harness against the simulated runtime."""

    if options.duration <= timedelta(0):
        raise ValueError("duration must be positive")
    if options.tick_interval <= timedelta(0):
        raise ValueError("tick_interval must be positive")
    if not (0.0 < options.minimum_uptime_ratio <= 1.0):
        raise ValueError("minimum_uptime_ratio must be between 0 and 1")

    output_dir = options.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    diary_path = output_dir / "decision_diary.jsonl"
    performance_path = output_dir / "performance_metrics.json"
    log_dir = output_dir / "logs"
    progress_path = output_dir / "progress.json"

    timestamp_token = datetime.now(tz=UTC).strftime("%Y%m%d-%H%M%S")
    review_label = f"Smoke run {timestamp_token}"

    tick_seconds = options.tick_interval.total_seconds()

    command = [
        sys.executable,
        "-m",
        "src.operations.final_dry_run_simulated_runtime",
        "--duration-seconds",
        f"{options.duration.total_seconds():.6f}",
        "--tick-interval",
        f"{tick_seconds:.6f}",
        "--diary",
        diary_path.as_posix(),
        "--performance",
        performance_path.as_posix(),
    ]

    metadata: MutableMapping[str, object] = {
        "mode": "smoke",
        "tick_interval_seconds": tick_seconds,
        "requested_duration_seconds": options.duration.total_seconds(),
        "review_label": review_label,
    }
    metadata.update(options.metadata)

    warn_gap = max(options.tick_interval * 3, timedelta(seconds=3))
    fail_gap = max(options.tick_interval * 6, timedelta(seconds=6))
    diary_warn = max(options.tick_interval * 3, timedelta(seconds=3))
    diary_fail = max(options.tick_interval * 6, timedelta(seconds=6))
    performance_warn = diary_warn
    performance_fail = diary_fail

    evidence_interval = max(options.tick_interval, timedelta(seconds=1))
    evidence_grace = max(options.tick_interval * 2, timedelta(seconds=2))

    config = FinalDryRunConfig(
        command=command,
        duration=options.duration,
        log_directory=log_dir,
        progress_path=progress_path,
        progress_interval=options.progress_interval,
        diary_path=diary_path,
        performance_path=performance_path,
        minimum_uptime_ratio=options.minimum_uptime_ratio,
        required_duration=options.duration,
        allow_warnings=options.allow_warnings,
        metadata=metadata,
        environment=options.environment,
        monitor_log_levels=True,
        log_gap_warn=warn_gap,
        log_gap_fail=fail_gap,
        live_gap_alert=warn_gap,
        live_gap_severity=DryRunStatus.warn,
        diary_stale_warn=diary_warn,
        diary_stale_fail=diary_fail,
        performance_stale_warn=performance_warn,
        performance_stale_fail=performance_fail,
        evidence_check_interval=evidence_interval,
        evidence_initial_grace=evidence_grace,
    )

    review_kwargs: dict[str, object] = {
        "review_run_label": review_label,
        "review_notes": options.review_notes,
        "create_review": bool(options.review_markdown or options.review_json),
    }

    result = run_final_dry_run_workflow(
        config,
        evidence_dir=options.packet_dir,
        evidence_archive=options.packet_archive,
        include_raw_artifacts=options.packet_include_raw,
        **review_kwargs,
    )

    if options.review_markdown and result.review is not None:
        options.review_markdown.parent.mkdir(parents=True, exist_ok=True)
        options.review_markdown.write_text(
            result.review.to_markdown(),
            encoding="utf-8",
        )

    if options.review_json and result.review is not None:
        options.review_json.parent.mkdir(parents=True, exist_ok=True)
        options.review_json.write_text(
            json.dumps(result.review.as_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )

    return result
