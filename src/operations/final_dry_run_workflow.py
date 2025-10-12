"""High-level workflow helpers for executing the final dry run."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from src.operations.dry_run_packet import DryRunPacketPaths, write_dry_run_packet
from src.operations.final_dry_run import (
    FinalDryRunConfig,
    FinalDryRunResult,
    perform_final_dry_run,
)
from src.operations.final_dry_run_review import FinalDryRunReview, build_review

__all__ = [
    "FinalDryRunWorkflowResult",
    "perform_final_dry_run_workflow",
    "run_final_dry_run_workflow",
]


@dataclass(frozen=True)
class FinalDryRunWorkflowResult:
    """Bundle returned after running the end-to-end dry run workflow."""

    run_result: FinalDryRunResult
    evidence_packet: DryRunPacketPaths | None
    review: FinalDryRunReview | None


async def perform_final_dry_run_workflow(
    config: FinalDryRunConfig,
    *,
    evidence_dir: Path | None = None,
    evidence_archive: Path | None = None,
    include_raw_artifacts: bool = True,
    review_run_label: str | None = None,
    review_attendees: Iterable[str] = (),
    review_notes: Iterable[str] = (),
    create_review: bool = True,
) -> FinalDryRunWorkflowResult:
    """Execute the dry run, evidence packet, and review assembly workflow."""

    run_result = await perform_final_dry_run(config)

    log_paths: list[Path] = [run_result.log_path]
    if run_result.raw_log_path != run_result.log_path:
        log_paths.append(run_result.raw_log_path)

    evidence_packet: DryRunPacketPaths | None = None
    if evidence_dir is not None:
        evidence_packet = write_dry_run_packet(
            summary=run_result.summary,
            sign_off_report=run_result.sign_off,
            output_dir=evidence_dir,
            log_paths=log_paths,
            diary_path=config.diary_path,
            performance_path=config.performance_path,
            include_raw_artifacts=include_raw_artifacts,
            archive_path=evidence_archive,
        )

    review: FinalDryRunReview | None = None
    if create_review:
        review = build_review(
            run_result.summary,
            run_result.sign_off,
            run_label=review_run_label,
            attendees=tuple(review_attendees),
            notes=tuple(review_notes),
            evidence_packet=evidence_packet,
        )

    return FinalDryRunWorkflowResult(
        run_result=run_result,
        evidence_packet=evidence_packet,
        review=review,
    )


def run_final_dry_run_workflow(
    config: FinalDryRunConfig,
    **kwargs: object,
) -> FinalDryRunWorkflowResult:
    """Synchronous wrapper around :func:`perform_final_dry_run_workflow`."""

    return asyncio.run(perform_final_dry_run_workflow(config, **kwargs))
