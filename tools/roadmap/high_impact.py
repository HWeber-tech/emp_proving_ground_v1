"""Evaluate the document-driven high-impact roadmap streams."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

from ._shared import (
    Requirement,
    evaluate_requirements,
    repo_root,
    require_module_attr,
)


SUMMARY_MARKER_START = "<!-- HIGH_IMPACT_SUMMARY:START -->"
SUMMARY_MARKER_END = "<!-- HIGH_IMPACT_SUMMARY:END -->"


@dataclass(frozen=True)
class StreamDefinition:
    """Definition describing a high-impact roadmap stream."""

    stream: str
    ready_summary: str
    attention_summary: str
    next_checkpoint: str
    requirements: Sequence[Requirement]

    def evaluate(self, repo_root: Path) -> "StreamStatus":
        evidence, missing = evaluate_requirements(self.requirements, repo_root)

        status = "Ready" if not missing else "Attention needed"
        summary = self.ready_summary if not missing else self.attention_summary

        return StreamStatus(
            stream=self.stream,
            status=status,
            summary=summary,
            next_checkpoint=self.next_checkpoint,
            evidence=tuple(evidence),
            missing=tuple(missing),
        )


@dataclass(frozen=True)
class StreamStatus:
    """Computed status for a high-impact roadmap stream."""

    stream: str
    status: str
    summary: str
    next_checkpoint: str
    evidence: tuple[str, ...]
    missing: tuple[str, ...]


def _stream_definitions() -> Sequence[StreamDefinition]:
    return (
        StreamDefinition(
            stream="Stream A – Institutional data backbone",
            ready_summary=(
                "Timescale ingest, Redis caching, Kafka streaming, and Spark exports ship "
                "with readiness telemetry and failover tooling."
            ),
            attention_summary=(
                "Data backbone scaffolding is incomplete; ensure ingest pipelines, caches, "
                "streaming publishers, and readiness evaluators exist."
            ),
            next_checkpoint=(
                "Exercise cross-region failover and automated scheduler cutover using the readiness feeds."
            ),
            requirements=(
                require_module_attr(
                    "data_foundation.ingest.timescale_pipeline",
                    "TimescaleBackboneOrchestrator",
                ),
                require_module_attr(
                    "data_foundation.ingest.configuration",
                    "build_institutional_ingest_config",
                ),
                require_module_attr(
                    "data_foundation.ingest.quality",
                    "evaluate_ingest_quality",
                ),
                require_module_attr(
                    "data_foundation.cache.redis_cache",
                    "ManagedRedisCache",
                ),
                require_module_attr(
                    "data_foundation.streaming.kafka_stream",
                    "KafkaIngestEventPublisher",
                ),
                require_module_attr(
                    "data_foundation.streaming.kafka_stream",
                    "KafkaIngestQualityPublisher",
                ),
                require_module_attr(
                    "data_foundation.batch.spark_export",
                    "execute_spark_export_plan",
                ),
                require_module_attr(
                    "operations.data_backbone",
                    "evaluate_data_backbone_readiness",
                ),
                require_module_attr(
                    "operations.ingest_trends",
                    "evaluate_ingest_trends",
                ),
                require_module_attr(
                    "data_foundation.ingest.failover",
                    "decide_ingest_failover",
                ),
                require_module_attr(
                    "operations.backup",
                    "evaluate_backup_readiness",
                ),
                require_module_attr(
                    "operations.spark_stress",
                    "execute_spark_stress_drill",
                ),
            ),
        ),
        StreamDefinition(
            stream="Stream B – Sensory cortex & evolution uplift",
            ready_summary=(
                "All five sensory organs operate with drift telemetry and catalogue-backed "
                "evolution lineage exports."
            ),
            attention_summary=(
                "Sensory cortex or evolution catalogue support is missing; ensure HOW/ANOMALY "
                "organs, drift telemetry, and catalogue exports are present."
            ),
            next_checkpoint=(
                "Extend live-paper experiments and automated tuning loops using evolution telemetry."
            ),
            requirements=(
                require_module_attr("sensory.how.how_sensor", "HowSensor"),
                require_module_attr("sensory.anomaly.anomaly_sensor", "AnomalySensor"),
                require_module_attr(
                    "sensory.when.gamma_exposure", "GammaExposureAnalyzer"
                ),
                require_module_attr("sensory.why.why_sensor", "WhySensor"),
                require_module_attr("sensory.what.what_sensor", "WhatSensor"),
                require_module_attr(
                    "operations.sensory_drift", "evaluate_sensory_drift"
                ),
                require_module_attr("genome.catalogue", "load_default_catalogue"),
                require_module_attr(
                    "evolution.lineage_telemetry", "EvolutionLineageSnapshot"
                ),
                require_module_attr(
                    "orchestration.evolution_cycle", "EvolutionCycleOrchestrator"
                ),
                require_module_attr(
                    "operations.evolution_experiments",
                    "evaluate_evolution_experiments",
                ),
                require_module_attr(
                    "operations.evolution_tuning",
                    "evaluate_evolution_tuning",
                ),
            ),
        ),
        StreamDefinition(
            stream="Stream C – Execution, risk, compliance, ops readiness",
            ready_summary=(
                "FIX pilots, risk/compliance workflows, ROI telemetry, and operational readiness "
                "publish evidence for operators."
            ),
            attention_summary=(
                "Execution, risk, or compliance telemetry is missing; ensure FIX pilots, compliance "
                "evaluators, ROI instrumentation, and readiness feeds exist."
            ),
            next_checkpoint=(
                "Expand broker connectivity with drop-copy reconciliation and extend regulatory telemetry coverage."
            ),
            requirements=(
                require_module_attr("runtime.fix_pilot", "FixIntegrationPilot"),
                require_module_attr("operations.fix_pilot", "evaluate_fix_pilot"),
                require_module_attr("runtime.fix_dropcopy", "FixDropcopyReconciler"),
                require_module_attr("operations.execution", "evaluate_execution_readiness"),
                require_module_attr(
                    "operations.professional_readiness", "evaluate_professional_readiness"
                ),
                require_module_attr("operations.roi", "evaluate_roi_posture"),
                require_module_attr(
                    "operations.strategy_performance", "evaluate_strategy_performance"
                ),
                require_module_attr(
                    "compliance.workflow", "evaluate_compliance_workflows"
                ),
                require_module_attr(
                    "operations.compliance_readiness", "evaluate_compliance_readiness"
                ),
                require_module_attr("compliance.trade_compliance", "TradeComplianceMonitor"),
                require_module_attr("compliance.kyc", "KycAmlMonitor"),
                require_module_attr(
                    "operations.configuration_audit", "evaluate_configuration_audit"
                ),
                require_module_attr(
                    "operations.system_validation", "evaluate_system_validation"
                ),
                require_module_attr("operations.slo", "evaluate_ingest_slos"),
            ),
        ),
    )


def evaluate_streams() -> list[StreamStatus]:
    """Return computed statuses for the high-impact roadmap streams."""

    root = repo_root()
    return [definition.evaluate(root) for definition in _stream_definitions()]


def format_markdown(statuses: Iterable[StreamStatus]) -> str:
    """Format stream statuses as a Markdown table."""

    lines = [
        "| Stream | Status | Summary | Next checkpoint |",
        "| --- | --- | --- | --- |",
    ]
    for status in statuses:
        lines.append(
            "| {stream} | {status} | {summary} | {next} |".format(
                stream=status.stream,
                status=status.status,
                summary=status.summary.replace("|", "\\|"),
                next=status.next_checkpoint.replace("|", "\\|"),
            )
        )
    return "\n".join(lines)


def format_detail(statuses: Iterable[StreamStatus]) -> str:
    """Return a richer Markdown report for dashboards and docs."""

    lines: list[str] = [
        "# High-impact roadmap status",
        "",
    ]
    for status in statuses:
        lines.extend(
            [
                f"## {status.stream}",
                "",
                f"**Status:** {status.status}",
                "",
                f"**Summary:** {status.summary}",
                "",
                f"**Next checkpoint:** {status.next_checkpoint}",
                "",
            ]
        )

        if status.evidence:
            lines.append("**Evidence:**")
            lines.extend(f"- {item}" for item in status.evidence)
            lines.append("")

        if status.missing:
            lines.append("**Missing:**")
            lines.extend(f"- {item}" for item in status.missing)
            lines.append("")

    return "\n".join(lines).rstrip()


def format_json(statuses: Iterable[StreamStatus]) -> str:
    """Return a JSON payload describing the stream statuses.

    The high-impact roadmap report feeds both docs and automation.  External
    tooling previously had to reach for the private ``_format_json`` helper to
    reuse the CLI output.  Exposing this public helper keeps those consumers out
    of the private namespace while preserving a single implementation shared by
    the CLI and tests.
    """

    return json.dumps([asdict(status) for status in statuses], indent=2)


def _format_json(statuses: Iterable[StreamStatus]) -> str:
    """Backward-compatible alias for :func:`format_json`."""

    return format_json(statuses)


def _inject_summary_table(existing: str, table: str) -> str:
    """Replace the summary table between the configured markers."""

    try:
        start_index = existing.index(SUMMARY_MARKER_START)
    except ValueError as exc:  # pragma: no cover - defensive guard
        raise ValueError(
            "Summary document missing HIGH_IMPACT_SUMMARY start marker"
        ) from exc

    try:
        end_index = existing.index(SUMMARY_MARKER_END, start_index)
    except ValueError as exc:  # pragma: no cover - defensive guard
        raise ValueError(
            "Summary document missing HIGH_IMPACT_SUMMARY end marker"
        ) from exc

    end_marker_index = end_index + len(SUMMARY_MARKER_END)

    before = existing[:start_index]
    after = existing[end_marker_index:]
    replacement = (
        f"{SUMMARY_MARKER_START}\n{table}\n{SUMMARY_MARKER_END}"
    )
    return f"{before}{replacement}{after}"


def refresh_docs(
    statuses: Sequence[StreamStatus] | None = None,
    *,
    summary_path: Path | None = None,
    detail_path: Path | None = None,
) -> None:
    """Refresh the roadmap status documentation files.

    The summary document retains its narrative wrapper and only replaces the
    table that sits between the ``HIGH_IMPACT_SUMMARY`` markers.  The detail
    document is overwritten with the richer report used by dashboards and
    narrative status updates.
    """

    if statuses is None:
        statuses = evaluate_streams()
    else:
        statuses = list(statuses)

    root = repo_root()
    summary_path = summary_path or root / "docs/status/high_impact_roadmap.md"
    detail_path = detail_path or root / "docs/status/high_impact_roadmap_detail.md"

    summary_text = summary_path.read_text(encoding="utf-8")
    table = format_markdown(statuses)
    updated_summary = _inject_summary_table(summary_text, table)
    if not updated_summary.endswith("\n"):
        updated_summary = f"{updated_summary}\n"
    summary_path.write_text(updated_summary, encoding="utf-8")

    detail_text = format_detail(statuses)
    if not detail_text.endswith("\n"):
        detail_text = f"{detail_text}\n"
    detail_path.write_text(detail_text, encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the high-impact roadmap CLI."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--format",
        choices=("markdown", "json", "detail"),
        default="markdown",
        help="Output format (default: markdown table)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path where the report should be written",
    )
    parser.add_argument(
        "--refresh-docs",
        action="store_true",
        help="Update the status documentation files in docs/status/",
    )
    args = parser.parse_args(argv)

    statuses = evaluate_streams()
    if args.refresh_docs:
        refresh_docs(statuses)
    if args.format == "json":
        output = format_json(statuses)
    elif args.format == "detail":
        output = format_detail(statuses)
    else:
        output = format_markdown(statuses)

    if args.output:
        args.output.write_text(output, encoding="utf-8")

    print(output)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI wrapper
    raise SystemExit(main())
