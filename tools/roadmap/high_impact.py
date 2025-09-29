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
    require_path,
)


SUMMARY_MARKER_START = "<!-- HIGH_IMPACT_SUMMARY:START -->"
SUMMARY_MARKER_END = "<!-- HIGH_IMPACT_SUMMARY:END -->"
PORTFOLIO_MARKER_START = "<!-- HIGH_IMPACT_PORTFOLIO:START -->"
PORTFOLIO_MARKER_END = "<!-- HIGH_IMPACT_PORTFOLIO:END -->"


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


@dataclass(frozen=True)
class PortfolioStatus:
    """Aggregate view across the high-impact roadmap streams."""

    total_streams: int
    ready: int
    attention_needed: int
    streams: tuple[StreamStatus, ...]

    def ready_streams(self) -> tuple[StreamStatus, ...]:
        """Return the streams marked as ready."""

        return tuple(status for status in self.streams if status.status == "Ready")

    def attention_streams(self) -> tuple[StreamStatus, ...]:
        """Return the streams still needing attention."""

        return tuple(status for status in self.streams if status.status != "Ready")

    @property
    def all_ready(self) -> bool:
        """Return ``True`` when every stream has passed its requirements."""

        return self.attention_needed == 0

    def missing_requirements(self) -> dict[str, tuple[str, ...]]:
        """Map stream names to the outstanding requirement labels."""

        return {
            status.stream: status.missing
            for status in self.attention_streams()
            if status.missing
        }

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable representation of the portfolio."""

        return {
            "total_streams": self.total_streams,
            "ready": self.ready,
            "attention_needed": self.attention_needed,
            "streams": [asdict(status) for status in self.streams],
            "ready_streams": [status.stream for status in self.ready_streams()],
            "attention_streams": [
                status.stream for status in self.attention_streams()
            ],
            "missing_requirements": {
                status.stream: list(status.missing)
                for status in self.attention_streams()
            },
        }


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
                    "data_foundation.ingest.anomaly_detection",
                    "detect_feed_anomalies",
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
                    "data_foundation.streaming.latency_benchmark",
                    "StreamingLatencyBenchmark",
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
                    "operations.cross_region_failover",
                    "evaluate_cross_region_failover",
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
                require_module_attr(
                    "operations.event_bus_health", "evaluate_event_bus_health"
                ),
                require_module_attr(
                    "operations.failover_drill", "execute_failover_drill"
                ),
                require_module_attr(
                    "operations.alerts", "build_default_alert_manager"
                ),
                require_module_attr(
                    "risk.analytics.var", "compute_parametric_var"
                ),
                require_module_attr(
                    "risk.analytics.expected_shortfall",
                    "compute_historical_expected_shortfall",
                ),
                require_module_attr(
                    "risk.analytics.volatility_target",
                    "determine_target_allocation",
                ),
                require_module_attr(
                    "risk.analytics.volatility_regime",
                    "classify_volatility_regime",
                ),
                require_module_attr(
                    "trading.order_management.lifecycle_processor",
                    "OrderLifecycleProcessor",
                ),
                require_module_attr(
                    "trading.order_management.position_tracker", "PositionTracker"
                ),
                require_module_attr(
                    "trading.order_management.event_journal", "OrderEventJournal"
                ),
                require_module_attr(
                    "trading.order_management.reconciliation", "replay_order_events"
                ),
                require_module_attr(
                    "trading.execution.market_regime", "classify_market_regime"
                ),
                require_path("scripts/order_lifecycle_dry_run.py"),
                require_path("scripts/reconcile_positions.py"),
                require_path("scripts/generate_risk_report.py"),
                require_path("docs/runbooks/execution_lifecycle.md"),
            ),
        ),
    )


def evaluate_streams(selected_streams: Sequence[str] | None = None) -> list[StreamStatus]:
    """Return computed statuses for the high-impact roadmap streams.

    Parameters
    ----------
    selected_streams:
        Optional iterable of stream names to evaluate.  When omitted, all
        streams are evaluated.  The names must match the ``stream`` field of the
        :class:`StreamDefinition` entries.
    """

    definitions = list(_stream_definitions())
    if selected_streams:
        cleaned = [
            name.strip()
            for name in selected_streams
            if name and name.strip()
        ]
        if not cleaned:
            definitions = []
        else:
            ordered_names: list[str] = []
            seen: set[str] = set()
            for name in cleaned:
                if name not in seen:
                    ordered_names.append(name)
                    seen.add(name)

            known = {definition.stream for definition in definitions}
            unknown = [name for name in ordered_names if name not in known]
            if unknown:
                raise ValueError(
                    "Unknown stream(s): " + ", ".join(unknown)
                )

            definitions_by_name = {
                definition.stream: definition for definition in definitions
            }
            definitions = [definitions_by_name[name] for name in ordered_names]

    root = repo_root()
    return [definition.evaluate(root) for definition in definitions]


def summarise_portfolio(statuses: Iterable[StreamStatus]) -> PortfolioStatus:
    """Return aggregate status counts for the high-impact roadmap portfolio."""

    stream_statuses = tuple(statuses)
    ready = sum(1 for status in stream_statuses if status.status == "Ready")
    attention = len(stream_statuses) - ready
    return PortfolioStatus(
        total_streams=len(stream_statuses),
        ready=ready,
        attention_needed=attention,
        streams=stream_statuses,
    )


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


def format_portfolio_summary(statuses: Iterable[StreamStatus]) -> str:
    """Return a narrative summary of the overall portfolio health."""

    portfolio = summarise_portfolio(statuses)
    lines = [
        "# High-impact roadmap summary",
        "",
        f"- Total streams: {portfolio.total_streams}",
        f"- Ready: {portfolio.ready}",
        f"- Attention needed: {portfolio.attention_needed}",
    ]

    if portfolio.all_ready:
        lines.extend([
            "",
            "All streams are Ready; no missing requirements.",
        ])
    elif portfolio.attention_streams():
        lines.extend([
            "",
            "Streams needing attention:",
        ])
        for status in portfolio.attention_streams():
            missing_count = len(status.missing)
            lines.append(
                f"- {status.stream} ({missing_count} missing requirement"
                f"{'s' if missing_count != 1 else ''})"
            )
            for requirement in status.missing:
                lines.append(f"  - {requirement}")

    if portfolio.streams:
        lines.extend(["", "## Streams", ""])
        for status in portfolio.streams:
            lines.extend(
                [
                    f"### {status.stream}",
                    "",
                    f"*Status:* {status.status}",
                    f"*Summary:* {status.summary}",
                    f"*Next checkpoint:* {status.next_checkpoint}",
                    "",
                ]
            )

    return "\n".join(lines).rstrip()


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


def format_detail_json(statuses: Iterable[StreamStatus]) -> str:
    """Return a JSON payload with evidence for every stream."""

    stream_statuses = tuple(statuses)
    portfolio = summarise_portfolio(stream_statuses)

    payload = {
        "portfolio": portfolio.as_dict(),
        "streams": [
            {
                "stream": status.stream,
                "status": status.status,
                "summary": status.summary,
                "next_checkpoint": status.next_checkpoint,
                "evidence": list(status.evidence),
                "missing": list(status.missing),
            }
            for status in stream_statuses
        ],
    }

    return json.dumps(payload, indent=2)


def format_attention(statuses: Iterable[StreamStatus]) -> str:
    """Highlight streams that still have missing requirements."""

    lines: list[str] = [
        "# High-impact roadmap attention",
        "",
    ]

    needing_attention = [
        status for status in statuses if status.missing
    ]

    if not needing_attention:
        lines.append("All streams are Ready; no missing requirements.")
        return "\n".join(lines).rstrip()

    for status in needing_attention:
        lines.extend(
            [
                f"## {status.stream}",
                "",
                f"*Status:* {status.status}",
                "",
                "**Missing requirements:**",
            ]
        )
        lines.extend(f"- {item}" for item in status.missing)

        if status.evidence:
            lines.extend(["", "**Evidence captured:**"])
            lines.extend(f"- {item}" for item in status.evidence)

        lines.append("")

    return "\n".join(lines).rstrip()


def format_attention_json(statuses: Iterable[StreamStatus]) -> str:
    """Return a JSON payload focused on attention-required streams."""

    portfolio = summarise_portfolio(statuses)
    attention_streams = [
        {
            "stream": status.stream,
            "status": status.status,
            "summary": status.summary,
            "next_checkpoint": status.next_checkpoint,
            "missing": list(status.missing),
            "evidence": list(status.evidence),
        }
        for status in portfolio.attention_streams()
    ]

    payload = {
        "portfolio": {
            "total_streams": portfolio.total_streams,
            "ready": portfolio.ready,
            "attention_needed": portfolio.attention_needed,
            "all_ready": portfolio.all_ready,
        },
        "streams": attention_streams,
        "missing_requirements": {
            stream: list(requirements)
            for stream, requirements in portfolio.missing_requirements().items()
        },
    }
    return json.dumps(payload, indent=2)


def format_json(statuses: Iterable[StreamStatus]) -> str:
    """Return a JSON payload describing the high-impact roadmap portfolio.

    Earlier versions returned only the per-stream list, forcing downstream
    automation to recompute overall readiness counts.  The roadmap is most
    useful when both the stream evidence and the portfolio roll-up ship
    together, so this formatter now mirrors the :class:`PortfolioStatus`
    structure.  Existing consumers still receive the same stream payload under
    the ``"streams"`` key while gaining the aggregate counts used in dashboards
    and alerts.
    """

    portfolio = summarise_portfolio(statuses)
    portfolio_dict = portfolio.as_dict()
    stream_payload = portfolio_dict.pop("streams", [])
    payload = {
        "portfolio": portfolio_dict,
        "streams": stream_payload,
    }
    return json.dumps(payload, indent=2)


def _format_json(statuses: Iterable[StreamStatus]) -> str:
    """Backward-compatible alias for :func:`format_json`."""

    return format_json(statuses)


def format_portfolio_json(statuses: Iterable[StreamStatus]) -> str:
    """Return a JSON payload describing aggregate portfolio health."""

    portfolio = summarise_portfolio(statuses)
    return json.dumps(portfolio.as_dict(), indent=2)


def _replace_between_markers(
    existing: str, *, start: str, end: str, payload: str
) -> str:
    """Replace the text between two markers with ``payload``."""

    try:
        start_index = existing.index(start)
    except ValueError as exc:  # pragma: no cover - defensive guard
        raise ValueError(
            f"Summary document missing start marker: {start!r}"
        ) from exc

    try:
        end_index = existing.index(end, start_index)
    except ValueError as exc:  # pragma: no cover - defensive guard
        raise ValueError(
            f"Summary document missing end marker: {end!r}"
        ) from exc

    end_marker_index = end_index + len(end)

    before = existing[:start_index]
    after = existing[end_marker_index:]
    replacement = f"{start}\n{payload}\n{end}"
    return f"{before}{replacement}{after}"


def _inject_summary_table(existing: str, table: str) -> str:
    """Replace the summary table between the configured markers."""

    return _replace_between_markers(
        existing,
        start=SUMMARY_MARKER_START,
        end=SUMMARY_MARKER_END,
        payload=table,
    )


def _inject_portfolio_summary(existing: str, summary: str) -> str:
    """Replace the portfolio narrative between the configured markers."""

    return _replace_between_markers(
        existing,
        start=PORTFOLIO_MARKER_START,
        end=PORTFOLIO_MARKER_END,
        payload=summary,
    )


def refresh_docs(
    statuses: Sequence[StreamStatus] | None = None,
    *,
    summary_path: Path | None = None,
    detail_path: Path | None = None,
    attention_path: Path | None = None,
    portfolio_json_path: Path | None = None,
    attention_json_path: Path | None = None,
) -> None:
    """Refresh the roadmap status documentation files.

    The summary document retains its narrative wrapper and only replaces the
    table that sits between the ``HIGH_IMPACT_SUMMARY`` markers.  The detail
    document is overwritten with the richer report used by dashboards and
    narrative status updates.  When JSON destinations are provided (or the
    defaults are used), the helper also emits the portfolio roll-up and the
    attention-focused payload used by dashboards.
    """

    if statuses is None:
        statuses = evaluate_streams()
    else:
        statuses = list(statuses)

    root = repo_root()
    summary_path = summary_path or root / "docs/status/high_impact_roadmap.md"
    detail_path = detail_path or root / "docs/status/high_impact_roadmap_detail.md"
    attention_path = (
        attention_path or root / "docs/status/high_impact_roadmap_attention.md"
    )
    portfolio_json_path = (
        portfolio_json_path
        or root / "docs/status/high_impact_roadmap_portfolio.json"
    )
    attention_json_path = (
        attention_json_path
        or root / "docs/status/high_impact_roadmap_attention.json"
    )

    summary_text = summary_path.read_text(encoding="utf-8")
    table = format_markdown(statuses)
    updated_summary = _inject_summary_table(summary_text, table)
    portfolio_block = format_portfolio_summary(statuses)
    updated_summary = _inject_portfolio_summary(updated_summary, portfolio_block)
    if not updated_summary.endswith("\n"):
        updated_summary = f"{updated_summary}\n"
    summary_path.write_text(updated_summary, encoding="utf-8")

    detail_text = format_detail(statuses)
    if not detail_text.endswith("\n"):
        detail_text = f"{detail_text}\n"
    detail_path.write_text(detail_text, encoding="utf-8")

    attention_text = format_attention(statuses)
    if not attention_text.endswith("\n"):
        attention_text = f"{attention_text}\n"
    attention_path.write_text(attention_text, encoding="utf-8")

    if portfolio_json_path:
        portfolio_payload = format_portfolio_json(statuses)
        if not portfolio_payload.endswith("\n"):
            portfolio_payload = f"{portfolio_payload}\n"
        portfolio_json_path.write_text(portfolio_payload, encoding="utf-8")

    if attention_json_path:
        attention_payload = format_attention_json(statuses)
        if not attention_payload.endswith("\n"):
            attention_payload = f"{attention_payload}\n"
        attention_json_path.write_text(attention_payload, encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the high-impact roadmap CLI."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--format",
        choices=(
            "markdown",
            "json",
            "detail",
            "detail-json",
            "summary",
            "attention",
            "attention-json",
            "portfolio-json",
        ),
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
    parser.add_argument(
        "--summary-path",
        type=Path,
        help=(
            "Optional path to the summary markdown file when refreshing docs. "
            "Defaults to docs/status/high_impact_roadmap.md"
        ),
    )
    parser.add_argument(
        "--detail-path",
        type=Path,
        help=(
            "Optional path to the detailed markdown file when refreshing docs. "
            "Defaults to docs/status/high_impact_roadmap_detail.md"
        ),
    )
    parser.add_argument(
        "--attention-path",
        type=Path,
        help=(
            "Optional path to the attention markdown file when refreshing docs. "
            "Defaults to docs/status/high_impact_roadmap_attention.md"
        ),
    )
    parser.add_argument(
        "--portfolio-json-path",
        type=Path,
        help=(
            "Optional path to the portfolio JSON file when refreshing docs. "
            "Defaults to docs/status/high_impact_roadmap_portfolio.json"
        ),
    )
    parser.add_argument(
        "--attention-json-path",
        type=Path,
        help=(
            "Optional path to the attention JSON file when refreshing docs. "
            "Defaults to docs/status/high_impact_roadmap_attention.json"
        ),
    )
    parser.add_argument(
        "--stream",
        action="append",
        dest="streams",
        help=(
            "Limit evaluation to the given stream name. Provide multiple times to "
            "include more than one stream."
        ),
    )
    args = parser.parse_args(argv)

    if args.refresh_docs and args.streams:
        parser.error(
            "--refresh-docs cannot be combined with --stream; refresh the full "
            "portfolio instead"
        )

    try:
        statuses = tuple(evaluate_streams(args.streams))
    except ValueError as exc:
        parser.error(str(exc))

    portfolio = summarise_portfolio(statuses)
    if args.refresh_docs:
        refresh_docs(
            statuses,
            summary_path=args.summary_path,
            detail_path=args.detail_path,
            attention_path=args.attention_path,
            portfolio_json_path=args.portfolio_json_path,
            attention_json_path=args.attention_json_path,
        )
    if args.format == "json":
        output = format_json(statuses)
    elif args.format == "detail":
        output = format_detail(statuses)
    elif args.format == "detail-json":
        output = format_detail_json(statuses)
    elif args.format == "summary":
        output = format_portfolio_summary(statuses)
    elif args.format == "attention":
        output = format_attention(statuses)
    elif args.format == "attention-json":
        output = format_attention_json(statuses)
    elif args.format == "portfolio-json":
        output = format_portfolio_json(statuses)
    else:
        output = format_markdown(statuses)

    if args.output:
        args.output.write_text(output, encoding="utf-8")

    print(output)
    return 0 if portfolio.all_ready else 1


if __name__ == "__main__":  # pragma: no cover - CLI wrapper
    raise SystemExit(main())
