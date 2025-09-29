"""Helpers for documenting data lineage and quality service levels."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
from typing import Iterable


@dataclass(frozen=True)
class DataLineageNode:
    """Description of a dataset within the market data lineage graph."""

    name: str
    layer: str
    description: str
    owners: tuple[str, ...]
    upstream_dependencies: tuple[str, ...]
    downstream_consumers: tuple[str, ...]
    freshness_sla: timedelta
    completeness_target: float
    retention_policy: str
    quality_controls: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    def sorted_quality_controls(self) -> tuple[str, ...]:
        """Return quality controls in stable order for deterministic docs."""

        return tuple(sorted(self.quality_controls))

    def sorted_notes(self) -> tuple[str, ...]:
        """Return notes in stable order for deterministic docs."""

        return tuple(sorted(self.notes))


@dataclass(frozen=True)
class DataLineageDocument:
    """Complete lineage description for a group of datasets."""

    title: str
    summary: str
    nodes: tuple[DataLineageNode, ...] = field(default_factory=tuple)
    assumptions: tuple[str, ...] = ()

    def iter_nodes(self) -> Iterable[DataLineageNode]:
        """Iterate over nodes ordered by layer then name."""

        return iter(sorted(self.nodes, key=lambda node: (node.layer, node.name)))


def _format_timedelta(value: timedelta) -> str:
    total_minutes = int(value.total_seconds() // 60)
    hours, minutes = divmod(total_minutes, 60)
    days, hours = divmod(hours, 24)
    parts: list[str] = []
    if days:
        parts.append(f"{days}d")
    if hours:
        parts.append(f"{hours}h")
    if minutes or not parts:
        parts.append(f"{minutes}m")
    return " ".join(parts)


def build_market_data_lineage() -> DataLineageDocument:
    """Return the canonical lineage for Tier-0 market data ingestion."""

    nodes: list[DataLineageNode] = []

    nodes.append(
        DataLineageNode(
            name="Raw Vendor Snapshots",
            layer="Source",
            description=(
                "Source files fetched directly from Yahoo Finance, Alpha Vantage, and "
                "FRED using provider adapters registered with the multi-source "
                "aggregator. Files are stored under data/vendor/ for replayability."
            ),
            owners=("Data Foundation",),
            upstream_dependencies=(),
            downstream_consumers=("MultiSourceAggregator",),
            freshness_sla=timedelta(minutes=15),
            completeness_target=0.99,
            retention_policy=(
                "Hot storage retains 30 days of tick/interval captures; cold storage "
                "archives monthly bundles for 2 years in artifacts/data/vendor/."
            ),
            quality_controls=(
                "HTTP fetch status and schema checks",
                "Provider-specific throttling alerts",
            ),
            notes=(
                "Sourcing windows align with Encyclopedia Tier-0 free vendor cadence.",
            ),
        )
    )

    nodes.append(
        DataLineageNode(
            name="Normalised OHLCV Bars",
            layer="Ingestion",
            description=(
                "Canonical bar set returned by MultiSourceAggregator after column "
                "normalisation, timezone harmonisation, and symbol reconciliation."
            ),
            owners=("Data Foundation", "Trading Ops"),
            upstream_dependencies=("Raw Vendor Snapshots",),
            downstream_consumers=(
                "CoverageValidator",
                "CrossSourceDriftValidator",
                "StalenessValidator",
                "PricingPipeline",
            ),
            freshness_sla=timedelta(minutes=20),
            completeness_target=0.995,
            retention_policy=(
                "Rolling 90 days accessible in parquet format under "
                "data_foundation/cache/hot/. Quarterly snapshots compressed into "
                "artifacts/data/cache/ for encyclopaedia-aligned cold retention."
            ),
            quality_controls=(
                "Column schema enforcement",
                "Timezone coercion to UTC",
                "Symbol canonicalisation",
            ),
            notes=(
                "Serves as the baseline dataset for pricing pipelines and sensors.",
            ),
        )
    )

    nodes.append(
        DataLineageNode(
            name="Quality Diagnostics",
            layer="Validation",
            description=(
                "Structured findings emitted by Coverage-, Drift-, and Staleness-"
                "validators in src.data_foundation.ingest.multi_source. Results "
                "drive alerting pipelines and reconciliation dashboards."
            ),
            owners=("Data Reliability", "Trading Ops"),
            upstream_dependencies=("Normalised OHLCV Bars",),
            downstream_consumers=(
                "Risk Analytics",
                "PnL Dashboard",
                "Data Foundation Runbooks",
            ),
            freshness_sla=timedelta(minutes=25),
            completeness_target=1.0,
            retention_policy=(
                "Last 180 days stored in structured JSON under artifacts/data/quality/. "
                "Findings older than 180 days summarised into monthly markdown reports."
            ),
            quality_controls=(
                "Severity escalation thresholds",
                "Missing-bar tolerance checks",
                "Latency watermark comparisons",
            ),
            notes=(
                "Validators cover Tier-0/Tier-1 encyclopedia requirements.",
            ),
        )
    )

    nodes.append(
        DataLineageNode(
            name="Sensor Feature Store",
            layer="Analytic",
            description=(
                "Feature parquet files produced by HOW/WHEN/WHY sensory organs after "
                "ingesting normalised bars and macro calendars. Serves both "
                "strategies and risk sizing."
            ),
            owners=("Sensory Team",),
            upstream_dependencies=(
                "Normalised OHLCV Bars",
                "Quality Diagnostics",
                "Macro Calendar Snapshots",
            ),
            downstream_consumers=(
                "Strategy Backtests",
                "Evolution Lab",
                "Risk Scenario Runner",
            ),
            freshness_sla=timedelta(minutes=30),
            completeness_target=0.98,
            retention_policy=(
                "Intraday features retained for 14 days; end-of-day aggregates kept for "
                "1 year to support encyclopedia Tier-1 audits."
            ),
            quality_controls=(
                "Sensor drift monitoring",
                "Schema fingerprinting",
                "Cross-sensor dependency checks",
            ),
            notes=(
                "Feature definitions documented via tools.sensory.registry exporter.",
            ),
        )
    )

    nodes.append(
        DataLineageNode(
            name="Macro Calendar Snapshots",
            layer="Source",
            description=(
                "Economic calendar events curated from open data sources and normalised "
                "into WHY-dimension signals consumed by strategies and risk modules."
            ),
            owners=("Data Foundation",),
            upstream_dependencies=(),
            downstream_consumers=(
                "Sensor Feature Store",
                "Risk Scenario Runner",
            ),
            freshness_sla=timedelta(hours=1),
            completeness_target=0.97,
            retention_policy=(
                "12 months retained to support seasonality studies; older entries archived "
                "into docs/reports/macro/ per encyclopedia data governance guidance."
            ),
            quality_controls=(
                "Duplicate event suppression",
                "Timezone reconciliation",
            ),
            notes=(
                "SLA follows encyclopedia Appendix B macro data timetable.",
            ),
        )
    )

    return DataLineageDocument(
        title="Market Data Lineage & Quality SLA",
        summary=(
            "This document captures the Tier-0/Tier-1 market data flow, owners, service "
            "levels, and retention expectations described in the EMP Encyclopedia. "
            "It is generated programmatically to ensure the documentation stays aligned "
            "with the implementation under src/data_foundation."
        ),
        nodes=tuple(nodes),
        assumptions=(
            "Freshness SLAs assume scheduled runs every 15 minutes within trading hours.",
            "Completeness targets exclude provider-wide outages acknowledged by vendor status pages.",
            "Retention policies inherit from config/data_foundation/retention.yaml when deployed in production.",
        ),
    )


def render_lineage_markdown(document: DataLineageDocument) -> str:
    """Render a lineage document to Markdown."""

    lines: list[str] = []
    lines.append(f"# {document.title}")
    lines.append("")
    lines.append(
        "_Auto-generated via `scripts/generate_data_lineage.py`; do not edit manually._"
    )
    lines.append("")
    lines.append(document.summary)
    lines.append("")

    if document.assumptions:
        lines.append("## Assumptions")
        for assumption in document.assumptions:
            lines.append(f"- {assumption}")
        lines.append("")

    lines.append("## Service Levels by Dataset")
    lines.append("")
    lines.append("| Layer | Dataset | Freshness SLA | Completeness Target | Retention | Owners |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for node in document.iter_nodes():
        lines.append(
            "| {layer} | {name} | {freshness} | {completeness:.1%} | {retention} | {owners} |".format(
                layer=node.layer,
                name=node.name,
                freshness=_format_timedelta(node.freshness_sla),
                completeness=node.completeness_target,
                retention=node.retention_policy,
                owners=", ".join(node.owners),
            )
        )
    lines.append("")

    for node in document.iter_nodes():
        lines.append(f"### {node.name}")
        lines.append("")
        lines.append(node.description)
        lines.append("")
        lines.append("- **Layer:** {layer}".format(layer=node.layer))
        lines.append(
            "- **Upstream:** {upstream}".format(
                upstream=", ".join(node.upstream_dependencies) or "(root)"
            )
        )
        lines.append(
            "- **Downstream:** {downstream}".format(
                downstream=", ".join(node.downstream_consumers) or "(leaf)"
            )
        )
        lines.append(
            "- **Freshness SLA:** {freshness}".format(
                freshness=_format_timedelta(node.freshness_sla)
            )
        )
        lines.append(
            "- **Completeness Target:** {completeness:.1%}".format(
                completeness=node.completeness_target
            )
        )
        lines.append(f"- **Retention:** {node.retention_policy}")
        if node.quality_controls:
            lines.append("- **Quality Controls:**")
            for control in node.sorted_quality_controls():
                lines.append(f"  - {control}")
        if node.notes:
            lines.append("- **Notes:**")
            for note in node.sorted_notes():
                lines.append(f"  - {note}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


__all__ = [
    "DataLineageDocument",
    "DataLineageNode",
    "build_market_data_lineage",
    "render_lineage_markdown",
]
