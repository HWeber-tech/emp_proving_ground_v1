"""Tests for the data lineage documentation helpers."""

from __future__ import annotations

from datetime import timedelta

from src.data_foundation.documentation import (
    DataLineageDocument,
    DataLineageNode,
    build_market_data_lineage,
    render_lineage_markdown,
)


def test_build_market_data_lineage_returns_expected_nodes() -> None:
    document = build_market_data_lineage()
    names = {node.name for node in document.nodes}
    assert {
        "Raw Vendor Snapshots",
        "Normalised OHLCV Bars",
        "Quality Diagnostics",
        "Sensor Feature Store",
        "Macro Calendar Snapshots",
    } <= names


def test_render_lineage_markdown_includes_table() -> None:
    document = DataLineageDocument(
        title="Example",
        summary="Summary",
        nodes=(
            DataLineageNode(
                name="Dataset",
                layer="Layer",
                description="Example dataset",
                owners=("Owner",),
                upstream_dependencies=(),
                downstream_consumers=(),
                freshness_sla=timedelta(minutes=5),
                completeness_target=0.9,
                retention_policy="7 days",
                quality_controls=("Check",),
                notes=("Note",),
            ),
        ),
    )
    markdown = render_lineage_markdown(document)
    assert "| Layer | Dataset |" in markdown
    assert "7d" not in markdown  # ensure formatting respects minutes for sub-day values
    assert "5m" in markdown
    assert "- **Quality Controls:**" in markdown
