from __future__ import annotations

from datetime import datetime

try:  # Python 3.10 compatibility
    from datetime import UTC
except ImportError:  # pragma: no cover - fallback for older runtimes
    from datetime import timezone

    UTC = timezone.utc

from src.understanding import UnderstandingDiagnosticsBuilder


def test_understanding_graph_serialisation_roundtrip() -> None:
    builder = UnderstandingDiagnosticsBuilder(now=lambda: datetime(2024, 1, 3, tzinfo=UTC))
    artifacts = builder.build()

    graph_dict = artifacts.graph.as_dict()
    assert graph_dict["status"] == "ok"
    assert graph_dict["metadata"]["decision_id"] == artifacts.decision.tactic_id
    assert len(graph_dict["nodes"]) == 4

    dot = artifacts.graph.to_dot()
    assert "UnderstandingLoop" in dot
    assert "sensory" in dot

    markdown = artifacts.graph.to_markdown()
    assert "Understanding loop" not in markdown
    assert "Sensory cortex" in markdown
