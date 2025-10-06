from __future__ import annotations

from datetime import datetime

try:  # Python 3.10 compatibility
    from datetime import UTC
except ImportError:  # pragma: no cover - fallback for older runtimes
    from datetime import timezone

    UTC = timezone.utc

import pytest

from src.understanding import (
    UnderstandingDiagnosticsBuilder,
    UnderstandingGraphStatus,
)


pytestmark = [pytest.mark.guardrail, pytest.mark.understanding_acceptance]


def test_understanding_acceptance_cycle_generates_synced_artifacts() -> None:
    builder = UnderstandingDiagnosticsBuilder(now=lambda: datetime(2024, 1, 2, tzinfo=UTC))
    artifacts = builder.build()

    graph = artifacts.graph
    snapshot = artifacts.to_snapshot()

    assert graph.status is UnderstandingGraphStatus.ok
    assert {node.node_id for node in graph.nodes} == {"sensory", "belief", "router", "policy"}
    assert graph.metadata["capsule_id"] == snapshot.capsule.capsule_id

    assert artifacts.decision.tactic_id == artifacts.ledger_diff.policy_id
    assert artifacts.ledger_diff.approvals
    assert not artifacts.drift_summary.exceeded
    assert snapshot.capsule.metadata["decision_id"] == artifacts.decision.tactic_id

    sensory_node = next(node for node in graph.nodes if node.node_id == "sensory")
    assert sensory_node.metadata["drift_exceeded"] is False
