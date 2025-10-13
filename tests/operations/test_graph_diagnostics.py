from datetime import datetime, timezone

import pytest

from src.operations.graph_diagnostics import (
    GraphEvaluation,
    GraphHealthStatus,
    GraphMetrics,
    GraphThresholds,
    compute_graph_metrics,
    evaluate_graph_metrics,
)
from src.understanding.diagnostics import (
    UnderstandingEdge,
    UnderstandingGraphDiagnostics,
    UnderstandingGraphStatus,
    UnderstandingNode,
    UnderstandingNodeKind,
)


@pytest.fixture
def sample_graph() -> UnderstandingGraphDiagnostics:
    generated_at = datetime(2025, 1, 1, tzinfo=timezone.utc)

    sensory = UnderstandingNode(
        node_id="sensory",
        name="Sensory",
        kind=UnderstandingNodeKind.sensory,
        status=UnderstandingGraphStatus.ok,
    )
    belief = UnderstandingNode(
        node_id="belief",
        name="Belief",
        kind=UnderstandingNodeKind.belief,
        status=UnderstandingGraphStatus.ok,
    )
    router = UnderstandingNode(
        node_id="router",
        name="Router",
        kind=UnderstandingNodeKind.router,
        status=UnderstandingGraphStatus.ok,
    )
    policy = UnderstandingNode(
        node_id="policy",
        name="Policy",
        kind=UnderstandingNodeKind.policy,
        status=UnderstandingGraphStatus.ok,
    )

    edges = (
        UnderstandingEdge(source="sensory", target="belief", relationship="feeds"),
        UnderstandingEdge(source="belief", target="router", relationship="conditions"),
        UnderstandingEdge(source="router", target="policy", relationship="selects"),
    )

    return UnderstandingGraphDiagnostics(
        status=UnderstandingGraphStatus.ok,
        nodes=(sensory, belief, router, policy),
        edges=edges,
        generated_at=generated_at,
    )


def test_compute_graph_metrics_chain(sample_graph: UnderstandingGraphDiagnostics) -> None:
    metrics = compute_graph_metrics(sample_graph)

    assert isinstance(metrics, GraphMetrics)
    assert metrics.node_degrees == {"sensory": 1, "belief": 2, "router": 2, "policy": 1}
    assert metrics.degree_histogram == {"1": 2, "2": 2}
    assert metrics.average_degree == pytest.approx(1.5)
    assert metrics.core_nodes == ("belief", "router")
    assert metrics.periphery_nodes == ("policy", "sensory")
    assert metrics.core_ratio == pytest.approx(0.5)
    assert metrics.modularity == pytest.approx(1.0 / 6.0, abs=1e-6)


def test_evaluate_graph_metrics_thresholds(sample_graph: UnderstandingGraphDiagnostics) -> None:
    metrics = compute_graph_metrics(sample_graph)
    thresholds = GraphThresholds(
        min_average_degree=1.0,
        min_modularity=0.2,
        min_core_ratio=0.6,
        max_core_ratio=0.8,
    )

    evaluation = evaluate_graph_metrics(metrics, thresholds)

    assert isinstance(evaluation, GraphEvaluation)
    assert evaluation.status is GraphHealthStatus.warn
    assert any("Modularity" in message for message in evaluation.messages)
    assert any("Core ratio" in message for message in evaluation.messages)


def test_evaluate_graph_metrics_ok(sample_graph: UnderstandingGraphDiagnostics) -> None:
    metrics = compute_graph_metrics(sample_graph)
    thresholds = GraphThresholds(
        min_average_degree=1.0,
        min_modularity=0.0,
        min_core_ratio=0.4,
        max_core_ratio=0.8,
    )

    evaluation = evaluate_graph_metrics(metrics, thresholds)

    assert evaluation.status is GraphHealthStatus.ok
    assert evaluation.messages == ()


def test_evaluate_graph_metrics_failure(sample_graph: UnderstandingGraphDiagnostics) -> None:
    metrics = compute_graph_metrics(sample_graph)
    thresholds = GraphThresholds(min_average_degree=2.0)

    evaluation = evaluate_graph_metrics(metrics, thresholds)

    assert evaluation.status is GraphHealthStatus.fail
    assert any("Average degree" in message for message in evaluation.messages)
