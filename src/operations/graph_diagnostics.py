"""Graph diagnostics metrics and evaluation helpers for nightly jobs."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from enum import StrEnum
from typing import Iterable, Mapping, Sequence

from src.understanding.diagnostics import (
    UnderstandingEdge,
    UnderstandingGraphDiagnostics,
    UnderstandingNodeKind,
)


class GraphHealthStatus(StrEnum):
    """Health levels emitted by graph diagnostics."""

    ok = "ok"
    warn = "warn"
    fail = "fail"


@dataclass(frozen=True)
class GraphMetrics:
    """Summary metrics derived from an understanding-loop graph."""

    node_degrees: Mapping[str, int]
    degree_histogram: Mapping[str, int]
    average_degree: float
    modularity: float
    core_nodes: Sequence[str]
    periphery_nodes: Sequence[str]
    core_ratio: float
    communities: Mapping[str, str]

    def as_dict(self) -> Mapping[str, object]:
        return {
            "node_degrees": dict(self.node_degrees),
            "degree_histogram": dict(self.degree_histogram),
            "average_degree": round(self.average_degree, 6),
            "modularity": round(self.modularity, 6),
            "core_nodes": list(self.core_nodes),
            "periphery_nodes": list(self.periphery_nodes),
            "core_ratio": round(self.core_ratio, 6),
            "communities": dict(self.communities),
        }


@dataclass(frozen=True)
class GraphThresholds:
    """Thresholds used to evaluate graph health."""

    min_average_degree: float = 1.0
    min_modularity: float = 0.0
    min_core_ratio: float = 0.2
    max_core_ratio: float = 0.75

    def as_dict(self) -> Mapping[str, float]:
        return {
            "min_average_degree": self.min_average_degree,
            "min_modularity": self.min_modularity,
            "min_core_ratio": self.min_core_ratio,
            "max_core_ratio": self.max_core_ratio,
        }


@dataclass(frozen=True)
class GraphEvaluation:
    """Evaluation of graph metrics against configured thresholds."""

    status: GraphHealthStatus
    messages: Sequence[str]
    thresholds: GraphThresholds

    def as_dict(self) -> Mapping[str, object]:
        return {
            "status": self.status.value,
            "messages": list(self.messages),
            "thresholds": self.thresholds.as_dict(),
        }


def compute_graph_metrics(graph: UnderstandingGraphDiagnostics) -> GraphMetrics:
    """Compute graph metrics required by the observability roadmap."""

    node_ids = tuple(node.node_id for node in graph.nodes)
    communities = {
        node.node_id: _resolve_community(node.kind)
        for node in graph.nodes
    }

    undirected_edges = _normalise_edges(graph.edges)
    degree_counter: Counter[str] = Counter({node_id: 0 for node_id in node_ids})
    for source, target in undirected_edges:
        degree_counter[source] += 1
        degree_counter[target] += 1

    total_nodes = len(node_ids)
    degree_histogram = Counter(degree_counter.values())
    histogram = {str(degree): degree_histogram[degree] for degree in sorted(degree_histogram)}

    if total_nodes:
        average_degree = sum(degree_counter.values()) / float(total_nodes)
    else:
        average_degree = 0.0

    modularity = _calculate_modularity(communities, undirected_edges, degree_counter)

    core_threshold = max(1.0, average_degree)
    core_nodes = tuple(sorted(node_id for node_id, degree in degree_counter.items() if degree >= core_threshold))
    periphery_nodes = tuple(sorted(node_id for node_id in node_ids if node_id not in core_nodes))
    core_ratio = (len(core_nodes) / float(total_nodes)) if total_nodes else 0.0

    return GraphMetrics(
        node_degrees=dict(degree_counter),
        degree_histogram=histogram,
        average_degree=average_degree,
        modularity=modularity,
        core_nodes=core_nodes,
        periphery_nodes=periphery_nodes,
        core_ratio=core_ratio,
        communities=communities,
    )


def evaluate_graph_metrics(metrics: GraphMetrics, thresholds: GraphThresholds) -> GraphEvaluation:
    """Evaluate metrics and emit a health status plus advisory messages."""

    messages: list[str] = []
    status = GraphHealthStatus.ok

    if metrics.average_degree < thresholds.min_average_degree:
        messages.append(
            "Average degree {:.3f} fell below the minimum {:.3f}.".format(
                metrics.average_degree, thresholds.min_average_degree
            )
        )
        status = GraphHealthStatus.fail

    if metrics.modularity < thresholds.min_modularity:
        messages.append(
            "Modularity {:.3f} below target {:.3f}; investigate graph sparsity or community leakage.".format(
                metrics.modularity, thresholds.min_modularity
            )
        )
        if status is GraphHealthStatus.ok:
            status = GraphHealthStatus.warn

    if metrics.core_ratio < thresholds.min_core_ratio:
        messages.append(
            "Core ratio {:.3f} under minimum {:.3f}; decision loop may lack stable hubs.".format(
                metrics.core_ratio, thresholds.min_core_ratio
            )
        )
        if status is GraphHealthStatus.ok:
            status = GraphHealthStatus.warn
    elif metrics.core_ratio > thresholds.max_core_ratio:
        messages.append(
            "Core ratio {:.3f} above maximum {:.3f}; graph trending toward over-concentration.".format(
                metrics.core_ratio, thresholds.max_core_ratio
            )
        )
        if status is GraphHealthStatus.ok:
            status = GraphHealthStatus.warn

    if not metrics.core_nodes:
        messages.append("No core nodes detected; verify router and belief hubs remain connected.")
        status = GraphHealthStatus.fail

    return GraphEvaluation(status=status, messages=tuple(messages), thresholds=thresholds)


def _resolve_community(kind: UnderstandingNodeKind) -> str:
    if kind in (UnderstandingNodeKind.sensory, UnderstandingNodeKind.belief):
        return "upstream"
    if kind in (UnderstandingNodeKind.router, UnderstandingNodeKind.policy):
        return "downstream"
    return kind.value


def _normalise_edges(edges: Sequence[UnderstandingEdge]) -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    for edge in edges:
        source = getattr(edge, "source", None)
        target = getattr(edge, "target", None)
        if not isinstance(source, str) or not isinstance(target, str):
            continue
        if source == target:
            continue
        ordered = tuple(sorted((source, target)))
        pairs.add(ordered)
    return pairs


def _calculate_modularity(
    communities: Mapping[str, str],
    edges: Iterable[tuple[str, str]],
    degrees: Mapping[str, int],
) -> float:
    edge_set = {tuple(sorted(edge)) for edge in edges}
    edge_count = len(edge_set)
    if edge_count == 0:
        return 0.0

    nodes = tuple(communities.keys())
    two_m = 2.0 * float(edge_count)
    modularity = 0.0

    for node_i in nodes:
        community_i = communities.get(node_i)
        degree_i = float(degrees.get(node_i, 0))
        for node_j in nodes:
            if communities.get(node_j) != community_i:
                continue
            if node_i == node_j:
                adjacency = 0.0
            else:
                key = tuple(sorted((node_i, node_j)))
                adjacency = 1.0 if key in edge_set else 0.0
            degree_j = float(degrees.get(node_j, 0))
            modularity += adjacency - (degree_i * degree_j) / two_m

    return modularity / two_m


__all__ = [
    "GraphEvaluation",
    "GraphHealthStatus",
    "GraphMetrics",
    "GraphThresholds",
    "compute_graph_metrics",
    "evaluate_graph_metrics",
]
