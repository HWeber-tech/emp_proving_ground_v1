"""Causal graph engine for macro → liquidity → microprice → fills."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Iterable, Mapping


Parents = Mapping[str, float]
Context = Mapping[str, object]


@dataclass(slots=True, frozen=True)
class CausalNode:
    """Structural equation node within a directed acyclic graph."""

    name: str
    parents: tuple[str, ...]
    compute: Callable[[Parents, Context], float]
    description: str = ""


@dataclass(slots=True, frozen=True)
class CausalInterventionResult:
    """Baseline versus intervened trajectories for a causal graph run."""

    baseline: Mapping[str, float]
    intervened: Mapping[str, float]
    delta: Mapping[str, float]
    affected_nodes: tuple[str, ...]


class CausalGraphEngine:
    """Evaluate structural causal models and run interventions."""

    def __init__(self, nodes: Iterable[CausalNode]):
        node_list = list(nodes)
        if not node_list:
            raise ValueError("causal graph requires at least one node")

        self._nodes: dict[str, CausalNode] = {}
        for node in node_list:
            if node.name in self._nodes:
                raise ValueError(f"duplicate node detected: {node.name}")
            self._nodes[node.name] = node

        for node in node_list:
            missing = [parent for parent in node.parents if parent not in self._nodes]
            if missing:
                joined = ", ".join(missing)
                raise ValueError(f"unknown parents for {node.name}: {joined}")

        self._order = self._resolve_topology()
        self._edges = tuple(
            (parent, node.name)
            for node in self._nodes.values()
            for parent in node.parents
        )

    @classmethod
    def default(cls) -> "CausalGraphEngine":
        """Return the canonical macro → liquidity → microprice → fills DAG."""

        nodes = (
            CausalNode(
                name="macro",
                parents=(),
                compute=_macro_signal,
                description="Macro factor input from higher-level telemetry.",
            ),
            CausalNode(
                name="liquidity",
                parents=("macro",),
                compute=_liquidity_response,
                description="Order book liquidity response to macro shifts.",
            ),
            CausalNode(
                name="microprice",
                parents=("liquidity",),
                compute=_microprice_response,
                description="Microprice adjustment driven by liquidity conditions.",
            ),
            CausalNode(
                name="fills",
                parents=("microprice",),
                compute=_fill_probability,
                description="Fill probability given limit price versus microprice.",
            ),
        )
        return cls(nodes)

    def evaluate(
        self,
        context: Context,
        *,
        interventions: Mapping[str, float] | None = None,
    ) -> dict[str, float]:
        """Evaluate the graph using the provided context and optional interventions."""

        interventions = interventions or {}
        result: dict[str, float] = {}
        for name in self._order:
            if name in interventions:
                result[name] = float(interventions[name])
                continue

            node = self._nodes[name]
            parent_values = {parent: result[parent] for parent in node.parents}
            value = node.compute(parent_values, context)
            result[name] = float(value)
        return result

    def run_intervention(
        self,
        context: Context,
        interventions: Mapping[str, float],
    ) -> CausalInterventionResult:
        """Compute baseline versus intervened trajectories for the supplied changes."""

        baseline = self.evaluate(context)
        intervened = self.evaluate(context, interventions=interventions)
        delta = {
            name: intervened[name] - baseline[name]
            for name in baseline
        }
        affected = tuple(
            name
            for name, diff in delta.items()
            if not math.isclose(diff, 0.0, rel_tol=1e-9, abs_tol=1e-9)
        )
        return CausalInterventionResult(
            baseline=baseline,
            intervened=intervened,
            delta=delta,
            affected_nodes=affected,
        )

    @property
    def edges(self) -> tuple[tuple[str, str], ...]:
        """Return the directed edges for the graph."""

        return self._edges

    @property
    def topology(self) -> tuple[str, ...]:
        """Return the topological order used during evaluations."""

        return self._order

    def _resolve_topology(self) -> tuple[str, ...]:
        indegree = {name: len(node.parents) for name, node in self._nodes.items()}
        ready = [name for name, degree in indegree.items() if degree == 0]
        order: list[str] = []

        while ready:
            name = ready.pop()
            order.append(name)
            for child, child_node in self._nodes.items():
                if name not in child_node.parents:
                    continue
                indegree[child] -= 1
                if indegree[child] == 0:
                    ready.append(child)

        if len(order) != len(self._nodes):
            raise ValueError("cycle detected in causal graph definition")
        return tuple(order)


def _macro_signal(_: Parents, context: Context) -> float:
    """Macro node expects a ``macro_signal`` context entry."""

    try:
        value = context["macro_signal"]
    except KeyError as exc:  # pragma: no cover - defensive, covered by tests
        raise KeyError("context missing 'macro_signal' for macro node") from exc
    return float(value)


def _liquidity_response(parents: Parents, context: Context) -> float:
    """Liquidity reacts to macro shifts with optional shock adjustments."""

    macro_signal = parents["macro"]
    base = float(context.get("base_liquidity", 1.0))
    shock = float(context.get("liquidity_shock", 0.0))
    beta = float(context.get("macro_to_liquidity_beta", 0.6))
    liquidity = base * (1.0 + beta * macro_signal) - shock
    return max(liquidity, 0.0)


def _microprice_response(parents: Parents, context: Context) -> float:
    """Microprice adjusts from mid-price under liquidity and imbalance."""

    liquidity = parents["liquidity"]
    try:
        mid_price = float(context["mid_price"])
    except KeyError as exc:  # pragma: no cover - defensive, covered by tests
        raise KeyError("context missing 'mid_price' for microprice node") from exc
    imbalance = float(context.get("order_imbalance", 0.0))
    spread = max(float(context.get("spread", 0.01)), 1e-9)
    sensitivity = float(context.get("microprice_sensitivity", 0.5))
    liquidity_effect = 1.0 / (1.0 + max(liquidity, 0.0))
    adjustment = imbalance * spread * sensitivity * liquidity_effect
    return mid_price + adjustment


def _fill_probability(parents: Parents, context: Context) -> float:
    """Compute expected fills using a logistic response to price edge."""

    microprice = parents["microprice"]
    try:
        limit_price = float(context["limit_price"])
    except KeyError as exc:  # pragma: no cover - defensive, covered by tests
        raise KeyError("context missing 'limit_price' for fills node") from exc
    spread = max(float(context.get("spread", 0.01)), 1e-9)
    urgency = float(context.get("fill_urgency", 0.0))
    order_size = float(context.get("order_size", 1.0))
    edge = (limit_price - microprice) / spread + urgency
    probability = 1.0 / (1.0 + math.exp(-edge))
    return order_size * probability

