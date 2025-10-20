"""Counterfactual simulator for belief robustness assessments.

This module pairs the :class:`~src.understanding.causal_graph_engine.CausalGraphEngine`
with belief snapshots so the understanding loop can exercise "do" interventions
and observe how downstream metrics respond.  The simulator focuses on a compact
API that returns structured results suitable for diagnostics, guardrails, and
auditing reports.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Mapping, MutableMapping, Sequence

from .causal_graph_engine import CausalGraphEngine, CausalInterventionResult
from .router import BeliefSnapshot

_DEFAULT_CONTEXT_VALUES: Mapping[str, float] = {
    "macro_signal": 0.0,
    "base_liquidity": 1.0,
    "liquidity_shock": 0.0,
    "macro_to_liquidity_beta": 0.6,
    "mid_price": 0.0,
    "order_imbalance": 0.0,
    "microprice_sensitivity": 0.5,
    "spread": 0.01,
    "limit_price": 0.0,
    "fill_urgency": 0.0,
    "order_size": 1.0,
}


@dataclass(frozen=True, slots=True)
class CounterfactualScenario:
    """Scenario describing a causal intervention to evaluate."""

    name: str
    interventions: Mapping[str, float]
    description: str = ""

    def as_dict(self) -> Mapping[str, object]:
        return {
            "name": self.name,
            "description": self.description,
            "interventions": {key: float(value) for key, value in self.interventions.items()},
        }


@dataclass(frozen=True, slots=True)
class CounterfactualScenarioResult:
    """Outcome bundle for a single counterfactual scenario."""

    scenario: CounterfactualScenario
    intervention: CausalInterventionResult
    max_abs_delta: float
    robust: bool

    def as_dict(self) -> Mapping[str, object]:
        return {
            "scenario": self.scenario.as_dict(),
            "baseline": dict(self.intervention.baseline),
            "intervened": dict(self.intervention.intervened),
            "delta": dict(self.intervention.delta),
            "affected_nodes": tuple(self.intervention.affected_nodes),
            "max_abs_delta": self.max_abs_delta,
            "robust": self.robust,
        }


@dataclass(frozen=True, slots=True)
class CounterfactualAssessment:
    """Aggregated result across all counterfactual scenarios."""

    baseline: Mapping[str, float]
    scenarios: tuple[CounterfactualScenarioResult, ...]
    tolerance: float
    robust: bool
    metadata: Mapping[str, object]

    def as_dict(self) -> Mapping[str, object]:
        return {
            "baseline": dict(self.baseline),
            "tolerance": self.tolerance,
            "robust": self.robust,
            "metadata": dict(self.metadata),
            "scenarios": [result.as_dict() for result in self.scenarios],
        }


class CounterfactualSimulator:
    """Simulate counterfactual interventions to probe belief robustness."""

    def __init__(
        self,
        causal_engine: CausalGraphEngine,
        *,
        tolerance: float = 0.05,
        context_resolver: Callable[[BeliefSnapshot | Mapping[str, Any]], Mapping[str, float]] | None = None,
        context_defaults: Mapping[str, float] | None = None,
    ) -> None:
        if tolerance < 0.0:
            raise ValueError("tolerance must be non-negative")
        self._engine = causal_engine
        self._tolerance = float(tolerance)
        defaults: MutableMapping[str, float] = dict(_DEFAULT_CONTEXT_VALUES)
        if context_defaults:
            for key, value in context_defaults.items():
                defaults[str(key)] = float(value)
        self._defaults = defaults
        self._context_resolver = context_resolver

    def simulate(
        self,
        subject: BeliefSnapshot | Mapping[str, Any] | None,
        scenarios: Sequence[CounterfactualScenario],
        *,
        context_overrides: Mapping[str, float] | None = None,
    ) -> CounterfactualAssessment:
        """Run the supplied scenarios and return an assessment bundle."""

        context = self._build_context(subject)
        if context_overrides:
            for key, value in context_overrides.items():
                context[str(key)] = float(value)

        baseline = self._engine.evaluate(context)
        scenario_results = []
        for scenario in scenarios:
            interventions = {str(name): float(value) for name, value in scenario.interventions.items()}
            intervention_result = self._engine.run_intervention(context, interventions)
            max_abs_delta = _max_abs_delta(intervention_result.delta)
            scenario_results.append(
                CounterfactualScenarioResult(
                    scenario=scenario,
                    intervention=intervention_result,
                    max_abs_delta=max_abs_delta,
                    robust=max_abs_delta <= self._tolerance,
                )
            )

        metadata = self._build_metadata(subject, context, len(scenarios))
        overall_robust = all(result.robust for result in scenario_results)
        return CounterfactualAssessment(
            baseline=baseline,
            scenarios=tuple(scenario_results),
            tolerance=self._tolerance,
            robust=overall_robust,
            metadata=metadata,
        )

    def _build_context(
        self,
        subject: BeliefSnapshot | Mapping[str, Any] | None,
    ) -> dict[str, float]:
        if self._context_resolver is not None:
            resolved = dict(self._context_resolver(subject))  # type: ignore[arg-type]
        else:
            resolved = _default_context(subject, self._defaults)
        context: dict[str, float] = {}
        for key, default in self._defaults.items():
            value = resolved.get(key)
            if value is None:
                context[key] = float(default)
            else:
                context[key] = float(value)
        for key, value in resolved.items():
            if key not in context:
                context[key] = float(value)
        return context

    def _build_metadata(
        self,
        subject: BeliefSnapshot | Mapping[str, Any] | None,
        context: Mapping[str, float],
        scenario_count: int,
    ) -> Mapping[str, object]:
        payload: MutableMapping[str, object] = {
            "scenario_count": int(scenario_count),
            "engine_topology": tuple(self._engine.topology),
            "context_keys": tuple(sorted(context.keys())),
        }

        if isinstance(subject, BeliefSnapshot):
            payload["belief_id"] = subject.belief_id
            payload["regime"] = subject.regime_state.regime
            payload["regime_confidence"] = float(subject.regime_state.confidence)
            payload["regime_timestamp"] = subject.regime_state.timestamp.isoformat()
        elif isinstance(subject, Mapping):
            belief_id = subject.get("belief_id")
            if belief_id is not None:
                payload["belief_id"] = str(belief_id)
            timestamp = subject.get("timestamp")
            if isinstance(timestamp, datetime):
                payload["timestamp"] = timestamp.isoformat()

        return payload


def _default_context(
    subject: BeliefSnapshot | Mapping[str, Any] | None,
    defaults: Mapping[str, float],
) -> dict[str, float]:
    sources: tuple[Mapping[str, Any], ...]
    if isinstance(subject, BeliefSnapshot):
        metadata = subject.metadata if isinstance(subject.metadata, Mapping) else {}
        sources = (
            subject.features,
            metadata,
        )
    elif isinstance(subject, Mapping):
        features = subject.get("features") if isinstance(subject.get("features"), Mapping) else {}
        metadata = subject.get("metadata") if isinstance(subject.get("metadata"), Mapping) else {}
        sources = (
            subject,
            features,  # type: ignore[arg-type]
            metadata,  # type: ignore[arg-type]
        )
    else:
        sources = ()

    lookup: MutableMapping[str, float] = {}
    for source in sources:
        for key, value in source.items():
            numeric = _coerce_float(value)
            if numeric is None:
                continue
            lookup[str(key).lower()] = numeric

    context: dict[str, float] = {}
    for key, default in defaults.items():
        lowered = key.lower()
        aliases = (
            lowered,
            lowered.replace("_", ""),
            f"why_{lowered}",
            f"how_{lowered}",
            f"what_{lowered}",
            f"when_{lowered}",
        )
        value = None
        for alias in aliases:
            value = lookup.get(alias)
            if value is not None:
                break
        context[key] = float(default if value is None else value)
    return context


def _coerce_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if numeric != numeric:  # NaN check
        return None
    return numeric


def _max_abs_delta(delta: Mapping[str, float]) -> float:
    if not delta:
        return 0.0
    return max(abs(float(value)) for value in delta.values())


__all__ = [
    "CounterfactualSimulator",
    "CounterfactualScenario",
    "CounterfactualScenarioResult",
    "CounterfactualAssessment",
]

