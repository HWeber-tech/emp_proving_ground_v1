"""PolicyRouter for fast-weight tactical experimentation and reflection summaries.

The roadmap's adaptation loop calls for routing tactics based on regime context,
fast-weight experimentation, and automated reviewer-facing summaries so AlphaTrade
operators can understand emerging strategies without combing through telemetry.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Deque, Iterable, Mapping, MutableMapping, Sequence


@dataclass(frozen=True)
class RegimeState:
    """Snapshot of the active market regime recognised by the understanding loop."""

    regime: str
    confidence: float
    features: Mapping[str, float]
    timestamp: datetime

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")


@dataclass(frozen=True)
class PolicyTactic:
    """Tactic configuration with regime-aware preferences."""

    tactic_id: str
    base_weight: float
    parameters: Mapping[str, object] = field(default_factory=dict)
    guardrails: Mapping[str, object] = field(default_factory=dict)
    regime_bias: Mapping[str, float] = field(default_factory=dict)
    confidence_sensitivity: float = 0.5
    description: str | None = None

    def score(self, regime_state: RegimeState) -> tuple[float, MutableMapping[str, float]]:
        """Return a regime-conditioned score and a breakdown for reflection summaries."""

        bias = self.regime_bias.get(regime_state.regime, 1.0)
        # Higher confidence should amplify the score while staying deterministic.
        confidence_multiplier = 1.0 + (regime_state.confidence - 0.5) * self.confidence_sensitivity
        score = self.base_weight * max(0.0, bias) * max(0.0, confidence_multiplier)
        breakdown: MutableMapping[str, float] = {
            "base_weight": self.base_weight,
            "regime_bias": float(bias),
            "confidence_multiplier": float(confidence_multiplier),
        }
        return score, breakdown

    def resolve_parameters(self, regime_state: RegimeState) -> Mapping[str, object]:
        """Materialise parameters with contextual hints for downstream consumers."""

        payload: dict[str, object] = dict(self.parameters)
        payload.setdefault("regime_hint", regime_state.regime)
        payload.setdefault("regime_confidence", regime_state.confidence)
        return payload


@dataclass(frozen=True)
class FastWeightExperiment:
    """Temporary multiplier applied to a tactic when gating conditions are satisfied."""

    experiment_id: str
    tactic_id: str
    delta: float
    rationale: str
    min_confidence: float = 0.0
    feature_gates: Mapping[str, tuple[float | None, float | None]] | None = None
    expires_at: datetime | None = None

    @property
    def multiplier(self) -> float:
        return max(0.0, 1.0 + self.delta)

    def applies(self, regime_state: RegimeState) -> bool:
        if regime_state.confidence < self.min_confidence:
            return False
        if self.expires_at and regime_state.timestamp > self.expires_at:
            return False
        if not self.feature_gates:
            return True
        for feature, bounds in self.feature_gates.items():
            value = regime_state.features.get(feature)
            if value is None:
                return False
            lower, upper = bounds
            if lower is not None and value < lower:
                return False
            if upper is not None and value > upper:
                return False
        return True

    def reflection_payload(self) -> Mapping[str, object]:
        return {
            "experiment_id": self.experiment_id,
            "tactic_id": self.tactic_id,
            "multiplier": self.multiplier,
            "rationale": self.rationale,
        }


@dataclass(frozen=True)
class PolicyDecision:
    """Output of the PolicyRouter contract described in the roadmap."""

    tactic_id: str
    parameters: Mapping[str, object]
    selected_weight: float
    guardrails: Mapping[str, object]
    rationale: str
    experiments_applied: tuple[str, ...]
    reflection_summary: Mapping[str, object]


class PolicyRouter:
    """Route tactics using fast-weight experimentation and automated reflection summaries."""

    def __init__(
        self,
        *,
        default_guardrails: Mapping[str, object] | None = None,
        reflection_history: int = 50,
        summary_top_k: int = 3,
    ) -> None:
        self._tactics: dict[str, PolicyTactic] = {}
        self._experiments: dict[str, FastWeightExperiment] = {}
        self._default_guardrails = dict(default_guardrails or {})
        self._history: Deque[Mapping[str, object]] = deque(maxlen=reflection_history)
        self._summary_top_k = summary_top_k

    def register_tactic(self, tactic: PolicyTactic) -> None:
        if tactic.tactic_id in self._tactics:
            raise ValueError(f"tactic '{tactic.tactic_id}' already registered")
        self._tactics[tactic.tactic_id] = tactic

    def register_experiment(self, experiment: FastWeightExperiment) -> None:
        self._experiments[experiment.experiment_id] = experiment

    def remove_experiment(self, experiment_id: str) -> None:
        self._experiments.pop(experiment_id, None)

    def route(
        self,
        regime_state: RegimeState,
        fast_weights: Mapping[str, float] | None = None,
    ) -> PolicyDecision:
        if not self._tactics:
            raise RuntimeError("no tactics registered")

        scoreboard: list[dict[str, object]] = []
        for tactic in self._tactics.values():
            base_score, breakdown = tactic.score(regime_state)
            multiplier = 1.0
            applied_experiments: list[FastWeightExperiment] = []

            if fast_weights and tactic.tactic_id in fast_weights:
                fast_multiplier = max(0.0, float(fast_weights[tactic.tactic_id]))
                multiplier *= fast_multiplier
                breakdown["fast_weight_multiplier"] = fast_multiplier

            for experiment in self._experiments.values():
                if experiment.tactic_id != tactic.tactic_id:
                    continue
                if not experiment.applies(regime_state):
                    continue
                applied_experiments.append(experiment)
                multiplier *= experiment.multiplier

            score = base_score * multiplier
            scoreboard.append(
                {
                    "tactic": tactic,
                    "score": score,
                    "breakdown": breakdown,
                    "experiments": tuple(applied_experiments),
                    "multiplier": multiplier,
                }
            )

        ranked = sorted(scoreboard, key=lambda item: item["score"], reverse=True)
        winner = ranked[0]
        tactic: PolicyTactic = winner["tactic"]  # type: ignore[assignment]
        experiments: Sequence[FastWeightExperiment] = winner["experiments"]  # type: ignore[assignment]

        guardrails = dict(self._default_guardrails)
        guardrails.update(tactic.guardrails)

        rationale = self._build_rationale(tactic, regime_state, experiments, winner)
        reflection_summary = self._build_reflection_summary(
            regime_state=regime_state,
            ranked=ranked,
            winner=winner,
            rationale=rationale,
        )
        self._history.append(reflection_summary)

        return PolicyDecision(
            tactic_id=tactic.tactic_id,
            parameters=tactic.resolve_parameters(regime_state),
            selected_weight=float(winner["score"]),
            guardrails=guardrails,
            rationale=rationale,
            experiments_applied=tuple(exp.experiment_id for exp in experiments),
            reflection_summary=reflection_summary,
        )

    def history(self) -> Sequence[Mapping[str, object]]:
        return tuple(self._history)

    def _build_rationale(
        self,
        tactic: PolicyTactic,
        regime_state: RegimeState,
        experiments: Sequence[FastWeightExperiment],
        winner: Mapping[str, object],
    ) -> str:
        experiment_clause = (
            "; experiments=" + ",".join(exp.experiment_id for exp in experiments)
            if experiments
            else ""
        )
        return (
            f"tactic={tactic.tactic_id} regime={regime_state.regime}"
            f" conf={regime_state.confidence:.2f} score={winner['score']:.4f}{experiment_clause}"
        )

    def _build_reflection_summary(
        self,
        *,
        regime_state: RegimeState,
        ranked: Sequence[Mapping[str, object]],
        winner: Mapping[str, object],
        rationale: str,
    ) -> Mapping[str, object]:
        top_candidates: list[Mapping[str, object]] = []
        for entry in ranked[: self._summary_top_k]:
            tactic: PolicyTactic = entry["tactic"]  # type: ignore[assignment]
            experiments: Iterable[FastWeightExperiment] = entry["experiments"]  # type: ignore[assignment]
            top_candidates.append(
                {
                    "tactic_id": tactic.tactic_id,
                    "score": float(entry["score"]),
                    "breakdown": dict(entry["breakdown"]),
                    "experiments": [exp.reflection_payload() for exp in experiments],
                }
            )

        winning_tactic: PolicyTactic = winner["tactic"]  # type: ignore[assignment]
        winning_experiments: Iterable[FastWeightExperiment] = winner["experiments"]  # type: ignore[assignment]

        summary = {
            "headline": (
                f"Selected {winning_tactic.tactic_id} for {regime_state.regime}"
                f" (confidence={regime_state.confidence:.2f})"
            ),
            "rationale": rationale,
            "regime_features": dict(regime_state.features),
            "top_candidates": top_candidates,
            "experiments": [exp.reflection_payload() for exp in winning_experiments],
            "timestamp": regime_state.timestamp.isoformat(),
        }
        return summary


__all__ = [
    "PolicyDecision",
    "PolicyRouter",
    "PolicyTactic",
    "FastWeightExperiment",
    "RegimeState",
]
