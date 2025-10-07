"""PolicyRouter for fast-weight tactical experimentation and reflection summaries.

The roadmap's adaptation loop calls for routing tactics based on regime context,
fast-weight experimentation, and automated reviewer-facing summaries so AlphaTrade
operators can understand emerging strategies without combing through telemetry.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Deque, Iterable, Mapping, MutableMapping, Sequence, TYPE_CHECKING


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
    objectives: Sequence[str] = field(default_factory=tuple)
    tags: Sequence[str] = field(default_factory=tuple)

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


if TYPE_CHECKING:  # pragma: no cover - import cycle guard for type checking
    from .policy_reflection import PolicyReflectionArtifacts


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

    def register_tactics(self, tactics: Iterable[PolicyTactic]) -> None:
        """Register multiple tactics in a single call."""

        for tactic in tactics:
            self.register_tactic(tactic)

    def update_tactic(self, tactic: PolicyTactic) -> None:
        """Replace an existing tactic with an updated definition."""

        if tactic.tactic_id not in self._tactics:
            raise KeyError(f"tactic '{tactic.tactic_id}' is not registered")
        self._tactics[tactic.tactic_id] = tactic

    def tactics(self) -> Mapping[str, PolicyTactic]:
        """Return a copy of the registered tactics keyed by identifier."""

        return dict(self._tactics)

    def register_experiment(self, experiment: FastWeightExperiment) -> None:
        self._experiments[experiment.experiment_id] = experiment

    def register_experiments(self, experiments: Iterable[FastWeightExperiment]) -> None:
        """Register multiple fast-weight experiments."""

        for experiment in experiments:
            self.register_experiment(experiment)

    def remove_experiment(self, experiment_id: str) -> None:
        self._experiments.pop(experiment_id, None)

    def prune_experiments(self, *, now: datetime | None = None) -> tuple[str, ...]:
        """Remove expired fast-weight experiments and return their identifiers."""

        current = now or datetime.now(tz=timezone.utc)
        if current.tzinfo is None:
            current = current.replace(tzinfo=timezone.utc)

        removed: list[str] = []
        for experiment_id, experiment in list(self._experiments.items()):
            expires_at = experiment.expires_at
            if expires_at is None:
                continue
            expiry = expires_at if expires_at.tzinfo else expires_at.replace(tzinfo=timezone.utc)
            if expiry <= current:
                self._experiments.pop(experiment_id, None)
                removed.append(experiment_id)
        return tuple(removed)

    def experiments(self) -> Mapping[str, FastWeightExperiment]:
        """Return a copy of the registered experiments keyed by identifier."""

        return dict(self._experiments)

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

    def reflection_digest(self, *, window: int | None = None) -> Mapping[str, object]:
        """Return an aggregated view of recent reflection summaries.

        The digest highlights tactic frequency, experiment usage, and regime
        distribution so reviewers can spot emerging strategies without paging
        through raw telemetry dumps. A ``window`` can be provided to focus the
        aggregation on the most recent N entries.
        """

        history = self.history()
        if window is not None:
            if window <= 0:
                raise ValueError("window must be a positive integer when provided")
            history = history[-window:]

        if not history:
            return {
                "total_decisions": 0,
                "as_of": None,
                "tactics": [],
                "experiments": [],
                "regimes": {},
                "recent_headlines": [],
                "current_streak": {"tactic_id": None, "length": 0},
                "longest_streak": {"tactic_id": None, "length": 0},
            }

        tactic_counts: Counter[str] = Counter()
        tactic_scores: dict[str, float] = {}
        tactic_last_seen: dict[str, datetime] = {}
        tactic_tags: dict[str, Sequence[str]] = {}
        tactic_objectives: dict[str, Sequence[str]] = {}

        experiment_counts: Counter[str] = Counter()
        experiment_last_seen: dict[str, datetime] = {}
        experiment_rationales: dict[str, str] = {}
        experiment_tactics: dict[str, Counter[str]] = {}

        regime_counts: Counter[str] = Counter()

        tag_counts: Counter[str] = Counter()
        tag_last_seen: dict[str, datetime] = {}
        tag_scores: dict[str, float] = {}
        tag_tactics: dict[str, Counter[str]] = {}

        objective_counts: Counter[str] = Counter()
        objective_last_seen: dict[str, datetime] = {}
        objective_scores: dict[str, float] = {}
        objective_tactics: dict[str, Counter[str]] = {}

        recent_headlines: list[str] = []
        as_of: datetime | None = None

        streak_tactic: str | None = None
        streak_length = 0
        longest_streak: tuple[str | None, int] = (None, 0)

        for entry in history:
            tactic_id = str(entry.get("tactic_id", "")) or None
            headline = entry.get("headline")
            if isinstance(headline, str):
                recent_headlines.append(headline)

            timestamp = self._parse_timestamp(entry.get("timestamp"))
            if timestamp and (as_of is None or timestamp > as_of):
                as_of = timestamp

            if tactic_id:
                tactic_counts[tactic_id] += 1
                score = float(entry.get("score", 0.0))
                tactic_scores[tactic_id] = tactic_scores.get(tactic_id, 0.0) + score
                if timestamp and (
                    tactic_id not in tactic_last_seen or timestamp > tactic_last_seen[tactic_id]
                ):
                    tactic_last_seen[tactic_id] = timestamp
                tags = entry.get("tactic_tags")
                if isinstance(tags, (list, tuple)):
                    cleaned_tags = tuple(str(tag) for tag in tags)
                    tactic_tags[tactic_id] = cleaned_tags
                    for tag in cleaned_tags:
                        tag_counts[tag] += 1
                        tag_scores[tag] = tag_scores.get(tag, 0.0) + score
                        if timestamp and (
                            tag not in tag_last_seen or timestamp > tag_last_seen[tag]
                        ):
                            tag_last_seen[tag] = timestamp
                        tag_tactics.setdefault(tag, Counter())[tactic_id] += 1
                objectives = entry.get("tactic_objectives")
                if isinstance(objectives, (list, tuple)):
                    cleaned_objectives = tuple(str(obj) for obj in objectives)
                    tactic_objectives[tactic_id] = cleaned_objectives
                    for objective in cleaned_objectives:
                        objective_counts[objective] += 1
                        objective_scores[objective] = objective_scores.get(objective, 0.0) + score
                        if timestamp and (
                            objective not in objective_last_seen
                            or timestamp > objective_last_seen[objective]
                        ):
                            objective_last_seen[objective] = timestamp
                        objective_tactics.setdefault(objective, Counter())[tactic_id] += 1

                if tactic_id == streak_tactic:
                    streak_length += 1
                else:
                    streak_tactic = tactic_id
                    streak_length = 1
                if streak_length > longest_streak[1]:
                    longest_streak = (tactic_id, streak_length)

            experiments = entry.get("experiments", ())
            for experiment in experiments:
                experiment_id = str(experiment.get("experiment_id", "")) or None
                if not experiment_id:
                    continue
                experiment_counts[experiment_id] += 1
                if timestamp and (
                    experiment_id not in experiment_last_seen
                    or timestamp > experiment_last_seen[experiment_id]
                ):
                    experiment_last_seen[experiment_id] = timestamp
                rationale = experiment.get("rationale")
                if isinstance(rationale, str):
                    experiment_rationales.setdefault(experiment_id, rationale)
                tactic = experiment.get("tactic_id")
                if isinstance(tactic, str):
                    experiment_tactics.setdefault(experiment_id, Counter())[tactic] += 1

            regime = entry.get("regime")
            if isinstance(regime, str) and regime:
                regime_counts[regime] += 1

        total_decisions = len(history)

        tactic_summaries = []
        for tactic_id, count in tactic_counts.most_common():
            avg_score = tactic_scores[tactic_id] / count if count else 0.0
            tactic_summaries.append(
                {
                    "tactic_id": tactic_id,
                    "count": count,
                    "share": count / total_decisions,
                    "avg_score": avg_score,
                    "last_seen": tactic_last_seen.get(tactic_id).isoformat()
                    if tactic_id in tactic_last_seen
                    else None,
                    "tags": list(tactic_tags.get(tactic_id, ())),
                    "objectives": list(tactic_objectives.get(tactic_id, ())),
                }
            )

        experiment_summaries = []
        for experiment_id, count in experiment_counts.most_common():
            top_tactic = None
            if experiment_id in experiment_tactics and experiment_tactics[experiment_id]:
                top_tactic = experiment_tactics[experiment_id].most_common(1)[0][0]
            experiment_summaries.append(
                {
                    "experiment_id": experiment_id,
                    "count": count,
                    "share": count / total_decisions,
                    "last_seen": experiment_last_seen.get(experiment_id).isoformat()
                    if experiment_id in experiment_last_seen
                    else None,
                    "rationale": experiment_rationales.get(experiment_id),
                    "most_common_tactic": top_tactic,
                }
            )

        regime_summary = {
            regime: {"count": count, "share": count / total_decisions}
            for regime, count in regime_counts.most_common()
        }

        def _sorted_entries(counter: Counter[str]) -> list[tuple[str, int]]:
            return sorted(counter.items(), key=lambda item: (-item[1], item[0]))

        tag_summaries = []
        for tag, count in _sorted_entries(tag_counts):
            top_tactics = []
            if tag in tag_tactics and tag_tactics[tag]:
                top_tactics = [tactic for tactic, _ in tag_tactics[tag].most_common(3)]
            tag_summaries.append(
                {
                    "tag": tag,
                    "count": count,
                    "share": count / total_decisions,
                    "avg_score": tag_scores[tag] / count if count else 0.0,
                    "last_seen": tag_last_seen.get(tag).isoformat()
                    if tag in tag_last_seen
                    else None,
                    "top_tactics": top_tactics,
                }
            )

        objective_summaries = []
        for objective, count in _sorted_entries(objective_counts):
            top_tactics = []
            if objective in objective_tactics and objective_tactics[objective]:
                top_tactics = [
                    tactic for tactic, _ in objective_tactics[objective].most_common(3)
                ]
            objective_summaries.append(
                {
                    "objective": objective,
                    "count": count,
                    "share": count / total_decisions,
                    "avg_score": objective_scores[objective] / count if count else 0.0,
                    "last_seen": objective_last_seen.get(objective).isoformat()
                    if objective in objective_last_seen
                    else None,
                    "top_tactics": top_tactics,
                }
            )

        recent_headlines = recent_headlines[-5:]

        return {
            "total_decisions": total_decisions,
            "as_of": as_of.isoformat() if as_of else None,
            "tactics": tactic_summaries,
            "experiments": experiment_summaries,
            "regimes": regime_summary,
            "recent_headlines": recent_headlines,
            "tags": tag_summaries,
            "objectives": objective_summaries,
            "current_streak": {
                "tactic_id": streak_tactic,
                "length": streak_length,
            },
            "longest_streak": {
                "tactic_id": longest_streak[0],
                "length": longest_streak[1],
            },
        }

    def reflection_report(
        self,
        *,
        window: int | None = None,
        now: datetime | None = None,
        max_tactics: int = 5,
        max_experiments: int = 5,
        max_headlines: int = 5,
    ) -> "PolicyReflectionArtifacts":
        """Convenience helper that builds reviewer-ready reflection artifacts."""

        from .policy_reflection import PolicyReflectionBuilder

        now_factory = (lambda now=now: now) if now is not None else None
        builder = PolicyReflectionBuilder(
            self,
            now=now_factory,
            max_tactics=max_tactics,
            max_experiments=max_experiments,
            max_headlines=max_headlines,
        )
        return builder.build(window=window)

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
                    "multiplier": float(entry.get("multiplier", 1.0)),
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
            "regime": regime_state.regime,
            "score": float(winner["score"]),
            "total_multiplier": float(winner.get("multiplier", 1.0)),
            "tactic_id": winning_tactic.tactic_id,
            "tactic_description": winning_tactic.description,
            "tactic_tags": list(winning_tactic.tags),
            "tactic_objectives": list(winning_tactic.objectives),
            "top_candidates": top_candidates,
            "experiments": [exp.reflection_payload() for exp in winning_experiments],
            "timestamp": regime_state.timestamp.isoformat(),
        }
        return summary

    @staticmethod
    def _parse_timestamp(value: object) -> datetime | None:
        if not isinstance(value, str):
            return None
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return None


__all__ = [
    "PolicyDecision",
    "PolicyRouter",
    "PolicyTactic",
    "FastWeightExperiment",
    "RegimeState",
]
