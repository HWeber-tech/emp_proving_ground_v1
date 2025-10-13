"""PolicyRouter for fast-weight tactical experimentation and reflection summaries.

The roadmap's adaptation loop calls for routing tactics based on regime context,
fast-weight experimentation, and automated reviewer-facing summaries so AlphaTrade
operators can understand emerging strategies without combing through telemetry.
"""

from __future__ import annotations

import math
from collections import Counter, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Deque, Iterable, Mapping, MutableMapping, Sequence, TYPE_CHECKING

from .fast_weights import FastWeightController


def _serialise_feature_gates(
    gates: Mapping[str, tuple[float | None, float | None]] | None,
) -> tuple[Mapping[str, float | None], ...]:
    """Return a serialisable view of feature gate bounds."""

    if not gates:
        return ()

    serialised: list[Mapping[str, float | None]] = []
    for feature in sorted(gates):
        lower, upper = gates[feature]
        serialised.append(
            {
                "feature": str(feature),
                "minimum": float(lower) if lower is not None else None,
                "maximum": float(upper) if upper is not None else None,
            }
        )
    return tuple(serialised)


def _normalise_expiry(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


@dataclass(frozen=True)
class RegimeState:
    """Snapshot of the active market regime recognised by the understanding loop."""

    regime: str
    confidence: float
    features: Mapping[str, float]
    timestamp: datetime
    volatility: float = 0.0
    volatility_state: str = "normal"

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        if self.volatility < 0.0:
            raise ValueError("volatility must be non-negative")


@dataclass(frozen=True)
class PolicyTactic:
    """Tactic configuration with regime-aware preferences."""

    tactic_id: str
    base_weight: float
    parameters: Mapping[str, object] = field(default_factory=dict)
    guardrails: Mapping[str, object] = field(default_factory=dict)
    regime_bias: Mapping[str, float] = field(default_factory=dict)
    topology: str | None = None
    confidence_sensitivity: float = 0.5
    description: str | None = None
    objectives: Sequence[str] = field(default_factory=tuple)
    tags: Sequence[str] = field(default_factory=tuple)
    exploration: bool = False

    def __post_init__(self) -> None:
        if self.topology is not None:
            topology_text = str(self.topology).strip()
            object.__setattr__(self, "topology", topology_text or None)
        cleaned_tags = tuple(str(tag).strip() for tag in self.tags if str(tag).strip())
        object.__setattr__(self, "tags", cleaned_tags)
        cleaned_objectives = tuple(
            str(objective).strip()
            for objective in self.objectives
            if str(objective).strip()
        )
        object.__setattr__(self, "objectives", cleaned_objectives)
        object.__setattr__(self, "exploration", bool(self.exploration))

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
        if self.topology:
            payload.setdefault("execution_topology", self.topology)
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
    regimes: Sequence[str] | None = None

    def __post_init__(self) -> None:
        if self.regimes is None:
            return
        cleaned = [
            str(regime).strip()
            for regime in self.regimes
            if isinstance(regime, str) and str(regime).strip()
        ]
        unique = tuple(dict.fromkeys(cleaned))
        object.__setattr__(self, "regimes", unique or None)

    @property
    def multiplier(self) -> float:
        return max(0.0, 1.0 + self.delta)

    def applies(self, regime_state: RegimeState) -> bool:
        if regime_state.confidence < self.min_confidence:
            return False
        if self.expires_at:
            expiry = _normalise_expiry(self.expires_at)
            if expiry and regime_state.timestamp > expiry:
                return False
        if self.regimes and regime_state.regime not in self.regimes:
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
        payload: dict[str, object] = {
            "experiment_id": self.experiment_id,
            "tactic_id": self.tactic_id,
            "multiplier": self.multiplier,
            "delta": self.delta,
            "rationale": self.rationale,
            "min_confidence": float(self.min_confidence),
        }
        gates = _serialise_feature_gates(self.feature_gates)
        if gates:
            payload["feature_gates"] = list(gates)
        if self.regimes:
            payload["regimes"] = list(self.regimes)
        expiry = _normalise_expiry(self.expires_at)
        if expiry:
            payload["expires_at"] = expiry.isoformat()
        return payload


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
    weight_breakdown: Mapping[str, object] = field(default_factory=dict)
    fast_weight_metrics: Mapping[str, object] = field(default_factory=dict)
    decision_timestamp: datetime | None = None
    exploration_metadata: Mapping[str, object] = field(default_factory=dict)


if TYPE_CHECKING:  # pragma: no cover - import cycle guard for type checking
    from .policy_reflection import PolicyReflectionArtifacts


@dataclass
class ExplorationBudget:
    """Track and enforce a global exploration budget for tactic routing."""

    max_fraction: float | None = None
    mutate_every: int | None = None
    _total_decisions: int = 0
    _exploration_decisions: int = 0
    _has_explored: bool = False
    _since_last_exploration: int = 0
    _blocked_attempts: int = 0
    _blocked_reasons: Counter = field(default_factory=Counter)
    _forced_decisions: int = 0
    _forced_reasons: Counter = field(default_factory=Counter)

    def __post_init__(self) -> None:
        if self.max_fraction is not None:
            if not isinstance(self.max_fraction, (int, float)):
                raise TypeError("max_fraction must be numeric or None")
            if math.isnan(self.max_fraction):
                raise ValueError("max_fraction cannot be NaN")
            if not 0.0 <= float(self.max_fraction) <= 1.0:
                raise ValueError("max_fraction must be between 0 and 1")
            self.max_fraction = float(self.max_fraction)
        if self.mutate_every is not None:
            if not isinstance(self.mutate_every, int):
                raise TypeError("mutate_every must be an integer or None")
            if self.mutate_every <= 0:
                raise ValueError("mutate_every must be positive")

    def can_select(self) -> tuple[bool, str | None]:
        """Return whether an exploration tactic can be selected now."""

        if self.max_fraction is not None:
            projected_total = self._total_decisions + 1
            projected_exploration = self._exploration_decisions + 1
            projected_fraction = projected_exploration / projected_total
            if projected_fraction > self.max_fraction + 1e-9:
                return False, "budget_exhausted"

        if self.mutate_every is not None and self._has_explored:
            if self._since_last_exploration < self.mutate_every:
                return False, "cadence"

        return True, None

    def record_decision(self, *, exploration: bool) -> None:
        """Update counters after a decision has been emitted."""

        self._total_decisions += 1
        if exploration:
            self._exploration_decisions += 1
            self._has_explored = True
            self._since_last_exploration = 0
        elif self._has_explored:
            self._since_last_exploration += 1

    def record_blocked(self, reason: str | None) -> None:
        """Record that an exploration attempt was blocked."""

        self._blocked_attempts += 1
        if reason:
            self._blocked_reasons[str(reason)] += 1

    def record_forced(self, reason: str | None) -> None:
        """Record that we had to ignore the budget due to lack of alternatives."""

        self._forced_decisions += 1
        if reason:
            self._forced_reasons[str(reason)] += 1

    def snapshot(self) -> Mapping[str, object]:
        """Return a JSON-serialisable view of the current budget state."""

        total = self._total_decisions
        exploration_total = self._exploration_decisions
        share = exploration_total / total if total else 0.0
        next_allowed = None
        if self.mutate_every is not None:
            if not self._has_explored:
                next_allowed = 0
            else:
                deficit = self.mutate_every - self._since_last_exploration
                next_allowed = max(0, deficit)
        return {
            "max_fraction": self.max_fraction,
            "mutate_every": self.mutate_every,
            "total_decisions": total,
            "exploration_decisions": exploration_total,
            "exploration_share": share,
            "since_last_exploration": (
                self._since_last_exploration if self._has_explored else None
            ),
            "next_exploration_in": next_allowed,
            "blocked_attempts": self._blocked_attempts,
            "blocked_reasons": dict(self._blocked_reasons),
            "forced_decisions": self._forced_decisions,
            "forced_reasons": dict(self._forced_reasons),
        }


class PolicyRouter:
    """Route tactics using fast-weight experimentation and automated reflection summaries."""

    def __init__(
        self,
        *,
        default_guardrails: Mapping[str, object] | None = None,
        reflection_history: int = 50,
        summary_top_k: int = 3,
        fast_weight_controller: "FastWeightController" | None = None,
        regime_switch_deadline_ms: int = 75,
        enforce_regime_topology: bool = True,
        exploration_max_fraction: float | None = None,
        exploration_mutate_every: int | None = None,
    ) -> None:
        self._tactics: dict[str, PolicyTactic] = {}
        self._experiments: dict[str, FastWeightExperiment] = {}
        self._default_guardrails = dict(default_guardrails or {})
        self._history: Deque[Mapping[str, object]] = deque(maxlen=reflection_history)
        self._summary_top_k = summary_top_k
        self._fast_weight_controller = fast_weight_controller or FastWeightController()
        self._regime_switch_deadline_ms = max(0, int(regime_switch_deadline_ms))
        self._enforce_regime_topology = bool(enforce_regime_topology)
        self._last_regime: str | None = None
        self._last_tactic_id: str | None = None
        self._last_topology: str | None = None
        if exploration_max_fraction is not None or exploration_mutate_every is not None:
            self._exploration_budget: ExplorationBudget | None = ExplorationBudget(
                max_fraction=exploration_max_fraction,
                mutate_every=exploration_mutate_every,
            )
        else:
            self._exploration_budget = None

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

    def tactic_catalog(self) -> Sequence[Mapping[str, object]]:
        """Return a reviewer-oriented catalogue of registered tactics."""

        catalogue: list[Mapping[str, object]] = []
        for tactic in sorted(self._tactics.values(), key=lambda item: item.tactic_id):
            catalogue.append(
                {
                    "tactic_id": tactic.tactic_id,
                    "description": tactic.description,
                    "base_weight": float(tactic.base_weight),
                    "confidence_sensitivity": float(tactic.confidence_sensitivity),
                    "regime_bias": dict(tactic.regime_bias),
                    "parameters": dict(tactic.parameters),
                    "guardrails": dict(tactic.guardrails),
                    "objectives": list(tactic.objectives),
                    "tags": list(tactic.tags),
                }
            )
        return tuple(catalogue)

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

    def experiment_registry(
        self,
        *,
        regime_state: RegimeState | None = None,
    ) -> Sequence[Mapping[str, object]]:
        """Summarise fast-weight experiments for reviewers and diagnostics."""

        entries: list[Mapping[str, object]] = []
        for experiment in sorted(self._experiments.values(), key=lambda item: item.experiment_id):
            feature_gates = _serialise_feature_gates(experiment.feature_gates)
            expiry = _normalise_expiry(experiment.expires_at)
            entry: dict[str, object] = {
                "experiment_id": experiment.experiment_id,
                "tactic_id": experiment.tactic_id,
                "delta": float(experiment.delta),
                "multiplier": float(experiment.multiplier),
                "rationale": experiment.rationale,
                "min_confidence": float(experiment.min_confidence),
                "feature_gates": list(feature_gates),
                "expires_at": expiry.isoformat() if expiry else None,
                "regimes": list(experiment.regimes) if experiment.regimes else [],
            }
            if regime_state is not None:
                entry["would_apply"] = experiment.applies(regime_state)
            entries.append(entry)
        return tuple(entries)

    def route(
        self,
        regime_state: RegimeState,
        fast_weights: Mapping[str, float] | None = None,
        *,
        decision_timestamp: datetime | None = None,
    ) -> PolicyDecision:
        if not self._tactics:
            raise RuntimeError("no tactics registered")

        effective_fast_weights: Mapping[str, float] | None = None
        fast_weight_metrics_payload: Mapping[str, object] | None = None
        if self._fast_weight_controller:
            result = self._fast_weight_controller.constrain(
                fast_weights=fast_weights,
                tactic_ids=self._tactics.keys(),
            )
            effective_fast_weights = result.weights
            fast_weight_metrics_payload = dict(result.metrics.as_dict())
        else:
            effective_fast_weights = dict(fast_weights or {})

        fast_weight_lookup = (
            dict(effective_fast_weights) if effective_fast_weights else {}
        )

        scoreboard: list[dict[str, object]] = []
        for tactic in self._tactics.values():
            base_score, breakdown = tactic.score(regime_state)
            multiplier = 1.0
            applied_experiments: list[FastWeightExperiment] = []

            if fast_weight_lookup and tactic.tactic_id in fast_weight_lookup:
                fast_multiplier = max(0.0, float(fast_weight_lookup[tactic.tactic_id]))
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
                    "base_score": base_score,
                    "fast_weight_metrics": (
                        dict(fast_weight_metrics_payload)
                        if fast_weight_metrics_payload
                        else None
                    ),
                }
            )

        ranked_entries = sorted(scoreboard, key=lambda item: item["score"], reverse=True)
        if not ranked_entries:
            raise RuntimeError("no tactics available after scoring")

        exploration_context: dict[str, object] | None = None
        if self._exploration_budget is not None:
            budget_before = self._exploration_budget.snapshot()
            working_entries, enforcement_context = self._apply_exploration_budget(
                ranked=ranked_entries,
            )
            exploration_context = dict(enforcement_context)
            exploration_context["budget_before"] = dict(budget_before)
        else:
            working_entries = list(ranked_entries)

        if not working_entries:
            raise RuntimeError("no tactics available after budgeting")

        regime_changed = (
            self._last_regime is not None and regime_state.regime != self._last_regime
        )
        previous_topology = self._last_topology

        winner_entry = working_entries[0]
        switch_forced = False
        if regime_changed and self._enforce_regime_topology:
            candidate_entry, forced = self._select_regime_transition_winner(
                ranked=working_entries,
                regime_state=regime_state,
                previous_topology=previous_topology,
                default_entry=winner_entry,
            )
            winner_entry = candidate_entry
            switch_forced = forced

        if winner_entry is working_entries[0]:
            ordered_ranked: list[Mapping[str, object]] = list(working_entries)
        else:
            ordered_ranked = [winner_entry] + [
                entry for entry in working_entries if entry is not winner_entry
            ]

        decision_time = decision_timestamp or datetime.now(tz=timezone.utc)
        if decision_time.tzinfo is None:
            decision_time = decision_time.replace(tzinfo=timezone.utc)
        else:
            decision_time = decision_time.astimezone(timezone.utc)

        tactic: PolicyTactic = winner_entry["tactic"]  # type: ignore[assignment]
        experiments: Sequence[FastWeightExperiment] = winner_entry["experiments"]  # type: ignore[assignment]

        guardrails = dict(self._default_guardrails)
        guardrails.update(tactic.guardrails)

        rationale = self._build_rationale(tactic, regime_state, experiments, winner_entry)
        if self._exploration_budget is not None:
            self._exploration_budget.record_decision(exploration=bool(tactic.exploration))
            after_snapshot = self._exploration_budget.snapshot()
            if exploration_context is None:
                exploration_context = {}
            exploration_context.setdefault("blocked_candidates", [])
            exploration_context.setdefault("forced", False)
            exploration_context["budget_after"] = dict(after_snapshot)
            if "budget_before" not in exploration_context:
                exploration_context["budget_before"] = dict(after_snapshot)
            exploration_context["selected_is_exploration"] = bool(tactic.exploration)
        else:
            exploration_context = None

        weight_breakdown = self._build_weight_breakdown(
            tactic=tactic,
            summary=winner_entry,
            experiments=experiments,
            exploration_context=exploration_context,
        )
        regime_transition = self._build_regime_transition_summary(
            regime_state=regime_state,
            decision_timestamp=decision_time,
            selected_tactic=tactic,
            ranked=ordered_ranked,
            switch_forced=switch_forced,
        )
        reflection_summary = dict(
            self._build_reflection_summary(
                regime_state=regime_state,
                ranked=ordered_ranked,
                winner=winner_entry,
                rationale=rationale,
                weight_breakdown=weight_breakdown,
                exploration_context=exploration_context,
            )
        )
        reflection_summary["decision_timestamp"] = decision_time.isoformat()
        reflection_summary["regime_transition"] = regime_transition
        self._history.append(reflection_summary)

        self._last_regime = regime_state.regime
        self._last_tactic_id = tactic.tactic_id
        self._last_topology = self._topology_identifier(tactic)

        return PolicyDecision(
            tactic_id=tactic.tactic_id,
            parameters=tactic.resolve_parameters(regime_state),
            selected_weight=float(winner_entry["score"]),
            guardrails=guardrails,
            rationale=rationale,
            experiments_applied=tuple(exp.experiment_id for exp in experiments),
            reflection_summary=reflection_summary,
            weight_breakdown=weight_breakdown,
            fast_weight_metrics=(
                dict(winner_entry.get("fast_weight_metrics", {}))
                if isinstance(winner_entry.get("fast_weight_metrics"), Mapping)
                else {}
            ),
            decision_timestamp=decision_time,
            exploration_metadata=dict(exploration_context or {}),
        )

    def history(self) -> Sequence[Mapping[str, object]]:
        return tuple(self._history)

    def ingest_reflection_history(self, entries: Iterable[Mapping[str, object]]) -> int:
        """Seed the reflection history with pre-recorded summaries.

        Decision diary exports and governance workflows often capture
        ``reflection_summary`` payloads emitted by the router.  This helper lets
        those persisted summaries be replayed into a fresh router instance so the
        automated digest and reflection builder can operate on historical data
        without re-running the full understanding loop.

        Entries missing a ``tactic_id`` or ``timestamp`` are ignored.  The method
        returns the number of summaries accepted.
        """

        appended = 0
        for entry in entries:
            if not isinstance(entry, Mapping):
                continue

            summary: dict[str, object] = dict(entry)

            tactic_id = summary.get("tactic_id")
            if not isinstance(tactic_id, str) or not tactic_id.strip():
                continue
            summary["tactic_id"] = tactic_id.strip()

            timestamp = summary.get("timestamp")
            if isinstance(timestamp, datetime):
                normalised_ts = timestamp.astimezone(timezone.utc)
                summary["timestamp"] = normalised_ts.isoformat()
            elif isinstance(timestamp, str):
                candidate = timestamp.strip()
                if not candidate:
                    continue
                try:
                    parsed = datetime.fromisoformat(candidate)
                except ValueError:
                    continue
                summary["timestamp"] = parsed.astimezone(timezone.utc).isoformat()
            else:
                continue

            score = summary.get("score", 0.0)
            try:
                summary["score"] = float(score)
            except (TypeError, ValueError):
                summary["score"] = 0.0

            experiments = summary.get("experiments")
            if isinstance(experiments, Sequence):
                cleaned_experiments: list[Mapping[str, object]] = []
                for experiment in experiments:
                    if not isinstance(experiment, Mapping):
                        continue
                    payload = dict(experiment)
                    exp_id = payload.get("experiment_id")
                    if not isinstance(exp_id, str) or not exp_id.strip():
                        continue
                    payload["experiment_id"] = exp_id.strip()
                    cleaned_experiments.append(payload)
                summary["experiments"] = cleaned_experiments
            else:
                summary["experiments"] = []

            tags = summary.get("tactic_tags")
            if isinstance(tags, Sequence) and not isinstance(tags, (str, bytes)):
                summary["tactic_tags"] = [str(tag) for tag in tags]

            objectives = summary.get("tactic_objectives")
            if isinstance(objectives, Sequence) and not isinstance(objectives, (str, bytes)):
                summary["tactic_objectives"] = [str(obj) for obj in objectives]

            self._history.append(summary)
            appended += 1

        return appended

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

        def _iso(timestamp: datetime | None) -> str | None:
            if timestamp is None:
                return None
            if timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=timezone.utc)
            else:
                timestamp = timestamp.astimezone(timezone.utc)
            return timestamp.isoformat()

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
                "confidence": {
                    "count": 0,
                    "average": None,
                    "min": None,
                    "max": None,
                    "latest": None,
                    "change": None,
                    "first_seen": None,
                    "last_seen": None,
                },
                "features": [],
                "weight_stats": {
                    "average_base_score": None,
                    "average_total_multiplier": None,
                    "average_final_score": None,
                    "fast_weight": {
                        "applications": 0,
                        "count": 0,
                        "average_multiplier": None,
                        "min_multiplier": None,
                        "max_multiplier": None,
                        "active_percentage": {
                            "latest": None,
                            "average": None,
                            "min": None,
                            "max": None,
                        },
                    },
                },
                "emerging_tactics": [],
                "emerging_experiments": [],
            }

        tactic_counts: Counter[str] = Counter()
        tactic_scores: dict[str, float] = {}
        tactic_first_seen: dict[str, datetime] = {}
        tactic_last_seen: dict[str, datetime] = {}
        tactic_tags: dict[str, Sequence[str]] = {}
        tactic_objectives: dict[str, Sequence[str]] = {}

        experiment_counts: Counter[str] = Counter()
        experiment_first_seen: dict[str, datetime] = {}
        experiment_last_seen: dict[str, datetime] = {}
        experiment_rationales: dict[str, str] = {}
        experiment_tactics: dict[str, Counter[str]] = {}
        experiment_regimes: dict[str, set[str]] = {}
        experiment_min_confidence: dict[str, float] = {}
        experiment_feature_gates: dict[str, tuple[Mapping[str, object], ...]] = {}
        experiment_expires_at: dict[str, str] = {}
        experiment_multipliers: dict[str, float] = {}
        experiment_deltas: dict[str, float] = {}

        regime_counts: Counter[str] = Counter()

        tag_counts: Counter[str] = Counter()
        tag_first_seen: dict[str, datetime] = {}
        tag_last_seen: dict[str, datetime] = {}
        tag_scores: dict[str, float] = {}
        tag_tactics: dict[str, Counter[str]] = {}

        objective_counts: Counter[str] = Counter()
        objective_first_seen: dict[str, datetime] = {}
        objective_last_seen: dict[str, datetime] = {}
        objective_scores: dict[str, float] = {}
        objective_tactics: dict[str, Counter[str]] = {}

        confidence_values: list[tuple[datetime | None, float]] = []
        confidence_first_value: float | None = None
        confidence_first_timestamp: datetime | None = None

        feature_counts: Counter[str] = Counter()
        feature_sums: dict[str, float] = {}
        feature_min: dict[str, float] = {}
        feature_max: dict[str, float] = {}
        feature_first_value: dict[str, float] = {}
        feature_first_seen: dict[str, datetime | None] = {}
        feature_last_value: dict[str, float] = {}
        feature_last_seen: dict[str, datetime | None] = {}

        recent_headlines: list[str] = []
        as_of: datetime | None = None

        streak_tactic: str | None = None
        streak_length = 0
        longest_streak: tuple[str | None, int] = (None, 0)

        base_score_sum = 0.0
        base_score_count = 0
        total_multiplier_sum = 0.0
        total_multiplier_count = 0
        total_score_sum = 0.0
        fast_weight_sum = 0.0
        fast_weight_count = 0
        fast_weight_applications = 0
        fast_weight_max: float | None = None
        fast_weight_min: float | None = None
        fast_weight_active_sum = 0.0
        fast_weight_active_count = 0
        fast_weight_active_min: float | None = None
        fast_weight_active_max: float | None = None
        fast_weight_active_latest: float | None = None

        for entry in history:
            tactic_id = str(entry.get("tactic_id", "")) or None
            headline = entry.get("headline")
            if isinstance(headline, str):
                recent_headlines.append(headline)

            timestamp = self._parse_timestamp(entry.get("timestamp"))
            if timestamp and (as_of is None or timestamp > as_of):
                as_of = timestamp

            confidence_value = entry.get("confidence")
            if isinstance(confidence_value, (int, float)):
                numeric_confidence = float(confidence_value)
                confidence_values.append((timestamp, numeric_confidence))
                if confidence_first_value is None:
                    confidence_first_value = numeric_confidence
                    confidence_first_timestamp = timestamp

            regime_features = entry.get("regime_features")
            if isinstance(regime_features, Mapping):
                for feature_name, raw_value in regime_features.items():
                    try:
                        numeric_value = float(raw_value)
                    except (TypeError, ValueError):
                        continue
                    feature_counts[feature_name] += 1
                    feature_sums[feature_name] = feature_sums.get(feature_name, 0.0) + numeric_value
                    current_min = feature_min.get(feature_name)
                    if current_min is None or numeric_value < current_min:
                        feature_min[feature_name] = numeric_value
                    current_max = feature_max.get(feature_name)
                    if current_max is None or numeric_value > current_max:
                        feature_max[feature_name] = numeric_value
                    if feature_name not in feature_first_value:
                        feature_first_value[feature_name] = numeric_value
                        feature_first_seen[feature_name] = timestamp
                    feature_last_value[feature_name] = numeric_value
                    feature_last_seen[feature_name] = timestamp

            if tactic_id:
                tactic_counts[tactic_id] += 1
                score_value = entry.get("score", 0.0)
                try:
                    score = float(score_value)
                except (TypeError, ValueError):
                    score = 0.0
                total_score_sum += score
                tactic_scores[tactic_id] = tactic_scores.get(tactic_id, 0.0) + score
                if timestamp and tactic_id not in tactic_first_seen:
                    tactic_first_seen[tactic_id] = timestamp
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
                        if timestamp and tag not in tag_first_seen:
                            tag_first_seen[tag] = timestamp
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
                        if timestamp and objective not in objective_first_seen:
                            objective_first_seen[objective] = timestamp
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
                if timestamp and experiment_id not in experiment_first_seen:
                    experiment_first_seen[experiment_id] = timestamp
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
                multiplier = experiment.get("multiplier")
                if isinstance(multiplier, (int, float)):
                    experiment_multipliers.setdefault(experiment_id, float(multiplier))
                delta = experiment.get("delta")
                if isinstance(delta, (int, float)):
                    experiment_deltas.setdefault(experiment_id, float(delta))
                min_confidence = experiment.get("min_confidence")
                if isinstance(min_confidence, (int, float)):
                    experiment_min_confidence.setdefault(experiment_id, float(min_confidence))
                regimes = experiment.get("regimes")
                if isinstance(regimes, Sequence) and not isinstance(regimes, (str, bytes)):
                    bucket = experiment_regimes.setdefault(experiment_id, set())
                    bucket.update(
                        str(regime).strip()
                        for regime in regimes
                        if isinstance(regime, str) and str(regime).strip()
                    )
                feature_gates = experiment.get("feature_gates")
                if isinstance(feature_gates, Sequence) and feature_gates:
                    cleaned: list[Mapping[str, object]] = []
                    for gate in feature_gates:
                        if isinstance(gate, Mapping):
                            cleaned.append(dict(gate))
                    if cleaned:
                        experiment_feature_gates.setdefault(
                            experiment_id,
                            tuple(cleaned),
                        )
                expiry = experiment.get("expires_at")
                if isinstance(expiry, str):
                    experiment_expires_at.setdefault(experiment_id, expiry)

            regime = entry.get("regime")
            if isinstance(regime, str) and regime:
                regime_counts[regime] += 1

            weight_breakdown = entry.get("weight_breakdown")
            if isinstance(weight_breakdown, Mapping):
                base_score = weight_breakdown.get("base_score")
                if isinstance(base_score, (int, float)):
                    base_score_sum += float(base_score)
                    base_score_count += 1
                total_multiplier = weight_breakdown.get("total_multiplier")
                if isinstance(total_multiplier, (int, float)):
                    multiplier_value = float(total_multiplier)
                    total_multiplier_sum += multiplier_value
                    total_multiplier_count += 1
                fast_weight_multiplier = weight_breakdown.get("fast_weight_multiplier")
                if isinstance(fast_weight_multiplier, (int, float)):
                    fast_multiplier_value = float(fast_weight_multiplier)
                    fast_weight_sum += fast_multiplier_value
                    fast_weight_count += 1
                    fast_weight_max = (
                        fast_multiplier_value
                        if fast_weight_max is None
                        else max(fast_weight_max, fast_multiplier_value)
                    )
                    fast_weight_min = (
                        fast_multiplier_value
                        if fast_weight_min is None
                        else min(fast_weight_min, fast_multiplier_value)
                    )
                    if not math.isclose(fast_multiplier_value, 1.0, rel_tol=1e-9, abs_tol=1e-9):
                        fast_weight_applications += 1
                elif isinstance(entry.get("total_multiplier"), (int, float)):
                    # fall back when fast weight multiplier missing but total recorded
                    total_multiplier_sum += float(entry["total_multiplier"])
                    total_multiplier_count += 1
            metrics_entry = entry.get("fast_weight_metrics")
            if isinstance(metrics_entry, Mapping):
                active_value = metrics_entry.get("active_percentage")
                if isinstance(active_value, (int, float)):
                    numeric_active = float(active_value)
                    fast_weight_active_sum += numeric_active
                    fast_weight_active_count += 1
                    fast_weight_active_latest = numeric_active
                    fast_weight_active_min = (
                        numeric_active
                        if fast_weight_active_min is None
                        else min(fast_weight_active_min, numeric_active)
                    )
                    fast_weight_active_max = (
                        numeric_active
                        if fast_weight_active_max is None
                        else max(fast_weight_active_max, numeric_active)
                    )
        if total_score_sum == 0.0 and tactic_counts:
            total_score_sum = sum(tactic_scores.values())

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
                    "first_seen": _iso(tactic_first_seen.get(tactic_id)),
                    "last_seen": _iso(tactic_last_seen.get(tactic_id)),
                    "tags": list(tactic_tags.get(tactic_id, ())),
                    "objectives": list(tactic_objectives.get(tactic_id, ())),
                }
            )

        experiment_summaries = []
        for experiment_id, count in experiment_counts.most_common():
            top_tactic = None
            if experiment_id in experiment_tactics and experiment_tactics[experiment_id]:
                top_tactic = experiment_tactics[experiment_id].most_common(1)[0][0]
            last_seen = (
                _iso(experiment_last_seen.get(experiment_id))
            )
            entry: dict[str, object] = {
                "experiment_id": experiment_id,
                "count": count,
                "share": count / total_decisions,
                "first_seen": _iso(experiment_first_seen.get(experiment_id)),
                "last_seen": last_seen,
                "rationale": experiment_rationales.get(experiment_id),
                "most_common_tactic": top_tactic,
            }
            experiment_model = self._experiments.get(experiment_id)
            if experiment_model:
                entry["min_confidence"] = float(experiment_model.min_confidence)
                entry["regimes"] = (
                    list(experiment_model.regimes) if experiment_model.regimes else []
                )
                entry["feature_gates"] = list(
                    _serialise_feature_gates(experiment_model.feature_gates)
                )
                expiry = _normalise_expiry(experiment_model.expires_at)
                entry["expires_at"] = expiry.isoformat() if expiry else None
                entry["multiplier"] = float(experiment_model.multiplier)
                entry["delta"] = float(experiment_model.delta)
            else:
                if experiment_id in experiment_min_confidence:
                    entry["min_confidence"] = experiment_min_confidence[experiment_id]
                if experiment_id in experiment_regimes:
                    entry["regimes"] = sorted(experiment_regimes[experiment_id])
                if experiment_id in experiment_feature_gates:
                    entry["feature_gates"] = list(experiment_feature_gates[experiment_id])
                if experiment_id in experiment_expires_at:
                    entry["expires_at"] = experiment_expires_at[experiment_id]
                if experiment_id in experiment_multipliers:
                    entry["multiplier"] = experiment_multipliers[experiment_id]
                if experiment_id in experiment_deltas:
                    entry["delta"] = experiment_deltas[experiment_id]
            experiment_summaries.append(entry)

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
                    "first_seen": _iso(tag_first_seen.get(tag)),
                    "last_seen": _iso(tag_last_seen.get(tag)),
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
                    "first_seen": _iso(objective_first_seen.get(objective)),
                    "last_seen": _iso(objective_last_seen.get(objective)),
                    "top_tactics": top_tactics,
                }
            )

        recent_headlines = recent_headlines[-5:]

        if confidence_values:
            sentinel = datetime.min.replace(tzinfo=timezone.utc)
            sorted_confidence = sorted(confidence_values, key=lambda item: item[0] or sentinel)
            values = [value for _, value in sorted_confidence]
            latest_timestamp = next((ts for ts, _ in reversed(sorted_confidence) if ts), None)
            confidence_summary: Mapping[str, object] = {
                "count": len(values),
                "average": sum(values) / len(values),
                "min": min(values),
                "max": max(values),
                "latest": values[-1],
                "change": (
                    values[-1] - confidence_first_value
                    if confidence_first_value is not None
                    else values[-1] - values[0]
                ),
                "first_seen": (
                    confidence_first_timestamp.isoformat()
                    if confidence_first_timestamp
                    else None
                ),
                "last_seen": latest_timestamp.isoformat() if latest_timestamp else None,
            }
        else:
            confidence_summary = {
                "count": 0,
                "average": None,
                "min": None,
                "max": None,
                "latest": None,
                "change": None,
                "first_seen": None,
                "last_seen": None,
            }

        feature_summaries: list[Mapping[str, object]] = []
        if feature_counts:
            for feature_name, count in feature_counts.most_common():
                total = feature_sums.get(feature_name, 0.0)
                average = total / count if count else 0.0
                latest_value = feature_last_value.get(feature_name)
                last_seen = feature_last_seen.get(feature_name)
                first_value = feature_first_value.get(feature_name)
                trend = None
                if latest_value is not None and first_value is not None:
                    trend = latest_value - first_value
                first_seen = feature_first_seen.get(feature_name)
                feature_summaries.append(
                    {
                        "feature": feature_name,
                        "count": count,
                        "average": average,
                        "latest": latest_value,
                        "min": feature_min.get(feature_name),
                        "max": feature_max.get(feature_name),
                        "trend": trend,
                        "first_seen": first_seen.isoformat()
                        if isinstance(first_seen, datetime)
                        else None,
                        "last_seen": last_seen.isoformat()
                        if isinstance(last_seen, datetime)
                        else None,
                    }
                )

        average_base_score = (
            base_score_sum / base_score_count if base_score_count else None
        )
        average_total_multiplier = (
            total_multiplier_sum / total_multiplier_count if total_multiplier_count else None
        )
        average_final_score = (
            total_score_sum / total_decisions if total_decisions else None
        )
        fast_weight_average = (
            fast_weight_sum / fast_weight_count if fast_weight_count else None
        )
        fast_weight_active_average = (
            fast_weight_active_sum / fast_weight_active_count
            if fast_weight_active_count
            else None
        )

        weight_stats = {
            "average_base_score": float(average_base_score)
            if average_base_score is not None
            else None,
            "average_total_multiplier": float(average_total_multiplier)
            if average_total_multiplier is not None
            else None,
            "average_final_score": float(average_final_score)
            if average_final_score is not None
            else None,
            "fast_weight": {
                "applications": fast_weight_applications,
                "count": fast_weight_count,
                "average_multiplier": float(fast_weight_average)
                if fast_weight_average is not None
                else None,
                "min_multiplier": float(fast_weight_min)
                if fast_weight_min is not None
                else None,
                "max_multiplier": float(fast_weight_max)
                if fast_weight_max is not None
                else None,
            },
        }
        active_percentage_summary = {
            "latest": float(fast_weight_active_latest)
            if fast_weight_active_latest is not None
            else None,
            "average": float(fast_weight_active_average)
            if fast_weight_active_average is not None
            else None,
            "min": float(fast_weight_active_min)
            if fast_weight_active_min is not None
            else None,
            "max": float(fast_weight_active_max)
            if fast_weight_active_max is not None
            else None,
        }
        weight_stats["fast_weight"]["active_percentage"] = active_percentage_summary

        emerging_tactics: list[Mapping[str, object]] = []
        if tactic_counts:
            min_timestamp = datetime.min.replace(tzinfo=timezone.utc)

            def _tactic_sort_key(tactic_id: str) -> tuple[datetime, int, str]:
                first_seen = tactic_first_seen.get(tactic_id) or min_timestamp
                if first_seen.tzinfo is None:
                    first_seen = first_seen.replace(tzinfo=timezone.utc)
                return (first_seen, tactic_counts[tactic_id], tactic_id)

            for tactic_id in sorted(
                tactic_counts,
                key=_tactic_sort_key,
                reverse=True,
            )[: self._summary_top_k]:
                count = tactic_counts[tactic_id]
                avg_score = tactic_scores.get(tactic_id, 0.0) / count if count else 0.0
                emerging_tactics.append(
                    {
                        "tactic_id": tactic_id,
                        "count": count,
                        "share": count / total_decisions,
                        "avg_score": avg_score,
                        "first_seen": _iso(tactic_first_seen.get(tactic_id)),
                        "last_seen": _iso(tactic_last_seen.get(tactic_id)),
                        "tags": list(tactic_tags.get(tactic_id, ())),
                        "objectives": list(tactic_objectives.get(tactic_id, ())),
                    }
                )

        emerging_experiments: list[Mapping[str, object]] = []
        if experiment_counts:
            min_timestamp = datetime.min.replace(tzinfo=timezone.utc)

            def _experiment_sort_key(experiment_id: str) -> tuple[datetime, int, str]:
                first_seen = experiment_first_seen.get(experiment_id) or min_timestamp
                if first_seen.tzinfo is None:
                    first_seen = first_seen.replace(tzinfo=timezone.utc)
                return (first_seen, experiment_counts[experiment_id], experiment_id)

            experiment_summary_index = {
                str(entry.get("experiment_id")): dict(entry) for entry in experiment_summaries
            }

            for experiment_id in sorted(
                experiment_counts,
                key=_experiment_sort_key,
                reverse=True,
            )[: self._summary_top_k]:
                summary = experiment_summary_index.get(experiment_id, {})
                enriched: dict[str, object] = dict(summary)
                enriched.setdefault("experiment_id", experiment_id)
                enriched["count"] = experiment_counts[experiment_id]
                enriched["share"] = experiment_counts[experiment_id] / total_decisions
                enriched["first_seen"] = _iso(experiment_first_seen.get(experiment_id))
                enriched["last_seen"] = _iso(experiment_last_seen.get(experiment_id))
                emerging_experiments.append(enriched)

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
            "confidence": confidence_summary,
            "features": feature_summaries,
            "weight_stats": weight_stats,
            "emerging_tactics": emerging_tactics,
            "emerging_experiments": emerging_experiments,
        }

    def reflection_report(
        self,
        *,
        window: int | None = None,
        now: datetime | None = None,
        max_tactics: int = 5,
        max_experiments: int = 5,
        max_headlines: int = 5,
        max_features: int = 5,
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
            max_features=max_features,
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
        weight_breakdown: Mapping[str, object],
        exploration_context: Mapping[str, object] | None = None,
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
                    "base_score": float(entry.get("base_score", 0.0)),
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
            "confidence": float(regime_state.confidence),
            "score": float(winner["score"]),
            "total_multiplier": float(winner.get("multiplier", 1.0)),
            "tactic_id": winning_tactic.tactic_id,
            "tactic_description": winning_tactic.description,
            "tactic_tags": list(winning_tactic.tags),
            "tactic_objectives": list(winning_tactic.objectives),
            "top_candidates": top_candidates,
            "experiments": [exp.reflection_payload() for exp in winning_experiments],
            "timestamp": regime_state.timestamp.isoformat(),
            "weight_breakdown": dict(weight_breakdown),
            "fast_weight_metrics": (
                dict(winner.get("fast_weight_metrics", {}))
                if isinstance(winner.get("fast_weight_metrics"), Mapping)
                else {}
            ),
        }
        if exploration_context:
            summary["exploration"] = {
                "selected": bool(exploration_context.get("selected_is_exploration", False)),
                "forced": bool(exploration_context.get("forced", False)),
                "blocked_candidates": [
                    {
                        "tactic_id": item.get("tactic_id"),
                        "reason": item.get("reason"),
                        "score": item.get("score"),
                    }
                    for item in exploration_context.get("blocked_candidates", [])
                ],
                "budget_before": dict(exploration_context.get("budget_before", {})),
                "budget_after": dict(exploration_context.get("budget_after", {})),
            }
        return summary

    @staticmethod
    def _build_weight_breakdown(
        *,
        tactic: PolicyTactic,
        summary: Mapping[str, object],
        experiments: Sequence[FastWeightExperiment],
        exploration_context: Mapping[str, object] | None = None,
    ) -> Mapping[str, object]:
        breakdown = dict(summary.get("breakdown", {}))
        total_multiplier = float(summary.get("multiplier", 1.0))
        fast_weight_multiplier = float(breakdown.get("fast_weight_multiplier", 1.0))
        experiment_multipliers = {
            experiment.experiment_id: float(experiment.multiplier)
            for experiment in experiments
        }
        base_score = float(summary.get("base_score", 0.0))

        payload: dict[str, object] = {
            "base_weight": float(tactic.base_weight),
            "regime_bias": float(breakdown.get("regime_bias", 1.0)),
            "confidence_multiplier": float(breakdown.get("confidence_multiplier", 1.0)),
            "fast_weight_multiplier": fast_weight_multiplier,
            "experiment_multipliers": experiment_multipliers,
            "total_multiplier": total_multiplier,
            "base_score": base_score,
            "final_score": float(summary.get("score", base_score * total_multiplier)),
        }
        metrics = summary.get("fast_weight_metrics")
        if isinstance(metrics, Mapping):
            active_percentage = metrics.get("active_percentage")
            payload["fast_weight_active_percentage"] = (
                float(active_percentage)
                if isinstance(active_percentage, (int, float))
                else None
            )
        else:
            payload["fast_weight_active_percentage"] = None
        if exploration_context:
            payload["exploration"] = {
                "selected": bool(exploration_context.get("selected_is_exploration", False)),
                "forced": bool(exploration_context.get("forced", False)),
                "blocked_candidates": [
                    {
                        "tactic_id": item.get("tactic_id"),
                        "reason": item.get("reason"),
                        "score": item.get("score"),
                    }
                    for item in exploration_context.get("blocked_candidates", [])
                ],
                "budget_before": dict(exploration_context.get("budget_before", {})),
                "budget_after": dict(exploration_context.get("budget_after", {})),
            }
        return payload

    @staticmethod
    def _topology_identifier(tactic: PolicyTactic) -> str:
        return tactic.topology or tactic.tactic_id

    def _apply_exploration_budget(
        self,
        *,
        ranked: Sequence[Mapping[str, object]],
    ) -> tuple[list[Mapping[str, object]], Mapping[str, object]]:
        if not ranked:
            return [], {}

        budget = self._exploration_budget
        if budget is None:
            return list(ranked), {}

        allowed: list[Mapping[str, object]] = []
        blocked_details: list[tuple[Mapping[str, object], str | None]] = []

        for entry in ranked:
            tactic: PolicyTactic = entry["tactic"]  # type: ignore[assignment]
            if not tactic.exploration:
                entry["exploration_budget_status"] = "allowed"
                entry["exploration_budget_reason"] = None
                allowed.append(entry)
                continue

            permitted, reason = budget.can_select()
            if permitted:
                entry["exploration_budget_status"] = "allowed"
                entry["exploration_budget_reason"] = None
                allowed.append(entry)
            else:
                entry["exploration_budget_status"] = "blocked"
                entry["exploration_budget_reason"] = reason
                blocked_details.append((entry, reason))

        metadata: dict[str, object] = {}

        if allowed:
            for _, reason in blocked_details:
                budget.record_blocked(reason)
            ordered = allowed + [entry for entry, _ in blocked_details]
            metadata["blocked_candidates"] = [
                {
                    "tactic_id": entry["tactic"].tactic_id,  # type: ignore[index]
                    "reason": reason,
                    "score": float(entry.get("score", 0.0)),
                }
                for entry, reason in blocked_details
            ]
            metadata["forced"] = False
            return ordered, metadata

        ordered = list(ranked)
        metadata["blocked_candidates"] = [
            {
                "tactic_id": entry["tactic"].tactic_id,  # type: ignore[index]
                "reason": reason,
                "score": float(entry.get("score", 0.0)),
            }
            for entry, reason in blocked_details
        ]
        metadata["forced"] = bool(blocked_details)
        if blocked_details:
            budget.record_forced(blocked_details[0][1])
        return ordered, metadata

    def _select_regime_transition_winner(
        self,
        *,
        ranked: Sequence[Mapping[str, object]],
        regime_state: RegimeState,
        previous_topology: str | None,
        default_entry: Mapping[str, object],
    ) -> tuple[Mapping[str, object], bool]:
        if not ranked:
            raise RuntimeError("no tactics available after scoring")

        best_bias = float("-inf")
        preferred_entry: Mapping[str, object] | None = None
        fallback_entry: Mapping[str, object] | None = None

        for entry in ranked:
            tactic: PolicyTactic = entry["tactic"]  # type: ignore[assignment]
            topology = self._topology_identifier(tactic)
            if previous_topology is not None and topology == previous_topology:
                continue
            if fallback_entry is None:
                fallback_entry = entry
            bias = float(tactic.regime_bias.get(regime_state.regime, 0.0))
            if bias > 0.0 and bias >= best_bias:
                preferred_entry = entry
                best_bias = bias

        candidate = preferred_entry or fallback_entry
        if candidate is None:
            return default_entry, False
        if candidate is default_entry:
            return default_entry, False
        return candidate, True

    def _build_regime_transition_summary(
        self,
        *,
        regime_state: RegimeState,
        decision_timestamp: datetime,
        selected_tactic: PolicyTactic,
        ranked: Sequence[Mapping[str, object]],
        switch_forced: bool,
    ) -> Mapping[str, object]:
        previous_regime = self._last_regime
        previous_topology = self._last_topology
        previous_tactic = self._last_tactic_id
        selected_topology = self._topology_identifier(selected_tactic)

        regime_changed = (
            previous_regime is not None and regime_state.regime != previous_regime
        )

        if regime_state.timestamp.tzinfo is None:
            regime_timestamp = regime_state.timestamp.replace(tzinfo=timezone.utc)
        else:
            regime_timestamp = regime_state.timestamp.astimezone(timezone.utc)

        decision_time = (
            decision_timestamp.replace(tzinfo=timezone.utc)
            if decision_timestamp.tzinfo is None
            else decision_timestamp.astimezone(timezone.utc)
        )

        latency_ms = max(
            0.0,
            (decision_time - regime_timestamp).total_seconds() * 1000.0,
        )

        if previous_topology is None:
            topology_changed: bool | None = None
        else:
            topology_changed = selected_topology != previous_topology

        transition: dict[str, object] = {
            "decision_timestamp": decision_time.isoformat(),
            "regime_timestamp": regime_timestamp.isoformat(),
            "current_regime": regime_state.regime,
            "previous_regime": previous_regime,
            "latency_ms": latency_ms,
            "deadline_ms": float(self._regime_switch_deadline_ms),
            "met_deadline": (
                latency_ms <= self._regime_switch_deadline_ms if regime_changed else None
            ),
            "previous_tactic": previous_tactic,
            "current_tactic": selected_tactic.tactic_id,
            "previous_topology": previous_topology,
            "current_topology": selected_topology,
            "topology_changed": topology_changed,
            "switch_forced": bool(switch_forced and regime_changed),
            "regime_changed": regime_changed,
        }

        if regime_changed:
            candidates: list[Mapping[str, object]] = []
            for entry in ranked[: self._summary_top_k]:
                tactic: PolicyTactic = entry["tactic"]  # type: ignore[assignment]
                candidates.append(
                    {
                        "tactic_id": tactic.tactic_id,
                        "topology": self._topology_identifier(tactic),
                        "score": float(entry.get("score", 0.0)),
                    }
                )
            transition["topology_candidates"] = candidates

        return transition

    @staticmethod
    def _parse_timestamp(value: object) -> datetime | None:
        if not isinstance(value, str):
            return None
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError:
            return None
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)


__all__ = [
    "PolicyDecision",
    "PolicyRouter",
    "PolicyTactic",
    "FastWeightExperiment",
    "RegimeState",
]
