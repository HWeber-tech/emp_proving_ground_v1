"""UnderstandingRouter fast-weight adapters with feature gating.

This module implements the roadmap task for the understanding loop sprint by
providing a façade around the `PolicyRouter`. The façade ingests rich belief
snapshots, evaluates feature-gated fast-weight adapters (including
Hebbian-inspired updates with deterministic decay), and orchestrates the
underlying policy routing call so downstream strategy slots see auditable
fast-weight behaviour tied to documented gates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Mapping, MutableMapping, Sequence, TYPE_CHECKING

from src.governance.system_config import RunMode
from src.thinking.adaptation.policy_router import (
    PolicyDecision,
    PolicyRouter,
    PolicyTactic,
    RegimeState,
)
from src.thinking.adaptation.fast_weights import (
    FastWeightController,
    build_fast_weight_controller,
)
from src.thinking.adaptation.feature_toggles import LINEAR_ATTENTION_FLAG

if TYPE_CHECKING:  # pragma: no cover - typing support
    from src.governance.system_config import SystemConfig


@dataclass(frozen=True)
class BeliefSnapshot:
    """Belief snapshot produced by the understanding loop pipeline."""

    belief_id: str
    regime_state: RegimeState
    features: Mapping[str, float]
    metadata: Mapping[str, object] = field(default_factory=dict)
    fast_weights_enabled: bool = True
    feature_flags: Mapping[str, bool] | None = None

    def feature_value(self, name: str) -> float | None:
        """Return the feature value if known, otherwise ``None``."""

        value = self.features.get(name)
        if value is None:
            return None
        return float(value)

    def flag_enabled(self, flag: str) -> bool:
        """Return True when the requested feature flag is enabled."""

        if not self.feature_flags:
            return False
        return bool(self.feature_flags.get(flag, False))


@dataclass(frozen=True)
class FeatureGate:
    """Inclusive bounds for a single feature used to gate fast-weight adapters."""

    feature: str
    minimum: float | None = None
    maximum: float | None = None

    def passes(self, snapshot: BeliefSnapshot) -> bool:
        value = snapshot.feature_value(self.feature)
        if value is None:
            return False
        if self.minimum is not None and value < self.minimum:
            return False
        if self.maximum is not None and value > self.maximum:
            return False
        return True

    def as_dict(self) -> Mapping[str, object]:
        payload: dict[str, object] = {"feature": self.feature}
        if self.minimum is not None:
            payload["minimum"] = float(self.minimum)
        if self.maximum is not None:
            payload["maximum"] = float(self.maximum)
        return payload


@dataclass(frozen=True)
class HebbianConfig:
    """Parameters for Hebbian-inspired multiplier updates with decay."""

    feature: str
    learning_rate: float = 0.2
    decay: float = 0.1
    baseline: float = 1.0
    floor: float = 0.0
    ceiling: float | None = None

    def update(self, previous: float | None, observation: float) -> tuple[float, float]:
        """Return the previous and updated multipliers after applying decay."""

        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if not 0.0 < self.decay <= 1.0:
            raise ValueError("decay must be in (0, 1]")

        prior = self.baseline if previous is None else float(previous)
        updated = (1.0 - self.decay) * prior + self.learning_rate * float(observation)

        if self.ceiling is not None:
            updated = min(updated, float(self.ceiling))
        updated = max(updated, self.floor)
        return prior, updated

    def as_dict(self) -> Mapping[str, object]:
        payload: dict[str, object] = {
            "feature": self.feature,
            "learning_rate": float(self.learning_rate),
            "decay": float(self.decay),
            "baseline": float(self.baseline),
            "floor": float(self.floor),
        }
        if self.ceiling is not None:
            payload["ceiling"] = float(self.ceiling)
        return payload


@dataclass(frozen=True)
class FastWeightAdapter:
    """Adapter that applies a fast-weight multiplier when gates pass."""

    adapter_id: str
    tactic_id: str
    rationale: str
    multiplier: float | None = None
    feature_gates: Sequence[FeatureGate] = field(default_factory=tuple)
    required_flags: Mapping[str, bool] = field(default_factory=dict)
    expires_at: datetime | None = None
    hebbian: HebbianConfig | None = None

    def applies(self, snapshot: BeliefSnapshot, *, as_of: datetime | None = None) -> bool:
        if self.hebbian is None:
            if self.multiplier is None or self.multiplier <= 0.0:
                return False
        if self.expires_at and (as_of or snapshot.regime_state.timestamp) > self.expires_at:
            return False
        for gate in self.feature_gates:
            if not gate.passes(snapshot):
                return False
        for flag, expected in self.required_flags.items():
            if snapshot.flag_enabled(flag) != expected:
                return False
        if self.hebbian:
            value = snapshot.feature_value(self.hebbian.feature)
            if value is None:
                return False
        return True

    def summary(self) -> Mapping[str, object]:
        payload: dict[str, object] = {
            "adapter_id": self.adapter_id,
            "tactic_id": self.tactic_id,
            "rationale": self.rationale,
            "feature_gates": [gate.as_dict() for gate in self.feature_gates],
            "required_flags": {flag: bool(expected) for flag, expected in self.required_flags.items()},
        }
        if self.multiplier is not None:
            payload["multiplier"] = float(self.multiplier)
        if self.expires_at:
            payload["expires_at"] = self.expires_at.isoformat()
        if self.hebbian:
            payload["hebbian"] = dict(self.hebbian.as_dict())
        return payload


@dataclass(frozen=True)
class UnderstandingDecision:
    """Understanding router result with adapter metadata."""

    decision: PolicyDecision
    belief_snapshot: BeliefSnapshot
    applied_adapters: tuple[str, ...]
    fast_weight_summary: Mapping[str, Mapping[str, object]]
    fast_weight_metrics: Mapping[str, object]


class UnderstandingRouter:
    """Route strategies with feature-gated fast-weight adapters."""

    def __init__(
        self,
        policy_router: PolicyRouter | None = None,
        *,
        fast_weight_controller: FastWeightController | None = None,
    ) -> None:
        if policy_router is not None and fast_weight_controller is not None:
            raise ValueError(
                "Provide either policy_router or fast_weight_controller when constructing UnderstandingRouter",
            )
        if policy_router is None:
            policy_router = PolicyRouter(fast_weight_controller=fast_weight_controller)
        self._router = policy_router
        self._adapters: dict[str, FastWeightAdapter] = {}
        self._heb_state: dict[str, float] = {}

    @classmethod
    def from_system_config(
        cls,
        config: "SystemConfig",
        *,
        default_guardrails: Mapping[str, object] | None = None,
        reflection_history: int = 50,
        summary_top_k: int = 3,
    ) -> "UnderstandingRouter":
        """Build a router with fast-weight constraints sourced from configuration."""

        extras = dict(config.extras or {})

        if config.run_mode is RunMode.live:
            extras["EXPLORATION_MAX_FRACTION"] = 0.0
            extras.pop("EXPLORATION_MUTATE_EVERY", None)

        controller = build_fast_weight_controller(extras)

        def _parse_fraction(value: object | None) -> float | None:
            if value is None:
                return None
            try:
                fraction = float(value)
            except (TypeError, ValueError):
                return None
            if not 0.0 <= fraction <= 1.0:
                return None
            return fraction

        def _parse_int(value: object | None) -> int | None:
            if value is None:
                return None
            try:
                integer = int(str(value))
            except (TypeError, ValueError):
                return None
            if integer <= 0:
                return None
            return integer

        exploration_fraction = _parse_fraction(extras.get("EXPLORATION_MAX_FRACTION"))
        exploration_cadence = _parse_int(extras.get("EXPLORATION_MUTATE_EVERY"))

        if config.run_mode is RunMode.live:
            exploration_fraction = 0.0
            exploration_cadence = None
        allow_forced_exploration = config.run_mode is not RunMode.live

        policy_router = PolicyRouter(
            default_guardrails=default_guardrails,
            reflection_history=reflection_history,
            summary_top_k=summary_top_k,
            fast_weight_controller=controller,
            exploration_max_fraction=exploration_fraction,
            exploration_mutate_every=exploration_cadence,
            allow_forced_exploration=allow_forced_exploration,
        )
        return cls(policy_router=policy_router)

    @property
    def policy_router(self) -> PolicyRouter:
        return self._router

    def exploration_freeze_active(self) -> bool:
        """Return ``True`` when policy routing is currently frozen for exploration."""

        return self._router.exploration_freeze_active()

    def exploration_freeze_state(self) -> Mapping[str, object]:
        """Expose the policy router's exploration freeze state."""

        return self._router.exploration_freeze_state()

    def freeze_exploration(
        self,
        *,
        reason: str,
        triggered_by: str,
        severity: str | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> None:
        """Freeze exploration tactics on the underlying policy router."""

        self._router.freeze_exploration(
            reason=reason,
            triggered_by=triggered_by,
            severity=severity,
            metadata=metadata,
        )

    def release_exploration(
        self,
        *,
        reason: str | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> None:
        """Release an active exploration freeze on the policy router."""

        self._router.release_exploration(reason=reason, metadata=metadata)

    def register_tactic(self, tactic: PolicyTactic) -> None:
        self._router.register_tactic(tactic)

    def register_adapter(self, adapter: FastWeightAdapter) -> None:
        self._adapters[adapter.adapter_id] = adapter
        if adapter.hebbian:
            self._heb_state.setdefault(adapter.adapter_id, adapter.hebbian.baseline)
        else:
            self._heb_state.pop(adapter.adapter_id, None)

    def remove_adapter(self, adapter_id: str) -> None:
        self._adapters.pop(adapter_id, None)
        self._heb_state.pop(adapter_id, None)

    def route(
        self,
        snapshot: BeliefSnapshot,
        *,
        fast_weights: Mapping[str, float] | None = None,
        as_of: datetime | None = None,
    ) -> UnderstandingDecision:
        weights: MutableMapping[str, float] = {}
        if snapshot.fast_weights_enabled and fast_weights:
            for tactic_id, value in fast_weights.items():
                try:
                    weights[str(tactic_id)] = float(value)
                except (TypeError, ValueError):
                    continue
        elif fast_weights:
            # Feature flag explicitly disables all fast-weight multipliers, so any
            # externally supplied overrides are ignored to restore baseline routing.
            weights.clear()

        applied: list[str] = []
        summaries: dict[str, Mapping[str, object]] = {}

        if snapshot.fast_weights_enabled:
            for adapter in self._adapters.values():
                if not adapter.applies(snapshot, as_of=as_of):
                    continue

                applied.append(adapter.adapter_id)
                current_weight = float(weights.get(adapter.tactic_id, 1.0))

                multiplier = adapter.multiplier if adapter.multiplier is not None else 1.0
                summary = dict(adapter.summary())

                if adapter.hebbian:
                    previous_value = self._heb_state.get(adapter.adapter_id)
                    feature_value = snapshot.feature_value(adapter.hebbian.feature)
                    # Feature value is guaranteed by applies(), but guard for mypy/static flow.
                    if feature_value is None:
                        continue
                    prior_multiplier, updated_multiplier = adapter.hebbian.update(
                        previous_value,
                        feature_value,
                    )
                    multiplier = updated_multiplier
                    self._heb_state[adapter.adapter_id] = updated_multiplier
                    summary.update(
                        {
                            "previous_multiplier": prior_multiplier,
                            "current_multiplier": updated_multiplier,
                            "feature_value": float(feature_value),
                        }
                    )
                else:
                    summary["current_multiplier"] = multiplier

                weights[adapter.tactic_id] = current_weight * multiplier
                summaries[adapter.adapter_id] = summary

        decision = self._router.route(
            snapshot.regime_state,
            fast_weights=dict(weights) if weights else None,
            decision_timestamp=as_of,
            linear_attention_enabled=snapshot.flag_enabled(LINEAR_ATTENTION_FLAG),
        )

        return UnderstandingDecision(
            decision=decision,
            belief_snapshot=snapshot,
            applied_adapters=tuple(applied),
            fast_weight_summary=summaries,
            fast_weight_metrics=dict(decision.fast_weight_metrics),
        )


__all__ = [
    "BeliefSnapshot",
    "FastWeightAdapter",
    "FeatureGate",
    "HebbianConfig",
    "UnderstandingDecision",
    "UnderstandingRouter",
]
