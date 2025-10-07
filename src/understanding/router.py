"""UnderstandingRouter fast-weight adapters with feature gating.

This module implements the roadmap task for the understanding loop sprint by
providing a façade around the `PolicyRouter`. The façade ingests rich belief
snapshots, evaluates feature-gated fast-weight adapters, and orchestrates the
underlying policy routing call so downstream strategy slots see deterministic
fast-weight behaviour tied to documented gates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Mapping, MutableMapping, Sequence

from src.thinking.adaptation.policy_router import (
    PolicyDecision,
    PolicyRouter,
    PolicyTactic,
    RegimeState,
)


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
class FastWeightAdapter:
    """Adapter that applies a fast-weight multiplier when gates pass."""

    adapter_id: str
    tactic_id: str
    multiplier: float
    rationale: str
    feature_gates: Sequence[FeatureGate] = field(default_factory=tuple)
    required_flags: Mapping[str, bool] = field(default_factory=dict)
    expires_at: datetime | None = None

    def applies(self, snapshot: BeliefSnapshot, *, as_of: datetime | None = None) -> bool:
        if self.multiplier <= 0.0:
            return False
        if self.expires_at and (as_of or snapshot.regime_state.timestamp) > self.expires_at:
            return False
        for gate in self.feature_gates:
            if not gate.passes(snapshot):
                return False
        for flag, expected in self.required_flags.items():
            if snapshot.flag_enabled(flag) != expected:
                return False
        return True

    def summary(self) -> Mapping[str, object]:
        payload: dict[str, object] = {
            "adapter_id": self.adapter_id,
            "tactic_id": self.tactic_id,
            "multiplier": float(self.multiplier),
            "rationale": self.rationale,
            "feature_gates": [gate.as_dict() for gate in self.feature_gates],
            "required_flags": {flag: bool(expected) for flag, expected in self.required_flags.items()},
        }
        if self.expires_at:
            payload["expires_at"] = self.expires_at.isoformat()
        return payload


@dataclass(frozen=True)
class UnderstandingDecision:
    """Understanding router result with adapter metadata."""

    decision: PolicyDecision
    belief_snapshot: BeliefSnapshot
    applied_adapters: tuple[str, ...]
    fast_weight_summary: Mapping[str, Mapping[str, object]]


class UnderstandingRouter:
    """Route strategies with feature-gated fast-weight adapters."""

    def __init__(self, policy_router: PolicyRouter | None = None) -> None:
        self._router = policy_router or PolicyRouter()
        self._adapters: dict[str, FastWeightAdapter] = {}

    @property
    def policy_router(self) -> PolicyRouter:
        return self._router

    def register_tactic(self, tactic: PolicyTactic) -> None:
        self._router.register_tactic(tactic)

    def register_adapter(self, adapter: FastWeightAdapter) -> None:
        self._adapters[adapter.adapter_id] = adapter

    def remove_adapter(self, adapter_id: str) -> None:
        self._adapters.pop(adapter_id, None)

    def route(
        self,
        snapshot: BeliefSnapshot,
        *,
        fast_weights: Mapping[str, float] | None = None,
        as_of: datetime | None = None,
    ) -> UnderstandingDecision:
        weights: MutableMapping[str, float] = dict(fast_weights or {})

        applied: list[str] = []
        summaries: dict[str, Mapping[str, object]] = {}

        if snapshot.fast_weights_enabled:
            for adapter in self._adapters.values():
                if not adapter.applies(snapshot, as_of=as_of):
                    continue
                applied.append(adapter.adapter_id)
                current = float(weights.get(adapter.tactic_id, 1.0))
                weights[adapter.tactic_id] = current * adapter.multiplier
                summaries[adapter.adapter_id] = adapter.summary()

        decision = self._router.route(
            snapshot.regime_state,
            fast_weights=dict(weights) if weights else None,
        )

        return UnderstandingDecision(
            decision=decision,
            belief_snapshot=snapshot,
            applied_adapters=tuple(applied),
            fast_weight_summary=summaries,
        )


__all__ = [
    "BeliefSnapshot",
    "FastWeightAdapter",
    "FeatureGate",
    "UnderstandingDecision",
    "UnderstandingRouter",
]
