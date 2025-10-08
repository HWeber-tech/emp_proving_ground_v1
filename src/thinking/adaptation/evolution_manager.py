"""Evolution manager that seeds basic strategy mutation from decision diaries.

This module implements the "Seed the Evolution Engine" roadmap task by
monitoring decision diary outcomes during paper-trade runs and adapting the
policy router when tactics underperform.  It keeps a short performance window
for each managed tactic, derives a simple win/loss outcome from the recorded
metrics, and – when win-rate drifts below a configured threshold – either
introduces a catalogue variant or dampens the existing tactic weight.

Feature gating is handled via :class:`~src.evolution.feature_flags.EvolutionFeatureFlags`
so adaptive trials only run when ``EVOLUTION_ENABLE_ADAPTIVE_RUNS`` is enabled.
The manager is also constrained to paper-trade stages to avoid mutating live
portfolios without explicit promotion.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, replace
from typing import Deque, Iterable, Mapping, MutableMapping, Sequence

from src.evolution.feature_flags import EvolutionFeatureFlags
from src.governance.policy_ledger import PolicyLedgerStage
from src.thinking.adaptation.policy_router import PolicyDecision, PolicyRouter, PolicyTactic

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StrategyVariant:
    """Definition of a tactic variant that can be trialled by the evolution manager."""

    variant_id: str
    tactic: PolicyTactic
    base_tactic_id: str
    rationale: str | None = None
    trial_weight_multiplier: float = 1.0


@dataclass(frozen=True)
class ManagedStrategyConfig:
    """Configuration describing how the manager should evolve a tactic."""

    base_tactic_id: str
    fallback_variants: Sequence[StrategyVariant] = ()
    degrade_multiplier: float = 0.6


@dataclass(slots=True)
class _ManagedStrategyState:
    config: ManagedStrategyConfig
    outcomes: Deque[int]
    trial_queue: Deque[StrategyVariant]
    introduced_variants: MutableMapping[str, StrategyVariant]

    def record(self, win: bool) -> None:
        self.outcomes.append(1 if win else 0)

    def next_variant(self) -> StrategyVariant | None:
        while self.trial_queue:
            candidate = self.trial_queue.popleft()
            if candidate.variant_id not in self.introduced_variants:
                self.introduced_variants[candidate.variant_id] = candidate
                return candidate
        return None


class EvolutionManager:
    """Monitor tactic outcomes and trigger lightweight evolutionary trials."""

    def __init__(
        self,
        *,
        policy_router: PolicyRouter,
        strategies: Iterable[ManagedStrategyConfig],
        window_size: int = 5,
        win_rate_threshold: float = 0.4,
        min_observations: int | None = None,
        feature_flags: EvolutionFeatureFlags | None = None,
        adaptive_override: bool | None = None,
        paper_only: bool = True,
    ) -> None:
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        if not 0.0 < win_rate_threshold <= 1.0:
            raise ValueError("win_rate_threshold must be in (0, 1]")

        self._router = policy_router
        self._window_size = window_size
        self._win_rate_threshold = win_rate_threshold
        self._min_observations = min_observations or window_size
        self._feature_flags = feature_flags or EvolutionFeatureFlags()
        self._adaptive_override = adaptive_override
        self._paper_only = paper_only
        self._states: dict[str, _ManagedStrategyState] = {}

        for config in strategies:
            base_id = config.base_tactic_id
            self._states[base_id] = _ManagedStrategyState(
                config=config,
                outcomes=deque(maxlen=window_size),
                trial_queue=deque(config.fallback_variants),
                introduced_variants={},
            )

    def observe_iteration(
        self,
        *,
        decision: PolicyDecision,
        stage: PolicyLedgerStage,
        outcomes: Mapping[str, object] | None,
        metadata: Mapping[str, object] | None = None,
    ) -> None:
        """Process a decision diary outcome produced by the AlphaTrade loop."""

        state = self._states.get(decision.tactic_id)
        if state is None:
            return
        if self._paper_only and stage is not PolicyLedgerStage.PAPER:
            return
        if not self._feature_flags.adaptive_runs_enabled(override=self._adaptive_override):
            return

        win = self._derive_outcome(outcomes)
        if win is None:
            return
        state.record(win)

        observations = len(state.outcomes)
        if observations < self._min_observations:
            return

        win_rate = sum(state.outcomes) / observations
        if win_rate >= self._win_rate_threshold:
            return

        logger.info(
            "EvolutionManager triggering adaptation for tactic %s: win_rate=%.3f (< %.3f)",
            decision.tactic_id,
            win_rate,
            self._win_rate_threshold,
        )
        self._run_adaptation(decision.tactic_id, state, metadata=metadata)
        state.outcomes.clear()

    def _run_adaptation(
        self,
        tactic_id: str,
        state: _ManagedStrategyState,
        *,
        metadata: Mapping[str, object] | None,
    ) -> None:
        variant = state.next_variant()
        if variant is not None:
            self._register_variant(variant, metadata=metadata)
            self._degrade_base(tactic_id, state.config.degrade_multiplier)
            return
        self._degrade_base(tactic_id, state.config.degrade_multiplier)

    def _register_variant(
        self,
        variant: StrategyVariant,
        *,
        metadata: Mapping[str, object] | None,
    ) -> None:
        tactic = variant.tactic
        if variant.trial_weight_multiplier != 1.0:
            tactic = replace(
                tactic,
                base_weight=max(0.0, tactic.base_weight * variant.trial_weight_multiplier),
            )
        registry = self._router.tactics()
        if tactic.tactic_id in registry:
            self._router.update_tactic(tactic)
            action = "updated"
        else:
            self._router.register_tactic(tactic)
            action = "registered"
        logger.info(
            "EvolutionManager %s variant %s for base %s", action, tactic.tactic_id, variant.base_tactic_id
        )
        if metadata:
            logger.debug("Variant metadata: %s", dict(metadata))

    def _degrade_base(self, tactic_id: str, multiplier: float) -> None:
        if multiplier <= 0:
            return
        tactics = self._router.tactics()
        base = tactics.get(tactic_id)
        if base is None:
            logger.debug("Cannot degrade missing tactic %s", tactic_id)
            return
        new_weight = max(0.0, base.base_weight * multiplier)
        if new_weight == base.base_weight:
            return
        updated = replace(base, base_weight=new_weight)
        self._router.update_tactic(updated)
        logger.info(
            "EvolutionManager degraded base weight for %s from %.4f to %.4f",
            tactic_id,
            base.base_weight,
            new_weight,
        )

    @staticmethod
    def _derive_outcome(outcomes: Mapping[str, object] | None) -> bool | None:
        if outcomes is None:
            return None
        if not isinstance(outcomes, Mapping):
            return None

        win_flag = outcomes.get("win")
        if isinstance(win_flag, bool):
            return win_flag

        numeric_keys = (
            "paper_pnl",
            "paper_return",
            "pnl",
            "return",
            "realised_pnl",
            "alpha_score",
        )
        for key in numeric_keys:
            value = outcomes.get(key)
            if isinstance(value, (int, float)):
                if value > 0:
                    return True
                if value < 0:
                    return False

        win_rate = outcomes.get("win_rate")
        if isinstance(win_rate, (int, float)):
            if 0.0 <= win_rate <= 1.0:
                return win_rate >= 0.5
            return win_rate > 0.0
        return None


__all__ = [
    "EvolutionManager",
    "ManagedStrategyConfig",
    "StrategyVariant",
]
