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
from dataclasses import dataclass, field, replace
from numbers import Real
from typing import Any, Callable, Deque, Iterable, Mapping, MutableMapping, Sequence
from src.evolution.mutation_ledger import get_mutation_ledger

from src.evolution.feature_flags import EvolutionFeatureFlags
from src.governance.policy_ledger import PolicyLedgerStage
from src.trading.strategies.catalog_loader import StrategyCatalog, load_strategy_catalog
from src.thinking.adaptation.operator_constraints import (
    OperatorConstraint,
    OperatorConstraintSet,
    OperatorConstraintViolation,
    OperatorContext,
    parse_operator_constraints,
)
from src.thinking.adaptation.policy_router import PolicyDecision, PolicyRouter, PolicyTactic, RegimeState

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StrategyVariant:
    """Definition of a tactic variant that can be trialled by the evolution manager."""

    variant_id: str
    tactic: PolicyTactic
    base_tactic_id: str
    rationale: str | None = None
    trial_weight_multiplier: float = 1.0
    exploration: bool = True
    metadata: Mapping[str, object] | None = None


@dataclass(frozen=True)
class CatalogueVariantRequest:
    """Definition for sourcing a variant from the strategy catalogue."""

    strategy_id: str
    rationale: str | None = None
    weight_multiplier: float = 1.0
    guardrails: Mapping[str, object] | None = None
    parameter_overrides: Mapping[str, object] | None = None
    tactic_id: str | None = None
    description: str | None = None
    objectives: Sequence[str] | None = None
    confidence_sensitivity: float | None = None
    base_weight: float | None = None


@dataclass(frozen=True)
class ParameterMutation:
    """Definition of a lightweight parameter mutation applied to the base tactic."""

    parameter: str
    scale: float | None = None
    offset: float | None = None
    min_value: float | None = None
    max_value: float | None = None
    suffix: str | None = None
    rationale: str | None = None
    weight_multiplier: float = 1.0


@dataclass(frozen=True)
class ManagedStrategyConfig:
    """Configuration describing how the manager should evolve a tactic."""

    base_tactic_id: str
    fallback_variants: Sequence[StrategyVariant] = ()
    degrade_multiplier: float = 0.6
    catalogue_variants: Sequence[CatalogueVariantRequest] = ()
    parameter_mutations: Sequence[ParameterMutation] = ()
    operator_constraints: (
        OperatorConstraintSet
        | Sequence[OperatorConstraint]
        | Mapping[str, Any]
        | OperatorConstraint
        | None
    ) = None


@dataclass(slots=True)
class _ManagedStrategyState:
    config: ManagedStrategyConfig
    outcomes: Deque[int]
    trial_queue: Deque[StrategyVariant]
    introduced_variants: MutableMapping[str, StrategyVariant]
    mutation_index: int = 0
    constraints: OperatorConstraintSet | None = None

    def record(self, win: bool) -> None:
        self.outcomes.append(1 if win else 0)

    def next_variant(self) -> StrategyVariant | None:
        while self.trial_queue:
            candidate = self.trial_queue.popleft()
            if candidate.variant_id not in self.introduced_variants:
                self.introduced_variants[candidate.variant_id] = candidate
                return candidate
        return None


@dataclass(frozen=True)
class EvolutionAdaptationResult:
    """Summary emitted when the manager applies an adaptation."""

    base_tactic_id: str
    actions: tuple[Mapping[str, object], ...]
    win_rate: float
    observations: int
    metadata: Mapping[str, object] = field(default_factory=dict)

    def as_dict(self) -> Mapping[str, object]:
        payload: dict[str, object] = {
            "base_tactic_id": self.base_tactic_id,
            "win_rate": float(self.win_rate),
            "observations": int(self.observations),
            "actions": [dict(action) for action in self.actions],
        }
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload


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
        catalogue: StrategyCatalog | None = None,
        catalogue_loader: Callable[[], StrategyCatalog] | None = None,
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
        self._catalogue = catalogue
        self._catalogue_loader = catalogue_loader or load_strategy_catalog

        for config in strategies:
            base_id = config.base_tactic_id
            trial_variants: list[StrategyVariant] = list(config.fallback_variants)
            if config.catalogue_variants:
                for request in config.catalogue_variants:
                    variant = self._build_catalogue_variant(base_id, request)
                    if variant is not None:
                        trial_variants.append(variant)
            constraint_set = self._resolve_constraints(config.operator_constraints)
            self._states[base_id] = _ManagedStrategyState(
                config=config,
                outcomes=deque(maxlen=window_size),
                trial_queue=deque(trial_variants),
                introduced_variants={},
                constraints=constraint_set,
            )

    @staticmethod
    def _resolve_constraints(
        source: OperatorConstraintSet
        | Sequence[OperatorConstraint]
        | Mapping[str, Any]
        | OperatorConstraint
        | None,
    ) -> OperatorConstraintSet | None:
        if source is None:
            return None
        if isinstance(source, OperatorConstraintSet):
            return source
        if isinstance(source, OperatorConstraint):
            return OperatorConstraintSet((source,))
        if isinstance(source, Mapping):
            return parse_operator_constraints(source)
        if isinstance(source, Sequence) and not isinstance(source, (str, bytes, bytearray)):
            constraints: list[OperatorConstraint] = []
            for item in source:
                if isinstance(item, OperatorConstraint):
                    constraints.append(item)
                elif isinstance(item, Mapping):
                    constraints.append(OperatorConstraint(**dict(item)))
                else:
                    raise TypeError(
                        "Operator constraints sequence must contain mappings or OperatorConstraint instances",
                    )
            if not constraints:
                return None
            return OperatorConstraintSet(tuple(constraints))
        raise TypeError(
            f"Unsupported operator_constraints payload type: {type(source)!r}",
        )

    def observe_iteration(
        self,
        *,
        decision: PolicyDecision,
        stage: PolicyLedgerStage,
        outcomes: Mapping[str, object] | None,
        metadata: Mapping[str, object] | None = None,
        regime_state: RegimeState | None = None,
    ) -> EvolutionAdaptationResult | None:
        """Process a decision diary outcome produced by the AlphaTrade loop."""

        state = self._states.get(decision.tactic_id)
        if state is None:
            return None
        if self._paper_only and stage is not PolicyLedgerStage.PAPER:
            return None
        if not self._feature_flags.adaptive_runs_enabled(override=self._adaptive_override):
            return None

        win = self._derive_outcome(outcomes)
        if win is None:
            return None
        state.record(win)

        observations = len(state.outcomes)
        if observations < self._min_observations:
            return None

        win_rate = sum(state.outcomes) / observations
        if win_rate >= self._win_rate_threshold:
            return None

        logger.info(
            "EvolutionManager triggering adaptation for tactic %s: win_rate=%.3f (< %.3f)",
            decision.tactic_id,
            win_rate,
            self._win_rate_threshold,
        )
        actions, violations = self._run_adaptation(
            decision.tactic_id,
            state,
            metadata=metadata,
            stage=stage,
            regime_state=regime_state,
        )
        state.outcomes.clear()
        if not actions and not violations:
            return None

        summary_metadata: dict[str, object] = {
            "threshold": float(self._win_rate_threshold),
            "window_size": self._window_size,
        }
        if metadata:
            summary_metadata.update({str(key): value for key, value in metadata.items()})
        if violations:
            summary_metadata["operator_constraints"] = [
                violation.as_dict() for violation in violations
            ]

        result_actions: list[Mapping[str, object]] = list(actions)
        if violations:
            for violation in violations:
                entry: dict[str, object] = {
                    "action": "operator_constraint_blocked",
                    "operation": violation.operation,
                    "reason": violation.reason,
                }
                if violation.details:
                    entry["details"] = dict(violation.details)
                result_actions.append(entry)

        return EvolutionAdaptationResult(
            base_tactic_id=decision.tactic_id,
            actions=tuple(result_actions),
            win_rate=float(win_rate),
            observations=observations,
            metadata=summary_metadata,
        )

    def _run_adaptation(
        self,
        tactic_id: str,
        state: _ManagedStrategyState,
        *,
        metadata: Mapping[str, object] | None,
        stage: PolicyLedgerStage,
        regime_state: RegimeState | None,
    ) -> tuple[list[Mapping[str, object]], list[OperatorConstraintViolation]]:
        actions: list[Mapping[str, object]] = []
        violations: list[OperatorConstraintViolation] = []
        constraints = state.constraints

        variant = state.next_variant()
        if variant is not None:
            allowed, variant_violations = self._constraints_allow(
                constraints=constraints,
                operation="register_variant",
                stage=stage,
                regime_state=regime_state,
                parameters=variant.tactic.parameters,
                metadata={
                    "variant_id": variant.variant_id,
                    "base_tactic_id": variant.base_tactic_id,
                },
            )
            if not allowed:
                state.introduced_variants.pop(variant.variant_id, None)
                state.trial_queue.appendleft(variant)
                self._log_constraint_violations(tactic_id, variant_violations)
                violations.extend(variant_violations)
            else:
                registered = self._register_variant(variant, metadata=metadata)
                if registered is not None:
                    actions.append(registered)
            degrade_action, degrade_violations = self._degrade_base(
                tactic_id,
                state.config.degrade_multiplier,
                constraints=constraints,
                stage=stage,
                regime_state=regime_state,
            )
            if degrade_action is not None:
                actions.append(degrade_action)
            if degrade_violations:
                self._log_constraint_violations(tactic_id, degrade_violations)
                violations.extend(degrade_violations)
            return actions, violations

        mutation_variant = self._build_mutation_variant(tactic_id, state)
        if mutation_variant is not None:
            allowed, mutation_violations = self._constraints_allow(
                constraints=constraints,
                operation="register_variant",
                stage=stage,
                regime_state=regime_state,
                parameters=mutation_variant.tactic.parameters,
                metadata={
                    "variant_id": mutation_variant.variant_id,
                    "base_tactic_id": mutation_variant.base_tactic_id,
                },
            )
            if not allowed:
                state.introduced_variants.pop(mutation_variant.variant_id, None)
                self._log_constraint_violations(tactic_id, mutation_violations)
                violations.extend(mutation_violations)
            else:
                registered = self._register_variant(mutation_variant, metadata=metadata)
                if registered is not None:
                    actions.append(registered)
            degrade_action, degrade_violations = self._degrade_base(
                tactic_id,
                state.config.degrade_multiplier,
                constraints=constraints,
                stage=stage,
                regime_state=regime_state,
            )
            if degrade_action is not None:
                actions.append(degrade_action)
            if degrade_violations:
                self._log_constraint_violations(tactic_id, degrade_violations)
                violations.extend(degrade_violations)
            return actions, violations

        degrade_action, degrade_violations = self._degrade_base(
            tactic_id,
            state.config.degrade_multiplier,
            constraints=constraints,
            stage=stage,
            regime_state=regime_state,
        )
        if degrade_action is not None:
            actions.append(degrade_action)
        if degrade_violations:
            self._log_constraint_violations(tactic_id, degrade_violations)
            violations.extend(degrade_violations)
        return actions, violations

    @staticmethod
    def _log_constraint_violations(
        tactic_id: str,
        violations: Sequence[OperatorConstraintViolation],
    ) -> None:
        for violation in violations:
            logger.info(
                "Operator constraint blocked %s for tactic %s: %s",
                violation.operation,
                tactic_id,
                violation.as_dict(),
            )

    def _constraints_allow(
        self,
        *,
        constraints: OperatorConstraintSet | None,
        operation: str,
        stage: PolicyLedgerStage,
        regime_state: RegimeState | None,
        parameters: Mapping[str, Any] | None,
        metadata: Mapping[str, Any] | None,
    ) -> tuple[bool, tuple[OperatorConstraintViolation, ...]]:
        if constraints is None:
            return True, ()
        features: Mapping[str, float]
        confidence: float | None
        regime: str | None
        if regime_state is None:
            features = {}
            confidence = None
            regime = None
        else:
            features = dict(regime_state.features or {})
            confidence = regime_state.confidence
            regime = regime_state.regime
        context = OperatorContext(
            operation=operation,
            stage=stage,
            regime=regime,
            regime_confidence=confidence,
            regime_features=features,
            parameters=dict(parameters or {}),
            metadata=dict(metadata or {}),
        )
        return constraints.validate(context)

    def _register_variant(
        self,
        variant: StrategyVariant,
        *,
        metadata: Mapping[str, object] | None,
    ) -> Mapping[str, object] | None:
        tactic = variant.tactic
        if variant.exploration:
            tactic = self._ensure_exploration_tactic(tactic)
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
        payload: dict[str, object] = {
            "action": "register_variant",
            "tactic_id": tactic.tactic_id,
            "base_tactic_id": variant.base_tactic_id,
            "base_weight": float(tactic.base_weight),
            "registration": action,
        }
        if variant.rationale:
            payload["rationale"] = variant.rationale
        if variant.trial_weight_multiplier != 1.0:
            payload["trial_weight_multiplier"] = float(variant.trial_weight_multiplier)
        if tactic.description:
            payload["description"] = tactic.description
        if tactic.tags:
            payload["tags"] = list(tactic.tags)
        payload["guardrails"] = dict(tactic.guardrails)
        payload["parameters"] = dict(tactic.parameters)
        if metadata:
            payload["metadata"] = {str(key): value for key, value in metadata.items()}
        if variant.metadata:
            payload["variant_metadata"] = {str(key): value for key, value in variant.metadata.items()}

        self._record_mutation_event(
            variant=variant,
            tactic=tactic,
            metadata=metadata,
        )
        return payload

    def _record_mutation_event(
        self,
        *,
        variant: StrategyVariant,
        tactic: PolicyTactic,
        metadata: Mapping[str, object] | None,
    ) -> None:
        variant_metadata = variant.metadata
        if not isinstance(variant_metadata, Mapping):
            return
        mutation_meta = variant_metadata.get("parameter_mutation")
        if not isinstance(mutation_meta, Mapping):
            return

        parameter = str(mutation_meta.get("parameter", "")).strip()
        original_value = mutation_meta.get("original_value")
        mutated_value = mutation_meta.get("mutated_value")

        ledger_metadata: dict[str, object] = {
            key: value
            for key, value in mutation_meta.items()
            if key not in {"parameter", "original_value", "mutated_value"}
        }
        if metadata:
            ledger_metadata["trigger_metadata"] = {str(key): value for key, value in metadata.items()}
        try:
            get_mutation_ledger().record_parameter_mutation(
                base_tactic_id=variant.base_tactic_id,
                variant_id=tactic.tactic_id,
                parameter=parameter or mutation_meta.get("parameter"),
                original_value=original_value,
                mutated_value=mutated_value,
                metadata=ledger_metadata or None,
            )
        except Exception:  # pragma: no cover - ledger persistence is best-effort
            logger.debug(
                "Mutation ledger recording failed for variant %s", tactic.tactic_id, exc_info=True
            )

    @staticmethod
    def _ensure_exploration_tactic(tactic: PolicyTactic) -> PolicyTactic:
        has_tag = "exploration" in {str(tag) for tag in tactic.tags}
        if tactic.exploration and has_tag:
            return tactic
        tags = tuple(dict.fromkeys((*tactic.tags, "exploration"))) if tactic.tags else ("exploration",)
        if tactic.exploration:
            return replace(tactic, tags=tags)
        return replace(tactic, exploration=True, tags=tags)

    def _degrade_base(
        self,
        tactic_id: str,
        multiplier: float,
        *,
        constraints: OperatorConstraintSet | None,
        stage: PolicyLedgerStage,
        regime_state: RegimeState | None,
    ) -> tuple[Mapping[str, object] | None, tuple[OperatorConstraintViolation, ...]]:
        if multiplier <= 0:
            return None, ()

        tactics = self._router.tactics()
        base = tactics.get(tactic_id)
        if base is None:
            logger.debug("Cannot degrade missing tactic %s", tactic_id)
            return None, ()

        new_weight = max(0.0, base.base_weight * multiplier)
        if new_weight == base.base_weight:
            return None, ()

        allowed, violations = self._constraints_allow(
            constraints=constraints,
            operation="degrade_base",
            stage=stage,
            regime_state=regime_state,
            parameters={
                "original_weight": base.base_weight,
                "target_weight": new_weight,
                "multiplier": multiplier,
            },
            metadata={"tactic_id": tactic_id},
        )
        if not allowed:
            return None, violations

        updated = replace(base, base_weight=new_weight)
        self._router.update_tactic(updated)
        logger.info(
            "EvolutionManager degraded base weight for %s from %.4f to %.4f",
            tactic_id,
            base.base_weight,
            new_weight,
        )
        return (
            {
                "action": "degrade_base",
                "tactic_id": tactic_id,
                "from_weight": float(base.base_weight),
                "to_weight": float(new_weight),
                "multiplier": float(multiplier),
            },
            (),
        )

    def _build_catalogue_variant(
        self,
        base_tactic_id: str,
        request: CatalogueVariantRequest,
    ) -> StrategyVariant | None:
        catalogue = self._ensure_catalogue()
        if catalogue is None:
            logger.debug("Catalogue variant requested but no catalogue available")
            return None

        definition = catalogue.get_definition(request.strategy_id)
        if definition is None:
            definition = catalogue.get_definition_by_key(request.strategy_id)
        if definition is None:
            logger.warning(
                "EvolutionManager could not resolve catalogue strategy '%s'", request.strategy_id
            )
            return None

        tactic_id = request.tactic_id or f"{definition.identifier}__trial"
        parameters = dict(definition.parameters)
        if request.parameter_overrides:
            parameters.update({str(k): v for k, v in request.parameter_overrides.items()})
        parameters.setdefault("symbols", tuple(definition.symbols))
        parameters.setdefault("catalogue_identifier", definition.identifier)
        parameters.setdefault("catalogue_key", definition.key)
        parameters.setdefault("catalogue_version", catalogue.version)

        guardrails: dict[str, object] = {
            "force_paper": True,
            "requires_diary": True,
            "catalogue_version": catalogue.version,
        }
        if request.guardrails:
            guardrails.update({str(k): v for k, v in request.guardrails.items()})

        confidence = (
            request.confidence_sensitivity
            if request.confidence_sensitivity is not None
            else 0.5
        )
        base_weight = (
            request.base_weight
            if request.base_weight is not None
            else self._normalise_catalogue_weight(definition.capital)
        )
        objectives = tuple(request.objectives) if request.objectives else definition.tags

        tactic = PolicyTactic(
            tactic_id=tactic_id,
            base_weight=float(max(0.0, base_weight)),
            parameters=parameters,
            guardrails=guardrails,
            description=request.description or definition.description,
            objectives=objectives,
            tags=definition.tags,
            confidence_sensitivity=float(confidence),
        )

        rationale = request.rationale or f"Catalogue variant {definition.name}"
        return StrategyVariant(
            variant_id=tactic_id,
            tactic=tactic,
            base_tactic_id=base_tactic_id,
            rationale=rationale,
            trial_weight_multiplier=request.weight_multiplier,
        )

    def _build_mutation_variant(
        self,
        tactic_id: str,
        state: _ManagedStrategyState,
    ) -> StrategyVariant | None:
        config = state.config
        if not config.parameter_mutations:
            return None
        tactics = self._router.tactics()
        base = tactics.get(tactic_id)
        if base is None:
            logger.debug("Cannot mutate missing tactic %s", tactic_id)
            return None
        base_parameters = dict(base.parameters)
        for mutation in config.parameter_mutations:
            parameter = mutation.parameter
            original = base_parameters.get(parameter)
            if not isinstance(original, Real):
                logger.debug(
                    "Skipping mutation for %s parameter %s (non-numeric or missing)",
                    tactic_id,
                    parameter,
                )
                continue

            mutated_value = float(original)
            if mutation.scale is not None:
                mutated_value *= float(mutation.scale)
            if mutation.offset is not None:
                mutated_value += float(mutation.offset)
            if mutation.min_value is not None:
                mutated_value = max(float(mutation.min_value), mutated_value)
            if mutation.max_value is not None:
                mutated_value = min(float(mutation.max_value), mutated_value)

            mutated_params = dict(base_parameters)
            mutated_params[parameter] = mutated_value

            state.mutation_index += 1
            suffix = mutation.suffix or parameter
            variant_id = f"{tactic_id}__mut_{suffix}_{state.mutation_index}"
            if variant_id in state.introduced_variants:
                # Extremely defensive, but keeps identifiers unique.
                continue

            adjusted_weight = max(0.0, base.base_weight * mutation.weight_multiplier)
            mutated_tactic = replace(
                base,
                tactic_id=variant_id,
                parameters=mutated_params,
                base_weight=adjusted_weight,
            )

            rationale = mutation.rationale or (
                f"Auto-mutated {parameter} from {original} to {mutated_value:.4f}"
            )
            mutation_details: dict[str, object] = {
                "parameter": parameter,
                "original_value": float(original) if original is not None else None,
                "mutated_value": mutated_value,
                "mutation_index": state.mutation_index,
                "suffix": suffix,
            }
            if mutation.scale is not None:
                mutation_details["scale"] = float(mutation.scale)
            if mutation.offset is not None:
                mutation_details["offset"] = float(mutation.offset)
            if mutation.min_value is not None:
                mutation_details["min_value"] = float(mutation.min_value)
            if mutation.max_value is not None:
                mutation_details["max_value"] = float(mutation.max_value)
            if mutation.weight_multiplier != 1.0:
                mutation_details["weight_multiplier"] = float(mutation.weight_multiplier)
            variant = StrategyVariant(
                variant_id=variant_id,
                tactic=mutated_tactic,
                base_tactic_id=tactic_id,
                rationale=rationale,
                trial_weight_multiplier=1.0,
                metadata={"parameter_mutation": mutation_details},
            )
            state.introduced_variants[variant_id] = variant
            return variant
        return None

    def _ensure_catalogue(self) -> StrategyCatalog | None:
        if self._catalogue is not None:
            return self._catalogue
        loader = self._catalogue_loader
        if loader is None:
            return None
        try:
            self._catalogue = loader()
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning("EvolutionManager failed to load strategy catalogue: %s", exc, exc_info=exc)
            self._catalogue_loader = None
            return None
        return self._catalogue

    def _normalise_catalogue_weight(self, capital: float) -> float:
        catalogue = self._catalogue
        if catalogue is None:
            return max(0.0, float(capital)) or 1.0
        default_capital = float(getattr(catalogue, "default_capital", 0.0))
        if default_capital <= 0:
            return max(0.0, float(capital)) or 1.0
        return max(0.0, float(capital) / default_capital)

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
    "CatalogueVariantRequest",
    "StrategyVariant",
    "ParameterMutation",
    "EvolutionAdaptationResult",
]
