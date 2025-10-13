"""Constraint helpers that gate adaptation operators by stage and regime context.

The adaptation roadmap calls for mutation operators that respect governance
boundaries (e.g. paper-only experimentation) and regime-aware domain checks
before altering strategy phenotypes.  This module provides a lightweight
constraint engine that evaluates whether an operator may execute in the current
release stage, market regime, and parameter bounds.  Evolution components can
attach constraint sets to their configuration so losing-streak responses (like
registering catalogue variants or mutating parameters) remain inside approved
safety envelopes documented in the context packs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, MutableMapping, Sequence

from src.governance.policy_ledger import PolicyLedgerStage

__all__ = [
    "OperatorContext",
    "OperatorConstraint",
    "OperatorConstraintSet",
    "OperatorConstraintViolation",
    "parse_operator_constraints",
]


def _normalise_stage(value: PolicyLedgerStage | str | None) -> PolicyLedgerStage:
    if isinstance(value, PolicyLedgerStage):
        return value
    return PolicyLedgerStage.from_value(value)


def _normalise_operations(operations: Sequence[str]) -> tuple[str, ...]:
    cleaned: list[str] = []
    seen: set[str] = set()
    for operation in operations:
        text = str(operation).strip()
        if not text:
            continue
        if text in seen:
            continue
        seen.add(text)
        cleaned.append(text)
    if not cleaned:
        raise ValueError("OperatorConstraint requires at least one operation name")
    return tuple(cleaned)


def _normalise_stage_set(stages: Sequence[PolicyLedgerStage | str] | None) -> tuple[PolicyLedgerStage, ...]:
    if not stages:
        return ()
    cleaned: list[PolicyLedgerStage] = []
    seen: set[PolicyLedgerStage] = set()
    for stage in stages:
        resolved = _normalise_stage(stage)
        if resolved in seen:
            continue
        seen.add(resolved)
        cleaned.append(resolved)
    return tuple(cleaned)


def _normalise_regime_set(regimes: Sequence[str] | None) -> tuple[str, ...]:
    if not regimes:
        return ()
    cleaned: list[str] = []
    seen: set[str] = set()
    for regime in regimes:
        text = str(regime).strip().lower()
        if not text or text in seen:
            continue
        seen.add(text)
        cleaned.append(text)
    return tuple(cleaned)


def _normalise_bounds(
    value: Mapping[str, Any] | Sequence[float | int | None] | tuple[float | int | None, ...],
    *,
    field_name: str,
) -> tuple[float | None, float | None]:
    minimum: float | None = None
    maximum: float | None = None
    if isinstance(value, Mapping):
        raw_min = value.get("minimum") if "minimum" in value else value.get("min")
        raw_max = value.get("maximum") if "maximum" in value else value.get("max")
    elif isinstance(value, Sequence):
        raw_min = value[0] if len(value) > 0 else None
        raw_max = value[1] if len(value) > 1 else None
    else:
        raise TypeError(f"{field_name} bounds must be a mapping or sequence")

    if raw_min is not None:
        try:
            minimum = float(raw_min)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"{field_name} minimum must be numeric") from exc
    if raw_max is not None:
        try:
            maximum = float(raw_max)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"{field_name} maximum must be numeric") from exc
    if minimum is not None and maximum is not None and minimum > maximum:
        raise ValueError(f"{field_name} minimum {minimum} exceeds maximum {maximum}")
    return minimum, maximum


@dataclass(frozen=True)
class OperatorContext:
    """Execution context supplied to operator constraints."""

    operation: str
    stage: PolicyLedgerStage
    regime: str | None = None
    regime_confidence: float | None = None
    regime_features: Mapping[str, float] = field(default_factory=dict)
    parameters: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def numeric_parameter(self, name: str) -> float | None:
        """Return the parameter as a float when available."""

        value = self.parameters.get(name)
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None


@dataclass(frozen=True)
class OperatorConstraintViolation:
    """Result describing why an operator was rejected."""

    operation: str
    reason: str
    details: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Mapping[str, Any]:
        payload: MutableMapping[str, Any] = {
            "operation": self.operation,
            "reason": self.reason,
        }
        if self.details:
            payload["details"] = dict(self.details)
        return payload


@dataclass(frozen=True)
class OperatorConstraint:
    """Constraint rule applied to one or more operator names."""

    operations: Sequence[str]
    name: str | None = None
    allowed_stages: Sequence[PolicyLedgerStage | str] | None = None
    blocked_stages: Sequence[PolicyLedgerStage | str] | None = None
    allowed_regimes: Sequence[str] | None = None
    blocked_regimes: Sequence[str] | None = None
    min_confidence: float | None = None
    max_confidence: float | None = None
    feature_gates: Mapping[str, Mapping[str, Any] | Sequence[float | int | None]] | None = None
    parameter_bounds: Mapping[str, Mapping[str, Any] | Sequence[float | int | None]] | None = None
    reason: str | None = None

    def __post_init__(self) -> None:
        operations = _normalise_operations(self.operations)
        object.__setattr__(self, "operations", operations)
        object.__setattr__(self, "allowed_stages", _normalise_stage_set(self.allowed_stages))
        object.__setattr__(self, "blocked_stages", _normalise_stage_set(self.blocked_stages))
        object.__setattr__(self, "allowed_regimes", _normalise_regime_set(self.allowed_regimes))
        object.__setattr__(self, "blocked_regimes", _normalise_regime_set(self.blocked_regimes))

        min_conf = self.min_confidence
        max_conf = self.max_confidence
        if min_conf is not None:
            object.__setattr__(self, "min_confidence", float(min_conf))
        if max_conf is not None:
            object.__setattr__(self, "max_confidence", float(max_conf))
        if (
            self.min_confidence is not None
            and self.max_confidence is not None
            and self.min_confidence > self.max_confidence
        ):
            raise ValueError("min_confidence cannot exceed max_confidence")

        feature_gates = {}
        if self.feature_gates:
            for feature, gate in self.feature_gates.items():
                feature_name = str(feature).strip()
                if not feature_name:
                    raise ValueError("feature_gates contain blank feature name")
                feature_gates[feature_name] = _normalise_bounds(
                    gate,
                    field_name=f"feature gate {feature_name}",
                )
        object.__setattr__(self, "feature_gates", feature_gates)

        parameter_bounds = {}
        if self.parameter_bounds:
            for parameter, bounds in self.parameter_bounds.items():
                parameter_name = str(parameter).strip()
                if not parameter_name:
                    raise ValueError("parameter_bounds contain blank parameter name")
                parameter_bounds[parameter_name] = _normalise_bounds(
                    bounds,
                    field_name=f"parameter bound {parameter_name}",
                )
        object.__setattr__(self, "parameter_bounds", parameter_bounds)

    def applies(self, operation: str) -> bool:
        return operation in self.operations

    def evaluate(self, context: OperatorContext) -> OperatorConstraintViolation | None:
        if not self.applies(context.operation):
            return None

        reasons: list[str] = []
        details: MutableMapping[str, Any] = {}

        allowed_stages = self.allowed_stages
        if allowed_stages and context.stage not in allowed_stages:
            reasons.append("stage_not_allowed")
            details["stage"] = context.stage.value
            details["allowed_stages"] = tuple(stage.value for stage in allowed_stages)

        blocked_stages = self.blocked_stages
        if blocked_stages and context.stage in blocked_stages:
            reasons.append("stage_blocked")
            details["stage"] = context.stage.value
            details["blocked_stages"] = tuple(stage.value for stage in blocked_stages)

        regime = context.regime.lower().strip() if context.regime else None
        allowed_regimes = self.allowed_regimes
        if allowed_regimes and (regime is None or regime not in allowed_regimes):
            reasons.append("regime_not_allowed")
            details["regime"] = regime
            details["allowed_regimes"] = allowed_regimes

        blocked_regimes = self.blocked_regimes
        if blocked_regimes and regime in blocked_regimes:
            reasons.append("regime_blocked")
            details["regime"] = regime
            details["blocked_regimes"] = blocked_regimes

        confidence = context.regime_confidence
        if self.min_confidence is not None:
            if confidence is None or confidence < self.min_confidence:
                reasons.append("confidence_below_minimum")
                details["confidence"] = confidence
                details["min_confidence"] = self.min_confidence
        if self.max_confidence is not None:
            if confidence is None or confidence > self.max_confidence:
                reasons.append("confidence_above_maximum")
                details["confidence"] = confidence
                details["max_confidence"] = self.max_confidence

        feature_gates = self.feature_gates
        if feature_gates:
            failed_features: dict[str, Mapping[str, float | None]] = {}
            regime_features = context.regime_features or {}
            for feature, (minimum, maximum) in feature_gates.items():
                value = regime_features.get(feature)
                feature_detail: MutableMapping[str, float | None] = {
                    "value": float(value) if value is not None else None,
                }
                if value is None:
                    reasons.append("feature_missing")
                    feature_detail["minimum"] = minimum
                    feature_detail["maximum"] = maximum
                    failed_features[feature] = feature_detail
                    continue
                try:
                    numeric_value = float(value)
                except (TypeError, ValueError):
                    reasons.append("feature_non_numeric")
                    feature_detail["minimum"] = minimum
                    feature_detail["maximum"] = maximum
                    failed_features[feature] = feature_detail
                    continue
                if minimum is not None and numeric_value < (minimum - 1e-12):
                    reasons.append("feature_below_minimum")
                    feature_detail["minimum"] = minimum
                    failed_features[feature] = feature_detail
                if maximum is not None and numeric_value > (maximum + 1e-12):
                    reasons.append("feature_above_maximum")
                    feature_detail["maximum"] = maximum
                    failed_features[feature] = feature_detail
            if failed_features:
                details.setdefault("features", {}).update(failed_features)

        parameter_bounds = self.parameter_bounds
        if parameter_bounds:
            failed_parameters: dict[str, Mapping[str, float | None]] = {}
            for parameter, (minimum, maximum) in parameter_bounds.items():
                value = context.parameters.get(parameter)
                parameter_detail: MutableMapping[str, float | None] = {
                    "value": float(value) if isinstance(value, (int, float)) else None,
                    "minimum": minimum,
                    "maximum": maximum,
                }
                numeric_value: float | None
                if value is None:
                    numeric_value = None
                else:
                    try:
                        numeric_value = float(value)
                        parameter_detail["value"] = numeric_value
                    except (TypeError, ValueError):
                        numeric_value = None
                if numeric_value is None:
                    reasons.append("parameter_missing_or_non_numeric")
                    failed_parameters[parameter] = parameter_detail
                    continue
                if minimum is not None and numeric_value < (minimum - 1e-12):
                    reasons.append("parameter_below_minimum")
                    failed_parameters[parameter] = parameter_detail
                if maximum is not None and numeric_value > (maximum + 1e-12):
                    reasons.append("parameter_above_maximum")
                    failed_parameters[parameter] = parameter_detail
            if failed_parameters:
                details.setdefault("parameters", {}).update(failed_parameters)

        if not reasons:
            return None

        if self.name:
            details.setdefault("constraint", self.name)
        return OperatorConstraintViolation(
            operation=context.operation,
            reason=self.reason or "operator_constraint_rejected",
            details=dict(details),
        )


@dataclass(frozen=True)
class OperatorConstraintSet:
    """Collection of operator constraints."""

    constraints: Sequence[OperatorConstraint]

    def __post_init__(self) -> None:
        object.__setattr__(self, "constraints", tuple(self.constraints))

    def validate(self, context: OperatorContext) -> tuple[bool, tuple[OperatorConstraintViolation, ...]]:
        violations: list[OperatorConstraintViolation] = []
        for constraint in self.constraints:
            violation = constraint.evaluate(context)
            if violation is not None:
                violations.append(violation)
        return not violations, tuple(violations)


def parse_operator_constraints(
    config: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None,
) -> OperatorConstraintSet | None:
    """Parse configuration payloads into an :class:`OperatorConstraintSet`.

    ``config`` accepts either a mapping keyed by constraint name or a sequence of
    individual constraint definitions.  Each definition may specify any keyword
    accepted by :class:`OperatorConstraint`.
    """

    if config is None:
        return None

    definitions: Iterable[Mapping[str, Any]]
    if isinstance(config, Mapping):
        definitions = (
            {**dict(payload), "name": name}
            for name, payload in config.items()
        )
    elif isinstance(config, Sequence):
        definitions = tuple(dict(item) for item in config)
    else:
        raise TypeError("Operator constraints config must be a mapping or sequence")

    constraints: list[OperatorConstraint] = []
    for definition in definitions:
        if not isinstance(definition, Mapping):
            raise TypeError("Operator constraint definition must be a mapping")
        constraints.append(OperatorConstraint(**definition))

    if not constraints:
        return None
    return OperatorConstraintSet(tuple(constraints))
