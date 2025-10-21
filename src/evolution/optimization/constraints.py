"""Constraint handling primitives for evolutionary strategy optimization."""
from __future__ import annotations

import math
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Callable, Deque, Iterable, Literal, Mapping, MutableMapping, Sequence

ComparisonOperator = Literal["<", "<=", ">", ">=", "==", "!="]


def _compare(op: ComparisonOperator, value: float, target: float) -> bool:
    if op == ">=":
        return value >= target
    if op == ">":
        return value > target
    if op == "<=":
        return value <= target
    if op == "<":
        return value < target
    if op == "==":
        return value == target
    if op == "!=":
        return value != target
    raise ValueError(f"Unsupported comparison operator '{op}'")


def _default_penalty(op: ComparisonOperator, value: float, limit: float) -> float:
    if op in ("<=", "<"):
        return max(0.0, value - limit)
    if op in (">=", ">"):
        return max(0.0, limit - value)
    if op == "==":
        return abs(value - limit)
    if op == "!=":
        return 1.0 if value == limit else 0.0
    raise ValueError(f"Unsupported comparison operator '{op}'")


@dataclass(frozen=True)
class ConstraintViolation:
    """Details about a constraint violation detected during evaluation."""

    constraint: str
    kind: Literal["hard", "soft"]
    message: str
    details: Mapping[str, object] = field(default_factory=dict)
    penalty: float = 0.0

    def with_penalty(self, penalty: float) -> "ConstraintViolation":
        return ConstraintViolation(
            constraint=self.constraint,
            kind=self.kind,
            message=self.message,
            details=self.details,
            penalty=penalty,
        )


@dataclass(frozen=True)
class ConstraintEvaluation:
    """Summary of an evaluation pass against the configured constraints."""

    satisfied: bool
    hard_violations: tuple[ConstraintViolation, ...]
    soft_violations: tuple[ConstraintViolation, ...]
    penalty: float

    @property
    def violations(self) -> tuple[ConstraintViolation, ...]:
        return self.hard_violations + self.soft_violations


@dataclass(frozen=True)
class HardConstraint:
    """Hard constraint that must be satisfied for a candidate to be valid."""

    name: str
    metric: str
    operator: ComparisonOperator
    limit: float
    message: str | None = None

    def evaluate(self, metrics: Mapping[str, float]) -> ConstraintViolation | None:
        value_obj = metrics.get(self.metric)
        if value_obj is None or (isinstance(value_obj, float) and math.isnan(value_obj)):
            message = self.message or f"Metric '{self.metric}' missing for hard constraint '{self.name}'"
            return ConstraintViolation(
                constraint=self.name,
                kind="hard",
                message=message,
                details={
                    "metric": self.metric,
                    "operator": self.operator,
                    "limit": self.limit,
                    "value": None,
                },
            )
        value = float(value_obj)
        if _compare(self.operator, value, float(self.limit)):
            return None
        message = self.message or (
            f"Constraint '{self.name}' violated: {self.metric} {self.operator} {self.limit} (value={value})"
        )
        return ConstraintViolation(
            constraint=self.name,
            kind="hard",
            message=message,
            details={
                "metric": self.metric,
                "operator": self.operator,
                "limit": float(self.limit),
                "value": value,
                "delta": value - float(self.limit),
            },
        )


@dataclass(frozen=True)
class SoftConstraint:
    """Soft constraint that yields a penalty when violated but does not block."""

    name: str
    metric: str
    operator: ComparisonOperator
    limit: float
    weight: float = 1.0
    message: str | None = None
    penalty_fn: Callable[[ComparisonOperator, float, float], float] | None = None

    def evaluate(self, metrics: Mapping[str, float]) -> tuple[float, ConstraintViolation | None]:
        value_obj = metrics.get(self.metric)
        if value_obj is None or (isinstance(value_obj, float) and math.isnan(value_obj)):
            penalty = abs(float(self.weight))
            message = self.message or f"Metric '{self.metric}' missing for soft constraint '{self.name}'"
            violation = ConstraintViolation(
                constraint=self.name,
                kind="soft",
                message=message,
                details={
                    "metric": self.metric,
                    "operator": self.operator,
                    "limit": self.limit,
                    "value": None,
                },
                penalty=penalty,
            )
            return penalty, violation
        value = float(value_obj)
        if _compare(self.operator, value, float(self.limit)):
            return 0.0, None
        penalty_func = self.penalty_fn or _default_penalty
        base_penalty = float(penalty_func(self.operator, value, float(self.limit)))
        penalty = max(0.0, base_penalty) * abs(float(self.weight))
        message = self.message or (
            f"Soft constraint '{self.name}' violated: {self.metric} {self.operator} {self.limit} (value={value})"
        )
        violation = ConstraintViolation(
            constraint=self.name,
            kind="soft",
            message=message,
            details={
                "metric": self.metric,
                "operator": self.operator,
                "limit": float(self.limit),
                "value": value,
                "delta": value - float(self.limit),
            },
            penalty=penalty,
        )
        return penalty, violation


class ConstraintSet:
    """Collection of hard and soft constraints applied to candidate strategies."""

    def __init__(
        self,
        *,
        hard: Sequence[HardConstraint] | None = None,
        soft: Sequence[SoftConstraint] | None = None,
        tracking_window: int | None = 128,
    ) -> None:
        self._hard = tuple(hard or ())
        self._soft = tuple(soft or ())
        self._history: Deque[ConstraintViolation] = deque(maxlen=tracking_window or None)
        self._counts: MutableMapping[str, int] = Counter()

    @property
    def hard_constraints(self) -> tuple[HardConstraint, ...]:
        return self._hard

    @property
    def soft_constraints(self) -> tuple[SoftConstraint, ...]:
        return self._soft

    @property
    def violation_history(self) -> tuple[ConstraintViolation, ...]:
        return tuple(self._history)

    @property
    def violation_counts(self) -> Mapping[str, int]:
        return dict(self._counts)

    def add_hard_constraints(self, constraints: Iterable[HardConstraint]) -> None:
        self._hard = tuple(self._hard) + tuple(constraints)

    def add_soft_constraints(self, constraints: Iterable[SoftConstraint]) -> None:
        self._soft = tuple(self._soft) + tuple(constraints)

    def reset_tracking(self) -> None:
        self._history.clear()
        self._counts.clear()

    def evaluate(self, metrics: Mapping[str, float]) -> ConstraintEvaluation:
        hard_violations: list[ConstraintViolation] = []
        for constraint in self._hard:
            violation = constraint.evaluate(metrics)
            if violation is not None:
                hard_violations.append(violation)
                self._record_violation(violation)

        total_penalty = 0.0
        soft_violations: list[ConstraintViolation] = []
        for constraint in self._soft:
            penalty, violation = constraint.evaluate(metrics)
            total_penalty += penalty
            if violation is not None:
                soft_violations.append(violation)
                self._record_violation(violation)

        satisfied = not hard_violations
        return ConstraintEvaluation(
            satisfied=satisfied,
            hard_violations=tuple(hard_violations),
            soft_violations=tuple(soft_violations),
            penalty=total_penalty,
        )

    def _record_violation(self, violation: ConstraintViolation) -> None:
        self._history.append(violation)
        self._counts[violation.constraint] = self._counts.get(violation.constraint, 0) + 1


__all__ = [
    "ComparisonOperator",
    "ConstraintViolation",
    "ConstraintEvaluation",
    "HardConstraint",
    "SoftConstraint",
    "ConstraintSet",
]
