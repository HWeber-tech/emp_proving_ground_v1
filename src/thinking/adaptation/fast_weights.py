"""Fast-weight constraint helpers enforcing sparse positive activations.

The AlphaTrade roadmap calls for fast-weight routing that prefers
excitatory (non-negative) adjustments and keeps the number of boosted
strategies sparse.  This module provides a small controller that takes
an incoming mapping of tactic multipliers, clamps them to non-negative
values, and prunes low-activation multipliers so that only the strongest
signals remain active at any decision step.  When inhibitory (below
baseline) multipliers are supplied, the controller can either surface
them explicitly or suppress them entirely depending on the configured
constraints so governance can reason about the adaptation posture.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Iterable, Mapping, MutableMapping, Any


@dataclass(frozen=True)
class FastWeightConstraints:
    """Configuration for constraining fast-weight multipliers."""

    baseline: float = 1.0
    minimum_multiplier: float = 0.0
    activation_threshold: float = 1.05
    max_active_fraction: float = 0.4
    prune_tolerance: float = 1e-6
    excitatory_only: bool = False

    def __post_init__(self) -> None:  # pragma: no cover - dataclass validation
        if self.baseline < 0.0:
            raise ValueError("baseline must be non-negative")
        if self.minimum_multiplier < 0.0:
            raise ValueError("minimum_multiplier must be non-negative")
        if self.baseline < self.minimum_multiplier:
            raise ValueError("baseline must be >= minimum_multiplier")
        if self.max_active_fraction < 0.0 or not math.isfinite(self.max_active_fraction):
            raise ValueError("max_active_fraction must be finite and non-negative")
        if self.activation_threshold < self.minimum_multiplier:
            raise ValueError("activation_threshold must be >= minimum_multiplier")
        if self.prune_tolerance < 0.0:
            raise ValueError("prune_tolerance must be non-negative")
        if not isinstance(self.excitatory_only, bool):
            raise TypeError("excitatory_only must be a boolean flag")


@dataclass(frozen=True)
class FastWeightMetrics:
    """Summary statistics for constrained multipliers."""

    total: int
    active: int
    dormant: int
    inhibitory: int
    suppressed_inhibitory: int
    active_percentage: float
    sparsity: float
    active_ids: tuple[str, ...]
    dormant_ids: tuple[str, ...]
    inhibitory_ids: tuple[str, ...]
    suppressed_inhibitory_ids: tuple[str, ...]
    max_multiplier: float | None
    min_multiplier: float | None

    def as_dict(self) -> Mapping[str, object]:
        return {
            "total": self.total,
            "active": self.active,
            "dormant": self.dormant,
            "inhibitory": self.inhibitory,
            "suppressed_inhibitory": self.suppressed_inhibitory,
            "active_percentage": self.active_percentage,
            "sparsity": self.sparsity,
            "active_ids": self.active_ids,
            "dormant_ids": self.dormant_ids,
            "inhibitory_ids": self.inhibitory_ids,
            "suppressed_inhibitory_ids": self.suppressed_inhibitory_ids,
            "max_multiplier": self.max_multiplier,
            "min_multiplier": self.min_multiplier,
        }


@dataclass(frozen=True)
class FastWeightResult:
    """Constrained multipliers and associated metrics."""

    weights: Mapping[str, float]
    metrics: FastWeightMetrics


class FastWeightController:
    """Apply sparsity constraints to fast-weight multipliers."""

    def __init__(self, constraints: FastWeightConstraints | None = None) -> None:
        self._constraints = constraints or FastWeightConstraints()

    @property
    def constraints(self) -> FastWeightConstraints:
        return self._constraints

    def constrain(
        self,
        *,
        fast_weights: Mapping[str, float] | None,
        tactic_ids: Iterable[str],
    ) -> FastWeightResult:
        """Return constrained multipliers and sparsity metrics.

        ``fast_weights`` is treated as multipliers relative to the baseline
        (1.0 by default).  Missing tactics are assumed to be at baseline.
        Values are clamped to ``minimum_multiplier`` and any non-finite
        values are ignored.  The strongest activations above
        ``activation_threshold`` are kept while the rest are pruned back to
        baseline so that only ``max_active_fraction`` of strategies remain
        boosted.
        """

        constraints = self._constraints
        tactic_order = tuple(dict.fromkeys(tactic_ids))
        total = len(tactic_order)
        baseline = constraints.baseline
        minimum = constraints.minimum_multiplier
        tolerance = constraints.prune_tolerance
        activation_threshold = constraints.activation_threshold
        excitatory_only = constraints.excitatory_only

        final_values: MutableMapping[str, float] = {tactic_id: baseline for tactic_id in tactic_order}
        raw_values: MutableMapping[str, float] = {tactic_id: baseline for tactic_id in tactic_order}
        activations: list[tuple[str, float]] = []

        if fast_weights:
            for tactic_id in tactic_order:
                if tactic_id not in fast_weights:
                    continue
                try:
                    value = float(fast_weights[tactic_id])
                except (TypeError, ValueError):
                    continue
                if not math.isfinite(value):
                    continue
                if value < minimum:
                    value = minimum
                if value < 0.0:
                    value = 0.0
                if excitatory_only and value < baseline:
                    value = baseline
                final_values[tactic_id] = value
                raw_values[tactic_id] = value
                if value > activation_threshold:
                    activations.append((tactic_id, value))

        max_active_fraction = constraints.max_active_fraction
        max_active = total
        if total and max_active_fraction < 1.0:
            # Keep at most ceil(total * fraction) activations.
            proposed = math.ceil(total * max_active_fraction)
            max_active = max(0, min(total, proposed))

        if max_active >= 0 and len(activations) > max_active:
            activations.sort(key=lambda item: item[1], reverse=True)
            keep_ids = {tactic_id for tactic_id, _ in activations[:max_active]}
            for tactic_id, _ in activations[max_active:]:
                final_values[tactic_id] = baseline

        active_count = 0
        active_ids: list[str] = []
        dormant_ids: list[str] = []
        inhibitory_ids: list[str] = []
        suppressed_inhibitory_ids: list[str] = []
        max_multiplier: float | None = None
        min_multiplier: float | None = None

        for tactic_id in tactic_order:
            value = final_values[tactic_id]
            raw_value = raw_values[tactic_id]
            if not constraints.allow_inhibitory and raw_value < baseline - tolerance:
                suppressed_inhibitory_ids.append(tactic_id)
            if not constraints.allow_inhibitory and value < baseline:
                value = baseline
            if value < 0.0:
                value = 0.0
            if not math.isfinite(value):
                value = baseline
            if excitatory_only and value < baseline:
                value = baseline
            final_values[tactic_id] = value

            if value > activation_threshold + tolerance:
                active_count += 1
                active_ids.append(tactic_id)
            else:
                dormant_ids.append(tactic_id)

            if value < baseline - tolerance:
                inhibitory_ids.append(tactic_id)

            max_multiplier = value if max_multiplier is None else max(max_multiplier, value)
            min_multiplier = value if min_multiplier is None else min(min_multiplier, value)

        dormant = max(0, total - active_count)
        active_percentage = (active_count / total * 100.0) if total else 0.0
        sparsity = 1.0 - (active_count / total) if total else 1.0
        inhibitory_count = len(inhibitory_ids)
        suppressed_inhibitory_count = len(suppressed_inhibitory_ids)

        constrained = {
            tactic_id: value
            for tactic_id, value in final_values.items()
            if not math.isclose(value, baseline, rel_tol=1e-12, abs_tol=tolerance)
        }

        metrics = FastWeightMetrics(
            total=total,
            active=active_count,
            dormant=dormant,
            inhibitory=inhibitory_count,
            suppressed_inhibitory=suppressed_inhibitory_count,
            active_percentage=active_percentage,
            sparsity=sparsity,
            active_ids=tuple(active_ids),
            dormant_ids=tuple(dormant_ids),
            inhibitory_ids=tuple(inhibitory_ids),
            suppressed_inhibitory_ids=tuple(suppressed_inhibitory_ids),
            max_multiplier=float(max_multiplier) if max_multiplier is not None else None,
            min_multiplier=float(min_multiplier) if min_multiplier is not None else None,
        )
        return FastWeightResult(weights=constrained, metrics=metrics)


def _normalise_keys(source: Mapping[str, Any]) -> Mapping[str, Any]:
    return {str(key).upper(): value for key, value in source.items()}


def _coerce_float(value: Any, *, key: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive guard
        raise ValueError(f"{key} must be a float-compatible value, got {value!r}") from exc


def _coerce_bool(value: Any, *, key: str) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        raise ValueError(f"{key} cannot be None")
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"{key} must be a boolean-compatible value, got {value!r}")


def parse_fast_weight_constraints(
    source: Mapping[str, Any] | None,
) -> FastWeightConstraints | None:
    """Return fast-weight constraints parsed from a mapping.

    The parser understands the ``FAST_WEIGHT_*`` keys documented for the
    understanding loop configuration.  Keys are matched case-insensitively and
    unspecified entries fall back to the dataclass defaults.
    """

    if not source:
        return None

    normalised = _normalise_keys(source)
    updates: dict[str, Any] = {}

    def _maybe_set(*keys: str, attr: str, coercer) -> None:
        for key in keys:
            if key in normalised:
                updates[attr] = coercer(normalised[key], key=key)
                return

    _maybe_set("FAST_WEIGHT_BASELINE", attr="baseline", coercer=_coerce_float)
    _maybe_set(
        "FAST_WEIGHT_MINIMUM_MULTIPLIER",
        "FAST_WEIGHT_MIN_MULTIPLIER",
        attr="minimum_multiplier",
        coercer=_coerce_float,
    )
    _maybe_set(
        "FAST_WEIGHT_ACTIVATION_THRESHOLD",
        attr="activation_threshold",
        coercer=_coerce_float,
    )
    _maybe_set(
        "FAST_WEIGHT_MAX_ACTIVE_FRACTION",
        attr="max_active_fraction",
        coercer=_coerce_float,
    )
    _maybe_set(
        "FAST_WEIGHT_PRUNE_TOLERANCE",
        attr="prune_tolerance",
        coercer=_coerce_float,
    )
    _maybe_set(
        "FAST_WEIGHT_EXCITATORY_ONLY",
        attr="excitatory_only",
        coercer=_coerce_bool,
    )

    if not updates:
        return None

    base = FastWeightConstraints()
    return replace(base, **updates)


def build_fast_weight_controller(source: Mapping[str, Any] | None) -> FastWeightController:
    """Helper that builds a controller from configuration overrides."""

    constraints = parse_fast_weight_constraints(source)
    if constraints is None:
        return FastWeightController()
    return FastWeightController(constraints)


__all__ = [
    "FastWeightConstraints",
    "FastWeightController",
    "FastWeightMetrics",
    "FastWeightResult",
    "build_fast_weight_controller",
    "parse_fast_weight_constraints",
]
