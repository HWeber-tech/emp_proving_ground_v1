"""Fast-weight constraint helpers enforcing sparse positive activations.

The AlphaTrade roadmap calls for fast-weight routing that prefers
excitatory (non-negative) adjustments and keeps the number of boosted
strategies sparse.  This module provides a small controller that takes
an incoming mapping of tactic multipliers, clamps them to non-negative
values, and prunes low-activation multipliers so that only the strongest
signals remain active at any decision step.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Mapping, MutableMapping


@dataclass(frozen=True)
class FastWeightConstraints:
    """Configuration for constraining fast-weight multipliers."""

    baseline: float = 1.0
    minimum_multiplier: float = 0.0
    activation_threshold: float = 1.05
    max_active_fraction: float = 0.4
    prune_tolerance: float = 1e-6

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


@dataclass(frozen=True)
class FastWeightMetrics:
    """Summary statistics for constrained multipliers."""

    total: int
    active: int
    dormant: int
    active_percentage: float
    sparsity: float
    active_ids: tuple[str, ...]
    dormant_ids: tuple[str, ...]
    max_multiplier: float | None
    min_multiplier: float | None

    def as_dict(self) -> Mapping[str, object]:
        return {
            "total": self.total,
            "active": self.active,
            "dormant": self.dormant,
            "active_percentage": self.active_percentage,
            "sparsity": self.sparsity,
            "active_ids": self.active_ids,
            "dormant_ids": self.dormant_ids,
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

        final_values: MutableMapping[str, float] = {tactic_id: baseline for tactic_id in tactic_order}
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
                final_values[tactic_id] = value
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
        max_multiplier: float | None = None
        min_multiplier: float | None = None

        for tactic_id in tactic_order:
            value = final_values[tactic_id]
            if value < 0.0:
                value = 0.0
            if not math.isfinite(value):
                value = baseline
            final_values[tactic_id] = value

            if value > activation_threshold + tolerance:
                active_count += 1
                active_ids.append(tactic_id)
            else:
                dormant_ids.append(tactic_id)

            max_multiplier = value if max_multiplier is None else max(max_multiplier, value)
            min_multiplier = value if min_multiplier is None else min(min_multiplier, value)

        dormant = max(0, total - active_count)
        active_percentage = (active_count / total * 100.0) if total else 0.0
        sparsity = 1.0 - (active_count / total) if total else 1.0

        constrained = {
            tactic_id: value
            for tactic_id, value in final_values.items()
            if not math.isclose(value, baseline, rel_tol=1e-12, abs_tol=tolerance)
        }

        metrics = FastWeightMetrics(
            total=total,
            active=active_count,
            dormant=dormant,
            active_percentage=active_percentage,
            sparsity=sparsity,
            active_ids=tuple(active_ids),
            dormant_ids=tuple(dormant_ids),
            max_multiplier=float(max_multiplier) if max_multiplier is not None else None,
            min_multiplier=float(min_multiplier) if min_multiplier is not None else None,
        )
        return FastWeightResult(weights=constrained, metrics=metrics)


__all__ = [
    "FastWeightConstraints",
    "FastWeightController",
    "FastWeightMetrics",
    "FastWeightResult",
]
