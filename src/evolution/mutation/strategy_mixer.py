"""Strategy mixing operator with switching friction penalties.

Implements roadmap backlog item "op_mix_strategies" by blending multiple
strategy candidates into an ensemble while applying stability penalties that
discourage abrupt weight changes (switching frictions).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Iterable, Mapping, MutableMapping, Sequence


def _normalise_weights(weights: Mapping[str, float]) -> dict[str, float]:
    """Return a normalised copy of ``weights`` with only positive entries."""

    cleaned: dict[str, float] = {}
    for key, value in weights.items():
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if numeric <= 0.0:
            continue
        cleaned[str(key)] = numeric

    total = sum(cleaned.values())
    if total <= 0.0:
        return {}
    scale = 1.0 / total
    return {key: value * scale for key, value in cleaned.items()}


def _compute_effective_friction(
    *,
    friction: float,
    previous_timestamp: datetime | None,
    current_timestamp: datetime,
    decay_half_life: float | None,
) -> float:
    friction = max(0.0, min(float(friction), 0.999))
    if not previous_timestamp or decay_half_life is None or decay_half_life <= 0.0:
        return friction

    previous = previous_timestamp.astimezone(UTC)
    current = current_timestamp.astimezone(UTC)
    delta_seconds = max(0.0, (current - previous).total_seconds())
    if delta_seconds <= 0.0:
        return friction

    decay_factor = pow(0.5, delta_seconds / decay_half_life)
    return friction * decay_factor


@dataclass(slots=True, frozen=True)
class StrategyMixCandidate:
    """Candidate tactic considered during ensemble mixing."""

    tactic_id: str
    score: float
    min_weight: float | None = None
    max_weight: float | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class StrategyMixState:
    """Stored ensemble allocation used for applying switching friction."""

    weights: Mapping[str, float]
    timestamp: datetime | None = None

    def normalised_weights(self) -> dict[str, float]:
        return _normalise_weights(self.weights)


@dataclass(slots=True, frozen=True)
class StrategyMixResult:
    """Outcome of a strategy mixing operation."""

    weights: Mapping[str, float]
    selected: tuple[str, ...]
    dropped: tuple[str, ...]
    penalty: float
    target_mix: Mapping[str, float]
    previous_weights: Mapping[str, float]
    effective_friction: float
    timestamp: datetime
    metadata: Mapping[str, object] = field(default_factory=dict)


def _dedupe_candidates(candidates: Sequence[StrategyMixCandidate]) -> list[StrategyMixCandidate]:
    """Return candidates deduplicated by tactic_id (keep highest score)."""

    best_by_id: dict[str, StrategyMixCandidate] = {}
    for candidate in candidates:
        tactic_id = candidate.tactic_id.strip()
        if not tactic_id:
            continue
        existing = best_by_id.get(tactic_id)
        if existing is None or float(candidate.score) > float(existing.score):
            best_by_id[tactic_id] = candidate
    return sorted(
        best_by_id.values(),
        key=lambda item: (float(item.score), item.tactic_id),
        reverse=True,
    )


def _build_bounds(
    candidates: Iterable[StrategyMixCandidate],
    *,
    global_min: float,
    global_max: float,
) -> dict[str, tuple[float, float]]:
    bounds: dict[str, tuple[float, float]] = {}
    for candidate in candidates:
        minimum = candidate.min_weight if candidate.min_weight is not None else global_min
        maximum = candidate.max_weight if candidate.max_weight is not None else global_max
        minimum = max(0.0, min(float(minimum), 1.0))
        maximum = max(minimum, min(float(maximum), 1.0))
        bounds[candidate.tactic_id] = (minimum, maximum)
    return bounds


def _bounded_mix(
    raw_scores: Mapping[str, float],
    bounds: Mapping[str, tuple[float, float]],
) -> dict[str, float]:
    if not raw_scores:
        return {}

    non_negative = {key: max(0.0, float(value)) for key, value in raw_scores.items()}
    total = sum(non_negative.values())
    if total <= 0.0:
        equal_share = 1.0 / len(non_negative)
        base = {key: equal_share for key in non_negative}
    else:
        base = {key: value / total for key, value in non_negative.items()}

    minima = {key: bounds.get(key, (0.0, 1.0))[0] for key in base}
    maxima = {key: bounds.get(key, (0.0, 1.0))[1] for key in base}

    min_total = sum(minima.values())
    if min_total >= 1.0:
        equal_share = 1.0 / len(base)
        return {key: equal_share for key in base}

    remainder = 1.0 - min_total
    weights = {key: minima[key] + base[key] * remainder for key in base}

    while True:
        overflow = 0.0
        free: list[str] = []
        for key in weights:
            maximum = maxima.get(key, 1.0)
            if weights[key] > maximum:
                overflow += weights[key] - maximum
                weights[key] = maximum
            else:
                free.append(key)
        if overflow <= 1e-12 or not free:
            break
        free_total = sum(base[key] for key in free)
        if free_total <= 0.0:
            share = overflow / len(free)
            for key in free:
                weights[key] += share
            continue
        for key in free:
            weights[key] += overflow * (base[key] / free_total)

    normalised = _normalise_weights(weights)
    if normalised:
        return normalised
    return {key: 1.0 / len(base) for key in base}


def op_mix_strategies(
    candidates: Sequence[StrategyMixCandidate],
    *,
    previous_state: StrategyMixState | None = None,
    friction: float = 0.35,
    max_components: int = 3,
    min_share: float = 0.05,
    max_share: float = 0.6,
    timestamp: datetime | None = None,
    decay_half_life: float | None = None,
    drop_threshold: float = 0.01,
) -> StrategyMixResult:
    """Blend strategy candidates into an ensemble with switching friction.

    Args:
        candidates: Candidate tactics ranked by score.
        previous_state: Prior ensemble allocation to penalise abrupt changes.
        friction: Base friction coefficient in ``[0, 1)`` controlling how quickly
            the mix can change. ``0`` allows immediate shifts while higher values
            retain prior weights longer.
        max_components: Maximum number of tactics to keep in the ensemble.
        min_share: Global minimum share per selected tactic before normalisation.
        max_share: Global maximum share per selected tactic.
        timestamp: Timestamp associated with the mix (defaults to ``datetime.now``).
        decay_half_life: Optional half-life (seconds) that decays friction over
            time gaps between mixes.
        drop_threshold: Minimum weight required to keep a tactic post-friction.

    Returns:
        StrategyMixResult describing the new ensemble allocation.
    """

    if max_components <= 0:
        raise ValueError("max_components must be positive")

    timestamp = (timestamp or datetime.now(tz=UTC)).astimezone(UTC)
    deduped = _dedupe_candidates(candidates)

    global_min = max(0.0, min(float(min_share), 1.0))
    global_max = max(global_min, min(float(max_share), 1.0))

    selected_candidates = deduped[:max_components]
    bounds = _build_bounds(selected_candidates, global_min=global_min, global_max=global_max)

    raw_scores = {candidate.tactic_id: float(candidate.score) for candidate in selected_candidates}
    target_mix = _bounded_mix(raw_scores, bounds)

    previous_weights = previous_state.normalised_weights() if previous_state else {}
    effective_friction = _compute_effective_friction(
        friction=friction,
        previous_timestamp=previous_state.timestamp if previous_state else None,
        current_timestamp=timestamp,
        decay_half_life=decay_half_life,
    )
    adaptation_rate = 1.0 - effective_friction

    all_ids = set(target_mix) | set(previous_weights)
    blended: dict[str, float] = {}
    for tactic_id in all_ids:
        target = target_mix.get(tactic_id, 0.0)
        previous = previous_weights.get(tactic_id, 0.0)
        updated = previous + adaptation_rate * (target - previous)
        if updated <= 0.0:
            continue
        blended[tactic_id] = updated

    if not blended:
        blended = dict(target_mix)

    filtered = {key: value for key, value in blended.items() if value >= max(drop_threshold, 0.0)}
    if not filtered:
        filtered = dict(target_mix) if target_mix else dict(previous_weights)

    sorted_items = sorted(filtered.items(), key=lambda item: item[1], reverse=True)
    top_items = sorted_items[:max_components]
    final_weights = _normalise_weights(dict(top_items))

    if not final_weights and target_mix:
        final_weights = dict(target_mix)

    selected = tuple(final_weights.keys())
    dropped = tuple(sorted(set(previous_weights) - set(selected)))

    penalty = 0.0
    union_ids = set(target_mix) | set(final_weights)
    for tactic_id in union_ids:
        penalty += abs(final_weights.get(tactic_id, 0.0) - target_mix.get(tactic_id, 0.0))

    metadata: dict[str, object] = {
        "switch_cost": sum(
            abs(final_weights.get(tactic_id, 0.0) - previous_weights.get(tactic_id, 0.0))
            for tactic_id in set(previous_weights) | set(final_weights)
        ),
        "adaptation_rate": adaptation_rate,
        "global_bounds": {
            "min_share": global_min,
            "max_share": global_max,
        },
        "candidate_count": len(deduped),
    }

    result = StrategyMixResult(
        weights=final_weights,
        selected=selected,
        dropped=dropped,
        penalty=penalty,
        target_mix=target_mix,
        previous_weights=previous_weights,
        effective_friction=effective_friction,
        timestamp=timestamp,
        metadata=metadata,
    )
    return result


__all__ = [
    "StrategyMixCandidate",
    "StrategyMixResult",
    "StrategyMixState",
    "op_mix_strategies",
]
