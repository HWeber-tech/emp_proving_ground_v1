"""Preference articulation utilities for evolutionary optimization."""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Callable, Iterable, Iterator, Mapping, MutableMapping

Number = float | int


@dataclass(frozen=True)
class ObjectivePreference:
    """Configuration for a single optimization objective preference."""

    objective: str
    weight: float
    minimum: float = 0.0
    maximum: float = math.inf
    description: str | None = None

    def clamp(self, value: Number) -> float:
        """Clamp *value* into the supported range for the preference."""

        lo = float(self.minimum)
        hi = float(self.maximum)
        if lo > hi:
            lo, hi = hi, lo
        return min(hi, max(lo, float(value)))

    def with_weight(self, weight: Number) -> "ObjectivePreference":
        """Return a copy with *weight* applied and clamped."""

        clamped = self.clamp(weight)
        return replace(self, weight=clamped)


class PreferenceProfile:
    """Mutable view over objective preferences and derived weights."""

    def __init__(
        self,
        preferences: Iterable[ObjectivePreference] | Mapping[str, ObjectivePreference] | None = None,
        *,
        metadata: Mapping[str, object] | None = None,
    ) -> None:
        if preferences is None:
            items: Iterable[ObjectivePreference] = ()
        elif isinstance(preferences, Mapping):
            items = preferences.values()
        else:
            items = preferences
        self._preferences: MutableMapping[str, ObjectivePreference] = {
            pref.objective: pref for pref in items
        }
        self._metadata = dict(metadata or {})

    @property
    def metadata(self) -> Mapping[str, object]:
        return self._metadata

    @property
    def objectives(self) -> tuple[str, ...]:
        return tuple(self._preferences)

    def get(self, objective: str) -> ObjectivePreference | None:
        return self._preferences.get(objective)

    def with_preference(
        self,
        objective: str,
        *,
        weight: Number | None = None,
        minimum: Number | None = None,
        maximum: Number | None = None,
        description: str | None = None,
    ) -> "PreferenceProfile":
        current = self._preferences.get(objective)
        if current is None:
            weight = float(weight if weight is not None else 1.0)
            minimum_f = float(minimum if minimum is not None else 0.0)
            maximum_f = float(maximum if maximum is not None else math.inf)
            pref = ObjectivePreference(
                objective=objective,
                weight=weight,
                minimum=minimum_f,
                maximum=maximum_f,
                description=description,
            )
        else:
            pref = current
            if minimum is not None or maximum is not None:
                minimum_f = float(minimum if minimum is not None else pref.minimum)
                maximum_f = float(maximum if maximum is not None else pref.maximum)
                pref = replace(pref, minimum=minimum_f, maximum=maximum_f)
            if description is not None:
                pref = replace(pref, description=description)
            if weight is not None:
                pref = pref.with_weight(weight)
        updated = PreferenceProfile(self._preferences.values(), metadata=self._metadata)
        updated._preferences[objective] = pref
        return updated

    def normalized_weights(self) -> Mapping[str, float]:
        weights = {name: max(0.0, pref.weight) for name, pref in self._preferences.items()}
        total = sum(weights.values())
        if total <= 0.0 and weights:
            uniform = 1.0 / float(len(weights))
            return {name: uniform for name in weights}
        if total <= 0.0:
            return {}
        return {name: value / total for name, value in weights.items()}

    def score(self, metrics: Mapping[str, Number], *, missing_value: float = 0.0) -> float:
        weights = self.normalized_weights()
        if not weights:
            return 0.0
        score = 0.0
        for name, weight in weights.items():
            raw = metrics.get(name, missing_value)
            if raw is None or (isinstance(raw, float) and math.isnan(raw)):
                raw = missing_value
            score += weight * float(raw)
        return score

    def ranked_preferences(self) -> tuple[ObjectivePreference, ...]:
        return tuple(sorted(self._preferences.values(), key=lambda pref: pref.weight, reverse=True))

    def to_dict(self) -> Mapping[str, float]:
        return {name: pref.weight for name, pref in self._preferences.items()}

    def __iter__(self) -> Iterator[ObjectivePreference]:
        return iter(self._preferences.values())


def _default_aggregator(metrics: Mapping[str, Number], weights: Mapping[str, float]) -> float:
    score = 0.0
    for objective, weight in weights.items():
        value = metrics.get(objective)
        if value is None or (isinstance(value, float) and math.isnan(value)):
            continue
        score += weight * float(value)
    return score


class PreferenceArticulator:
    """High-level helper for aligning evolution with articulated preferences."""

    def __init__(
        self,
        profile: PreferenceProfile | None = None,
        *,
        aggregator: Callable[[Mapping[str, Number], Mapping[str, float]], float] | None = None,
        input_fn: Callable[[str], str] | None = None,
        output_fn: Callable[[str], None] | None = None,
    ) -> None:
        self._profile = profile or PreferenceProfile()
        self._aggregator = aggregator or _default_aggregator
        self._input = input_fn or input
        self._output = output_fn or print

    @property
    def profile(self) -> PreferenceProfile:
        return self._profile

    def articulate(self, metrics: Mapping[str, Number]) -> float:
        weights = self._profile.normalized_weights()
        return self._aggregator(metrics, weights)

    def update(self, objective: str, weight: Number) -> None:
        self._profile = self._profile.with_preference(objective, weight=float(weight))

    def tune(self) -> PreferenceProfile:
        self._profile = interactive_preference_tuning(
            self._profile, input_fn=self._input, output_fn=self._output
        )
        return self._profile


def interactive_preference_tuning(
    profile: PreferenceProfile,
    *,
    input_fn: Callable[[str], str] | None = None,
    output_fn: Callable[[str], None] | None = None,
) -> PreferenceProfile:
    """Interactively tune preferences using *input_fn*/*output_fn* callbacks."""

    reader = input_fn or input
    writer = output_fn or print
    if not profile.objectives:
        writer("No objectives configured; skipping preference tuning.")
        return profile

    writer("Preference tuning: press <enter> to keep the current value.")
    updated = PreferenceProfile(profile, metadata=profile.metadata)
    for pref in profile:
        prompt = f"{pref.objective} weight [{pref.weight:.3f}] -> "
        while True:
            response = reader(prompt)
            if response.strip() == "":
                break
            try:
                value = float(response.strip())
            except ValueError:
                writer("Invalid weight, please enter a numeric value.")
                continue
            updated = updated.with_preference(pref.objective, weight=value)
            break
    writer("Updated preferences: " + ", ".join(
        f"{pref.objective}={pref.weight:.3f}" for pref in updated.ranked_preferences()
    ))
    return updated


__all__ = [
    "ObjectivePreference",
    "PreferenceArticulator",
    "PreferenceProfile",
    "interactive_preference_tuning",
]
