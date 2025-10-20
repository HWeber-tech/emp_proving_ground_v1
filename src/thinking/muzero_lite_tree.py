"""MuZero-lite short-horizon tree simulation with causal edge adjustments.

Implements roadmap item **F.1.3** style planning for short-horizon futures.
The helper exposes a small surface area so it can run inside tests or research
notebooks without heavy dependencies.  It accepts lightweight transition
payloads, applies causal edge adjustments, and reports expectations by horizon.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable, Mapping, Sequence

import math

__all__ = [
    "CausalEdgeAdjustment",
    "CausalContribution",
    "ShortHorizonTransition",
    "ShortHorizonFuture",
    "MuZeroLiteTreeSimulation",
    "simulate_short_horizon_futures",
]


def _coerce_float(value: float | int) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
        raise ValueError("value must be numeric") from exc
    if not math.isfinite(numeric):
        raise ValueError("value must be finite")
    return numeric


def _coerce_probability(value: float | int | None) -> float:
    if value is None:
        return 1.0
    numeric = _coerce_float(value)
    if numeric < 0.0:
        raise ValueError("probability must be non-negative")
    return numeric


@dataclass(frozen=True, slots=True)
class CausalEdgeAdjustment:
    """Describe how a causal trigger alters imagined edge."""

    tag: str
    delta_bps: float
    confidence: float = 1.0

    def __post_init__(self) -> None:
        if not self.tag or not self.tag.strip():
            raise ValueError("tag must be a non-empty string")
        if self.confidence < 0.0:
            raise ValueError("confidence must be non-negative")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> "CausalEdgeAdjustment":
        """Build an adjustment from a mapping payload."""

        if "tag" in payload:
            tag = str(payload["tag"])
        elif "cause" in payload:
            tag = str(payload["cause"])
        elif "trigger" in payload:
            tag = str(payload["trigger"])
        else:  # pragma: no cover - defensive
            raise ValueError("adjustment payload requires a 'tag'/'cause' field")

        if "delta_bps" in payload:
            delta = payload["delta_bps"]
        elif "magnitude" in payload:
            delta = payload["magnitude"]
        elif "edge_delta_bps" in payload:
            delta = payload["edge_delta_bps"]
        else:  # pragma: no cover - defensive
            raise ValueError("adjustment payload requires a delta magnitude")

        confidence = payload.get("confidence") or payload.get("weight") or payload.get("strength")

        return cls(
            tag=tag,
            delta_bps=_coerce_float(delta),
            confidence=_coerce_probability(confidence),
        )

    def contribution(self, occurrences: int) -> float:
        if occurrences <= 0:
            return 0.0
        return self.delta_bps * self.confidence * occurrences


@dataclass(frozen=True, slots=True)
class CausalContribution:
    """Applied adjustment for a simulated path."""

    tag: str
    delta_bps: float
    confidence: float
    occurrences: int

    def as_dict(self) -> dict[str, object]:
        return {
            "tag": self.tag,
            "delta_bps": self.delta_bps,
            "confidence": self.confidence,
            "occurrences": self.occurrences,
        }


@dataclass(frozen=True, slots=True)
class ShortHorizonTransition:
    """Transition specification for one layer of the MuZero-lite tree."""

    action: str
    probability: float
    delta_bps: float | None = None
    edge_bps: float | None = None
    causal_tag: str | None = None

    def __post_init__(self) -> None:
        if not self.action or not self.action.strip():
            raise ValueError("action must be a non-empty string")
        if self.delta_bps is None and self.edge_bps is None:
            raise ValueError("transition requires edge_bps or delta_bps")
        if self.probability < 0.0:
            raise ValueError("probability must be non-negative")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> "ShortHorizonTransition":
        action = payload.get("action") or payload.get("move") or payload.get("decision")
        if action is None:  # pragma: no cover - defensive
            raise ValueError("transition payload requires an 'action'/'move' field")

        probability = _coerce_probability(
            payload.get("probability")
            or payload.get("weight")
            or payload.get("likelihood")
        )

        if "edge_bps" in payload:
            edge = _coerce_float(payload["edge_bps"])
        elif "edge" in payload:
            edge = _coerce_float(payload["edge"])
        else:
            edge = None

        if "delta_bps" in payload:
            delta = _coerce_float(payload["delta_bps"])
        elif "edge_delta_bps" in payload:
            delta = _coerce_float(payload["edge_delta_bps"])
        elif "delta" in payload:
            delta = _coerce_float(payload["delta"])
        else:
            delta = None

        tag_value = (
            payload.get("causal_tag")
            or payload.get("cause")
            or payload.get("trigger")
            or payload.get("factor")
            or payload.get("tag")
        )
        causal_tag = str(tag_value) if tag_value is not None else None

        return cls(
            action=str(action),
            probability=probability,
            delta_bps=delta,
            edge_bps=edge,
            causal_tag=causal_tag,
        )

    def apply(self, parent_edge_bps: float, *, discount: float, depth: int) -> float:
        if self.edge_bps is not None:
            return self.edge_bps
        assert self.delta_bps is not None  # guarded in __post_init__
        factor = discount ** depth if discount != 1.0 else 1.0
        return parent_edge_bps + self.delta_bps * factor


@dataclass(frozen=True, slots=True)
class ShortHorizonFuture:
    """Imagined future leaf or intermediate horizon."""

    horizon: int
    actions: tuple[str, ...]
    base_edge_bps: float
    adjusted_edge_bps: float
    probability: float
    causal_contributions: tuple[CausalContribution, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "horizon": self.horizon,
            "actions": list(self.actions),
            "base_edge_bps": self.base_edge_bps,
            "adjusted_edge_bps": self.adjusted_edge_bps,
            "probability": self.probability,
            "causal_contributions": [c.as_dict() for c in self.causal_contributions],
        }


@dataclass(frozen=True, slots=True)
class MuZeroLiteTreeSimulation:
    """Aggregate view of simulated futures."""

    root_edge_bps: float
    futures: tuple[ShortHorizonFuture, ...]
    expected_base_edge_bps: float
    expected_adjusted_edge_bps: float
    base_by_horizon: Mapping[int, float]
    adjusted_by_horizon: Mapping[int, float]

    @property
    def best_future(self) -> ShortHorizonFuture:
        if not self.futures:  # pragma: no cover - defensive
            raise ValueError("simulation produced no futures")
        return max(self.futures, key=lambda future: future.adjusted_edge_bps)

    def expected_by_horizon(self, *, adjusted: bool = False) -> dict[int, float]:
        source = self.adjusted_by_horizon if adjusted else self.base_by_horizon
        return dict(source)

    def as_dict(self) -> dict[str, object]:
        return {
            "root_edge_bps": self.root_edge_bps,
            "expected_base_edge_bps": self.expected_base_edge_bps,
            "expected_adjusted_edge_bps": self.expected_adjusted_edge_bps,
            "base_by_horizon": dict(self.base_by_horizon),
            "adjusted_by_horizon": dict(self.adjusted_by_horizon),
            "futures": [future.as_dict() for future in self.futures],
        }


def _normalise_transitions(transitions: Sequence[ShortHorizonTransition]) -> list[ShortHorizonTransition]:
    total = sum(transition.probability for transition in transitions)
    if total <= 0.0:
        raise ValueError("transition probabilities must sum to a positive value")
    if math.isclose(total, 1.0, rel_tol=1e-9, abs_tol=1e-12):
        return list(transitions)
    return [
        replace(transition, probability=transition.probability / total)
        for transition in transitions
    ]


def _coerce_transition(spec: object) -> ShortHorizonTransition:
    if isinstance(spec, ShortHorizonTransition):
        return spec
    if isinstance(spec, Mapping):
        payload = dict(spec)
        payload.setdefault("action", payload.get("move") or payload.get("decision"))
        return ShortHorizonTransition.from_mapping(payload)
    if isinstance(spec, tuple) and spec:
        action = spec[0]
        if isinstance(action, str):
            if len(spec) == 2 and isinstance(spec[1], Mapping):
                payload = dict(spec[1])
                payload.setdefault("action", action)
                return ShortHorizonTransition.from_mapping(payload)
            if len(spec) == 2:
                return ShortHorizonTransition(
                    action=action,
                    probability=1.0,
                    delta_bps=_coerce_float(spec[1]),
                )
    raise TypeError(f"unsupported transition specification: {spec!r}")


def _coerce_adjustments(
    adjustments: Sequence[CausalEdgeAdjustment | Mapping[str, object]] | Mapping[str, float] | None,
) -> list[CausalEdgeAdjustment]:
    if adjustments is None:
        return []
    if isinstance(adjustments, Mapping):
        return [
            CausalEdgeAdjustment(tag=str(tag), delta_bps=_coerce_float(delta))
            for tag, delta in adjustments.items()
        ]

    result: list[CausalEdgeAdjustment] = []
    for item in adjustments:
        if isinstance(item, CausalEdgeAdjustment):
            result.append(item)
        elif isinstance(item, Mapping):
            result.append(CausalEdgeAdjustment.from_mapping(item))
        else:
            raise TypeError(f"unsupported adjustment specification: {item!r}")
    return result


def _apply_adjustments(
    edge_bps: float,
    tags: tuple[str, ...],
    adjustments: Sequence[CausalEdgeAdjustment],
) -> tuple[float, tuple[CausalContribution, ...]]:
    if not adjustments or not tags:
        return edge_bps, ()

    contributions: list[CausalContribution] = []
    adjusted = edge_bps

    for adjustment in adjustments:
        occurrences = tags.count(adjustment.tag)
        if occurrences:
            delta = adjustment.contribution(occurrences)
            adjusted += delta
            contributions.append(
                CausalContribution(
                    tag=adjustment.tag,
                    delta_bps=delta,
                    confidence=adjustment.confidence,
                    occurrences=occurrences,
                )
            )

    return adjusted, tuple(contributions)


def simulate_short_horizon_futures(
    root_edge_bps: float | int,
    layers: Sequence[Sequence[object] | Mapping[str, object]],
    *,
    causal_edge_adjustments: Sequence[CausalEdgeAdjustment | Mapping[str, object]]
    | Mapping[str, float]
    | None = None,
    discount: float = 0.9,
) -> MuZeroLiteTreeSimulation:
    """Simulate short-horizon futures using a MuZero-lite style tree."""

    root_edge = _coerce_float(root_edge_bps)
    if not layers:
        raise ValueError("layers must not be empty")
    if discount <= 0.0:
        raise ValueError("discount must be positive")

    adjustments = _coerce_adjustments(causal_edge_adjustments)

    parsed_layers: list[list[ShortHorizonTransition]] = []
    for index, layer in enumerate(layers):
        if isinstance(layer, Mapping):
            candidates = [
                _coerce_transition((action, spec))
                for action, spec in layer.items()
            ]
        else:
            candidates = [_coerce_transition(spec) for spec in layer]
        if not candidates:
            raise ValueError(f"layer {index} must contain at least one transition")
        parsed_layers.append(_normalise_transitions(candidates))

    futures: list[ShortHorizonFuture] = []

    def _explore(
        depth: int,
        current_edge: float,
        probability: float,
        actions: tuple[str, ...],
        tags: tuple[str, ...],
    ) -> None:
        if depth >= len(parsed_layers):
            return

        for transition in parsed_layers[depth]:
            next_probability = probability * transition.probability
            if next_probability <= 0.0:
                continue

            next_edge = transition.apply(current_edge, discount=discount, depth=depth)
            next_actions = actions + (transition.action,)
            next_tags = tags + ((transition.causal_tag,) if transition.causal_tag else ())

            adjusted_edge, contributions = _apply_adjustments(next_edge, next_tags, adjustments)
            futures.append(
                ShortHorizonFuture(
                    horizon=depth + 1,
                    actions=next_actions,
                    base_edge_bps=next_edge,
                    adjusted_edge_bps=adjusted_edge,
                    probability=next_probability,
                    causal_contributions=contributions,
                )
            )

            _explore(depth + 1, next_edge, next_probability, next_actions, next_tags)

    _explore(0, root_edge, 1.0, tuple(), tuple())

    if not futures:
        raise ValueError("no futures generated; check transition probabilities")

    base_totals: dict[int, float] = {}
    adjusted_totals: dict[int, float] = {}
    probability_totals: dict[int, float] = {}

    for future in futures:
        horizon = future.horizon
        base_totals[horizon] = base_totals.get(horizon, 0.0) + future.base_edge_bps * future.probability
        adjusted_totals[horizon] = adjusted_totals.get(horizon, 0.0) + future.adjusted_edge_bps * future.probability
        probability_totals[horizon] = probability_totals.get(horizon, 0.0) + future.probability

    base_by_horizon = {
        horizon: base_totals[horizon] / probability_totals[horizon]
        for horizon in base_totals
        if probability_totals[horizon] > 0.0
    }
    adjusted_by_horizon = {
        horizon: adjusted_totals[horizon] / probability_totals[horizon]
        for horizon in adjusted_totals
        if probability_totals[horizon] > 0.0
    }

    terminal_horizon = max(probability_totals)
    expected_base = base_by_horizon[terminal_horizon]
    expected_adjusted = adjusted_by_horizon[terminal_horizon]

    return MuZeroLiteTreeSimulation(
        root_edge_bps=root_edge,
        futures=tuple(futures),
        expected_base_edge_bps=expected_base,
        expected_adjusted_edge_bps=expected_adjusted,
        base_by_horizon=base_by_horizon,
        adjusted_by_horizon=adjusted_by_horizon,
    )

