"""Lightweight MuZero-inspired tree search for short-horizon planning.

This module fulfils the roadmap item **MuZero-Lite Tree** by providing a
compact tree roll-out helper that can project short-horizon futures while
incorporating causal edge adjustments.  The helper accepts either callables or
precomputed policy/transition/value mappings, making it easy to script small
scenario studies inside tests or the planner orchestrator.

The public API surfaces a :func:`simulate_short_horizon_futures` helper that
returns structured path metadata so downstream code can inspect best-action
sequences, expected returns, and per-step adjustments.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from collections.abc import Callable

import math

from src.operations.regulatory_telemetry import RegulatoryTelemetryStatus

__all__ = [
    "MuZeroLiteStep",
    "MuZeroLitePath",
    "MuZeroLiteTreeResult",
    "simulate_short_horizon_futures",
]


def _normalise_label(value: object | None) -> str:
    if value is None:
        return ""
    return str(value).strip().lower()


def _append_label(labels: list[str], seen: set[str], candidate: object | None) -> None:
    if candidate is None:
        return
    label = _normalise_label(candidate)
    if not label:
        return
    if label not in seen:
        seen.add(label)
        labels.append(label)


def _normalise_label_sequence(value: object | None) -> tuple[str, ...]:
    labels: list[str] = []
    seen: set[str] = set()

    def _collect(payload: object | None) -> None:
        if payload is None:
            return
        if isinstance(payload, str):
            _append_label(labels, seen, payload)
            return
        if isinstance(payload, (list, tuple, set, frozenset)):
            for item in payload:
                _collect(item)
            return
        if isinstance(payload, Mapping):
            for key in ("names", "name", "label", "id"):
                if key in payload:
                    _collect(payload[key])
            for key in ("regulations", "regulation", "venues", "venue"):
                if key in payload:
                    _collect(payload[key])
            for value in payload.values():
                if isinstance(value, (Mapping, list, tuple, set, frozenset)):
                    _collect(value)
            return
        _append_label(labels, seen, payload)

    _collect(value)
    return tuple(labels)


def _metadata_labels(metadata: object | None, keys: tuple[str, ...]) -> tuple[str, ...]:
    collected: list[str] = []
    seen: set[str] = set()

    def _collect(payload: object | None) -> None:
        if payload is None:
            return
        if isinstance(payload, Mapping):
            for key in keys:
                if key in payload:
                    for label in _normalise_label_sequence(payload[key]):
                        if label not in seen:
                            seen.add(label)
                            collected.append(label)
            for value in payload.values():
                if isinstance(value, (Mapping, list, tuple, set, frozenset)):
                    _collect(value)
            return
        if isinstance(payload, (list, tuple, set, frozenset)):
            for item in payload:
                _collect(item)

    _collect(metadata)
    return tuple(collected)


def _status_blocks(status: object | None) -> bool:
    if status is None:
        return False
    if isinstance(status, RegulatoryTelemetryStatus):
        return status is not RegulatoryTelemetryStatus.ok
    if isinstance(status, bool):
        return not status
    if isinstance(status, Mapping):
        for key in ("status", "state", "value", "label"):
            if key in status:
                return _status_blocks(status[key])
        return True
    if isinstance(status, (list, tuple, set, frozenset)):
        return any(_status_blocks(item) for item in status)
    label = _normalise_label(status)
    if not label:
        return True
    if label in {"ok", "pass", "allowed", "green", "open", "available"}:
        return False
    return True


def _normalise_regulatory_blocklist(statuses: object | None) -> frozenset[str]:
    blocked: list[str] = []
    seen: set[str] = set()

    def _add(name: object | None) -> None:
        label = _normalise_label(name)
        if not label:
            return
        if label not in seen:
            seen.add(label)
            blocked.append(label)

    if statuses is None:
        return frozenset()
    if isinstance(statuses, Mapping):
        for name, status in statuses.items():
            if _status_blocks(status):
                _add(name)
        return frozenset(blocked)
    if isinstance(statuses, (list, tuple, set, frozenset)):
        for name in statuses:
            _add(name)
        return frozenset(blocked)
    _add(statuses)
    return frozenset(blocked)


def _venue_closed(status: object | None) -> bool:
    if isinstance(status, RegulatoryTelemetryStatus):
        return status is not RegulatoryTelemetryStatus.ok
    if isinstance(status, bool):
        return not status
    if isinstance(status, Mapping):
        for key in ("status", "state", "value", "label"):
            if key in status:
                return _venue_closed(status[key])
        return True
    if isinstance(status, (list, tuple, set, frozenset)):
        return any(_venue_closed(item) for item in status)
    label = _normalise_label(status)
    if not label:
        return True
    if label in {"open", "available", "ok", "green", "trading", "live"}:
        return False
    return True


def _normalise_venue_blocklist(statuses: object | None) -> frozenset[str]:
    closed: list[str] = []
    seen: set[str] = set()

    def _add(name: object | None) -> None:
        label = _normalise_label(name)
        if not label:
            return
        if label not in seen:
            seen.add(label)
            closed.append(label)

    if statuses is None:
        return frozenset()
    if isinstance(statuses, Mapping):
        for name, status in statuses.items():
            if _venue_closed(status):
                _add(name)
        return frozenset(closed)
    if isinstance(statuses, (list, tuple, set, frozenset)):
        for name in statuses:
            _add(name)
        return frozenset(closed)
    _add(statuses)
    return frozenset(closed)


class _ConstraintEvaluator:
    def __init__(self, regulatory_status: object | None, venue_status: object | None) -> None:
        self._blocked_regulations = _normalise_regulatory_blocklist(regulatory_status)
        self._closed_venues = _normalise_venue_blocklist(venue_status)

    def allows(self, metadata: object | None) -> bool:
        if not self._blocked_regulations and not self._closed_venues:
            return True
        regulations = _metadata_labels(metadata, ("regulations", "regulation"))
        if regulations and any(reg in self._blocked_regulations for reg in regulations):
            return False
        venues = _metadata_labels(metadata, ("venues", "venue"))
        if venues and any(venue in self._closed_venues for venue in venues):
            return False
        return True

    @property
    def has_constraints(self) -> bool:
        return bool(self._blocked_regulations or self._closed_venues)


def _prepare_transition_metadata(
    metadata: object | None, action_entry: Mapping[str, object] | None
) -> object | None:
    base_metadata: Mapping[str, object] | None
    if isinstance(metadata, Mapping):
        base_metadata = dict(metadata)
    elif isinstance(metadata, (list, tuple, set, frozenset)):
        base_metadata = {"factors": list(metadata)}
    elif metadata is None:
        base_metadata = None
    else:
        base_metadata = {"factors": [metadata]}

    extras: dict[str, object] = {}
    if isinstance(action_entry, Mapping):
        constraints = action_entry.get("constraints")
        if isinstance(constraints, Mapping):
            for key, value in constraints.items():
                extras.setdefault(key, value)
        for key in ("regulations", "regulation", "venues", "venue"):
            if key in action_entry:
                extras.setdefault(key, action_entry[key])

    if extras:
        if base_metadata is None:
            base_metadata = dict(extras)
        else:
            for key, value in extras.items():
                base_metadata.setdefault(key, value)
        return base_metadata

    return base_metadata if base_metadata is not None else metadata


@dataclass(frozen=True)
class MuZeroLiteStep:
    """Single decision step inside the simulated tree."""

    depth: int
    action: str
    prior: float
    base_edge: float
    adjusted_edge: float
    discount: float
    discounted_contribution: float

    def as_dict(self) -> dict[str, float | int | str]:
        return {
            "depth": self.depth,
            "action": self.action,
            "prior": self.prior,
            "base_edge": self.base_edge,
            "adjusted_edge": self.adjusted_edge,
            "discount": self.discount,
            "discounted_contribution": self.discounted_contribution,
        }


@dataclass(frozen=True)
class MuZeroLitePath:
    """Rollout path summary returned by the MuZero-lite simulation."""

    steps: tuple[MuZeroLiteStep, ...]
    leaf_value: float
    total_return: float
    probability: float

    @property
    def actions(self) -> tuple[str, ...]:
        return tuple(step.action for step in self.steps)

    def as_dict(self) -> dict[str, object]:
        return {
            "actions": list(self.actions),
            "leaf_value": self.leaf_value,
            "probability": self.probability,
            "total_return": self.total_return,
            "steps": [step.as_dict() for step in self.steps],
        }


@dataclass(frozen=True)
class MuZeroLiteTreeResult:
    """Aggregate result for a MuZero-lite short-horizon rollout."""

    root_state: object
    horizon: int
    discount: float
    root_value: float
    expected_return: float
    paths: tuple[MuZeroLitePath, ...]

    @property
    def best_path(self) -> MuZeroLitePath | None:
        return self.paths[0] if self.paths else None

    def as_dict(self) -> dict[str, object]:
        return {
            "root_state": self.root_state,
            "horizon": self.horizon,
            "discount": self.discount,
            "root_value": self.root_value,
            "expected_return": self.expected_return,
            "paths": [path.as_dict() for path in self.paths],
            "best_path": self.best_path.as_dict() if self.best_path is not None else None,
        }


PolicyLike = Mapping[object, Mapping[str, float] | Sequence[tuple[str, float]]] | Callable[[Any], Mapping[str, float]]
TransitionLike = Mapping[
    object,
    Mapping[str, Mapping[str, object] | Sequence[object] | tuple[object, ...]] |
    Sequence[tuple[str, object]] |
    Sequence[tuple[str, object, object]]
] | Callable[[Any, str], tuple[Any, float, object | None]]
ValueLike = Mapping[object, float] | Callable[[Any], float]


def simulate_short_horizon_futures(
    root_state: object,
    *,
    policy: PolicyLike,
    transition_model: TransitionLike,
    value_model: ValueLike | None = None,
    horizon: int = 2,
    discount: float = 0.97,
    causal_edge_adjustments: Mapping[str, float] | None = None,
    max_branches: int | None = None,
    regulatory_status: object | None = None,
    venue_status: object | None = None,
) -> MuZeroLiteTreeResult:
    """Roll out a short MuZero-style tree with optional causal adjustments.

    Parameters
    ----------
    root_state:
        Identifier of the starting state.
    policy:
        Callable or mapping yielding per-action priors for a state.  Mappings may
        supply nested mappings (``{"buy": 0.6}``) or sequences of ``(action, prior)``.
    transition_model:
        Callable or mapping describing state transitions.  When provided as a
        mapping, each entry can be a mapping with keys ``state``/``next_state``
        and ``edge``/``reward`` or a tuple ``(next_state, edge[, metadata])``.
    value_model:
        Optional callable/mapping providing a leaf value for a state.
    horizon:
        Maximum depth (number of decisions) to roll out.  Must be positive.
    discount:
        Per-step discount factor applied to edges and leaf value.
    causal_edge_adjustments:
        Mapping of causal factors to additive edge adjustments.  Keys matching
        action names are also applied.
    max_branches:
        Optional limit on how many actions to expand per node, ranked by prior.
    regulatory_status:
        Optional mapping or sequence describing regulatory statuses.  Actions
        associated with blocked regulations are skipped.
    venue_status:
        Optional mapping or sequence describing venue availability.  Actions
        targeting unavailable venues are skipped.
    """

    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if not math.isfinite(discount) or discount <= 0.0:
        raise ValueError("discount must be a positive finite value")

    policy_fn = _build_policy_fn(policy)
    transition_fn = _build_transition_fn(transition_model)
    value_fn = _build_value_fn(value_model)

    root_value = value_fn(root_state)

    paths: list[MuZeroLitePath] = []
    constraint_evaluator = _ConstraintEvaluator(regulatory_status, venue_status)
    blocked_due_to_constraints = False

    def _dfs(
        state: object,
        depth: int,
        cumulative: float,
        probability: float,
        steps: tuple[MuZeroLiteStep, ...],
    ) -> None:
        nonlocal blocked_due_to_constraints
        if depth >= horizon:
            leaf_val = value_fn(state)
            discounted_leaf = leaf_val * (discount ** depth)
            paths.append(
                MuZeroLitePath(
                    steps=steps,
                    leaf_value=leaf_val,
                    total_return=cumulative + discounted_leaf,
                    probability=probability,
                )
            )
            return

        priors = policy_fn(state)
        if not priors:
            leaf_val = value_fn(state)
            discounted_leaf = leaf_val * (discount ** depth)
            paths.append(
                MuZeroLitePath(
                    steps=steps,
                    leaf_value=leaf_val,
                    total_return=cumulative + discounted_leaf,
                    probability=probability,
                )
            )
            return

        branches = sorted(priors.items(), key=lambda item: item[1], reverse=True)
        if max_branches is not None and max_branches > 0:
            branches = branches[:max_branches]

        for action, prior in branches:
            next_state, base_edge, metadata = transition_fn(state, action)
            if not constraint_evaluator.allows(metadata):
                if constraint_evaluator.has_constraints:
                    blocked_due_to_constraints = True
                continue
            adjusted_edge = _apply_causal_adjustments(
                action,
                base_edge,
                metadata,
                causal_edge_adjustments,
            )
            discount_factor = discount ** depth
            contribution = adjusted_edge * discount_factor
            step = MuZeroLiteStep(
                depth=depth + 1,
                action=action,
                prior=prior,
                base_edge=base_edge,
                adjusted_edge=adjusted_edge,
                discount=discount_factor,
                discounted_contribution=contribution,
            )
            _dfs(
                next_state,
                depth + 1,
                cumulative + contribution,
                probability * max(prior, 0.0),
                steps + (step,),
            )

    _dfs(root_state, 0, 0.0, 1.0, tuple())

    if not paths:
        if constraint_evaluator.has_constraints and blocked_due_to_constraints:
            raise ValueError(
                "no futures generated; transitions blocked by regulatory or venue constraints"
            )
        # Should not occur, but keep defensive fall-back.
        leaf_val = value_fn(root_state)
        paths = [
            MuZeroLitePath(
                steps=tuple(),
                leaf_value=leaf_val,
                total_return=leaf_val * (discount ** horizon),
                probability=1.0,
            )
        ]

    probability_mass = sum(max(path.probability, 0.0) for path in paths)
    if probability_mass > 0.0:
        expected_return = sum(path.total_return * max(path.probability, 0.0) for path in paths) / probability_mass
    else:
        expected_return = root_value

    ordered_paths = tuple(sorted(paths, key=lambda path: (-path.total_return, path.actions)))

    return MuZeroLiteTreeResult(
        root_state=root_state,
        horizon=horizon,
        discount=discount,
        root_value=root_value,
        expected_return=expected_return,
        paths=ordered_paths,
    )


def _build_policy_fn(model: PolicyLike) -> Callable[[Any], Mapping[str, float]]:
    if callable(model):
        return model
    if not isinstance(model, Mapping):
        raise TypeError("policy must be a callable or mapping")

    def _policy(state: object) -> Mapping[str, float]:
        entry = model.get(state)
        if entry is None:
            return {}
        if isinstance(entry, Mapping):
            return {str(action): float(prior) for action, prior in entry.items()}
        if isinstance(entry, Sequence):
            result: dict[str, float] = {}
            for item in entry:
                if not isinstance(item, Sequence) or len(item) < 2:
                    raise ValueError("policy sequence entries must be (action, prior)")
                action = str(item[0])
                prior = float(item[1])
                result[action] = prior
            return result
        raise TypeError("policy mapping values must be mappings or sequences of (action, prior)")

    return _policy


def _build_transition_fn(model: TransitionLike) -> Callable[[Any, str], tuple[Any, float, object | None]]:
    if callable(model):
        return model
    if not isinstance(model, Mapping):
        raise TypeError("transition_model must be a callable or mapping")

    def _transition(state: object, action: str) -> tuple[Any, float, object | None]:
        state_entry = model.get(state)
        if state_entry is None:
            raise KeyError(f"no transition defined for state {state!r}")

        if isinstance(state_entry, Mapping):
            action_entry = state_entry.get(action)
            if action_entry is None:
                raise KeyError(f"no transition defined for action {action!r} in state {state!r}")
            if isinstance(action_entry, Mapping):
                next_state = action_entry.get("next_state")
                if next_state is None:
                    next_state = action_entry.get("state")
                if next_state is None:
                    raise KeyError("transition mapping must include 'state' or 'next_state'")
                if "edge" in action_entry:
                    edge = action_entry["edge"]
                elif "reward" in action_entry:
                    edge = action_entry["reward"]
                else:
                    raise KeyError("transition mapping must include 'edge' or 'reward'")
                raw_metadata = action_entry.get("metadata")
                if raw_metadata is None:
                    for meta_key in ("causal", "causal_keys", "factors"):
                        if meta_key in action_entry:
                            raw_metadata = action_entry[meta_key]
                            break
                metadata = _prepare_transition_metadata(raw_metadata, action_entry)
                return next_state, float(edge), metadata
            if isinstance(action_entry, Sequence):
                if len(action_entry) < 2:
                    raise ValueError("transition sequence must provide at least (state, edge)")
                next_state = action_entry[0]
                edge = action_entry[1]
                raw_metadata = action_entry[2] if len(action_entry) >= 3 else None
                metadata = _prepare_transition_metadata(raw_metadata, None)
                return next_state, float(edge), metadata
            raise TypeError("transition entries must be mappings or sequences")

        if isinstance(state_entry, Sequence):
            for candidate in state_entry:
                if not isinstance(candidate, Sequence) or len(candidate) < 2:
                    raise ValueError("transition sequence entries must be (action, state[, edge])")
                if str(candidate[0]) == action:
                    if len(candidate) == 2:
                        next_state, edge = candidate[1], 0.0
                        metadata = None
                    else:
                        next_state = candidate[1]
                        edge = candidate[2]
                        raw_metadata = candidate[3] if len(candidate) >= 4 else None
                        metadata = _prepare_transition_metadata(raw_metadata, None)
                    return next_state, float(edge), metadata
            raise KeyError(f"no transition defined for action {action!r} in state {state!r}")
        raise TypeError("state transition entries must be mappings or sequences")

    return _transition


def _build_value_fn(model: ValueLike | None) -> Callable[[Any], float]:
    if model is None:
        return lambda _state: 0.0
    if callable(model):
        return model
    if not isinstance(model, Mapping):
        raise TypeError("value_model must be a callable or mapping")

    def _value(state: object) -> float:
        value = model.get(state)
        if value is None:
            return 0.0
        return float(value)

    return _value


def _apply_causal_adjustments(
    action: str,
    base_edge: float,
    metadata: object | None,
    adjustments: Mapping[str, float] | None,
) -> float:
    if adjustments is None:
        return float(base_edge)

    total = float(base_edge) + adjustments.get(action, 0.0)

    if metadata is None:
        return total

    if isinstance(metadata, Mapping):
        keys: list[str] = []
        for special_key in ("causal_keys", "factors", "drivers"):
            if special_key in metadata:
                keys.extend(_coerce_iterable(metadata[special_key]))
        for key, weight in metadata.items():
            if key in {"causal_keys", "factors", "drivers"}:
                continue
            if isinstance(weight, (int, float)) and math.isfinite(weight):
                total += adjustments.get(str(key), 0.0) * float(weight)
            elif isinstance(weight, str):
                keys.append(weight)
        for key in keys:
            total += adjustments.get(str(key), 0.0)
        return total

    if isinstance(metadata, (list, tuple, set, frozenset)):
        for key in metadata:
            total += adjustments.get(str(key), 0.0)
        return total

    return total + adjustments.get(str(metadata), 0.0)


def _coerce_iterable(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, (list, tuple, set, frozenset)):
        return [str(item) for item in value]
    return [str(value)]
