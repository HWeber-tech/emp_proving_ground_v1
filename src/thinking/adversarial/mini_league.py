"""Lightweight mini-league coordination for self-play curricula.

Implements roadmap task F.2.1 by providing a structured league with four pools
of agents: ``current`` training policy, ``best`` historically promoted policy,
``exploit`` specialists probing weaknesses, and ``chaos`` adversaries that
maintain exploration pressure.

The implementation favours deterministic, easily serialisable state so other
systems (planner, orchestrators, telemetry) can consume snapshots without
needing to understand live Python objects.  The API intentionally stays small:
register agents, generate matchups, and record match results.
"""

from __future__ import annotations

import math
from collections import deque
import uuid
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Iterable, Mapping, MutableMapping, Sequence

__all__ = [
    "LeagueSlot",
    "LeagueEntry",
    "LeagueMatchup",
    "LeagueResult",
    "ExploitabilityComparison",
    "ExploitabilityObservation",
    "MiniLeague",
]


class LeagueSlot(str, Enum):
    """Role buckets for the self-play mini-league."""

    CURRENT = "current"
    BEST = "best"
    EXPLOIT = "exploit"
    CHAOS = "chaos"


@dataclass(slots=True)
class LeagueEntry:
    """Agent metadata tracked within the league."""

    agent_id: str
    score: float | None = None
    games_played: int = 0
    tags: tuple[str, ...] = ()
    metadata: MutableMapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.agent_id:
            raise ValueError("agent_id must be a non-empty string")
        if self.score is not None:
            self.score = float(self.score)
        self.games_played = int(self.games_played)
        if self.games_played < 0:
            raise ValueError("games_played cannot be negative")
        # Normalise metadata to a detached, mutable dict for consumer updates.
        if isinstance(self.metadata, Mapping):
            self.metadata = dict(self.metadata)
        else:
            self.metadata = {}

    def as_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "agent_id": self.agent_id,
            "games_played": int(self.games_played),
        }
        if self.score is not None:
            payload["score"] = float(self.score)
        if self.tags:
            payload["tags"] = list(self.tags)
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload

    def record_game(self, *, score: float | None = None) -> None:
        self.games_played += 1
        if score is not None:
            self.score = score


@dataclass(slots=True, frozen=True)
class LeagueMatchup:
    """Planned match between two agents in the league."""

    challenger: LeagueEntry
    opponent: LeagueEntry
    challenger_slot: LeagueSlot
    opponent_slot: LeagueSlot

    def as_dict(self) -> dict[str, object]:
        return {
            "challenger": self.challenger.as_dict(),
            "opponent": self.opponent.as_dict(),
            "challenger_slot": self.challenger_slot.value,
            "opponent_slot": self.opponent_slot.value,
        }


@dataclass(slots=True, frozen=True)
class LeagueResult:
    """Outcome of a league matchup."""

    matchup: LeagueMatchup
    challenger_score: float
    opponent_score: float
    metadata: Mapping[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        payload = {
            "matchup": self.matchup.as_dict(),
            "challenger_score": float(self.challenger_score),
            "opponent_score": float(self.opponent_score),
        }
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload

    def winner(self) -> str | None:
        if self.challenger_score == self.opponent_score:
            return None
        return (
            self.matchup.challenger.agent_id
            if self.challenger_score > self.opponent_score
            else self.matchup.opponent.agent_id
        )


@dataclass(slots=True, frozen=True)
class ExploitabilityComparison:
    """Comparison between the current policy and a reference agent."""

    slot: LeagueSlot
    agent_id: str
    metric: float
    turnover: float | None
    turnover_diff_pct: float | None
    gap: float
    turnover_variance: float | None = None
    inventory_variance: float | None = None
    lagrangian_penalty: float | None = None
    lagrangian_adjusted_gap: float | None = None


@dataclass(slots=True, frozen=True)
class ExploitabilityObservation:
    """Exploitability metric snapshot derived from mini-league rosters."""

    metric: str
    tolerance_pct: float
    current_agent_id: str | None
    current_metric: float | None
    current_turnover: float | None
    comparisons: tuple[ExploitabilityComparison, ...]
    selected_gap: float | None
    selected_slot: LeagueSlot | None
    selected_agent_id: str | None
    selected_penalty: float | None = None
    wow_delta: float | None = None


def _coerce_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _extract_metric(entry: LeagueEntry, metric: str) -> float | None:
    if metric in entry.metadata:
        candidate = _coerce_float(entry.metadata[metric])
        if candidate is not None:
            return candidate
    if metric == "score" and entry.score is not None:
        return float(entry.score)
    if entry.score is not None and metric in {"sharpe", "performance"}:
        return float(entry.score)
    if "score" in entry.metadata:
        candidate = _coerce_float(entry.metadata["score"])
        if candidate is not None:
            return candidate
    return None


def _extract_turnover(entry: LeagueEntry, turnover_key: str) -> float | None:
    if turnover_key not in entry.metadata:
        return None
    return _coerce_float(entry.metadata[turnover_key])


def _turnover_diff_pct(base: float, other: float) -> float | None:
    if base is None or other is None:
        return None
    if base == 0.0:
        return 0.0 if other == 0.0 else float("inf")
    return abs(other - base) / abs(base) * 100.0


def _extract_variance(entry: LeagueEntry, variance_key: str) -> float | None:
    if variance_key not in entry.metadata:
        return None
    value = _coerce_float(entry.metadata[variance_key])
    if value is None:
        return None
    if value < 0.0:
        return 0.0
    return value


def _constraint_difference(candidate: float | None, target: float | None) -> float | None:
    if candidate is None or target is None:
        return None
    return float(candidate - target)


def _resolve_float_field(payload: Mapping[str, object], *candidates: str) -> float | None:
    for key in candidates:
        if key in payload:
            value = _coerce_float(payload[key])
            if value is not None:
                return value
    meta = payload.get("metadata")
    if isinstance(meta, Mapping):
        for key in candidates:
            if key in meta:
                value = _coerce_float(meta[key])
                if value is not None:
                    return value
    return None


def _resolve_str_field(payload: Mapping[str, object], *candidates: str) -> str | None:
    for key in candidates:
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    meta = payload.get("metadata")
    if isinstance(meta, Mapping):
        for key in candidates:
            value = meta.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def _normalise_observer_payloads(data: object) -> tuple[Mapping[str, object], ...]:
    if data is None:
        return ()

    payloads: list[Mapping[str, object]] = []

    def _extend_from_sequence(sequence: Sequence[object]) -> None:
        for item in sequence:
            if isinstance(item, Mapping):
                payloads.append(item)

    if isinstance(data, Mapping):
        for key in ("observers", "observer_profiles", "pattern_observers", "watchers", "agents", "entries"):
            candidate = data.get(key)
            if isinstance(candidate, Sequence) and not isinstance(candidate, (str, bytes)):
                _extend_from_sequence(candidate)
        if not payloads:
            values = list(data.values())
            if values and all(isinstance(value, Mapping) for value in values):
                payloads.extend(value for value in values if isinstance(value, Mapping))
        if not payloads and isinstance(data, Mapping):
            payloads.append(data)
    elif isinstance(data, Sequence) and not isinstance(data, (str, bytes)):
        _extend_from_sequence(data)

    return tuple(payloads)


class _LagrangianConstraintState:
    """Track dual variables enforcing turnover/inventory variance limits."""

    __slots__ = (
        "turnover_lambda",
        "inventory_lambda",
        "turnover_updates",
        "inventory_updates",
    )

    def __init__(self) -> None:
        self.turnover_lambda: float = 0.0
        self.inventory_lambda: float = 0.0
        self.turnover_updates: int = 0
        self.inventory_updates: int = 0

    def penalty(
        self,
        turnover_violation: float | None,
        inventory_violation: float | None,
    ) -> float:
        penalty = 0.0
        if (
            turnover_violation is not None
            and turnover_violation > 0.0
            and self.turnover_lambda > 0.0
        ):
            penalty += self.turnover_lambda * turnover_violation
        if (
            inventory_violation is not None
            and inventory_violation > 0.0
            and self.inventory_lambda > 0.0
        ):
            penalty += self.inventory_lambda * inventory_violation
        return penalty

    def update(
        self,
        turnover_violation: float | None,
        inventory_violation: float | None,
    ) -> None:
        if turnover_violation is not None:
            self.turnover_updates += 1
            lr = 1.0 / math.sqrt(self.turnover_updates)
            candidate = self.turnover_lambda + lr * turnover_violation
            self.turnover_lambda = max(0.0, candidate)
        if inventory_violation is not None:
            self.inventory_updates += 1
            lr = 1.0 / math.sqrt(self.inventory_updates)
            candidate = self.inventory_lambda + lr * inventory_violation
            self.inventory_lambda = max(0.0, candidate)


class MiniLeague:
    """Coordinate match scheduling between league roles."""

    def __init__(
        self,
        *,
        max_exploit: int = 6,
        max_chaos: int = 6,
        history_limit: int = 64,
        sharpness_floor: float = 0.15,
        calibration_brier_ceiling: float = 0.12,
        exploitability_gap_ceiling: float = 0.05,
    ) -> None:
        if max_exploit <= 0:
            raise ValueError("max_exploit must be positive")
        if max_chaos <= 0:
            raise ValueError("max_chaos must be positive")
        if history_limit <= 0:
            raise ValueError("history_limit must be positive")
        if sharpness_floor <= 0.0:
            raise ValueError("sharpness_floor must be positive")
        if calibration_brier_ceiling <= 0.0:
            raise ValueError("calibration_brier_ceiling must be positive")
        if exploitability_gap_ceiling < 0.0:
            raise ValueError("exploitability_gap_ceiling cannot be negative")

        self._slots: dict[LeagueSlot, list[LeagueEntry]] = {
            LeagueSlot.CURRENT: [],
            LeagueSlot.BEST: [],
            LeagueSlot.EXPLOIT: [],
            LeagueSlot.CHAOS: [],
        }
        self._max_entries = {
            LeagueSlot.EXPLOIT: max_exploit,
            LeagueSlot.CHAOS: max_chaos,
        }
        self._history: deque[LeagueResult] = deque(maxlen=history_limit)
        self._exploitability_observations: deque[ExploitabilityObservation] = deque(
            maxlen=history_limit
        )
        self._lagrangian_state = _LagrangianConstraintState()
        self._sharpness_floor = float(sharpness_floor)
        self._calibration_brier_ceiling = float(calibration_brier_ceiling)
        self._exploitability_gap_ceiling = float(exploitability_gap_ceiling)
        self._observer_watchlist: dict[str, dict[str, object]] = {}

    def register(self, slot: LeagueSlot, entry: LeagueEntry) -> None:
        roster = self._slots[slot]
        if slot in (LeagueSlot.CURRENT, LeagueSlot.BEST):
            roster.clear()
            roster.append(entry)
            return

        existing_index = self._find(slot, entry.agent_id)
        if existing_index is not None:
            roster[existing_index] = entry
        else:
            roster.append(entry)
        self._sort_slot(slot)
        self._trim_slot(slot)

    def current(self) -> LeagueEntry | None:
        return self._slots[LeagueSlot.CURRENT][0] if self._slots[LeagueSlot.CURRENT] else None

    def best(self) -> LeagueEntry | None:
        return self._slots[LeagueSlot.BEST][0] if self._slots[LeagueSlot.BEST] else None

    def roster(self, slot: LeagueSlot) -> tuple[LeagueEntry, ...]:
        return tuple(self._slots[slot])

    def promote_current_to_best(self, *, copy_metadata: bool = True) -> LeagueEntry | None:
        current_entry = self.current()
        if current_entry is None:
            return None
        sharpness = _extract_metric(current_entry, "sharpness")
        if sharpness is None or sharpness + 1e-9 < self._sharpness_floor:
            return None
        calibration_brier = _extract_metric(current_entry, "calibration_brier")
        if (
            calibration_brier is None
            or calibration_brier - 1e-9 > self._calibration_brier_ceiling
        ):
            return None
        latest_observation: ExploitabilityObservation | None = next(
            (obs for obs in reversed(self._exploitability_observations) if obs.selected_gap is not None),
            None,
        )
        if latest_observation is None or latest_observation.selected_gap is None:
            return None

        latest_gap = latest_observation.selected_gap
        if latest_gap > self._exploitability_gap_ceiling + 1e-9:
            return None

        previous_gap: float | None = next(
            (
                obs.selected_gap
                for obs in reversed(self._exploitability_observations)
                if obs is not latest_observation and obs.selected_gap is not None
            ),
            None,
        )
        if (
            previous_gap is not None
            and latest_gap > previous_gap + 1e-9
        ):
            return None
        promoted = replace(
            current_entry,
            metadata=dict(current_entry.metadata) if copy_metadata else {},
        )
        self.register(LeagueSlot.BEST, promoted)
        return promoted

    def remove(self, slot: LeagueSlot, agent_id: str) -> bool:
        roster = self._slots[slot]
        idx = self._find(slot, agent_id)
        if idx is None:
            return False
        removed_entry = roster[idx]
        del roster[idx]
        if slot is LeagueSlot.EXPLOIT:
            self._observer_watchlist.pop(removed_entry.agent_id, None)
        return True

    def schedule_round(self) -> tuple[LeagueMatchup, ...]:
        current_entry = self.current()
        if current_entry is None:
            return ()
        opponents: list[LeagueMatchup] = []
        for slot in (LeagueSlot.BEST, LeagueSlot.EXPLOIT, LeagueSlot.CHAOS):
            for opponent in self._slots[slot]:
                opponents.append(
                    LeagueMatchup(
                        challenger=current_entry,
                        opponent=opponent,
                        challenger_slot=LeagueSlot.CURRENT,
                        opponent_slot=slot,
                    )
                )
        return tuple(opponents)

    def record_result(self, result: LeagueResult) -> None:
        self._history.append(result)
        # Update the challenger slot (always current) with new score metadata
        matchup = result.matchup
        if matchup.challenger_slot is LeagueSlot.CURRENT:
            challenger = self.current()
            if challenger is not None and challenger.agent_id == matchup.challenger.agent_id:
                challenger.record_game(score=result.challenger_score)
        if matchup.opponent_slot in (LeagueSlot.EXPLOIT, LeagueSlot.CHAOS, LeagueSlot.BEST):
            roster = self._slots[matchup.opponent_slot]
            idx = self._find(matchup.opponent_slot, matchup.opponent.agent_id)
            if idx is not None:
                roster[idx].record_game(score=result.opponent_score)

    def history(self, *, limit: int | None = None) -> tuple[LeagueResult, ...]:
        if limit is None or limit >= len(self._history):
            return tuple(self._history)
        count = max(0, limit)
        items = list(self._history)[-count:]
        return tuple(items)

    def snapshot(self) -> dict[str, object]:
        snapshot = {
            slot.value: [entry.as_dict() for entry in roster]
            for slot, roster in self._slots.items()
        }
        if self._observer_watchlist:
            snapshot["observer_watchlist"] = [dict(entry) for entry in self._observer_watchlist.values()]
        return snapshot

    def observer_watchlist(self) -> tuple[dict[str, object], ...]:
        """Return a snapshot of observers the league is mimicking."""

        return tuple(dict(entry) for entry in self._observer_watchlist.values())

    def exploitability_observations(self) -> tuple[ExploitabilityObservation, ...]:
        return tuple(self._exploitability_observations)

    def compute_exploitability_observation(
        self,
        *,
        metric: str = "sharpe",
        turnover_key: str = "turnover",
        turnover_tolerance_pct: float = 10.0,
        turnover_variance_key: str = "turnover_variance",
        inventory_variance_key: str = "inventory_variance",
    ) -> ExploitabilityObservation:
        if turnover_tolerance_pct < 0:
            raise ValueError("turnover_tolerance_pct must be non-negative")

        current_entry = self.current()
        if current_entry is None:
            return ExploitabilityObservation(
                metric=metric,
                tolerance_pct=turnover_tolerance_pct,
                current_agent_id=None,
                current_metric=None,
                current_turnover=None,
                comparisons=(),
                selected_gap=None,
                selected_slot=None,
                selected_agent_id=None,
            )

        current_metric = _extract_metric(current_entry, metric)
        current_turnover = _extract_turnover(current_entry, turnover_key)
        current_turnover_variance = _extract_variance(current_entry, turnover_variance_key)
        current_inventory_variance = _extract_variance(current_entry, inventory_variance_key)

        comparisons: list[ExploitabilityComparison] = []
        turnover_violations: list[float] = []
        inventory_violations: list[float] = []
        if current_metric is not None and current_turnover is not None:
            for slot in (LeagueSlot.BEST, LeagueSlot.EXPLOIT):
                for entry in self._slots[slot]:
                    candidate_metric = _extract_metric(entry, metric)
                    candidate_turnover = _extract_turnover(entry, turnover_key)
                    if candidate_metric is None or candidate_turnover is None:
                        continue
                    turnover_diff = _turnover_diff_pct(current_turnover, candidate_turnover)
                    if turnover_diff is None or turnover_diff > turnover_tolerance_pct:
                        continue
                    candidate_turnover_variance = _extract_variance(
                        entry, turnover_variance_key
                    )
                    candidate_inventory_variance = _extract_variance(
                        entry, inventory_variance_key
                    )
                    turnover_violation = _constraint_difference(
                        candidate_turnover_variance, current_turnover_variance
                    )
                    inventory_violation = _constraint_difference(
                        candidate_inventory_variance, current_inventory_variance
                    )
                    gap = max(0.0, candidate_metric - current_metric)
                    penalty = self._lagrangian_state.penalty(
                        turnover_violation, inventory_violation
                    )
                    penalty_value = penalty if penalty > 0.0 else 0.0
                    adjusted_gap = max(0.0, gap - penalty_value)
                    comparisons.append(
                        ExploitabilityComparison(
                            slot=slot,
                            agent_id=entry.agent_id,
                            metric=candidate_metric,
                            turnover=candidate_turnover,
                            turnover_diff_pct=turnover_diff,
                            gap=gap,
                            turnover_variance=candidate_turnover_variance,
                            inventory_variance=candidate_inventory_variance,
                            lagrangian_penalty=penalty_value,
                            lagrangian_adjusted_gap=adjusted_gap,
                        )
                    )
                    if turnover_violation is not None:
                        turnover_violations.append(turnover_violation)
                    if inventory_violation is not None:
                        inventory_violations.append(inventory_violation)

        if comparisons:
            def _score(item: ExploitabilityComparison) -> float:
                if item.lagrangian_adjusted_gap is not None:
                    return float(item.lagrangian_adjusted_gap)
                return float(item.gap)

            selected = max(comparisons, key=_score)
            selected_gap = _score(selected)
            selected_slot = selected.slot
            selected_agent_id = selected.agent_id
            selected_penalty = selected.lagrangian_penalty
        else:
            selected_gap = None
            selected_slot = None
            selected_agent_id = None
            selected_penalty = None

        turnover_violation_avg = (
            sum(turnover_violations) / len(turnover_violations)
            if turnover_violations
            else None
        )
        inventory_violation_avg = (
            sum(inventory_violations) / len(inventory_violations)
            if inventory_violations
            else None
        )
        if turnover_violation_avg is not None or inventory_violation_avg is not None:
            self._lagrangian_state.update(turnover_violation_avg, inventory_violation_avg)

        return ExploitabilityObservation(
            metric=metric,
            tolerance_pct=turnover_tolerance_pct,
            current_agent_id=current_entry.agent_id,
            current_metric=current_metric,
            current_turnover=current_turnover,
            comparisons=tuple(comparisons),
            selected_gap=selected_gap,
            selected_slot=selected_slot,
            selected_agent_id=selected_agent_id,
            selected_penalty=selected_penalty,
        )

    def train_observer_mimics(
        self,
        observers: Sequence[Mapping[str, object]] | Mapping[str, object] | None,
        *,
        confidence_floor: float = 0.15,
    ) -> tuple[LeagueEntry, ...]:
        """
        Populate exploit slot with agents that mimic external observers.

        The observers payload is intentionally flexible.  Accepts:
        - Sequence[Mapping]: direct observer descriptors
        - Mapping containing keys such as ``observers`` or ``pattern_observers``
        - Single mapping describing one observer

        Each observer contributes a mimic entry tagged as an exploit specialist.
        Metadata records the source observer, observed pattern signature, and
        countermeasures that randomise behaviour to remain unpredictable.
        """

        payloads = _normalise_observer_payloads(observers)
        if not payloads:
            return ()

        created: list[LeagueEntry] = []
        for payload in payloads:
            confidence = _resolve_float_field(
                payload,
                "confidence",
                "likelihood",
                "strength",
                "weight",
                "observer_confidence",
                "score",
            )
            if confidence is None:
                confidence = 0.0
            else:
                confidence = max(0.0, confidence)

            anticipation = _resolve_float_field(
                payload,
                "anticipation",
                "anticipation_score",
                "anticipation_risk",
                "tracking_confidence",
                "watcher_confidence",
            )
            if anticipation is None:
                anticipation = 0.0
            else:
                anticipation = max(0.0, anticipation)

            if confidence < confidence_floor and len(payloads) > 1:
                continue

            source_id = _resolve_str_field(
                payload,
                "observer_id",
                "agent_id",
                "id",
                "name",
            )
            mimic_pattern = _resolve_str_field(
                payload,
                "pattern",
                "pattern_signature",
                "observed_pattern",
                "signature",
                "behavior_pattern",
            )

            mimic_agent_id = (
                f"mimic::{source_id}"
                if source_id
                else f"mimic::{uuid.uuid4()}"
            )

            countermeasures = {
                "entropy_rotation": True,
                "decoy_signals": anticipation >= 0.5,
                "shadow_execution": confidence >= 0.5,
                "jitter_window": max(0.05, min(0.5, 0.5 - min(0.45, anticipation * 0.2))),
            }

            mimic_score = max(0.0, confidence - anticipation * 0.5)
            entropy_baffle = max(0.1, min(0.9, 1.0 - min(confidence, 1.0)))

            metadata = {
                "league_slot": LeagueSlot.EXPLOIT.value,
                "mimic_source": source_id or "anonymous-observer",
                "mimicked_pattern": mimic_pattern,
                "observation_confidence": confidence,
                "anticipation_risk": anticipation,
                "entropy_baffle": entropy_baffle,
                "countermeasures": countermeasures,
                "tags": ["observer-mimic"],
            }

            extra_metadata = payload.get("metadata")
            if isinstance(extra_metadata, Mapping):
                for key, value in extra_metadata.items():
                    metadata.setdefault(f"observer_{key}", value)

            entry = LeagueEntry(
                agent_id=mimic_agent_id,
                score=mimic_score,
                tags=("observer-mimic", "exploit"),
                metadata=metadata,
            )

            self.register(LeagueSlot.EXPLOIT, entry)
            idx = self._find(LeagueSlot.EXPLOIT, mimic_agent_id)
            if idx is None:
                continue

            stored_entry = self._slots[LeagueSlot.EXPLOIT][idx]
            self._observer_watchlist[stored_entry.agent_id] = {
                "mimic_id": stored_entry.agent_id,
                "source_observer": metadata["mimic_source"],
                "pattern": metadata.get("mimicked_pattern"),
                "confidence": confidence,
                "anticipation_risk": anticipation,
                "entropy_baffle": entropy_baffle,
                "countermeasures": countermeasures,
            }

            created.append(stored_entry)

        limit = max(8, self._max_entries.get(LeagueSlot.EXPLOIT, 6) * 2)
        while len(self._observer_watchlist) > limit:
            oldest_key = next(iter(self._observer_watchlist))
            self._observer_watchlist.pop(oldest_key, None)

        return tuple(created)

    def record_exploitability_observation(
        self,
        *,
        metric: str = "sharpe",
        turnover_key: str = "turnover",
        turnover_tolerance_pct: float = 10.0,
        turnover_variance_key: str = "turnover_variance",
        inventory_variance_key: str = "inventory_variance",
    ) -> ExploitabilityObservation:
        observation = self.compute_exploitability_observation(
            metric=metric,
            turnover_key=turnover_key,
            turnover_tolerance_pct=turnover_tolerance_pct,
            turnover_variance_key=turnover_variance_key,
            inventory_variance_key=inventory_variance_key,
        )
        if observation.selected_gap is not None:
            previous_gap = next(
                (
                    obs.selected_gap
                    for obs in reversed(self._exploitability_observations)
                    if obs.selected_gap is not None
                ),
                None,
            )
            if previous_gap is not None:
                observation = replace(
                    observation,
                    wow_delta=observation.selected_gap - previous_gap,
                )
        self._exploitability_observations.append(observation)
        return observation

    def _find(self, slot: LeagueSlot, agent_id: str) -> int | None:
        for idx, entry in enumerate(self._slots[slot]):
            if entry.agent_id == agent_id:
                return idx
        return None

    def _sort_slot(self, slot: LeagueSlot) -> None:
        if slot not in (LeagueSlot.EXPLOIT, LeagueSlot.CHAOS):
            return
        roster = self._slots[slot]
        roster.sort(
            key=lambda entry: (
                -float(entry.score)
                if entry.score is not None
                else float("inf"),
                entry.agent_id,
            )
        )

    def _trim_slot(self, slot: LeagueSlot) -> None:
        limit = self._max_entries.get(slot)
        if limit is None:
            return
        roster = self._slots[slot]
        if len(roster) > limit:
            del roster[limit:]

    def __len__(self) -> int:  # pragma: no cover - convenience method
        return sum(len(roster) for roster in self._slots.values())

    def __contains__(self, agent_id: str) -> bool:  # pragma: no cover - convenience method
        return any(entry.agent_id == agent_id for roster in self._slots.values() for entry in roster)

    def __iter__(self) -> Iterable[LeagueEntry]:  # pragma: no cover - convenience helper
        for slot in LeagueSlot:
            yield from self._slots[slot]

    def __repr__(self) -> str:  # pragma: no cover - diagnostics
        rosters = {
            slot.value: [entry.agent_id for entry in roster]
            for slot, roster in self._slots.items()
        }
        return f"MiniLeague({rosters})"
