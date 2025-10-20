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

from collections import deque
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Iterable, Mapping, MutableMapping

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


class MiniLeague:
    """Coordinate match scheduling between league roles."""

    def __init__(
        self,
        *,
        max_exploit: int = 6,
        max_chaos: int = 6,
        history_limit: int = 64,
    ) -> None:
        if max_exploit <= 0:
            raise ValueError("max_exploit must be positive")
        if max_chaos <= 0:
            raise ValueError("max_chaos must be positive")
        if history_limit <= 0:
            raise ValueError("history_limit must be positive")

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
        latest_gap: float | None = None
        previous_gap: float | None = None
        gaps_found = 0
        for observation in reversed(self._exploitability_observations):
            if observation.selected_gap is None:
                continue
            if gaps_found == 0:
                latest_gap = observation.selected_gap
                gaps_found += 1
                continue
            previous_gap = observation.selected_gap
            break
        if (
            latest_gap is not None
            and previous_gap is not None
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
        del roster[idx]
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
        return {
            slot.value: [entry.as_dict() for entry in roster]
            for slot, roster in self._slots.items()
        }

    def exploitability_observations(self) -> tuple[ExploitabilityObservation, ...]:
        return tuple(self._exploitability_observations)

    def compute_exploitability_observation(
        self,
        *,
        metric: str = "sharpe",
        turnover_key: str = "turnover",
        turnover_tolerance_pct: float = 10.0,
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

        comparisons: list[ExploitabilityComparison] = []
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
                    gap = max(0.0, candidate_metric - current_metric)
                    comparisons.append(
                        ExploitabilityComparison(
                            slot=slot,
                            agent_id=entry.agent_id,
                            metric=candidate_metric,
                            turnover=candidate_turnover,
                            turnover_diff_pct=turnover_diff,
                            gap=gap,
                        )
                    )

        if comparisons:
            selected = max(comparisons, key=lambda item: item.gap)
            selected_gap = selected.gap
            selected_slot = selected.slot
            selected_agent_id = selected.agent_id
        else:
            selected_gap = None
            selected_slot = None
            selected_agent_id = None

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
        )

    def record_exploitability_observation(
        self,
        *,
        metric: str = "sharpe",
        turnover_key: str = "turnover",
        turnover_tolerance_pct: float = 10.0,
    ) -> ExploitabilityObservation:
        observation = self.compute_exploitability_observation(
            metric=metric,
            turnover_key=turnover_key,
            turnover_tolerance_pct=turnover_tolerance_pct,
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
