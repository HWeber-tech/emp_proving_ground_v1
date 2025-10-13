"""Regime-aware fitness aggregation helpers used by the policy router.

The roadmap calls for tournament selection across a regime grid so tactic
decisions can account for historical fitness instead of relying solely on the
latest score.  The lightweight table defined here keeps running aggregates for
each tactic/regime pair and exposes normalised metrics the router can consume
without introducing heavyweight dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Mapping, MutableMapping, Sequence


@dataclass(slots=True)
class RegimeFitnessStats:
    """Aggregate metrics for a single tactic within a specific regime."""

    observations: int = 0
    total_score: float = 0.0
    total_base_score: float = 0.0
    total_multiplier: float = 0.0
    max_score: float | None = None
    min_score: float | None = None
    last_score: float | None = None
    last_updated_at: datetime | None = None

    def update(
        self,
        *,
        score: float,
        base_score: float,
        multiplier: float,
        timestamp: datetime | None,
    ) -> None:
        self.observations += 1
        self.total_score += score
        self.total_base_score += base_score
        self.total_multiplier += multiplier
        self.last_score = score

        if self.max_score is None or score > self.max_score:
            self.max_score = score
        if self.min_score is None or score < self.min_score:
            self.min_score = score

        if timestamp is not None:
            if self.last_updated_at is None or timestamp > self.last_updated_at:
                self.last_updated_at = timestamp

    def average_score(self) -> float:
        if not self.observations:
            return 0.0
        return self.total_score / self.observations

    def average_base_score(self) -> float:
        if not self.observations:
            return 0.0
        return self.total_base_score / self.observations

    def average_multiplier(self) -> float:
        if not self.observations:
            return 0.0
        return self.total_multiplier / self.observations

    def as_dict(self) -> Mapping[str, object]:
        return {
            "observations": self.observations,
            "average_score": self.average_score(),
            "average_base_score": self.average_base_score(),
            "average_multiplier": self.average_multiplier(),
            "max_score": self.max_score,
            "min_score": self.min_score,
            "last_score": self.last_score,
            "last_updated_at": self.last_updated_at.isoformat()
            if self.last_updated_at is not None
            else None,
        }


@dataclass(slots=True)
class TacticRegimeFitness:
    """Aggregate fitness metrics for a tactic across all regimes."""

    tactic_id: str
    total_observations: int = 0
    total_score: float = 0.0
    total_base_score: float = 0.0
    total_multiplier: float = 0.0
    last_updated_at: datetime | None = None
    regimes: MutableMapping[str, RegimeFitnessStats] = field(default_factory=dict)

    def update(
        self,
        *,
        regime: str,
        score: float,
        base_score: float,
        multiplier: float,
        timestamp: datetime | None,
    ) -> None:
        stats = self.regimes.setdefault(regime, RegimeFitnessStats())
        stats.update(score=score, base_score=base_score, multiplier=multiplier, timestamp=timestamp)

        self.total_observations += 1
        self.total_score += score
        self.total_base_score += base_score
        self.total_multiplier += multiplier
        if timestamp is not None:
            if self.last_updated_at is None or timestamp > self.last_updated_at:
                self.last_updated_at = timestamp

    def average_score(self) -> float:
        if not self.total_observations:
            return 0.0
        return self.total_score / self.total_observations

    def average_base_score(self) -> float:
        if not self.total_observations:
            return 0.0
        return self.total_base_score / self.total_observations

    def average_multiplier(self) -> float:
        if not self.total_observations:
            return 0.0
        return self.total_multiplier / self.total_observations

    def regime_average(self, regime: str) -> float:
        stats = self.regimes.get(regime)
        return stats.average_score() if stats else 0.0

    def regime_observations(self, regime: str) -> int:
        stats = self.regimes.get(regime)
        return stats.observations if stats else 0

    def regime_diversity(self) -> int:
        return sum(1 for stats in self.regimes.values() if stats.observations > 0)

    def as_dict(self) -> Mapping[str, object]:
        return {
            "tactic_id": self.tactic_id,
            "total_observations": self.total_observations,
            "average_score": self.average_score(),
            "average_base_score": self.average_base_score(),
            "average_multiplier": self.average_multiplier(),
            "last_updated_at": self.last_updated_at.isoformat()
            if self.last_updated_at is not None
            else None,
            "regimes": {regime: stats.as_dict() for regime, stats in self.regimes.items()},
        }


class RegimeFitnessTable:
    """Maintain tactic fitness aggregates across the regime grid."""

    def __init__(self) -> None:
        self._records: MutableMapping[str, TacticRegimeFitness] = {}
        self._regime_decision_counts: MutableMapping[str, int] = {}

    def update(
        self,
        *,
        regime: str,
        entries: Sequence[Mapping[str, object]],
        decision_timestamp: datetime | None,
    ) -> None:
        regime_key = str(regime).strip().lower() or "unknown"
        for entry in entries:
            tactic = entry.get("tactic")
            tactic_id = getattr(tactic, "tactic_id", None)
            if not tactic_id:
                continue

            score = float(entry.get("score", 0.0) or 0.0)
            base_score = float(entry.get("base_score", score) or 0.0)
            multiplier = float(entry.get("multiplier", 1.0) or 0.0)

            record = self._records.setdefault(tactic_id, TacticRegimeFitness(tactic_id=tactic_id))
            record.update(
                regime=regime_key,
                score=score,
                base_score=base_score,
                multiplier=multiplier,
                timestamp=decision_timestamp,
            )

        self._regime_decision_counts[regime_key] = self._regime_decision_counts.get(regime_key, 0) + 1

    def record(self, tactic_id: str) -> TacticRegimeFitness | None:
        return self._records.get(tactic_id)

    def regime_decision_count(self, regime: str) -> int:
        regime_key = str(regime).strip().lower() or "unknown"
        return self._regime_decision_counts.get(regime_key, 0)

    def regimes(self) -> Mapping[str, Mapping[str, int]]:
        return {
            regime: {"decisions": count}
            for regime, count in self._regime_decision_counts.items()
        }

    def snapshot(self) -> Mapping[str, object]:
        return {
            "tactics": {tactic_id: record.as_dict() for tactic_id, record in self._records.items()},
            "regimes": self.regimes(),
        }

