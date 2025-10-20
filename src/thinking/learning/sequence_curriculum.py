"""Sequence length curriculum scheduler for chunked TBPTT (C.3.2).

The roadmap requires the trainer to respect a curriculum that expands the
effective sequence length from 4k tokens up to 16k once the model stabilises.
This module provides a light-weight state machine that tracks curriculum
stages, enforces minimum dwell times, and exposes explicit transitions so
training loops can log the reasons behind each promotion.

The implementation intentionally keeps a small API surface: define stages,
record progress (optionally with evaluation metrics), and advance when the
exit conditions are met.  Downstream callers remain responsible for deciding
when the curriculum should progress, but the helper centralises bookkeeping
and guards against accidental regressions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal, Sequence

__all__ = [
    "SequenceLengthStage",
    "CurriculumTransition",
    "SequenceLengthCurriculum",
    "DEFAULT_SEQUENCE_LENGTH_STAGES",
    "build_curriculum",
]


@dataclass(frozen=True)
class SequenceLengthStage:
    """Definition for a single curriculum stage."""

    name: str
    sequence_length: int
    min_steps: int = 0
    max_steps: int | None = None
    metric_threshold: float | None = None
    metric_comparator: Literal["<=", ">="] = "<="
    patience: int = 1

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("stage name must be a non-empty string")
        if self.sequence_length <= 0:
            raise ValueError("sequence_length must be positive")
        if self.min_steps < 0:
            raise ValueError("min_steps must be non-negative")
        if self.max_steps is not None and self.max_steps < self.min_steps:
            raise ValueError("max_steps cannot be less than min_steps")
        if self.metric_comparator not in {"<=", ">="}:
            raise ValueError("metric_comparator must be '<=' or '>='")
        if self.patience <= 0:
            raise ValueError("patience must be positive")


@dataclass(frozen=True)
class CurriculumTransition:
    """Record describing a curriculum promotion."""

    from_stage: str
    to_stage: str
    reason: str
    steps_spent: int
    total_steps: int
    metric_value: float | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "from_stage": self.from_stage,
            "to_stage": self.to_stage,
            "reason": self.reason,
            "steps_spent": self.steps_spent,
            "total_steps": self.total_steps,
            "metric_value": self.metric_value,
        }


class SequenceLengthCurriculum:
    """Stateful helper enforcing the sequence length curriculum."""

    def __init__(
        self,
        stages: Sequence[SequenceLengthStage] | None = None,
        *,
        epsilon: float = 1e-6,
    ) -> None:
        if stages is None:
            stages = DEFAULT_SEQUENCE_LENGTH_STAGES
        if not stages:
            raise ValueError("stages must contain at least one entry")

        validated: list[SequenceLengthStage] = []
        previous_length: int | None = None
        name_seen: set[str] = set()
        for stage in stages:
            if previous_length is not None and stage.sequence_length < previous_length:
                raise ValueError("sequence lengths must be non-decreasing")
            if stage.name in name_seen:
                raise ValueError("stage names must be unique")
            name_seen.add(stage.name)
            previous_length = stage.sequence_length
            validated.append(stage)

        self._stages: tuple[SequenceLengthStage, ...] = tuple(validated)
        if epsilon < 0:
            raise ValueError("epsilon must be non-negative")
        self._epsilon = float(epsilon)

        self._index = 0
        self._steps_in_stage = 0
        self._total_steps = 0
        self._metric_success_streak = 0
        self._history: list[CurriculumTransition] = []

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def stages(self) -> tuple[SequenceLengthStage, ...]:
        return self._stages

    @property
    def current_stage(self) -> SequenceLengthStage:
        return self._stages[self._index]

    @property
    def current_sequence_length(self) -> int:
        return self.current_stage.sequence_length

    @property
    def stage_index(self) -> int:
        return self._index

    @property
    def steps_in_stage(self) -> int:
        return self._steps_in_stage

    @property
    def total_steps(self) -> int:
        return self._total_steps

    @property
    def history(self) -> tuple[CurriculumTransition, ...]:
        return tuple(self._history)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def can_advance(self) -> bool:
        return self._index < len(self._stages) - 1

    def advance(self, reason: str, *, metric: float | None = None) -> SequenceLengthStage:
        if not reason:
            raise ValueError("reason must be a non-empty string")
        if not self.can_advance():
            return self.current_stage
        return self._advance(reason, metric)

    def record_progress(
        self,
        *,
        steps: int = 0,
        metric: float | None = None,
        force: bool = False,
        reason: str | None = None,
    ) -> SequenceLengthStage:
        if steps < 0:
            raise ValueError("steps must be non-negative")
        if steps:
            self._steps_in_stage += int(steps)
            self._total_steps += int(steps)

        if not self.can_advance():
            return self.current_stage

        stage = self.current_stage

        if force:
            used_reason = reason or "forced"
            return self._advance(used_reason, metric)

        if stage.max_steps is not None and self._steps_in_stage >= stage.max_steps:
            used_reason = reason or f"max_steps={stage.max_steps}"
            return self._advance(used_reason, metric)

        if stage.metric_threshold is None or metric is None:
            self._metric_success_streak = 0
            return stage

        if self._steps_in_stage < stage.min_steps:
            self._metric_success_streak = 0
            return stage

        if stage.metric_comparator == "<=" and metric <= stage.metric_threshold + self._epsilon:
            self._metric_success_streak += 1
        elif stage.metric_comparator == ">=" and metric >= stage.metric_threshold - self._epsilon:
            self._metric_success_streak += 1
        else:
            self._metric_success_streak = 0
            return stage

        if self._metric_success_streak < stage.patience:
            return stage

        used_reason = reason or (
            f"metric {stage.metric_comparator} {stage.metric_threshold} for {stage.patience}"
        )
        return self._advance(used_reason, metric)

    def reset(self) -> None:
        self._index = 0
        self._steps_in_stage = 0
        self._total_steps = 0
        self._metric_success_streak = 0
        self._history.clear()

    def summary(self) -> dict[str, object]:
        return {
            "stages": [stage.name for stage in self._stages],
            "current_stage": self.current_stage.name,
            "stage_index": self._index,
            "steps_in_stage": self._steps_in_stage,
            "total_steps": self._total_steps,
            "history": [transition.as_dict() for transition in self._history],
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _advance(self, reason: str, metric: float | None) -> SequenceLengthStage:
        previous = self.current_stage
        self._metric_success_streak = 0
        self._history.append(
            CurriculumTransition(
                from_stage=previous.name,
                to_stage=self._stages[self._index + 1].name,
                reason=reason,
                steps_spent=self._steps_in_stage,
                total_steps=self._total_steps,
                metric_value=metric,
            )
        )
        self._index += 1
        self._steps_in_stage = 0
        return self.current_stage


DEFAULT_SEQUENCE_LENGTH_STAGES: tuple[SequenceLengthStage, ...] = (
    SequenceLengthStage(name="seq_len_4k", sequence_length=4096, max_steps=50_000),
    SequenceLengthStage(name="seq_len_8k", sequence_length=8192, min_steps=50_000, max_steps=150_000),
    SequenceLengthStage(name="seq_len_16k", sequence_length=16384, min_steps=150_000),
)


def build_curriculum(stages: Iterable[SequenceLengthStage] | None = None) -> SequenceLengthCurriculum:
    """Convenience helper mirroring historic factory naming."""

    return SequenceLengthCurriculum(tuple(stages) if stages is not None else None)
