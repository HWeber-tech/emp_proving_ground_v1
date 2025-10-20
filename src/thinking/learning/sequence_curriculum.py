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

import math
from dataclasses import dataclass
from typing import Iterable, Literal, Mapping, Sequence

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
    rare_event_ratio: float = 0.15

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
        if not 0.0 <= self.rare_event_ratio <= 1.0:
            raise ValueError("rare_event_ratio must be between 0 and 1 inclusive")


def _default_stage_name(sequence_length: int, index: int) -> str:
    if sequence_length % 1024 == 0:
        base = f"seq_len_{sequence_length // 1024}k"
    else:
        base = f"seq_len_{sequence_length}"
    return base if index == 0 else f"{base}_{index}"


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
        sequence_lengths: Sequence[int] | None = None,
        milestones: Sequence[int] | None = None,
    ) -> None:
        if stages is not None and sequence_lengths is not None:
            raise ValueError("stages and sequence_lengths are mutually exclusive")

        if stages is None:
            base_lengths = sequence_lengths
            if base_lengths is None:
                base_lengths = [stage.sequence_length for stage in DEFAULT_SEQUENCE_LENGTH_STAGES]
            normalised_lengths = tuple(int(length) for length in base_lengths)
            if not normalised_lengths:
                raise ValueError("stages must contain at least one entry")
            if any(length <= 0 for length in normalised_lengths):
                raise ValueError("sequence lengths must be positive")
            if list(normalised_lengths) != sorted(normalised_lengths):
                raise ValueError("sequence lengths must be non-decreasing")

            default_ratios = [stage.rare_event_ratio for stage in DEFAULT_SEQUENCE_LENGTH_STAGES]
            derived: list[SequenceLengthStage] = []
            for index, length in enumerate(normalised_lengths):
                ratio = default_ratios[min(index, len(default_ratios) - 1)]
                derived.append(
                    SequenceLengthStage(
                        name=_default_stage_name(length, index),
                        sequence_length=length,
                        rare_event_ratio=ratio,
                    )
                )
            stages = tuple(derived)
        if not stages:
            raise ValueError("stages must contain at least one entry")

        validated: list[SequenceLengthStage] = []
        previous_length: int | None = None
        name_seen: set[str] = set()
        for index, stage in enumerate(stages):
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

        if milestones is None:
            self._milestones: tuple[int, ...] | None = None
        else:
            normalised_milestones = tuple(int(value) for value in milestones)
            if len(normalised_milestones) != len(self._stages) - 1:
                raise ValueError("milestones must contain len(stages) - 1 entries")
            if any(value <= 0 for value in normalised_milestones):
                raise ValueError("milestones must be positive integers")
            if list(normalised_milestones) != sorted(normalised_milestones):
                raise ValueError("milestones must be non-decreasing")
            self._milestones = normalised_milestones

        self._sequence_lengths = tuple(stage.sequence_length for stage in self._stages)

        self._index = 0
        self._steps_in_stage = 0
        self._total_steps = 0
        self._tokens_observed = 0
        self._metric_success_streak = 0
        self._history: list[CurriculumTransition] = []
        self._rare_event_residual = 0.0
        self._rare_event_counts = {"rare": 0, "main": 0}

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
    def milestones(self) -> tuple[int, ...] | None:
        return self._milestones

    @property
    def current_sequence_length(self) -> int:
        return self.current_stage.sequence_length

    @property
    def current_length(self) -> int:
        return self.current_sequence_length

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
    def tokens_observed(self) -> int:
        return self._tokens_observed

    @property
    def history(self) -> tuple[CurriculumTransition, ...]:
        return tuple(self._history)

    @property
    def rare_event_ratio(self) -> float:
        return self.current_stage.rare_event_ratio

    @property
    def rare_event_counts(self) -> tuple[int, int]:
        return self._rare_event_counts["rare"], self._rare_event_counts["main"]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def can_advance(self) -> bool:
        return self._index < len(self._stages) - 1

    def advance(self, reason: str | None = None, *, metric: float | None = None) -> SequenceLengthStage:
        if reason is None:
            reason = "manual"
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
            self._tokens_observed += int(steps)

        if force:
            used_reason = reason or "forced"
            return self._advance(used_reason, metric)

        self._maybe_promote_for_milestones()

        if not self.can_advance():
            return self.current_stage

        stage = self.current_stage

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
        self._tokens_observed = 0
        self._metric_success_streak = 0
        self._history.clear()
        self._rare_event_residual = 0.0
        self._rare_event_counts = {"rare": 0, "main": 0}

    def summary(self) -> dict[str, object]:
        return {
            "stages": [stage.name for stage in self._stages],
            "current_stage": self.current_stage.name,
            "stage_index": self._index,
            "steps_in_stage": self._steps_in_stage,
            "total_steps": self._total_steps,
            "tokens_observed": self._tokens_observed,
            "history": [transition.as_dict() for transition in self._history],
            "rare_event_ratio": self.rare_event_ratio,
            "rare_events_assigned": self._rare_event_counts["rare"],
            "main_events_assigned": self._rare_event_counts["main"],
        }

    def observe_tokens(self, tokens: int) -> int:
        stage = self.record_progress(steps=tokens)
        return stage.sequence_length

    def state_dict(self) -> dict[str, object]:
        return {
            "stage_index": self._index,
            "tokens_observed": self._tokens_observed,
            "sequence_lengths": self._sequence_lengths,
            "milestones": self._milestones,
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        stage_index = int(state.get("stage_index", 0))
        tokens_observed = int(state.get("tokens_observed", 0))
        if tokens_observed < 0:
            raise ValueError("tokens_observed must be non-negative")

        self._tokens_observed = tokens_observed
        self._total_steps = tokens_observed
        self._steps_in_stage = 0

        self._rare_event_residual = 0.0
        self._rare_event_counts = {"rare": 0, "main": 0}

        if self._milestones:
            self._index = 0
            self._maybe_promote_for_milestones()
        else:
            if stage_index < 0 or stage_index >= len(self._stages):
                raise ValueError("stage_index out of range")
            self._index = stage_index

        self._metric_success_streak = 0
        self._history.clear()

    def allocate_event_mix(self, batches: int) -> dict[str, int]:
        """Return how many rare vs. main samples to draw for ``batches``.

        Uses a fractional-residual scheduler so callers can supply any batch size
        while still achieving the stage's target rare-event ratio over time.
        """

        if batches < 0:
            raise ValueError("batches must be non-negative")
        if batches == 0:
            return {"rare": 0, "main": 0}

        ratio = self.rare_event_ratio
        if ratio >= 1.0:
            rare = batches
            main = 0
            self._rare_event_residual = 0.0
        elif ratio <= 0.0:
            rare = 0
            main = batches
            self._rare_event_residual = 0.0
        else:
            expected = ratio * batches + self._rare_event_residual
            rare = min(batches, math.floor(expected + 1e-9))
            # Ensure rare events surface periodically even when batches are tiny.
            if rare == 0 and expected >= 0.999:
                rare = 1
            self._rare_event_residual = expected - rare
            main = batches - rare
        self._rare_event_counts["rare"] += rare
        self._rare_event_counts["main"] += main
        return {"rare": rare, "main": main}

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
        self._rare_event_residual = 0.0
        self._ensure_tokens_cover_milestone()
        return self.current_stage

    def _ensure_tokens_cover_milestone(self) -> None:
        if not self._milestones or self._index == 0:
            return
        milestone = self._milestones[self._index - 1]
        if self._tokens_observed < milestone:
            self._tokens_observed = milestone
        if self._total_steps < milestone:
            self._total_steps = milestone

    def _maybe_promote_for_milestones(self) -> None:
        if not self._milestones:
            return
        while self.can_advance() and self._tokens_observed >= self._milestones[self._index]:
            milestone_value = self._milestones[self._index]
            self._advance(f"milestone_tokens={milestone_value}", None)


DEFAULT_SEQUENCE_LENGTH_STAGES: tuple[SequenceLengthStage, ...] = (
    SequenceLengthStage(
        name="seq_len_4k",
        sequence_length=4096,
        max_steps=50_000,
        rare_event_ratio=0.2,
    ),
    SequenceLengthStage(
        name="seq_len_8k",
        sequence_length=8192,
        min_steps=50_000,
        max_steps=150_000,
        rare_event_ratio=0.25,
    ),
    SequenceLengthStage(
        name="seq_len_16k",
        sequence_length=16384,
        min_steps=150_000,
        rare_event_ratio=0.3,
    ),
)


def build_curriculum(stages: Iterable[SequenceLengthStage] | None = None) -> SequenceLengthCurriculum:
    """Convenience helper mirroring historic factory naming."""

    return SequenceLengthCurriculum(tuple(stages) if stages is not None else None)
