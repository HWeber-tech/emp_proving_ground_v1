"""Sequence length curriculum utilities for Phase C (C.3.2)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence


@dataclass(frozen=True)
class CurriculumStage:
    """Descriptor for a single curriculum stage."""

    index: int
    sequence_length: int
    milestone_tokens: int | None


class SequenceLengthCurriculum:
    """Manage the progression of TBPTT sequence lengths during training.

    The curriculum defaults to the Phase C specification and exposes a small API
    that lets training loops incrementally increase the sequence length as
    stability improves.  Progression can be manual (via :meth:`advance`) or
    automatically driven by cumulative tokens observed through
    :meth:`observe_tokens` when milestones are provided.
    """

    DEFAULT_SEQUENCE_LENGTHS: tuple[int, int, int] = (4096, 8192, 16384)

    def __init__(
        self,
        *,
        sequence_lengths: Sequence[int] | None = None,
        milestones: Sequence[int] | None = None,
    ) -> None:
        if sequence_lengths is None:
            sequence_lengths = self.DEFAULT_SEQUENCE_LENGTHS
        if not sequence_lengths:
            raise ValueError("sequence_lengths must contain at least one stage")

        normalised = tuple(int(length) for length in sequence_lengths)
        if any(length <= 0 for length in normalised):
            raise ValueError("sequence_lengths must be positive integers")
        if list(normalised) != sorted(normalised):
            raise ValueError("sequence_lengths must be non-decreasing")

        self._sequence_lengths: tuple[int, ...] = normalised
        self._max_stage_index = len(self._sequence_lengths) - 1

        if milestones is not None:
            milestones_tuple = tuple(int(value) for value in milestones)
            if len(milestones_tuple) != len(self._sequence_lengths) - 1:
                raise ValueError(
                    "milestones must contain len(sequence_lengths) - 1 entries"
                )
            if any(value <= 0 for value in milestones_tuple):
                raise ValueError("milestones must be positive integers")
            if list(milestones_tuple) != sorted(milestones_tuple):
                raise ValueError("milestones must be non-decreasing")
            self._milestones: tuple[int, ...] | None = milestones_tuple
        else:
            self._milestones = None

        self._stage_index = 0
        self._tokens_observed = 0

    @property
    def stage_index(self) -> int:
        return self._stage_index

    @property
    def current_length(self) -> int:
        return self._sequence_lengths[self._stage_index]

    @property
    def tokens_observed(self) -> int:
        return self._tokens_observed

    @property
    def milestones(self) -> tuple[int, ...] | None:
        return self._milestones

    @property
    def stages(self) -> tuple[CurriculumStage, ...]:
        entries: list[CurriculumStage] = []
        for index, length in enumerate(self._sequence_lengths):
            milestone = None
            if self._milestones and index > 0:
                milestone = self._milestones[index - 1]
            entries.append(CurriculumStage(index=index, sequence_length=length, milestone_tokens=milestone))
        return tuple(entries)

    def observe_tokens(self, tokens: int) -> int:
        if tokens < 0:
            raise ValueError("tokens must be non-negative")
        self._tokens_observed += tokens
        if self._milestones:
            self._recompute_stage_from_tokens()
        return self.current_length

    def advance(self) -> int:
        if self._stage_index < self._max_stage_index:
            self._stage_index += 1
            if self._milestones:
                target = self._milestones[self._stage_index - 1]
                if self._tokens_observed < target:
                    self._tokens_observed = target
        return self.current_length

    def reset(self) -> None:
        self._stage_index = 0
        self._tokens_observed = 0

    def state_dict(self) -> dict[str, int | tuple[int, ...] | None]:
        return {
            "stage_index": self._stage_index,
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

        if self._milestones:
            self._recompute_stage_from_tokens()
        else:
            if stage_index < 0 or stage_index > self._max_stage_index:
                raise ValueError("stage_index out of range")
            self._stage_index = stage_index

    def _recompute_stage_from_tokens(self) -> None:
        assert self._milestones is not None  # for type checkers
        stage = 0
        for milestone in self._milestones:
            if self._tokens_observed >= milestone:
                stage += 1
            else:
                break
        if stage > self._max_stage_index:
            stage = self._max_stage_index
        self._stage_index = stage

    def __repr__(self) -> str:  # pragma: no cover - diagnostic helper
        return (
            "SequenceLengthCurriculum(stage_index="
            f"{self._stage_index}, current_length={self.current_length}, tokens_observed="
            f"{self._tokens_observed})"
        )


__all__ = ["CurriculumStage", "SequenceLengthCurriculum"]
