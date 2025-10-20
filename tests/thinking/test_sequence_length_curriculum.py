from __future__ import annotations

import pytest

from src.thinking.learning import SequenceLengthCurriculum, SequenceLengthStage


def test_curriculum_default_progression_manual_advance() -> None:
    curriculum = SequenceLengthCurriculum()

    assert curriculum.current_length == 4096
    assert curriculum.stage_index == 0

    curriculum.advance()
    assert curriculum.stage_index == 1
    assert curriculum.current_length == 8192

    curriculum.advance()
    assert curriculum.stage_index == 2
    assert curriculum.current_length == 16384

    curriculum.advance()
    assert curriculum.stage_index == 2
    assert curriculum.current_length == 16384


@pytest.mark.parametrize(
    ("chunks", "expected"),
    [([5, 5, 10, 10], 16384), ([12, 18, 0, 1], 16384)],
)
def test_curriculum_advances_with_milestones(
    chunks: list[int], expected: int
) -> None:
    curriculum = SequenceLengthCurriculum(milestones=(10, 30))
    total = 0
    for tokens in chunks:
        length = curriculum.observe_tokens(tokens)
        total += tokens

    assert curriculum.current_length == expected
    assert curriculum.tokens_observed == total
    assert curriculum.stage_index == 2


def test_curriculum_state_dict_roundtrip() -> None:
    curriculum = SequenceLengthCurriculum(milestones=(10, 30))
    curriculum.observe_tokens(15)

    state = curriculum.state_dict()

    restored = SequenceLengthCurriculum(milestones=(10, 30))
    restored.load_state_dict(state)

    assert restored.stage_index == curriculum.stage_index
    assert restored.current_length == curriculum.current_length
    assert restored.tokens_observed == curriculum.tokens_observed


def test_curriculum_rejects_invalid_tokens() -> None:
    curriculum = SequenceLengthCurriculum()
    with pytest.raises(ValueError):
        curriculum.observe_tokens(-1)


def test_curriculum_manual_advance_snaps_to_milestone() -> None:
    curriculum = SequenceLengthCurriculum(milestones=(10, 30))
    curriculum.advance()

    assert curriculum.stage_index == 1
    assert curriculum.tokens_observed == 10
    assert curriculum.current_length == 8192


def test_allocate_event_mix_tracks_ratio() -> None:
    curriculum = SequenceLengthCurriculum()
    rare_accumulated = 0
    main_accumulated = 0
    for _ in range(20):
        mix = curriculum.allocate_event_mix(1)
        rare_accumulated += mix["rare"]
        main_accumulated += mix["main"]

    assert rare_accumulated == 4
    assert main_accumulated == 16


def test_allocate_event_mix_resets_on_stage_transition() -> None:
    stages = (
        SequenceLengthStage(
            name="stage_a",
            sequence_length=512,
            max_steps=1,
            rare_event_ratio=0.55,
        ),
        SequenceLengthStage(
            name="stage_b",
            sequence_length=1024,
            rare_event_ratio=0.55,
        ),
    )
    curriculum = SequenceLengthCurriculum(stages=stages)
    curriculum.allocate_event_mix(1)

    curriculum.record_progress(force=True, reason="advance")

    mix = curriculum.allocate_event_mix(1)
    assert mix["rare"] == 0
    assert mix["main"] == 1


def test_sequence_length_stage_validates_rare_ratio() -> None:
    with pytest.raises(ValueError):
        SequenceLengthStage(name="bad", sequence_length=128, rare_event_ratio=1.5)
