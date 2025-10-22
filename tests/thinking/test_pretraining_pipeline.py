from __future__ import annotations

import math

import pytest

from src.thinking.learning import (
    PreTrainingBatch,
    PreTrainingPipeline,
    PreTrainingPipelineConfig,
    SequenceLengthCurriculum,
    SequenceLengthStage,
)
from src.thinking.learning.horizon_evaluation import HorizonObservation


def _build_pipeline() -> PreTrainingPipeline:
    stages = (
        SequenceLengthStage(name="bootstrap", sequence_length=4, rare_event_ratio=0.25),
        SequenceLengthStage(name="stability", sequence_length=6, rare_event_ratio=0.5),
    )
    curriculum = SequenceLengthCurriculum(stages=stages)
    config = PreTrainingPipelineConfig(
        curriculum=curriculum,
        burn_in_tokens=2,
        horizon_bins=5,
        freeze_fraction=0.5,
        lora_fraction=0.5,
        lora_rank_range=(4, 8),
        lora_alpha_multiplier=1.5,
        lora_dropout=0.1,
    )
    return PreTrainingPipeline(config)


def _sample_predictions() -> dict[str, object]:
    return {
        "delta": [0.1, -0.05, 0.0, 0.02],
        "quantiles": [0.05, 0.0, -0.02],
        "direction": 0.65,
        "big_move": 0.2,
        "next_event": {"up": 0.6, "down": 0.3, "flat": 0.1},
        "masked_depth": [0.5, 0.6],
        "queue_depletion": 0.35,
    }


def _sample_targets() -> dict[str, object]:
    return {
        "delta": [0.0, -0.1, 0.05, 0.03],
        "direction": 1.0,
        "big_move": 0.0,
        "next_event": "up",
        "masked_depth": [0.45, 0.55],
        "depth_mask": [1, 1],
        "queue_depletion": 0.0,
    }


def test_pretraining_pipeline_run_produces_summary() -> None:
    pipeline = _build_pipeline()

    observations = [
        HorizonObservation("ev1", "event", probability=0.6, outcome=1, gross_alpha_bps=5, fees_bps=1, weight=2),
        HorizonObservation("100ms", "time", probability=0.55, outcome=0, gross_alpha_bps=3, fees_bps=1, weight=1),
    ]
    layer_names = [f"layer_{idx}" for idx in range(6)]

    batch = PreTrainingBatch(
        sequence=list(range(12)),
        predictions=_sample_predictions(),
        targets=_sample_targets(),
        horizon_observations=observations,
        layer_names=layer_names,
    )

    result = pipeline.run(batch)

    assert result.stage_before == "bootstrap"
    assert result.stage_index_before == 0
    assert result.sequence_length == 4
    assert result.tokens_processed == 8
    assert len(result.chunk_summaries) == 2
    assert result.chunk_summaries[0].burn_in_tokens == 2
    assert result.event_mix["rare"] + result.event_mix["main"] == len(result.chunk_summaries)
    assert result.loss.total > 0.0
    assert math.isfinite(result.loss.component("huber"))
    assert result.horizon_report is not None
    assert result.horizon_report.overall.count == len(observations)
    assert result.lora_plan is not None
    assert result.lora_plan.total_layers == len(layer_names)
    assert result.curriculum_summary["tokens_observed"] == result.tokens_processed

    serialised = result.as_dict()
    assert serialised["loss"]["total"] == pytest.approx(result.loss.total)
    assert serialised["chunk_summaries"][0]["burn_in_tokens"] == 2


def test_pretraining_pipeline_force_advance_and_no_optional_inputs() -> None:
    pipeline = _build_pipeline()

    batch = PreTrainingBatch(
        sequence=list(range(8)),
        predictions=_sample_predictions(),
        targets=_sample_targets(),
    )

    result = pipeline.run(batch, metric=0.2, force_advance=True)

    assert result.stage_before == "bootstrap"
    assert result.stage_after == "stability"
    assert result.horizon_report is None
    assert result.lora_plan is None

    # Running a second batch should now use the new curriculum stage length.
    batch_two = PreTrainingBatch(
        sequence=list(range(18)),
        predictions=_sample_predictions(),
        targets=_sample_targets(),
    )

    result_two = pipeline.run(batch_two)

    assert result_two.stage_before == "stability"
    assert all(chunk.train_tokens == 6 for chunk in result_two.chunk_summaries)
    assert result_two.event_mix["rare"] == 1
    assert result_two.event_mix["main"] == len(result_two.chunk_summaries) - 1
