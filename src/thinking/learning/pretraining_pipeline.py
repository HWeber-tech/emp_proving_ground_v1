"""Pre-training pipeline orchestration utilities (E.1.x roadmap slice).

The LOBSTER pre-training roadmap outlines a staged curriculum that gradually
extends context length, mixes in rare events, and applies domain-specific
regularisation.  This module ties together the existing building blocks
(curriculum scheduling, multi-task losses, LoRA planning, and horizon
calibration) into a light-weight pipeline that research harnesses can use
during experimentation.

The pipeline deliberately focuses on deterministic, dependency-free logic so it
can run inside CI and unit tests.  Downstream training loops remain responsible
for GPU execution and gradient computation; the pipeline simply orchestrates
bookkeeping and metric computation around each batch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import copy
from typing import Mapping, MutableMapping, Sequence

from .horizon_evaluation import (
    HorizonEvaluationReport,
    HorizonObservation,
    evaluate_predictions_by_horizon,
)
from .lora_freeze import LoRAFreezePlan, plan_lora_freeze
from .multitask_losses import MultiTaskLossResult, compute_multitask_losses
from .sequence_curriculum import (
    SequenceLengthCurriculum,
    SequenceLengthStage,
    build_curriculum,
)
from .trainer_chunker import TrainerChunker


@dataclass(frozen=True)
class PreTrainingChunkSummary:
    """Summary describing a single TBPTT chunk processed by the pipeline."""

    index: int
    train_tokens: int
    burn_in_tokens: int

    @property
    def total_tokens(self) -> int:
        return self.train_tokens + self.burn_in_tokens

    def as_dict(self) -> dict[str, int]:
        return {
            "index": self.index,
            "train_tokens": self.train_tokens,
            "burn_in_tokens": self.burn_in_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass(slots=True)
class PreTrainingPipelineConfig:
    """Configuration bundle for :class:`PreTrainingPipeline`."""

    curriculum: SequenceLengthCurriculum = field(default_factory=build_curriculum)
    burn_in_tokens: int = 512
    loss_weights: Mapping[str, float] | None = None
    quantile_levels: tuple[float, ...] = (0.25, 0.5, 0.75)
    huber_delta: float = 1.0
    horizon_bins: int = 10
    freeze_fraction: float = 0.7
    freeze_range: tuple[float, float] = (0.6, 0.8)
    lora_fraction: float = 0.35
    lora_range: tuple[float, float] = (0.3, 0.4)
    lora_rank_range: tuple[int, int] = (8, 16)
    lora_alpha_multiplier: float = 2.0
    lora_dropout: float = 0.05

    def __post_init__(self) -> None:
        if self.burn_in_tokens < 0:
            raise ValueError("burn_in_tokens must be non-negative")
        if self.horizon_bins <= 0:
            raise ValueError("horizon_bins must be positive")


@dataclass(frozen=True)
class PreTrainingBatch:
    """Input payload consumed by :class:`PreTrainingPipeline.run`."""

    sequence: Sequence[object]
    predictions: Mapping[str, object]
    targets: Mapping[str, object]
    horizon_observations: Sequence[HorizonObservation | Mapping[str, object]] | None = None
    layer_names: Sequence[object] = ()
    initial_state: object | None = None


@dataclass(frozen=True)
class PreTrainingRunResult:
    """Structured summary returned by :meth:`PreTrainingPipeline.run`."""

    stage_before: str
    stage_after: str
    stage_index_before: int
    stage_index_after: int
    sequence_length: int
    tokens_processed: int
    chunk_summaries: tuple[PreTrainingChunkSummary, ...]
    event_mix: Mapping[str, int]
    loss: MultiTaskLossResult
    horizon_report: HorizonEvaluationReport | None
    lora_plan: LoRAFreezePlan | None
    curriculum_summary: Mapping[str, object]

    def as_dict(self) -> dict[str, object]:
        summary: MutableMapping[str, object] = {
            "stage_before": self.stage_before,
            "stage_after": self.stage_after,
            "stage_index_before": self.stage_index_before,
            "stage_index_after": self.stage_index_after,
            "sequence_length": self.sequence_length,
            "tokens_processed": self.tokens_processed,
            "chunk_summaries": [chunk.as_dict() for chunk in self.chunk_summaries],
            "event_mix": dict(self.event_mix),
            "loss": self.loss.as_dict(),
            "curriculum_summary": dict(self.curriculum_summary),
        }
        if self.horizon_report is not None:
            summary["horizon_report"] = self.horizon_report.as_dict()
        if self.lora_plan is not None:
            summary["lora_plan"] = self.lora_plan.as_dict()
        return dict(summary)


class PreTrainingPipeline:
    """Coordinate the curriculum, loss computation, and evaluation helpers."""

    def __init__(self, config: PreTrainingPipelineConfig | None = None) -> None:
        self._config = config or PreTrainingPipelineConfig()

    @property
    def config(self) -> PreTrainingPipelineConfig:
        return self._config

    @property
    def curriculum(self) -> SequenceLengthCurriculum:
        return self._config.curriculum

    def current_stage(self) -> SequenceLengthStage:
        return self.curriculum.current_stage

    def run(
        self,
        batch: PreTrainingBatch,
        *,
        metric: float | None = None,
        force_advance: bool = False,
    ) -> PreTrainingRunResult:
        stage_before = self.curriculum.current_stage
        stage_index_before = self.curriculum.stage_index

        chunker = TrainerChunker(
            burn_in=self._config.burn_in_tokens,
            train_len=stage_before.sequence_length,
        )
        chunk_summaries = tuple(
            PreTrainingChunkSummary(
                index=chunk.index,
                train_tokens=chunk.train_length,
                burn_in_tokens=chunk.burn_in_length,
            )
            for chunk in chunker.iter_chunks(
                batch.sequence,
                initial_state=batch.initial_state,
            )
        )
        tokens_processed = sum(chunk.train_tokens for chunk in chunk_summaries)

        event_mix = self.curriculum.allocate_event_mix(len(chunk_summaries))

        if tokens_processed or force_advance or metric is not None:
            self.curriculum.record_progress(
                steps=tokens_processed,
                metric=metric,
                force=force_advance,
            )
        stage_after = self.curriculum.current_stage
        stage_index_after = self.curriculum.stage_index

        observations = self._normalise_observations(batch.horizon_observations)
        horizon_report = None
        if observations:
            horizon_report = evaluate_predictions_by_horizon(
                observations,
                num_bins=self._config.horizon_bins,
            )

        loss = compute_multitask_losses(
            batch.predictions,
            batch.targets,
            weights=self._config.loss_weights,
            quantile_levels=self._config.quantile_levels,
            huber_delta=self._config.huber_delta,
        )

        lora_plan = None
        if batch.layer_names:
            lora_plan = plan_lora_freeze(
                batch.layer_names,
                freeze_fraction=self._config.freeze_fraction,
                freeze_range=self._config.freeze_range,
                lora_fraction=self._config.lora_fraction,
                lora_range=self._config.lora_range,
                rank_range=self._config.lora_rank_range,
                lora_alpha_multiplier=self._config.lora_alpha_multiplier,
                lora_dropout=self._config.lora_dropout,
            )

        curriculum_snapshot = copy.deepcopy(self.curriculum.summary())

        return PreTrainingRunResult(
            stage_before=stage_before.name,
            stage_after=stage_after.name,
            stage_index_before=stage_index_before,
            stage_index_after=stage_index_after,
            sequence_length=stage_before.sequence_length,
            tokens_processed=tokens_processed,
            chunk_summaries=chunk_summaries,
            event_mix=dict(event_mix),
            loss=loss,
            horizon_report=horizon_report,
            lora_plan=lora_plan,
            curriculum_summary=curriculum_snapshot,
        )

    @staticmethod
    def _normalise_observations(
        observations: Sequence[HorizonObservation | Mapping[str, object]] | None,
    ) -> tuple[HorizonObservation, ...]:
        if not observations:
            return tuple()
        normalised: list[HorizonObservation] = []
        for entry in observations:
            if isinstance(entry, HorizonObservation):
                normalised.append(entry)
            elif isinstance(entry, Mapping):
                normalised.append(HorizonObservation(**entry))
            else:
                raise TypeError(
                    "horizon_observations must contain HorizonObservation instances "
                    "or mapping objects"
                )
        return tuple(normalised)


__all__ = [
    "PreTrainingBatch",
    "PreTrainingChunkSummary",
    "PreTrainingPipeline",
    "PreTrainingPipelineConfig",
    "PreTrainingRunResult",
]
