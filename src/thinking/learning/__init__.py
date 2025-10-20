from __future__ import annotations

from .meta_cognition_engine import MetaCognitionEngine
from .multitask_losses import MultiTaskLossResult, compute_multitask_losses
from .horizon_evaluation import (
    HorizonEvaluationReport,
    HorizonMetrics,
    HorizonObservation,
    evaluate_predictions_by_horizon,
)
from .lora_freeze import LoRAFreezePlan, LoRALayerConfig, plan_lora_freeze
from .l2sp_rehearsal import L2SPRegularizer, EquityRehearsalPlan, plan_equity_rehearsal
from .trainer_chunker import SequenceChunk, TrainerChunker
from .sequence_curriculum import (
    CurriculumTransition,
    SequenceLengthCurriculum,
    SequenceLengthStage,
    DEFAULT_SEQUENCE_LENGTH_STAGES,
    build_curriculum,
)

__all__ = [
    "MetaCognitionEngine",
    "MultiTaskLossResult",
    "compute_multitask_losses",
    "HorizonEvaluationReport",
    "HorizonMetrics",
    "HorizonObservation",
    "evaluate_predictions_by_horizon",
    "LoRAFreezePlan",
    "LoRALayerConfig",
    "plan_lora_freeze",
    "L2SPRegularizer",
    "EquityRehearsalPlan",
    "plan_equity_rehearsal",
    "SequenceChunk",
    "TrainerChunker",
    "SequenceLengthStage",
    "SequenceLengthCurriculum",
    "CurriculumTransition",
    "DEFAULT_SEQUENCE_LENGTH_STAGES",
    "build_curriculum",
]
