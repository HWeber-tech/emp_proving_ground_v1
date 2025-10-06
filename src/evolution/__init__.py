"""
EMP Evolution Layer v1.1

The Evolution Layer orchestrates the genetic programming and evolution of
trading strategies. It manages populations, selection, variation, and
evaluation to drive continuous improvement and adaptation.

Architecture:
- engine/: Genetic engine and population management
- selection/: Selection algorithms (tournament, fitness proportionate)
- variation/: Crossover, mutation, and recombination operators
- evaluation/: Fitness evaluation and backtesting
- meta/: Meta-evolution for self-improving evolution
"""

# Legacy facade: re-export core evolution interfaces
from src.core.evolution.engine import EvolutionConfig, EvolutionEngine
from src.core.evolution.fitness import FitnessEvaluator
from src.core.evolution.operators import *  # noqa: F401,F403
from src.core.evolution.population import Population
from src.evolution.catalogue_telemetry import (
    EvolutionCatalogueEntrySnapshot,
    EvolutionCatalogueSnapshot,
    build_catalogue_snapshot,
)
from src.evolution.evaluation import (
    EVENT_SOURCE_RECORDED_REPLAY,
    EVENT_TYPE_RECORDED_REPLAY,
    build_recorded_replay_event,
    format_recorded_replay_markdown,
    RecordedEvaluationResult,
    RecordedSensoryEvaluator,
    RecordedSensorySnapshot,
    RecordedTrade,
    publish_recorded_replay_snapshot,
)
from src.evolution.evaluation.telemetry import (
    RecordedReplayTelemetrySnapshot,
    summarise_recorded_replay,
)
from src.evolution.feature_flags import EvolutionFeatureFlags
from src.evolution.lineage_telemetry import (
    EvolutionLineageSnapshot,
    build_lineage_snapshot,
)

_ = None  # Legacy facade: re-export core evolution interfaces

__version__ = "1.1.0"
__author__ = "EMP System"
__description__ = "Evolution Layer - Genetic Programming and Evolution (core-consolidated)"

__all__ = [
    "EvolutionEngine",
    "EvolutionConfig",
    "FitnessEvaluator",
    "Population",
    "EvolutionCatalogueEntrySnapshot",
    "EvolutionCatalogueSnapshot",
    "build_catalogue_snapshot",
    "EvolutionLineageSnapshot",
    "build_lineage_snapshot",
    "EvolutionFeatureFlags",
    "RecordedEvaluationResult",
    "RecordedReplayTelemetrySnapshot",
    "RecordedSensoryEvaluator",
    "RecordedSensorySnapshot",
    "RecordedTrade",
    "summarise_recorded_replay",
    "publish_recorded_replay_snapshot",
    "format_recorded_replay_markdown",
    "build_recorded_replay_event",
    "EVENT_TYPE_RECORDED_REPLAY",
    "EVENT_SOURCE_RECORDED_REPLAY",
]
