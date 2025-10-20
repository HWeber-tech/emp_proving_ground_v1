from __future__ import annotations

from .policy_reflection import PolicyReflectionArtifacts, PolicyReflectionBuilder
from .policy_router import (
    FastWeightExperiment,
    LinearAttentionRouter,
    PolicyDecision,
    PolicyRouter,
    PolicyTactic,
    RegimeState,
)
from .regime_fitness import RegimeFitnessTable
from .fast_weights import (
    FastWeightConstraints,
    FastWeightController,
    FastWeightMetrics,
    FastWeightResult,
    build_fast_weight_controller,
    parse_fast_weight_constraints,
)
from .novelty_archive import (
    NoveltyArchive,
    NoveltyProbe,
    compute_genotype_signature,
    compute_probe_vector,
)
from .operator_constraints import (
    OperatorContext,
    OperatorConstraint,
    OperatorConstraintSet,
    OperatorConstraintViolation,
    parse_operator_constraints,
)
from .evolution_manager import EvolutionManager, ManagedStrategyConfig, StrategyVariant
from .operators import (
    GenotypeOperatorResult,
    op_add_feature,
    op_drop_feature,
    op_swap_execution_topology,
    op_tighten_risk,
)
from .entropy_governor import EntropyGovernor, EntropyGovernorConfig
from .feature_toggles import (
    AdaptationFeatureToggles,
    EXPLORATION_FLAG,
    FAST_WEIGHTS_FLAG,
    LINEAR_ATTENTION_FLAG,
)
from .replay_harness import (
    StageDecision,
    StageThresholds,
    TacticEvaluationResult,
    TacticReplayHarness,
)
from .strategy_contracts import (
    StrategyExecutionTopology,
    StrategyFeature,
    StrategyGenotype,
    StrategyPhenotype,
    StrategyRiskTemplate,
    StrategyTunable,
)

__all__ = [
    "PolicyReflectionArtifacts",
    "PolicyReflectionBuilder",
    "FastWeightExperiment",
    "LinearAttentionRouter",
    "PolicyDecision",
    "PolicyRouter",
    "PolicyTactic",
    "RegimeState",
    "EntropyGovernor",
    "EntropyGovernorConfig",
    "RegimeFitnessTable",
    "FastWeightConstraints",
    "FastWeightController",
    "FastWeightMetrics",
    "FastWeightResult",
    "build_fast_weight_controller",
    "parse_fast_weight_constraints",
    "NoveltyArchive",
    "NoveltyProbe",
    "compute_genotype_signature",
    "compute_probe_vector",
    "OperatorContext",
    "OperatorConstraint",
    "OperatorConstraintSet",
    "OperatorConstraintViolation",
    "parse_operator_constraints",
    "EvolutionManager",
    "ManagedStrategyConfig",
    "StrategyVariant",
    "GenotypeOperatorResult",
    "op_add_feature",
    "op_drop_feature",
    "op_swap_execution_topology",
    "op_tighten_risk",
    "AdaptationFeatureToggles",
    "FAST_WEIGHTS_FLAG",
    "LINEAR_ATTENTION_FLAG",
    "EXPLORATION_FLAG",
    "StageDecision",
    "StageThresholds",
    "TacticEvaluationResult",
    "TacticReplayHarness",
    "StrategyFeature",
    "StrategyExecutionTopology",
    "StrategyRiskTemplate",
    "StrategyTunable",
    "StrategyGenotype",
    "StrategyPhenotype",
]
