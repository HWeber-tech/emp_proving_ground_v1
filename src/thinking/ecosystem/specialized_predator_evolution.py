"""
Canonical ecosystem shim (thinking layer)

This module re-exports canonical ecosystem classes so legacy imports from
src.thinking.ecosystem.specialized_predator_evolution continue to work while
the single source of truth lives under src.ecosystem.*.

Re-exported:
- NicheDetector -> src.ecosystem.evaluation.niche_detector.NicheDetector
- SpeciesManager -> src.ecosystem.species.species_manager.SpeciesManager
- CoordinationEngine -> src.ecosystem.coordination.coordination_engine.CoordinationEngine
- EcosystemOptimizer -> src.ecosystem.optimization.ecosystem_optimizer.EcosystemOptimizer
- SpecializedPredatorEvolution -> src.ecosystem.evolution.specialized_predator_evolution.SpecializedPredatorEvolution
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.ecosystem.coordination.coordination_engine import CoordinationEngine as CoordinationEngine
from src.ecosystem.evaluation.niche_detector import (
    MarketNiche as MarketNiche,  # also re-export commonly used type
)
from src.ecosystem.evaluation.niche_detector import NicheDetector as NicheDetector
from src.ecosystem.evolution.specialized_predator_evolution import (
    SpecializedPredatorEvolution as SpecializedPredatorEvolution,
)
from src.ecosystem.optimization.ecosystem_optimizer import EcosystemOptimizer as EcosystemOptimizer

if TYPE_CHECKING:  # pragma: no cover
    from src.ecosystem.species.species_manager import SpeciesManager as SpeciesManager
else:  # runtime fallback for type-check-time only symbol

    class _SpeciesManagerRT:
        pass

    SpeciesManager = _SpeciesManagerRT

__all__: list[str] = [
    "NicheDetector",
    "MarketNiche",
    "SpeciesManager",
    "CoordinationEngine",
    "EcosystemOptimizer",
    "SpecializedPredatorEvolution",
]
