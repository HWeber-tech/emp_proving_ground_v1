#!/usr/bin/env python3
"""
Canonical ecosystem shim (intelligence layer)

This module re-exports canonical ecosystem classes so legacy imports from
src.intelligence.specialized_predators continue to work while the single
source of truth lives under src.ecosystem.*.

Re-exported:
- NicheDetector -> src.ecosystem.evaluation.niche_detector.NicheDetector
- SpeciesManager -> src.ecosystem.species.species_manager.SpeciesManager
- CoordinationEngine -> src.ecosystem.coordination.coordination_engine.CoordinationEngine
- EcosystemOptimizer -> src.ecosystem.optimization.ecosystem_optimizer.EcosystemOptimizer
- SpecializedPredatorEvolution -> src.ecosystem.evolution.specialized_predator_evolution.SpecializedPredatorEvolution
"""

from __future__ import annotations

from src.ecosystem.evaluation.niche_detector import (
    NicheDetector as NicheDetector,
)
from src.ecosystem.species.species_manager import (
    SpeciesManager as SpeciesManager,
)
from src.ecosystem.coordination.coordination_engine import (
    CoordinationEngine as CoordinationEngine,
)
from src.ecosystem.optimization.ecosystem_optimizer import (
    EcosystemOptimizer as EcosystemOptimizer,
)
from src.ecosystem.evolution.specialized_predator_evolution import (
    SpecializedPredatorEvolution as SpecializedPredatorEvolution,
)

__all__ = [
    "NicheDetector",
    "SpeciesManager",
    "CoordinationEngine",
    "EcosystemOptimizer",
    "SpecializedPredatorEvolution",
]
