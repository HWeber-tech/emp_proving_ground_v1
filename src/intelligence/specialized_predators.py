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

import importlib
from typing import Any

from src.ecosystem.coordination.coordination_engine import CoordinationEngine as CoordinationEngine
from src.ecosystem.evaluation.niche_detector import NicheDetector as NicheDetector


def _eco_mod(mod: str) -> Any:
    return importlib.import_module(f"src.ecosystem.{mod}")


def _species_manager() -> Any:
    return getattr(_eco_mod("species.species_manager"), "SpeciesManager")


def _predator_evolution() -> Any:
    return getattr(
        _eco_mod("evolution.specialized_predator_evolution"), "SpecializedPredatorEvolution"
    )


def _ecosystem_optimizer() -> Any:
    return getattr(_eco_mod("optimization.ecosystem_optimizer"), "EcosystemOptimizer")


SpeciesManager = _species_manager()
EcosystemOptimizer = _ecosystem_optimizer()
SpecializedPredatorEvolution = _predator_evolution()

__all__ = [
    "NicheDetector",
    "SpeciesManager",
    "CoordinationEngine",
    "EcosystemOptimizer",
    "SpecializedPredatorEvolution",
]
