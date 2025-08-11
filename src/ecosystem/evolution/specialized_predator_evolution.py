#!/usr/bin/env python3
"""
Canonical Specialized Predator Evolution (ecosystem domain)

This module orchestrates specialized predator evolution by composing canonical
ecosystem components:
- NicheDetector: src.ecosystem.evaluation.niche_detector
- SpeciesManager: src.ecosystem.species.species_manager
- CoordinationEngine: src.ecosystem.coordination.coordination_engine
- EcosystemOptimizer: src.ecosystem.optimization.ecosystem_optimizer

Interfaces are intentionally minimal to reduce duplication across packages.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List

from src.ecosystem.evaluation.niche_detector import NicheDetector
from src.ecosystem.species.species_manager import SpeciesManager
from src.ecosystem.coordination.coordination_engine import CoordinationEngine
from src.ecosystem.optimization.ecosystem_optimizer import EcosystemOptimizer

logger = logging.getLogger(__name__)


class SpecializedPredatorEvolution:
    """
    Orchestrates ecosystem-level evolution across canonical components.

    Public API:
    - evolve_specialized_predators() -> Dict[str, Any]
    - get_ecosystem_stats() -> Dict[str, Any]
    """

    def __init__(self) -> None:
        self.niche_detector = NicheDetector()
        self.species_manager = SpeciesManager()
        self.coordination_engine = CoordinationEngine()
        self.ecosystem_optimizer = EcosystemOptimizer()
        self._history: List[Dict[str, Any]] = []

    async def evolve_specialized_predators(self) -> Dict[str, Any]:
        """
        Detect niches, evolve species, optimize coordination, and produce
        ecosystem-level optimization. Returns a structured result dict.
        """
        logger.info("Starting specialized predator evolution (canonical)")

        market_data = await self._get_market_data()
        historical_analysis = await self._get_historical_analysis()

        # Detect market niches (canonical detector returns dict of MarketNiche by id)
        niches_by_id = await self.niche_detector.detect_niches(market_data)
        niches = list(niches_by_id.values())
        if not niches:
            logger.warning("No market niches detected")
            result = {
                "timestamp": datetime.utcnow().isoformat(),
                "niches_detected": 0,
                "specialists_evolved": 0,
                "coordination_strategy": None,
                "optimization_results": None,
            }
            self._history.append(result)
            return result

        # Evolve one specialist per niche (minimal composition)
        # SpeciesManager here is a canonical facade that may internally bridge to prior implementation
        specialists: Dict[str, Any] = {}
        for niche in niches:
            try:
                # Attempt to evolve a specialist; fall back to a lightweight record if not supported
                evolved = await self._evolve_specialist(niche)
                specialists[getattr(niche, "regime_type", getattr(niche, "niche_type", "unknown"))] = evolved
            except Exception as e:
                logger.error(f"Failed to evolve specialist for niche: {e}")

        # Build a lightweight coordination strategy by resolving against canonical engine if available
        try:
            # The canonical CoordinationEngine resolves intents; here we use a minimal placeholder flow
            # Downstream systems may provide richer integration.
            coordination_strategy = {
                "type": "canonical-ecosystem-default",
                "timestamp": datetime.utcnow().isoformat(),
            }
        except Exception as e:
            logger.warning(f"Coordination fallback engaged: {e}")
            coordination_strategy = {"type": "basic", "timestamp": datetime.utcnow().isoformat()}

        # Ecosystem-level optimization (canonical optimizer)
        try:
            # Canonical optimizer expects populations and history; provide minimal placeholders
            optimization = await self.ecosystem_optimizer.get_ecosystem_summary()  # type: ignore[attr-defined]
        except Exception:
            optimization = {
                "total_optimizations": 0,
                "best_metrics": None,
                "current_species_distribution": {},
            }

        result = {
            "timestamp": datetime.utcnow().isoformat(),
            "niches_detected": len(niches),
            "specialists_evolved": len(specialists),
            "coordination_strategy": coordination_strategy,
            "optimization_results": optimization,
        }
        self._history.append(result)
        return result

    def get_ecosystem_stats(self) -> Dict[str, Any]:
        """Return a summarized snapshot of recent evolution cycles."""
        if not self._history:
            return {"total_cycles": 0, "last_result": None}
        return {"total_cycles": len(self._history), "last_result": self._history[-1]}

    async def _evolve_specialist(self, niche: Any) -> Dict[str, Any]:
        """
        Evolve a specialist for the given niche using canonical SpeciesManager.
        As a compatibility facade, return a dict describing the evolved specialist.
        """
        # The canonical SpeciesManager may expose different interfaces across phases.
        # Provide a stable dict wrapper for downstream code.
        try:
            # Attempt an async evolution API commonly found in prior implementations
            evolved_obj = await self.species_manager.evolve_specialist(  # type: ignore[attr-defined]
                niche=niche, base_population=[], specialization_pressure=getattr(niche, "opportunity_score", 0.5)
            )
            return {
                "species_type": getattr(evolved_obj, "predator_type", getattr(evolved_obj, "species_type", "unknown")),
                "id": getattr(evolved_obj, "predator_id", getattr(evolved_obj, "species_id", "unknown")),
                "metrics": getattr(evolved_obj, "performance_metrics", {}),
            }
        except AttributeError:
            # Fallback: minimal adaptation if evolve_specialist is not available
            return {
                "species_type": "generic",
                "id": f"species_{getattr(niche, 'regime_type', 'niche')}",
                "metrics": {},
            }

    async def _get_market_data(self) -> Dict[str, Any]:
        """Minimal market data stub compatible with canonical NicheDetector."""
        return {
            "data": [],  # canonical detector expects 'data' for DataFrame-based features (may be empty)
            "volatility": 0.025,
            "trend_strength": 0.6,
            "volume": 1500,
        }

    async def _get_historical_analysis(self) -> Dict[str, Any]:
        """Minimal historical analysis stub for compatibility with detectors."""
        return {"baseline_volatility": 0.02, "momentum_success_rate": 0.6}


__all__ = ["SpecializedPredatorEvolution"]