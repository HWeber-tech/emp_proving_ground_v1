#!/usr/bin/env python3
"""
Advanced Intelligence Facade (Thin Lazy Aggregator)
===================================================

This package-level module provides a lightweight public API for intelligence
components without importing heavy dependencies at import time. All public
symbols are lazily resolved on first attribute access via PEP 562 (__getattr__).

Design goals:
- Preserve legacy public paths (e.g., `src.intelligence.RedTeamAI`)
- Avoid import-time side effects (no logging/I/O/thread starts)
- Keep heavy imports localized inside functions/methods
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Explicit public exports
__all__ = [
    "SentientAdaptationEngine",
    "PredictiveMarketModeler",
    "MarketGAN",
    "AdversarialTrainer",
    "RedTeamAI",
    "SpecializedPredatorEvolution",
    "CompetitiveIntelligenceSystem",
    "Phase3IntelligenceOrchestrator",
]

# Lazy mapping of public names to their canonical modules. This keeps import
# time light while ensuring callers resolve the single source of truth instead
# of legacy shim modules.
_LAZY_EXPORTS: dict[str, str] = {
    "SentientAdaptationEngine": (
        "src.sentient.adaptation.sentient_adaptation_engine:SentientAdaptationEngine"
    ),
    "PredictiveMarketModeler": (
        "src.thinking.prediction.predictive_market_modeler:PredictiveMarketModeler"
    ),
    "MarketGAN": "src.thinking.adversarial.market_gan:MarketGAN",
    "AdversarialTrainer": "src.thinking.adversarial.adversarial_trainer:AdversarialTrainer",
    "RedTeamAI": "src.thinking.adversarial.red_team_ai:RedTeamAI",
    "SpecializedPredatorEvolution": "src.ecosystem.evolution.specialized_predator_evolution:SpecializedPredatorEvolution",
    "CompetitiveIntelligenceSystem": (
        "src.thinking.competitive.competitive_intelligence_system:CompetitiveIntelligenceSystem"
    ),
}


class _PortfolioEvolutionFallback:
    """Lightweight fallback when the portfolio evolution engine is unavailable."""

    def __init__(self) -> None:
        self._history: list[dict[str, Any]] = []

    async def initialize(self) -> bool:
        return True

    async def stop(self) -> bool:
        return True

    async def evolve_portfolio(
        self, strategies: list[Any], market_data: Mapping[str, Any]
    ) -> dict[str, Any]:
        snapshot = {
            "status": "unavailable",
            "strategies": [getattr(s, "strategy_id", repr(s)) for s in strategies],
            "context_keys": sorted(market_data.keys()),
            "reason": "portfolio evolution engine fallback",
        }
        self._history.append(snapshot)
        return snapshot

    def get_evolution_stats(self) -> dict[str, Any]:
        return {
            "available": False,
            "total_cycles": len(self._history),
            "last_result": self._history[-1] if self._history else None,
        }

    def get_status(self) -> dict[str, Any]:
        return {
            "available": False,
            "reason": "Portfolio evolution engine fallback in use",
        }


def __getattr__(name: str) -> Any:
    # Lazy import to reduce import-time cost; preserves legacy public path.
    target = _LAZY_EXPORTS.get(name)
    if target:
        import importlib

        mod_path, attr = target.split(":")
        try:
            mod = importlib.import_module(mod_path)
        except ImportError as exc:
            if name == "PortfolioEvolutionEngine":
                logger.warning(
                    "Falling back to stub portfolio evolution engine: %s", exc
                )
                globals()[name] = _PortfolioEvolutionFallback
                return _PortfolioEvolutionFallback
            raise

        try:
            value = getattr(mod, attr)
        except AttributeError as exc:
            if name == "PortfolioEvolutionEngine":
                logger.warning(
                    "Portfolio evolution engine missing in %s: %s; using fallback",
                    mod_path,
                    exc,
                )
                value = _PortfolioEvolutionFallback
            else:
                raise
        globals()[name] = value
        return value
    if name == "PortfolioEvolutionEngine":
        logger.warning("Portfolio evolution engine unavailable; using fallback")
        globals()[name] = _PortfolioEvolutionFallback
        return _PortfolioEvolutionFallback
    raise AttributeError(name)


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + __all__)


class Phase3IntelligenceOrchestrator:
    """Main orchestrator for intelligence systems.

    Heavy imports are localized in __init__/initialize to avoid import-time cost.
    """

    def __init__(self) -> None:
        # Defer imports until initialize() to keep import-time light
        self.sentient_engine = None
        self.predictive_modeler = None
        self.adversarial_trainer = None
        self.red_team = None
        self.specialized_evolution = None
        self.competitive_intelligence = None
        # Portfolio evolution surface was removed during dead-code cleanup; keep
        # attribute stub so callers referencing it get a clear None.
        self.portfolio_evolution: Any | None = None

    async def initialize_phase3(self) -> None:
        # Localized imports to avoid import-time cost
        from datetime import timedelta  # noqa: F401  (used by clients)

        # Resolve facades lazily from this package to preserve public path
        SentientAdaptationEngine = __getattr__("SentientAdaptationEngine")
        PredictiveMarketModeler = __getattr__("PredictiveMarketModeler")
        MarketGAN = __getattr__("MarketGAN")
        AdversarialTrainer = __getattr__("AdversarialTrainer")
        RedTeamAI = __getattr__("RedTeamAI")
        SpecializedPredatorEvolution = __getattr__("SpecializedPredatorEvolution")
        CompetitiveIntelligenceSystem = __getattr__("CompetitiveIntelligenceSystem")

        # Instantiate components
        self.sentient_engine = SentientAdaptationEngine()
        # Predictive modeling surface may include async init; keep consistent
        self.predictive_modeler = PredictiveMarketModeler()
        self.adversarial_trainer = MarketGAN()
        self.red_team = RedTeamAI()
        self.specialized_evolution = SpecializedPredatorEvolution()
        self.portfolio_evolution = None
        self.competitive_intelligence = CompetitiveIntelligenceSystem()

        # If any components expose initialize coroutines, call them defensively
        import inspect

        for comp in (
            self.sentient_engine,
            self.predictive_modeler,
            self.adversarial_trainer,
            self.red_team,
            self.specialized_evolution,
            self.competitive_intelligence,
        ):
            init = getattr(comp, "initialize", None)
            if callable(init):
                res = init()
                if inspect.isawaitable(res):
                    await res

    async def run_intelligence_cycle(
        self, market_data: dict[str, Any], current_strategies: list[Any]
    ) -> dict[str, Any]:
        """Run complete intelligence cycle."""
        from datetime import datetime, timedelta

        assert self.sentient_engine is not None
        assert self.predictive_modeler is not None
        assert self.adversarial_trainer is not None
        assert self.red_team is not None
        assert self.specialized_evolution is not None
        assert self.competitive_intelligence is not None

        results: dict[str, Any] = {
            "timestamp": datetime.utcnow(),
            "sentient_adaptations": [],
            "predictions": [],
            "adversarial_results": [],
            "red_team_findings": [],
            "specialized_predators": [],
            "portfolio_evolution": None,
            "competitive_understanding": None,
        }

        # 1. Sentient adaptation
        for strategy in current_strategies:
            adaptation = await self.sentient_engine.adapt_in_real_time(market_data, strategy, {})
            results["sentient_adaptations"].append(adaptation)

        # 2. Predictive modeling
        predictions = await self.predictive_modeler.predict_market_scenarios(
            market_data, timedelta(hours=24)
        )
        results["predictions"] = predictions

        # 3. Adversarial training
        improved_strategies = await self.adversarial_trainer.train_adversarial_strategies(
            current_strategies
        )
        results["adversarial_results"] = improved_strategies

        # 4. Red team testing
        red_team_findings = []
        for strategy in improved_strategies:
            findings = await self.red_team.attack_strategy(strategy)
            red_team_findings.append(findings)
        results["red_team_findings"] = red_team_findings

        # 5. Specialized predator evolution
        specialized_predators = await self.specialized_evolution.evolve_specialized_predators()
        results["specialized_predators"] = specialized_predators

        # 6. Portfolio evolution
        results["portfolio_evolution"] = {
            "status": "removed",
            "detail": "Portfolio evolution surface retired during dead-code cleanup",
        }

        # 7. Competitive understanding
        competitive_report = await self.competitive_intelligence.identify_competitors(
            market_data
        )
        results["competitive_understanding"] = competitive_report

        return results

    async def get_phase3_status(self) -> dict[str, Any]:
        """Get status of all Phase 3 systems."""
        assert self.sentient_engine is not None
        assert self.predictive_modeler is not None
        assert self.adversarial_trainer is not None
        assert self.red_team is not None
        assert self.specialized_evolution is not None
        assert self.competitive_intelligence is not None

        competitive_stats = await self.competitive_intelligence.get_understanding_stats()

        return {
            "sentient_engine": self.sentient_engine.get_status(),
            "predictive_modeler": self.predictive_modeler.get_status(),
            "adversarial_trainer": self.adversarial_trainer.get_status(),
            "red_team": self.red_team.get_status(),
            "specialized_evolution": self.specialized_evolution.get_status(),
            "portfolio_evolution": {
                "status": "removed",
                "detail": "Legacy portfolio evolution facade retired",
            },
            "competitive_understanding": competitive_stats,
        }


# Global instance (lazy)
_orchestrator: Optional[Phase3IntelligenceOrchestrator] = None


async def get_phase3_orchestrator() -> Phase3IntelligenceOrchestrator:
    """Get or create global Phase 3 orchestrator (lazy)."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = Phase3IntelligenceOrchestrator()
        await _orchestrator.initialize_phase3()
    return _orchestrator
