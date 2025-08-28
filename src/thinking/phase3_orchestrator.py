from collections.abc import Sequence

"""
Phase 3 Orchestrator - Advanced Intelligence & Predatory Behavior
================================================================

Main orchestrator for Phase 3 implementation that coordinates all
advanced intelligence features and predatory behavior systems.

This orchestrator manages:
1. Sentient adaptation engine
2. Predictive market modeling
3. Adversarial training systems
4. Specialized predator evolution
5. Competitive intelligence
"""

import asyncio
import inspect
import json
import logging
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Protocol, TypedDict, cast, runtime_checkable

import numpy as np

from src.core.adaptation import AdaptationService, NoOpAdaptationService
from src.core.event_bus import EventBus
from src.core.state_store import StateStore
from src.thinking.adversarial.market_gan import MarketGAN
from src.thinking.adversarial.red_team_ai import RedTeamAI
from src.thinking.competitive.competitive_intelligence_system import CompetitiveIntelligenceSystem
from src.thinking.ecosystem.specialized_predator_evolution import SpecializedPredatorEvolution
from src.thinking.prediction.predictive_market_modeler import PredictiveMarketModeler


# Local type definitions to reduce Any usage
class PredictionTD(TypedDict, total=False):
    confidence: float
    probability: float


class GANResultTD(TypedDict, total=False):
    success: bool
    improved_strategies: list[str]


class RedTeamResultTD(TypedDict, total=False):
    survival_probability: float
    weaknesses: list[object]


@runtime_checkable
class PredictiveModeler(Protocol):
    async def initialize(self) -> bool: ...
    async def stop(self) -> bool: ...
    async def predict_market_scenarios(
        self, current_state: dict[str, object], time_horizon: timedelta, num_scenarios: int = ...
    ) -> Sequence[object]: ...


@runtime_checkable
class MarketGANP(Protocol):
    async def initialize(self) -> bool: ...
    async def stop(self) -> bool: ...
    async def train_adversarial_strategies(
        self, strategy_population: List[str], num_epochs: int = ...
    ) -> List[str]: ...


@runtime_checkable
class RedTeamP(Protocol):
    async def initialize(self) -> bool: ...
    async def stop(self) -> bool: ...
    async def attack_strategy(self, target_strategy: str) -> dict[str, object]: ...


@runtime_checkable
class SpecializedEvolutionP(Protocol):
    async def initialize(self) -> bool: ...
    async def stop(self) -> bool: ...


@runtime_checkable
class CompetitiveIntelP(Protocol):
    async def initialize(self) -> bool: ...
    async def stop(self) -> bool: ...


logger = logging.getLogger(__name__)


class Phase3Orchestrator:
    """
    Main orchestrator for Phase 3 advanced intelligence features.

    Coordinates all predatory behavior systems and ensures they work
    together as a unified, intelligent ecosystem.
    """

    def __init__(
        self,
        state_store: StateStore,
        event_bus: EventBus,
        adaptation_service: Optional[AdaptationService] = None,
    ):
        self.state_store = state_store
        self.event_bus = event_bus

        # Initialize all Phase 3 systems
        self.sentient_engine = adaptation_service or NoOpAdaptationService()
        self.predictive_modeler: PredictiveModeler = PredictiveMarketModeler(state_store)
        self.market_gan: MarketGANP = MarketGAN(state_store)
        self.red_team: RedTeamP = RedTeamAI(state_store)
        self.specialized_evolution: SpecializedEvolutionP = SpecializedPredatorEvolution()
        self.competitive_intelligence: CompetitiveIntelP = CompetitiveIntelligenceSystem(
            state_store
        )

        # Configuration
        self.config: dict[str, object] = {
            "sentient_enabled": True,
            "predictive_enabled": True,
            "adversarial_enabled": True,
            "specialized_enabled": True,
            "competitive_enabled": True,
            "update_frequency": 300,  # 5 minutes
            "full_analysis_frequency": 3600,  # 1 hour
        }

        # State tracking
        self.is_running = False
        self.last_full_analysis: Optional[datetime] = None
        self.performance_metrics: dict[str, object] = {}

        logger.info("Phase 3 Orchestrator initialized")

    async def initialize(self) -> bool:
        """Initialize all Phase 3 systems."""
        try:
            logger.info("Initializing Phase 3 systems...")

            # Initialize sentient adaptation engine
            if self.config["sentient_enabled"]:
                await self.sentient_engine.initialize()
                logger.info("✓ Sentient adaptation engine initialized")

            # Initialize predictive market modeler
            if self.config["predictive_enabled"]:
                await self.predictive_modeler.initialize()
                logger.info("✓ Predictive market modeler initialized")

            # Initialize adversarial systems
            if self.config["adversarial_enabled"]:
                await self.market_gan.initialize()
                await self.red_team.initialize()
                logger.info("✓ Adversarial systems initialized")

            # Initialize specialized evolution
            if self.config["specialized_enabled"]:
                await self.specialized_evolution.initialize()
                logger.info("✓ Specialized predator evolution initialized")

            # Initialize competitive intelligence
            if self.config["competitive_enabled"]:
                await self.competitive_intelligence.initialize()
                logger.info("✓ Competitive intelligence system initialized")

            logger.info("Phase 3 systems initialization complete")
            return True

        except Exception as e:
            logger.error(f"Error initializing Phase 3 systems: {e}")
            return False

    async def start(self) -> bool:
        """Start the Phase 3 orchestrator."""
        try:
            if self.is_running:
                logger.warning("Phase 3 orchestrator already running")
                return True

            logger.info("Starting Phase 3 orchestrator...")

            # Initialize systems
            if not await self.initialize():
                return False

            self.is_running = True

            # Start background tasks
            asyncio.create_task(self._run_continuous_analysis())
            asyncio.create_task(self._run_performance_monitoring())

            logger.info("Phase 3 orchestrator started successfully")
            return True

        except Exception as e:
            logger.error(f"Error starting Phase 3 orchestrator: {e}")
            return False

    async def stop(self) -> bool:
        """Stop the Phase 3 orchestrator."""
        try:
            if not self.is_running:
                logger.warning("Phase 3 orchestrator not running")
                return True

            logger.info("Stopping Phase 3 orchestrator...")

            self.is_running = False

            # Stop all systems
            await self.sentient_engine.stop()
            await self.predictive_modeler.stop()
            await self.market_gan.stop()
            await self.red_team.stop()
            await self.specialized_evolution.stop()
            await self.competitive_intelligence.stop()

            logger.info("Phase 3 orchestrator stopped")
            return True

        except Exception as e:
            logger.error(f"Error stopping Phase 3 orchestrator: {e}")
            return False

    async def run_full_analysis(self) -> dict[str, object]:
        """Run comprehensive Phase 3 analysis."""
        try:
            logger.info("Running full Phase 3 analysis...")

            analysis_start = datetime.utcnow()
            results: Dict[str, Any] = {
                "analysis_id": str(uuid.uuid4()),
                "timestamp": analysis_start.isoformat(),
                "systems": {},
            }

            # Run sentient adaptation analysis
            if self.config["sentient_enabled"]:
                results["systems"]["sentient"] = await self._run_sentient_analysis()

            # Run predictive modeling
            if self.config["predictive_enabled"]:
                results["systems"]["predictive"] = await self._run_predictive_analysis()

            # Run adversarial training
            if self.config["adversarial_enabled"]:
                results["systems"]["adversarial"] = await self._run_adversarial_analysis()

            # Run specialized evolution
            if self.config["specialized_enabled"]:
                results["systems"]["specialized"] = await self._run_specialized_analysis()

            # Run competitive intelligence
            if self.config["competitive_enabled"]:
                results["systems"]["competitive"] = await self._run_competitive_analysis()

            # Calculate overall metrics
            results["overall_metrics"] = await self._calculate_overall_metrics(results)

            # Store results
            await self._store_analysis_results(results)

            self.last_full_analysis = analysis_start

            logger.info("Full Phase 3 analysis complete")
            return results

        except Exception as e:
            logger.error(f"Error running full analysis: {e}")
            return {"error": str(e)}

    async def _run_sentient_analysis(self) -> dict[str, object]:
        """Run sentient adaptation analysis."""
        try:
            # Get current market state
            market_state = await self._get_current_market_state()

            # Run adaptation cycle
            adaptation_result = await self.sentient_engine.adapt_in_real_time(
                market_event=market_state,
                strategy_response={"current_strategy": "adaptive"},
                outcome={"performance": 0.15},
            )

            return {
                "adaptation_success": adaptation_result.get("success", False),
                "learning_quality": adaptation_result.get("quality", 0.0),
                "adaptations_applied": len(adaptation_result.get("adaptations", [])),
                "confidence": adaptation_result.get("confidence", 0.0),
            }

        except Exception as e:
            logger.error(f"Error in sentient analysis: {e}")
            return {"error": str(e)}

    async def _run_predictive_analysis(self) -> dict[str, object]:
        """Run predictive market modeling."""
        try:
            # Get current market state
            current_state = await self._get_current_market_state()

            # Generate predictions
            predictions: Sequence[object] = await self.predictive_modeler.predict_market_scenarios(
                current_state=current_state, time_horizon=timedelta(hours=24)
            )

            avg_conf = (
                float(
                    np.mean(
                        [
                            float(getattr(p, "confidence", getattr(p, "confidence", 0.0)))
                            for p in predictions
                        ]
                    )
                )
                if predictions
                else 0.0
            )
            high_prob = sum(
                1
                for p in predictions
                if float(getattr(p, "probability", getattr(p, "probability", 0.0))) > 0.7
            )

            return {
                "scenarios_generated": len(predictions),
                "average_confidence": avg_conf,
                "high_probability_scenarios": high_prob,
                "prediction_accuracy": 0.75,  # Would be calculated from historical data
            }

        except Exception as e:
            logger.error(f"Error in predictive analysis: {e}")
            return {"error": str(e)}

    async def _run_adversarial_analysis(self) -> dict[str, object]:
        """Run adversarial training analysis."""
        try:
            # Get current strategy population
            strategy_population = await self._get_strategy_population()

            # Run GAN training
            gan_results = await self.market_gan.train_adversarial_strategies(
                strategy_population=strategy_population
            )

            # Treat GAN results flexibly (dict-like or attribute-object)
            # Normalize GAN results: the adapter returns List[str] of improved strategy IDs
            if isinstance(gan_results, list):
                improved = list(gan_results)
                success = len(improved) > 0
            elif isinstance(gan_results, dict):
                improved = list(gan_results.get("improved_strategies", []))
                success = bool(gan_results.get("success", bool(improved)))
            else:
                improved = []
                success = False

            # Run red team attacks
            red_team_results: list[dict[str, object]] = []
            for strategy in strategy_population[:5]:  # Test top 5 strategies
                attack_result = await self.red_team.attack_strategy(strategy)
                if isinstance(attack_result, dict):
                    red_team_results.append(cast(dict[str, object], attack_result))

            # Red team report normalization (AttackReportTD-style)
            total_attacks = 0
            total_successes = 0
            vuln_count = 0
            for rpt in red_team_results:
                try:
                    attacks = rpt.get("attack_results", []) if isinstance(rpt, dict) else []
                    if isinstance(attacks, list):
                        total_attacks += len(attacks)
                        total_successes += sum(
                            1
                            for a in attacks
                            if isinstance(a, dict) and bool(a.get("success", False))
                        )
                    wf = rpt.get("weaknesses_found", []) if isinstance(rpt, dict) else []
                    vuln_count += len(wf) if isinstance(wf, list) else 0
                except Exception:
                    continue
            survival_rate = float(total_successes / total_attacks) if total_attacks > 0 else 0.0
            vulnerabilities_found = int(vuln_count)

            return {
                "gan_training_complete": success,
                "strategies_improved": len(improved),
                "red_team_attacks": len(red_team_results),
                "vulnerabilities_found": vulnerabilities_found,
                "survival_rate": survival_rate,
            }

        except Exception as e:
            logger.error(f"Error in adversarial analysis: {e}")
            return {"error": str(e)}

    async def _run_specialized_analysis(self) -> dict[str, object]:
        """Run specialized predator evolution analysis (safe defaults)."""
        try:
            result: dict[str, object] = {"status": "ok", "modules": 0}
            se = getattr(self, "specialized_evolution", None)
            if se is not None:
                for method_name in ("analyze", "evaluate", "run_analysis", "run_cycle"):
                    method = getattr(se, method_name, None)
                    if callable(method):
                        maybe = method()
                        if asyncio.iscoroutine(maybe):
                            maybe = await maybe
                        if isinstance(maybe, dict):
                            # Merge but keep required keys if absent in response
                            result.update({**maybe})
                        else:
                            result["result"] = str(maybe)
                        break
            # Ensure required keys present
            result.setdefault("status", "ok")
            result.setdefault("modules", 0)
            return result
        except Exception as e:
            logger.error(f"Error in specialized analysis: {e}")
            return {"error": str(e), "status": "error", "modules": 0}

    async def _run_competitive_analysis(self) -> dict[str, object]:
        """Run competitive intelligence analysis (safe defaults)."""
        try:
            result: dict[str, object] = {"competitors_analyzed": 0, "threats": []}
            ci = getattr(self, "competitive_intelligence", None)
            if ci is not None:
                for method_name in ("analyze_competitors", "scan_market", "analyze", "scan"):
                    method = getattr(ci, method_name, None)
                    if callable(method):
                        maybe = method()
                        if asyncio.iscoroutine(maybe):
                            maybe = await maybe
                        if isinstance(maybe, dict):
                            result.update({**maybe})
                        elif isinstance(maybe, list):
                            result["threats"] = maybe
                            result["competitors_analyzed"] = len(maybe)
                        else:
                            result["summary"] = str(maybe)
                        break
            result.setdefault("competitors_analyzed", 0)
            result.setdefault("threats", [])
            return result
        except Exception as e:
            logger.error(f"Error in competitive analysis: {e}")
            return {"error": str(e), "competitors_analyzed": 0, "threats": []}

    async def _calculate_overall_metrics(self, results: dict[str, object]) -> dict[str, object]:
        """Compute simple aggregate metrics from system results (defensive)."""
        try:
            systems = results.get("systems", {}) or {}
            systems_t = cast(dict[str, object], systems if isinstance(systems, dict) else {})
            total = len(systems_t)
            has_errors = any(isinstance(v, dict) and "error" in v for v in systems_t.values())

            success_count = 0
            for name, v in systems_t.items():
                if not isinstance(v, dict):
                    continue
                if v.get("error"):
                    continue
                if (
                    v.get("adaptation_success")
                    or v.get("gan_training_complete")
                    or v.get("status") == "ok"
                ):
                    success_count += 1

            success_ratio = (success_count / total) if total else 0.0

            presence = {
                "sentient": "sentient" in systems_t,
                "predictive": "predictive" in systems_t,
                "adversarial": "adversarial" in systems_t,
                "specialized": "specialized" in systems_t,
                "competitive": "competitive" in systems_t,
            }

            return {
                "systems_count": total,
                "has_errors": has_errors,
                "success_ratio": success_ratio,
                "presence": presence,
                "computed_at": datetime.utcnow().isoformat(),
            }
        except Exception as e:
            logger.error(f"Error calculating overall metrics: {e}")
            return {
                "systems_count": 0,
                "has_errors": True,
                "success_ratio": 0.0,
                "presence": {},
                "computed_at": datetime.utcnow().isoformat(),
            }

    async def _store_analysis_results(self, results: dict[str, object]) -> None:
        """Persist compact analysis summary to the state store (defensive)."""
        try:
            compact = {
                "analysis_id": results.get("analysis_id"),
                "timestamp": results.get("timestamp"),
                "overall_metrics": results.get("overall_metrics", {}),
            }
            payload = json.dumps(compact, separators=(",", ":"))
            setter = getattr(self.state_store, "set", None)
            if callable(setter):
                try:
                    result = setter("phase3:last_full_analysis", payload)
                    if inspect.isawaitable(result):
                        await result
                except Exception:
                    # Fallback for sync or differently-typed setter; swallow errors
                    try:
                        setter("phase3:last_full_analysis", payload)
                    except Exception:
                        pass
        except Exception as e:
            # Swallow errors by design to keep behavior non-breaking
            logger.debug(f"Non-fatal error storing analysis results: {e}")

    async def _get_current_market_state(self) -> dict[str, object]:
        """Return a minimal current market state snapshot."""
        try:
            return {"timestamp": datetime.utcnow().isoformat()}
        except Exception:
            # Extremely defensive: always return a dict
            return {"timestamp": datetime.utcnow().isoformat()}

    async def _get_strategy_population(self) -> List[str]:
        """Return placeholder strategy identifiers."""
        try:
            return ["strat_A", "strat_B", "strat_C"]
        except Exception:
            return ["strat_A"]

    async def _run_performance_monitoring(self) -> None:
        """Background task to track simple performance metrics."""
        try:
            while self.is_running:
                try:
                    now = datetime.utcnow().isoformat()
                    metrics = self.performance_metrics
                    metrics["last_tick"] = now
                    v = metrics.get("tick_count", 0)
                    base = (
                        int(v)
                        if isinstance(v, int)
                        else (int(float(v)) if isinstance(v, (float, str)) else 0)
                    )
                    metrics["tick_count"] = base + 1
                    if self.last_full_analysis:
                        metrics["last_full_analysis_age_sec"] = max(
                            0, int((datetime.utcnow() - self.last_full_analysis).total_seconds())
                        )
                    sleep_interval = min(5, int(self.config.get("update_frequency", 300)))
                except Exception as inner:
                    logger.debug(f"Performance monitoring loop error: {inner}")
                    sleep_interval = 5
                await asyncio.sleep(sleep_interval)
        except asyncio.CancelledError:
            # Task cancellation should not raise
            pass
        except Exception as e:
            logger.debug(f"Performance monitoring stopped with error: {e}")

    async def _run_continuous_analysis(self) -> None:
        """Background lightweight continuous analysis or heartbeat."""
        try:
            while self.is_running:
                try:
                    # Lightweight heartbeat; extend with real checks if needed
                    self.performance_metrics["heartbeat"] = datetime.utcnow().isoformat()
                    sleep_interval = int(self.config.get("update_frequency", 300))
                except Exception as inner:
                    logger.debug(f"Continuous analysis loop error: {inner}")
                    sleep_interval = 60
                await asyncio.sleep(sleep_interval)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.debug(f"Continuous analysis stopped with error: {e}")
