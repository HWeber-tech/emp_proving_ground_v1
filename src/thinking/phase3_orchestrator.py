from collections.abc import Mapping

"""
Phase 3 Orchestrator - Understanding & Predatory Behavior
========================================================

Main orchestrator for Phase 3 implementation that coordinates all
advanced understanding features and predatory behavior systems.

This orchestrator manages:
1. Sentient adaptation engine
2. Predictive market modeling
3. Adversarial training systems
4. Specialized predator evolution
5. Competitive understanding
"""

import asyncio
import inspect
import json
import logging
import uuid
from datetime import datetime, timedelta
from typing import Any, Coroutine, Dict, List, Optional, Protocol, Sequence, TypedDict, cast, runtime_checkable

import numpy as np

from src.core.adaptation import AdaptationService, NoOpAdaptationService
from src.core.event_bus import EventBus
from src.core.state_store import StateStore
from src.ecosystem.evolution.specialized_predator_evolution import SpecializedPredatorEvolution
from src.runtime.task_supervisor import TaskSupervisor
from src.thinking.adversarial.market_gan import MarketGAN
from src.thinking.adversarial.red_team_ai import RedTeamAI
from src.thinking.cognitive_scheduler import (
    CognitiveScheduler,
    CognitiveTask,
    CognitiveTaskDecision,
)
from src.thinking.competitive.competitive_understanding_system import (
    CompetitiveUnderstandingSystem,
)
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
class CompetitiveUnderstandingP(Protocol):
    async def initialize(self) -> bool: ...
    async def stop(self) -> bool: ...


logger = logging.getLogger(__name__)


class Phase3Orchestrator:
    """
    Main orchestrator for Phase 3 advanced understanding features.

    Coordinates all predatory behavior systems and ensures they work
    together as a unified, understanding-first ecosystem.
    """

    def __init__(
        self,
        state_store: StateStore,
        event_bus: EventBus,
        adaptation_service: Optional[AdaptationService] = None,
        *,
        task_supervisor: TaskSupervisor | None = None,
    ):
        self.state_store = state_store
        self.event_bus = event_bus
        if task_supervisor is None:
            task_supervisor = TaskSupervisor(namespace="phase3-orchestrator")
            self._owns_task_supervisor = True
        else:
            self._owns_task_supervisor = False
        self._task_supervisor = task_supervisor
        self._background_tasks: set[asyncio.Task[Any]] = set()

        # Initialize all Phase 3 systems
        self.sentient_engine = adaptation_service or NoOpAdaptationService()
        self.predictive_modeler: PredictiveModeler = PredictiveMarketModeler(state_store)
        self.market_gan: MarketGANP = MarketGAN(state_store)
        self.red_team: RedTeamP = RedTeamAI(state_store)
        self.specialized_evolution: SpecializedEvolutionP = SpecializedPredatorEvolution()
        self.competitive_understanding: CompetitiveUnderstandingP = (
            CompetitiveUnderstandingSystem(state_store)
        )
        # Legacy attribute retained for downstream compatibility during the
        # intelligence -> understanding terminology migration.
        self.competitive_intelligence = self.competitive_understanding

        # Configuration
        self.config = {
            "sentient_enabled": True,
            "predictive_enabled": True,
            "adversarial_enabled": True,
            "specialized_enabled": True,
            "competitive_enabled": True,
            "update_frequency": 300,  # 5 minutes
            "full_analysis_frequency": 3600,  # 1 hour
        }

        self.config.setdefault("cognitive_compute_budget", 3.0)

        self._cognitive_scheduler = CognitiveScheduler()
        self._analysis_history: dict[str, dict[str, object]] = {}
        self._cognitive_task_profiles: dict[str, dict[str, object]] = {
            "predictive": {
                "compute_cost": 1.2,
                "base_gain": 0.75,
                "priority": 5,
                "staleness_seconds": 300,
                "first_run_boost": 1.2,
                "max_gain": 2.4,
                "min_allocation": 0.6,
            },
            "adversarial": {
                "compute_cost": 1.0,
                "base_gain": 0.7,
                "priority": 4,
                "staleness_seconds": 540,
                "first_run_boost": 1.15,
                "max_gain": 2.2,
                "min_allocation": 0.5,
            },
            "sentient": {
                "compute_cost": 1.1,
                "base_gain": 0.55,
                "priority": 3,
                "staleness_seconds": 480,
                "first_run_boost": 1.4,
                "max_gain": 2.5,
                "min_allocation": 0.5,
            },
            "understanding": {
                "compute_cost": 0.9,
                "base_gain": 0.45,
                "priority": 2,
                "staleness_seconds": 900,
                "first_run_boost": 1.1,
                "max_gain": 2.0,
            },
            "specialized": {
                "compute_cost": 0.8,
                "base_gain": 0.4,
                "priority": 1,
                "staleness_seconds": 900,
                "first_run_boost": 1.1,
                "max_gain": 1.8,
            },
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

            # Initialize competitive understanding
            if self._understanding_enabled():
                await self.competitive_understanding.initialize()
                logger.info("✓ Competitive understanding system initialized")

            logger.info("Phase 3 systems initialization complete")
            return True

        except Exception as e:
            logger.error(f"Error initializing Phase 3 systems: {e}")
            return False

    def _spawn_background_task(
        self,
        coro: Coroutine[Any, Any, Any],
        *,
        name: str,
        task_label: str,
    ) -> asyncio.Task[Any]:
        """Create and track a background task under the shared supervisor when available."""

        metadata = {
            "component": "thinking.phase3_orchestrator",
            "task": task_label,
        }
        supervisor = self._task_supervisor
        if supervisor is None:  # pragma: no cover - defensive guard
            raise RuntimeError("Task supervisor is not configured for Phase3Orchestrator")
        task = supervisor.create(coro, name=name, metadata=metadata)
        self._background_tasks.add(task)
        task.add_done_callback(lambda completed: self._background_tasks.discard(completed))
        return task

    async def _cancel_background_tasks(self) -> None:
        """Cancel managed background tasks and await their completion."""

        if not self._background_tasks:
            return

        tasks = tuple(self._background_tasks)
        self._background_tasks.clear()
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    async def start(self, *, task_supervisor: TaskSupervisor | None = None) -> bool:
        """Start the Phase 3 orchestrator."""
        try:
            if self.is_running:
                logger.warning("Phase 3 orchestrator already running")
                return True

            if task_supervisor is not None:
                self._task_supervisor = task_supervisor
                self._owns_task_supervisor = False

            logger.info("Starting Phase 3 orchestrator...")

            # Initialize systems
            if not await self.initialize():
                return False

            self.is_running = True

            # Start background tasks
            self._spawn_background_task(
                self._run_continuous_analysis(),
                name="phase3-continuous-analysis",
                task_label="continuous_analysis",
            )
            self._spawn_background_task(
                self._run_performance_monitoring(),
                name="phase3-performance-monitor",
                task_label="performance_monitoring",
            )

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

            await self._cancel_background_tasks()

            if self._owns_task_supervisor and self._task_supervisor is not None:
                await self._task_supervisor.cancel_all()

            # Stop all systems
            await self.sentient_engine.stop()
            await self.predictive_modeler.stop()
            await self.market_gan.stop()
            await self.red_team.stop()
            await self.specialized_evolution.stop()
            await self.competitive_understanding.stop()

            logger.info("Phase 3 orchestrator stopped")
            return True

        except Exception as e:
            logger.error(f"Error stopping Phase 3 orchestrator: {e}")
            return False

    def _is_task_enabled(self, task_name: str) -> bool:
        mapping = {
            "sentient": "sentient_enabled",
            "predictive": "predictive_enabled",
            "adversarial": "adversarial_enabled",
            "specialized": "specialized_enabled",
            "understanding": "competitive_enabled",
        }
        key = mapping.get(task_name)
        return bool(self.config.get(key, True))

    @staticmethod
    def _safe_float(value: object, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _clamp(value: float, *, min_value: float = 0.0, max_value: float = 1.0) -> float:
        if value < min_value:
            return min_value
        if value > max_value:
            return max_value
        return value

    def _estimate_expected_information_gain(self, task_name: str, now: datetime) -> float:
        profile = self._cognitive_task_profiles.get(task_name, {})
        base_gain = self._safe_float(profile.get("base_gain"), default=0.3)
        history = self._analysis_history.get(task_name)
        last_gain = self._safe_float(history.get("information_gain")) if history else base_gain
        expected = max(base_gain, last_gain)
        last_run = history.get("last_run") if history else None
        staleness_seconds = self._safe_float(profile.get("staleness_seconds"), default=0.0)
        if last_run is None:
            expected *= self._safe_float(profile.get("first_run_boost"), default=1.2)
        elif staleness_seconds > 0:
            age = max((now - cast(datetime, last_run)).total_seconds(), 0.0)
            boost_cap = self._safe_float(profile.get("max_staleness_boost"), default=2.5)
            staleness_boost = 1.0 + age / staleness_seconds
            expected *= min(boost_cap, staleness_boost)
        minimum_gain = self._safe_float(profile.get("minimum_gain"), default=0.05)
        maximum_gain = self._safe_float(profile.get("max_gain"), default=expected)
        return self._clamp(expected, min_value=minimum_gain, max_value=maximum_gain)

    def _build_cognitive_tasks(self, now: datetime) -> list[CognitiveTask]:
        tasks: list[CognitiveTask] = []
        for name, profile in self._cognitive_task_profiles.items():
            enabled = self._is_task_enabled(name)
            info_gain = self._estimate_expected_information_gain(name, now) if enabled else 0.0
            history = self._analysis_history.get(name)
            task = CognitiveTask(
                name=name,
                information_gain=info_gain,
                compute_cost=self._safe_float(profile.get("compute_cost"), default=1.0),
                priority=int(profile.get("priority", 0)),
                last_run=cast(datetime | None, history.get("last_run") if history else None),
                min_allocation=self._safe_float(profile.get("min_allocation"), default=0.0),
                max_allocation=(
                    self._safe_float(profile.get("max_allocation"))
                    if profile.get("max_allocation") is not None
                    else None
                ),
                metadata={
                    "enabled": enabled,
                    "baseline_gain": profile.get("base_gain"),
                },
            )
            tasks.append(task)
        return tasks

    def _plan_cognitive_schedule(self, now: datetime) -> list[CognitiveTaskDecision]:
        tasks = self._build_cognitive_tasks(now)
        budget = self._safe_float(self.config.get("cognitive_compute_budget"), default=0.0)
        return self._cognitive_scheduler.allocate(tasks, compute_budget=budget)

    def _record_information_gain(self, task_name: str, information_gain: float, run_time: datetime) -> None:
        self._analysis_history[task_name] = {
            "last_run": run_time,
            "information_gain": max(0.0, float(information_gain)),
        }

    def _derive_information_gain(self, task_name: str, result: Mapping[str, object]) -> float:
        if not isinstance(result, Mapping):
            return 0.0
        if result.get("error") or result.get("skipped"):
            return 0.0

        if task_name == "sentient":
            quality = self._safe_float(result.get("learning_quality"))
            confidence = self._safe_float(result.get("confidence"))
            adaptations = self._safe_float(result.get("adaptations_applied"))
            info_gain = quality * 0.6 + confidence * 0.3 + self._clamp(adaptations / 5.0) * 0.1
            return self._clamp(info_gain, max_value=2.5)

        if task_name == "predictive":
            avg_conf = self._safe_float(result.get("average_confidence"))
            accuracy = self._safe_float(result.get("prediction_accuracy"))
            scenarios = max(int(self._safe_float(result.get("scenarios_generated"), default=0.0)), 1)
            high_prob = self._safe_float(result.get("high_probability_scenarios"))
            scenario_ratio = self._clamp(high_prob / scenarios)
            info_gain = avg_conf * 0.5 + accuracy * 0.3 + scenario_ratio * 0.2
            return self._clamp(info_gain, max_value=2.5)

        if task_name == "adversarial":
            improved = self._safe_float(result.get("strategies_improved"))
            vulnerabilities = self._safe_float(result.get("vulnerabilities_found"))
            survival_rate = self._clamp(self._safe_float(result.get("survival_rate")), max_value=1.0)
            info_gain = (
                self._clamp(improved / 3.0) * 0.45
                + self._clamp(vulnerabilities / 4.0) * 0.35
                + (1.0 - survival_rate) * 0.2
            )
            return self._clamp(info_gain, max_value=2.3)

        if task_name == "specialized":
            modules = self._safe_float(result.get("modules"))
            status_ok = 1.0 if str(result.get("status", "")).lower() == "ok" else 0.0
            info_gain = status_ok * 0.3 + self._clamp(modules / 4.0) * 0.7
            return self._clamp(info_gain, max_value=1.8)

        if task_name == "understanding":
            competitors = self._safe_float(result.get("competitors_analyzed"))
            threats_payload = result.get("threats", [])
            if isinstance(threats_payload, (list, tuple)):
                threat_count = len(threats_payload)
            else:
                threat_count = 0
            info_gain = self._clamp(competitors / 4.0) * 0.6 + self._clamp(threat_count / 4.0) * 0.4
            return self._clamp(info_gain, max_value=2.0)

        return 0.0

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

            decisions = self._plan_cognitive_schedule(analysis_start)
            systems = cast(Dict[str, dict[str, object]], results["systems"])
            schedule_summary: list[dict[str, object]] = []

            task_runners: dict[str, Any] = {
                "sentient": self._run_sentient_analysis,
                "predictive": self._run_predictive_analysis,
                "adversarial": self._run_adversarial_analysis,
                "specialized": self._run_specialized_analysis,
                "understanding": self._run_understanding_analysis,
            }

            for decision in decisions:
                task_name = decision.task.name
                summary_entry: dict[str, object] = {
                    "task": task_name,
                    "selected": decision.selected,
                    "allocated_compute": decision.allocated_compute,
                    "score": decision.score,
                    "expected_information_gain": decision.task.information_gain,
                    "priority": decision.task.priority,
                }
                if decision.task.last_run is not None:
                    summary_entry["last_run_at"] = decision.task.last_run.isoformat()
                if decision.reason:
                    summary_entry["reason"] = decision.reason
                if decision.task.metadata:
                    summary_entry["metadata"] = dict(decision.task.metadata)
                schedule_summary.append(summary_entry)

                enabled = bool(decision.task.metadata.get("enabled", True))
                runner = task_runners.get(task_name)

                if not enabled:
                    systems[task_name] = {
                        "skipped": True,
                        "reason": "disabled",
                    }
                    continue

                if not decision.selected:
                    systems[task_name] = {
                        "skipped": True,
                        "reason": decision.reason or "not_selected",
                    }
                    continue

                if runner is None:
                    systems[task_name] = {
                        "skipped": True,
                        "reason": "no_runner",
                    }
                    continue

                runner_coro = runner()
                payload = await runner_coro if asyncio.iscoroutine(runner_coro) else runner_coro
                if isinstance(payload, dict):
                    task_result: dict[str, object] = dict(payload)
                else:
                    task_result = {"result": payload}

                task_result["compute_allocation"] = decision.allocated_compute
                information_gain = self._derive_information_gain(task_name, task_result)
                task_result["information_gain"] = information_gain
                systems[task_name] = task_result
                self._record_information_gain(task_name, information_gain, analysis_start)

            # Calculate overall metrics
            results["overall_metrics"] = await self._calculate_overall_metrics(results)

            budget = self._safe_float(self.config.get("cognitive_compute_budget"), default=0.0)
            consumed = sum(
                decision.allocated_compute for decision in decisions if decision.selected
            )
            results["scheduler"] = {
                "budget": budget,
                "consumed": consumed,
                "remaining": max(budget - consumed, 0.0),
                "generated_at": analysis_start.isoformat(),
                "decisions": schedule_summary,
            }

            self.performance_metrics["cognitive_scheduler"] = {
                "generated_at": analysis_start.isoformat(),
                "decisions": schedule_summary,
            }

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

            # Normalize adaptation_result which may be an object (e.g., dataclass) rather than a dict
            adaptations_obj = getattr(adaptation_result, "adaptations", [])
            if isinstance(adaptations_obj, dict):
                adaptations_count = len(cast(dict[str, object], adaptations_obj))
            elif isinstance(adaptations_obj, (list, tuple, set)):
                adaptations_count = len(cast(Sequence[object], adaptations_obj))
            else:
                adaptations_count = 0
            return {
                "adaptation_success": bool(getattr(adaptation_result, "success", False)),
                "learning_quality": float(getattr(adaptation_result, "learning_quality", 0.0)),
                "adaptations_applied": adaptations_count,
                "confidence": float(getattr(adaptation_result, "confidence", 0.0)),
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
            red_team_results = []
            for strategy in strategy_population[:5]:  # Test top 5 strategies
                attack_result = await self.red_team.attack_strategy(strategy)
                red_team_results.append(attack_result)

            # Red team report normalization (AttackReportTD-style)
            total_attacks = 0
            total_successes = 0
            vuln_count = 0
            for rpt in red_team_results:
                if not isinstance(rpt, Mapping):
                    logger.debug(
                        "Skipping red team report with unexpected type %s",
                        type(rpt).__name__,
                    )
                    continue

                attacks = rpt.get("attack_results", [])
                if isinstance(attacks, list):
                    total_attacks += len(attacks)
                    total_successes += sum(
                        1
                        for a in attacks
                        if isinstance(a, Mapping) and bool(a.get("success", False))
                    )
                elif attacks:
                    logger.debug(
                        "Ignoring attack_results payload of type %s",
                        type(attacks).__name__,
                    )

                weaknesses = rpt.get("weaknesses_found", [])
                if isinstance(weaknesses, list):
                    vuln_count += len(weaknesses)
                elif weaknesses:
                    logger.debug(
                        "Ignoring weaknesses_found payload of type %s",
                        type(weaknesses).__name__,
                    )
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

    def _understanding_enabled(self) -> bool:
        """Return the feature flag for competitive understanding systems."""

        cfg = self.config
        return bool(cfg.get("competitive_enabled", True))

    async def _run_understanding_analysis(self) -> dict[str, object]:
        """Run competitive understanding analysis (safe defaults)."""
        try:
            result: dict[str, object] = {"competitors_analyzed": 0, "threats": []}
            ci = getattr(self, "competitive_understanding", None)
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
            logger.error(f"Error in understanding analysis: {e}")
            return {"error": str(e), "competitors_analyzed": 0, "threats": []}

    async def _run_competitive_analysis(self) -> dict[str, object]:  # pragma: no cover - legacy alias
        """Compatibility shim for callers using the deprecated intelligence name."""

        return await self._run_understanding_analysis()

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
                "understanding": "understanding" in systems_t,
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
                except TypeError as exc:
                    logger.debug(
                        "Retrying state_store.set after TypeError: %s",
                        exc,
                        exc_info=exc,
                    )
                    try:
                        setter("phase3:last_full_analysis", payload)
                    except Exception as fallback_exc:  # pragma: no cover - defensive logging
                        logger.warning(
                            "Failed fallback persistence via state store: %s",
                            fallback_exc,
                            exc_info=fallback_exc,
                        )
                except Exception as exc:  # pragma: no cover - defensive logging
                    logger.warning(
                        "Failed to persist analysis results via state store: %s",
                        exc,
                        exc_info=exc,
                    )
        except Exception as e:
            # Swallow errors by design to keep behavior non-breaking
            logger.debug(f"Non-fatal error storing analysis results: {e}")

    async def _get_current_market_state(self) -> dict[str, object]:
        """Return a minimal current market state snapshot."""
        state_key = "phase3:market_state:last"

        def _as_float(value: object, default: float = 0.0) -> float:
            try:
                return float(value)
            except (TypeError, ValueError):
                return default

        def _as_int(value: object, default: int = 0) -> int:
            try:
                if isinstance(value, bool):
                    return int(value)
                return int(float(value))
            except (TypeError, ValueError):
                return default

        try:
            previous_raw = await self.state_store.get(state_key)
        except Exception as exc:  # pragma: no cover - defensive logging only
            logger.debug("Unable to read previous market state", exc_info=exc)
            previous_raw = None

        previous_snapshot: dict[str, object] | None = None
        if previous_raw:
            try:
                decoded = json.loads(previous_raw)
                if isinstance(decoded, dict):
                    previous_snapshot = decoded
            except json.JSONDecodeError as exc:  # pragma: no cover - corrupt payload guard
                logger.debug("Discarding corrupt market state payload", exc_info=exc)

        now = datetime.utcnow()
        previous_state = (
            previous_snapshot.get("market_state") if isinstance(previous_snapshot, dict) else None
        )
        if not isinstance(previous_state, dict):
            previous_state = None

        prev_price = _as_float(previous_state.get("price"), 1.0) if previous_state else 1.0
        prev_vol = _as_float(previous_state.get("volatility"), 0.015) if previous_state else 0.015
        prev_spread = _as_float(previous_state.get("spread"), 0.0001) if previous_state else 0.0001
        prev_volume = _as_float(previous_state.get("volume"), 750.0) if previous_state else 750.0

        prev_timestamp = (
            previous_snapshot.get("timestamp") if isinstance(previous_snapshot, dict) else None
        )
        elapsed_ms = 0.0
        if isinstance(prev_timestamp, str):
            try:
                prev_dt = datetime.fromisoformat(prev_timestamp)
                elapsed_ms = max(0.0, (now - prev_dt).total_seconds() * 1000.0)
            except ValueError:
                elapsed_ms = 0.0

        tick_raw = self.performance_metrics.get("tick_count", 0)
        tick_count = _as_int(tick_raw, 0)
        cycle_position = (tick_count % 8) - 4
        momentum = 0.0 if previous_state is None else cycle_position * 0.000025

        price = max(0.0, prev_price + momentum)
        volatility = max(0.0, 0.9 * prev_vol + abs(momentum) * 12.0 + 0.0005)
        spread = max(0.00001, 0.96 * prev_spread + 0.00001)
        volume = max(100.0, prev_volume * 0.75 + 150.0 + abs(momentum) * 50_000.0)

        directional_edge = momentum / max(prev_price if prev_price else price, 1e-6)
        volatility_penalty = max(0.0, volatility - prev_vol) * 50.0
        spread_cost = spread * 5000.0
        liquidity_bonus = min(volume / 2000.0, 1.5) * 2.5
        reward_proxy = directional_edge * 10_000.0 + liquidity_bonus - volatility_penalty - spread_cost

        essentials = [
            round(price, 6),
            round(volatility, 6),
            round(momentum, 6),
            round(spread, 6),
            round(volume, 2),
        ]

        transition = {
            "price_change": round(momentum, 6),
            "volatility_change": round(volatility - prev_vol, 6),
            "volume_change": round(volume - prev_volume, 2),
            "elapsed_ms": round(elapsed_ms, 3),
            "trend": "up" if momentum > 0 else ("down" if momentum < 0 else "flat"),
        }

        previous_compact = None
        if previous_state:
            previous_compact = {
                "price": round(prev_price, 6),
                "volatility": round(prev_vol, 6),
                "spread": round(prev_spread, 6),
                "volume": round(prev_volume, 2),
            }

        snapshot: dict[str, object] = {
            "timestamp": now.isoformat(),
            "market_state": {
                "price": essentials[0],
                "volatility": essentials[1],
                "momentum": essentials[2],
                "spread": essentials[3],
                "volume": essentials[4],
            },
            "previous_state": previous_compact,
            "transition": transition,
            "reward_proxy": round(reward_proxy, 6),
            "essentials": essentials,
            "meta": {"tick_count": tick_count},
        }

        try:
            encoded = json.dumps(snapshot, separators=(",", ":"), sort_keys=True)
            await self.state_store.set(state_key, encoded, expire=3600)
        except Exception as exc:  # pragma: no cover - persistence best effort
            logger.debug("Unable to persist market state snapshot", exc_info=exc)

        self.performance_metrics["last_market_state"] = snapshot["market_state"]
        return snapshot

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
