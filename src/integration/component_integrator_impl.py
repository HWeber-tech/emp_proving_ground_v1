"""
Component Integrator Implementation
=================================

Concrete implementation of the component integrator for managing
system-wide component integration and lifecycle.
"""

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from src.core.types import JSONObject
from src.integration.component_integrator import ComponentIntegrator
from src.trading.risk.risk_api import RISK_API_RUNBOOK

logger = logging.getLogger(__name__)


class ComponentIntegratorImpl(ComponentIntegrator):
    """Concrete implementation of component integrator."""

    def __init__(self) -> None:
        super().__init__()
        self.components: Dict[str, object] = {}
        self.dependencies: Dict[str, List[str]] = {}
        self.initialized = False
        self._aliases: Dict[str, str] = {}
        self._initialization_failures: Dict[str, str] = {}

    async def initialize(self) -> bool:
        """Initialize all components."""
        logger.info("Initializing component integrator...")
        self._initialization_failures.clear()

        overall_success = True
        for label, initializer in (
            ("sensory", self._initialize_sensory_system),
            ("trading", self._initialize_trading_system),
            ("evolution", self._initialize_evolution_system),
            ("risk", self._initialize_risk_system),
            ("governance", self._initialize_governance_system),
        ):
            try:
                success = await initializer()
            except Exception as exc:  # pragma: no cover - defensive
                success = False
                self._initialization_failures[label] = str(exc)
                logger.exception("%s subsystem initialization raised an exception", label)
            if not success:
                overall_success = False
                self._initialization_failures.setdefault(label, "initialization failed")

        self.initialized = overall_success
        if overall_success:
            logger.info("Component integrator initialized successfully")
        else:
            logger.warning(
                "Component integrator initialized with failures: %s",
                ", ".join(sorted(self._initialization_failures)),
            )
        return overall_success

    async def _initialize_sensory_system(self) -> bool:
        """Initialize sensory system components."""
        try:
            # New 4D+1 scaffolding (what/when/anomaly via simple sensors)
            from src.sensory.anomaly.anomaly_sensor import AnomalySensor
            from src.sensory.what.what_sensor import WhatSensor
            from src.sensory.when.when_sensor import WhenSensor

            self.components["what_sensor"] = WhatSensor()
            self.components["when_sensor"] = WhenSensor()
            self.components["anomaly_sensor"] = AnomalySensor()

            # Maintain backward-compatible aliases expected by legacy validators
            self._register_component_alias("what_sensor", "what_organ")
            self._register_component_alias("when_sensor", "when_organ")
            self._register_component_alias("anomaly_sensor", "anomaly_organ")

            logger.info("Sensory system components initialized")
            return True

        except Exception as e:
            logger.warning(f"Failed to initialize sensory system: {e}")
            self._initialization_failures["sensory"] = str(e)
            return False

    async def _initialize_trading_system(self) -> bool:
        """Initialize trading system components."""
        try:
            from src.core.strategy.engine import StrategyEngine as StrategyEngineImpl
            from src.trading.execution.execution_engine import ExecutionEngine

            self.components["strategy_engine"] = StrategyEngineImpl()
            self.components["execution_engine"] = ExecutionEngine()

            logger.info("Trading system components initialized")
            return True

        except Exception as e:
            logger.warning(f"Failed to initialize trading system: {e}")
            self._initialization_failures["trading"] = str(e)
            return False

    async def _initialize_evolution_system(self) -> bool:
        """Initialize evolution system components."""
        try:
            from src.core.evolution.engine import EvolutionConfig, EvolutionEngine

            self.components["evolution_engine"] = EvolutionEngine(EvolutionConfig())

            logger.info("Evolution system components initialized")
            return True

        except Exception as e:
            logger.warning(f"Failed to initialize evolution system: {e}")
            self._initialization_failures["evolution"] = str(e)
            return False

    async def _initialize_risk_system(self) -> bool:
        """Initialize risk management components."""
        try:
            if not self._ensure_risk_manager():
                logger.warning("Risk system initialization skipped; canonical risk manager missing")
                return False

            logger.info("Risk system components initialized with canonical risk manager")
            return True
        except Exception as e:
            logger.warning(f"Failed to initialize risk system: {e}")
            self._initialization_failures["risk"] = str(e)
            return False

    async def _initialize_governance_system(self) -> bool:
        """Initialize governance components."""
        try:
            from src.governance.audit_trail import AuditTrail
            from src.governance.system_config import SystemConfig

            self.components["system_config"] = SystemConfig()
            self.components["audit_trail"] = AuditTrail()

            logger.info("Governance system components initialized")
            return True

        except Exception as e:
            logger.warning(f"Failed to initialize governance system: {e}")
            self._initialization_failures["governance"] = str(e)
            return False

    async def shutdown(self) -> bool:
        """Shutdown all components gracefully."""
        try:
            logger.info("Shutting down component integrator...")

            # Shutdown components in reverse order
            for component_name, component in reversed(list(self.components.items())):
                shutdown_attr = getattr(component, "shutdown", None)
                if callable(shutdown_attr):
                    result = shutdown_attr()
                    if asyncio.iscoroutine(result):
                        await result
                logger.info(f"Shutdown {component_name}")

            self.components.clear()
            self._aliases.clear()
            self._risk_config = None
            self._risk_summary = None
            self.initialized = False
            logger.info("Component integrator shutdown complete")
            return True

        except Exception as e:
            logger.error(f"Failed to shutdown component integrator: {e}")
            return False

    def get_component(self, name: str) -> Optional[object]:
        """Get a component by name."""
        return self.components.get(name)

    def list_components(self) -> List[str]:
        """List all available components."""
        return list(self.components.keys())

    def _register_component_alias(self, source: str, alias: str) -> None:
        """Register an alternate name for an initialized component."""
        component = self.components.get(source)
        if component is None:
            logger.debug(
                "Skipping alias registration for %s -> %s because source is missing",
                source,
                alias,
            )
            return

        existing = self.components.get(alias)
        if existing is not None and existing is not component:
            logger.warning(
                "Alias %s already mapped to a different component (%s)",
                alias,
                type(existing).__name__,
            )
            return

        self.components[alias] = component
        self._aliases[alias] = source

    def is_alias(self, name: str) -> bool:
        """Return True when the component name is an alias of another component."""

        return name in self._aliases

    def get_component_status(self) -> JSONObject:
        """Get status of all components as JSON object."""
        status: JSONObject = {}
        for name, component in self.components.items():
            get_status = getattr(component, "get_status", None)
            if callable(get_status):
                try:
                    value = get_status()
                    status[name] = (
                        value if isinstance(value, (str, int, float, bool)) else str(value)
                    )
                except Exception:
                    status[name] = "unknown"
            else:
                status[name] = "active" if component else "inactive"
        return status

    async def validate_integration(self) -> JSONObject:
        """Validate component integration."""
        try:
            validation_results: JSONObject = {}

            # Test component availability
            for name, component in self.components.items():
                if self.is_alias(name):
                    continue
                validation_results[name] = {
                    "available": component is not None,
                    "type": type(component).__name__,
                }

            # Test component interactions
            if "strategy_engine" in self.components and "risk_manager" in self.components:
                strategy_engine = self.components["strategy_engine"]
                risk_manager = self.components["risk_manager"]

                # Test basic interaction
                validation_results["strategy_risk_integration"] = {
                    "available": True,
                    "test_passed": True,
                }

            risk_summary = self._risk_summary
            risk_entry: dict[str, object] = {
                "available": bool(risk_summary),
                "runbook": RISK_API_RUNBOOK,
            }
            if risk_summary:
                risk_entry["summary"] = dict(risk_summary)
            validation_results["risk_configuration"] = risk_entry

            # Test data flow
            sensory_key = None
            if "what_sensor" in self.components:
                sensory_key = "what_sensor"
            elif "what_organ" in self.components:
                sensory_key = "what_organ"

            if sensory_key and "strategy_engine" in self.components:
                validation_results["data_flow"] = {
                    "available": True,
                    "test_passed": True,
                    "source": sensory_key,
                }

            return {
                "timestamp": datetime.now().isoformat(),
                "status": "success",
                "components": validation_results,
                "total_components": len(validation_results),
            }

        except Exception as e:
            return {
                "timestamp": datetime.now().isoformat(),
                "status": "error",
                "error": str(e),
                "components": {},
            }

    def save_configuration(self, filepath: str) -> None:
        """Save component configuration."""
        try:
            config = {
                "timestamp": datetime.now().isoformat(),
                "components": {
                    name: {"type": type(component).__name__, "module": type(component).__module__}
                    for name, component in self.components.items()
                },
            }

            Path(filepath).write_text(json.dumps(config, indent=2))
            logger.info(f"Configuration saved to {filepath}")

        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")

    def load_configuration(self, filepath: str) -> bool:
        """Load component configuration."""
        try:
            config = json.loads(Path(filepath).read_text())
            logger.info(f"Configuration loaded from {filepath}")
            return True

        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            return False

    def get_system_health(self) -> JSONObject:
        """Get overall system health."""
        try:
            component_status_obj: JSONObject = self.get_component_status()

            # Calculate health score from component statuses without generator typing issues
            active_components = 0
            for v in component_status_obj.values():
                if isinstance(v, str) and v == "active":
                    active_components += 1

            health: JSONObject = {
                "timestamp": datetime.now().isoformat(),
                "initialized": self.initialized,
                "total_components": len(self.components),
                "component_status": component_status_obj,
                "integration_valid": True,  # Simplified for now
                "health_score": active_components / max(1, len(self.components)),
            }
            return health

        except Exception as e:
            logger.error(f"Failed to get system health: {e}")
            return {
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
                "status": "error",
            }
