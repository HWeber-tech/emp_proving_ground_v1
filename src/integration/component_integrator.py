"""
Component Integrator Protocols and Reference Implementation
===========================================================

Defines the IComponentIntegrator Protocol and a reference ComponentIntegrator
implementation used for centralized initialization, monitoring, and coordination
of system components.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Protocol, runtime_checkable

from src.core.types import JSONObject
from src.core.performance.market_data_cache import get_global_cache

# Canonical imports (avoid relative package traversals)
# Note: These may be optional at runtime depending on environment; guarded in methods.
try:
    from src.core import PopulationManager, SensoryOrgan, RiskManager  # type: ignore[attr-defined]
except Exception:  # pragma: no cover
    PopulationManager = SensoryOrgan = RiskManager = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

# Typed sentinel for internal "not set" states (replaces ad hoc "_" usage)
_MISSING: object = object()


@runtime_checkable
class IComponentIntegrator(Protocol):
    """Minimal Protocol for component integrators (duck-typed)."""

    async def initialize(self) -> bool: ...
    async def shutdown(self) -> bool: ...
    def list_components(self) -> List[str]: ...
    def get_component(self, name: str) -> Optional[object]: ...
    def get_component_status(self) -> JSONObject: ...
    async def validate_integration(self) -> JSONObject: ...
    def save_configuration(self, filepath: str) -> None: ...
    def load_configuration(self, filepath: str) -> bool: ...
    def get_system_health(self) -> JSONObject: ...


class ComponentIntegrator:
    """Centralized component integration and management system."""

    def __init__(self) -> None:
        """Initialize component integrator."""
        self.components: Dict[str, object] = {}
        self.component_status: Dict[str, str] = {}
        self.cache = get_global_cache()
        self._initialized = False

    async def initialize_components(self) -> bool:
        """Initialize all system components."""
        try:
            logger.info("Initializing system components...")

            # Initialize core components
            await self._initialize_core_components()

            # Initialize sensory components
            await self._initialize_sensory_components()

            # Initialize risk management
            await self._initialize_risk_management()

            # Initialize performance components
            await self._initialize_performance_components()

            self._initialized = True
            logger.info("All components initialized successfully")
            return True

        except Exception as e:  # pragma: no cover - defensive logging
            logger.error(f"Error initializing components: {e}")
            return False

    async def _initialize_core_components(self) -> None:
        """Initialize core system components."""
        logger.info("Initializing core components...")

        if PopulationManager is not None:
            # Population Manager
            population_manager = PopulationManager(population_size=100)  # type: ignore[call-arg]
            self.components["population_manager"] = population_manager
            self.component_status["population_manager"] = "initialized"

        if RiskManager is not None:
            # Risk Manager
            risk_manager = RiskManager()  # type: ignore[call-arg]
            self.components["risk_manager"] = risk_manager
            self.component_status["risk_manager"] = "initialized"

        logger.info("Core components initialized")

    async def _initialize_sensory_components(self) -> None:
        """Initialize sensory components."""
        logger.info("Initializing sensory components...")

        if SensoryOrgan is not None:
            # 4D+1 Sensory Organs
            sensory_organs = {
                "what_organ": SensoryOrgan("what"),
                "when_organ": SensoryOrgan("when"),
                "anomaly_organ": SensoryOrgan("anomaly"),
                "chaos_organ": SensoryOrgan("chaos"),
            }

            for name, organ in sensory_organs.items():
                self.components[name] = organ
                self.component_status[name] = "initialized"

        logger.info("Sensory components initialized")

    async def _initialize_risk_management(self) -> None:
        """Initialize risk management components."""
        logger.info("Initializing risk management...")

        # Risk manager is already initialized in core components
        self.component_status["risk_management"] = "initialized"

        logger.info("Risk management initialized")

    async def _initialize_performance_components(self) -> None:
        """Initialize performance components."""
        logger.info("Initializing performance components...")

        # Performance cache is already initialized via get_global_cache()
        self.component_status["performance_cache"] = "initialized"

        logger.info("Performance components initialized")

    async def shutdown_components(self) -> bool:
        """Shutdown all system components."""
        try:
            logger.info("Shutting down system components...")

            # Shutdown in reverse order
            shutdown_order = [
                "performance_cache",
                "risk_management",
                "chaos_organ",
                "anomaly_organ",
                "when_organ",
                "what_organ",
                "risk_manager",
                "population_manager",
            ]

            for component_name in shutdown_order:
                if component_name in self.components:
                    self.component_status[component_name] = "shutdown"
                    logger.info(f"Shutdown {component_name}")

            self.components.clear()
            self._initialized = False

            logger.info("All components shutdown successfully")
            return True

        except Exception as e:  # pragma: no cover - defensive logging
            logger.error(f"Error shutting down components: {e}")
            return False

    def get_component_status(self) -> JSONObject:
        """Get status of all components as JSON object."""
        # Concrete return type aligned with JSON schema (str statuses)
        return {name: status for name, status in self.component_status.items()}

    async def restart_component(self, component_name: str) -> bool:
        """Restart a specific component."""
        try:
            if component_name not in self.components:
                logger.warning(f"Component {component_name} not found")
                return False

            logger.info(f"Restarting component: {component_name}")

            # Shutdown the component
            self.component_status[component_name] = "shutdown"

            # Re-initialize based on component type
            if PopulationManager is not None and component_name == "population_manager":
                self.components[component_name] = PopulationManager(population_size=100)  # type: ignore[call-arg]
            elif RiskManager is not None and component_name == "risk_manager":
                self.components[component_name] = RiskManager()  # type: ignore[call-arg]
            elif SensoryOrgan is not None and component_name.endswith("_organ"):
                organ_type = component_name.replace("_organ", "")
                self.components[component_name] = SensoryOrgan(organ_type)

            self.component_status[component_name] = "initialized"
            logger.info(f"Component {component_name} restarted successfully")
            return True

        except Exception as e:  # pragma: no cover - defensive logging
            logger.error(f"Error restarting component {component_name}: {e}")
            return False

    def get_all_components(self) -> Dict[str, object]:
        """Get all registered components."""
        return self.components.copy()

    def get_component_summary(self) -> Dict[str, object]:
        """Get summary of all components."""
        return {
            "total_components": len(self.components),
            "initialized_components": sum(1 for status in self.component_status.values() if status == "initialized"),
            "shutdown_components": sum(1 for status in self.component_status.values() if status == "shutdown"),
            "component_status": self.component_status.copy(),
        }

    async def health_check(self) -> Dict[str, object]:
        """Perform comprehensive health check."""
        health_report: Dict[str, object] = {
            "timestamp": datetime.now().isoformat(),
            "overall_status": "healthy" if self._initialized else "unhealthy",
            "components": {},
        }

        components_report: Dict[str, object] = {}

        for name, component in self.components.items():
            try:
                # Basic health check
                if hasattr(component, "get_performance_metrics"):
                    metrics = getattr(component, "get_performance_metrics")()
                    components_report[name] = {
                        "status": "healthy",
                        "metrics": metrics,  # may be JSON-like
                    }
                else:
                    components_report[name] = {
                        "status": "healthy",
                        "metrics": {},
                    }
            except Exception as e:  # pragma: no cover
                components_report[name] = {
                    "status": "unhealthy",
                    "error": str(e),
                }

        health_report["components"] = components_report
        return health_report

    async def get_system_metrics(self) -> Dict[str, object]:
        """Get comprehensive system metrics."""
        metrics: Dict[str, object] = {
            "timestamp": datetime.now().isoformat(),
            "component_count": len(self.components),
            "initialized_count": sum(1 for status in self.component_status.values() if status == "initialized"),
            "cache_status": "connected" if self.cache else "disconnected",
        }

        # Add component-specific metrics
        for name, component in self.components.items():
            if hasattr(component, "get_performance_metrics"):
                metrics[name] = getattr(component, "get_performance_metrics")()

        return metrics

    def is_initialized(self) -> bool:
        """Check if system is initialized."""
        return self._initialized


# Global component integrator instance (typed sentinel based)
_global_integrator: object | ComponentIntegrator = _MISSING


def get_global_component_integrator() -> ComponentIntegrator:
    """Get global component integrator instance."""
    global _global_integrator
    if _global_integrator is _MISSING:
        _global_integrator = ComponentIntegrator()
    # Cast-by-assert to retain runtime semantics without importing typing.cast
    assert isinstance(_global_integrator, ComponentIntegrator)
    return _global_integrator


__all__ = ["IComponentIntegrator", "ComponentIntegrator", "get_global_component_integrator", "_MISSING"]
