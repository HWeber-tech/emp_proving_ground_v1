"""
Component Integrator Implementation
===================================

Concrete implementation of IComponentIntegrator for system-wide component management.
Provides centralized initialization, monitoring, and coordination of all system components.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from src.core.interfaces import IComponentIntegrator
from ..core import PopulationManager, SensoryOrgan, RiskManager
from src.core.performance.market_data_cache import get_global_cache

logger = logging.getLogger(__name__)


class ComponentIntegrator(IComponentIntegrator):
    """Centralized component integration and management system."""
    
    def __init__(self):
        """Initialize component integrator."""
        self.components: Dict[str, Any] = {}
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
            
        except Exception as e:
            logger.error(f"Error initializing components: {e}")
            return False
    
    async def _initialize_core_components(self) -> None:
        """Initialize core system components."""
        logger.info("Initializing core components...")
        
        # Population Manager
        population_manager = PopulationManager(population_size=100)
        self.components['population_manager'] = population_manager
        self.component_status['population_manager'] = 'initialized'
        
        # Risk Manager
        risk_manager = RiskManager()
        self.components['risk_manager'] = risk_manager
        self.component_status['risk_manager'] = 'initialized'
        
        logger.info("Core components initialized")
    
    async def _initialize_sensory_components(self) -> None:
        """Initialize sensory components."""
        logger.info("Initializing sensory components...")
        
        # 4D+1 Sensory Organs
        sensory_organs = {
            'what_organ': SensoryOrgan('what'),
            'when_organ': SensoryOrgan('when'),
            'anomaly_organ': SensoryOrgan('anomaly'),
            'chaos_organ': SensoryOrgan('chaos')
        }
        
        for name, organ in sensory_organs.items():
            self.components[name] = organ
            self.component_status[name] = 'initialized'
        
        logger.info("Sensory components initialized")
    
    async def _initialize_risk_management(self) -> None:
        """Initialize risk management components."""
        logger.info("Initializing risk management...")
        
        # Risk manager is already initialized in core components
        self.component_status['risk_management'] = 'initialized'
        
        logger.info("Risk management initialized")
    
    async def _initialize_performance_components(self) -> None:
        """Initialize performance components."""
        logger.info("Initializing performance components...")
        
        # Performance cache is already initialized via get_global_cache()
        self.component_status['performance_cache'] = 'initialized'
        
        logger.info("Performance components initialized")
    
    async def shutdown_components(self) -> bool:
        """Shutdown all system components."""
        try:
            logger.info("Shutting down system components...")
            
            # Shutdown in reverse order
            shutdown_order = [
                'performance_cache',
                'risk_management',
                'chaos_organ',
                'anomaly_organ',
                'when_organ',
                'what_organ',
                'risk_manager',
                'population_manager'
            ]
            
            for component_name in shutdown_order:
                if component_name in self.components:
                    self.component_status[component_name] = 'shutdown'
                    logger.info(f"Shutdown {component_name}")
            
            self.components.clear()
            self._initialized = False
            
            logger.info("All components shutdown successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error shutting down components: {e}")
            return False
    
    async def get_component_status(self, component_name: str) -> Optional[str]:
        """Get status of a specific component."""
        return self.component_status.get(component_name, 'unknown')
    
    async def restart_component(self, component_name: str) -> bool:
        """Restart a specific component."""
        try:
            if component_name not in self.components:
                logger.warning(f"Component {component_name} not found")
                return False
            
            logger.info(f"Restarting component: {component_name}")
            
            # Shutdown the component
            self.component_status[component_name] = 'shutdown'
            
            # Re-initialize based on component type
            if component_name == 'population_manager':
                self.components[component_name] = PopulationManager(population_size=100)
            elif component_name == 'risk_manager':
                self.components[component_name] = RiskManager()
            elif component_name.endswith('_organ'):
                organ_type = component_name.replace('_organ', '')
                self.components[component_name] = SensoryOrgan(organ_type)
            
            self.component_status[component_name] = 'initialized'
            logger.info(f"Component {component_name} restarted successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error restarting component {component_name}: {e}")
            return False
    
    def get_all_components(self) -> Dict[str, Any]:
        """Get all registered components."""
        return self.components.copy()
    
    def get_component_summary(self) -> Dict[str, Any]:
        """Get summary of all components."""
        return {
            'total_components': len(self.components),
            'initialized_components': sum(1 for status in self.component_status.values() if status == 'initialized'),
            'shutdown_components': sum(1 for status in self.component_status.values() if status == 'shutdown'),
            'component_status': self.component_status.copy()
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check."""
        health_report = {
            'timestamp': datetime.now().isoformat(),
            'overall_status': 'healthy' if self._initialized else 'unhealthy',
            'components': {}
        }
        
        for name, component in self.components.items():
            try:
                # Basic health check
                if hasattr(component, 'get_performance_metrics'):
                    metrics = component.get_performance_metrics()
                    health_report['components'][name] = {
                        'status': 'healthy',
                        'metrics': metrics
                    }
                else:
                    health_report['components'][name] = {
                        'status': 'healthy',
                        'metrics': {}
                    }
            except Exception as e:
                health_report['components'][name] = {
                    'status': 'unhealthy',
                    'error': str(e)
                }
        
        return health_report
    
    async def get_system_metrics(self) -> Dict[str, Any]:
        """Get comprehensive system metrics."""
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'component_count': len(self.components),
            'initialized_count': sum(1 for status in self.component_status.values() if status == 'initialized'),
            'cache_status': 'connected' if self.cache else 'disconnected'
        }
        
        # Add component-specific metrics
        for name, component in self.components.items():
            if hasattr(component, 'get_performance_metrics'):
                metrics[name] = component.get_performance_metrics()
        
        return metrics
    
    def is_initialized(self) -> bool:
        """Check if system is initialized."""
        return self._initialized


# Global component integrator instance
_global_integrator = None


def get_global_component_integrator() -> ComponentIntegrator:
    """Get global component integrator instance."""
    global _global_integrator
    if _global_integrator is None:
        _global_integrator = ComponentIntegrator()
    return _
