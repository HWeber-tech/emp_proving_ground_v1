#!/usr/bin/env python3
"""
Complete Core Interfaces for the EMP Trading System
==================================================

This module contains all the missing interfaces that need to be implemented
for the EMP system to function properly.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple, Callable
from datetime import datetime
import uuid
from pydantic import BaseModel, Field


class IExecutionEngine(ABC):
    """Interface for order execution engine."""
    
    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize the execution engine."""
        pass
    
    @abstractmethod
    async def execute_order(self, order: Any) -> bool:
        """Execute a trading order."""
        pass
    
    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an active order."""
        pass
    
    @abstractmethod
    async def get_position(self, symbol: str) -> Optional[Any]:
        """Get current position for a symbol."""
        pass
    
    @abstractmethod
    async def get_active_orders(self) -> List[Any]:
        """Get all active orders."""
        pass


class IStrategyEngine(ABC):
    """Interface for strategy engine functionality."""
    
    @abstractmethod
    def register_strategy(self, strategy: Any) -> bool:
        """Register a new strategy with the engine."""
        pass
    
    @abstractmethod
    def unregister_strategy(self, strategy_id: str) -> bool:
        """Unregister a strategy from the engine."""
        pass
    
    @abstractmethod
    def start_strategy(self, strategy_id: str) -> bool:
        """Start strategy execution."""
        pass
    
    @abstractmethod
    def stop_strategy(self, strategy_id: str) -> bool:
        """Stop strategy execution."""
        pass
    
    @abstractmethod
    def pause_strategy(self, strategy_id: str) -> bool:
        """Pause strategy execution."""
        pass
    
    @abstractmethod
    async def execute_strategy(self, strategy_id: str, market_data: Dict[str, Any]) -> Any:
        """Execute a strategy with market data."""
        pass
    
    @abstractmethod
    async def execute_all_strategies(self, market_data: Dict[str, Any]) -> List[Any]:
        """Execute all active strategies with market data."""
        pass
    
    @abstractmethod
    def get_strategy_status(self, strategy_id: str) -> Optional[str]:
        """Get the current status of a strategy."""
        pass
    
    @abstractmethod
    def get_strategy_performance(self, strategy_id: str) -> Optional[Dict[str, Any]]:
        """Get performance metrics for a strategy."""
        pass
    
    @abstractmethod
    def get_all_strategies(self) -> List[str]:
        """Get list of all registered strategy IDs."""
        pass
    
    @abstractmethod
    def get_active_strategies(self) -> List[str]:
        """Get list of active strategy IDs."""
        pass


class IPopulationManager(ABC):
    """Interface for managing populations in genetic algorithms."""
    
    @abstractmethod
    def initialize_population(self, genome_factory: Callable) -> None:
        """Initialize the population with new genomes."""
        pass
    
    @abstractmethod
    def get_population(self) -> List[Any]:
        """Get the current population."""
        pass
    
    @abstractmethod
    def get_best_genomes(self, count: int) -> List[Any]:
        """Get the top N genomes by fitness."""
        pass
    
    @abstractmethod
    def update_population(self, new_population: List[Any]) -> None:
        """Replace the current population with a new one."""
        pass
    
    @abstractmethod
    def get_population_statistics(self) -> Dict[str, Any]:
        """Get statistics about the current population."""
        pass
    
    @abstractmethod
    def advance_generation(self) -> None:
        """Increment the generation counter."""
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """Reset the population manager to initial state."""
        pass


class IValidationEngine(ABC):
    """Interface for validation engine."""
    
    @abstractmethod
    async def validate_market_data(self, data: Dict[str, Any]) -> bool:
        """Validate market data integrity."""
        pass
    
    @abstractmethod
    async def validate_strategy(self, strategy: Any) -> bool:
        """Validate strategy parameters."""
        pass
    
    @abstractmethod
    async def validate_order(self, order: Any) -> bool:
        """Validate order parameters."""
        pass


class IComponentIntegrator(ABC):
    """Interface for component integration."""
    
    @abstractmethod
    async def initialize_components(self) -> bool:
        """Initialize all system components."""
        pass
    
    @abstractmethod
    async def shutdown_components(self) -> bool:
        """Shutdown all system components."""
        pass
    
    @abstractmethod
    async def get_component_status(self, component_name: str) -> Optional[str]:
        """Get status of a specific component."""
        pass
    
    @abstractmethod
    async def restart_component(self, component_name: str) -> bool:
        """Restart a specific component."""
        pass
