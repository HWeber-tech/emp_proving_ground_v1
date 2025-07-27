"""
Core Interfaces
===============

Interface definitions for the EMP trading system.
Provides abstract base classes for all system components.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass


@dataclass
class DecisionGenome:
    """Represents a decision genome in the genetic algorithm."""
    id: str
    fitness: float = 0.0
    species_type: str = 'generic'
    parameters: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.parameters is None:
            self.parameters = {}


@dataclass
class SensorySignal:
    """Represents a signal from a sensory organ."""
    signal_type: str
    strength: float
    confidence: float
    source: str
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class IPopulationManager(ABC):
    """Interface for population management in genetic algorithms."""
    
    @abstractmethod
    def initialize_population(self, genome_factory: Callable) -> None:
        """Initialize population with new genomes."""
        pass
    
    @abstractmethod
    def get_population(self) -> List[DecisionGenome]:
        """Get current population."""
        pass
    
    @abstractmethod
    def get_best_genomes(self, count: int) -> List[DecisionGenome]:
        """Get top N genomes by fitness."""
        pass
    
    @abstractmethod
    def update_population(self, new_population: List[DecisionGenome]) -> None:
        """Replace current population with new one."""
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


class ISensoryOrgan(ABC):
    """Interface for sensory organs in the 4D+1 sensory cortex."""
    
    @abstractmethod
    async def process_market_data(self, market_data: Dict[str, Any]) -> List[SensorySignal]:
        """Process market data and return sensory signals."""
        pass
    
    @abstractmethod
    def get_organ_type(self) -> str:
        """Get the type of sensory organ."""
        pass
    
    @abstractmethod
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for the organ."""
        pass


class IRiskManager(ABC):
    """Interface for risk management."""
    
    @abstractmethod
    async def validate_position(self, position: Dict[str, Any]) -> bool:
        """Validate if position meets risk criteria."""
        pass
    
    @abstractmethod
    async def calculate_position_size(self, signal: Dict[str, Any]) -> float:
        """Calculate appropriate position size for signal."""
        pass
    
    @abstractmethod
    async def calculate_risk_metrics(self, portfolio: Dict[str, Any]) -> Dict[str, float]:
        """Calculate comprehensive risk metrics."""
        pass
    
    @abstractmethod
    async def validate_order(self, order: Dict[str, Any]) -> bool:
        """Validate order parameters."""
        pass
    
    @abstractmethod
    async def get_risk_limits(self) -> Dict[str, float]:
        """Get current risk limits."""
        pass
    
    @abstractmethod
    async def update_risk_limits(self, limits: Dict[str, float]) -> bool:
        """Update risk limits."""
        pass
    
    @abstractmethod
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for the risk manager."""
        pass


class IStrategy(ABC):
    """Interface for trading strategies."""
    
    @abstractmethod
    async def analyze(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze market data and return trading signals."""
        pass
    
    @abstractmethod
    async def generate_signals(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate trading signals from analysis."""
        pass
    
    @abstractmethod
    def get_strategy_name(self) -> str:
        """Get the name of the strategy."""
        pass
    
    @abstractmethod
    def get_parameters(self) -> Dict[str, Any]:
        """Get strategy parameters."""
        pass


class IComponentIntegrator(ABC):
    """Interface for component integration and management."""
    
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
    
    @abstractmethod
    def get_all_components(self) -> Dict[str, Any]:
        """Get all registered components."""
        pass
    
    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check."""
        pass


class IDataSource(ABC):
    """Interface for market data sources."""
    
    @abstractmethod
    async def connect(self) -> bool:
        """Connect to the data source."""
        pass
    
    @abstractmethod
    async def disconnect(self) -> bool:
        """Disconnect from the data source."""
        pass
    
    @abstractmethod
    async def get_market_data(self, symbol: str, timeframe: str) -> Dict[str, Any]:
        """Get market data for a symbol."""
        pass
    
    @abstractmethod
    async def subscribe_to_market_data(self, symbols: List[str], callback: Callable) -> bool:
        """Subscribe to real-time market data."""
        pass


class IStrategyEngine(ABC):
    """Interface for strategy engine."""
    
    @abstractmethod
    async def register_strategy(self, strategy: IStrategy) -> bool:
        """Register a new strategy."""
        pass
    
    @abstractmethod
    async def unregister_strategy(self, strategy_name: str) -> bool:
        """Unregister a strategy."""
        pass
    
    @abstractmethod
    async def execute_strategy(self, strategy_name: str, market_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Execute a specific strategy."""
        pass
    
    @abstractmethod
    def get_active_strategies(self) -> List[str]:
        """Get list of active strategies."""
        pass


class IPortfolioManager(ABC):
    """Interface for portfolio management."""
    
    @abstractmethod
    async def add_position(self, position: Dict[str, Any]) -> bool:
        """Add a new position to the portfolio."""
        pass
    
    @abstractmethod
    async def remove_position(self, position_id: str) -> bool:
        """Remove a position from the portfolio."""
        pass
    
    @abstractmethod
    async def update_position(self, position_id: str, updates: Dict[str, Any]) -> bool:
        """Update an existing position."""
        pass
    
    @abstractmethod
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get summary of the current portfolio."""
        pass
    
    @abstractmethod
    def get_positions(self) -> List[Dict[str, Any]]:
        """Get all current positions."""
        pass


class IMarketAnalyzer(ABC):
    """Interface for market analysis."""
    
    @abstractmethod
    async def analyze_market(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze market data and return analysis."""
        pass
    
    @abstractmethod
    def get_analysis_type(self) -> str:
        """Get the type of analysis performed."""
        pass


class IEvolutionEngine(ABC):
    """Interface for evolution engine."""
    
    @abstractmethod
    async def evolve_population(self, population: List[DecisionGenome]) -> List[DecisionGenome]:
        """Evolve the population using genetic algorithms."""
        pass
    
    @abstractmethod
    def get_evolution_parameters(self) -> Dict[str, Any]:
        """Get current evolution parameters."""
        pass
    
    @abstractmethod
    def update_evolution_parameters(self, parameters: Dict[str, Any]) -> bool:
        """Update evolution parameters."""
        pass


class ISelectionStrategy(ABC):
    """Interface for selection strategies in genetic algorithms."""
    
    @abstractmethod
    def select(self, population: List[DecisionGenome], count: int) -> List[DecisionGenome]:
        """Select genomes for reproduction."""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Get the name of the selection strategy."""
        pass


class ICrossoverStrategy(ABC):
    """Interface for crossover strategies in genetic algorithms."""
    
    @abstractmethod
    def crossover(self, parent1: DecisionGenome, parent2: DecisionGenome) -> DecisionGenome:
        """Perform crossover between two parents."""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Get the name of the crossover strategy."""
        pass


class IMutationStrategy(ABC):
    """Interface for mutation strategies in genetic algorithms."""
    
    @abstractmethod
    def mutate(self, genome: DecisionGenome) -> DecisionGenome:
        """Apply mutation to a genome."""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Get the name of the mutation strategy."""
        pass


class IFitnessFunction(ABC):
    """Interface for fitness functions in genetic algorithms."""
    
    @abstractmethod
    def calculate_fitness(self, genome: DecisionGenome) -> float:
        """Calculate fitness score for a genome."""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Get the name of the fitness function."""
        pass


class IFitnessEvaluator(ABC):
    """Interface for fitness evaluation in genetic algorithms."""
    
    @abstractmethod
    def evaluate(self, genome: DecisionGenome) -> float:
        """Evaluate fitness of a genome."""
        pass
    
    @abstractmethod
    def get_evaluation_metrics(self) -> Dict[str, Any]:
        """Get evaluation metrics."""
        pass


class IGenomeFactory(ABC):
    """Interface for genome factory in genetic algorithms."""
    
    @abstractmethod
    def create_genome(self) -> DecisionGenome:
        """Create a new genome."""
        pass
    
    @abstractmethod
    def create_random_genome(self) -> DecisionGenome:
        """Create a random genome."""
        pass
    
    @abstractmethod
    def create_genome_from_parents(self, parent1: DecisionGenome, parent2: DecisionGenome) -> DecisionGenome:
        """Create a genome from two parents."""
        pass


class IEvolutionLogger(ABC):
    """Interface for evolution logging."""
    
    @abstractmethod
    def log_generation(self, generation: int, population: List[DecisionGenome]) -> None:
        """Log generation information."""
        pass
    
    @abstractmethod
    def log_fitness(self, genome: DecisionGenome, fitness: float) -> None:
        """Log fitness information."""
        pass
    
    @abstractmethod
    def log_statistics(self, statistics: Dict[str, Any]) -> None:
        """Log evolution statistics."""
        pass
