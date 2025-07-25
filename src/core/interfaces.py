#!/usr/bin/env python3
"""
Core Interfaces for the EMP Trading System
==========================================

This module defines the abstract base classes and interfaces that form the
contract between different components of the EMP system.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
import numpy as np
from pydantic import BaseModel, Field
import uuid


class IStrategy(ABC):
    """Interface for trading strategies."""
    
    @abstractmethod
    async def analyze(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze market data and return trading signals."""
        pass
    
    @abstractmethod
    async def generate_signal(self, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Generate trading signal based on current context."""
        pass
    
    @abstractmethod
    def get_parameters(self) -> Dict[str, Any]:
        """Get current strategy parameters."""
        pass
    
    @abstractmethod
    def set_parameters(self, parameters: Dict[str, Any]) -> None:
        """Set strategy parameters."""
        pass


class IDataSource(ABC):
    """Interface for market data sources."""
    
    @abstractmethod
    async def get_data(self, symbol: str, start: datetime, end: datetime) -> Dict[str, Any]:
        """Retrieve market data for given symbol and time range."""
        pass
    
    @abstractmethod
    async def get_latest(self, symbol: str) -> Dict[str, Any]:
        """Get latest market data for symbol."""
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


class IEvolutionEngine(ABC):
    """Interface for genetic evolution engines."""
    
    @abstractmethod
    async def evolve(self, population: List[Any], fitness_function) -> List[Any]:
        """Evolve population using genetic algorithms."""
        pass
    
    @abstractmethod
    async def evaluate_fitness(self, individual: Any) -> float:
        """Evaluate fitness of an individual."""
        pass


class IPortfolioManager(ABC):
    """Interface for portfolio management."""
    
    @abstractmethod
    async def add_strategy(self, strategy: IStrategy) -> None:
        """Add strategy to portfolio."""
        pass
    
    @abstractmethod
    async def remove_strategy(self, strategy_id: str) -> None:
        """Remove strategy from portfolio."""
        pass
    
    @abstractmethod
    async def rebalance(self) -> None:
        """Rebalance portfolio allocations."""
        pass


class IMarketAnalyzer(ABC):
    """Interface for market analysis."""
    
    @abstractmethod
    async def analyze_regime(self, market_data: Dict[str, Any]) -> str:
        """Determine current market regime."""
        pass
    
    @abstractmethod
    async def calculate_volatility(self, market_data: Dict[str, Any]) -> float:
        """Calculate market volatility."""
        pass
    
    @abstractmethod
    async def detect_trend(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Detect market trend characteristics."""
        pass


class IPatternMemory(ABC):
    """Interface for pattern memory systems."""
    
    @abstractmethod
    async def store_pattern(self, pattern: Dict[str, Any], outcome: Dict[str, Any]) -> None:
        """Store pattern with associated outcome."""
        pass
    
    @abstractmethod
    async def find_similar(self, pattern: Dict[str, Any], limit: int = 5) -> List[Dict[str, Any]]:
        """Find similar patterns in memory."""
        pass


class IPredatorStrategy(ABC):
    """Interface for specialized predator strategies."""
    
    @abstractmethod
    def get_species_type(self) -> str:
        """Return the species type of this predator."""
        pass
    
    @abstractmethod
    def get_preferred_regimes(self) -> List[str]:
        """Return list of market regimes this predator prefers."""
        pass
    
    @abstractmethod
    async def hunt(self, market_data: Dict[str, Any], regime: str) -> Optional[Dict[str, Any]]:
        """Hunt for opportunities in given regime."""
        pass


class ISpecialistGenomeFactory(ABC):
    """Factory interface for creating specialist genomes with species-specific biases."""
    
    @abstractmethod
    def create_genome(self) -> 'DecisionGenome':
        """Create a genome with bias towards specific strategy type."""
        pass
    
    @abstractmethod
    def get_species_name(self) -> str:
        """Return the species name this factory creates."""
        pass
    
    @abstractmethod
    def get_parameter_ranges(self) -> Dict[str, Tuple[float, float]]:
        """Return parameter ranges specific to this species."""
        pass


class ICoordinationEngine(ABC):
    """Interface for coordinating multiple predator strategies."""
    
    @abstractmethod
    async def resolve_intents(self, intents: List[Dict[str, Any]], market_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Resolve conflicting intents from multiple strategies."""
        pass
    
    @abstractmethod
    async def prioritize_strategies(self, strategies: List[IPredatorStrategy], regime: str) -> List[IPredatorStrategy]:
        """Prioritize strategies based on current market regime."""
        pass


class INicheDetector(ABC):
    """Interface for detecting market niches and regimes."""
    
    @abstractmethod
    async def detect_niches(self, market_data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Detect and segment market into different niches."""
        pass
    
    @abstractmethod
    async def classify_regime(self, market_data: Dict[str, Any]) -> str:
        """Classify current market regime."""
        pass


class IPortfolioFitnessEvaluator(ABC):
    """Interface for evaluating portfolio-level fitness."""
    
    @abstractmethod
    async def evaluate_portfolio(self, portfolio: 'PortfolioGenome', niche_data: Dict[str, Any]) -> float:
        """Evaluate fitness of entire portfolio across niches."""
        pass
    
    @abstractmethod
    async def calculate_correlation_score(self, portfolio: 'PortfolioGenome') -> float:
        """Calculate diversification score for portfolio."""
        pass


# Data Models
class DecisionGenome(BaseModel):
    """Represents a trading strategy genome."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    parameters: Dict[str, Any] = Field(default_factory=dict)
    indicators: List[str] = Field(default_factory=list)
    rules: Dict[str, Any] = Field(default_factory=dict)
    risk_profile: Dict[str, float] = Field(default_factory=dict)
    species_type: str = Field(default="generic")
    generation: int = Field(default=0)
    parent_ids: List[str] = Field(default_factory=list)
    
    def mutate(self, mutation_rate: float = 0.1) -> 'DecisionGenome':
        """Apply mutation to genome."""
        import random
        mutated = self.model_copy()
        
        # Mutate parameters
        for key, value in mutated.parameters.items():
            if random.random() < mutation_rate:
                if isinstance(value, (int, float)):
                    mutated.parameters[key] = value * (1 + random.uniform(-0.2, 0.2))
        
        return mutated
    
    def crossover(self, other: 'DecisionGenome') -> 'DecisionGenome':
        """Perform crossover with another genome."""
        import random
        child = self.model_copy()
        child.id = str(uuid.uuid4())
        child.parent_ids = [self.id, other.id]
        child.generation = max(self.generation, other.generation) + 1
        
        # Crossover parameters
        for key in child.parameters:
            if key in other.parameters and random.random() < 0.5:
                child.parameters[key] = other.parameters[key]
        
        return child


class PortfolioGenome(BaseModel):
    """Represents a portfolio of specialized predator strategies."""
    id: str = Field(default_factory=lambda: f"port_{uuid.uuid4().hex[:8]}")
    species: Dict[str, DecisionGenome] = Field(default_factory=dict)
    fitness: float = Field(default=0.0)
    correlation_score: float = Field(default=0.0)
    diversification_score: float = Field(default=0.0)
    created_at: datetime = Field(default_factory=datetime.now)
    generation: int = Field(default=0)
    
    def add_species(self, species_type: str, genome: DecisionGenome) -> None:
        """Add a specialist species to the portfolio."""
        self.species[species_type] = genome
    
    def remove_species(self, species_type: str) -> None:
        """Remove a specialist species from the portfolio."""
        if species_type in self.species:
            del self.species[species_type]
    
    def get_species_list(self) -> List[str]:
        """Get list of species in portfolio."""
        return list(self.species.keys())


class TradeIntent(BaseModel):
    """Represents a trading intent from a predator strategy."""
    strategy_id: str
    species_type: str
    symbol: str
    direction: str  # BUY, SELL, HOLD
    confidence: float
    size: float
    timestamp: datetime = Field(default_factory=datetime.now)
    regime: str
    priority: int = Field(default=5)  # 1-10 scale


class MarketContext(BaseModel):
    """Represents current market context for coordination."""
    symbol: str
    regime: str
    volatility: float
    trend_strength: float
    volume_anomaly: float
    timestamp: datetime = Field(default_factory=datetime.now)


# Event Types
class EventType:
    """Event types for the EMP system."""
    MARKET_DATA = "market_data"
    TRADE_SIGNAL = "trade_signal"
    PORTFOLIO_UPDATE = "portfolio_update"
    REGIME_CHANGE = "regime_change"
    STRATEGY_ALERT = "strategy_alert"
    COORDINATION_REQUEST = "coordination_request"


# Additional classes for thinking patterns
class ThinkingPattern(BaseModel):
    """Represents a thinking pattern for market analysis."""
    pattern_id: str
    pattern_type: str
    confidence: float
    parameters: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.now)


class SensorySignal(BaseModel):
    """Represents a signal from sensory analysis."""
    signal_type: str
    strength: float
    confidence: float
    source: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.now)


class AnalysisResult(BaseModel):
    """Represents the result of market analysis."""
    analysis_id: str
    analysis_type: str
    symbol: str
    confidence: float
    results: Dict[str, Any]
    signals: List[SensorySignal] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.now)
