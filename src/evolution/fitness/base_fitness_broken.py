"""
Base fitness interface for evolution engine.

Provides the interface that fitness evaluators must implement.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any
from src.core.interfaces import DecisionGenome
from src.core.market_data import MarketData


@dataclass
class FitnessResult:
    """Standardized fitness result container."""
    score: float  # 0-1 normalized score
    raw_value: float  # Original calculated value
    confidence: float  # 0-1 confidence in the score
    metadata: Dict[str, Any]  # Additional context
    weight: float  # Dynamic weight for this dimension
    
    def __post_init__(self):
        """Validate fitness result."""
        self.score = max(0.0, min(1.0, self.score))
        self.confidence = max(0.0, min(1.0, self.confidence))
        self.weight = max(0.0, min(1.0, self.weight))


class BaseFitness(ABC):
    """Abstract base class for all fitness dimensions."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize fitness dimension with configuration."""
        self.config = config or {}
        self.name = self.__class__.__name__
        self.weight = self.config.get('weight', 1.0)
        self.min_samples = self.config.get('min_samples', 10)
        
    @abstractmethod
    def calculate_fitness(self, 
                         strategy_performance: Dict[str, Any],
                         market_data: Dict[str, Any],
                          def calculate_fitness(self, genome) -> float:
        """Calculate fitness score for this dimension."""
        try:
            # Get required data
            data = self.get_required_data()
            if not data:
                logger.warning(f"No data available for fitness calculation in {self.__class__.__name__}")
                return 0.0
            
            # Calculate base fitness score
            base_score = self._calculate_base_score(genome, data)
            
            # Apply market regime adjustment
            regime_weight = self.get_market_regime_weight(data.get('market_regime', 'neutral'))
            adjusted_score = base_score * regime_weight
            
            # Apply normalization
            normalized_score = self._normalize_score(adjusted_score)
            
            logger.debug(f"Fitness calculated for {genome.id}: {normalized_score:.4f}")
            return normalized_score
            
        except Exception as e:
            logger.error(f"Fitness calculation failed for {genome.id}: {e}")
            return 0.0
    
    @abstractmethod
    def get_optimal_weight(self, market_regime: str) -> float:
        """Get optimal weight for this fitness dimension based on market regime."""
        pass
    
    def normalize_score(self, raw_score: float, 
                       min_val: float, max_val: float) -> float:
        """Normalize raw score to 0-1 range."""
        if max_val == min_val:
            return 0.5
        
        normalized = (raw_score - min_val) / (max_val - min_val)
        return max(0.0, min(1.0, normalized))
    
    def calculate_confidence(self, sample_size: int, 
                           min_required: int = None) -> float:
        """Calculate confidence based on sample size."""
        min_samples = min_required or self.min_samples
        if sample_size >= min_samples:
            return 1.0
        return min(1.0, sample_size / min_samples)
    
    def validate_inputs(self, data: Dict[str, Any]) -> bool:
        """Validate input data completeness."""
        required_keys = self.get_required_keys()
        missing_keys = [key for key in required_keys if key not in data]
        
        if missing_keys:
            return False
        
        return True
    
    @abstractmethod
    def get_required_keys(self) -> list:
        """Return list of required data keys for this fitness dimension."""
        pass
    
    def get_description(self) -> str:
        """Return description of this fitness dimension."""
        return f"{self.name}: {self.__doc__ or 'No description provided'}"


class IFitnessEvaluator(ABC):
    """Interface for fitness evaluators used by the evolution engine."""
    
    @abstractmethod
    def evaluate(self, genome: DecisionGenome, market_data: MarketData) -> float:
        """
        Evaluate the fitness of a genome based on market data.
        
        Args:
            genome: The decision genome to evaluate
            market_data: Market data for evaluation
            
        Returns:
            Fitness score (higher is better)
        """
        pass


class MockFitnessEvaluator(IFitnessEvaluator):
    """Mock fitness evaluator for testing purposes."""
    
    def evaluate(self, genome: DecisionGenome, market_data: MarketData) -> float:
        """Return a random fitness score for testing."""
        import random
        return random.random()
