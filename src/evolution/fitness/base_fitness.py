"""
Base fitness class for multi-dimensional fitness evaluation.

Provides abstract base class and common utilities for all fitness dimensions.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Optional
import numpy as np
import logging

logger = logging.getLogger(__name__)


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
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize fitness dimension with configuration."""
        self.config = config or {}
        self.name = self.__class__.__name__
        self.weight = self.config.get('weight', 1.0)
        self.min_samples = self.config.get('min_samples', 10)
        
    @abstractmethod
    def calculate_fitness(self, 
                         strategy_performance: Dict[str, Any],
                         market_data: Dict[str, Any],
                         regime_data: Dict[str, Any]) -> FitnessResult:
        """
        Calculate fitness score for this dimension.
        
        Args:
            strategy_performance: Strategy performance metrics
            market_data: Market conditions and data
            regime_data: Market regime information
            
        Returns:
            FitnessResult with normalized score and metadata
        """
        pass
    
    @abstractmethod
    def get_optimal_weight(self, market_regime: str) -> float:
        """
        Get optimal weight for this fitness dimension based on market regime.
        
        Args:
            market_regime: Current market regime
            
        Returns:
            Weight multiplier for this dimension
        """
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
            logger.warning(f"{self.name}: Missing required keys: {missing_keys}")
            return False
        
        return True
    
    @abstractmethod
    def get_required_keys(self) -> list:
        """Return list of required data keys for this fitness dimension."""
        pass
    
    def get_description(self) -> str:
        """Return description of this fitness dimension."""
        return f"{self.name}: {self.__doc__ or 'No description provided'}"
