"""
EMP Uniform Crossover v1.1

Uniform crossover strategy implementation for genetic algorithms.
Implements the ICrossoverStrategy interface for creating offspring.
"""

import logging
import random
from typing import Dict, Any

from src.core.interfaces import ICrossoverStrategy
from src.genome.models.genome import DecisionGenome

logger = logging.getLogger(__name__)


class UniformCrossover(ICrossoverStrategy):
    """
    Uniform crossover strategy for genetic algorithms.
    
    Creates offspring by randomly selecting genes from each parent with equal probability.
    """
    
    def __init__(self, crossover_rate: float = 0.8):
        """
        Initialize uniform crossover.
        
        Args:
            crossover_rate: Probability of performing crossover (vs cloning)
        """
        self.crossover_rate = max(0.0, min(1.0, crossover_rate))
        logger.info(f"UniformCrossover initialized with rate {crossover_rate}")
    
    def crossover(self, parent1: DecisionGenome, parent2: DecisionGenome) -> tuple[DecisionGenome, DecisionGenome]:
        """
        Perform uniform crossover between two parent genomes.
        
        Args:
            parent1: First parent genome
            parent2: Second parent genome
            
        Returns:
            Tuple of two child genomes
        """
        # Create deep copies of parents
        child1 = self._create_child(parent1, parent2)
        child2 = self._create_child(parent2, parent1)
        
        logger.debug("Uniform crossover completed")
        
        return child1, child2
    
    def _create_child(self, parent1: DecisionGenome, parent2: DecisionGenome) -> DecisionGenome:
        """Create a child genome using uniform crossover."""
        import copy
        
        # Create a deep copy of parent1 as base
        child = copy.deepcopy(parent1)
        
        # Update genome ID
        child.genome_id = f"child_{parent1.genome_id}_{parent2.genome_id}_{random.randint(1000, 9999)}"
        child.crossover_count = parent1.crossover_count + 1
        
        # Perform uniform crossover on strategy parameters
        if random.random() < 0.5:
            child.strategy.strategy_type = parent2.strategy.strategy_type
        if random.random() < 0.5:
            child.strategy.entry_threshold = parent2.strategy.entry_threshold
        if random.random() < 0.5:
            child.strategy.exit_threshold = parent2.strategy.exit_threshold
        if random.random() < 0.5:
            child.strategy.momentum_weight = parent2.strategy.momentum_weight
        if random.random() < 0.5:
            child.strategy.trend_weight = parent2.strategy.trend_weight
        if random.random() < 0.5:
            child.strategy.volume_weight = parent2.strategy.volume_weight
        if random.random() < 0.5:
            child.strategy.sentiment_weight = parent2.strategy.sentiment_weight
        if random.random() < 0.5:
            child.strategy.lookback_period = parent2.strategy.lookback_period
        
        # Perform uniform crossover on risk parameters
        if random.random() < 0.5:
            child.risk.risk_tolerance = parent2.risk.risk_tolerance
        if random.random() < 0.5:
            child.risk.position_size_multiplier = parent2.risk.position_size_multiplier
        if random.random() < 0.5:
            child.risk.stop_loss_threshold = parent2.risk.stop_loss_threshold
        if random.random() < 0.5:
            child.risk.take_profit_threshold = parent2.risk.take_profit_threshold
        if random.random() < 0.5:
            child.risk.max_drawdown_limit = parent2.risk.max_drawdown_limit
        if random.random() < 0.5:
            child.risk.volatility_threshold = parent2.risk.volatility_threshold
        if random.random() < 0.5:
            child.risk.correlation_threshold = parent2.risk.correlation_threshold
        
        # Perform uniform crossover on timing parameters
        if random.random() < 0.5:
            child.timing.entry_timing = parent2.timing.entry_timing
        if random.random() < 0.5:
            child.timing.exit_timing = parent2.timing.exit_timing
        if random.random() < 0.5:
            child.timing.holding_period_min = parent2.timing.holding_period_min
        if random.random() < 0.5:
            child.timing.holding_period_max = parent2.timing.holding_period_max
        if random.random() < 0.5:
            child.timing.reentry_delay = parent2.timing.reentry_delay
        
        # Perform uniform crossover on sensory weights
        sensory_weights = ['price_weight', 'volume_weight', 'orderbook_weight', 
                          'news_weight', 'sentiment_weight', 'economic_weight']
        for weight_name in sensory_weights:
            if random.random() < 0.5:
                setattr(child.sensory, weight_name, getattr(parent2.sensory, weight_name))
        
        # Perform uniform crossover on thinking weights
        thinking_weights = ['trend_analysis_weight', 'risk_analysis_weight', 
                           'performance_analysis_weight', 'pattern_recognition_weight']
        for weight_name in thinking_weights:
            if random.random() < 0.5:
                setattr(child.thinking, weight_name, getattr(parent2.thinking, weight_name))
        
        # Ensure weights are normalized
        child._normalize_weights()
        
        return child
    
    @property
    def name(self) -> str:
        """Return the name of this crossover strategy."""
        return f"UniformCrossover(rate={self.crossover_rate})"
    
    def __repr__(self) -> str:
        """String representation of the crossover strategy."""
        return f"UniformCrossover(crossover_rate={self.crossover_rate})"
