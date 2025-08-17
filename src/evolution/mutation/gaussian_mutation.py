"""
EMP Gaussian Mutation v1.1

Gaussian mutation strategy implementation for genetic algorithms.
Implements the IMutationStrategy interface for introducing genetic diversity.
"""

import logging
import random

from src.core.interfaces import DecisionGenome, IMutationStrategy

logger = logging.getLogger(__name__)


class GaussianMutation(IMutationStrategy):
    """
    Gaussian mutation strategy for genetic algorithms.
    
    Applies Gaussian noise to numeric parameters with configurable standard deviation.
    """
    
    def __init__(self, mutation_strength: float = 0.1):
        """
        Initialize Gaussian mutation.
        
        Args:
            mutation_strength: Standard deviation for Gaussian noise (as fraction of parameter range)
        """
        self.mutation_strength = max(0.01, min(1.0, mutation_strength))
        logger.info(f"GaussianMutation initialized with strength {mutation_strength}")
    
    def mutate(self, genome: DecisionGenome, mutation_rate: float) -> DecisionGenome:
        """
        Apply Gaussian mutation to a genome.
        
        Args:
            genome: Genome to mutate
            mutation_rate: Probability of mutation for each gene
            
        Returns:
            Mutated genome
        """
        import copy

        # Preserve original behavior: no validation/exceptions/logging changes
        self._validate_inputs(genome, mutation_rate)
        
        # Create a deep copy to avoid modifying the original
        mutated = copy.deepcopy(genome)
        
        # Update genome ID and mutation count (ordering preserved)
        mutated.genome_id = f"{genome.genome_id}_mutated_{random.randint(1000, 9999)}"
        mutated.mutation_count = genome.mutation_count + 1
        
        # Mutate strategy parameters (order preserved)
        self._mutate_strategy(mutated, mutation_rate)
        
        # Mutate risk parameters (order preserved)
        self._mutate_risk(mutated, mutation_rate)
        
        # Mutate timing parameters (order preserved)
        self._mutate_timing(mutated, mutation_rate)
        
        # Mutate sensory and thinking weights (ensure they sum to 1.0) - order preserved
        sensory_weights = ['price_weight', 'volume_weight', 'orderbook_weight',
                           'news_weight', 'sentiment_weight', 'economic_weight']
        thinking_weights = ['trend_analysis_weight', 'risk_analysis_weight',
                           'performance_analysis_weight', 'pattern_recognition_weight']
        mutated_any = False
        mutated_any |= self._mutate_weight_block(mutated.sensory, sensory_weights, mutation_rate)
        mutated_any |= self._mutate_weight_block(mutated.thinking, thinking_weights, mutation_rate)
        
        # Normalize weights after mutation
        mutated._normalize_weights()
        
        return mutated

    # --- Private helpers extracted to reduce mutate() complexity while preserving behavior ---

    def _validate_inputs(self, genome: DecisionGenome, mutation_rate: float) -> None:
        # Original mutate() performed no validation; keep behavior identical.
        # This hook exists for readability and future-proofing without altering behavior.
        return None

    def _should_mutate(self, mutation_rate: float) -> bool:
        # Keep RNG source and call count semantics identical to original
        return random.random() < mutation_rate

    def _maybe_mutate_attr(self, obj, attr: str, min_val: float, max_val: float, mutation_rate: float) -> bool:
        """
        Mutate a single attribute on an object with per-attribute probability.
        Returns True if mutation occurred (for parity with original mutated_any flag usage).
        """
        if self._should_mutate(mutation_rate):
            setattr(obj, attr, self._mutate_parameter(getattr(obj, attr), min_val, max_val))
            return True
        return False

    def _mutate_weight_block(self, container, names, mutation_rate: float) -> bool:
        """
        Mutate a block of weight attributes on a container with consistent bounds [0.0, 1.0].
        Returns True if any weight mutated.
        """
        any_mutated = False
        for name in names:
            if self._should_mutate(mutation_rate):
                setattr(
                    container,
                    name,
                    self._mutate_parameter(getattr(container, name), 0.0, 1.0),
                )
                any_mutated = True
        return any_mutated

    def _mutate_strategy(self, mutated: DecisionGenome, mutation_rate: float) -> None:
        self._maybe_mutate_attr(mutated.strategy, "entry_threshold", 0.0, 1.0, mutation_rate)
        self._maybe_mutate_attr(mutated.strategy, "exit_threshold", 0.0, 1.0, mutation_rate)
        self._maybe_mutate_attr(mutated.strategy, "momentum_weight", 0.0, 1.0, mutation_rate)
        self._maybe_mutate_attr(mutated.strategy, "trend_weight", 0.0, 1.0, mutation_rate)
        self._maybe_mutate_attr(mutated.strategy, "volume_weight", 0.0, 1.0, mutation_rate)
        self._maybe_mutate_attr(mutated.strategy, "sentiment_weight", 0.0, 1.0, mutation_rate)
        if self._should_mutate(mutation_rate):
            mutated.strategy.lookback_period = int(self._mutate_parameter(
                mutated.strategy.lookback_period, 1, 100
            ))

    def _mutate_risk(self, mutated: DecisionGenome, mutation_rate: float) -> None:
        self._maybe_mutate_attr(mutated.risk, "risk_tolerance", 0.0, 1.0, mutation_rate)
        self._maybe_mutate_attr(mutated.risk, "position_size_multiplier", 0.1, 5.0, mutation_rate)
        self._maybe_mutate_attr(mutated.risk, "stop_loss_threshold", 0.001, 0.1, mutation_rate)
        self._maybe_mutate_attr(mutated.risk, "take_profit_threshold", 0.001, 0.2, mutation_rate)
        self._maybe_mutate_attr(mutated.risk, "max_drawdown_limit", 0.01, 0.5, mutation_rate)
        self._maybe_mutate_attr(mutated.risk, "volatility_threshold", 0.01, 1.0, mutation_rate)
        self._maybe_mutate_attr(mutated.risk, "correlation_threshold", 0.0, 1.0, mutation_rate)

    def _mutate_timing(self, mutated: DecisionGenome, mutation_rate: float) -> None:
        if self._should_mutate(mutation_rate):
            mutated.timing.holding_period_min = max(0, int(self._mutate_parameter(
                mutated.timing.holding_period_min, 0, 10
            )))
        if self._should_mutate(mutation_rate):
            mutated.timing.holding_period_max = max(
                mutated.timing.holding_period_min + 1,
                int(self._mutate_parameter(mutated.timing.holding_period_max, 1, 100))
            )
        if self._should_mutate(mutation_rate):
            mutated.timing.reentry_delay = max(0, int(self._mutate_parameter(
                mutated.timing.reentry_delay, 0, 20
            )))
    
    def _mutate_parameter(self, value: float, min_val: float, max_val: float) -> float:
        """
        Apply Gaussian mutation to a single parameter.
        
        Args:
            value: Current parameter value
            min_val: Minimum allowed value
            max_val: Maximum allowed value
            
        Returns:
            Mutated parameter value
        """
        if min_val >= max_val:
            return value
            
        # Calculate mutation range based on parameter range
        range_size = max_val - min_val
        std_dev = range_size * self.mutation_strength
        
        # Apply Gaussian noise
        mutated_value = value + random.gauss(0, std_dev)
        
        # Clamp to valid range
        return max(min_val, min(max_val, mutated_value))
    
    @property
    def name(self) -> str:
        """Return the name of this mutation strategy."""
        return f"GaussianMutation(strength={self.mutation_strength})"
    
    def __repr__(self) -> str:
        """String representation of the mutation strategy."""
        return f"GaussianMutation(mutation_strength={self.mutation_strength})"
