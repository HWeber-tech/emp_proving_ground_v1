"""
EMP Gaussian Mutation v1.1

Gaussian mutation strategy implementation for genetic algorithms.
Implements the IMutationStrategy interface for introducing genetic diversity.
"""

from __future__ import annotations

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

        # Create a deep copy to avoid modifying the original
        mutated = copy.deepcopy(genome)

        # Update genome ID and mutation count
        mutated.genome_id = f"{genome.genome_id}_mutated_{random.randint(1000, 9999)}"
        mutated.mutation_count = genome.mutation_count + 1

        # Mutate strategy parameters
        if random.random() < mutation_rate:
            mutated.strategy.entry_threshold = self._mutate_parameter(
                mutated.strategy.entry_threshold, 0.0, 1.0
            )
        if random.random() < mutation_rate:
            mutated.strategy.exit_threshold = self._mutate_parameter(
                mutated.strategy.exit_threshold, 0.0, 1.0
            )
        if random.random() < mutation_rate:
            mutated.strategy.momentum_weight = self._mutate_parameter(
                mutated.strategy.momentum_weight, 0.0, 1.0
            )
        if random.random() < mutation_rate:
            mutated.strategy.trend_weight = self._mutate_parameter(
                mutated.strategy.trend_weight, 0.0, 1.0
            )
        if random.random() < mutation_rate:
            mutated.strategy.volume_weight = self._mutate_parameter(
                mutated.strategy.volume_weight, 0.0, 1.0
            )
        if random.random() < mutation_rate:
            mutated.strategy.sentiment_weight = self._mutate_parameter(
                mutated.strategy.sentiment_weight, 0.0, 1.0
            )
        if random.random() < mutation_rate:
            mutated.strategy.lookback_period = int(
                self._mutate_parameter(mutated.strategy.lookback_period, 1, 100)
            )

        # Mutate risk parameters
        if random.random() < mutation_rate:
            mutated.risk.risk_tolerance = self._mutate_parameter(
                mutated.risk.risk_tolerance, 0.0, 1.0
            )
        if random.random() < mutation_rate:
            mutated.risk.position_size_multiplier = self._mutate_parameter(
                mutated.risk.position_size_multiplier, 0.1, 5.0
            )
        if random.random() < mutation_rate:
            mutated.risk.stop_loss_threshold = self._mutate_parameter(
                mutated.risk.stop_loss_threshold, 0.001, 0.1
            )
        if random.random() < mutation_rate:
            mutated.risk.take_profit_threshold = self._mutate_parameter(
                mutated.risk.take_profit_threshold, 0.001, 0.2
            )
        if random.random() < mutation_rate:
            mutated.risk.max_drawdown_limit = self._mutate_parameter(
                mutated.risk.max_drawdown_limit, 0.01, 0.5
            )
        if random.random() < mutation_rate:
            mutated.risk.volatility_threshold = self._mutate_parameter(
                mutated.risk.volatility_threshold, 0.01, 1.0
            )
        if random.random() < mutation_rate:
            mutated.risk.correlation_threshold = self._mutate_parameter(
                mutated.risk.correlation_threshold, 0.0, 1.0
            )

        # Mutate timing parameters
        if random.random() < mutation_rate:
            mutated.timing.holding_period_min = max(
                0, int(self._mutate_parameter(mutated.timing.holding_period_min, 0, 10))
            )
        if random.random() < mutation_rate:
            mutated.timing.holding_period_max = max(
                mutated.timing.holding_period_min + 1,
                int(self._mutate_parameter(mutated.timing.holding_period_max, 1, 100)),
            )
        if random.random() < mutation_rate:
            mutated.timing.reentry_delay = max(
                0, int(self._mutate_parameter(mutated.timing.reentry_delay, 0, 20))
            )

        # Mutate sensory weights (ensure they sum to 1.0)
        sensory_weights = [
            "price_weight",
            "volume_weight",
            "orderbook_weight",
            "news_weight",
            "sentiment_weight",
            "economic_weight",
        ]
        mutated_any = False
        for weight_name in sensory_weights:
            if random.random() < mutation_rate:
                setattr(
                    mutated.sensory,
                    weight_name,
                    self._mutate_parameter(getattr(mutated.sensory, weight_name), 0.0, 1.0),
                )
                mutated_any = True

        # Mutate thinking weights (ensure they sum to 1.0)
        thinking_weights = [
            "trend_analysis_weight",
            "risk_analysis_weight",
            "performance_analysis_weight",
            "pattern_recognition_weight",
        ]
        for weight_name in thinking_weights:
            if random.random() < mutation_rate:
                setattr(
                    mutated.thinking,
                    weight_name,
                    self._mutate_parameter(getattr(mutated.thinking, weight_name), 0.0, 1.0),
                )
                mutated_any = True

        # Normalize weights after mutation
        mutated._normalize_weights()

        return mutated

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
