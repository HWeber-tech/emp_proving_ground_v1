"""
EMP Genome Model v1.1

Defines the genetic encoding structure for trading strategies,
risk parameters, and timing preferences in the adaptive core.
"""

import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum

from src.core.exceptions import GenomeException

logger = logging.getLogger(__name__)


class GenomeType(Enum):
    """Types of genome components."""
    STRATEGY = "strategy"
    RISK = "risk"
    TIMING = "timing"
    SENSORY = "sensory"
    THINKING = "thinking"


@dataclass
class StrategyGenome:
    """Strategy-related genetic parameters."""
    strategy_type: str
    entry_threshold: float
    exit_threshold: float
    momentum_weight: float
    trend_weight: float
    volume_weight: float
    sentiment_weight: float
    lookback_period: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RiskGenome:
    """Risk-related genetic parameters."""
    risk_tolerance: float
    position_size_multiplier: float
    stop_loss_threshold: float
    take_profit_threshold: float
    max_drawdown_limit: float
    volatility_threshold: float
    correlation_threshold: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TimingGenome:
    """Timing-related genetic parameters."""
    entry_timing: str  # 'immediate', 'confirmation', 'pullback'
    exit_timing: str   # 'immediate', 'trailing', 'time_based'
    holding_period_min: int
    holding_period_max: int
    reentry_delay: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SensoryGenome:
    """Sensory-related genetic parameters."""
    price_weight: float
    volume_weight: float
    orderbook_weight: float
    news_weight: float
    sentiment_weight: float
    economic_weight: float
    signal_thresholds: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ThinkingGenome:
    """Thinking-related genetic parameters."""
    trend_analysis_weight: float
    risk_analysis_weight: float
    performance_analysis_weight: float
    pattern_recognition_weight: float
    inference_confidence_threshold: float
    analysis_methods: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DecisionGenome:
    """Complete decision genome for trading strategies."""
    genome_id: str
    version: str = "1.1.0"
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    # Genome components
    strategy: StrategyGenome = field(default_factory=lambda: StrategyGenome(
        strategy_type="trend_following",
        entry_threshold=0.5,
        exit_threshold=0.5,
        momentum_weight=0.3,
        trend_weight=0.4,
        volume_weight=0.2,
        sentiment_weight=0.1,
        lookback_period=20
    ))
    
    risk: RiskGenome = field(default_factory=lambda: RiskGenome(
        risk_tolerance=0.5,
        position_size_multiplier=1.0,
        stop_loss_threshold=0.02,
        take_profit_threshold=0.04,
        max_drawdown_limit=0.15,
        volatility_threshold=0.3,
        correlation_threshold=0.7
    ))
    
    timing: TimingGenome = field(default_factory=lambda: TimingGenome(
        entry_timing="confirmation",
        exit_timing="trailing",
        holding_period_min=1,
        holding_period_max=30,
        reentry_delay=5
    ))
    
    sensory: SensoryGenome = field(default_factory=lambda: SensoryGenome(
        price_weight=0.4,
        volume_weight=0.2,
        orderbook_weight=0.2,
        news_weight=0.1,
        sentiment_weight=0.05,
        economic_weight=0.05
    ))
    
    thinking: ThinkingGenome = field(default_factory=lambda: ThinkingGenome(
        trend_analysis_weight=0.4,
        risk_analysis_weight=0.3,
        performance_analysis_weight=0.2,
        pattern_recognition_weight=0.1,
        inference_confidence_threshold=0.7
    ))
    
    # Evolution metadata
    fitness_score: float = 0.0
    generation: int = 0
    parent_ids: List[str] = field(default_factory=list)
    mutation_count: int = 0
    crossover_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert genome to dictionary for serialization."""
        return {
            'genome_id': self.genome_id,
            'version': self.version,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'strategy': {
                'strategy_type': self.strategy.strategy_type,
                'entry_threshold': self.strategy.entry_threshold,
                'exit_threshold': self.strategy.exit_threshold,
                'momentum_weight': self.strategy.momentum_weight,
                'trend_weight': self.strategy.trend_weight,
                'volume_weight': self.strategy.volume_weight,
                'sentiment_weight': self.strategy.sentiment_weight,
                'lookback_period': self.strategy.lookback_period,
                'metadata': self.strategy.metadata
            },
            'risk': {
                'risk_tolerance': self.risk.risk_tolerance,
                'position_size_multiplier': self.risk.position_size_multiplier,
                'stop_loss_threshold': self.risk.stop_loss_threshold,
                'take_profit_threshold': self.risk.take_profit_threshold,
                'max_drawdown_limit': self.risk.max_drawdown_limit,
                'volatility_threshold': self.risk.volatility_threshold,
                'correlation_threshold': self.risk.correlation_threshold,
                'metadata': self.risk.metadata
            },
            'timing': {
                'entry_timing': self.timing.entry_timing,
                'exit_timing': self.timing.exit_timing,
                'holding_period_min': self.timing.holding_period_min,
                'holding_period_max': self.timing.holding_period_max,
                'reentry_delay': self.timing.reentry_delay,
                'metadata': self.timing.metadata
            },
            'sensory': {
                'price_weight': self.sensory.price_weight,
                'volume_weight': self.sensory.volume_weight,
                'orderbook_weight': self.sensory.orderbook_weight,
                'news_weight': self.sensory.news_weight,
                'sentiment_weight': self.sensory.sentiment_weight,
                'economic_weight': self.sensory.economic_weight,
                'signal_thresholds': self.sensory.signal_thresholds,
                'metadata': self.sensory.metadata
            },
            'thinking': {
                'trend_analysis_weight': self.thinking.trend_analysis_weight,
                'risk_analysis_weight': self.thinking.risk_analysis_weight,
                'performance_analysis_weight': self.thinking.performance_analysis_weight,
                'pattern_recognition_weight': self.thinking.pattern_recognition_weight,
                'inference_confidence_threshold': self.thinking.inference_confidence_threshold,
                'analysis_methods': self.thinking.analysis_methods,
                'metadata': self.thinking.metadata
            },
            'fitness_score': self.fitness_score,
            'generation': self.generation,
            'parent_ids': self.parent_ids,
            'mutation_count': self.mutation_count,
            'crossover_count': self.crossover_count
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DecisionGenome':
        """Create genome from dictionary."""
        try:
            # Create genome components
            strategy = StrategyGenome(**data.get('strategy', {}))
            risk = RiskGenome(**data.get('risk', {}))
            timing = TimingGenome(**data.get('timing', {}))
            sensory = SensoryGenome(**data.get('sensory', {}))
            thinking = ThinkingGenome(**data.get('thinking', {}))
            
            # Create genome
            genome = cls(
                genome_id=data['genome_id'],
                version=data.get('version', '1.1.0'),
                created_at=datetime.fromisoformat(data.get('created_at', datetime.now().isoformat())),
                updated_at=datetime.fromisoformat(data.get('updated_at', datetime.now().isoformat())),
                strategy=strategy,
                risk=risk,
                timing=timing,
                sensory=sensory,
                thinking=thinking,
                fitness_score=data.get('fitness_score', 0.0),
                generation=data.get('generation', 0),
                parent_ids=data.get('parent_ids', []),
                mutation_count=data.get('mutation_count', 0),
                crossover_count=data.get('crossover_count', 0)
            )
            
            return genome
            
        except Exception as e:
            raise GenomeException(f"Error creating genome from dict: {e}")
    
    def validate(self) -> bool:
        """Validate genome parameters."""
        try:
            # Validate strategy parameters
            if not (0 <= self.strategy.entry_threshold <= 1):
                raise GenomeException("Entry threshold must be between 0 and 1")
            if not (0 <= self.strategy.exit_threshold <= 1):
                raise GenomeException("Exit threshold must be between 0 and 1")
            if self.strategy.lookback_period < 1:
                raise GenomeException("Lookback period must be positive")
                
            # Validate risk parameters
            if not (0 <= self.risk.risk_tolerance <= 1):
                raise GenomeException("Risk tolerance must be between 0 and 1")
            if self.risk.position_size_multiplier < 0:
                raise GenomeException("Position size multiplier must be non-negative")
            if self.risk.max_drawdown_limit < 0 or self.risk.max_drawdown_limit > 1:
                raise GenomeException("Max drawdown limit must be between 0 and 1")
                
            # Validate timing parameters
            if self.timing.holding_period_min < 0:
                raise GenomeException("Minimum holding period must be non-negative")
            if self.timing.holding_period_max < self.timing.holding_period_min:
                raise GenomeException("Maximum holding period must be >= minimum")
                
            # Validate sensory weights sum to 1
            sensory_weights = [
                self.sensory.price_weight,
                self.sensory.volume_weight,
                self.sensory.orderbook_weight,
                self.sensory.news_weight,
                self.sensory.sentiment_weight,
                self.sensory.economic_weight
            ]
            if abs(sum(sensory_weights) - 1.0) > 0.01:
                raise GenomeException("Sensory weights must sum to 1.0")
                
            # Validate thinking weights sum to 1
            thinking_weights = [
                self.thinking.trend_analysis_weight,
                self.thinking.risk_analysis_weight,
                self.thinking.performance_analysis_weight,
                self.thinking.pattern_recognition_weight
            ]
            if abs(sum(thinking_weights) - 1.0) > 0.01:
                raise GenomeException("Thinking weights must sum to 1.0")
                
            return True
            
        except Exception as e:
            logger.error(f"Genome validation failed: {e}")
            return False
    
    def mutate(self, mutation_rate: float = 0.1) -> 'DecisionGenome':
        """Create a mutated copy of the genome."""
        import random
        
        mutated = self.from_dict(self.to_dict())
        mutated.genome_id = f"{self.genome_id}_mutated_{datetime.now().timestamp()}"
        mutated.parent_ids = [self.genome_id]
        mutated.mutation_count = self.mutation_count + 1
        mutated.generation = self.generation + 1
        
        # Mutate strategy parameters
        if random.random() < mutation_rate:
            mutated.strategy.entry_threshold += random.uniform(-0.1, 0.1)
            mutated.strategy.entry_threshold = max(0, min(1, mutated.strategy.entry_threshold))
            
        if random.random() < mutation_rate:
            mutated.strategy.exit_threshold += random.uniform(-0.1, 0.1)
            mutated.strategy.exit_threshold = max(0, min(1, mutated.strategy.exit_threshold))
            
        # Mutate risk parameters
        if random.random() < mutation_rate:
            mutated.risk.risk_tolerance += random.uniform(-0.1, 0.1)
            mutated.risk.risk_tolerance = max(0, min(1, mutated.risk.risk_tolerance))
            
        # Mutate timing parameters
        if random.random() < mutation_rate:
            mutated.timing.holding_period_min += random.randint(-2, 2)
            mutated.timing.holding_period_min = max(0, mutated.timing.holding_period_min)
            
        # Normalize weights after mutation
        mutated._normalize_weights()
        
        return mutated
    
    def _normalize_weights(self):
        """Normalize sensory and thinking weights to sum to 1."""
        # Normalize sensory weights
        sensory_weights = [
            self.sensory.price_weight,
            self.sensory.volume_weight,
            self.sensory.orderbook_weight,
            self.sensory.news_weight,
            self.sensory.sentiment_weight,
            self.sensory.economic_weight
        ]
        total = sum(sensory_weights)
        if total > 0:
            self.sensory.price_weight /= total
            self.sensory.volume_weight /= total
            self.sensory.orderbook_weight /= total
            self.sensory.news_weight /= total
            self.sensory.sentiment_weight /= total
            self.sensory.economic_weight /= total
            
        # Normalize thinking weights
        thinking_weights = [
            self.thinking.trend_analysis_weight,
            self.thinking.risk_analysis_weight,
            self.thinking.performance_analysis_weight,
            self.thinking.pattern_recognition_weight
        ]
        total = sum(thinking_weights)
        if total > 0:
            self.thinking.trend_analysis_weight /= total
            self.thinking.risk_analysis_weight /= total
            self.thinking.performance_analysis_weight /= total
            self.thinking.pattern_recognition_weight /= total 