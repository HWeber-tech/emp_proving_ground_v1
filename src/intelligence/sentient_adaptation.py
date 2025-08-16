#!/usr/bin/env python3
"""
SENTIENT-30: Dynamic Adaptation Engine
=====================================

Real-time learning and adaptation without generational delays.
Implements the core sentient predator system with intra-generation learning,
active pattern memory, meta-cognition, and dynamic risk evolution.

This module provides the foundation for transforming the EMP into a truly
sentient trading system that learns and adapts in real-time.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
from src.sentient.learning.real_time_learning_engine import (
    RealTimeLearningEngine as RealTimeLearningEngine,
)
from src.sentient.memory.faiss_pattern_memory import (
    FAISSPatternMemory as FAISSPatternMemory,
)
from src.sentient.adaptation.adaptation_controller import (
    AdaptationController as AdaptationController,  # type: ignore
)

try:
    from src.evolution.episodic_memory_system import EpisodicMemorySystem  # legacy
except Exception:  # pragma: no cover
    class EpisodicMemorySystem:  # type: ignore
        pass

logger = logging.getLogger(__name__)


@dataclass
class AdaptationSignal:
    """Represents a learning signal from market interaction."""
    confidence: float
    adaptation_strength: float
    pattern_relevance: float
    risk_adjustment: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)

# Backward-compat alias to avoid breaking imports while removing duplicate class name
LearningSignal = AdaptationSignal


@dataclass
class MarketEvent:
    """Represents a market event with extracted patterns and context."""
    symbol: str
    timestamp: datetime
    price_change: float
    volume_change: float
    volatility_change: float
    pattern_vector: np.ndarray
    context: Dict[str, Any]
    
    def extract_pattern(self) -> np.ndarray:
        """Extract the pattern vector for memory storage."""
        return self.pattern_vector




class MetaCognitionEngineImpl:
    """Meta-cognitive system for assessing learning quality and decision confidence."""
    
    def __init__(self):
        self.confidence_history = []
        self.accuracy_tracker = []
        self.learning_quality_threshold = 0.7
        
    async def assess_learning_quality(self, learning_signal: AdaptationSignal,
                                    historical_performance: List[float]) -> Dict[str, Any]:
        """Assess the quality of learning and whether adaptation should occur."""
        
        # Calculate learning quality metrics
        prediction_accuracy = self._calculate_prediction_accuracy(historical_performance)
        confidence_consistency = self._calculate_confidence_consistency()
        pattern_reliability = learning_signal.pattern_relevance
        
        # Overall learning quality score
        learning_quality = (
            prediction_accuracy * 0.4 +
            confidence_consistency * 0.3 +
            pattern_reliability * 0.3
        )
        
        # Determine if adaptation should occur
        should_adapt = learning_quality > self.learning_quality_threshold
        
        return {
            'learning_quality': learning_quality,
            'should_adapt': should_adapt,
            'prediction_accuracy': prediction_accuracy,
            'confidence_consistency': confidence_consistency,
            'pattern_reliability': pattern_reliability,
            'recommendation': 'adapt' if should_adapt else 'maintain'
        }
    
    def _calculate_prediction_accuracy(self, historical_performance: List[float]) -> float:
        """Calculate accuracy of recent predictions."""
        if len(historical_performance) < 10:
            return 0.5
        
        # Use last 20 performance values
        recent = historical_performance[-20:]
        trend = np.polyfit(range(len(recent)), recent, 1)[0]
        
        # Convert trend to accuracy score
        return max(0, min(1, 0.5 + trend * 10))
    
    def _calculate_confidence_consistency(self) -> float:
        """Calculate consistency of confidence scores."""
        if len(self.confidence_history) < 5:
            return 0.5
        
        recent_confidence = self.confidence_history[-10:]
        consistency = 1.0 - np.std(recent_confidence)
        return max(0, min(1, consistency))

# Backward-compat alias to preserve public name without introducing duplicate ClassDef
MetaCognitionEngine = MetaCognitionEngineImpl


# Use canonical AdaptationController implementation to avoid duplicate ClassDef


class SentientAdaptationEngine:
    """Main engine for real-time adaptation and learning."""
    
    def __init__(self, episodic_memory: Optional[EpisodicMemorySystem] = None):
        self.real_time_learner = RealTimeLearningEngine()
        self.pattern_memory = FAISSPatternMemory()
        self.meta_cognition = MetaCognitionEngine()
        self.adaptation_controller = AdaptationController()
        self.episodic_memory = episodic_memory or EpisodicMemorySystem()
        
        # Performance tracking
        self.recent_performance = []
        self.adaptation_count = 0
        self.last_adaptation = None
        
    async def adapt_in_real_time(self, market_event: MarketEvent, 
                               strategy_response: Dict[str, Any], 
                               outcome: Dict[str, float]) -> LearningSignal:
        """Main adaptation method - process market event and adapt strategy."""
        
        # Immediate learning from trade outcome
        learning_signal = await self.real_time_learner.process_outcome(
            market_event, strategy_response, outcome
        )
        
        # Update pattern memory with new experience
        await self.pattern_memory.store_pattern(
            pattern=market_event.extract_pattern(),
            context=market_event.context,
            outcome=outcome,
            confidence=learning_signal.confidence
        )
        
        # Meta-cognitive assessment of learning quality
        learning_quality = await self.meta_cognition.assess_learning_quality(
            learning_signal,
            historical_performance=self.get_recent_performance()
        )
        
        # Adapt strategy parameters based on learning
        if learning_quality['should_adapt']:
            adaptations = await self.adaptation_controller.generate_adaptations(
                learning_signal,
                current_strategy_state=self.get_strategy_state()
            )
            await self.apply_adaptations(adaptations)
            
            self.adaptation_count += 1
            self.last_adaptation = datetime.utcnow()
        
        # Update episodic memory
        self._update_episodic_memory(market_event, outcome, learning_signal)
        
        return learning_signal
    
    async def apply_adaptations(self, adaptations: Dict[str, Any]):
        """Apply generated adaptations to the system."""
        # This would interface with the actual strategy system
        logger.info(f"Applying adaptations: {adaptations}")
        
    def get_recent_performance(self) -> List[float]:
        """Get recent performance metrics."""
        return self.recent_performance[-20:] if self.recent_performance else []
    
    def get_strategy_state(self) -> Dict[str, Any]:
        """Get current strategy state for adaptation."""
        return {
            'risk_parameters': self.adaptation_controller.risk_parameters,
            'adaptation_count': self.adaptation_count,
            'last_adaptation': self.last_adaptation
        }
    
    def _update_episodic_memory(self, market_event: MarketEvent, 
                              outcome: Dict[str, float], 
                              learning_signal: LearningSignal):
        """Update episodic memory with new experience."""
        # This would integrate with the episodic memory system
        pass
    
    def get_adaptation_stats(self) -> Dict[str, Any]:
        """Get statistics about adaptations."""
        return {
            'total_adaptations': self.adaptation_count,
            'last_adaptation': self.last_adaptation,
            'pattern_memory_stats': self.pattern_memory.get_memory_stats(),
            'recent_performance': self.get_recent_performance()
        }


# Example usage and testing
async def test_sentient_adaptation():
    """Test the sentient adaptation engine."""
    engine = SentientAdaptationEngine()
    
    # Create test market event
    market_event = MarketEvent(
        symbol="EURUSD",
        timestamp=datetime.utcnow(),
        price_change=0.0015,
        volume_change=0.05,
        volatility_change=0.02,
        pattern_vector=np.random.randn(20),
        context={'regime': 'trending', 'volatility': 'high'}
    )
    
    # Create test strategy response
    strategy_response = {
        'strategy_id': 'test_strategy_1',
        'confidence': 0.75,
        'position_size': 0.1,
        'risk_level': 0.02
    }
    
    # Create test outcome
    outcome = {
        'pnl': 150.0,
        'win': 1,
        'duration': 3600,
        'max_drawdown': 0.01,
        'sharpe_ratio': 2.5,
        'volatility': 0.15,
        'volume_ratio': 1.2,
        'trend_strength': 0.8,
        'support_distance': 0.005,
        'resistance_distance': 0.008,
        'momentum': 0.7,
        'rsi': 65,
        'macd_signal': 0.02
    }
    
    # Test adaptation
    learning_signal = await engine.adapt_in_real_time(
        market_event, strategy_response, outcome
    )
    
    print("Learning Signal:")
    print(f"  Confidence: {learning_signal.confidence}")
    print(f"  Adaptation Strength: {learning_signal.adaptation_strength}")
    print(f"  Pattern Relevance: {learning_signal.pattern_relevance}")
    print(f"  Risk Adjustment: {learning_signal.risk_adjustment}")
    
    stats = engine.get_adaptation_stats()
    print("\nAdaptation Stats:")
    print(f"  Total Adaptations: {stats['total_adaptations']}")
    print(f"  Pattern Memory: {stats['pattern_memory_stats']}")


if __name__ == "__main__":
    asyncio.run(test_sentient_adaptation())
