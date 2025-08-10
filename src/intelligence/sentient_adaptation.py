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
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import faiss

try:
    from src.evolution.episodic_memory_system import EpisodicMemorySystem  # legacy
except Exception:  # pragma: no cover
    class EpisodicMemorySystem:  # type: ignore
        pass
from src.thinking.memory.pattern_memory import PatternMemory

logger = logging.getLogger(__name__)


@dataclass
class LearningSignal:
    """Represents a learning signal from market interaction."""
    confidence: float
    adaptation_strength: float
    pattern_relevance: float
    risk_adjustment: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


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


class RealTimeLearningEngine:
    """Real-time learning engine for immediate adaptation."""
    
    def __init__(self, learning_rate: float = 0.001):
        self.learning_rate = learning_rate
        self.memory_buffer = []
        self.adaptation_network = self._build_adaptation_network()
        self.scaler = StandardScaler()
        
    def _build_adaptation_network(self) -> nn.Module:
        """Build neural network for adaptation learning."""
        return nn.Sequential(
            nn.Linear(20, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 4)  # Outputs: confidence, strength, relevance, risk
        )
    
    async def process_outcome(self, market_event: MarketEvent, 
                            strategy_response: Dict[str, Any], 
                            outcome: Dict[str, float]) -> LearningSignal:
        """Process trade outcome and generate learning signal."""
        
        # Prepare input features
        features = self._prepare_features(market_event, strategy_response, outcome)
        
        # Normalize features
        if len(self.memory_buffer) > 100:
            self.scaler.fit(np.array(self.memory_buffer))
            features_norm = self.scaler.transform(features.reshape(1, -1))
        else:
            features_norm = features.reshape(1, -1)
        
        # Generate learning signal
        with torch.no_grad():
            outputs = self.adaptation_network(torch.FloatTensor(features_norm))
            confidence, strength, relevance, risk = outputs[0].numpy()
        
        # Store in memory buffer for future learning
        self.memory_buffer.append(features)
        if len(self.memory_buffer) > 1000:
            self.memory_buffer = self.memory_buffer[-500:]
        
        return LearningSignal(
            confidence=float(confidence),
            adaptation_strength=float(strength),
            pattern_relevance=float(relevance),
            risk_adjustment=float(risk),
            metadata={
                'market_event': market_event.symbol,
                'strategy_id': strategy_response.get('strategy_id'),
                'outcome_pnl': outcome.get('pnl', 0)
            }
        )
    
    def _prepare_features(self, market_event: MarketEvent, 
                         strategy_response: Dict[str, Any], 
                         outcome: Dict[str, float]) -> np.ndarray:
        """Prepare feature vector for learning."""
        features = [
            market_event.price_change,
            market_event.volume_change,
            market_event.volatility_change,
            *market_event.pattern_vector[:10],  # First 10 pattern features
            strategy_response.get('confidence', 0.5),
            strategy_response.get('position_size', 0.1),
            strategy_response.get('risk_level', 0.5),
            outcome.get('pnl', 0),
            outcome.get('win', 0),
            outcome.get('duration', 0),
            outcome.get('max_drawdown', 0),
            outcome.get('sharpe_ratio', 0),
            outcome.get('volatility', 0),
            outcome.get('volume_ratio', 1),
            outcome.get('trend_strength', 0),
            outcome.get('support_distance', 0),
            outcome.get('resistance_distance', 0),
            outcome.get('momentum', 0),
            outcome.get('rsi', 50),
            outcome.get('macd_signal', 0)
        ]
        
        return np.array(features, dtype=np.float32)


class FAISSPatternMemory:
    """FAISS-based pattern memory for instant pattern recognition and retrieval."""
    
    def __init__(self, dimension: int = 128):
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.pattern_store = []
        self.context_store = []
        self.outcome_store = []
        self.confidence_store = []
        
    async def store_pattern(self, pattern: np.ndarray, context: Dict[str, Any], 
                          outcome: Dict[str, float], confidence: float):
        """Store pattern with context and outcome in FAISS index."""
        # Ensure pattern is correct dimension
        if len(pattern) < self.dimension:
            pattern = np.pad(pattern, (0, self.dimension - len(pattern)))
        elif len(pattern) > self.dimension:
            pattern = pattern[:self.dimension]
        
        # Add to FAISS index
        self.index.add(pattern.astype(np.float32).reshape(1, -1))
        
        # Store associated data
        self.pattern_store.append(pattern)
        self.context_store.append(context)
        self.outcome_store.append(outcome)
        self.confidence_store.append(confidence)
        
    async def find_similar_patterns(self, query_pattern: np.ndarray, 
                                  k: int = 5) -> List[Tuple[Dict[str, Any], Dict[str, float], float]]:
        """Find similar patterns and return their contexts and outcomes."""
        if self.index.ntotal == 0:
            return []
        
        # Ensure query is correct dimension
        if len(query_pattern) < self.dimension:
            query_pattern = np.pad(query_pattern, (0, self.dimension - len(query_pattern)))
        elif len(query_pattern) > self.dimension:
            query_pattern = query_pattern[:self.dimension]
        
        # Search for similar patterns
        distances, indices = self.index.search(
            query_pattern.astype(np.float32).reshape(1, -1), k
        )
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.context_store):
                results.append((
                    self.context_store[idx],
                    self.outcome_store[idx],
                    1.0 - (distances[0][i] / max(distances[0]))  # Convert distance to similarity
                ))
        
        return results
    
    def get_memory_stats(self) -> Dict[str, int]:
        """Get statistics about stored patterns."""
        return {
            'total_patterns': len(self.pattern_store),
            'index_size': self.index.ntotal,
            'dimension': self.dimension
        }


class MetaCognitionEngine:
    """Meta-cognitive system for assessing learning quality and decision confidence."""
    
    def __init__(self):
        self.confidence_history = []
        self.accuracy_tracker = []
        self.learning_quality_threshold = 0.7
        
    async def assess_learning_quality(self, learning_signal: LearningSignal,
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


class AdaptationController:
    """Controller for generating and applying strategy adaptations."""
    
    def __init__(self):
        self.adaptation_history = []
        self.risk_parameters = {
            'base_risk': 0.02,
            'max_risk': 0.05,
            'min_risk': 0.005,
            'volatility_adjustment': 1.0,
            'performance_adjustment': 1.0
        }
        
    async def generate_adaptations(self, learning_signal: LearningSignal,
                                 current_strategy_state: Dict[str, Any]) -> Dict[str, Any]:
        """Generate specific adaptations based on learning signal."""
        
        adaptations = {
            'risk_adjustment': self._calculate_risk_adjustment(learning_signal),
            'parameter_updates': self._generate_parameter_updates(learning_signal),
            'pattern_weights': self._update_pattern_weights(learning_signal),
            'confidence_thresholds': self._adjust_confidence_thresholds(learning_signal)
        }
        
        # Store adaptation for history
        self.adaptation_history.append({
            'timestamp': datetime.utcnow(),
            'learning_signal': learning_signal,
            'adaptations': adaptations
        })
        
        return adaptations
    
    def _calculate_risk_adjustment(self, learning_signal: LearningSignal) -> float:
        """Calculate dynamic risk adjustment based on learning."""
        base_risk = self.risk_parameters['base_risk']
        
        # Adjust based on learning signal
        risk_factor = 1.0 + learning_signal.risk_adjustment * learning_signal.confidence
        
        # Apply bounds
        adjusted_risk = base_risk * risk_factor
        return max(self.risk_parameters['min_risk'], 
                  min(self.risk_parameters['max_risk'], adjusted_risk))
    
    def _generate_parameter_updates(self, learning_signal: LearningSignal) -> Dict[str, float]:
        """Generate parameter updates based on learning."""
        return {
            'learning_rate': max(0.0001, min(0.01, 0.001 * learning_signal.adaptation_strength)),
            'confidence_threshold': max(0.5, min(0.9, 0.7 + learning_signal.confidence * 0.2)),
            'pattern_sensitivity': max(0.1, min(1.0, learning_signal.pattern_relevance))
        }
    
    def _update_pattern_weights(self, learning_signal: LearningSignal) -> Dict[str, float]:
        """Update weights for different pattern types."""
        return {
            'trend_patterns': 0.4 + learning_signal.pattern_relevance * 0.2,
            'mean_reversion': 0.3 + (1 - learning_signal.pattern_relevance) * 0.2,
            'momentum': 0.3 + learning_signal.confidence * 0.2
        }
    
    def _adjust_confidence_thresholds(self, learning_signal: LearningSignal) -> Dict[str, float]:
        """Adjust confidence thresholds for decision making."""
        return {
            'entry_threshold': max(0.6, min(0.9, 0.75 + learning_signal.confidence * 0.1)),
            'exit_threshold': max(0.3, min(0.7, 0.5 + learning_signal.confidence * 0.1)),
            'stop_loss_threshold': max(0.01, min(0.05, 0.02 - learning_signal.risk_adjustment * 0.01))
        }


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
