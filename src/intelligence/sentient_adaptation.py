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
from typing import Any, Dict, List, Optional, cast

import numpy as np

from src.sentient.adaptation.adaptation_controller import (
    AdaptationController as AdaptationController,
)
from src.sentient.adaptation.adaptation_controller import (
    TacticalAdaptation as TacticalAdaptation,
)
from src.sentient.learning.real_time_learning_engine import (
    RealTimeLearningEngine as RealTimeLearningEngine,
)
from src.sentient.memory.faiss_pattern_memory import FAISSPatternMemory as FAISSPatternMemory

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

    def __init__(self) -> None:
        self.confidence_history: List[float] = []
        self.accuracy_tracker: List[float] = []
        self.learning_quality_threshold = 0.7

    async def assess_learning_quality(
        self, learning_signal: AdaptationSignal, historical_performance: List[float]
    ) -> Dict[str, Any]:
        """Assess the quality of learning and whether adaptation should occur."""

        # Calculate learning quality metrics
        prediction_accuracy = self._calculate_prediction_accuracy(historical_performance)
        confidence_consistency = self._calculate_confidence_consistency()
        pattern_reliability = learning_signal.pattern_relevance

        # Overall learning quality score
        learning_quality = (
            prediction_accuracy * 0.4 + confidence_consistency * 0.3 + pattern_reliability * 0.3
        )

        # Determine if adaptation should occur
        should_adapt = learning_quality > self.learning_quality_threshold

        return {
            "learning_quality": learning_quality,
            "should_adapt": should_adapt,
            "prediction_accuracy": prediction_accuracy,
            "confidence_consistency": confidence_consistency,
            "pattern_reliability": pattern_reliability,
            "recommendation": "adapt" if should_adapt else "maintain",
        }

    def _calculate_prediction_accuracy(self, historical_performance: List[float]) -> float:
        """Calculate accuracy of recent predictions."""
        if len(historical_performance) < 10:
            return 0.5

        # Use last 20 performance values
        recent = historical_performance[-20:]
        trend = np.polyfit(range(len(recent)), recent, 1)[0]

        # Convert trend to accuracy score
        return float(max(0.0, min(1.0, 0.5 + float(trend) * 10)))

    def _calculate_confidence_consistency(self) -> float:
        """Calculate consistency of confidence scores."""
        if len(self.confidence_history) < 5:
            return 0.5

        recent_confidence = self.confidence_history[-10:]
        consistency = 1.0 - float(np.std(recent_confidence))
        return float(max(0.0, min(1.0, consistency)))


# Backward-compat alias to preserve public name without introducing duplicate ClassDef
MetaCognitionEngine = MetaCognitionEngineImpl


# Use canonical AdaptationController implementation to avoid duplicate ClassDef


class SentientAdaptationEngine:
    """Main engine for real-time adaptation and learning."""

    def __init__(self, episodic_memory: Optional[EpisodicMemorySystem] = None) -> None:
        # Initialize with minimal configs to satisfy component contracts
        self.real_time_learner = RealTimeLearningEngine(
            {
                "max_buffer_size": 1000,
                "min_pnl_threshold": 0.0001,
            }
        )
        self.pattern_memory = FAISSPatternMemory(
            {
                "vector_dimension": 64,
                "index_path": "data/memory/faiss_index",
                "metadata_path": "data/memory/metadata.json",
            }
        )
        self.meta_cognition = MetaCognitionEngine()
        self.adaptation_controller = AdaptationController(
            {
                "min_confidence": 0.7,
                "max_adaptations": 10,
                # optional: "risk_parameters": {...}
            }
        )
        self.episodic_memory = episodic_memory or EpisodicMemorySystem()

        # Performance tracking
        self.recent_performance: List[float] = []
        self.adaptation_count = 0
        self.last_adaptation: Optional[datetime] = None

    async def adapt_in_real_time(
        self,
        market_event: MarketEvent,
        strategy_response: Dict[str, Any],
        outcome: Dict[str, float],
    ) -> LearningSignal:
        """Main adaptation method - process market event and adapt strategy."""

        # Construct a trade_data payload expected by RealTimeLearningEngine
        now = datetime.utcnow()
        trade_data: Dict[str, Any] = {
            "trade_id": f"{strategy_response.get('strategy_id', 'strategy')}-{int(now.timestamp())}",
            "close_time": now.isoformat(),
            "pnl": float(outcome.get("pnl", 0.0)),
            "duration": float(outcome.get("duration", 0.0)),
            "max_drawdown": float(outcome.get("max_drawdown", 0.0)),
            "max_profit": float(outcome.get("max_profit", 0.0)),
            # context features used by detectors
            "price_change": float(market_event.price_change),
            "volume": float(market_event.volume_change),
            "avg_volume": float(market_event.context.get("avg_volume", 1) or 1),
            "entry_price": float(market_event.context.get("entry_price", 0.0)),
            "spread": float(market_event.context.get("spread", 0.0)),
            "volatility": float(market_event.volatility_change),
            "order_imbalance": float(market_event.context.get("order_imbalance", 0.0)),
            "liquidity_depth": float(market_event.context.get("liquidity_depth", 0.0)),
            "recent_trades": market_event.context.get("recent_trades", []),
        }

        # Immediate learning from trade outcome (canonical API)
        ls = await self.real_time_learner.process_closed_trade(trade_data)

        # Map learning signal into AdaptationSignal for compatibility
        adaptation_confidence = float(max(0.0, min(1.0, abs(float(ls.outcome.get("pnl", 0.0))))))
        learning_signal = AdaptationSignal(
            confidence=adaptation_confidence,
            adaptation_strength=adaptation_confidence,
            pattern_relevance=0.5,
            risk_adjustment=0.0,
            timestamp=now,
            metadata={
                "trade_id": ls.trade_id,
                "signal_type": ls.signal_type.value,
                "context": ls.context,
                "outcome": ls.outcome,
                "features": ls.features,
            },
        )

        # Update pattern memory with new experience (canonical API)
        _ = self.pattern_memory.add_experience(
            vector=market_event.extract_pattern(),
            metadata={
                "context": market_event.context,
                "outcome": outcome,
                "learning_signal_id": ls.trade_id,
                "generated_at": now.isoformat(),
            },
        )

        # Meta-cognitive assessment of learning quality
        learning_quality = await self.meta_cognition.assess_learning_quality(
            learning_signal, historical_performance=self.get_recent_performance()
        )

        # Adapt strategy parameters based on learning
        if learning_quality["should_adapt"]:
            # Build the memories payload expected by AdaptationController
            memories: List[Dict[str, Any]] = [
                {"metadata": {"outcome": outcome, "context": market_event.context}}
            ]
            adaptations = await self.adaptation_controller.generate_adaptations(
                memories, current_context=self.get_strategy_state()
            )
            await self.apply_adaptations(adaptations)

            self.adaptation_count += 1
            self.last_adaptation = now

        # Update episodic memory
        self._update_episodic_memory(market_event, outcome, learning_signal)

        return learning_signal

    async def apply_adaptations(self, adaptations: List[TacticalAdaptation]) -> None:
        """Apply generated adaptations to the system."""
        # This would interface with the actual strategy system
        logger.info(f"Applying adaptations: {[a.adaptation_id for a in adaptations]}")
        return None

    def get_recent_performance(self) -> List[float]:
        """Get recent performance metrics."""
        return self.recent_performance[-20:] if self.recent_performance else []

    def get_strategy_state(self) -> Dict[str, Any]:
        """Get current strategy state for adaptation."""
        return {
            "risk_parameters": cast(Dict[str, Any], self.adaptation_controller.config).get(
                "risk_parameters", {}
            ),
            "adaptation_count": self.adaptation_count,
            "last_adaptation": self.last_adaptation,
        }

    def _update_episodic_memory(
        self, market_event: MarketEvent, outcome: Dict[str, float], learning_signal: LearningSignal
    ) -> None:
        """Update episodic memory with new experience."""
        # This would integrate with the episodic memory system
        pass

    def get_adaptation_stats(self) -> Dict[str, Any]:
        """Get statistics about adaptations."""
        return {
            "total_adaptations": self.adaptation_count,
            "last_adaptation": self.last_adaptation,
            "pattern_memory_stats": self.pattern_memory.get_memory_stats(),
            "recent_performance": self.get_recent_performance(),
        }


# Example usage and testing
async def test_sentient_adaptation() -> None:
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
        context={"regime": "trending", "volatility": "high"},
    )

    # Create test strategy response
    strategy_response = {
        "strategy_id": "test_strategy_1",
        "confidence": 0.75,
        "position_size": 0.1,
        "risk_level": 0.02,
    }

    # Create test outcome
    outcome = {
        "pnl": 150.0,
        "win": 1,
        "duration": 3600,
        "max_drawdown": 0.01,
        "sharpe_ratio": 2.5,
        "volatility": 0.15,
        "volume_ratio": 1.2,
        "trend_strength": 0.8,
        "support_distance": 0.005,
        "resistance_distance": 0.008,
        "momentum": 0.7,
        "rsi": 65,
        "macd_signal": 0.02,
    }

    # Test adaptation
    learning_signal = await engine.adapt_in_real_time(market_event, strategy_response, outcome)

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
