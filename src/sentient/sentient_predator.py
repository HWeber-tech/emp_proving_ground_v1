#!/usr/bin/env python3
"""
SentientPredator - Epic 1: The Predator's Instinct
Main orchestrator for the sentient predator capabilities.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List

import numpy as np

from src.sentient.adaptation.adaptation_controller import AdaptationController, TacticalAdaptation
from src.sentient.learning.real_time_learning_engine import LearningSignal, RealTimeLearningEngine
from src.sentient.memory.faiss_pattern_memory import FAISSPatternMemory

logger = logging.getLogger(__name__)


class SentientPredator:
    """Main orchestrator for sentient predator capabilities."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config

        # Initialize components
        self.learning_engine = RealTimeLearningEngine(config.get("learning", {}))
        self.memory = FAISSPatternMemory(config.get("memory", {}))
        self.adaptation_controller = AdaptationController(config.get("adaptation", {}))

        # State tracking
        self.is_active = False
        self.adaptations_applied: List[TacticalAdaptation] = []
        self.performance_metrics = {
            "total_trades": 0,
            "total_adaptations": 0,
            "successful_adaptations": 0,
            "total_pnl": 0.0,
        }

    async def start(self) -> None:
        """Start the sentient predator."""
        logger.info("Starting Sentient Predator...")
        self.is_active = True

    async def stop(self) -> None:
        """Stop the sentient predator."""
        logger.info("Stopping Sentient Predator...")
        self.is_active = False

    async def process_closed_trade(self, trade_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a closed trade through the entire sentient pipeline."""
        if not self.is_active:
            logger.warning("Sentient Predator is not active")
            return {}

        logger.info(f"Processing closed trade: {trade_data.get('trade_id')}")

        # Step 1: Generate learning signal
        learning_signal = await self.learning_engine.process_closed_trade(trade_data)

        # Step 2: Create vector representation
        vector = self._create_vector_representation(learning_signal)

        # Step 3: Store in memory
        memory_id = self.memory.add_experience(
            vector,
            {
                "learning_signal": {
                    "trade_id": learning_signal.trade_id,
                    "signal_type": learning_signal.signal_type.value,
                    "pnl": learning_signal.pnl,
                    "duration": learning_signal.duration,
                },
                "context": learning_signal.context,
                "features": learning_signal.features,
                "outcome": learning_signal.outcome,
            },
        )

        # Step 4: Search for similar patterns
        similar_memories = self.memory.search_similar(vector, k=20)

        # Step 5: Generate adaptations
        current_context = self._extract_current_context(trade_data)
        adaptations = await self.adaptation_controller.generate_adaptations(
            similar_memories, current_context
        )

        # Step 6: Apply adaptations to genome
        applied_adaptations = await self._apply_adaptations(adaptations)

        # Update metrics
        self.performance_metrics["total_trades"] += 1
        self.performance_metrics["total_adaptations"] += len(adaptations)
        self.performance_metrics["total_pnl"] += learning_signal.pnl

        return {
            "learning_signal": learning_signal,
            "memory_id": memory_id,
            "similar_memories": len(similar_memories),
            "adaptations": adaptations,
            "applied_adaptations": applied_adaptations,
            "metrics": self.get_metrics(),
        }

    def _create_vector_representation(self, signal: LearningSignal) -> np.ndarray:
        """Create vector representation from learning signal."""
        # Combine features into vector
        features = signal.features
        context = signal.context

        vector = np.asarray(
            [
                float(features.get("price_momentum", 0.0)),
                float(features.get("volume_ratio", 1.0)),
                float(features.get("volatility_ratio", 1.0)),
                float(features.get("liquidity_ratio", 1.0)),
                float(features.get("order_flow_imbalance", 0.0)),
                float(features.get("microstructure_score", 0.0)),
                float(context.get("volatility", 0.0)),
                float(context.get("spread", 0.0)),
                float(signal.pnl),
                float(signal.duration) / 3600.0,  # Normalize duration to hours
            ],
            dtype=float,
        )
        # Explicitly annotate to satisfy typing
        vector: np.ndarray = vector

        # Pad or truncate to fixed dimension
        target_dim = self.memory.dimension
        if len(vector) < target_dim:
            vector = np.pad(vector, (0, target_dim - len(vector)))
        elif len(vector) > target_dim:
            vector = vector[:target_dim]

        return vector

    def _extract_current_context(self, trade_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract current market context."""
        return {
            "volatility": trade_data.get("volatility", 0),
            "liquidity_depth": trade_data.get("liquidity_depth", 0),
            "order_imbalance": trade_data.get("order_imbalance", 0),
            "spread": trade_data.get("spread", 0),
            "market_condition": trade_data.get("market_condition", "neutral"),
        }

    async def _apply_adaptations(
        self, adaptations: List[TacticalAdaptation]
    ) -> List[TacticalAdaptation]:
        """Apply adaptations to the genome."""
        applied = []

        for adaptation in adaptations:
            if adaptation.is_high_confidence:
                # Here we would apply the adaptation to the genome
                # For now, we'll just log it
                logger.info(f"Applying adaptation: {adaptation.adaptation_id}")
                self.adaptations_applied.append(adaptation)
                applied.append(adaptation)

        return applied

    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        return {
            **self.performance_metrics,
            "learning_engine": self.learning_engine.get_performance_summary(),
            "memory": self.memory.get_memory_stats(),
            "adaptation_controller": self.adaptation_controller.get_adaptation_summary(),
            "active_adaptations": len(self.adaptations_applied),
        }

    def get_sentient_state(self) -> Dict[str, Any]:
        """Get the current sentient state."""
        return {
            "is_active": self.is_active,
            "components": {
                "learning_engine": "active",
                "memory": "active",
                "adaptation_controller": "active",
            },
            "metrics": self.get_metrics(),
            "last_update": datetime.utcnow().isoformat(),
        }

    async def reset(self) -> None:
        """Reset the sentient predator."""
        logger.info("Resetting Sentient Predator...")
        self.memory.clear_memory()
        self.adaptations_applied.clear()
        self.performance_metrics = {
            "total_trades": 0,
            "total_adaptations": 0,
            "successful_adaptations": 0,
            "total_pnl": 0.0,
        }
        logger.info("Sentient Predator reset complete")
