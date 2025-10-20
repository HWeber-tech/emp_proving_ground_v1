#!/usr/bin/env python3
"""
SentientPredator - Epic 1: The Predator's Instinct
Main orchestrator for the sentient predator capabilities.
"""

import logging
from datetime import datetime
from numbers import Number
from typing import Any, Dict, List, Optional

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

        episode_info = self._identify_extreme_episode(learning_signal)
        if episode_info is not None:
            latent_summary = self._build_episode_latent_summary(vector, learning_signal, episode_info)
            episode_metadata = {
                "episode_type": episode_info["type"],
                "severity": episode_info["severity"],
                "reason": episode_info["reason"],
                "evidence": self._normalise_mapping(episode_info.get("evidence", {})),
                "trade_id": learning_signal.trade_id,
                "signal_type": learning_signal.signal_type.value,
                "context": self._normalise_mapping(learning_signal.context),
                "features": self._normalise_mapping(learning_signal.features),
                "outcome": self._normalise_mapping(learning_signal.outcome),
                "metadata": self._normalise_mapping(learning_signal.metadata),
                "captured_by": "sentient_predator",
            }
            self.memory.store_extreme_episode(latent_summary, episode_metadata)
            logger.info(
                "Stored extreme episode",
                extra={
                    "episode_type": episode_info["type"],
                    "severity": episode_info["severity"],
                    "trade_id": learning_signal.trade_id,
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

    def _identify_extreme_episode(self, signal: LearningSignal) -> Optional[dict[str, Any]]:
        """Determine if the learning signal corresponds to an extreme market episode."""

        market_condition = str(
            signal.metadata.get("market_condition")
            or signal.context.get("market_condition")
            or ""
        ).lower()

        volatility = float(self._safe_numeric(signal.context.get("volatility", 0.0)))
        volume_ratio = float(self._safe_numeric(signal.features.get("volume_ratio", 1.0)))
        price_momentum = float(self._safe_numeric(signal.features.get("price_momentum", 0.0)))
        order_flow = float(self._safe_numeric(signal.features.get("order_flow_imbalance", 0.0)))

        thresholds = {
            "volatility": float(self.config.get("extreme_volatility_threshold", 0.08)),
            "volume_ratio": float(self.config.get("extreme_volume_threshold", 4.0)),
            "momentum": float(self.config.get("extreme_momentum_threshold", 0.03)),
            "order_flow": float(self.config.get("extreme_order_flow_threshold", 0.015)),
        }

        evidence = {
            "volatility": volatility,
            "volume_ratio": volume_ratio,
            "price_momentum": price_momentum,
            "order_flow_imbalance": order_flow,
        }

        if "flash" in market_condition and "crash" in market_condition:
            episode_type = "flash_crash"
            reason = "market_condition flagged flash crash"
        elif "news" in market_condition and "shock" in market_condition:
            episode_type = "news_shock"
            reason = "market_condition flagged news shock"
        elif volatility >= thresholds["volatility"] and volume_ratio >= thresholds["volume_ratio"]:
            episode_type = "volatility_spike"
            reason = "volatility and volume ratio exceeded thresholds"
        elif price_momentum >= thresholds["momentum"] and abs(order_flow) >= thresholds["order_flow"]:
            episode_type = "order_flow_shock"
            reason = "momentum and order flow imbalance exceeded thresholds"
        else:
            return None

        severity = self._calculate_episode_severity(
            volatility=volatility,
            volume_ratio=volume_ratio,
            price_momentum=price_momentum,
            order_flow=order_flow,
        )

        return {
            "type": episode_type,
            "severity": severity,
            "reason": reason,
            "evidence": evidence,
        }

    def _calculate_episode_severity(
        self,
        volatility: float,
        volume_ratio: float,
        price_momentum: float,
        order_flow: float,
    ) -> float:
        """Combine multiple signals into a severity score in [0, 1]."""

        volatility_component = min(1.0, abs(volatility) / 0.12)
        volume_component = min(1.0, max(0.0, volume_ratio - 1.0) / 5.0)
        momentum_component = min(1.0, abs(price_momentum) / 0.05)
        order_flow_component = min(1.0, abs(order_flow) / 0.025)

        severity = (
            0.4 * volatility_component
            + 0.25 * volume_component
            + 0.2 * momentum_component
            + 0.15 * order_flow_component
        )
        return float(max(0.0, min(1.0, severity)))

    def _build_episode_latent_summary(
        self,
        base_vector: np.ndarray,
        signal: LearningSignal,
        episode_info: dict[str, Any],
    ) -> np.ndarray:
        """Construct latent summary vector for an extreme episode."""

        summary = np.asarray(base_vector, dtype=float).copy()
        if summary.ndim != 1:
            summary = summary.reshape(-1)

        if summary.size > self.memory.dimension:
            summary = summary[: self.memory.dimension]
        elif summary.size < self.memory.dimension:
            summary = np.pad(summary, (0, self.memory.dimension - summary.size))

        summary[0] = float(episode_info["severity"])
        if summary.size > 1:
            summary[1] = float(self._safe_numeric(signal.context.get("volatility", 0.0)))
        if summary.size > 2:
            summary[2] = float(self._safe_numeric(signal.features.get("volume_ratio", 1.0)))
        if summary.size > 3:
            summary[3] = float(self._safe_numeric(signal.features.get("price_momentum", 0.0)))
        if summary.size > 4:
            summary[4] = float(self._safe_numeric(signal.features.get("order_flow_imbalance", 0.0)))
        if summary.size > 5:
            summary[5] = float(self._safe_numeric(signal.outcome.get("max_drawdown", 0.0)))

        return summary.astype(np.float32)

    @staticmethod
    def _safe_numeric(value: Any) -> float:
        """Convert numeric-like values to float, defaulting to 0.0."""

        if isinstance(value, Number):
            return float(value)
        if isinstance(value, np.generic):  # pragma: no cover - defensive guard
            return float(value)
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _normalise_mapping(source: Dict[str, Any]) -> Dict[str, Any]:
        """Normalise mapping values to JSON-serialisable primitives."""

        normalised: Dict[str, Any] = {}
        for key, value in source.items():
            if isinstance(value, (str, bool)) or value is None:
                normalised[key] = value
            elif isinstance(value, Number):
                normalised[key] = float(value)
            elif isinstance(value, np.generic):  # pragma: no cover - defensive guard
                normalised[key] = float(value)
            else:
                try:
                    normalised[key] = float(value)
                except (TypeError, ValueError):
                    normalised[key] = str(value)
        return normalised
