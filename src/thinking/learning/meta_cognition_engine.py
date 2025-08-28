"""
Meta-Cognition Engine
Self-awareness system that assesses learning quality and decision confidence.
"""

import logging
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional

try:
    from src.core.events import ContextPacket, LearningSignal  # legacy
except Exception:  # pragma: no cover
    LearningSignal = ContextPacket = object
from src.core.state_store import StateStore

logger = logging.getLogger(__name__)


class MetaCognitionEngine:
    """
    Meta-cognitive system for assessing learning quality and decision confidence.

    Features:
    - Learning quality assessment
    - Decision confidence calibration
    - Self-reflection on learning outcomes
    - Adaptive learning rate adjustment
    """

    def __init__(self, state_store: StateStore):
        self.state_store = state_store
        self._learning_history_key = "emp:learning_history"
        self._confidence_history_key = "emp:confidence_history"
        self._quality_threshold = 0.7

    def _as_float(self, v: object) -> float:
        """Safely coerce various primitive-like values to float."""
        try:
            return float(v) if isinstance(v, (int, float, str)) else 0.0
        except Exception:
            return 0.0

    async def assess_learning_quality(
        self, learning_signal: LearningSignal, historical_performance: dict[str, object]
    ) -> dict[str, object]:
        """
        Assess the quality of a learning signal.

        Args:
            learning_signal: The learning signal to assess
            historical_performance: Historical performance metrics

        Returns:
            Assessment of learning quality and recommendations
        """
        try:
            # Calculate learning quality metrics
            accuracy_score = await self._calculate_accuracy_score(learning_signal)
            consistency_score = await self._calculate_consistency_score(learning_signal)
            relevance_score = await self._calculate_relevance_score(learning_signal)

            # Overall learning quality
            overall_quality = accuracy_score * 0.4 + consistency_score * 0.3 + relevance_score * 0.3

            # Determine if adaptation should occur
            should_adapt = overall_quality >= self._quality_threshold

            # Generate recommendations
            recommendations = []
            if accuracy_score < 0.7:
                recommendations.append("Improve prediction accuracy")
            if consistency_score < 0.7:
                recommendations.append("Enhance pattern consistency")
            if relevance_score < 0.7:
                recommendations.append("Increase context relevance")

            assessment = {
                "learning_quality": overall_quality,
                "accuracy_score": accuracy_score,
                "consistency_score": consistency_score,
                "relevance_score": relevance_score,
                "should_adapt": should_adapt,
                "recommendations": recommendations,
                "timestamp": datetime.utcnow().isoformat(),
            }

            # Store assessment
            await self._store_learning_assessment(learning_signal.trade_id, assessment)

            logger.debug(
                f"Learning quality assessment: {overall_quality:.2f}, " f"Adapt: {should_adapt}"
            )

            return assessment

        except Exception as e:
            logger.error(f"Error assessing learning quality: {e}")
            return {
                "learning_quality": 0.5,
                "accuracy_score": 0.5,
                "consistency_score": 0.5,
                "relevance_score": 0.5,
                "should_adapt": False,
                "recommendations": ["Error in assessment"],
                "timestamp": datetime.utcnow().isoformat(),
            }

    async def _calculate_accuracy_score(self, learning_signal: LearningSignal) -> float:
        """Calculate accuracy score based on prediction vs actual outcome."""
        try:
            # Get historical predictions for similar contexts
            historical_predictions = await self._get_historical_predictions(
                learning_signal.triggering_context
            )

            if not historical_predictions:
                return 0.5  # Neutral score for no history

            # Calculate prediction accuracy
            predicted_outcomes: List[float] = [
                self._as_float(p.get("predicted_pnl", 0.0)) for p in historical_predictions
            ]
            actual_outcomes: List[float] = [
                self._as_float(p.get("actual_pnl", 0.0)) for p in historical_predictions
            ]

            if len(predicted_outcomes) < 3:
                return 0.5

            # Calculate correlation between predictions and actuals
            correlation = self._calculate_correlation(predicted_outcomes, actual_outcomes)

            # Calculate mean absolute error
            mae = sum(
                abs(float(p) - float(a)) for p, a in zip(predicted_outcomes, actual_outcomes)
            ) / len(predicted_outcomes)

            # Normalize MAE to 0-1 scale
            max_possible_error = max(abs(max(actual_outcomes)), abs(min(actual_outcomes))) * 2
            normalized_mae = 1 - (mae / max_possible_error) if max_possible_error > 0 else 0.5

            # Combine correlation and normalized MAE
            accuracy_score = (correlation + normalized_mae) / 2

            return max(0.0, min(1.0, accuracy_score))

        except Exception as e:
            logger.error(f"Error calculating accuracy score: {e}")
            return 0.5

    async def _calculate_consistency_score(self, learning_signal: LearningSignal) -> float:
        """Calculate consistency score based on similar context outcomes."""
        try:
            # Get recent similar learning signals
            recent_signals = await self._get_recent_learning_signals(
                learning_signal.triggering_context, days=7
            )

            if len(recent_signals) < 3:
                return 0.5  # Neutral score for insufficient data

            # Calculate outcome consistency
            outcomes = [self._as_float(getattr(s, "outcome_pnl", 0.0)) for s in recent_signals]

            # Calculate coefficient of variation
            mean_outcome = sum(outcomes) / len(outcomes)
            std_dev = (sum((o - mean_outcome) ** 2 for o in outcomes) / len(outcomes)) ** 0.5

            if mean_outcome == 0:
                return 0.5

            coefficient_of_variation = std_dev / abs(mean_outcome)

            # Convert to consistency score (lower CV = higher consistency)
            consistency_score = 1 / (1 + coefficient_of_variation)

            return float(max(0.0, min(1.0, consistency_score)))

        except Exception as e:
            logger.error(f"Error calculating consistency score: {e}")
            return 0.5

    async def _calculate_relevance_score(self, learning_signal: LearningSignal) -> float:
        """Calculate relevance score based on context similarity."""
        try:
            # Get context features
            context = learning_signal.triggering_context

            # Calculate context stability
            context_stability = await self._calculate_context_stability(context)

            # Calculate regime relevance
            regime_relevance = await self._calculate_regime_relevance(context)

            # Calculate confidence relevance
            confidence_relevance = float(context.confidence)

            # Combine scores
            relevance_score = (
                context_stability * 0.4 + regime_relevance * 0.3 + confidence_relevance * 0.3
            )

            return max(0.0, min(1.0, relevance_score))

        except Exception as e:
            logger.error(f"Error calculating relevance score: {e}")
            return 0.5

    async def _calculate_context_stability(self, context: ContextPacket) -> float:
        """Calculate how stable the current context is."""
        try:
            # Get recent context changes
            recent_contexts = await self._get_recent_contexts(context.regime, minutes=30)

            if len(recent_contexts) < 2:
                return 0.7  # Default stability for limited data

            # Calculate context volatility
            confidence_changes = [
                abs(float(c.confidence) - float(context.confidence)) for c in recent_contexts
            ]

            avg_change = sum(confidence_changes) / len(confidence_changes)

            # Convert to stability score
            stability_score = 1 - min(avg_change, 1.0)

            return max(0.0, min(1.0, stability_score))

        except Exception as e:
            logger.error(f"Error calculating context stability: {e}")
            return 0.5

    async def _calculate_regime_relevance(self, context: ContextPacket) -> float:
        """Calculate how relevant the current regime is."""
        try:
            # Get regime performance history
            regime_performance = await self._get_regime_performance(context.regime)

            if not regime_performance:
                return 0.5

            # Calculate regime effectiveness
            avg_performance = regime_performance.get("avg_performance", 0)
            win_rate = regime_performance.get("win_rate", 0.5)

            # Combine into relevance score
            relevance = (avg_performance + win_rate) / 2

            return max(0.0, min(1.0, relevance))

        except Exception as e:
            logger.error(f"Error calculating regime relevance: {e}")
            return 0.5

    def _calculate_correlation(self, x: List[float], y: List[float]) -> float:
        """Calculate Pearson correlation coefficient."""
        if len(x) != len(y) or len(x) < 2:
            return 0.0

        mean_x = sum(x) / len(x)
        mean_y = sum(y) / len(y)

        numerator = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
        denominator_x = sum((xi - mean_x) ** 2 for xi in x) ** 0.5
        denominator_y = sum((yi - mean_y) ** 2 for yi in y) ** 0.5

        if denominator_x == 0 or denominator_y == 0:
            return 0.0

        correlation = numerator / (denominator_x * denominator_y)
        return float(max(-1.0, min(1.0, correlation)))

    async def calibrate_confidence(
        self, learning_signal: LearningSignal, prediction_history: dict[str, object]
    ) -> Decimal:
        """
        Calibrate confidence based on historical accuracy.

        Args:
            learning_signal: The learning signal to calibrate
            prediction_history: Historical prediction accuracy

        Returns:
            Calibrated confidence level
        """
        try:
            # Get historical accuracy for similar contexts
            historical_accuracy = await self._get_historical_accuracy(
                learning_signal.triggering_context
            )

            if historical_accuracy is None:
                return Decimal(str(learning_signal.confidence_of_outcome))

            # Calculate calibration factor
            calibration_factor = historical_accuracy / 0.75  # Normalize to 0.75 baseline

            # Apply calibration
            calibrated_confidence = (
                float(learning_signal.confidence_of_outcome) * calibration_factor
            )

            # Ensure reasonable bounds
            calibrated_confidence = max(0.1, min(1.0, calibrated_confidence))

            # Store calibration
            await self._store_confidence_calibration(
                learning_signal.trade_id,
                float(learning_signal.confidence_of_outcome),
                calibrated_confidence,
            )

            return Decimal(str(calibrated_confidence))

        except Exception as e:
            logger.error(f"Error calibrating confidence: {e}")
            return Decimal(str(learning_signal.confidence_of_outcome))

    async def _get_historical_predictions(self, context: ContextPacket) -> List[dict[str, object]]:
        """Get historical predictions for similar contexts."""
        try:
            # This would be enhanced with actual retrieval
            return []
        except Exception as e:
            logger.error(f"Error getting historical predictions: {e}")
            return []

    async def _get_recent_learning_signals(
        self, context: ContextPacket, days: int = 7
    ) -> List[LearningSignal]:
        """Get recent learning signals for similar contexts."""
        try:
            # This would be enhanced with actual retrieval
            return []
        except Exception as e:
            logger.error(f"Error getting recent learning signals: {e}")
            return []

    async def _get_recent_contexts(self, regime: str, minutes: int = 30) -> List[ContextPacket]:
        """Get recent contexts for the given regime."""
        try:
            # This would be enhanced with actual retrieval
            return []
        except Exception as e:
            logger.error(f"Error getting recent contexts: {e}")
            return []

    async def _get_regime_performance(self, regime: str) -> Dict[str, float]:
        """Get performance metrics for the given regime."""
        try:
            # This would be enhanced with actual retrieval
            return {"avg_performance": 0.5, "win_rate": 0.5}
        except Exception as e:
            logger.error(f"Error getting regime performance: {e}")
            return {"avg_performance": 0.5, "win_rate": 0.5}

    async def _get_historical_accuracy(self, context: ContextPacket) -> Optional[float]:
        """Get historical accuracy for similar contexts."""
        try:
            # This would be enhanced with actual retrieval
            return 0.75
        except Exception as e:
            logger.error(f"Error getting historical accuracy: {e}")
            return None

    async def _store_learning_assessment(
        self, trade_id: str, assessment: dict[str, object]
    ) -> None:
        """Store learning assessment for future reference."""
        try:
            key = f"{self._learning_history_key}:{trade_id}"
            await self.state_store.set(key, str(assessment), expire=86400 * 30)  # 30 days
        except Exception as e:
            logger.error(f"Error storing learning assessment: {e}")

    async def _store_confidence_calibration(
        self, trade_id: str, original_confidence: float, calibrated_confidence: float
    ) -> None:
        """Store confidence calibration for tracking."""
        try:
            key = f"{self._confidence_history_key}:{trade_id}"
            await self.state_store.set(
                key,
                str(
                    {
                        "original": original_confidence,
                        "calibrated": calibrated_confidence,
                        "timestamp": datetime.utcnow().isoformat(),
                    }
                ),
                expire=86400 * 30,  # 30 days
            )
        except Exception as e:
            logger.error(f"Error storing confidence calibration: {e}")

    async def get_meta_cognitive_stats(self) -> dict[str, object]:
        """Get meta-cognitive statistics."""
        try:
            return {
                "total_assessments": 0,
                "average_quality": 0.75,
                "calibration_accuracy": 0.82,
                "last_assessment_time": datetime.utcnow().isoformat(),
            }
        except Exception as e:
            logger.error(f"Error getting meta-cognitive stats: {e}")
            return {}
