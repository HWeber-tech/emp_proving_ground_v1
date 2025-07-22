"""
Sentient Adaptation Engine
The apex predator of real-time learning and adaptation.
"""

import asyncio
import logging
from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Optional, Any
import uuid

from src.core.events import LearningSignal, ContextPacket, TacticalAdaptation
from src.thinking.learning.real_time_learner import RealTimeLearningEngine
from src.thinking.memory.faiss_memory import FAISSPatternMemory
from src.thinking.learning.meta_cognition_engine import MetaCognitionEngine
from src.thinking.adaptation.tactical_adaptation_engine import TacticalAdaptationEngine
from src.operational.state_store import StateStore

logger = logging.getLogger(__name__)


class SentientAdaptationEngine:
    """
    The apex predator of real-time learning and adaptation.
    
    Features:
    - Real-time learning from trade outcomes
    - FAISS-based pattern memory
    - Meta-cognitive assessment
    - Tactical adaptation generation
    - Self-improving intelligence
    """
    
    def __init__(self, state_store: StateStore):
        self.state_store = state_store
        
        # Core components
        self.real_time_learner = RealTimeLearningEngine(state_store)
        self.pattern_memory = FAISSPatternMemory(state_store)
        self.meta_cognition = MetaCognitionEngine(state_store)
        self.tactical_adaptation = TacticalAdaptationEngine(state_store, self.pattern_memory)
        
        # State tracking
        self._adaptation_history_key = "emp:adaptation_history"
        self._performance_metrics_key = "emp:performance_metrics"
        
    async def initialize(self) -> None:
        """Initialize the sentient adaptation engine."""
        await self.pattern_memory.initialize()
        logger.info("Sentient adaptation engine initialized")
    
    async def adapt_in_real_time(
        self,
        market_event: ContextPacket,
        strategy_response: Dict[str, Any],
        outcome: Dict[str, Any]
    ) -> LearningSignal:
        """
        The apex method for real-time adaptation.
        
        Args:
            market_event: The market context that triggered the trade
            strategy_response: The strategy's response to the market event
            outcome: The actual outcome of the trade
            
        Returns:
            Learning signal with adaptation decisions
        """
        try:
            # Step 1: Create learning signal from outcome
            learning_signal = await self._create_learning_signal(
                market_event,
                strategy_response,
                outcome
            )
            
            # Step 2: Store experience in pattern memory
            await self.pattern_memory.add_experience(learning_signal)
            
            # Step 3: Meta-cognitive assessment
            historical_performance = await self._get_historical_performance()
            quality_assessment = await self.meta_cognition.assess_learning_quality(
                learning_signal,
                historical_performance
            )
            
            # Step 4: Generate adaptations if quality is sufficient
            adaptations = []
            if quality_assessment['should_adapt']:
                current_strategy_state = await self._get_current_strategy_state()
                adaptations = await self.tactical_adaptation.generate_adaptations(
                    learning_signal,
                    current_strategy_state
                )
                
                # Step 5: Apply adaptations
                strategy_id = strategy_response.get('strategy_id', 'default')
                await self.tactical_adaptation.apply_adaptations(adaptations, strategy_id)
            
            # Step 6: Calibrate confidence
            calibrated_confidence = await self.meta_cognition.calibrate_confidence(
                learning_signal,
                historical_performance
            )
            
            # Step 7: Store adaptation history
            await self._store_adaptation_record(
                learning_signal,
                quality_assessment,
                adaptations,
                calibrated_confidence
            )
            
            logger.info(
                f"Sentient adaptation complete: "
                f"Trade={learning_signal.trade_id}, "
                f"Quality={quality_assessment['learning_quality']:.2f}, "
                f"Adaptations={len(adaptations)}, "
                f"Confidence={calibrated_confidence:.2f}"
            )
            
            return learning_signal
            
        except Exception as e:
            logger.error(f"Error in sentient adaptation: {e}")
            return await self._create_fallback_learning_signal(market_event, outcome)
    
    async def _create_learning_signal(
        self,
        market_event: ContextPacket,
        strategy_response: Dict[str, Any],
        outcome: Dict[str, Any]
    ) -> LearningSignal:
        """Create a learning signal from trade data."""
        try:
            # Extract outcome data
            pnl = Decimal(str(outcome.get('pnl', 0)))
            duration = float(outcome.get('duration', 0))
            confidence = Decimal(str(outcome.get('confidence', 0.5)))
            
            # Create learning signal
            learning_signal = LearningSignal(
                event_id=str(uuid.uuid4()),
                timestamp=datetime.utcnow(),
                source="SentientAdaptationEngine",
                correlation_id=str(uuid.uuid4()),
                triggering_context=market_event,
                outcome_pnl=pnl,
                trade_duration_seconds=duration,
                confidence_of_outcome=confidence,
                trade_id=strategy_response.get('trade_id', str(uuid.uuid4())),
                metadata={
                    'strategy_id': strategy_response.get('strategy_id', 'unknown'),
                    'market_regime': market_event.regime,
                    'adaptation_triggered': True
                }
            )
            
            return learning_signal
            
        except Exception as e:
            logger.error(f"Error creating learning signal: {e}")
            return await self._create_fallback_learning_signal(market_event, outcome)
    
    async def _create_fallback_learning_signal(
        self,
        market_event: ContextPacket,
        outcome: Dict[str, Any]
    ) -> LearningSignal:
        """Create a fallback learning signal when processing fails."""
        return LearningSignal(
            event_id=str(uuid.uuid4()),
            timestamp=datetime.utcnow(),
            source="SentientAdaptationEngine",
            correlation_id=str(uuid.uuid4()),
            triggering_context=market_event,
            outcome_pnl=Decimal(str(outcome.get('pnl', 0))),
            trade_duration_seconds=float(outcome.get('duration', 0)),
            confidence_of_outcome=Decimal('0.1'),
            trade_id=str(uuid.uuid4()),
            metadata={'fallback': True}
        )
    
    async def _get_historical_performance(self) -> Dict[str, Any]:
        """Get historical performance metrics."""
        try:
            data = await self.state_store.get(self._performance_metrics_key)
            if data:
                return eval(data)
            return {
                'total_trades': 0,
                'win_rate': 0.5,
                'avg_pnl': 0.0,
                'sharpe_ratio': 0.0
            }
        except Exception as e:
            logger.error(f"Error getting historical performance: {e}")
            return {
                'total_trades': 0,
                'win_rate': 0.5,
                'avg_pnl': 0.0,
                'sharpe_ratio': 0.0
            }
    
    async def _get_current_strategy_state(self) -> Dict[str, Any]:
        """Get current strategy state."""
        try:
            # This would be enhanced with actual retrieval
            return {
                'position_size_multiplier': 1.0,
                'stop_loss_multiplier': 1.0,
                'take_profit_multiplier': 1.0,
                'max_position_size': 0.1,
                'max_drawdown_threshold': 0.05
            }
        except Exception as e:
            logger.error(f"Error getting current strategy state: {e}")
            return {}
    
    async def _store_adaptation_record(
        self,
        learning_signal: LearningSignal,
        quality_assessment: Dict[str, Any],
        adaptations: List[TacticalAdaptation],
        calibrated_confidence: Decimal
    ) -> None:
        """Store adaptation record for tracking."""
        try:
            record = {
                'trade_id': learning_signal.trade_id,
                'timestamp': datetime.utcnow().isoformat(),
                'learning_quality': quality_assessment['learning_quality'],
                'adaptations_applied': len(adaptations),
                'calibrated_confidence': float(calibrated_confidence),
                'market_regime': learning_signal.triggering_context.regime,
                'outcome_pnl': float(learning_signal.outcome_pnl),
                'adaptations': [a.dict() for a in adaptations] if adaptations else []
            }
            
            key = f"{self._adaptation_history_key}:{learning_signal.trade_id}"
            await self.state_store.set(
                key,
                str(record),
                expire=86400 * 30  # 30 days
            )
        except Exception as e:
            logger.error(f"Error storing adaptation record: {e}")
    
    async def get_sentient_stats(self) -> Dict[str, Any]:
        """Get comprehensive sentient adaptation statistics."""
        try:
            # Get component statistics
            pattern_stats = await self.pattern_memory.get_pattern_statistics()
            meta_stats = await self.meta_cognition.get_meta_cognitive_stats()
            adaptation_stats = await self.tactical_adaptation.get_adaptation_statistics()
            
            return {
                'pattern_memory': pattern_stats,
                'meta_cognition': meta_stats,
                'adaptations': adaptation_stats,
                'engine_status': 'active',
                'last_update': datetime.utcnow().isoformat()
            }
        except Exception as e:
            logger.error(f"Error getting sentient stats: {e}")
            return {
                'pattern_memory': {},
                'meta_cognition': {},
                'adaptations': {},
                'engine_status': 'error',
                'last_update': datetime.utcnow().isoformat()
            }
    
    async def reset_sentient_memory(self) -> bool:
        """Reset all sentient memory and start fresh."""
        try:
            await self.pattern_memory.clear_memory()
            
            # Clear adaptation history
            keys = await self.state_store.keys(f"{self._adaptation_history_key}:*")
            for key in keys:
                await self.state_store.delete(key)
            
            logger.info("Sentient adaptation engine memory reset")
            return True
            
        except Exception as e:
            logger.error(f"Error resetting sentient memory: {e}")
            return False
    
    async def get_learning_summary(self, days: int = 7) -> Dict[str, Any]:
        """Get summary of learning over specified period."""
        try:
            # This would be enhanced with actual retrieval
            return {
                'total_trades': 0,
                'successful_adaptations': 0,
                'average_improvement': 0.0,
                'best_performing_regime': 'unknown',
                'learning_rate': 0.01
            }
        except Exception as e:
            logger.error(f"Error getting learning summary: {e}")
            return {}
