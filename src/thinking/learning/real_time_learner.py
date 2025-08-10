"""
Real-Time Learning Engine
Processes trade outcomes and generates learning signals for immediate adaptation.
"""

import asyncio
import logging
from datetime import datetime
from decimal import Decimal
from typing import Dict, Optional, Any
import uuid

try:
    from src.core.events import LearningSignal, ContextPacket, ExecutionReport  # legacy
except Exception:  # pragma: no cover
    LearningSignal = ContextPacket = ExecutionReport = object  # type: ignore
from src.operational.state_store import StateStore

logger = logging.getLogger(__name__)


class RealTimeLearningEngine:
    """
    Processes trade outcomes in real-time to generate learning signals.
    
    Features:
    - Real-time trade outcome processing
    - Context-to-trade mapping
    - Confidence calculation
    - Learning signal generation
    """
    
    def __init__(self, state_store: StateStore):
        self.state_store = state_store
        self._trade_journal_key = "emp:trade_journal"
        self._context_mapping_key = "emp:context_mapping"
        
    async def process_trade_outcome(
        self,
        execution_report: ExecutionReport,
        portfolio_monitor: Any
    ) -> LearningSignal:
        """
        Process a completed trade and generate a learning signal.
        
        Args:
            execution_report: The execution report from the trade
            portfolio_monitor: Portfolio monitor for trade history
            
        Returns:
            LearningSignal with trade outcome analysis
        """
        try:
            # Get the original context that triggered this trade
            context_packet = await self._get_triggering_context(execution_report.trade_intent_id)
            
            if not context_packet:
                logger.warning(f"No context found for trade {execution_report.trade_intent_id}")
                return self._create_fallback_signal(execution_report)
            
            # Calculate trade metrics
            trade_metrics = await self._calculate_trade_metrics(
                execution_report,
                portfolio_monitor
            )
            
            # Calculate confidence of outcome
            confidence = self._calculate_confidence(execution_report, trade_metrics)
            
            # Create learning signal
            learning_signal = LearningSignal(
                event_id=str(uuid.uuid4()),
                timestamp=datetime.utcnow(),
                source="RealTimeLearningEngine",
                correlation_id=execution_report.correlation_id,
                triggering_context=context_packet,
                outcome_pnl=trade_metrics['pnl'],
                trade_duration_seconds=trade_metrics['duration'],
                confidence_of_outcome=confidence,
                trade_id=execution_report.trade_intent_id,
                metadata={
                    'execution_status': execution_report.status,
                    'fees': float(execution_report.fees),
                    'slippage': trade_metrics.get('slippage', 0.0),
                    'max_adverse_excursion': trade_metrics.get('max_adverse_excursion', 0.0),
                    'max_favorable_excursion': trade_metrics.get('max_favorable_excursion', 0.0)
                }
            )
            
            # Store the learning signal
            await self._store_learning_signal(learning_signal)
            
            logger.info(
                f"Generated learning signal for trade {execution_report.trade_intent_id}: "
                f"PNL={trade_metrics['pnl']:.4f}, "
                f"Confidence={confidence:.2f}, "
                f"Duration={trade_metrics['duration']:.1f}s"
            )
            
            return learning_signal
            
        except Exception as e:
            logger.error(f"Error processing trade outcome: {e}")
            return self._create_fallback_signal(execution_report)
    
    async def _get_triggering_context(self, trade_intent_id: str) -> Optional[ContextPacket]:
        """Retrieve the context packet that triggered a trade."""
        try:
            mapping_data = await self.state_store.get(self._context_mapping_key)
            if mapping_data:
                mapping = eval(mapping_data)  # Safe for internal use
                context_data = mapping.get(trade_intent_id)
                if context_data:
                    return ContextPacket(**context_data)
        except Exception as e:
            logger.error(f"Error retrieving triggering context: {e}")
        return None
    
    async def _calculate_trade_metrics(
        self,
        execution_report: ExecutionReport,
        portfolio_monitor: Any
    ) -> Dict[str, float]:
        """Calculate comprehensive trade metrics."""
        try:
            # Get trade history for this trade
            trade_history = await self._get_trade_history(execution_report.trade_intent_id)
            
            # Calculate P&L
            pnl = float(execution_report.metadata.get('pnl', 0))
            
            # Calculate duration
            start_time = trade_history.get('start_time', execution_report.timestamp)
            duration = (execution_report.timestamp - start_time).total_seconds()
            
            # Calculate slippage
            intended_price = float(trade_history.get('intended_price', float(execution_report.price)))
            actual_price = float(execution_report.price)
            slippage = abs(actual_price - intended_price) / intended_price
            
            # Get excursion metrics if available
            max_adverse_excursion = float(execution_report.metadata.get('max_adverse_excursion', 0))
            max_favorable_excursion = float(execution_report.metadata.get('max_favorable_excursion', 0))
            
            return {
                'pnl': pnl,
                'duration': duration,
                'slippage': slippage,
                'max_adverse_excursion': max_adverse_excursion,
                'max_favorable_excursion': max_favorable_excursion
            }
            
        except Exception as e:
            logger.error(f"Error calculating trade metrics: {e}")
            return {
                'pnl': 0.0,
                'duration': 0.0,
                'slippage': 0.0,
                'max_adverse_excursion': 0.0,
                'max_favorable_excursion': 0.0
            }
    
    def _calculate_confidence(
        self,
        execution_report: ExecutionReport,
        trade_metrics: Dict[str, float]
    ) -> Decimal:
        """
        Calculate confidence in the trade outcome.
        
        Factors:
        - Execution quality (slippage)
        - Trade duration (longer = less confidence)
        - Outcome clarity (large P&L = higher confidence)
        """
        try:
            confidence = Decimal('1.0')
            
            # Reduce confidence based on slippage
            slippage = trade_metrics.get('slippage', 0)
            if slippage > 0.01:  # 1% slippage
                confidence *= Decimal('0.8')
            elif slippage > 0.005:  # 0.5% slippage
                confidence *= Decimal('0.9')
            
            # Reduce confidence based on duration
            duration = trade_metrics.get('duration', 0)
            if duration > 3600:  # 1 hour
                confidence *= Decimal('0.9')
            elif duration > 86400:  # 1 day
                confidence *= Decimal('0.7')
            
            # Increase confidence for clear outcomes
            pnl = abs(trade_metrics.get('pnl', 0))
            if pnl > 0.02:  # 2% P&L
                confidence *= Decimal('1.1')
            elif pnl < 0.001:  # Very small P&L
                confidence *= Decimal('0.8')
            
            # Cap confidence between 0.1 and 1.0
            confidence = max(Decimal('0.1'), min(Decimal('1.0'), confidence))
            
            return confidence
            
        except Exception as e:
            logger.error(f"Error calculating confidence: {e}")
            return Decimal('0.5')
    
    def _create_fallback_signal(self, execution_report: ExecutionReport) -> LearningSignal:
        """Create a fallback learning signal when processing fails."""
        return LearningSignal(
            event_id=str(uuid.uuid4()),
            timestamp=datetime.utcnow(),
            source="RealTimeLearningEngine",
            correlation_id=execution_report.correlation_id,
            triggering_context=ContextPacket(
                event_id=str(uuid.uuid4()),
                timestamp=datetime.utcnow(),
                source="fallback",
                regime="unknown",
                confidence=Decimal('0.5'),
                latent_vec=[],
                patterns={},
                risk_metrics={},
                market_state={}
            ),
            outcome_pnl=Decimal('0'),
            trade_duration_seconds=0.0,
            confidence_of_outcome=Decimal('0.1'),
            trade_id=execution_report.trade_intent_id,
            metadata={'fallback': True}
        )
    
    async def _store_learning_signal(self, signal: LearningSignal) -> None:
        """Store the learning signal for future reference."""
        try:
            key = f"{self._trade_journal_key}:{signal.trade_id}"
            await self.state_store.set(
                key,
                signal.json(),
                expire=86400 * 30  # 30 days
            )
        except Exception as e:
            logger.error(f"Error storing learning signal: {e}")
    
    async def _get_trade_history(self, trade_id: str) -> Dict[str, Any]:
        """Retrieve trade history for a specific trade."""
        try:
            key = f"{self._trade_journal_key}:{trade_id}"
            data = await self.state_store.get(key)
            if data:
                return eval(data)  # Safe for internal use
        except Exception as e:
            logger.error(f"Error retrieving trade history: {e}")
        return {}
    
    async def store_context_mapping(
        self,
        trade_intent_id: str,
        context_packet: ContextPacket
    ) -> None:
        """Store mapping between trade intent and triggering context."""
        try:
            mapping_data = await self.state_store.get(self._context_mapping_key)
            if mapping_data:
                mapping = eval(mapping_data)
            else:
                mapping = {}
            
            mapping[trade_intent_id] = context_packet.dict()
            
            await self.state_store.set(
                self._context_mapping_key,
                str(mapping),
                expire=86400 * 7  # 7 days
            )
        except Exception as e:
            logger.error(f"Error storing context mapping: {e}")
    
    async def get_learning_statistics(self) -> Dict[str, Any]:
        """Get statistics about learning signals."""
        try:
            # This would be enhanced with actual statistics
            return {
                'total_signals_processed': 0,
                'average_confidence': 0.75,
                'last_signal_time': datetime.utcnow().isoformat()
            }
        except Exception as e:
            logger.error(f"Error getting learning statistics: {e}")
            return {}
