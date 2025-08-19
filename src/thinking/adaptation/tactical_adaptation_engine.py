"""
Tactical Adaptation Engine
Generates real-time tactical adjustments based on learning signals and pattern memory.
"""

import logging
from ast import literal_eval
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, List

try:
    from src.core.events import ContextPacket, LearningSignal, TacticalAdaptation  # legacy
except Exception:  # pragma: no cover
    LearningSignal = TacticalAdaptation = ContextPacket = object  # type: ignore
from src.core.state_store import StateStore
from src.thinking.memory.faiss_memory import FAISSPatternMemory

logger = logging.getLogger(__name__)


class TacticalAdaptationEngine:
    """
    Generates tactical adaptations based on real-time learning and pattern memory.
    
    Features:
    - Real-time adaptation generation
    - Pattern-based decision making
    - Confidence-weighted adjustments
    - Adaptive parameter tuning
    """
    
    def __init__(self, state_store: StateStore, pattern_memory: FAISSPatternMemory):
        self.state_store = state_store
        self.pattern_memory = pattern_memory
        self._adaptation_key = "emp:tactical_adaptations"
        self._strategy_params_key = "emp:strategy_parameters"
        
    async def generate_adaptations(
        self,
        learning_signal: LearningSignal,
        current_strategy_state: Dict[str, Any]
    ) -> List[TacticalAdaptation]:
        """
        Generate tactical adaptations based on learning signal.
        
        Args:
            learning_signal: The learning signal from trade outcome
            current_strategy_state: Current strategy parameters
            
        Returns:
            List of tactical adaptations to apply
        """
        try:
            adaptations = []
            
            # Find similar historical experiences
            similar_experiences = await self.pattern_memory.find_similar_experiences(
                query_vector=learning_signal.triggering_context.latent_vec,
                max_results=10,
                min_confidence=0.6,
                time_window=timedelta(days=7)
            )
            
            if not similar_experiences:
                logger.debug("No similar experiences found for adaptation")
                return adaptations
            
            # Analyze patterns from similar experiences
            pattern_analysis = await self._analyze_patterns(
                similar_experiences,
                learning_signal
            )
            
            # Generate adaptations based on analysis
            adaptations.extend(
                await self._generate_parameter_adaptations(
                    pattern_analysis,
                    current_strategy_state
                )
            )
            
            adaptations.extend(
                await self._generate_risk_adaptations(
                    pattern_analysis,
                    current_strategy_state
                )
            )
            
            # Store adaptations
            for adaptation in adaptations:
                await self._store_adaptation(adaptation)
            
            logger.info(
                f"Generated {len(adaptations)} tactical adaptations "
                f"based on learning signal {learning_signal.trade_id}"
            )
            
            return adaptations
            
        except Exception as e:
            logger.error(f"Error generating adaptations: {e}")
            return []
    
    async def _analyze_patterns(
        self,
        similar_experiences: List,
        learning_signal: LearningSignal
    ) -> Dict[str, Any]:
        """Analyze patterns from similar experiences."""
        try:
            # Extract outcomes and contexts
            outcomes = [exp[1].outcome_pnl for exp in similar_experiences]
            contexts = [exp[1].context for exp in similar_experiences]
            
            # Calculate statistics
            avg_outcome = sum(outcomes) / len(outcomes) if outcomes else 0
            win_rate = len([o for o in outcomes if o > 0]) / len(outcomes) if outcomes else 0
            
            # Analyze regime patterns
            regimes = [ctx.regime for ctx in contexts]
            regime_distribution = {}
            for regime in regimes:
                regime_distribution[regime] = regime_distribution.get(regime, 0) + 1
            
            # Calculate confidence-weighted outcomes
            weighted_outcomes = [
                exp[1].outcome_pnl * exp[0]  # outcome * similarity_score
                for exp in similar_experiences
            ]
            weighted_avg = sum(weighted_outcomes) / len(weighted_outcomes) if weighted_outcomes else 0
            
            return {
                'avg_outcome': avg_outcome,
                'win_rate': win_rate,
                'weighted_avg': weighted_avg,
                'regime_distribution': regime_distribution,
                'num_similar': len(similar_experiences),
                'current_outcome': float(learning_signal.outcome_pnl)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing patterns: {e}")
            return {
                'avg_outcome': 0,
                'win_rate': 0,
                'weighted_avg': 0,
                'regime_distribution': {},
                'num_similar': 0,
                'current_outcome': 0
            }
    
    async def _generate_parameter_adaptations(
        self,
        pattern_analysis: Dict[str, Any],
        current_strategy_state: Dict[str, Any]
    ) -> List[TacticalAdaptation]:
        """Generate parameter-based adaptations."""
        adaptations = []
        
        try:
            # Determine if we need to adjust based on pattern analysis
            avg_outcome = pattern_analysis['avg_outcome']
            current_outcome = pattern_analysis['current_outcome']
            
            # Calculate adjustment factor
            if avg_outcome > 0 and current_outcome < 0:
                # Current strategy is underperforming similar contexts
                adjustment_factor = Decimal('1.1')  # Increase aggressiveness
                reason = "Underperforming in similar contexts"
            elif avg_outcome < 0 and current_outcome > 0:
                # Current strategy is outperforming similar contexts
                adjustment_factor = Decimal('0.9')  # Decrease aggressiveness
                reason = "Outperforming in similar contexts"
            else:
                # Performance is aligned with patterns
                adjustment_factor = Decimal('1.0')
                reason = "Performance aligned with patterns"
            
            # Generate specific parameter adaptations
            if adjustment_factor != Decimal('1.0'):
                # Position sizing
                adaptations.append(TacticalAdaptation(
                    parameter_to_adjust="position_size_multiplier",
                    adjustment_factor=adjustment_factor,
                    reason=reason,
                    duration_ticks=100,
                    confidence=Decimal(str(pattern_analysis['win_rate'])),
                    metadata={'pattern_based': True}
                ))
                
                # Stop loss adjustment
                stop_loss_factor = Decimal('1.0') / adjustment_factor
                adaptations.append(TacticalAdaptation(
                    parameter_to_adjust="stop_loss_multiplier",
                    adjustment_factor=stop_loss_factor,
                    reason=f"Risk adjustment: {reason}",
                    duration_ticks=100,
                    confidence=Decimal(str(pattern_analysis['win_rate'])),
                    metadata={'pattern_based': True}
                ))
                
                # Take profit adjustment
                take_profit_factor = adjustment_factor
                adaptations.append(TacticalAdaptation(
                    parameter_to_adjust="take_profit_multiplier",
                    adjustment_factor=take_profit_factor,
                    reason=f"Profit target adjustment: {reason}",
                    duration_ticks=100,
                    confidence=Decimal(str(pattern_analysis['win_rate'])),
                    metadata={'pattern_based': True}
                ))
        
        except Exception as e:
            logger.error(f"Error generating parameter adaptations: {e}")
        
        return adaptations
    
    async def _generate_risk_adaptations(
        self,
        pattern_analysis: Dict[str, Any],
        current_strategy_state: Dict[str, Any]
    ) -> List[TacticalAdaptation]:
        """Generate risk-based adaptations."""
        adaptations = []
        
        try:
            win_rate = pattern_analysis['win_rate']
            
            # Adjust risk based on win rate
            if win_rate < 0.4:
                # Low win rate - reduce risk
                adaptations.append(TacticalAdaptation(
                    parameter_to_adjust="max_position_size",
                    adjustment_factor=Decimal('0.8'),
                    reason="Low win rate in similar contexts",
                    duration_ticks=200,
                    confidence=Decimal(str(1 - win_rate)),
                    metadata={'risk_based': True}
                ))
                
                adaptations.append(TacticalAdaptation(
                    parameter_to_adjust="max_drawdown_threshold",
                    adjustment_factor=Decimal('0.9'),
                    reason="Conservative adjustment for low win rate",
                    duration_ticks=200,
                    confidence=Decimal(str(1 - win_rate)),
                    metadata={'risk_based': True}
                ))
                
            elif win_rate > 0.7:
                # High win rate - increase risk tolerance
                adaptations.append(TacticalAdaptation(
                    parameter_to_adjust="max_position_size",
                    adjustment_factor=Decimal('1.2'),
                    reason="High win rate in similar contexts",
                    duration_ticks=200,
                    confidence=Decimal(str(win_rate)),
                    metadata={'risk_based': True}
                ))
        
        except Exception as e:
            logger.error(f"Error generating risk adaptations: {e}")
        
        return adaptations
    
    async def _store_adaptation(self, adaptation: TacticalAdaptation) -> None:
        """Store adaptation for tracking."""
        try:
            key = f"{self._adaptation_key}:{adaptation.parameter_to_adjust}"
            await self.state_store.set(
                key,
                adaptation.json(),
                expire=86400  # 1 day
            )
        except Exception as e:
            logger.error(f"Error storing adaptation: {e}")
    
    async def apply_adaptations(
        self,
        adaptations: List[TacticalAdaptation],
        strategy_id: str
    ) -> bool:
        """
        Apply tactical adaptations to strategy parameters.
        
        Args:
            adaptations: List of adaptations to apply
            strategy_id: ID of the strategy to modify
            
        Returns:
            True if adaptations were successfully applied
        """
        try:
            # Get current strategy parameters
            key = f"{self._strategy_params_key}:{strategy_id}"
            current_params = await self.state_store.get(key)
            
            if current_params:
                # Bandit B307: replaced eval with safe parsing
                try:
                    params = literal_eval(current_params)
                except (ValueError, SyntaxError):
                    params = {}
            else:
                params = {}
            
            # Apply adaptations
            for adaptation in adaptations:
                param_name = adaptation.parameter_to_adjust
                current_value = params.get(param_name, 1.0)
                new_value = float(current_value) * float(adaptation.adjustment_factor)
                
                # Ensure reasonable bounds
                if 'size' in param_name or 'threshold' in param_name:
                    new_value = max(0.1, min(5.0, new_value))
                else:
                    new_value = max(0.5, min(2.0, new_value))
                
                params[param_name] = new_value
            
            # Store updated parameters
            await self.state_store.set(
                key,
                str(params),
                expire=86400 * 7  # 7 days
            )
            
            logger.info(
                f"Applied {len(adaptations)} adaptations to strategy {strategy_id}"
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Error applying adaptations: {e}")
            return False
    
    async def get_active_adaptations(self, strategy_id: str) -> List[TacticalAdaptation]:
        """Get currently active adaptations for a strategy."""
        try:
            # This would be enhanced with actual retrieval
            return []
        except Exception as e:
            logger.error(f"Error getting active adaptations: {e}")
            return []
    
    async def get_adaptation_statistics(self) -> Dict[str, Any]:
        """Get statistics about tactical adaptations."""
        try:
            return {
                'total_adaptations_generated': 0,
                'success_rate': 0.85,
                'last_adaptation_time': datetime.utcnow().isoformat()
            }
        except Exception as e:
            logger.error(f"Error getting adaptation statistics: {e}")
            return {}
