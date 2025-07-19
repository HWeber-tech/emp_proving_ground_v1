"""
EMP Fitness Evaluator v1.1

Fitness evaluation orchestrator for the simulation envelope.
Orchestrates thinking layer analysis without performing calculations itself.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass

from ...core.events import TradeIntent, FitnessReport, PerformanceMetrics, RiskMetrics
from ...core.event_bus import publish_event, EventType
from ...thinking.analysis.performance_analyzer import PerformanceAnalyzer
from ...thinking.analysis.risk_analyzer import RiskAnalyzer
from ...governance.fitness_store import FitnessStore

logger = logging.getLogger(__name__)


@dataclass
class EvaluationContext:
    """Context for fitness evaluation."""
    strategy_id: str
    genome_id: str
    generation: int
    initial_capital: float
    evaluation_period: int
    market_data: Optional[List[Any]] = None
    metadata: Optional[Dict[str, Any]] = None


class FitnessEvaluator:
    """Fitness evaluator that orchestrates thinking layer analysis."""
    
    def __init__(self, fitness_store: Optional[FitnessStore] = None):
        self.fitness_store = fitness_store or FitnessStore()
        self.performance_analyzer = PerformanceAnalyzer()
        self.risk_analyzer = RiskAnalyzer()
        self.evaluation_history: List[Dict[str, Any]] = []
        
        logger.info("Fitness Evaluator initialized")
        
    async def evaluate_fitness(self, trade_history: List[TradeIntent], 
                             context: EvaluationContext) -> FitnessReport:
        """Evaluate fitness by orchestrating thinking layer analysis."""
        try:
            if not trade_history:
                logger.warning("Empty trade history provided for fitness evaluation")
                return self._create_default_fitness_report(context)
                
            # Step 1: Orchestrate performance analysis
            performance_result = await self._orchestrate_performance_analysis(trade_history, context)
            
            # Step 2: Orchestrate risk analysis
            risk_result = await self._orchestrate_risk_analysis(trade_history, context)
            
            # Step 3: Extract metrics from thinking layer results
            performance_metrics = self._extract_performance_metrics(performance_result)
            risk_metrics = self._extract_risk_metrics(risk_result)
            
            # Step 4: Calculate fitness score using governance layer
            fitness_score = self._calculate_fitness_score(performance_metrics, risk_metrics, context)
            
            # Step 5: Create fitness report
            fitness_report = FitnessReport(
                timestamp=datetime.now(),
                genome_id=context.genome_id,
                strategy_id=context.strategy_id,
                performance_metrics=performance_metrics,
                risk_metrics=risk_metrics,
                fitness_score=fitness_score,
                generation=context.generation,
                metadata={
                    "evaluator_version": "1.1.0",
                    "method": "orchestrated_thinking_analysis",
                    "evaluation_period": context.evaluation_period,
                    "initial_capital": context.initial_capital,
                    "trade_count": len(trade_history)
                }
            )
            
            # Step 6: Publish fitness report event
            await publish_event(fitness_report)
            
            # Step 7: Store evaluation history
            self._store_evaluation_history(fitness_report, context)
            
            logger.info(f"Fitness evaluation completed: {fitness_score:.4f} for {context.strategy_id}")
            return fitness_report
            
        except Exception as e:
            logger.error(f"Error evaluating fitness: {e}")
            return self._create_default_fitness_report(context)
            
    async def _orchestrate_performance_analysis(self, trade_history: List[TradeIntent], 
                                              context: EvaluationContext) -> Any:
        """Orchestrate performance analysis through thinking layer."""
        try:
            # Delegate to thinking layer
            performance_result = self.performance_analyzer.analyze_performance(
                trade_history, context.initial_capital
            )
            
            logger.debug(f"Performance analysis orchestrated for {context.strategy_id}")
            return performance_result
            
        except Exception as e:
            logger.error(f"Error orchestrating performance analysis: {e}")
            raise
            
    async def _orchestrate_risk_analysis(self, trade_history: List[TradeIntent], 
                                       context: EvaluationContext) -> Any:
        """Orchestrate risk analysis through thinking layer."""
        try:
            # Delegate to thinking layer
            risk_result = self.risk_analyzer.analyze_risk(
                trade_history, context.market_data
            )
            
            logger.debug(f"Risk analysis orchestrated for {context.strategy_id}")
            return risk_result
            
        except Exception as e:
            logger.error(f"Error orchestrating risk analysis: {e}")
            raise
            
    def _extract_performance_metrics(self, performance_result: Any) -> PerformanceMetrics:
        """Extract performance metrics from thinking layer result."""
        try:
            if hasattr(performance_result, 'result') and 'performance_metrics' in performance_result.result:
                metrics_dict = performance_result.result['performance_metrics']
                return PerformanceMetrics(**metrics_dict)
            else:
                logger.warning("Could not extract performance metrics from result")
                return PerformanceMetrics()
                
        except Exception as e:
            logger.error(f"Error extracting performance metrics: {e}")
            return PerformanceMetrics()
            
    def _extract_risk_metrics(self, risk_result: Any) -> RiskMetrics:
        """Extract risk metrics from thinking layer result."""
        try:
            if hasattr(risk_result, 'result') and 'risk_metrics' in risk_result.result:
                metrics_dict = risk_result.result['risk_metrics']
                return RiskMetrics(**metrics_dict)
            else:
                logger.warning("Could not extract risk metrics from result")
                return RiskMetrics()
                
        except Exception as e:
            logger.error(f"Error extracting risk metrics: {e}")
            return RiskMetrics()
            
    def _calculate_fitness_score(self, performance_metrics: PerformanceMetrics, 
                               risk_metrics: RiskMetrics, 
                               context: EvaluationContext) -> float:
        """Calculate fitness score using governance layer."""
        try:
            # Use governance layer fitness store
            fitness_score = self.fitness_store.calculate_fitness(
                performance_metrics, risk_metrics
            )
            
            return fitness_score
            
        except Exception as e:
            logger.error(f"Error calculating fitness score: {e}")
            return 0.0
            
    def _store_evaluation_history(self, fitness_report: FitnessReport, 
                                context: EvaluationContext):
        """Store evaluation in history."""
        history_entry = {
            "timestamp": fitness_report.timestamp,
            "strategy_id": context.strategy_id,
            "genome_id": context.genome_id,
            "generation": context.generation,
            "fitness_score": fitness_report.fitness_score,
            "performance_metrics": fitness_report.performance_metrics.__dict__,
            "risk_metrics": fitness_report.risk_metrics.__dict__,
            "evaluation_period": context.evaluation_period
        }
        
        self.evaluation_history.append(history_entry)
        
    def _create_default_fitness_report(self, context: EvaluationContext) -> FitnessReport:
        """Create default fitness report when evaluation fails."""
        return FitnessReport(
            timestamp=datetime.now(),
            genome_id=context.genome_id,
            strategy_id=context.strategy_id,
            performance_metrics=PerformanceMetrics(),
            risk_metrics=RiskMetrics(),
            fitness_score=0.0,
            generation=context.generation,
            metadata={
                "evaluator_version": "1.1.0",
                "method": "default_fallback",
                "error": "Evaluation failed"
            }
        )
        
    async def evaluate_batch(self, evaluations: List[tuple]) -> List[FitnessReport]:
        """Evaluate fitness for a batch of strategies."""
        results = []
        
        for trade_history, context in evaluations:
            try:
                result = await self.evaluate_fitness(trade_history, context)
                results.append(result)
            except Exception as e:
                logger.error(f"Error in batch evaluation: {e}")
                default_result = self._create_default_fitness_report(context)
                results.append(default_result)
                
        return results
        
    def get_evaluation_history(self, strategy_id: Optional[str] = None, 
                             limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get evaluation history."""
        history = self.evaluation_history
        
        if strategy_id:
            history = [h for h in history if h['strategy_id'] == strategy_id]
            
        if limit:
            history = history[-limit:]
            
        return history
        
    def get_evaluation_statistics(self) -> Dict[str, Any]:
        """Get statistics about fitness evaluations."""
        if not self.evaluation_history:
            return {}
            
        fitness_scores = [h['fitness_score'] for h in self.evaluation_history]
        generations = [h['generation'] for h in self.evaluation_history]
        
        return {
            'total_evaluations': len(self.evaluation_history),
            'average_fitness': np.mean(fitness_scores),
            'fitness_std': np.std(fitness_scores),
            'best_fitness': max(fitness_scores) if fitness_scores else 0,
            'worst_fitness': min(fitness_scores) if fitness_scores else 0,
            'generations_evaluated': len(set(generations)),
            'latest_generation': max(generations) if generations else 0
        }
        
    def clear_history(self):
        """Clear evaluation history."""
        self.evaluation_history.clear()
        logger.info("Fitness evaluation history cleared") 