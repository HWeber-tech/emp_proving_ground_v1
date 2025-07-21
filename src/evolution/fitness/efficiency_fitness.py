"""
Efficiency fitness dimension for strategy evaluation.

Measures resource utilization and trade efficiency optimization.
"""

import numpy as np
from typing import Dict, Any
from .base_fitness import BaseFitness, FitnessResult


class EfficiencyFitness(BaseFitness):
    """Efficiency optimization through resource utilization and trade optimization."""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.target_trades_per_month = self.config.get('target_trades_per_month', 20)
        self.target_profit_per_trade = self.config.get('target_profit_per_trade', 0.005)  # 0.5%
        self.max_slippage = self.config.get('max_slippage', 0.001)  # 0.1%
        
    def calculate_fitness(self, 
                         strategy_performance: Dict[str, Any],
                         market_data: Dict[str, Any],
                         regime_data: Dict[str, Any]) -> FitnessResult:
        """Calculate efficiency fitness score."""
        try:
            if not self.validate_inputs(strategy_performance):
                return FitnessResult(
                    score=0.0,
                    raw_value=0.0,
                    confidence=0.0,
                    metadata={'error': 'Invalid inputs'},
                    weight=self.weight
                )
            
            # Extract efficiency metrics
            total_trades = strategy_performance.get('total_trades', 0)
            total_return = strategy_performance.get('total_return', 0.0)
            returns = strategy_performance.get('returns', [])
            slippage = strategy_performance.get('avg_slippage', 0.0)
            transaction_costs = strategy_performance.get('transaction_costs', 0.0)
            holding_periods = strategy_performance.get('avg_holding_period', 1.0)
            
            if total_trades == 0 or not returns:
                return FitnessResult(
                    score=0.0,
                    raw_value=0.0,
                    confidence=0.0,
                    metadata={'error': 'No trades or returns'},
                    weight=self.weight
                )
            
            # Trade frequency score
            trades_per_month = total_trades / 12  # Assume 1 year data
            frequency_score = min(1.0, trades_per_month / self.target_trades_per_month)
            
            # Profit per trade score
            profit_per_trade = total_return / total_trades
            profit_score = min(1.0, profit_per_trade / self.target_profit_per_trade)
            
            # Slippage score (lower slippage = higher score)
            slippage_score = max(0.0, 1.0 - slippage / self.max_slippage)
            
            # Transaction cost efficiency
            cost_efficiency = self._calculate_cost_efficiency(transaction_costs, total_return)
            
            # Capital utilization
            utilization_score = self._calculate_utilization_score(returns, holding_periods)
            
            # Composite efficiency score
            composite_score = (
                frequency_score * 0.2 +
                profit_score * 0.3 +
                slippage_score * 0.2 +
                cost_efficiency * 0.15 +
                utilization_score * 0.15
            )
            
            final_score = max(0.0, min(1.0, composite_score))
            
            # Calculate confidence
            confidence = self.calculate_confidence(total_trades)
            
            metadata = {
                'frequency_score': frequency_score,
                'profit_score': profit_score,
                'slippage_score': slippage_score,
                'cost_efficiency': cost_efficiency,
                'utilization_score': utilization_score,
                'trades_per_month': trades_per_month,
                'profit_per_trade': profit_per_trade,
                'avg_slippage': slippage
            }
            
            return FitnessResult(
                score=final_score,
                raw_value=profit_per_trade,
                confidence=confidence,
                metadata=metadata,
                weight=self.weight
            )
            
        except Exception as e:
            return FitnessResult(
                score=0.0,
                raw_value=0.0,
                confidence=0.0,
                metadata={'error': str(e)},
                weight=self.weight
            )
    
    def get_optimal_weight(self, market_regime: str) -> float:
        """Get optimal weight based on market regime."""
        weights = {
            'TRENDING_UP': 1.1,      # Important for trend efficiency
            'TRENDING_DOWN': 1.0,    # Standard importance
            'RANGING': 1.3,          # Critical for ranging efficiency
            'VOLATILE': 1.2,         # Important for volatile efficiency
            'CRISIS': 0.9,           # Less focus on efficiency during crisis
            'RECOVERY': 1.1,         # Important for recovery efficiency
            'LOW_VOLATILITY': 1.2,   # High importance in low vol
            'HIGH_VOLATILITY': 1.1   # Moderate importance in high vol
        }
        return weights.get(market_regime, 1.0)
    
    def get_required_keys(self) -> list:
        """Return required data keys."""
        return ['total_trades', 'total_return', 'returns', 'avg_slippage', 'transaction_costs', 'avg_holding_period']
    
    def _calculate_cost_efficiency(self, transaction_costs: float, total_return: float) -> float:
        """Calculate transaction cost efficiency."""
        if total_return == 0:
            return 0.5
        
        # Cost ratio (lower is better)
        cost_ratio = abs(transaction_costs / total_return)
        
        # Efficiency score
        max_acceptable_ratio = 0.2  # 20% of returns
        efficiency_score = max(0.0, 1.0 - min(cost_ratio / max_acceptable_ratio, 1.0))
        
        return efficiency_score
    
    def _calculate_utilization_score(self, returns: list, holding_periods: float) -> float:
        """Calculate capital utilization score."""
        if not returns or holding_periods == 0:
            return 0.5
        
        # Utilization based on return frequency and holding periods
        annual_turnover = 252 / holding_periods  # Trading days per year
        
        # Utilization score (optimal turnover)
        optimal_turnover = 12  # 12 trades per year
        utilization_score = max(0.0, 1.0 - abs(annual_turnover - optimal_turnover) / optimal_turnover)
        
        return utilization_score
