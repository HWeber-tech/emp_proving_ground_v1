"""
Profit fitness dimension for strategy evaluation.

Measures profit generation capability with risk-adjusted metrics.
"""

import numpy as np
from typing import Dict, Any
from .base_fitness import BaseFitness, FitnessResult


class ProfitFitness(BaseFitness):
    """Profit optimization with risk-adjusted returns."""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.target_return = self.config.get('target_return', 0.20)  # 20% annual
        self.target_sharpe = self.config.get('target_sharpe', 1.5)
        self.target_sortino = self.config.get('target_sortino', 2.0)
        self.risk_free_rate = self.config.get('risk_free_rate', 0.02)
        
    def calculate_fitness(self, 
                         strategy_performance: Dict[str, Any],
                         market_data: Dict[str, Any],
                         regime_data: Dict[str, Any]) -> FitnessResult:
        """Calculate profit fitness score."""
        try:
            if not self.validate_inputs(strategy_performance):
                return FitnessResult(
                    score=0.0,
                    raw_value=0.0,
                    confidence=0.0,
                    metadata={'error': 'Invalid inputs'},
                    weight=self.weight
                )
            
            # Extract performance metrics
            total_return = strategy_performance.get('total_return', 0.0)
            returns = strategy_performance.get('returns', [])
            volatility = strategy_performance.get('volatility', 0.0)
            downside_volatility = strategy_performance.get('downside_volatility', 0.0)
            
            if not returns or len(returns) < 2:
                return FitnessResult(
                    score=0.0,
                    raw_value=0.0,
                    confidence=0.0,
                    metadata={'error': 'Insufficient data'},
                    weight=self.weight
                )
            
            # Calculate risk-adjusted metrics
            annual_return = self._annualize_return(total_return, len(returns))
            annual_vol = self._annualize_volatility(returns)
            annual_downside_vol = self._annualize_volatility([r for r in returns if r < 0])
            
            # Sharpe ratio
            sharpe = (annual_return - self.risk_free_rate) / annual_vol if annual_vol > 0 else 0
            
            # Sortino ratio
            sortino = (annual_return - self.risk_free_rate) / annual_downside_vol if annual_downside_vol > 0 else 0
            
            # Calmar ratio (return / max drawdown)
            max_drawdown = strategy_performance.get('max_drawdown', 0.01)
            calmar = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
            
            # Composite profit score
            return_score = min(annual_return / self.target_return, 2.0)
            sharpe_score = min(sharpe / self.target_sharpe, 2.0)
            sortino_score = min(sortino / self.target_sortino, 2.0)
            calmar_score = min(calmar / 3.0, 2.0)  # Target Calmar of 3.0
            
            # Weighted composite
            composite_score = (
                return_score * 0.3 +
                sharpe_score * 0.3 +
                sortino_score * 0.2 +
                calmar_score * 0.2
            )
            
            # Normalize to 0-1
            final_score = min(1.0, composite_score / 2.0)
            
            # Calculate confidence
            confidence = self.calculate_confidence(len(returns))
            
            metadata = {
                'annual_return': annual_return,
                'sharpe_ratio': sharpe,
                'sortino_ratio': sortino,
                'calmar_ratio': calmar,
                'volatility': annual_vol,
                'downside_volatility': annual_downside_vol
            }
            
            return FitnessResult(
                score=final_score,
                raw_value=annual_return,
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
            'TRENDING_UP': 1.3,      # Emphasize profit in trending markets
            'TRENDING_DOWN': 1.1,    # Still important in down trends
            'RANGING': 0.9,          # Less emphasis in ranging
            'VOLATILE': 1.2,         # Profit from volatility
            'CRISIS': 0.7,           # Focus on survival during crisis
            'RECOVERY': 1.4,         # Maximize recovery gains
            'LOW_VOLATILITY': 1.0,   # Standard weight
            'HIGH_VOLATILITY': 1.2   # Profit from high vol
        }
        return weights.get(market_regime, 1.0)
    
    def get_required_keys(self) -> list:
        """Return required data keys."""
        return ['total_return', 'returns', 'volatility', 'max_drawdown']
    
    def _annualize_return(self, total_return: float, periods: int) -> float:
        """Annualize return based on number of periods."""
        if periods == 0:
            return 0.0
        # Assume daily data, 252 trading days per year
        return (1 + total_return) ** (252 / periods) - 1
    
    def _annualize_volatility(self, returns: list) -> float:
        """Annualize volatility."""
        if len(returns) < 2:
            return 0.0
        return np.std(returns) * np.sqrt(252)
