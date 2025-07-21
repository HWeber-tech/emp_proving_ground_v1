"""
Survival fitness dimension for strategy evaluation.

Measures strategy survival capability through drawdown control and longevity metrics.
"""

import numpy as np
from typing import Dict, Any
from .base_fitness import BaseFitness, FitnessResult


class SurvivalFitness(BaseFitness):
    """Survival optimization through drawdown control and longevity."""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.max_acceptable_drawdown = self.config.get('max_acceptable_drawdown', 0.15)  # 15%
        self.target_win_rate = self.config.get('target_win_rate', 0.55)  # 55%
        self.max_consecutive_losses = self.config.get('max_consecutive_losses', 5)
        
    def calculate_fitness(self, 
                         strategy_performance: Dict[str, Any],
                         market_data: Dict[str, Any],
                         regime_data: Dict[str, Any]) -> FitnessResult:
        """Calculate survival fitness score."""
        try:
            if not self.validate_inputs(strategy_performance):
                return FitnessResult(
                    score=0.0,
                    raw_value=0.0,
                    confidence=0.0,
                    metadata={'error': 'Invalid inputs'},
                    weight=self.weight
                )
            
            # Extract survival metrics
            max_drawdown = strategy_performance.get('max_drawdown', 0.0)
            win_rate = strategy_performance.get('win_rate', 0.0)
            consecutive_losses = strategy_performance.get('max_consecutive_losses', 0)
            returns = strategy_performance.get('returns', [])
            equity_curve = strategy_performance.get('equity_curve', [])
            
            if not returns:
                return FitnessResult(
                    score=0.0,
                    raw_value=0.0,
                    confidence=0.0,
                    metadata={'error': 'No returns data'},
                    weight=self.weight
                )
            
            # Drawdown score (inverse - lower drawdown = higher score)
            drawdown_score = max(0.0, 1.0 - abs(max_drawdown) / self.max_acceptable_drawdown)
            
            # Win rate score
            win_rate_score = min(1.0, win_rate / self.target_win_rate)
            
            # Consecutive losses score
            loss_score = max(0.0, 1.0 - consecutive_losses / self.max_consecutive_losses)
            
            # Recovery score (time to recover from drawdown)
            recovery_score = self._calculate_recovery_score(equity_curve, max_drawdown)
            
            # Stability score (volatility of returns)
            stability_score = self._calculate_stability_score(returns)
            
            # Composite survival score
            composite_score = (
                drawdown_score * 0.4 +
                win_rate_score * 0.3 +
                loss_score * 0.15 +
                recovery_score * 0.1 +
                stability_score * 0.05
            )
            
            # Ensure score is within 0-1 range
            final_score = max(0.0, min(1.0, composite_score))
            
            # Calculate confidence
            confidence = self.calculate_confidence(len(returns))
            
            metadata = {
                'max_drawdown': max_drawdown,
                'win_rate': win_rate,
                'consecutive_losses': consecutive_losses,
                'recovery_score': recovery_score,
                'stability_score': stability_score,
                'drawdown_score': drawdown_score
            }
            
            return FitnessResult(
                score=final_score,
                raw_value=1.0 - abs(max_drawdown),  # Higher is better
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
            'TRENDING_UP': 0.8,      # Less focus on survival in trends
            'TRENDING_DOWN': 1.5,    # High focus on survival in downtrends
            'RANGING': 1.0,          # Balanced approach
            'VOLATILE': 1.3,         # High survival focus in volatility
            'CRISIS': 2.0,           # Maximum survival focus during crisis
            'RECOVERY': 1.2,         # Moderate survival focus
            'LOW_VOLATILITY': 0.9,   # Standard approach
            'HIGH_VOLATILITY': 1.4   # High survival focus
        }
        return weights.get(market_regime, 1.0)
    
    def get_required_keys(self) -> list:
        """Return required data keys."""
        return ['max_drawdown', 'win_rate', 'max_consecutive_losses', 'returns', 'equity_curve']
    
    def _calculate_recovery_score(self, equity_curve: list, max_drawdown: float) -> float:
        """Calculate recovery score based on time to recover from drawdown."""
        if not equity_curve or max_drawdown >= 0:
            return 1.0
        
        # Find peak and trough
        peak = max(equity_curve)
        trough = min(equity_curve)
        
        if peak == 0:
            return 0.0
        
        # Calculate recovery time
        peak_idx = equity_curve.index(peak)
        trough_idx = equity_curve.index(trough)
        
        if peak_idx >= trough_idx:
            return 1.0
        
        # Find recovery point
        recovery_idx = None
        for i in range(trough_idx + 1, len(equity_curve)):
            if equity_curve[i] >= peak:
                recovery_idx = i
                break
        
        if recovery_idx is None:
            return 0.5  # Partial recovery
        
        # Recovery time in periods
        recovery_time = recovery_idx - peak_idx
        
        # Score based on recovery speed (faster = better)
        max_acceptable_recovery = 252  # 1 year
        recovery_score = max(0.0, 1.0 - recovery_time / max_acceptable_recovery)
        
        return recovery_score
    
    def _calculate_stability_score(self, returns: list) -> float:
        """Calculate stability score based on return volatility."""
        if len(returns) < 2:
            return 0.5
        
        # Calculate coefficient of variation
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        if mean_return == 0:
            return 0.5
        
        cv = abs(std_return / mean_return)
        
        # Stability score (lower CV = higher stability)
        max_acceptable_cv = 2.0
        stability_score = max(0.0, 1.0 - min(cv / max_acceptable_cv, 1.0))
        
        return stability_score
