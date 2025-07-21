"""
Robustness fitness dimension for strategy evaluation.

Measures stability and consistency of performance across different conditions.
"""

import numpy as np
from typing import Dict, Any
from .base_fitness import BaseFitness, FitnessResult


class RobustnessFitness(BaseFitness):
    """Robustness optimization through stability and consistency."""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.target_volatility = self.config.get('target_volatility', 0.15)  # 15% annual
        self.max_tracking_error = self.config.get('max_tracking_error', 0.05)  # 5%
        
    def calculate_fitness(self, 
                         strategy_performance: Dict[str, Any],
                         market_data: Dict[str, Any],
                         regime_data: Dict[str, Any]) -> FitnessResult:
        """Calculate robustness fitness score."""
        try:
            if not self.validate_inputs(strategy_performance):
                return FitnessResult(
                    score=0.0,
                    raw_value=0.0,
                    confidence=0.0,
                    metadata={'error': 'Invalid inputs'},
                    weight=self.weight
                )
            
            # Extract robustness metrics
            returns = strategy_performance.get('returns', [])
            volatility = strategy_performance.get('volatility', 0.0)
            beta = strategy_performance.get('beta', 0.0)
            tracking_error = strategy_performance.get('tracking_error', 0.0)
            max_consecutive_wins = strategy_performance.get('max_consecutive_wins', 0)
            max_consecutive_losses = strategy_performance.get('max_consecutive_losses', 0)
            
            if not returns:
                return FitnessResult(
                    score=0.0,
                    raw_value=0.0,
                    confidence=0.0,
                    metadata={'error': 'No returns data'},
                    weight=self.weight
                )
            
            # Volatility score (target volatility)
            volatility_score = max(0.0, 1.0 - abs(volatility - self.target_volatility) / self.target_volatility)
            
            # Beta stability score (closer to 1.0 = more stable)
            beta_score = max(0.0, 1.0 - abs(beta - 1.0) / 2.0)
            
            # Tracking error score
            tracking_score = max(0.0, 1.0 - tracking_error / self.max_tracking_error)
            
            # Win/loss streak balance
            streak_balance = self._calculate_streak_balance(max_consecutive_wins, max_consecutive_losses)
            
            # Return consistency
            consistency_score = self._calculate_consistency_score(returns)
            
            # Composite robustness score
            composite_score = (
                volatility_score * 0.3 +
                beta_score * 0.2 +
                tracking_score * 0.2 +
                streak_balance * 0.15 +
                consistency_score * 0.15
            )
            
            final_score = max(0.0, min(1.0, composite_score))
            
            # Calculate confidence
            confidence = self.calculate_confidence(len(returns))
            
            metadata = {
                'volatility_score': volatility_score,
                'beta_score': beta_score,
                'tracking_score': tracking_score,
                'streak_balance': streak_balance,
                'consistency_score': consistency_score,
                'volatility': volatility,
                'beta': beta,
                'tracking_error': tracking_error
            }
            
            return FitnessResult(
                score=final_score,
                raw_value=1.0 / (1.0 + volatility),  # Lower volatility = higher score
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
            'TRENDING_UP': 0.8,      # Less critical in stable trends
            'TRENDING_DOWN': 1.2,    # Important for stability in downtrends
            'RANGING': 1.4,          # Critical for ranging markets
            'VOLATILE': 1.5,         # Maximum importance in volatility
            'CRISIS': 1.3,           # High importance during crisis
            'RECOVERY': 1.1,         # Moderate importance during recovery
            'LOW_VOLATILITY': 1.0,   # Standard importance
            'HIGH_VOLATILITY': 1.5   # Maximum importance in high vol
        }
        return weights.get(market_regime, 1.0)
    
    def get_required_keys(self) -> list:
        """Return required data keys."""
        return ['returns', 'volatility', 'beta', 'tracking_error', 'max_consecutive_wins', 'max_consecutive_losses']
    
    def _calculate_streak_balance(self, wins: int, losses: int) -> float:
        """Calculate balance between win and loss streaks."""
        if wins + losses == 0:
            return 0.5
        
        # Balance score (equal wins and losses = perfect balance)
        total_streaks = wins + losses
        balance = 1.0 - abs(wins - losses) / total_streaks
        
        return balance
    
    def _calculate_consistency_score(self, returns: list) -> float:
        """Calculate return consistency score."""
        if len(returns) < 2:
            return 0.5
        
        # Calculate rolling consistency
        window_size = min(20, len(returns))
        rolling_means = []
        
        for i in range(len(returns) - window_size + 1):
            window = returns[i:i + window_size]
            rolling_means.append(np.mean(window))
        
        if not rolling_means:
            return 0.5
        
        # Consistency based on rolling mean stability
        mean_consistency = np.mean(rolling_means)
        std_consistency = np.std(rolling_means)
        
        if mean_consistency == 0:
            return 0.5
        
        cv = abs(std_consistency / mean_consistency)
        consistency_score = max(0.0, 1.0 - min(cv, 1.0))
        
        return consistency_score
