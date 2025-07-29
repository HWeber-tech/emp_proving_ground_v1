"""
Base fitness evaluation class for the EMP Proving Ground.
Provides abstract interface for fitness calculation with real implementations.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import logging
import numpy as np

logger = logging.getLogger(__name__)

class BaseFitness(ABC):
    """
    Abstract base class for fitness evaluation.
    All fitness dimensions must inherit from this class.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize fitness evaluator with configuration."""
        self.config = config
        self.name = self.__class__.__name__
        self.weight = self.config.get('weight', 1.0)
        self.min_samples = self.config.get('min_samples', 10)
        
    @abstractmethod
    def calculate_fitness(self, genome) -> float:
        """Calculate fitness score for this dimension."""
        pass
    
    @abstractmethod
    def get_optimal_weight(self, market_regime: str) -> float:
        """Get optimal weight for this fitness dimension based on market regime."""
        pass
    
    def normalize_score(self, raw_score: float, 
                       min_score: float = 0.0, 
                       max_score: float = 1.0) -> float:
        """Normalize score to [0, 1] range."""
        if max_score <= min_score:
            return 0.0
        return max(0.0, min(1.0, (raw_score - min_score) / (max_score - min_score)))
    
    def validate_data(self, data: Dict[str, Any]) -> bool:
        """Validate that required data is available and sufficient."""
        required_keys = self.get_required_data_keys()
        
        for key in required_keys:
            if key not in data:
                logger.warning(f"Missing required data key: {key}")
                return False
                
            if isinstance(data[key], (list, np.ndarray)) and len(data[key]) < self.min_samples:
                logger.warning(f"Insufficient data for key {key}: {len(data[key])} < {self.min_samples}")
                return False
        
        return True
    
    @abstractmethod
    def get_required_data_keys(self) -> List[str]:
        """Return list of required data keys for this fitness dimension."""
        pass
    
    def get_market_regime_weight(self, market_regime: str) -> float:
        """Get weight adjustment based on market regime."""
        regime_weights = {
            'bull': 1.2,
            'bear': 0.8,
            'sideways': 1.0,
            'volatile': 0.9,
            'calm': 1.1,
            'neutral': 1.0
        }
        return regime_weights.get(market_regime.lower(), 1.0)
    
    def _calculate_base_score(self, genome, data: Dict[str, Any]) -> float:
        """Calculate base fitness score - to be implemented by subclasses."""
        # Default implementation - should be overridden
        return 0.5
    
    def _normalize_score(self, score: float) -> float:
        """Normalize score to [0, 1] range with sigmoid function."""
        return 1.0 / (1.0 + np.exp(-score))
    
    def get_required_data(self) -> Dict[str, Any]:
        """Get required data for fitness calculation."""
        # This would typically fetch from a data manager
        # For now, return empty dict - subclasses should override
        return {}
    
    def update_fitness_history(self, genome, fitness: float) -> None:
        """Update fitness history for the genome."""
        if not hasattr(genome, 'fitness_history'):
            genome.fitness_history = {}
        
        if self.name not in genome.fitness_history:
            genome.fitness_history[self.name] = []
        
        genome.fitness_history[self.name].append({
            'fitness': fitness,
            'timestamp': np.datetime64('now'),
            'generation': getattr(genome, 'generation', 0)
        })
        
        # Keep only last 100 entries
        if len(genome.fitness_history[self.name]) > 100:
            genome.fitness_history[self.name] = genome.fitness_history[self.name][-100:]
    
    def get_fitness_trend(self, genome) -> float:
        """Get fitness trend for this dimension."""
        if not hasattr(genome, 'fitness_history') or self.name not in genome.fitness_history:
            return 0.0
        
        history = genome.fitness_history[self.name]
        if len(history) < 2:
            return 0.0
        
        # Calculate trend over last 10 generations
        recent_history = history[-10:]
        if len(recent_history) < 2:
            return 0.0
        
        # Simple linear trend
        x = np.arange(len(recent_history))
        y = [entry['fitness'] for entry in recent_history]
        
        try:
            slope = np.polyfit(x, y, 1)[0]
            return float(slope)
        except:
            return 0.0


class ProfitFitness(BaseFitness):
    """Fitness evaluator based on profit/return performance."""
    
    def calculate_fitness(self, genome) -> float:
        """Calculate fitness based on profit metrics."""
        try:
            data = self.get_required_data()
            if not self.validate_data(data):
                return 0.0
            
            # Calculate profit-based fitness
            total_return = data.get('total_return', 0.0)
            win_rate = data.get('win_rate', 0.0)
            profit_factor = data.get('profit_factor', 1.0)
            
            # Combine metrics
            profit_score = total_return * 0.5
            consistency_score = win_rate * 0.3
            efficiency_score = (profit_factor - 1.0) * 0.2
            
            base_score = profit_score + consistency_score + efficiency_score
            return self._normalize_score(base_score)
            
        except Exception as e:
            logger.error(f"Profit fitness calculation failed: {e}")
            return 0.0
    
    def get_optimal_weight(self, market_regime: str) -> float:
        """Get optimal weight for profit fitness."""
        weights = {
            'bull': 1.3,
            'bear': 0.7,
            'sideways': 1.0,
            'volatile': 0.9,
            'calm': 1.2
        }
        return weights.get(market_regime.lower(), 1.0)
    
    def get_required_data_keys(self) -> List[str]:
        """Required data keys for profit fitness."""
        return ['total_return', 'win_rate', 'profit_factor', 'trade_count']


class RiskFitness(BaseFitness):
    """Fitness evaluator based on risk management performance."""
    
    def calculate_fitness(self, genome) -> float:
        """Calculate fitness based on risk metrics."""
        try:
            data = self.get_required_data()
            if not self.validate_data(data):
                return 0.0
            
            # Calculate risk-based fitness
            max_drawdown = data.get('max_drawdown', 1.0)
            sharpe_ratio = data.get('sharpe_ratio', 0.0)
            sortino_ratio = data.get('sortino_ratio', 0.0)
            var_95 = data.get('var_95', 0.0)
            
            # Lower drawdown and higher ratios are better
            drawdown_score = (1.0 - max_drawdown) * 0.4
            sharpe_score = min(3.0, max(0.0, sharpe_ratio)) / 3.0 * 0.3
            sortino_score = min(3.0, max(0.0, sortino_ratio)) / 3.0 * 0.2
            var_score = (1.0 - min(1.0, abs(var_95))) * 0.1
            
            base_score = drawdown_score + sharpe_score + sortino_score + var_score
            return self._normalize_score(base_score)
            
        except Exception as e:
            logger.error(f"Risk fitness calculation failed: {e}")
            return 0.0
    
    def get_optimal_weight(self, market_regime: str) -> float:
        """Get optimal weight for risk fitness."""
        weights = {
            'bull': 0.8,
            'bear': 1.4,
            'sideways': 1.1,
            'volatile': 1.5,
            'calm': 0.9
        }
        return weights.get(market_regime.lower(), 1.0)
    
    def get_required_data_keys(self) -> List[str]:
        """Required data keys for risk fitness."""
        return ['max_drawdown', 'sharpe_ratio', 'sortino_ratio', 'var_95', 'volatility']

