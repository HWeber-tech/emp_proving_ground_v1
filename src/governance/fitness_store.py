"""
EMP Fitness Store v1.1

Manages fitness definitions and scoring for the governance layer
in the EMP Ultimate Architecture v1.1.
"""

import yaml
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path
from datetime import datetime

from ..core.events import PerformanceMetrics, RiskMetrics

logger = logging.getLogger(__name__)


class FitnessDefinition:
    """Fitness definition loaded from YAML configuration."""
    
    def __init__(self, config: Dict[str, Any]):
        self.version = config.get('version', '1.0.0')
        self.name = config.get('name', 'default')
        self.description = config.get('description', '')
        self.created_at = config.get('created_at', '')
        self.author = config.get('author', 'EMP System')
        
        # Performance weights
        self.performance_weights = config.get('performance_weights', {})
        
        # Risk weights
        self.risk_weights = config.get('risk_weights', {})
        
        # Thresholds
        self.minimum_thresholds = config.get('minimum_thresholds', {})
        
        # Penalty factors
        self.penalty_factors = config.get('penalty_factors', {})
        
        # Bonus factors
        self.bonus_factors = config.get('bonus_factors', {})
        
        # Time adjustments
        self.time_adjustments = config.get('time_adjustments', {})
        
        # Regime adjustments
        self.regime_adjustments = config.get('regime_adjustments', {})
        
        # Compliance requirements
        self.compliance = config.get('compliance', {})
        
        # Evaluation parameters
        self.evaluation = config.get('evaluation', {})
        
    def validate(self) -> bool:
        """Validate the fitness definition."""
        try:
            # Check required fields
            required_fields = ['performance_weights', 'risk_weights', 'minimum_thresholds']
            for field in required_fields:
                if not hasattr(self, field) or not getattr(self, field):
                    logger.error(f"Missing required field: {field}")
                    return False
                    
            # Validate weights sum to reasonable values
            perf_weight_sum = sum(self.performance_weights.values())
            risk_weight_sum = sum(self.risk_weights.values())
            
            if not (0.9 <= perf_weight_sum <= 1.1):
                logger.error(f"Performance weights sum to {perf_weight_sum}, should be ~1.0")
                return False
                
            if not (0.9 <= risk_weight_sum <= 1.1):
                logger.error(f"Risk weights sum to {risk_weight_sum}, should be ~1.0")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Error validating fitness definition: {e}")
            return False


class FitnessStore:
    """Store for managing fitness definitions."""
    
    def __init__(self, config_dir: str = "config/fitness"):
        self.config_dir = Path(config_dir)
        self.definitions: Dict[str, FitnessDefinition] = {}
        self.active_definition: Optional[str] = None
        
        logger.info(f"Fitness Store initialized with config directory: {config_dir}")
        
    def load_definitions(self) -> bool:
        """Load all fitness definitions from the config directory."""
        try:
            if not self.config_dir.exists():
                logger.error(f"Config directory does not exist: {self.config_dir}")
                return False
                
            # Load all YAML files
            yaml_files = list(self.config_dir.glob("*.yaml")) + list(self.config_dir.glob("*.yml"))
            
            if not yaml_files:
                logger.warning(f"No fitness definition files found in {self.config_dir}")
                return False
                
            for yaml_file in yaml_files:
                try:
                    with open(yaml_file, 'r') as f:
                        config = yaml.safe_load(f)
                        
                    if 'fitness_definition' in config:
                        definition = FitnessDefinition(config['fitness_definition'])
                        
                        if definition.validate():
                            self.definitions[definition.name] = definition
                            logger.info(f"Loaded fitness definition: {definition.name}")
                        else:
                            logger.error(f"Invalid fitness definition: {definition.name}")
                            
                except Exception as e:
                    logger.error(f"Error loading {yaml_file}: {e}")
                    
            # Set default active definition
            if self.definitions and not self.active_definition:
                self.active_definition = list(self.definitions.keys())[0]
                logger.info(f"Set active definition: {self.active_definition}")
                
            return len(self.definitions) > 0
            
        except Exception as e:
            logger.error(f"Error loading fitness definitions: {e}")
            return False
            
    def get_definition(self, name: str) -> Optional[FitnessDefinition]:
        """Get a fitness definition by name."""
        return self.definitions.get(name)
        
    def get_active_definition(self) -> Optional[FitnessDefinition]:
        """Get the currently active fitness definition."""
        if self.active_definition:
            return self.definitions.get(self.active_definition)
        return None
        
    def set_active_definition(self, name: str) -> bool:
        """Set the active fitness definition."""
        if name in self.definitions:
            self.active_definition = name
            logger.info(f"Set active fitness definition: {name}")
            return True
        else:
            logger.error(f"Fitness definition not found: {name}")
            return False
            
    def list_definitions(self) -> List[str]:
        """List all available fitness definitions."""
        return list(self.definitions.keys())
        
    def calculate_fitness(self, performance: PerformanceMetrics, 
                         risk: RiskMetrics, 
                         definition_name: Optional[str] = None) -> float:
        """Calculate fitness score using the specified definition."""
        try:
            # Get definition
            if definition_name:
                definition = self.get_definition(definition_name)
            else:
                definition = self.get_active_definition()
                
            if not definition:
                logger.error("No fitness definition available")
                return 0.0
                
            # Calculate performance score
            performance_score = self._calculate_performance_score(performance, definition)
            
            # Calculate risk score
            risk_score = self._calculate_risk_score(risk, definition)
            
            # Apply penalties
            penalty = self._calculate_penalties(performance, risk, definition)
            
            # Apply bonuses
            bonus = self._calculate_bonuses(performance, risk, definition)
            
            # Combine scores
            fitness_score = (performance_score * 0.7 + risk_score * 0.3) * penalty * bonus
            
            # Normalize to [0, 1]
            fitness_score = max(0.0, min(1.0, fitness_score))
            
            logger.debug(f"Fitness score: {fitness_score:.4f} (perf: {performance_score:.4f}, risk: {risk_score:.4f}, penalty: {penalty:.4f}, bonus: {bonus:.4f})")
            
            return fitness_score
            
        except Exception as e:
            logger.error(f"Error calculating fitness score: {e}")
            return 0.0
            
    def _calculate_performance_score(self, performance: PerformanceMetrics, 
                                   definition: FitnessDefinition) -> float:
        """Calculate performance score component."""
        score = 0.0
        
        # Total return
        if 'total_return' in definition.performance_weights:
            weight = definition.performance_weights['total_return']
            score += (performance.total_return * weight)
            
        # Annualized return
        if 'annualized_return' in definition.performance_weights:
            weight = definition.performance_weights['annualized_return']
            score += (performance.annualized_return * weight)
            
        # Sharpe ratio
        if 'sharpe_ratio' in definition.performance_weights:
            weight = definition.performance_weights['sharpe_ratio']
            score += (performance.sharpe_ratio * weight)
            
        # Sortino ratio
        if 'sortino_ratio' in definition.performance_weights:
            weight = definition.performance_weights['sortino_ratio']
            score += (performance.sortino_ratio * weight)
            
        # Win rate
        if 'win_rate' in definition.performance_weights:
            weight = definition.performance_weights['win_rate']
            score += (performance.win_rate * weight)
            
        # Profit factor
        if 'profit_factor' in definition.performance_weights:
            weight = definition.performance_weights['profit_factor']
            score += (performance.profit_factor * weight)
            
        return max(0.0, score)
        
    def _calculate_risk_score(self, risk: RiskMetrics, 
                            definition: FitnessDefinition) -> float:
        """Calculate risk score component."""
        score = 1.0  # Start with perfect score
        
        # Max drawdown penalty
        if 'max_drawdown' in definition.risk_weights:
            weight = definition.risk_weights['max_drawdown']
            penalty = max(0, risk.max_drawdown)
            score -= (penalty * weight)
            
        # Volatility penalty
        if 'volatility' in definition.risk_weights:
            weight = definition.risk_weights['volatility']
            penalty = max(0, risk.volatility)
            score -= (penalty * weight)
            
        # VaR penalty
        if 'var_95' in definition.risk_weights:
            weight = definition.risk_weights['var_95']
            penalty = max(0, abs(risk.var_95))
            score -= (penalty * weight)
            
        # CVaR penalty
        if 'cvar_95' in definition.risk_weights:
            weight = definition.risk_weights['cvar_95']
            penalty = max(0, abs(risk.cvar_95))
            score -= (penalty * weight)
            
        # Beta penalty
        if 'beta' in definition.risk_weights:
            weight = definition.risk_weights['beta']
            penalty = max(0, abs(risk.beta - 1.0))
            score -= (penalty * weight)
            
        return max(0.0, score)
        
    def _calculate_penalties(self, performance: PerformanceMetrics, 
                           risk: RiskMetrics, 
                           definition: FitnessDefinition) -> float:
        """Calculate penalty factors."""
        penalty = 1.0
        
        # Check minimum thresholds
        if 'total_return' in definition.minimum_thresholds:
            min_return = definition.minimum_thresholds['total_return']
            if performance.total_return < min_return:
                penalty *= definition.penalty_factors.get('below_minimum_return', 0.5)
                
        if 'sharpe_ratio' in definition.minimum_thresholds:
            min_sharpe = definition.minimum_thresholds['sharpe_ratio']
            if performance.sharpe_ratio < min_sharpe:
                penalty *= definition.penalty_factors.get('below_minimum_sharpe', 0.7)
                
        if 'max_drawdown' in definition.minimum_thresholds:
            max_dd = definition.minimum_thresholds['max_drawdown']
            if risk.max_drawdown > max_dd:
                penalty *= definition.penalty_factors.get('above_maximum_drawdown', 0.8)
                
        return penalty
        
    def _calculate_bonuses(self, performance: PerformanceMetrics, 
                          risk: RiskMetrics, 
                          definition: FitnessDefinition) -> float:
        """Calculate bonus factors."""
        bonus = 1.0
        
        # High return bonus
        if 'high_return_bonus' in definition.bonus_factors:
            if performance.total_return > 0.15:  # 15% return
                bonus *= definition.bonus_factors['high_return_bonus']
                
        # High Sharpe bonus
        if 'high_sharpe_bonus' in definition.bonus_factors:
            if performance.sharpe_ratio > 1.5:
                bonus *= definition.bonus_factors['high_sharpe_bonus']
                
        # Low drawdown bonus
        if 'low_drawdown_bonus' in definition.bonus_factors:
            if risk.max_drawdown < 0.10:  # 10% drawdown
                bonus *= definition.bonus_factors['low_drawdown_bonus']
                
        return bonus
        
    def get_summary(self) -> Dict[str, Any]:
        """Get fitness store summary."""
        return {
            'total_definitions': len(self.definitions),
            'active_definition': self.active_definition,
            'available_definitions': list(self.definitions.keys()),
            'config_directory': str(self.config_dir)
        } 