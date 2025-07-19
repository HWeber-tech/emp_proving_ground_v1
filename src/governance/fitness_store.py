"""
EMP Fitness Store v1.1

Manages fitness definitions for strategy evolution. Provides
centralized access to fitness configurations and validation.
"""

import yaml
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
from dataclasses import dataclass
import numpy as np

from src.core.exceptions import GovernanceException

logger = logging.getLogger(__name__)


@dataclass
class FitnessDefinition:
    """Fitness definition structure."""
    name: str
    version: str
    description: str
    components: Dict[str, Any]
    thresholds: Dict[str, Any]
    penalties: Dict[str, Any]
    bonuses: Dict[str, Any]


class FitnessStore:
    """Manages fitness definitions for strategy evolution."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = Path(config_path) if config_path else Path("config/fitness")
        self.definitions: Dict[str, FitnessDefinition] = {}
        self.active_definition: Optional[str] = None
        self._load_definitions()
        
    def _load_definitions(self):
        """Load fitness definitions from configuration files."""
        try:
            if not self.config_path.exists():
                logger.warning(f"Fitness config path does not exist: {self.config_path}")
                return
                
            for config_file in self.config_path.glob("*.yaml"):
                try:
                    definition = self._load_definition_from_file(config_file)
                    self.definitions[definition.name] = definition
                    logger.info(f"Loaded fitness definition: {definition.name}")
                except Exception as e:
                    logger.error(f"Error loading fitness definition from {config_file}: {e}")
                    
            # Set default active definition
            if self.definitions and not self.active_definition:
                self.active_definition = list(self.definitions.keys())[0]
                logger.info(f"Set active fitness definition: {self.active_definition}")
                
        except Exception as e:
            raise GovernanceException(f"Error loading fitness definitions: {e}")
            
    def _load_definition_from_file(self, config_file: Path) -> FitnessDefinition:
        """Load a fitness definition from a YAML file."""
        with open(config_file, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
            
        fitness_data = data.get('fitness_definition', {})
        
        return FitnessDefinition(
            name=fitness_data.get('name', config_file.stem),
            version=fitness_data.get('version', '1.0.0'),
            description=fitness_data.get('description', ''),
            components=fitness_data.get('components', {}),
            thresholds=fitness_data.get('thresholds', {}),
            penalties=fitness_data.get('penalties', {}),
            bonuses=fitness_data.get('bonuses', {})
        )
        
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
        
    def validate_definition(self, name: str) -> bool:
        """Validate a fitness definition."""
        definition = self.get_definition(name)
        if not definition:
            return False
            
        # Check required fields
        required_fields = ['components', 'thresholds']
        for field in required_fields:
            if not hasattr(definition, field) or not getattr(definition, field):
                logger.error(f"Missing required field in fitness definition: {field}")
                return False
                
        # Validate components
        for component_name, component in definition.components.items():
            if 'weight' not in component:
                logger.error(f"Missing weight in component: {component_name}")
                return False
                
        logger.info(f"Fitness definition validated: {name}")
        return True
        
    def calculate_fitness(self, performance_metrics: Dict[str, Any], 
                         definition_name: Optional[str] = None) -> float:
        """Calculate fitness score using the specified definition."""
        definition = self.get_definition(definition_name) if definition_name else self.get_active_definition()
        
        if not definition:
            raise GovernanceException("No fitness definition available")
            
        try:
            total_score = 0.0
            total_weight = 0.0
            
            # Calculate component scores
            for component_name, component in definition.components.items():
                weight = component.get('weight', 0.0)
                metrics = component.get('metrics', [])
                
                component_score = self._calculate_component_score(
                    component_name, metrics, performance_metrics, definition
                )
                
                total_score += component_score * weight
                total_weight += weight
                
            # Normalize by total weight
            if total_weight > 0:
                fitness_score = total_score / total_weight
            else:
                fitness_score = 0.0
                
            # Apply penalties and bonuses
            fitness_score = self._apply_penalties_and_bonuses(
                fitness_score, performance_metrics, definition
            )
            
            return max(0.0, min(1.0, fitness_score))  # Clamp to [0, 1]
            
        except Exception as e:
            raise GovernanceException(f"Error calculating fitness: {e}")
            
    def _calculate_component_score(self, component_name: str, metrics: List[str], 
                                  performance_metrics: Dict[str, Any], 
                                  definition: FitnessDefinition) -> float:
        """Calculate score for a specific component."""
        if not metrics:
            return 0.0
            
        component_scores = []
        
        for metric in metrics:
            if metric in performance_metrics:
                value = performance_metrics[metric]
                
                # Normalize metric value based on thresholds
                normalized_value = self._normalize_metric(
                    metric, value, definition.thresholds
                )
                
                component_scores.append(normalized_value)
                
        return np.mean(component_scores) if component_scores else 0.0
        
    def _normalize_metric(self, metric: str, value: float, 
                         thresholds: Dict[str, Any]) -> float:
        """Normalize a metric value based on thresholds."""
        # Default normalization (0-1 scale)
        if metric == 'sharpe_ratio':
            return max(0.0, min(1.0, value / 2.0))  # 2.0 Sharpe = perfect
        elif metric == 'sortino_ratio':
            return max(0.0, min(1.0, value / 2.0))  # 2.0 Sortino = perfect
        elif metric == 'calmar_ratio':
            return max(0.0, min(1.0, value / 1.0))  # 1.0 Calmar = perfect
        elif metric == 'win_rate':
            return value  # Already 0-1
        elif metric == 'profit_factor':
            return max(0.0, min(1.0, (value - 1.0) / 2.0))  # 3.0 PF = perfect
        elif metric == 'max_drawdown':
            return max(0.0, 1.0 - abs(value))  # 0% drawdown = perfect
        elif metric == 'volatility':
            return max(0.0, 1.0 - value)  # 0% volatility = perfect
        else:
            # Generic normalization
            return float(max(0.0, min(1.0, value)))
            
    def _apply_penalties_and_bonuses(self, base_score: float, 
                                   performance_metrics: Dict[str, Any],
                                   definition: FitnessDefinition) -> float:
        """Apply penalties and bonuses to the base fitness score."""
        final_score = base_score
        
        # Apply penalties
        for penalty_name, penalty_value in definition.penalties.items():
            if penalty_name in performance_metrics:
                metric_value = performance_metrics[penalty_name]
                threshold = definition.thresholds.get(penalty_name)
                
                if threshold and self._check_penalty_condition(penalty_name, metric_value, threshold):
                    final_score *= (1.0 - penalty_value)
                    
        # Apply bonuses
        for bonus_name, bonus_value in definition.bonuses.items():
            if bonus_name in performance_metrics:
                metric_value = performance_metrics[bonus_name]
                threshold = definition.thresholds.get(bonus_name)
                
                if threshold and self._check_bonus_condition(bonus_name, metric_value, threshold):
                    final_score *= (1.0 + bonus_value)
                    
        return final_score
        
    def _check_penalty_condition(self, penalty_name: str, metric_value: float, 
                               threshold: float) -> bool:
        """Check if a penalty condition is met."""
        if penalty_name in ['max_drawdown', 'volatility']:
            return abs(metric_value) > threshold
        elif penalty_name in ['sharpe_ratio', 'sortino_ratio', 'calmar_ratio', 'win_rate']:
            return metric_value < threshold
        else:
            return False
            
    def _check_bonus_condition(self, bonus_name: str, metric_value: float, 
                             threshold: float) -> bool:
        """Check if a bonus condition is met."""
        if bonus_name in ['sharpe_ratio', 'sortino_ratio', 'calmar_ratio', 'win_rate']:
            return metric_value > threshold
        elif bonus_name in ['max_drawdown', 'volatility']:
            return abs(metric_value) < threshold
        else:
            return False 