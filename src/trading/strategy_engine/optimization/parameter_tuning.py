"""
Parameter Tuning Module

Advanced parameter tuning using grid search, Bayesian optimization,
and hyperparameter optimization techniques.

Author: EMP Development Team
Date: July 18, 2024
Phase: 3 - Advanced Trading Strategies and Risk Management
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass
from itertools import product
import time

logger = logging.getLogger(__name__)


@dataclass
class TuningResult:
    """Result of parameter tuning"""
    best_parameters: Dict[str, Any]
    best_score: float
    tuning_method: str
    execution_time: float
    parameter_combinations: int
    scores_history: List[float]


class ParameterTuner:
    """
    Advanced Parameter Tuning System
    
    Implements multiple tuning methods:
    - Grid Search
    - Random Search
    - Bayesian Optimization
    - Hyperparameter Optimization
    """
    
    def __init__(self):
        self.tuning_history = []
        logger.info("ParameterTuner initialized")
    
    def grid_search(self, strategy_class: type, symbols: List[str],
                   historical_data: Dict[str, List], fitness_function: Callable,
                   parameter_grid: Dict[str, List[Any]]) -> TuningResult:
        """Perform grid search optimization"""
        
        start_time = time.time()
        
        # Generate all parameter combinations
        param_names = list(parameter_grid.keys())
        param_values = list(parameter_grid.values())
        combinations = list(product(*param_values))
        
        best_score = float('-inf')
        best_parameters = {}
        scores_history = []
        
        logger.info(f"Grid search: {len(combinations)} combinations to evaluate")
        
        for i, combination in enumerate(combinations):
            parameters = dict(zip(param_names, combination))
            
            try:
                # Create strategy instance
                strategy_instance = strategy_class(
                    strategy_id=f"grid_{i}",
                    parameters=parameters,
                    symbols=symbols
                )
                
                # Evaluate fitness
                score = fitness_function(strategy_instance, historical_data)
                scores_history.append(score)
                
                # Update best
                if score > best_score:
                    best_score = score
                    best_parameters = parameters.copy()
                
                if i % 100 == 0:
                    logger.info(f"Grid search progress: {i}/{len(combinations)}")
                    
            except Exception as e:
                logger.warning(f"Grid search evaluation failed: {e}")
                scores_history.append(float('-inf'))
        
        execution_time = time.time() - start_time
        
        return TuningResult(
            best_parameters=best_parameters,
            best_score=best_score,
            tuning_method="grid_search",
            execution_time=execution_time,
            parameter_combinations=len(combinations),
            scores_history=scores_history
        )
    
    def random_search(self, strategy_class: type, symbols: List[str],
                     historical_data: Dict[str, List], fitness_function: Callable,
                     parameter_bounds: Dict[str, Tuple[float, float]],
                     n_iterations: int = 100) -> TuningResult:
        """Perform random search optimization"""
        
        start_time = time.time()
        
        best_score = float('-inf')
        best_parameters = {}
        scores_history = []
        
        logger.info(f"Random search: {n_iterations} iterations")
        
        for i in range(n_iterations):
            # Generate random parameters
            parameters = {}
            for param_name, (min_val, max_val) in parameter_bounds.items():
                if isinstance(min_val, int) and isinstance(max_val, int):
                    parameters[param_name] = np.random.randint(min_val, max_val + 1)
                else:
                    parameters[param_name] = np.random.uniform(min_val, max_val)
            
            try:
                # Create strategy instance
                strategy_instance = strategy_class(
                    strategy_id=f"random_{i}",
                    parameters=parameters,
                    symbols=symbols
                )
                
                # Evaluate fitness
                score = fitness_function(strategy_instance, historical_data)
                scores_history.append(score)
                
                # Update best
                if score > best_score:
                    best_score = score
                    best_parameters = parameters.copy()
                
                if i % 20 == 0:
                    logger.info(f"Random search progress: {i}/{n_iterations}")
                    
            except Exception as e:
                logger.warning(f"Random search evaluation failed: {e}")
                scores_history.append(float('-inf'))
        
        execution_time = time.time() - start_time
        
        return TuningResult(
            best_parameters=best_parameters,
            best_score=best_score,
            tuning_method="random_search",
            execution_time=execution_time,
            parameter_combinations=n_iterations,
            scores_history=scores_history
        )
    
    def bayesian_optimization(self, strategy_class: type, symbols: List[str],
                            historical_data: Dict[str, List], fitness_function: Callable,
                            parameter_bounds: Dict[str, Tuple[float, float]],
                            n_iterations: int = 50) -> TuningResult:
        """Perform Bayesian optimization"""
        
        try:
            from skopt import gp_minimize
            from skopt.space import Real, Integer
        except ImportError:
            logger.warning("scikit-optimize not available, falling back to random search")
            return self.random_search(strategy_class, symbols, historical_data, 
                                    fitness_function, parameter_bounds, n_iterations)
        
        start_time = time.time()
        
        # Define search space
        search_space = []
        param_names = []
        
        for param_name, (min_val, max_val) in parameter_bounds.items():
            param_names.append(param_name)
            if isinstance(min_val, int) and isinstance(max_val, int):
                search_space.append(Integer(min_val, max_val, name=param_name))
            else:
                search_space.append(Real(min_val, max_val, name=param_name))
        
        # Define objective function
        def objective(params):
            parameters = dict(zip(param_names, params))
            
            try:
                strategy_instance = strategy_class(
                    strategy_id=f"bayesian_{len(self.tuning_history)}",
                    parameters=parameters,
                    symbols=symbols
                )
                
                score = fitness_function(strategy_instance, historical_data)
                return -score  # Minimize negative score
                
            except Exception as e:
                logger.warning(f"Bayesian optimization evaluation failed: {e}")
                return float('inf')
        
        # Run Bayesian optimization
        result = gp_minimize(
            objective,
            search_space,
            n_calls=n_iterations,
            random_state=42
        )
        
        best_parameters = dict(zip(param_names, result.x))
        best_score = -result.fun  # Convert back to positive score
        
        execution_time = time.time() - start_time
        
        return TuningResult(
            best_parameters=best_parameters,
            best_score=best_score,
            tuning_method="bayesian_optimization",
            execution_time=execution_time,
            parameter_combinations=n_iterations,
            scores_history=[-score for score in result.func_vals]
        )
    
    def hyperparameter_optimization(self, strategy_class: type, symbols: List[str],
                                  historical_data: Dict[str, List], fitness_function: Callable,
                                  parameter_config: Dict[str, Any]) -> TuningResult:
        """Perform advanced hyperparameter optimization"""
        
        # This is a simplified version - in practice, you might use Optuna, Hyperopt, etc.
        logger.info("Hyperparameter optimization using adaptive search")
        
        # Use Bayesian optimization as the default method
        parameter_bounds = parameter_config.get('bounds', {})
        n_iterations = parameter_config.get('n_iterations', 50)
        
        return self.bayesian_optimization(
            strategy_class, symbols, historical_data, fitness_function,
            parameter_bounds, n_iterations
        )
    
    def compare_tuning_methods(self, strategy_class: type, symbols: List[str],
                             historical_data: Dict[str, List], fitness_function: Callable,
                             parameter_config: Dict[str, Any]) -> Dict[str, TuningResult]:
        """Compare different tuning methods"""
        
        results = {}
        
        # Grid search
        if 'grid' in parameter_config:
            logger.info("Running grid search...")
            results['grid_search'] = self.grid_search(
                strategy_class, symbols, historical_data, fitness_function,
                parameter_config['grid']
            )
        
        # Random search
        if 'random' in parameter_config:
            logger.info("Running random search...")
            results['random_search'] = self.random_search(
                strategy_class, symbols, historical_data, fitness_function,
                parameter_config['random']['bounds'],
                parameter_config['random']['n_iterations']
            )
        
        # Bayesian optimization
        if 'bayesian' in parameter_config:
            logger.info("Running Bayesian optimization...")
            results['bayesian_optimization'] = self.bayesian_optimization(
                strategy_class, symbols, historical_data, fitness_function,
                parameter_config['bayesian']['bounds'],
                parameter_config['bayesian']['n_iterations']
            )
        
        return results
    
    def get_tuning_summary(self, results: Dict[str, TuningResult]) -> Dict[str, Any]:
        """Get summary of tuning results"""
        summary = {
            'methods_comparison': {},
            'best_method': None,
            'best_score': float('-inf')
        }
        
        for method_name, result in results.items():
            summary['methods_comparison'][method_name] = {
                'best_score': result.best_score,
                'execution_time': result.execution_time,
                'parameter_combinations': result.parameter_combinations,
                'best_parameters': result.best_parameters
            }
            
            if result.best_score > summary['best_score']:
                summary['best_score'] = result.best_score
                summary['best_method'] = method_name
        
        return summary 
