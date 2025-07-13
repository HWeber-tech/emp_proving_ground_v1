#!/usr/bin/env python3
"""
Configuration loader for EMP Proving Ground v2.0 Pilot Experiment
"""

import yaml
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
from dataclasses import dataclass


@dataclass
class RegimeConfig:
    """Configuration for a market regime"""
    name: str
    start_date: str
    end_date: str
    symbol: str
    description: str
    characteristics: List[str]


@dataclass
class EvolutionConfig:
    """Configuration for evolution parameters"""
    elite_ratio: float
    crossover_ratio: float
    mutation_ratio: float
    mutation_rate_per_node: float
    tournament_size: int
    max_tree_depth: int
    complexity_penalty_lambda: float
    diversity_injection_threshold: float
    diversity_injection_ratio: float


@dataclass
class FitnessConfig:
    """Configuration for fitness evaluation"""
    weights: Dict[str, float]
    consistency_penalty_multiplier: float


@dataclass
class ExecutionConfig:
    """Configuration for execution parameters"""
    checkpoint_interval: int
    max_simulation_steps: int
    initial_balance: float
    base_position_size: float


@dataclass
class LoggingConfig:
    """Configuration for logging"""
    level: str
    log_to_file: bool
    log_to_console: bool
    metrics_csv_interval: int


@dataclass
class SuccessCriteria:
    """Success criteria thresholds"""
    min_fitness_improvement: float
    trap_rate_min: float
    trap_rate_max: float
    max_crash_count: int


@dataclass
class PilotConfig:
    """Complete pilot configuration"""
    experiment_name: str
    population_size: int
    generations: int
    random_seed: int
    adversarial_sweep_levels: List[float]
    regime_datasets: Dict[str, RegimeConfig]
    evolution: EvolutionConfig
    fitness: FitnessConfig
    execution: ExecutionConfig
    logging: LoggingConfig
    success_criteria: SuccessCriteria


def load_config(config_path: str = "pilot_config.yaml") -> PilotConfig:
    """
    Load configuration from YAML file
    
    Args:
        config_path: Path to configuration file
    
    Returns:
        PilotConfig object with all parameters
    """
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config_data = yaml.safe_load(f)
    
    # Parse regime datasets
    regime_datasets = {}
    for regime_name, regime_data in config_data.get("regime_datasets", {}).items():
        regime_datasets[regime_name] = RegimeConfig(
            name=regime_data["name"],
            start_date=regime_data["start_date"],
            end_date=regime_data["end_date"],
            symbol=regime_data["symbol"],
            description=regime_data["description"],
            characteristics=regime_data["characteristics"]
        )
    
    # Parse evolution config
    evolution_data = config_data.get("evolution", {})
    evolution_config = EvolutionConfig(
        elite_ratio=evolution_data["elite_ratio"],
        crossover_ratio=evolution_data["crossover_ratio"],
        mutation_ratio=evolution_data["mutation_ratio"],
        mutation_rate_per_node=evolution_data["mutation_rate_per_node"],
        tournament_size=evolution_data["tournament_size"],
        max_tree_depth=evolution_data["max_tree_depth"],
        complexity_penalty_lambda=evolution_data["complexity_penalty_lambda"],
        diversity_injection_threshold=evolution_data["diversity_injection_threshold"],
        diversity_injection_ratio=evolution_data["diversity_injection_ratio"]
    )
    
    # Parse fitness config
    fitness_data = config_data.get("fitness", {})
    fitness_config = FitnessConfig(
        weights=fitness_data["weights"],
        consistency_penalty_multiplier=fitness_data["consistency_penalty_multiplier"]
    )
    
    # Parse execution config
    execution_data = config_data.get("execution", {})
    execution_config = ExecutionConfig(
        checkpoint_interval=execution_data["checkpoint_interval"],
        max_simulation_steps=execution_data["max_simulation_steps"],
        initial_balance=execution_data["initial_balance"],
        base_position_size=execution_data["base_position_size"]
    )
    
    # Parse logging config
    logging_data = config_data.get("logging", {})
    logging_config = LoggingConfig(
        level=logging_data["level"],
        log_to_file=logging_data["log_to_file"],
        log_to_console=logging_data["log_to_console"],
        metrics_csv_interval=logging_data["metrics_csv_interval"]
    )
    
    # Parse success criteria
    success_data = config_data.get("success_criteria", {})
    success_criteria = SuccessCriteria(
        min_fitness_improvement=success_data["min_fitness_improvement"],
        trap_rate_min=success_data["trap_rate_min"],
        trap_rate_max=success_data["trap_rate_max"],
        max_crash_count=success_data["max_crash_count"]
    )
    
    # Create complete config
    config = PilotConfig(
        experiment_name=config_data["experiment_name"],
        population_size=config_data["population_size"],
        generations=config_data["generations"],
        random_seed=config_data["random_seed"],
        adversarial_sweep_levels=config_data["adversarial_sweep_levels"],
        regime_datasets=regime_datasets,
        evolution=evolution_config,
        fitness=fitness_config,
        execution=execution_config,
        logging=logging_config,
        success_criteria=success_criteria
    )
    
    return config


def validate_config(config: PilotConfig) -> bool:
    """
    Validate configuration parameters
    
    Args:
        config: Configuration to validate
    
    Returns:
        True if valid, raises ValueError if invalid
    """
    
    # Validate ratios sum to 1.0
    total_ratio = (config.evolution.elite_ratio + 
                   config.evolution.crossover_ratio + 
                   config.evolution.mutation_ratio)
    
    if abs(total_ratio - 1.0) > 0.01:
        raise ValueError(f"Evolution ratios must sum to 1.0, got {total_ratio}")
    
    # Validate fitness weights sum to 1.0
    total_weights = sum(config.fitness.weights.values())
    if abs(total_weights - 1.0) > 0.01:
        raise ValueError(f"Fitness weights must sum to 1.0, got {total_weights}")
    
    # Validate adversarial levels
    for level in config.adversarial_sweep_levels:
        if not 0.0 <= level <= 1.0:
            raise ValueError(f"Adversarial level must be between 0.0 and 1.0, got {level}")
    
    # Validate population size
    if config.population_size < 10:
        raise ValueError(f"Population size must be at least 10, got {config.population_size}")
    
    # Validate generations
    if config.generations < 1:
        raise ValueError(f"Generations must be at least 1, got {config.generations}")
    
    return True


def create_experiment_directories(config: PilotConfig) -> Dict[str, Path]:
    """
    Create experiment directories
    
    Args:
        config: Configuration object
    
    Returns:
        Dictionary of directory paths
    """
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = Path(f"experiments/{config.experiment_name}_{timestamp}")
    
    directories = {
        "experiment_root": experiment_dir,
        "checkpoints": experiment_dir / "checkpoints",
        "logs": experiment_dir / "logs",
        "metrics": experiment_dir / "metrics",
        "results": experiment_dir / "results"
    }
    
    # Create directories
    for dir_path in directories.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    return directories 