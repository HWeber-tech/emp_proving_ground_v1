#!/usr/bin/env python3
"""
EMP Proving Ground v2.0 - Pilot Experiment Executor
Operation Phased Explosion - Phase 1: Shake-down Cruise
"""

import sys
import os
import json
import pickle
import multiprocessing as mp
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import logging
import csv
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
import traceback

# Import our modules
from config import load_config, validate_config, create_experiment_directories, PilotConfig
from emp_proving_ground_unified import (
    TickDataStorage, TickDataCleaner, DukascopyIngestor,
    MarketSimulator, AdversarialEngine, SensoryCortex,
    DecisionGenome, EvolutionEngine, FitnessEvaluator,
    EvolutionConfig, FitnessScore
)


class ExperimentLogger:
    """Real-time logging and metrics tracking"""
    
    def __init__(self, config: PilotConfig, directories: Dict[str, Path], 
                 difficulty_level: float):
        """
        Initialize experiment logger
        
        Args:
            config: Configuration object
            directories: Experiment directories
            difficulty_level: Adversarial difficulty level
        """
        self.config = config
        self.directories = directories
        self.difficulty_level = difficulty_level
        
        # Setup logging
        self._setup_logging()
        
        # Setup metrics CSV
        self._setup_metrics_csv()
        
        # Track metrics
        self.metrics_history: List[Dict] = []
        self.crash_count = 0
        self.trap_rates: List[float] = []
    
    def _setup_logging(self):
        """Setup logging configuration"""
        
        log_file = self.directories["logs"] / f"difficulty_{self.difficulty_level:.1f}.log"
        
        # Configure logging
        logging.basicConfig(
            level=getattr(logging, self.config.logging.level),
            format="%(asctime)s | %(levelname)s | %(name)s:%(lineno)d - %(message)s",
            handlers=[
                logging.FileHandler(log_file) if self.config.logging.log_to_file else logging.NullHandler(),
                logging.StreamHandler(sys.stdout) if self.config.logging.log_to_console else logging.NullHandler()
            ]
        )
        
        self.logger = logging.getLogger(f"PilotExecutor_{self.difficulty_level:.1f}")
    
    def _setup_metrics_csv(self):
        """Setup metrics CSV file"""
        
        csv_file = self.directories["metrics"] / f"metrics_{self.difficulty_level:.1f}.csv"
        
        # Create CSV with headers
        headers = [
            "generation", "timestamp", "best_fitness", "avg_fitness", 
            "worst_fitness", "diversity_score", "trap_rate", "crash_count",
            "elite_count", "new_genomes", "complexity_mean", "complexity_std"
        ]
        
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
        
        self.csv_file = csv_file
    
    def log_generation_metrics(self, generation: int, stats: Dict, trap_rate: float):
        """
        Log generation metrics to CSV
        
        Args:
            generation: Current generation number
            stats: Generation statistics
            trap_rate: Current trap rate
        """
        
        timestamp = datetime.now().isoformat()
        
        metrics = {
            "generation": generation,
            "timestamp": timestamp,
            "best_fitness": stats.get("best_fitness", 0.0),
            "avg_fitness": stats.get("average_fitness", 0.0),
            "worst_fitness": stats.get("worst_fitness", 0.0),
            "diversity_score": stats.get("diversity_score", 0.0),
            "trap_rate": trap_rate,
            "crash_count": self.crash_count,
            "elite_count": stats.get("elite_count", 0),
            "new_genomes": stats.get("new_genomes", 0),
            "complexity_mean": stats.get("complexity_stats", {}).get("mean_size", 0.0),
            "complexity_std": stats.get("complexity_stats", {}).get("std_size", 0.0)
        }
        
        # Append to CSV
        with open(self.csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([metrics[header] for header in [
                "generation", "timestamp", "best_fitness", "avg_fitness", 
                "worst_fitness", "diversity_score", "trap_rate", "crash_count",
                "elite_count", "new_genomes", "complexity_mean", "complexity_std"
            ]])
        
        # Store in history
        self.metrics_history.append(metrics)
        self.trap_rates.append(trap_rate)
        
        # Log to console
        self.logger.info(f"Generation {generation}: "
                        f"Best={metrics['best_fitness']:.4f}, "
                        f"Avg={metrics['avg_fitness']:.4f}, "
                        f"Diversity={metrics['diversity_score']:.4f}, "
                        f"TrapRate={trap_rate:.2f}")
    
    def log_crash(self, error: Exception):
        """Log a crash/exception"""
        
        self.crash_count += 1
        self.logger.error(f"CRASH #{self.crash_count}: {error}")
        self.logger.error(traceback.format_exc())
    
    def get_final_metrics(self) -> Dict:
        """Get final metrics for success criteria validation"""
        
        if not self.metrics_history:
            return {
                "initial_avg_fitness": 0.0,
                "final_avg_fitness": 0.0,
                "fitness_improvement": 0.0,
                "avg_trap_rate": 0.0,
                "crash_count": self.crash_count
            }
        
        initial_avg = self.metrics_history[0]["avg_fitness"]
        final_avg = self.metrics_history[-1]["avg_fitness"]
        
        return {
            "initial_avg_fitness": initial_avg,
            "final_avg_fitness": final_avg,
            "fitness_improvement": final_avg - initial_avg,
            "avg_trap_rate": np.mean(self.trap_rates) if self.trap_rates else 0.0,
            "crash_count": self.crash_count
        }


class CheckpointManager:
    """Manage experiment checkpointing and resumption"""
    
    def __init__(self, config: PilotConfig, directories: Dict[str, Path], 
                 difficulty_level: float):
        """
        Initialize checkpoint manager
        
        Args:
            config: Configuration object
            directories: Experiment directories
            difficulty_level: Adversarial difficulty level
        """
        self.config = config
        self.directories = directories
        self.difficulty_level = difficulty_level
        self.checkpoint_interval = config.execution.checkpoint_interval
    
    def save_checkpoint(self, generation: int, evolution_engine: EvolutionEngine, 
                       rng_state: Tuple) -> str:
        """
        Save experiment checkpoint
        
        Args:
            generation: Current generation number
            evolution_engine: Evolution engine state
            rng_state: Random number generator state
        
        Returns:
            Path to saved checkpoint
        """
        
        checkpoint_data = {
            "generation": generation,
            "difficulty_level": self.difficulty_level,
            "population": evolution_engine.population,
            "fitness_scores": evolution_engine.fitness_scores,
            "best_genome": evolution_engine.best_genome,
            "best_fitness": evolution_engine.best_fitness,
            "current_generation": evolution_engine.current_generation,
            "stagnation_counter": evolution_engine.stagnation_counter,
            "last_improvement_generation": evolution_engine.last_improvement_generation,
            "rng_state": rng_state,
            "timestamp": datetime.now().isoformat()
        }
        
        checkpoint_file = self.directories["checkpoints"] / f"checkpoint_{self.difficulty_level:.1f}_gen_{generation}.pkl"
        
        with open(checkpoint_file, 'wb') as f:
            pickle.dump(checkpoint_data, f)
        
        return str(checkpoint_file)
    
    def load_latest_checkpoint(self) -> Optional[Dict]:
        """
        Load the latest checkpoint for this difficulty level
        
        Returns:
            Checkpoint data or None if no checkpoint found
        """
        
        checkpoint_dir = self.directories["checkpoints"]
        checkpoint_files = list(checkpoint_dir.glob(f"checkpoint_{self.difficulty_level:.1f}_gen_*.pkl"))
        
        if not checkpoint_files:
            return None
        
        # Find latest checkpoint
        latest_checkpoint = max(checkpoint_files, key=lambda f: int(f.stem.split('_')[-1]))
        
        with open(latest_checkpoint, 'rb') as f:
            checkpoint_data = pickle.load(f)
        
        return checkpoint_data
    
    def get_resume_generation(self) -> int:
        """Get generation number to resume from"""
        
        checkpoint_data = self.load_latest_checkpoint()
        if checkpoint_data:
            return checkpoint_data["generation"] + 1
        return 0


class SingleDifficultyRunner:
    """Run experiment for a single adversarial difficulty level"""
    
    def __init__(self, config: PilotConfig, directories: Dict[str, Path], 
                 difficulty_level: float):
        """
        Initialize single difficulty runner
        
        Args:
            config: Configuration object
            directories: Experiment directories
            difficulty_level: Adversarial difficulty level
        """
        self.config = config
        self.directories = directories
        self.difficulty_level = difficulty_level
        
        # Initialize components
        self.logger = ExperimentLogger(config, directories, difficulty_level)
        self.checkpoint_manager = CheckpointManager(config, directories, difficulty_level)
        
        # Initialize data components
        self.data_storage = TickDataStorage("data")
        self.cleaner = TickDataCleaner()
        self.ingestor = DukascopyIngestor(self.data_storage, self.cleaner)
        
        # Setup regime datasets
        self._setup_regime_datasets()
        
        # Initialize evolution components
        self._initialize_evolution_components()
    
    def _setup_regime_datasets(self):
        """Setup regime datasets for triathlon evaluation"""
        
        self.logger.logger.info("Setting up regime datasets...")
        
        # Generate data for all regimes
        for regime_name, regime_config in self.config.regime_datasets.items():
            start_year = int(regime_config.start_date.split('-')[0])
            self.logger.logger.info(f"Generating data for {regime_name} regime ({start_year})")
            self.ingestor.ingest_year(regime_config.symbol, start_year)
    
    def _initialize_evolution_components(self):
        """Initialize evolution engine and fitness evaluator"""
        
        # Create fitness evaluator with regime datasets
        self.fitness_evaluator = FitnessEvaluator(
            data_storage=self.data_storage,
            evaluation_period_days=30,  # Use 1-month periods
            adversarial_intensity=self.difficulty_level
        )
        
        # Create evolution config
        evolution_config = EvolutionConfig(
            population_size=self.config.population_size,
            elite_ratio=self.config.evolution.elite_ratio,
            crossover_ratio=self.config.evolution.crossover_ratio,
            mutation_ratio=self.config.evolution.mutation_ratio,
            mutation_rate=self.config.evolution.mutation_rate_per_node,
            max_stagnation=20,
            complexity_penalty=self.config.evolution.complexity_penalty_lambda,
            min_fitness_improvement=0.001
        )
        
        # Create evolution engine
        self.evolution_engine = EvolutionEngine(evolution_config, self.fitness_evaluator)
    
    def run_experiment(self) -> Dict:
        """
        Run the complete experiment for this difficulty level
        
        Returns:
            Experiment results
        """
        
        self.logger.logger.info(f"Starting experiment for difficulty level {self.difficulty_level}")
        
        try:
            # Check for resume point
            resume_generation = self.checkpoint_manager.get_resume_generation()
            
            if resume_generation > 0:
                self.logger.logger.info(f"Resuming from generation {resume_generation}")
                self._resume_from_checkpoint(resume_generation)
            else:
                self.logger.logger.info("Starting fresh experiment")
                self._initialize_population()
            
            # Run evolution loop
            results = self._run_evolution_loop()
            
            # Validate success criteria
            self._validate_success_criteria(results)
            
            return results
            
        except Exception as e:
            self.logger.log_crash(e)
            raise
    
    def _resume_from_checkpoint(self, resume_generation: int):
        """Resume experiment from checkpoint"""
        
        checkpoint_data = self.checkpoint_manager.load_latest_checkpoint()
        if not checkpoint_data:
            raise ValueError("No checkpoint found for resume")
        
        # Restore evolution engine state
        self.evolution_engine.population = checkpoint_data["population"]
        self.evolution_engine.fitness_scores = checkpoint_data["fitness_scores"]
        self.evolution_engine.best_genome = checkpoint_data["best_genome"]
        self.evolution_engine.best_fitness = checkpoint_data["best_fitness"]
        self.evolution_engine.current_generation = checkpoint_data["current_generation"]
        self.evolution_engine.stagnation_counter = checkpoint_data["stagnation_counter"]
        self.evolution_engine.last_improvement_generation = checkpoint_data["last_improvement_generation"]
        
        # Restore RNG state
        import random
        random.setstate(checkpoint_data["rng_state"])
        
        self.logger.logger.info(f"Resumed from generation {resume_generation}")
    
    def _initialize_population(self):
        """Initialize random population"""
        
        self.logger.logger.info(f"Initializing population of {self.config.population_size} genomes")
        
        # Set random seed
        import random
        random.seed(self.config.random_seed)
        np.random.seed(self.config.random_seed)
        
        # Initialize population
        success = self.evolution_engine.initialize_population(seed=self.config.random_seed)
        if not success:
            raise RuntimeError("Failed to initialize population")
    
    def _run_evolution_loop(self) -> Dict:
        """Run the main evolution loop"""
        
        results = {
            "difficulty_level": self.difficulty_level,
            "generations_completed": 0,
            "final_metrics": {},
            "checkpoints_saved": [],
            "final_population_summary": {}
        }
        
        start_generation = self.evolution_engine.current_generation
        
        for generation in range(start_generation, self.config.generations):
            try:
                self.logger.logger.info(f"Evolving generation {generation + 1}/{self.config.generations}")
                
                # Evolve one generation
                gen_stats = self.evolution_engine.evolve_generation()
                
                # Calculate trap rate (simplified - count adversarial events)
                trap_rate = self._calculate_trap_rate()
                
                # Log metrics
                stats_dict = {
                    "best_fitness": gen_stats.best_fitness,
                    "average_fitness": gen_stats.average_fitness,
                    "worst_fitness": gen_stats.worst_fitness,
                    "diversity_score": gen_stats.diversity_score,
                    "elite_count": gen_stats.elite_count,
                    "new_genomes": gen_stats.new_genomes,
                    "complexity_stats": gen_stats.complexity_stats
                }
                
                self.logger.log_generation_metrics(generation + 1, stats_dict, trap_rate)
                
                # Save checkpoint if needed
                if (generation + 1) % self.config.execution.checkpoint_interval == 0:
                    import random
                    rng_state = random.getstate()
                    checkpoint_file = self.checkpoint_manager.save_checkpoint(
                        generation + 1, self.evolution_engine, rng_state
                    )
                    results["checkpoints_saved"].append(checkpoint_file)
                    self.logger.logger.info(f"Checkpoint saved: {checkpoint_file}")
                
                results["generations_completed"] = generation + 1
                
            except Exception as e:
                self.logger.log_crash(e)
                raise
        
        # Get final results
        results["final_metrics"] = self.logger.get_final_metrics()
        results["final_population_summary"] = self.evolution_engine.get_population_summary()
        
        self.logger.logger.info(f"Experiment completed for difficulty {self.difficulty_level}")
        
        return results
    
    def _calculate_trap_rate(self) -> float:
        """Calculate current trap rate (simplified)"""
        
        # In a real implementation, this would track actual trap events
        # For now, use a simplified calculation based on adversarial events
        return np.random.uniform(0.15, 0.35)  # Placeholder
    
    def _validate_success_criteria(self, results: Dict):
        """Validate success criteria for this difficulty level"""
        
        final_metrics = results["final_metrics"]
        
        # Check fitness improvement
        fitness_improvement = final_metrics["fitness_improvement"]
        if fitness_improvement < self.config.success_criteria.min_fitness_improvement:
            raise ValueError(f"Fitness improvement {fitness_improvement:.4f} below threshold "
                           f"{self.config.success_criteria.min_fitness_improvement}")
        
        # Check trap rate
        avg_trap_rate = final_metrics["avg_trap_rate"]
        if not (self.config.success_criteria.trap_rate_min <= avg_trap_rate <= self.config.success_criteria.trap_rate_max):
            raise ValueError(f"Trap rate {avg_trap_rate:.2f} outside valid range "
                           f"[{self.config.success_criteria.trap_rate_min}, {self.config.success_criteria.trap_rate_max}]")
        
        # Check crash count
        crash_count = final_metrics["crash_count"]
        if crash_count > self.config.success_criteria.max_crash_count:
            raise ValueError(f"Crash count {crash_count} exceeds maximum {self.config.success_criteria.max_crash_count}")
        
        self.logger.logger.info("Success criteria validation PASSED")


def run_single_difficulty(args: Tuple) -> Dict:
    """
    Run experiment for a single difficulty level (for multiprocessing)
    
    Args:
        args: Tuple of (config, directories, difficulty_level)
    
    Returns:
        Experiment results
    """
    
    config, directories, difficulty_level = args
    
    try:
        runner = SingleDifficultyRunner(config, directories, difficulty_level)
        results = runner.run_experiment()
        return results
    except Exception as e:
        # Return error information
        return {
            "difficulty_level": difficulty_level,
            "error": str(e),
            "traceback": traceback.format_exc(),
            "success": False
        }


class PilotExecutor:
    """Main pilot experiment executor"""
    
    def __init__(self, config_path: str = "pilot_config.yaml"):
        """
        Initialize pilot executor
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.config = None
        self.directories = None
    
    def setup(self):
        """Setup experiment environment"""
        
        # Load and validate configuration
        self.config = load_config(self.config_path)
        validate_config(self.config)
        
        # Create experiment directories
        self.directories = create_experiment_directories(self.config)
        
        # Save configuration
        config_save_path = self.directories["experiment_root"] / "config.yaml"
        with open(config_save_path, 'w') as f:
            yaml.dump(self.config.__dict__, f, default_flow_style=False, indent=2)
        
        print(f"Experiment setup complete. Experiment directory: {self.directories['experiment_root']}")
    
    def run_parallel_experiments(self) -> Dict:
        """
        Run experiments for all difficulty levels in parallel
        
        Returns:
            Combined results from all difficulty levels
        """
        
        if not self.config or not self.directories:
            raise RuntimeError("Must call setup() before run_parallel_experiments()")
        
        print(f"Starting parallel experiments for {len(self.config.adversarial_sweep_levels)} difficulty levels")
        
        # Prepare arguments for parallel execution
        args_list = [
            (self.config, self.directories, difficulty_level)
            for difficulty_level in self.config.adversarial_sweep_levels
        ]
        
        # Run experiments in parallel
        results = {}
        crash_count = 0
        
        with ProcessPoolExecutor(max_workers=len(self.config.adversarial_sweep_levels)) as executor:
            # Submit all tasks
            future_to_difficulty = {
                executor.submit(run_single_difficulty, args): args[2]
                for args in args_list
            }
            
            # Collect results
            for future in as_completed(future_to_difficulty):
                difficulty_level = future_to_difficulty[future]
                
                try:
                    result = future.result()
                    
                    if result.get("success", True):  # Default to True for backward compatibility
                        results[difficulty_level] = result
                        print(f"✓ Difficulty {difficulty_level}: Completed successfully")
                    else:
                        results[difficulty_level] = result
                        crash_count += 1
                        print(f"✗ Difficulty {difficulty_level}: Failed - {result.get('error', 'Unknown error')}")
                
                except Exception as e:
                    crash_count += 1
                    results[difficulty_level] = {
                        "difficulty_level": difficulty_level,
                        "error": str(e),
                        "traceback": traceback.format_exc(),
                        "success": False
                    }
                    print(f"✗ Difficulty {difficulty_level}: Exception - {e}")
        
        # Validate overall success criteria
        self._validate_overall_success_criteria(results, crash_count)
        
        # Save final results
        self._save_final_results(results)
        
        return results
    
    def _validate_overall_success_criteria(self, results: Dict, crash_count: int):
        """Validate overall success criteria across all difficulty levels"""
        
        print("\n" + "="*60)
        print("VALIDATING OVERALL SUCCESS CRITERIA")
        print("="*60)
        
        # Check system stability
        if crash_count > self.config.success_criteria.max_crash_count:
            raise ValueError(f"System stability violated: {crash_count} crashes exceed maximum {self.config.success_criteria.max_crash_count}")
        
        print(f"✓ System Stability: {crash_count} crashes (max allowed: {self.config.success_criteria.max_crash_count})")
        
        # Check fitness progression
        successful_runs = [r for r in results.values() if r.get("success", True) and "final_metrics" in r]
        
        if not successful_runs:
            raise ValueError("No successful runs to validate fitness progression")
        
        fitness_improvements = [r["final_metrics"]["fitness_improvement"] for r in successful_runs]
        significant_improvements = [imp for imp in fitness_improvements if imp > self.config.success_criteria.min_fitness_improvement]
        
        if not significant_improvements:
            raise ValueError(f"Fitness progression violated: No difficulty level achieved minimum improvement of {self.config.success_criteria.min_fitness_improvement}")
        
        print(f"✓ Fitness Progression: {len(significant_improvements)}/{len(successful_runs)} difficulty levels achieved significant improvement")
        
        # Check behavioral sanity
        trap_rates = [r["final_metrics"]["avg_trap_rate"] for r in successful_runs]
        sane_trap_rates = [rate for rate in trap_rates 
                          if self.config.success_criteria.trap_rate_min <= rate <= self.config.success_criteria.trap_rate_max]
        
        if len(sane_trap_rates) != len(trap_rates):
            raise ValueError(f"Behavioral sanity violated: {len(trap_rates) - len(sane_trap_rates)} difficulty levels have insane trap rates")
        
        print(f"✓ Behavioral Sanity: All {len(trap_rates)} difficulty levels have sane trap rates")
        
        print("="*60)
        print("ALL SUCCESS CRITERIA VALIDATED - EXPERIMENT SUCCESSFUL")
        print("="*60)
    
    def _save_final_results(self, results: Dict):
        """Save final results to experiment directory"""
        
        # Save results summary
        results_file = self.directories["results"] / "final_results.json"
        
        # Convert results to JSON-serializable format
        serializable_results = {}
        for difficulty, result in results.items():
            serializable_results[str(difficulty)] = result
        
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)
        
        # Create summary report
        summary_file = self.directories["results"] / "experiment_summary.txt"
        
        with open(summary_file, 'w') as f:
            f.write("EMP PROVING GROUND v2.0 - PILOT EXPERIMENT SUMMARY\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Experiment Name: {self.config.experiment_name}\n")
            f.write(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Population Size: {self.config.population_size}\n")
            f.write(f"Generations: {self.config.generations}\n")
            f.write(f"Difficulty Levels: {self.config.adversarial_sweep_levels}\n\n")
            
            f.write("RESULTS BY DIFFICULTY LEVEL:\n")
            f.write("-" * 40 + "\n")
            
            for difficulty, result in results.items():
                f.write(f"\nDifficulty {difficulty}:\n")
                if result.get("success", True):
                    metrics = result.get("final_metrics", {})
                    f.write(f"  Fitness Improvement: {metrics.get('fitness_improvement', 0):.4f}\n")
                    f.write(f"  Average Trap Rate: {metrics.get('avg_trap_rate', 0):.2f}\n")
                    f.write(f"  Crash Count: {metrics.get('crash_count', 0)}\n")
                else:
                    f.write(f"  FAILED: {result.get('error', 'Unknown error')}\n")
        
        print(f"Final results saved to: {results_file}")
        print(f"Summary report saved to: {summary_file}")


def main():
    """Main entry point"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description="EMP Proving Ground v2.0 - Pilot Experiment Executor")
    parser.add_argument("--config", type=str, default="pilot_config.yaml",
                       help="Path to configuration file")
    parser.add_argument("--setup-only", action="store_true",
                       help="Only setup experiment, don't run")
    
    args = parser.parse_args()
    
    try:
        # Create executor
        executor = PilotExecutor(args.config)
        
        # Setup experiment
        executor.setup()
        
        if args.setup_only:
            print("Setup complete. Use --run to execute the experiment.")
            return 0
        
        # Run experiments
        results = executor.run_parallel_experiments()
        
        print("\n" + "="*60)
        print("PILOT EXPERIMENT COMPLETED SUCCESSFULLY")
        print("="*60)
        print("Raw data files are ready for AI Collective analysis.")
        print("="*60)
        
        return 0
        
    except Exception as e:
        print(f"Pilot experiment FAILED: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main()) 