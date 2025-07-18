#!/usr/bin/env python3
"""
EMP Proving Ground v1.0 - Complete Evolutionary Trading System

This script integrates all components to run a complete evolutionary
trading strategy optimization system.

Components:
- Data Pipeline (DukascopyIngestor, TickDataCleaner, TickDataStorage)
- Market Simulation (MarketSimulator, AdversarialEngine)
- Sensory Cortex (4D+1 perception system)
- Decision Genome (Evolutionary decision trees)
- Fitness Evaluation (Multi-objective fitness scoring)
- Evolution Engine (Population management and evolution)
"""

import os
import sys
import logging
import argparse
from datetime import datetime, timedelta
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from emp.data.ingestion import DukascopyIngestor
from emp.data.cleaning import TickDataCleaner
from emp.data.storage import TickDataStorage
from emp.data.regimes import MarketRegimeIdentifier
from emp.simulation.market import MarketSimulator
from emp.simulation.adversary import AdversarialEngine
from emp.agent.sensory import SensoryCortex
from emp.agent.genome import DecisionGenome
from emp.evolution.fitness import FitnessEvaluator, EvolutionConfig
from emp.evolution.engine import EvolutionEngine

def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('emp_evolution.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )

def download_data(symbols: list, years: list, data_dir: str = "data"):
    """Download and prepare market data."""
    logger = logging.getLogger(__name__)
    logger.info("Starting data download and preparation")
    
    # Initialize components
    storage = TickDataStorage(data_dir=data_dir)
    cleaner = TickDataCleaner()
    ingestor = DukascopyIngestor(storage, cleaner)
    
    # Download data for each symbol and year
    for symbol in symbols:
        for year in years:
            logger.info(f"Downloading {symbol} data for {year}")
            success = ingestor.ingest_year(symbol, year)
            if success:
                logger.info(f"Successfully downloaded {symbol} {year}")
            else:
                logger.warning(f"Failed to download {symbol} {year}")
    
    logger.info("Data download completed")

def identify_regimes(symbols: list, start_year: int, end_year: int, data_dir: str = "data"):
    """Identify market regimes in the data."""
    logger = logging.getLogger(__name__)
    logger.info("Starting regime identification")
    
    # Initialize components
    storage = TickDataStorage(data_dir=data_dir)
    regime_identifier = MarketRegimeIdentifier(storage)
    
    # Identify regimes for each symbol
    for symbol in symbols:
        logger.info(f"Identifying regimes for {symbol}")
        regimes = regime_identifier.identify_regimes(symbol, start_year, end_year)
        logger.info(f"Found {len(regimes)} regimes for {symbol}")
    
    logger.info("Regime identification completed")

def run_evolution(config: dict):
    """Run the complete evolutionary optimization."""
    logger = logging.getLogger(__name__)
    logger.info("Starting evolutionary optimization")
    
    # Initialize data components
    storage = TickDataStorage(data_dir=config.get('data_dir', 'data'))
    cleaner = TickDataCleaner()
    
    # Initialize simulation components
    simulator = MarketSimulator(
        data_storage=storage,
        initial_balance=config.get('initial_balance', 100000.0),
        leverage=config.get('leverage', 1.0)
    )
    
    # Initialize adversarial engine
    adversary = AdversarialEngine(
        difficulty_level=config.get('adversarial_intensity', 0.7)
    )
    simulator.add_adversarial_callback(adversary.apply_adversarial_effects)
    
    # Initialize sensory cortex
    sensory_cortex = SensoryCortex(
        symbol=config.get('symbol', 'EURUSD'),
        data_storage=storage
    )
    
    # Calibrate sensory cortex
    end_time = datetime.now()
    start_time = end_time - timedelta(days=config.get('calibration_days', 30))
    sensory_cortex.calibrate(start_time, end_time)
    
    # Initialize fitness evaluator
    fitness_evaluator = FitnessEvaluator(
        data_storage=storage,
        evaluation_period_days=config.get('evaluation_period_days', 30),
        adversarial_intensity=config.get('adversarial_intensity', 0.7),
        commission_rate=config.get('commission_rate', 0.0001),
        slippage_bps=config.get('slippage_bps', 0.5)
    )
    
    # Initialize evolution engine
    evolution_config = EvolutionConfig(
        population_size=config.get('population_size', 100),
        elite_ratio=config.get('elite_ratio', 0.1),
        crossover_ratio=config.get('crossover_ratio', 0.6),
        mutation_ratio=config.get('mutation_ratio', 0.3),
        mutation_rate=config.get('mutation_rate', 0.1),
        max_stagnation=config.get('max_stagnation', 20),
        complexity_penalty=config.get('complexity_penalty', 0.01),
        min_fitness_improvement=config.get('min_fitness_improvement', 0.001)
    )
    
    evolution_engine = EvolutionEngine(evolution_config, fitness_evaluator)
    
    # Initialize population
    logger.info("Initializing population")
    success = evolution_engine.initialize_population(seed=config.get('random_seed', 42))
    if not success:
        logger.error("Failed to initialize population")
        return
    
    # Run evolution
    max_generations = config.get('max_generations', 50)
    logger.info(f"Starting evolution for {max_generations} generations")
    
    for generation in range(max_generations):
        logger.info(f"Evolving generation {generation + 1}/{max_generations}")
        
        # Evolve one generation
        stats = evolution_engine.evolve_generation()
        
        # Log progress
        logger.info(f"Generation {stats.generation}: Best fitness = {stats.best_fitness:.4f}, "
                   f"Avg fitness = {stats.average_fitness:.4f}, "
                   f"Diversity = {stats.diversity_score:.4f}")
        
        # Save checkpoint every 10 generations
        if (generation + 1) % 10 == 0:
            checkpoint_file = f"checkpoints/population_gen_{generation + 1}.json"
            os.makedirs("checkpoints", exist_ok=True)
            evolution_engine.save_population(checkpoint_file)
            logger.info(f"Checkpoint saved: {checkpoint_file}")
        
        # Early stopping if fitness is good enough
        if stats.best_fitness > config.get('target_fitness', 0.8):
            logger.info(f"Target fitness reached: {stats.best_fitness:.4f}")
            break
    
    # Final results
    logger.info("Evolution completed")
    
    # Get best genomes
    best_genomes = evolution_engine.get_best_genomes(count=5)
    logger.info("Top 5 genomes:")
    for i, genome in enumerate(best_genomes):
        logger.info(f"  {i+1}. Genome {genome.genome_id}: Fitness = {genome.fitness_score:.4f}")
    
    # Save final population
    final_population_file = f"results/final_population_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    os.makedirs("results", exist_ok=True)
    evolution_engine.save_population(final_population_file)
    
    # Get evolution summary
    summary = evolution_engine.get_evolution_summary()
    logger.info("Evolution Summary:")
    for key, value in summary.items():
        logger.info(f"  {key}: {value}")
    
    return evolution_engine, best_genomes

def test_components():
    """Test individual components."""
    logger = logging.getLogger(__name__)
    logger.info("Testing individual components")
    
    # Test data storage
    storage = TickDataStorage()
    logger.info("✓ Data storage initialized")
    
    # Test market simulator
    simulator = MarketSimulator(storage)
    logger.info("✓ Market simulator initialized")
    
    # Test adversarial engine
    adversary = AdversarialEngine()
    logger.info("✓ Adversarial engine initialized")
    
    # Test sensory cortex
    sensory_cortex = SensoryCortex("EURUSD", storage)
    logger.info("✓ Sensory cortex initialized")
    
    # Test decision genome
    genome = DecisionGenome()
    logger.info("✓ Decision genome initialized")
    
    # Test fitness evaluator
    fitness_evaluator = FitnessEvaluator(storage)
    logger.info("✓ Fitness evaluator initialized")
    
    # Test evolution engine
    config = EvolutionConfig(population_size=10)
    evolution_engine = EvolutionEngine(config, fitness_evaluator)
    logger.info("✓ Evolution engine initialized")
    
    logger.info("All components tested successfully")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="EMP Proving Ground v1.0 - Evolutionary Trading System")
    parser.add_argument("--mode", choices=["download", "regimes", "evolution", "test"], 
                       default="evolution", help="Operation mode")
    parser.add_argument("--symbols", nargs="+", default=["EURUSD"], 
                       help="Trading symbols")
    parser.add_argument("--years", nargs="+", type=int, default=[2023], 
                       help="Years to download")
    parser.add_argument("--config", type=str, default="config.yaml", 
                       help="Configuration file")
    parser.add_argument("--log-level", default="INFO", 
                       help="Logging level")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    logger.info("EMP Proving Ground v1.0 - Evolutionary Trading System")
    logger.info(f"Mode: {args.mode}")
    
    # Create necessary directories
    os.makedirs("data", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    if args.mode == "download":
        # Download market data
        download_data(args.symbols, args.years)
        
    elif args.mode == "regimes":
        # Identify market regimes
        start_year = min(args.years)
        end_year = max(args.years)
        identify_regimes(args.symbols, start_year, end_year)
        
    elif args.mode == "test":
        # Test components
        test_components()
        
    elif args.mode == "evolution":
        # Run evolutionary optimization
        config = {
            'data_dir': 'data',
            'symbol': args.symbols[0],
            'initial_balance': 100000.0,
            'leverage': 1.0,
            'adversarial_intensity': 0.7,
            'calibration_days': 30,
            'evaluation_period_days': 30,
            'commission_rate': 0.0001,
            'slippage_bps': 0.5,
            'population_size': 100,
            'elite_ratio': 0.1,
            'crossover_ratio': 0.6,
            'mutation_ratio': 0.3,
            'mutation_rate': 0.1,
            'max_stagnation': 20,
            'complexity_penalty': 0.01,
            'min_fitness_improvement': 0.001,
            'max_generations': 50,
            'target_fitness': 0.8,
            'random_seed': 42
        }
        
        # Load config file if it exists
        if os.path.exists(args.config):
            import yaml
            with open(args.config, 'r') as f:
                file_config = yaml.safe_load(f)
                config.update(file_config)
        
        evolution_engine, best_genomes = run_evolution(config)
        
        if evolution_engine and best_genomes:
            logger.info("Evolution completed successfully!")
            logger.info(f"Best genome fitness: {best_genomes[0].fitness_score:.4f}")
        else:
            logger.error("Evolution failed")
            sys.exit(1)

if __name__ == "__main__":
    main() 