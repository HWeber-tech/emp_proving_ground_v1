#!/usr/bin/env python3
"""
EMP Proving Ground - Main Entry Point

This script demonstrates the modular EMP Proving Ground system with:
- Risk Management Core
- PnL Engine
- 4D+1 Sensory Cortex
- Evolutionary Decision Trees
- Adversarial Market Simulation
"""

import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from decimal import Decimal

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.core import RiskConfig, InstrumentProvider, CurrencyConverter
from src.risk import RiskManager
from src.pnl import EnhancedPosition, TradeRecord
from src.data import TickDataStorage, TickDataCleaner, DukascopyIngestor
from src.sensory import SensoryCortex, SensoryReading
from src.evolution import EvolutionEngine, DecisionGenome, EvolutionConfig, FitnessEvaluator
from src.simulation import MarketSimulator, AdversarialEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_system():
    """Set up the EMP Proving Ground system"""
    logger.info("Setting up EMP Proving Ground system...")
    
    # Initialize core components
    risk_config = RiskConfig(
        max_risk_per_trade_pct=Decimal("0.02"),
        max_leverage=Decimal("10.0"),
        max_total_exposure_pct=Decimal("0.5"),
        max_drawdown_pct=Decimal("0.25")
    )
    
    instrument_provider = InstrumentProvider()
    currency_converter = CurrencyConverter()
    
    # Initialize data pipeline
    data_storage = TickDataStorage()
    data_cleaner = TickDataCleaner()
    data_ingestor = DukascopyIngestor(data_storage, data_cleaner)
    
    # Initialize risk management
    risk_manager = RiskManager(risk_config, instrument_provider)
    
    # Initialize sensory cortex
    sensory_cortex = SensoryCortex("EURUSD", data_storage)
    
    # Initialize evolution engine
    evolution_config = EvolutionConfig(
        population_size=100,
        elite_ratio=0.1,
        crossover_ratio=0.6,
        mutation_ratio=0.3
    )
    
    fitness_evaluator = FitnessEvaluator(data_storage)
    evolution_engine = EvolutionEngine(evolution_config, fitness_evaluator)
    
    # Initialize market simulation
    market_simulator = MarketSimulator(data_storage, initial_balance=100000.0)
    adversarial_engine = AdversarialEngine(difficulty_level=0.5)
    
    return {
        'risk_config': risk_config,
        'instrument_provider': instrument_provider,
        'currency_converter': currency_converter,
        'data_storage': data_storage,
        'data_cleaner': data_cleaner,
        'data_ingestor': data_ingestor,
        'risk_manager': risk_manager,
        'sensory_cortex': sensory_cortex,
        'evolution_engine': evolution_engine,
        'market_simulator': market_simulator,
        'adversarial_engine': adversarial_engine
    }


def generate_test_data(system):
    """Generate test data for the system"""
    logger.info("Generating test data...")
    
    # Generate data for EURUSD for 2023
    success = system['data_ingestor'].ingest_year("EURUSD", 2023)
    
    if success:
        logger.info("Test data generated successfully")
    else:
        logger.warning("Failed to generate test data, using synthetic data")
    
    return success


def calibrate_sensory_cortex(system):
    """Calibrate the sensory cortex"""
    logger.info("Calibrating sensory cortex...")
    
    # Calibrate with 30 days of historical data
    end_time = datetime.now()
    start_time = end_time - timedelta(days=30)
    
    system['sensory_cortex'].calibrate(start_time, end_time)
    logger.info("Sensory cortex calibrated")


def run_evolution(system, generations=10):
    """Run the evolution engine"""
    logger.info(f"Running evolution for {generations} generations...")
    
    # Initialize population
    success = system['evolution_engine'].initialize_population(seed=42)
    if not success:
        logger.error("Failed to initialize population")
        return
    
    # Run evolution
    for generation in range(generations):
        stats = system['evolution_engine'].evolve_generation()
        logger.info(f"Generation {stats.generation}: Best fitness = {stats.best_fitness:.4f}")
    
    # Get best genomes
    best_genomes = system['evolution_engine'].get_best_genomes(5)
    logger.info(f"Evolution completed. Best fitness: {best_genomes[0].fitness:.4f}")


def run_simulation(system, symbol="EURUSD", days=7):
    """Run a market simulation"""
    logger.info(f"Running simulation for {symbol} over {days} days...")
    
    # Set up simulation period
    end_time = datetime.now()
    start_time = end_time - timedelta(days=days)
    
    # Load data
    success = system['market_simulator'].load_data(symbol, start_time, end_time)
    if not success:
        logger.error("Failed to load data for simulation")
        return
    
    # Add adversarial engine
    system['market_simulator'].add_adversarial_callback(
        lambda: system['adversarial_engine'].update(
            system['market_simulator'].current_data.iloc[system['market_simulator'].current_index],
            system['market_simulator']
        )
    )
    
    # Run simulation
    step_count = 0
    while True:
        market_state = system['market_simulator'].step()
        if market_state is None:
            break
        
        step_count += 1
        if step_count % 1000 == 0:
            logger.info(f"Simulation step {step_count}")
    
    # Get results
    performance = system['market_simulator'].get_performance_stats()
    account_summary = system['market_simulator'].get_account_summary()
    
    logger.info(f"Simulation completed:")
    logger.info(f"  Total return: {performance.get('total_return', 0):.2%}")
    logger.info(f"  Sharpe ratio: {performance.get('sharpe_ratio', 0):.2f}")
    logger.info(f"  Max drawdown: {performance.get('max_drawdown', 0):.2%}")
    logger.info(f"  Total trades: {performance.get('total_trades', 0)}")
    logger.info(f"  Final equity: ${account_summary.get('equity', 0):,.2f}")


def demonstrate_components(system):
    """Demonstrate individual components"""
    logger.info("Demonstrating system components...")
    
    # Demonstrate risk management
    logger.info("=== Risk Management Demo ===")
    instrument = system['instrument_provider'].get_instrument("EURUSD")
    if instrument:
        position_size = system['risk_manager'].calculate_position_size(
            account_equity=100000,
            stop_loss_pips=50,
            instrument=instrument,
            account_currency="USD"
        )
        logger.info(f"Calculated position size: {position_size}")
    
    # Demonstrate PnL engine
    logger.info("=== PnL Engine Demo ===")
    position = EnhancedPosition(
        symbol="EURUSD",
        quantity=10000,
        avg_price=Decimal("1.1000"),
        entry_timestamp=datetime.now(),
        last_swap_time=datetime.now()
    )
    # Pass Decimal to update_unrealized_pnl to avoid type errors
    position.update_unrealized_pnl(Decimal("1.1050"))
    logger.info(f"Position PnL: ${position.get_total_pnl():.2f}")
    
    # Demonstrate sensory cortex
    logger.info("=== Sensory Cortex Demo ===")
    # Create a mock market state for demonstration
    from src.simulation import MarketState
    mock_market_state = MarketState(
        timestamp=datetime.now(),
        symbol="EURUSD",
        bid=1.1000,
        ask=1.1002,
        bid_volume=1000,
        ask_volume=1000,
        spread_bps=2.0,
        mid_price=1.1001
    )
    
    # This would require calibration data, so we'll just show the structure
    logger.info("Sensory cortex ready for market perception")
    
    # Demonstrate evolution engine
    logger.info("=== Evolution Engine Demo ===")
    population_summary = system['evolution_engine'].get_population_summary()
    logger.info(f"Population size: {population_summary.get('population_size', 0)}")
    
    # Demonstrate data pipeline
    logger.info("=== Data Pipeline Demo ===")
    cache_stats = system['data_storage'].get_cache_stats()
    logger.info(f"Cache hits: {cache_stats.get('hits', 0)}, misses: {cache_stats.get('misses', 0)}")


def main():
    """Main function"""
    logger.info("Starting EMP Proving Ground system...")
    
    try:
        # Set up system
        system = setup_system()
        
        # Generate test data
        generate_test_data(system)
        
        # Demonstrate components
        demonstrate_components(system)
        
        # Calibrate sensory cortex
        calibrate_sensory_cortex(system)
        
        # Run evolution (short run for demo)
        run_evolution(system, generations=3)
        
        # Run simulation
        run_simulation(system, days=1)
        
        logger.info("EMP Proving Ground system demonstration completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise


if __name__ == "__main__":
    main() 