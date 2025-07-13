"""
Test script to verify Phase 1 completion.
Tests the definition of done: MarketSimulator with real data and simple strategy.
"""
import sys
import os
import yaml
import logging
from datetime import datetime, timedelta

# Add the parent directory to the path so we can import emp modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from emp.data_pipeline.storage import TickDataStorage
from emp.simulation.simulator import MarketSimulator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        logger.error(f"Failed to load config from {config_path}: {e}")
        raise

def test_phase1_completion():
    """
    Test that Phase 1 meets the definition of done.
    """
    logger.info("=" * 60)
    logger.info("PHASE 1 COMPLETION TEST")
    logger.info("=" * 60)
    
    try:
        # Load configuration
        config = load_config()
        symbol = config['data']['symbol']
        processed_dir = config['data']['processed_dir']
        
        logger.info(f"Testing with symbol: {symbol}")
        logger.info(f"Data directory: {processed_dir}")
        
        # Initialize components
        storage = TickDataStorage(processed_dir)
        
        # Check if data is available
        data_range = storage.get_data_range(symbol)
        if data_range is None:
            logger.error(f"No data available for {symbol}")
            logger.error("Please run the download script first: python scripts/download_data.py")
            return False
        
        logger.info(f"Data available from {data_range[0]} to {data_range[1]}")
        
        # Define test period (1 year of data)
        end_date = data_range[1]
        start_date = end_date - timedelta(days=365)
        
        logger.info(f"Test period: {start_date} to {end_date}")
        
        # Initialize simulator
        simulator = MarketSimulator(
            data_storage=storage,
            initial_balance=100000.0,
            leverage=1.0
        )
        
        # Load data
        logger.info("Loading data into simulator...")
        simulator.load_data(symbol, start_date, end_date)
        
        # Run simple strategy
        logger.info("Running simple strategy test...")
        results = simulator.run_simple_strategy()
        
        # Analyze results
        logger.info("=" * 60)
        logger.info("TEST RESULTS")
        logger.info("=" * 60)
        
        # Check definition of done criteria
        criteria_met = []
        
        # 1. Can load a year of real EURUSD tick data
        if results['total_trades'] > 0:
            criteria_met.append("✓ Can load real tick data")
            logger.info("✓ Can load a year of real EURUSD tick data")
        else:
            logger.error("✗ Cannot load real tick data")
        
        # 2. Can execute a simple, hard-coded strategy
        if results['strategy_name'] == 'Simple MA Crossover':
            criteria_met.append("✓ Can execute simple strategy")
            logger.info("✓ Can execute a simple, hard-coded strategy")
        else:
            logger.error("✗ Cannot execute simple strategy")
        
        # 3. Produces a trade log
        if results['total_trades'] > 0:
            criteria_met.append("✓ Produces trade log")
            logger.info(f"✓ Produces a trade log ({results['total_trades']} trades)")
        else:
            logger.error("✗ No trades executed")
        
        # 4. Realistic, non-zero PnL
        total_pnl = results['total_pnl']
        if abs(total_pnl) > 0.01:  # More than 1 cent
            criteria_met.append("✓ Non-zero PnL")
            logger.info(f"✓ Realistic, non-zero PnL: ${total_pnl:.2f}")
        else:
            logger.error(f"✗ Zero or negligible PnL: ${total_pnl:.2f}")
        
        # 5. Transaction costs
        total_costs = results['total_commission'] + results['total_slippage']
        if total_costs > 0:
            criteria_met.append("✓ Transaction costs")
            logger.info(f"✓ Transaction costs: ${total_costs:.2f}")
            logger.info(f"  - Commission: ${results['total_commission']:.2f}")
            logger.info(f"  - Slippage: ${results['total_slippage']:.2f}")
        else:
            logger.error("✗ No transaction costs")
        
        # Print detailed results
        logger.info("\nDetailed Results:")
        logger.info(f"Strategy: {results['strategy_name']}")
        logger.info(f"Total Trades: {results['total_trades']}")
        logger.info(f"Final Balance: ${results['final_balance']:.2f}")
        logger.info(f"Total PnL: ${total_pnl:.2f}")
        logger.info(f"Return: {results['return_pct']:.2f}%")
        logger.info(f"Total Commission: ${results['total_commission']:.2f}")
        logger.info(f"Total Slippage: ${results['total_slippage']:.2f}")
        
        # Check if all criteria are met
        if len(criteria_met) == 5:
            logger.info("\n" + "=" * 60)
            logger.info("🎉 PHASE 1 DEFINITION OF DONE: ACHIEVED! 🎉")
            logger.info("=" * 60)
            logger.info("All criteria met:")
            for criterion in criteria_met:
                logger.info(f"  {criterion}")
            return True
        else:
            logger.error("\n" + "=" * 60)
            logger.error("❌ PHASE 1 DEFINITION OF DONE: NOT ACHIEVED ❌")
            logger.error("=" * 60)
            logger.error("Missing criteria:")
            missing_criteria = [
                "Can load real tick data",
                "Can execute simple strategy", 
                "Produces trade log",
                "Non-zero PnL",
                "Transaction costs"
            ]
            for criterion in missing_criteria:
                if criterion not in criteria_met:
                    logger.error(f"  ✗ {criterion}")
            return False
            
    except Exception as e:
        logger.error(f"Phase 1 test failed: {e}")
        return False

def main():
    """Main function to run Phase 1 completion test."""
    success = test_phase1_completion()
    
    if success:
        logger.info("\nPhase 1 is ready for Phase 2 implementation!")
        return 0
    else:
        logger.error("\nPhase 1 needs more work before proceeding to Phase 2.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 