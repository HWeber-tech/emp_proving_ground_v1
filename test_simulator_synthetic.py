"""
Test script to verify MarketSimulator with synthetic data.
"""
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

# Add the parent directory to the path so we can import emp modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from emp.simulation.simulator import MarketSimulator, OrderSide, OrderType

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SyntheticDataStorage:
    """Mock data storage for testing with synthetic data."""
    
    def __init__(self):
        self.synthetic_data = None
    
    def load_tick_data(self, symbol: str, start_time: datetime, end_time: datetime) -> pd.DataFrame:
        """Generate synthetic tick data."""
        logger.info(f"Generating synthetic data for {symbol} from {start_time} to {end_time}")
        
        # Generate time series
        time_range = pd.date_range(start=start_time, end=end_time, freq='1min')
        
        # Generate synthetic price data with some trend and noise
        np.random.seed(42)  # For reproducible results
        
        # Base price around 1.1000 (typical EURUSD level)
        base_price = 1.1000
        
        # Add trend and noise
        trend = np.linspace(0, 0.01, len(time_range))  # Small upward trend
        noise = np.random.normal(0, 0.0001, len(time_range))  # Small noise
        price_changes = trend + noise
        
        # Calculate mid prices
        mid_prices = base_price + np.cumsum(price_changes)
        
        # Generate bid/ask spreads
        spreads = np.random.uniform(0.0001, 0.0003, len(time_range))  # 1-3 pips
        
        # Calculate bid and ask prices
        bids = mid_prices - spreads / 2
        asks = mid_prices + spreads / 2
        
        # Generate volumes
        bid_volumes = np.random.uniform(1000, 10000, len(time_range))
        ask_volumes = np.random.uniform(1000, 10000, len(time_range))
        
        # Create DataFrame
        df = pd.DataFrame({
            'timestamp': time_range,
            'bid': bids,
            'ask': asks,
            'bid_volume': bid_volumes,
            'ask_volume': ask_volumes,
            'mid_price': mid_prices,
            'spread': spreads
        })
        
        self.synthetic_data = df
        logger.info(f"Generated {len(df)} synthetic ticks")
        
        return df

def test_simulator_with_synthetic_data():
    """Test the MarketSimulator with synthetic data."""
    logger.info("=" * 60)
    logger.info("SYNTHETIC DATA SIMULATOR TEST")
    logger.info("=" * 60)
    
    try:
        # Create synthetic data storage
        storage = SyntheticDataStorage()
        
        # Define test period
        start_time = datetime(2023, 1, 1, 0, 0, 0)
        end_time = datetime(2023, 1, 7, 23, 59, 59)  # 1 week of data
        
        # Initialize simulator
        simulator = MarketSimulator(
            data_storage=storage,
            initial_balance=100000.0,
            leverage=1.0
        )
        
        # Load synthetic data
        logger.info("Loading synthetic data...")
        simulator.load_data("EURUSD", start_time, end_time)
        
        # Run simple strategy
        logger.info("Running simple strategy test...")
        results = simulator.run_simple_strategy()
        
        # Analyze results
        logger.info("=" * 60)
        logger.info("SYNTHETIC TEST RESULTS")
        logger.info("=" * 60)
        
        logger.info(f"Strategy: {results['strategy_name']}")
        logger.info(f"Total Trades: {results['total_trades']}")
        logger.info(f"Final Balance: ${results['final_balance']:.2f}")
        logger.info(f"Total PnL: ${results['total_pnl']:.2f}")
        logger.info(f"Return: {results['return_pct']:.2f}%")
        logger.info(f"Total Commission: ${results['total_commission']:.2f}")
        logger.info(f"Total Slippage: ${results['total_slippage']:.2f}")
        
        # Check if results are reasonable
        success = True
        
        if results['total_trades'] == 0:
            logger.warning("No trades executed - this might be expected with synthetic data")
        
        if abs(results['total_pnl']) > 10000:
            logger.warning("Very large PnL - might indicate an issue")
            success = False
        
        if results['total_commission'] < 0 or results['total_slippage'] < 0:
            logger.error("Negative transaction costs - this is wrong")
            success = False
        
        if success:
            logger.info("\n✅ Synthetic data test passed!")
            logger.info("MarketSimulator is working correctly with synthetic data.")
        else:
            logger.error("\n❌ Synthetic data test failed!")
        
        return success
        
    except Exception as e:
        logger.error(f"Synthetic data test failed: {e}")
        return False

def main():
    """Main function to run synthetic data test."""
    success = test_simulator_with_synthetic_data()
    
    if success:
        logger.info("\nReady to test with real data!")
        return 0
    else:
        logger.error("\nSimulator needs fixes before testing with real data.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 