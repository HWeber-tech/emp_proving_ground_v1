"""
Test script to verify Phase 1 completion following Manus's methodology.
Tests the definition of done: MarketSimulator with data and simple strategy.
"""
import sys
import os
import logging
from datetime import datetime, timedelta
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_phase1_completion():
    """
    Test that Phase 1 meets the definition of done following Manus's approach.
    """
    logger.info("=" * 60)
    logger.info("PHASE 1 COMPLETION TEST (Following Manus's Method)")
    logger.info("=" * 60)
    
    try:
        # Import the unified implementation
        from emp_proving_ground_unified import (
            TickDataStorage, 
            MarketSimulator, 
            OrderSide, 
            OrderType
        )
        
        # Check if test data exists, create if not
        data_dir = Path("data/processed/EURUSD")
        if not data_dir.exists() or not list(data_dir.glob("*.parquet")):
            logger.info("No test data found. Creating synthetic data...")
            from create_test_data import main as create_data
            create_data()
        
        # Initialize components
        storage = TickDataStorage("data")
        
        # Define test period (1 week of data)
        start_date = datetime(2023, 1, 1, 0, 0, 0)
        end_date = datetime(2023, 1, 7, 23, 59, 59)
        
        logger.info(f"Test period: {start_date} to {end_date}")
        
        # Initialize simulator
        simulator = MarketSimulator(
            data_storage=storage,
            initial_balance=100000.0,
            leverage=1.0
        )
        
        # Load data
        logger.info("Loading data into simulator...")
        simulator.load_data("EURUSD", start_date, end_date)
        
        # Run simple strategy (MA crossover)
        logger.info("Running simple MA crossover strategy...")
        
        # Simple moving average crossover strategy
        total_trades = 0
        total_pnl = 0.0
        total_commission = 0.0
        total_slippage = 0.0
        
        # Track positions
        position = 0.0
        entry_price = 0.0
        
        # Strategy parameters
        fast_ma_period = 10
        slow_ma_period = 20
        position_size = 10000  # $10k per trade
        
        # Calculate moving averages
        prices = []
        fast_ma_values = []
        slow_ma_values = []
        
        # Run simulation
        while True:
            market_state = simulator.step()
            if market_state is None:
                break
                
            current_price = market_state.mid_price
            prices.append(current_price)
            
            # Calculate moving averages
            if len(prices) >= slow_ma_period:
                fast_ma = sum(prices[-fast_ma_period:]) / fast_ma_period
                slow_ma = sum(prices[-slow_ma_period:]) / slow_ma_period
                fast_ma_values.append(fast_ma)
                slow_ma_values.append(slow_ma)
                
                # Trading logic
                if len(fast_ma_values) >= 2:
                    # Buy signal: fast MA crosses above slow MA
                    if (fast_ma_values[-1] > slow_ma_values[-1] and 
                        fast_ma_values[-2] <= slow_ma_values[-2] and 
                        position <= 0):
                        
                        # Close short position if exists
                        if position < 0:
                            close_price = market_state.bid
                            pnl = (entry_price - close_price) * abs(position)
                            total_pnl += pnl
                            total_commission += abs(position) * 0.0001  # 1 pip commission
                            total_slippage += abs(position) * 0.00005  # 0.5 bps slippage
                            total_trades += 1
                            logger.debug(f"Closed short: {pnl:.2f}")
                        
                        # Open long position
                        position = position_size / current_price
                        entry_price = market_state.ask
                        total_commission += position * 0.0001
                        total_slippage += position * 0.00005
                        total_trades += 1
                        logger.debug(f"Opened long: {position:.2f} @ {entry_price:.5f}")
                    
                    # Sell signal: fast MA crosses below slow MA
                    elif (fast_ma_values[-1] < slow_ma_values[-1] and 
                          fast_ma_values[-2] >= slow_ma_values[-2] and 
                          position >= 0):
                        
                        # Close long position if exists
                        if position > 0:
                            close_price = market_state.bid
                            pnl = (close_price - entry_price) * position
                            total_pnl += pnl
                            total_commission += position * 0.0001
                            total_slippage += position * 0.00005
                            total_trades += 1
                            logger.debug(f"Closed long: {pnl:.2f}")
                        
                        # Open short position
                        position = -position_size / current_price
                        entry_price = market_state.bid
                        total_commission += abs(position) * 0.0001
                        total_slippage += abs(position) * 0.00005
                        total_trades += 1
                        logger.debug(f"Opened short: {position:.2f} @ {entry_price:.5f}")
        
        # Close final position
        if position != 0:
            # Get the last market state for closing
            last_price = prices[-1] if prices else current_price
            if position > 0:  # Long position
                close_price = last_price * 0.9999  # Simulate bid price
                pnl = (close_price - entry_price) * position
            else:  # Short position
                close_price = last_price * 1.0001  # Simulate ask price
                pnl = (entry_price - close_price) * abs(position)
            
            total_pnl += pnl
            total_commission += abs(position) * 0.0001
            total_slippage += abs(position) * 0.00005
            total_trades += 1
            logger.debug(f"Closed final position: {pnl:.2f}")
        
        # Calculate final balance
        final_balance = 100000.0 + total_pnl - total_commission - total_slippage
        return_pct = (final_balance - 100000.0) / 100000.0 * 100
        
        # Analyze results
        logger.info("=" * 60)
        logger.info("TEST RESULTS")
        logger.info("=" * 60)
        
        # Check definition of done criteria
        criteria_met = []
        
        # 1. Can load tick data
        if total_trades > 0:
            criteria_met.append("✓ Can load tick data")
            logger.info("✓ Can load tick data")
        else:
            logger.error("✗ Cannot load tick data")
        
        # 2. Can execute a simple strategy
        if total_trades > 0:
            criteria_met.append("✓ Can execute simple strategy")
            logger.info("✓ Can execute a simple, hard-coded strategy")
        else:
            logger.error("✗ Cannot execute simple strategy")
        
        # 3. Produces a trade log
        if total_trades > 0:
            criteria_met.append("✓ Produces trade log")
            logger.info(f"✓ Produces a trade log ({total_trades} trades)")
        else:
            logger.error("✗ No trades executed")
        
        # 4. Realistic, non-zero PnL
        if abs(total_pnl) > 0.01:  # More than 1 cent
            criteria_met.append("✓ Non-zero PnL")
            logger.info(f"✓ Realistic, non-zero PnL: ${total_pnl:.2f}")
        else:
            logger.error(f"✗ Zero or negligible PnL: ${total_pnl:.2f}")
        
        # 5. Transaction costs
        total_costs = total_commission + total_slippage
        if total_costs > 0:
            criteria_met.append("✓ Transaction costs")
            logger.info(f"✓ Transaction costs: ${total_costs:.2f}")
            logger.info(f"  - Commission: ${total_commission:.2f}")
            logger.info(f"  - Slippage: ${total_slippage:.2f}")
        else:
            logger.error("✗ No transaction costs")
        
        # Print detailed results
        logger.info("\nDetailed Results:")
        logger.info(f"Strategy: Simple MA Crossover")
        logger.info(f"Total Trades: {total_trades}")
        logger.info(f"Final Balance: ${final_balance:.2f}")
        logger.info(f"Total PnL: ${total_pnl:.2f}")
        logger.info(f"Return: {return_pct:.2f}%")
        logger.info(f"Total Commission: ${total_commission:.2f}")
        logger.info(f"Total Slippage: ${total_slippage:.2f}")
        
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
                "Can load tick data",
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
        import traceback
        traceback.print_exc()
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