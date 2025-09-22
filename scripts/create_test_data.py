"""
Create synthetic test data for EMP Proving Ground.
This script generates realistic EURUSD tick data for testing when real data is unavailable.
"""

import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def generate_test_tick_data(
    symbol: str, num_days: int = 7, base_price: float = 1.1000
) -> pd.DataFrame:
    """
    Generate realistic synthetic tick data for testing.

    Args:
        symbol: Trading symbol (e.g., 'EURUSD')
        num_days: Number of days to generate
        base_price: Starting price level

    Returns:
        DataFrame with tick data
    """
    logger.info(f"Generating {num_days} days of synthetic {symbol} data starting at {base_price}")

    # Generate time series (1-minute intervals)
    start_time = datetime(2023, 1, 1, 0, 0, 0)
    end_time = start_time + timedelta(days=num_days)
    time_range = pd.date_range(start=start_time, end=end_time, freq="1min")

    # Set random seed for reproducible results
    np.random.seed(42)

    # Generate realistic price movements
    # EURUSD typically moves 0.5-1.5% per day
    daily_volatility = 0.01  # 1% daily volatility
    minute_volatility = daily_volatility / np.sqrt(1440)  # Convert to minute volatility

    # Generate price changes with some trend and mean reversion
    price_changes = np.random.normal(0, minute_volatility, len(time_range))

    # Add small trend component
    trend = np.linspace(0, 0.002, len(time_range))  # Small upward trend
    price_changes += trend

    # Add mean reversion
    for i in range(1, len(price_changes)):
        if abs(price_changes[i]) > minute_volatility * 2:
            price_changes[i] *= 0.5  # Reduce extreme moves

    # Calculate mid prices
    mid_prices = base_price + np.cumsum(price_changes)

    # Generate realistic bid/ask spreads
    # EURUSD spreads are typically 0.5-2 pips
    spreads = np.random.uniform(0.00005, 0.0002, len(time_range))  # 0.5-2 pips

    # Calculate bid and ask prices
    bids = mid_prices - spreads / 2
    asks = mid_prices + spreads / 2

    # Generate realistic volumes
    # Higher volumes during active sessions
    base_volume = 1000
    session_multiplier = np.ones(len(time_range))

    for i, timestamp in enumerate(time_range):
        hour = timestamp.hour
        # European session (8-16 UTC)
        if 8 <= hour <= 16:
            session_multiplier[i] = 2.0
        # US session (13-21 UTC)
        elif 13 <= hour <= 21:
            session_multiplier[i] = 1.8
        # Asian session (0-8 UTC)
        elif 0 <= hour <= 8:
            session_multiplier[i] = 1.2
        # Weekend (lower volumes)
        if timestamp.weekday() >= 5:
            session_multiplier[i] *= 0.3

    bid_volumes = (
        np.random.uniform(base_volume, base_volume * 10, len(time_range)) * session_multiplier
    )
    ask_volumes = (
        np.random.uniform(base_volume, base_volume * 10, len(time_range)) * session_multiplier
    )

    # Create DataFrame
    df = pd.DataFrame(
        {
            "timestamp": time_range,
            "symbol": symbol,
            "bid": bids,
            "ask": asks,
            "bid_volume": bid_volumes,
            "ask_volume": ask_volumes,
            "mid_price": mid_prices,
            "spread": spreads,
        }
    )

    logger.info(f"Generated {len(df)} synthetic ticks")
    logger.info(f"Price range: {df['bid'].min():.5f} - {df['ask'].max():.5f}")
    logger.info(f"Average spread: {df['spread'].mean():.5f}")

    return df


def save_test_data(symbol: str, df: pd.DataFrame, processed_dir: str):
    """
    Save test data in the proper directory structure.

    Args:
        symbol: Trading symbol
        df: DataFrame with tick data
        processed_dir: Directory to save processed data
    """
    # Create directory structure
    data_dir = Path(processed_dir)
    symbol_dir = data_dir / symbol
    symbol_dir.mkdir(parents=True, exist_ok=True)

    # Save as Parquet file
    filename = f"{symbol}_2023_01.parquet"
    filepath = symbol_dir / filename

    df.to_parquet(filepath, index=False)
    logger.info(f"Saved test data to {filepath}")

    # Verify file was created
    if filepath.exists():
        file_size = filepath.stat().st_size / (1024 * 1024)  # MB
        logger.info(f"File size: {file_size:.2f} MB")
        return True
    else:
        logger.error(f"Failed to save file to {filepath}")
        return False


def main():
    """Main function to create test data."""
    logger.info("=" * 60)
    logger.info("CREATING SYNTHETIC TEST DATA")
    logger.info("=" * 60)

    try:
        # Configuration
        symbol = "EURUSD"
        num_days = 7  # One week of data
        base_price = 1.1000
        processed_dir = "data/processed"

        # Generate synthetic data
        logger.info("Generating synthetic tick data...")
        df = generate_test_tick_data(symbol, num_days, base_price)

        # Save data
        logger.info("Saving test data...")
        success = save_test_data(symbol, df, processed_dir)

        if success:
            logger.info("=" * 60)
            logger.info("✅ TEST DATA CREATION SUCCESSFUL! ✅")
            logger.info("=" * 60)
            logger.info(f"Symbol: {symbol}")
            logger.info(f"Period: {num_days} days")
            logger.info(f"Records: {len(df)}")
            logger.info(f"Location: {processed_dir}/{symbol}/")
            logger.info("\nReady for Phase 1 testing!")
            return 0
        else:
            logger.error("❌ Failed to save test data")
            return 1

    except Exception as e:
        logger.error(f"Test data creation failed: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
