#!/usr/bin/env python3
"""
Data Pipeline for MLOps - Epic 1: The MLOps Foundation
Transforms raw historical data into a clean, feature-rich dataset for model training.

This script:
1. Loads historical EURUSD 1h data
2. Calculates technical indicators as features
3. Creates classification target variable
4. Saves processed dataset for training
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataPipeline:
    """Handles the complete data pipeline from raw data to training dataset."""
    
    def __init__(self, data_dir: str = "data", symbol: str = "EURUSD"):
        self.data_dir = Path(data_dir)
        self.symbol = symbol
        self.raw_data_path = self.data_dir / "processed" / symbol
        self.output_path = self.data_dir / "training" / "v1_training_dataset.parquet"
        
    def load_historical_data(self) -> pd.DataFrame:
        """Load and concatenate all historical parquet files."""
        logger.info(f"Loading historical data from {self.raw_data_path}")
        
        parquet_files = list(self.raw_data_path.glob("*.parquet"))
        if not parquet_files:
            raise FileNotFoundError(f"No parquet files found in {self.raw_data_path}")
        
        logger.info(f"Found {len(parquet_files)} parquet files")
        
        # Load and concatenate all files
        dfs = []
        for file in sorted(parquet_files):
            df = pd.read_parquet(file)
            dfs.append(df)
        
        data = pd.concat(dfs, ignore_index=True)
        logger.info(f"Loaded {len(data)} total rows")
        
        # Ensure datetime index
        if 'timestamp' in data.columns:
            data['timestamp'] = pd.to_datetime(data['timestamp'])
            data = data.set_index('timestamp')
        
        return data
    
    def resample_to_ohlc(self, df: pd.DataFrame) -> pd.DataFrame:
        """Resample tick data to 1-hour OHLC data."""
        logger.info("Resampling tick data to 1-hour OHLC...")
        
        # Create OHLC data from bid/ask
        # Use bid as base price, calculate mid-price
        df['mid_price'] = (df['bid'] + df['ask']) / 2
        
        # Resample to 1-hour bars
        ohlc = df['mid_price'].resample('1H').ohlc()
        volume = df['total_volume'].resample('1H').sum()
        
        # Combine OHLC and volume
        ohlc_df = pd.concat([ohlc, volume.rename('volume')], axis=1)
        
        # Drop NaN values
        ohlc_df = ohlc_df.dropna()
        
        logger.info(f"Resampled to {len(ohlc_df)} hourly bars")
        
        return ohlc_df
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators to use as features."""
        logger.info("Calculating technical indicators...")
        
        features_df = df.copy()
        
        # Price-based features
        features_df['returns'] = features_df['close'].pct_change()
        features_df['log_returns'] = np.log(features_df['close'] / features_df['close'].shift(1))
        
        # Moving averages
        for period in [5, 10, 20, 50]:
            features_df[f'sma_{period}'] = features_df['close'].rolling(window=period).mean()
            features_df[f'ema_{period}'] = features_df['close'].ewm(span=period, adjust=False).mean()
        
        # RSI calculation
        delta = features_df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        features_df['rsi_14'] = 100 - (100 / (1 + rs))
        
        # MACD calculation
        ema_12 = features_df['close'].ewm(span=12, adjust=False).mean()
        ema_26 = features_df['close'].ewm(span=26, adjust=False).mean()
        features_df['macd'] = ema_12 - ema_26
        features_df['macd_signal'] = features_df['macd'].ewm(span=9, adjust=False).mean()
        features_df['macd_histogram'] = features_df['macd'] - features_df['macd_signal']
        
        # Bollinger Bands
        sma_20 = features_df['close'].rolling(window=20).mean()
        std_20 = features_df['close'].rolling(window=20).std()
        features_df['bb_upper'] = sma_20 + (std_20 * 2)
        features_df['bb_lower'] = sma_20 - (std_20 * 2)
        features_df['bb_middle'] = sma_20
        features_df['bb_width'] = features_df['bb_upper'] - features_df['bb_lower']
        features_df['bb_position'] = (features_df['close'] - features_df['bb_lower']) / features_df['bb_width']
        
        # ATR (Average True Range)
        high_low = features_df['high'] - features_df['low']
        high_close = np.abs(features_df['high'] - features_df['close'].shift(1))
        low_close = np.abs(features_df['low'] - features_df['close'].shift(1))
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        features_df['atr_14'] = true_range.rolling(window=14).mean()
        
        # Volume indicators
        features_df['volume_sma_20'] = features_df['volume'].rolling(window=20).mean()
        features_df['volume_ratio'] = features_df['volume'] / features_df['volume_sma_20']
        
        # Price position indicators
        features_df['price_sma_ratio'] = features_df['close'] / features_df['sma_20']
        features_df['price_ema_ratio'] = features_df['close'] / features_df['ema_20']
        
        # Volatility
        features_df['volatility_20'] = features_df['returns'].rolling(window=20).std()
        
        logger.info(f"Calculated {len(features_df.columns)} features")
        return features_df
    
    def create_target_variable(self, df: pd.DataFrame, 
                             threshold_pips: float = 10.0, 
                             forecast_horizon: int = 4) -> pd.DataFrame:
        """Create classification target variable."""
        logger.info("Creating target variable...")
        
        target_df = df.copy()
        
        # Calculate future price change
        future_price = target_df['close'].shift(-forecast_horizon)
        price_change = future_price - target_df['close']
        
        # Convert pips to decimal (1 pip = 0.0001 for EURUSD)
        threshold = threshold_pips * 0.0001
        
        # Create target classes
        conditions = [
            price_change > threshold,
            price_change < -threshold,
            (price_change >= -threshold) & (price_change <= threshold)
        ]
        
        choices = ['UP', 'DOWN', 'FLAT']
        
        target_df['target'] = np.select(conditions, choices, default='FLAT')
        
        # Calculate target probabilities for debugging
        target_counts = target_df['target'].value_counts()
        logger.info(f"Target distribution: {target_counts.to_dict()}")
        
        return target_df
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean the dataset by removing NaN values and outliers."""
        logger.info("Cleaning data...")
        
        # Remove rows with NaN values
        cleaned_df = df.dropna()
        
        # Remove extreme outliers (winsorize at 5th and 95th percentiles)
        for col in cleaned_df.select_dtypes(include=[np.number]).columns:
            if col != 'target':
                q_low = cleaned_df[col].quantile(0.05)
                q_high = cleaned_df[col].quantile(0.95)
                cleaned_df = cleaned_df[(cleaned_df[col] >= q_low) & (cleaned_df[col] <= q_high)]
        
        logger.info(f"Cleaned dataset: {len(cleaned_df)} rows remaining")
        return cleaned_df
    
    def run_pipeline(self, threshold_pips: float = 10.0, forecast_horizon: int = 4) -> None:
        """Run the complete data pipeline."""
        logger.info("Starting data pipeline...")
        
        # Load data
        raw_data = self.load_historical_data()
        
        # Resample to OHLC
        ohlc_data = self.resample_to_ohlc(raw_data)
        
        # Calculate features
        features_data = self.calculate_technical_indicators(ohlc_data)
        
        # Create target
        labeled_data = self.create_target_variable(features_data, threshold_pips, forecast_horizon)
        
        # Clean data
        clean_data = self.clean_data(labeled_data)
        
        # Save to parquet
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        clean_data.to_parquet(self.output_path)
        logger.info(f"Saved training dataset to {self.output_path}")
        
        # Print summary statistics
        print("\n=== Dataset Summary ===")
        print(f"Total samples: {len(clean_data)}")
        print(f"Features: {len(clean_data.columns) - 1}")  # Exclude target
        print(f"Target distribution:")
        print(clean_data['target'].value_counts())
        print(f"\nDataset saved to: {self.output_path}")


def main():
    """Main function to run the data pipeline."""
    parser = argparse.ArgumentParser(description='Create training dataset from historical data')
    parser.add_argument('--symbol', default='EURUSD', help='Symbol to process')
    parser.add_argument('--threshold-pips', type=float, default=10.0, help='Threshold in pips for target creation')
    parser.add_argument('--forecast-horizon', type=int, default=4, help='Number of bars to forecast ahead')
    
    args = parser.parse_args()
    
    pipeline = DataPipeline(symbol=args.symbol)
    pipeline.run_pipeline(
        threshold_pips=args.threshold_pips,
        forecast_horizon=args.forecast_horizon
    )


if __name__ == "__main__":
    main()
