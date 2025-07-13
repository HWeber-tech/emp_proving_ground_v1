"""
TickDataCleaner: Cleans and validates raw tick data with instrument-specific logic.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class TickDataCleaner:
    """
    Cleans and validates raw tick data with robust, instrument-specific logic.
    """
    
    def __init__(self, symbol: str):
        self.symbol = symbol
        
        # Instrument-specific cleaning parameters
        self.cleaning_params = self._get_cleaning_params(symbol)
        
    def _get_cleaning_params(self, symbol: str) -> Dict[str, Any]:
        """
        Get cleaning parameters specific to the instrument.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Dictionary of cleaning parameters
        """
        # Base parameters for all instruments
        base_params = {
            'min_spread_bps': 0.1,      # Minimum spread in basis points
            'max_spread_bps': 100.0,    # Maximum spread in basis points
            'min_price': 0.1,           # Minimum valid price
            'max_price': 1000000.0,     # Maximum valid price
            'min_volume': 0.0,          # Minimum volume
            'max_volume': 1000000000.0, # Maximum volume
            'outlier_threshold': 5.0,   # Standard deviations for outlier detection
            'min_tick_interval_ms': 1,  # Minimum time between ticks (ms)
            'max_price_change_pct': 10.0, # Maximum price change between ticks (%)
        }
        
        # Instrument-specific overrides
        instrument_params = {
            'EURUSD': {
                'min_price': 0.5,
                'max_price': 2.0,
                'min_spread_bps': 0.1,
                'max_spread_bps': 50.0,
            },
            'GBPUSD': {
                'min_price': 0.5,
                'max_price': 3.0,
                'min_spread_bps': 0.1,
                'max_spread_bps': 50.0,
            },
            'USDJPY': {
                'min_price': 50.0,
                'max_price': 200.0,
                'min_spread_bps': 0.1,
                'max_spread_bps': 50.0,
            },
            'USDCHF': {
                'min_price': 0.5,
                'max_price': 2.0,
                'min_spread_bps': 0.1,
                'max_spread_bps': 50.0,
            },
            'AUDUSD': {
                'min_price': 0.5,
                'max_price': 2.0,
                'min_spread_bps': 0.1,
                'max_spread_bps': 50.0,
            },
            'USDCAD': {
                'min_price': 0.5,
                'max_price': 2.0,
                'min_spread_bps': 0.1,
                'max_spread_bps': 50.0,
            },
        }
        
        # Merge base params with instrument-specific params
        params = base_params.copy()
        if symbol in instrument_params:
            params.update(instrument_params[symbol])
        
        return params

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and validate tick data for the given symbol.
        
        Args:
            df: Raw tick data DataFrame with columns: timestamp, bid, ask, bid_volume, ask_volume
            
        Returns:
            Cleaned DataFrame
        """
        if df.empty:
            logger.warning("Empty DataFrame provided for cleaning")
            return df
        
        logger.info(f"Cleaning {len(df)} ticks for {self.symbol}")
        
        # Make a copy to avoid modifying original
        cleaned_df = df.copy()
        
        # Step 1: Basic data type validation and conversion
        cleaned_df = self._validate_data_types(cleaned_df)
        
        # Step 2: Remove duplicate timestamps
        cleaned_df = self._remove_duplicates(cleaned_df)
        
        # Step 3: Sort by timestamp
        cleaned_df = cleaned_df.sort_values('timestamp').reset_index(drop=True)
        
        # Step 4: Validate price ranges
        cleaned_df = self._validate_price_ranges(cleaned_df)
        
        # Step 5: Validate spread ranges
        cleaned_df = self._validate_spread_ranges(cleaned_df)
        
        # Step 6: Validate volume ranges
        cleaned_df = self._validate_volume_ranges(cleaned_df)
        
        # Step 7: Detect and handle outliers
        cleaned_df = self._handle_outliers(cleaned_df)
        
        # Step 8: Validate tick intervals
        cleaned_df = self._validate_tick_intervals(cleaned_df)
        
        # Step 9: Validate price changes
        cleaned_df = self._validate_price_changes(cleaned_df)
        
        # Step 10: Final validation and statistics
        cleaned_df = self._final_validation(cleaned_df)
        
        logger.info(f"Cleaning complete. Kept {len(cleaned_df)} out of {len(df)} ticks")
        
        return cleaned_df
    
    def _validate_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and convert data types."""
        try:
            # Ensure timestamp is datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Ensure numeric columns are float
            numeric_cols = ['bid', 'ask', 'bid_volume', 'ask_volume']
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Remove rows with invalid numeric data
            df = df.dropna(subset=numeric_cols)
            
            return df
            
        except Exception as e:
            logger.error(f"Data type validation failed: {e}")
            raise
    
    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate timestamps, keeping the first occurrence."""
        initial_count = len(df)
        df = df.drop_duplicates(subset=['timestamp'], keep='first')
        removed_count = initial_count - len(df)
        
        if removed_count > 0:
            logger.info(f"Removed {removed_count} duplicate timestamps")
        
        return df
    
    def _validate_price_ranges(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate that prices are within acceptable ranges."""
        min_price = self.cleaning_params['min_price']
        max_price = self.cleaning_params['max_price']
        
        # Check bid and ask prices
        valid_bid = (df['bid'] >= min_price) & (df['bid'] <= max_price)
        valid_ask = (df['ask'] >= min_price) & (df['ask'] <= max_price)
        
        # Keep only rows where both bid and ask are valid
        valid_prices = valid_bid & valid_ask
        df = df[valid_prices].copy()
        
        removed_count = len(valid_prices) - valid_prices.sum()
        if removed_count > 0:
            logger.info(f"Removed {removed_count} ticks with invalid prices")
        
        return df
    
    def _validate_spread_ranges(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate that spreads are within acceptable ranges."""
        min_spread_bps = self.cleaning_params['min_spread_bps']
        max_spread_bps = self.cleaning_params['max_spread_bps']
        
        # Calculate spread in basis points
        df['spread_bps'] = (df['ask'] - df['bid']) / df['bid'] * 10000
        
        # Validate spread range
        valid_spread = (df['spread_bps'] >= min_spread_bps) & (df['spread_bps'] <= max_spread_bps)
        df = df[valid_spread].copy()
        
        removed_count = len(valid_spread) - valid_spread.sum()
        if removed_count > 0:
            logger.info(f"Removed {removed_count} ticks with invalid spreads")
        
        # Drop the temporary spread_bps column
        df = df.drop(columns=['spread_bps'])
        
        return df
    
    def _validate_volume_ranges(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate that volumes are within acceptable ranges."""
        min_volume = self.cleaning_params['min_volume']
        max_volume = self.cleaning_params['max_volume']
        
        # Check bid and ask volumes
        valid_bid_vol = (df['bid_volume'] >= min_volume) & (df['bid_volume'] <= max_volume)
        valid_ask_vol = (df['ask_volume'] >= min_volume) & (df['ask_volume'] <= max_volume)
        
        # Keep only rows where both volumes are valid
        valid_volumes = valid_bid_vol & valid_ask_vol
        df = df[valid_volumes].copy()
        
        removed_count = len(valid_volumes) - valid_volumes.sum()
        if removed_count > 0:
            logger.info(f"Removed {removed_count} ticks with invalid volumes")
        
        return df
    
    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect and handle outliers using statistical methods."""
        threshold = self.cleaning_params['outlier_threshold']
        
        # Calculate rolling statistics for outlier detection
        window_size = min(1000, len(df) // 10)  # Adaptive window size
        if window_size < 10:
            return df  # Not enough data for outlier detection
        
        # Calculate rolling median and MAD for robust outlier detection
        df = df.copy()
        df['bid_median'] = df['bid'].rolling(window=window_size, center=True).median()
        df['ask_median'] = df['ask'].rolling(window=window_size, center=True).median()
        
        # Calculate Median Absolute Deviation (MAD)
        df['bid_mad'] = df['bid'].rolling(window=window_size, center=True).apply(
            lambda x: np.median(np.abs(x - np.median(x)))
        )
        df['ask_mad'] = df['ask'].rolling(window=window_size, center=True).apply(
            lambda x: np.median(np.abs(x - np.median(x)))
        )
        
        # Identify outliers (prices more than threshold * MAD from median)
        bid_outliers = np.abs(df['bid'] - df['bid_median']) > threshold * df['bid_mad']
        ask_outliers = np.abs(df['ask'] - df['ask_median']) > threshold * df['ask_mad']
        
        # Remove outliers
        outliers = bid_outliers | ask_outliers
        df = df[~outliers].copy()
        
        # Clean up temporary columns
        df = df.drop(columns=['bid_median', 'ask_median', 'bid_mad', 'ask_mad'])
        
        removed_count = outliers.sum()
        if removed_count > 0:
            logger.info(f"Removed {removed_count} outlier ticks")
        
        return df
    
    def _validate_tick_intervals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate minimum time intervals between ticks."""
        min_interval_ms = self.cleaning_params['min_tick_interval_ms']
        
        # Calculate time differences
        df['time_diff_ms'] = df['timestamp'].diff().dt.total_seconds() * 1000
        
        # Remove ticks that are too close together (except first tick)
        valid_intervals = (df['time_diff_ms'] >= min_interval_ms) | df['time_diff_ms'].isna()
        df = df[valid_intervals].copy()
        
        # Clean up temporary column
        df = df.drop(columns=['time_diff_ms'])
        
        removed_count = len(valid_intervals) - valid_intervals.sum()
        if removed_count > 0:
            logger.info(f"Removed {removed_count} ticks with invalid intervals")
        
        return df
    
    def _validate_price_changes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate maximum price changes between consecutive ticks."""
        max_change_pct = self.cleaning_params['max_price_change_pct']
        
        # Calculate percentage changes
        df['bid_change_pct'] = df['bid'].pct_change().abs() * 100
        df['ask_change_pct'] = df['ask'].pct_change().abs() * 100
        
        # Remove ticks with excessive price changes
        valid_bid_change = (df['bid_change_pct'] <= max_change_pct) | df['bid_change_pct'].isna()
        valid_ask_change = (df['ask_change_pct'] <= max_change_pct) | df['ask_change_pct'].isna()
        
        valid_changes = valid_bid_change & valid_ask_change
        df = df[valid_changes].copy()
        
        # Clean up temporary columns
        df = df.drop(columns=['bid_change_pct', 'ask_change_pct'])
        
        removed_count = len(valid_changes) - valid_changes.sum()
        if removed_count > 0:
            logger.info(f"Removed {removed_count} ticks with excessive price changes")
        
        return df
    
    def _final_validation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Final validation and statistics."""
        if df.empty:
            logger.warning("No valid ticks remaining after cleaning")
            return df
        
        # Ensure bid <= ask
        valid_order = df['bid'] <= df['ask']
        df = df[valid_order].copy()
        
        removed_count = len(valid_order) - valid_order.sum()
        if removed_count > 0:
            logger.info(f"Removed {removed_count} ticks with invalid bid/ask order")
        
        # Add final calculated columns
        df['mid_price'] = (df['ask'] + df['bid']) / 2
        df['spread'] = df['ask'] - df['bid']
        
        # Log final statistics
        logger.info(f"Final cleaned data: {len(df)} ticks")
        logger.info(f"Price range: {df['bid'].min():.5f} - {df['ask'].max():.5f}")
        logger.info(f"Spread range: {df['spread'].min():.5f} - {df['spread'].max():.5f}")
        
        return df 