"""
Data Pipeline for the EMP Proving Ground system.

This module provides:
- TickDataStorage: Efficient storage and retrieval of tick data
- TickDataCleaner: Data cleaning and preprocessing
- DukascopyIngestor: Data ingestion from Dukascopy
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from src.data.real_data_ingestor import RealDataIngestor

logger = logging.getLogger(__name__)


class TickDataStorage:
    """
    Manages storage and retrieval of historical tick data.

    - Stores data in Parquet format for efficiency.
    - Implements caching for frequently accessed data.
    - Provides OHLCV aggregation on the fly.
    """

    def __init__(self, data_dir: str = "data"):
        """
        Initialize tick data storage

        Args:
            data_dir: Directory to store tick data
        """
        self.data_dir = Path(data_dir)
        self.processed_dir = self.data_dir / "processed"
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        # In-memory cache for tick data
        self.cache: Dict[str, pd.DataFrame] = {}
        self.cache_stats = {"hits": 0, "misses": 0, "cached_datasets": 0}

        logger.info(f"Initialized tick data storage: {data_dir}")

    def load_tick_data(
        self, symbol: str, start_time: datetime, end_time: datetime
    ) -> pd.DataFrame:
        """
        Load tick data for a given symbol and time range

        Args:
            symbol: Trading symbol
            start_time: Start time for data
            end_time: End time for data

        Returns:
            DataFrame with tick data
        """
        # Check cache first
        cache_key = f"{symbol}_{start_time.date()}_{end_time.date()}"
        if cache_key in self.cache:
            self.cache_stats["hits"] += 1
            df = self.cache[cache_key]
            # Filter by time range
            mask = (df["timestamp"] >= start_time) & (df["timestamp"] <= end_time)
            filtered_df = df[mask].copy()
            return filtered_df
        else:
            self.cache_stats["misses"] += 1

            # Load from processed files
            symbol_dir = self.processed_dir / symbol
            if not symbol_dir.exists():
                logger.warning(f"No data directory found for {symbol}")
                return pd.DataFrame()

            # Find relevant files
            data_files = []
            current_date = start_time.date()
            while current_date <= end_time.date():
                file_path = symbol_dir / f"{current_date}.parquet"
                if file_path.exists():
                    data_files.append(file_path)
                current_date += timedelta(days=1)

            if not data_files:
                logger.warning(f"No data files found for {symbol} in date range")
                return pd.DataFrame()

            # Load and combine data
            dfs = []
            for file_path in data_files:
                try:
                    df = pd.read_parquet(file_path)
                    dfs.append(df)
                except Exception as e:
                    logger.error(f"Error loading {file_path}: {e}")

            if not dfs:
                return pd.DataFrame()

            # Combine all data
            combined_df = pd.concat(dfs, ignore_index=True)
            combined_df = combined_df.sort_values("timestamp").reset_index(drop=True)

            # Filter by time range
            mask = (combined_df["timestamp"] >= start_time) & (
                combined_df["timestamp"] <= end_time
            )
            filtered_df = combined_df[mask].copy()

            # Cache the result
            if len(filtered_df) > 0:
                self._add_to_cache(cache_key, filtered_df)

            return filtered_df

    def get_ohlcv(
        self, symbol: str, start_time: datetime, end_time: datetime, freq: str = "M1"
    ) -> pd.DataFrame:
        """
        Get OHLCV data for a given symbol and time range

        Args:
            symbol: Trading symbol
            start_time: Start time for data
            end_time: End time for data
            freq: Frequency (e.g., 'M1', 'M5', 'H1', 'D1')

        Returns:
            DataFrame with OHLCV data
        """
        # Load tick data
        tick_data = self.load_tick_data(symbol, start_time, end_time)

        if tick_data.empty:
            return pd.DataFrame()

        # Convert to OHLCV
        ohlcv = self._ticks_to_ohlcv(tick_data, freq)

        return ohlcv

    def get_data_range(
        self, symbol: str, start_time: datetime, end_time: datetime
    ) -> pd.DataFrame:
        """
        Get data for a specific date range (used by evolution engine)

        Args:
            symbol: Trading symbol
            start_time: Start time
            end_time: End time

        Returns:
            DataFrame with OHLCV data
        """
        # Try to load real data first
        try:
            ingestor = RealDataIngestor()

            # Try to load existing data
            real_data = ingestor.load_symbol_data(symbol, start_time, end_time)

            if real_data is not None and not real_data.empty:
                logger.info(f"Loaded {len(real_data)} real data records for {symbol}")
                # Convert to OHLCV format if needed
                if "open" in real_data.columns and "high" in real_data.columns:
                    return real_data[["open", "high", "low", "close", "volume"]]
                else:
                    # Convert tick data to OHLCV
                    return self._ticks_to_ohlcv(real_data.reset_index(), "H")

            # If no real data available, try to download it
            logger.info(
                f"No existing real data found for {symbol}, attempting download..."
            )
            success = ingestor.download_symbol_data(
                symbol, start_time, end_time, "yahoo"
            )

            if success:
                real_data = ingestor.load_symbol_data(symbol, start_time, end_time)
                if real_data is not None and not real_data.empty:
                    logger.info(
                        f"Successfully downloaded and loaded {len(real_data)} real data records"
                    )
                    if "open" in real_data.columns and "high" in real_data.columns:
                        return real_data[["open", "high", "low", "close", "volume"]]
                    else:
                        return self._ticks_to_ohlcv(real_data.reset_index(), "H")

        except Exception as e:
            logger.warning(f"Error loading real data for {symbol}: {e}")

        # Fallback to existing OHLCV method
        logger.info(f"Using fallback data method for {symbol}")
        return self.get_ohlcv(symbol, start_time, end_time, freq="H")

    def _ticks_to_ohlcv(self, tick_data: pd.DataFrame, freq: str) -> pd.DataFrame:
        """
        Convert tick data to OHLCV format

        Args:
            tick_data: DataFrame with tick data
            freq: Frequency string

        Returns:
            DataFrame with OHLCV data
        """
        if tick_data.empty:
            return pd.DataFrame()

        # Ensure timestamp is datetime
        tick_data = tick_data.copy()
        tick_data["timestamp"] = pd.to_datetime(tick_data["timestamp"])

        # Set timestamp as index for resampling
        tick_data.set_index("timestamp", inplace=True)

        # Check which columns are available
        available_columns = tick_data.columns.tolist()

        # Define aggregation rules based on available columns
        agg_rules = {}

        if "bid" in available_columns:
            agg_rules["bid"] = "ohlc"
        if "ask" in available_columns:
            agg_rules["ask"] = "ohlc"
        if "bid_volume" in available_columns:
            agg_rules["bid_volume"] = "sum"
        if "ask_volume" in available_columns:
            agg_rules["ask_volume"] = "sum"
        if "volume" in available_columns:
            agg_rules["volume"] = "sum"

        # If no volume columns available, create synthetic volume
        if (
            "bid_volume" not in available_columns
            and "ask_volume" not in available_columns
            and "volume" not in available_columns
        ):
            tick_data["volume"] = np.random.randint(1000, 10000, len(tick_data))
            agg_rules["volume"] = "sum"

        # Resample to desired frequency
        ohlcv = tick_data.resample(freq).agg(agg_rules)

        # Flatten column names
        ohlcv.columns = ["_".join(col).strip() for col in ohlcv.columns]

        # Calculate OHLCV based on available columns
        if "bid_open" in ohlcv.columns and "ask_open" in ohlcv.columns:
            ohlcv["open"] = (ohlcv["bid_open"] + ohlcv["ask_open"]) / 2
            ohlcv["high"] = (ohlcv["bid_high"] + ohlcv["ask_high"]) / 2
            ohlcv["low"] = (ohlcv["bid_low"] + ohlcv["ask_low"]) / 2
            ohlcv["close"] = (ohlcv["bid_close"] + ohlcv["ask_close"]) / 2
        elif "bid_open" in ohlcv.columns:
            ohlcv["open"] = ohlcv["bid_open"]
            ohlcv["high"] = ohlcv["bid_high"]
            ohlcv["low"] = ohlcv["bid_low"]
            ohlcv["close"] = ohlcv["bid_close"]
        elif "ask_open" in ohlcv.columns:
            ohlcv["open"] = ohlcv["ask_open"]
            ohlcv["high"] = ohlcv["ask_high"]
            ohlcv["low"] = ohlcv["ask_low"]
            ohlcv["close"] = ohlcv["ask_close"]
        else:
            # Fallback to close price for all OHLC
            ohlcv["open"] = ohlcv["close"]
            ohlcv["high"] = ohlcv["close"]
            ohlcv["low"] = ohlcv["close"]

        # Calculate volume
        if "bid_volume" in ohlcv.columns and "ask_volume" in ohlcv.columns:
            ohlcv["volume"] = ohlcv["bid_volume"] + ohlcv["ask_volume"]
        elif "bid_volume" in ohlcv.columns:
            ohlcv["volume"] = ohlcv["bid_volume"]
        elif "ask_volume" in ohlcv.columns:
            ohlcv["volume"] = ohlcv["ask_volume"]
        elif "volume" in ohlcv.columns:
            ohlcv["volume"] = ohlcv["volume"]
        else:
            # Create synthetic volume
            ohlcv["volume"] = np.random.randint(1000, 10000, len(ohlcv))

        # Keep only OHLCV columns
        ohlcv = ohlcv[["open", "high", "low", "close", "volume"]]

        # Remove rows with all NaN values
        ohlcv = ohlcv.dropna()

        # Reset index and return as DataFrame
        result_df = ohlcv.reset_index()
        return result_df

    def _add_to_cache(self, key: str, df: pd.DataFrame):
        """Add DataFrame to cache"""
        # Limit cache size
        if len(self.cache) >= 50:
            # Remove oldest entry
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]

        self.cache[key] = df
        self.cache_stats["cached_datasets"] += 1

    def get_cache_stats(self) -> Dict:
        """Get cache statistics"""
        return self.cache_stats.copy()


class TickDataCleaner:
    """Cleans and preprocesses tick data"""

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize data cleaner

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.min_spread_bps = self.config.get("min_spread_bps", 0.1)
        self.max_spread_bps = self.config.get("max_spread_bps", 100.0)
        self.min_volume = self.config.get("min_volume", 0.0)
        self.max_price_change_pct = self.config.get("max_price_change_pct", 10.0)

    def clean(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Clean tick data

        Args:
            df: Raw tick data DataFrame
            symbol: Trading symbol

        Returns:
            Cleaned DataFrame
        """
        if df.empty:
            return df

        df = df.copy()

        # Remove duplicates
        df = df.drop_duplicates(subset=["timestamp"])

        # Sort by timestamp
        df = df.sort_values("timestamp").reset_index(drop=True)

        # Calculate spread
        df["spread"] = df["ask"] - df["bid"]
        df["spread_bps"] = (df["spread"] / df["bid"]) * 10000

        # Filter by spread
        df = df[
            (df["spread_bps"] >= self.min_spread_bps)
            & (df["spread_bps"] <= self.max_spread_bps)
        ]

        # Filter by volume
        if "bid_volume" in df.columns and "ask_volume" in df.columns:
            df["total_volume"] = df["bid_volume"] + df["ask_volume"]
            df = df[df["total_volume"] >= self.min_volume]

        # Remove extreme price changes
        df["price_change_pct"] = df["bid"].pct_change().abs() * 100
        df = df[df["price_change_pct"] <= self.max_price_change_pct]

        # Remove rows with invalid prices
        df = df[(df["bid"] > 0) & (df["ask"] > 0) & (df["bid"] < df["ask"])]

        # Reset index
        df = df.reset_index(drop=True)

        logger.info(f"Cleaned {symbol} data: {len(df)} rows remaining")

        return df


class DukascopyIngestor:
    """Ingests data from Dukascopy or generates synthetic data"""

    def __init__(self, storage: TickDataStorage, cleaner: TickDataCleaner):
        """
        Initialize data ingestor

        Args:
            storage: TickDataStorage instance
            cleaner: TickDataCleaner instance
        """
        self.storage = storage
        self.cleaner = cleaner

    def ingest_year(self, symbol: str, year: int) -> bool:
        """
        Ingest data for a specific year

        Args:
            symbol: Trading symbol
            year: Year to ingest

        Returns:
            True if successful, False otherwise
        """
        try:
            start_time = datetime(year, 1, 1)
            end_time = datetime(year, 12, 31, 23, 59, 59)

            # Try to download real data first
            data = self._download_real_data(symbol, start_time, end_time)

            if data is None or data.empty:
                # Generate synthetic data as fallback
                logger.info(f"Generating synthetic data for {symbol} {year}")
                data = self._generate_fallback_data(symbol, start_time, end_time)

            if data.empty:
                logger.error(f"Failed to get data for {symbol} {year}")
                return False

            # Clean the data
            cleaned_data = self.cleaner.clean(data, symbol)

            if cleaned_data.empty:
                logger.error(f"No data remaining after cleaning for {symbol} {year}")
                return False

            # Save to storage
            self._save_data(symbol, cleaned_data, year)

            logger.info(
                f"Successfully ingested {len(cleaned_data)} ticks for {symbol} {year}"
            )
            return True

        except Exception as e:
            logger.error(f"Error ingesting data for {symbol} {year}: {e}")
            return False

    def _download_real_data(
        self, symbol: str, start_time: datetime, end_time: datetime
    ) -> Optional[pd.DataFrame]:
        """
        Download real data from multiple sources.

        Args:
            symbol: Trading symbol
            start_time: Start time
            end_time: End time

        Returns:
            DataFrame with tick data or None if failed
        """
        logger.info(
            f"Attempting to download real data for {symbol} from {start_time} to {end_time}"
        )

        # Try multiple data sources in order of preference
        sources = [
            ("dukascopy", self._download_from_dukascopy),
            ("yahoo", self._download_from_yahoo),
            ("alpha_vantage", self._download_from_alpha_vantage),
        ]

        for source_name, download_func in sources:
            try:
                logger.info(f"Trying {source_name} for {symbol}")
                data = download_func(symbol, start_time, end_time)

                if data is not None and not data.empty:
                    logger.info(
                        f"Successfully downloaded {len(data)} records from {source_name}"
                    )
                    return data
                else:
                    logger.warning(f"No data available from {source_name}")

            except Exception as e:
                logger.error(f"Error downloading from {source_name}: {e}")
                continue

        logger.warning(f"All data sources failed for {symbol}")
        return None

    def _download_from_dukascopy(
        self, symbol: str, start_time: datetime, end_time: datetime
    ) -> Optional[pd.DataFrame]:
        """Download data from Dukascopy."""
        try:
            # Use the class directly since it's in the same module
            ingestor = DukascopyIngestor(self.storage, self.cleaner)
            data = ingestor._download_real_data(symbol, start_time, end_time)

            if data is not None and not data.empty:
                # Convert to expected format
                if "bid_volume" not in data.columns:
                    data["bid_volume"] = 1000  # Default volume
                if "ask_volume" not in data.columns:
                    data["ask_volume"] = 1000  # Default volume

                return data

        except Exception as e:
            logger.error(f"Error downloading from Dukascopy: {e}")

        return None

    def _download_from_yahoo(
        self, symbol: str, start_time: datetime, end_time: datetime
    ) -> Optional[pd.DataFrame]:
        """Download data from Yahoo Finance."""
        try:
            # Use the class directly since it's in the same module
            ingestor = RealDataIngestor()
            data = ingestor.download_yahoo_data(symbol, start_time, end_time)

            if data is not None and not data.empty:
                # Convert OHLCV to tick-like format
                tick_data = self._ohlcv_to_ticks(data)
                return tick_data

        except Exception as e:
            logger.error(f"Error downloading from Yahoo: {e}")

        return None

    def _download_from_alpha_vantage(
        self, symbol: str, start_time: datetime, end_time: datetime
    ) -> Optional[pd.DataFrame]:
        """Download data from Alpha Vantage."""
        try:
            # Use the class directly since it's in the same module
            ingestor = RealDataIngestor()
            data = ingestor.download_alpha_vantage_data(symbol, start_time, end_time)

            if data is not None and not data.empty:
                # Convert OHLCV to tick-like format
                tick_data = self._ohlcv_to_ticks(data)
                return tick_data

        except Exception as e:
            logger.error(f"Error downloading from Alpha Vantage: {e}")

        return None

    def _ohlcv_to_ticks(self, ohlcv_data: pd.DataFrame) -> pd.DataFrame:
        """Convert OHLCV data to tick-like format."""
        try:
            # Create tick data from OHLCV
            ticks = []

            for timestamp, row in ohlcv_data.iterrows():
                # Create multiple ticks per OHLCV bar for more realistic data
                base_time = timestamp

                # Create ticks at different times within the bar
                tick_times = [
                    base_time,
                    base_time + timedelta(minutes=15),
                    base_time + timedelta(minutes=30),
                    base_time + timedelta(minutes=45),
                ]

                for tick_time in tick_times:
                    # Interpolate price between open and close
                    # Progress through hour
                    progress = (tick_time - base_time).total_seconds() / 3600
                    price = row["open"] + (row["close"] - row["open"]) * progress

                    # Add some spread
                    spread = 0.0001  # 1 pip
                    bid = price - spread / 2
                    ask = price + spread / 2

                    ticks.append(
                        {
                            "timestamp": tick_time,
                            "bid": bid,
                            "ask": ask,
                            # Distribute volume
                            "bid_volume": row.get("volume", 1000) / 4,
                            "ask_volume": row.get("volume", 1000) / 4,
                        }
                    )

            return pd.DataFrame(ticks)

        except Exception as e:
            logger.error(f"Error converting OHLCV to ticks: {e}")
            return pd.DataFrame()

    def _generate_fallback_data(
        self, symbol: str, start_time: datetime, end_time: datetime
    ) -> pd.DataFrame:
        """
        Generate synthetic tick data as fallback

        Args:
            symbol: Trading symbol
            start_time: Start time
            end_time: End time

        Returns:
            DataFrame with synthetic tick data
        """
        # Generate timestamps
        timestamps = pd.date_range(start=start_time, end=end_time, freq="1s")

        # Base price (typical for forex pairs)
        base_prices = {
            "EURUSD": 1.1000,
            "GBPUSD": 1.2500,
            "USDJPY": 110.00,
            "USDCHF": 0.9000,
            "AUDUSD": 0.7500,
            "USDCAD": 1.3500,
        }

        base_price = base_prices.get(symbol, 1.0000)

        # Generate price movements
        np.random.seed(hash(symbol + str(start_time.date())) % 2**32)

        # Random walk with mean reversion
        returns = np.random.normal(0, 0.0001, len(timestamps))  # 1 pip volatility
        prices = base_price * np.exp(np.cumsum(returns))

        # Add some realistic features
        # Intraday volatility
        hour_volatility = np.sin(2 * np.pi * timestamps.hour / 24) * 0.0002
        prices *= np.exp(hour_volatility)

        # Weekend gaps
        weekend_mask = timestamps.weekday >= 5
        weekend_gaps = np.random.normal(0, 0.001, len(timestamps))
        prices[weekend_mask] *= np.exp(weekend_gaps[weekend_mask])

        # Generate bid/ask spreads
        spreads = np.random.uniform(0.0001, 0.0003, len(timestamps))  # 1-3 pips

        # Create DataFrame
        data = pd.DataFrame(
            {
                "timestamp": timestamps,
                "bid": prices - spreads / 2,
                "ask": prices + spreads / 2,
                "bid_volume": np.random.uniform(1000, 10000, len(timestamps)),
                "ask_volume": np.random.uniform(1000, 10000, len(timestamps)),
            }
        )

        return data

    def _save_data(self, symbol: str, data: pd.DataFrame, year: int):
        """
        Save data to storage

        Args:
            symbol: Trading symbol
            data: DataFrame to save
            year: Year for organization
        """
        # Create symbol directory
        symbol_dir = self.storage.processed_dir / symbol
        symbol_dir.mkdir(parents=True, exist_ok=True)

        # Group by date and save
        data["date"] = data["timestamp"].dt.date

        for date, group in data.groupby("date"):
            file_path = symbol_dir / f"{date}.parquet"
            group = group.drop("date", axis=1)
            group.to_parquet(file_path, index=False)

        logger.info(f"Saved {len(data)} ticks for {symbol} {year}")
