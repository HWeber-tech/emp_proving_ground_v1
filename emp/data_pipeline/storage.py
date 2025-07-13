"""
TickDataStorage: Memory-aware LRU caching and partitioned Parquet storage for tick data.
"""
import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple
import logging
from collections import OrderedDict
import gc

logger = logging.getLogger(__name__)

class LRUCache:
    """
    Simple LRU cache implementation for tick data.
    """
    
    def __init__(self, max_size: int = 10):
        self.max_size = max_size
        self.cache = OrderedDict()
        self.memory_usage = 0
        self.max_memory_mb = 1024  # 1GB max memory usage
        
    def get(self, key: str) -> Optional[pd.DataFrame]:
        """Get item from cache, moving it to the end (most recently used)."""
        if key in self.cache:
            # Move to end (most recently used)
            self.cache.move_to_end(key)
            return self.cache[key]
        return None
    
    def put(self, key: str, value: pd.DataFrame):
        """Add item to cache, evicting least recently used if necessary."""
        # Estimate memory usage (rough approximation)
        estimated_memory_mb = len(value) * len(value.columns) * 8 / (1024 * 1024)  # 8 bytes per float
        
        # If item is already in cache, remove it first
        if key in self.cache:
            old_value = self.cache[key]
            old_memory_mb = len(old_value) * len(old_value.columns) * 8 / (1024 * 1024)
            self.memory_usage -= old_memory_mb
            del self.cache[key]
        
        # Check if adding this item would exceed memory limit
        while (self.memory_usage + estimated_memory_mb > self.max_memory_mb or 
               len(self.cache) >= self.max_size):
            if not self.cache:
                logger.warning("Cache is empty but still can't fit new item")
                return
            
            # Remove least recently used item
            lru_key, lru_value = self.cache.popitem(last=False)
            lru_memory_mb = len(lru_value) * len(lru_value.columns) * 8 / (1024 * 1024)
            self.memory_usage -= lru_memory_mb
            logger.debug(f"Evicted {lru_key} from cache")
        
        # Add new item
        self.cache[key] = value
        self.memory_usage += estimated_memory_mb
        logger.debug(f"Added {key} to cache. Memory usage: {self.memory_usage:.2f}MB")
    
    def clear(self):
        """Clear all items from cache."""
        self.cache.clear()
        self.memory_usage = 0
        gc.collect()  # Force garbage collection
    
    def get_stats(self) -> Dict:
        """Get cache statistics."""
        return {
            'size': len(self.cache),
            'max_size': self.max_size,
            'memory_usage_mb': self.memory_usage,
            'max_memory_mb': self.max_memory_mb,
            'keys': list(self.cache.keys())
        }

class TickDataStorage:
    """
    Handles memory-aware LRU caching and partitioned Parquet storage for tick data.
    """
    
    def __init__(self, processed_dir: str, cache_size: int = 10):
        self.processed_dir = processed_dir
        self.cache = LRUCache(max_size=cache_size)
        
        # Create directory if it doesn't exist
        os.makedirs(processed_dir, exist_ok=True)
        
        logger.info(f"Initialized TickDataStorage with cache size {cache_size}")
    
    def load_tick_data(self, symbol: str, start_time: datetime, end_time: datetime) -> pd.DataFrame:
        """
        Load tick data for a given symbol and time range from Parquet storage.
        
        Args:
            symbol: Trading symbol
            start_time: Start time for data
            end_time: End time for data
            
        Returns:
            DataFrame with tick data
        """
        logger.info(f"Loading tick data for {symbol} from {start_time} to {end_time}")
        
        # Generate cache key
        cache_key = f"{symbol}_{start_time.strftime('%Y%m%d_%H%M%S')}_{end_time.strftime('%Y%m%d_%H%M%S')}"
        
        # Check cache first
        cached_data = self.cache.get(cache_key)
        if cached_data is not None:
            logger.info(f"Retrieved {len(cached_data)} ticks from cache")
            return cached_data
        
        # Load from Parquet files
        df = self._load_from_parquet(symbol, start_time, end_time)
        
        if not df.empty:
            # Add to cache
            self.cache.put(cache_key, df)
            logger.info(f"Loaded {len(df)} ticks from storage and cached")
        else:
            logger.warning(f"No tick data found for {symbol} in specified time range")
        
        return df
    
    def _load_from_parquet(self, symbol: str, start_time: datetime, end_time: datetime) -> pd.DataFrame:
        """
        Load data from partitioned Parquet files.
        
        Args:
            symbol: Trading symbol
            start_time: Start time
            end_time: End time
            
        Returns:
            Combined DataFrame
        """
        symbol_dir = os.path.join(self.processed_dir, symbol)
        if not os.path.exists(symbol_dir):
            logger.warning(f"No data directory found for {symbol}")
            return pd.DataFrame()
        
        # Find all relevant Parquet files
        parquet_files = []
        for root, dirs, files in os.walk(symbol_dir):
            for file in files:
                if file.endswith('.parquet'):
                    file_path = os.path.join(root, file)
                    # Check if file might contain data in our time range
                    if self._file_might_contain_data(file_path, start_time, end_time):
                        parquet_files.append(file_path)
        
        if not parquet_files:
            logger.warning(f"No Parquet files found for {symbol} in time range")
            return pd.DataFrame()
        
        # Load and combine data from all relevant files
        dfs = []
        for file_path in parquet_files:
            try:
                df = pd.read_parquet(file_path)
                if not df.empty:
                    # Filter by timestamp
                    mask = (df['timestamp'] >= start_time) & (df['timestamp'] <= end_time)
                    filtered_df = df[mask]
                    if not filtered_df.empty:
                        dfs.append(filtered_df)
            except Exception as e:
                logger.error(f"Failed to load {file_path}: {e}")
                continue
        
        if not dfs:
            logger.warning("No data found in specified time range")
            return pd.DataFrame()
        
        # Combine all DataFrames
        combined_df = pd.concat(dfs, ignore_index=True)
        combined_df = combined_df.sort_values('timestamp').reset_index(drop=True)
        
        logger.info(f"Loaded {len(combined_df)} ticks from {len(parquet_files)} files")
        return combined_df
    
    def _file_might_contain_data(self, file_path: str, start_time: datetime, end_time: datetime) -> bool:
        """
        Check if a Parquet file might contain data in the specified time range.
        This is a heuristic based on the file path structure.
        """
        try:
            # Extract year and month from path (format: year=YYYY/month=MM/data.parquet)
            path_parts = file_path.split(os.sep)
            year_part = None
            month_part = None
            
            for part in path_parts:
                if part.startswith('year='):
                    year_part = int(part.split('=')[1])
                elif part.startswith('month='):
                    month_part = int(part.split('=')[1])
            
            if year_part is None or month_part is None:
                return True  # If we can't determine, assume it might contain data
            
            # Check if the file's time range overlaps with our query
            file_start = datetime(year_part, month_part, 1)
            if month_part == 12:
                file_end = datetime(year_part + 1, 1, 1) - timedelta(seconds=1)
            else:
                file_end = datetime(year_part, month_part + 1, 1) - timedelta(seconds=1)
            
            return (file_start <= end_time) and (file_end >= start_time)
            
        except Exception:
            return True  # If we can't determine, assume it might contain data
    
    def save_tick_data(self, symbol: str, year: int, month: int, df: pd.DataFrame):
        """
        Save tick data to partitioned Parquet storage.
        
        Args:
            symbol: Trading symbol
            year: Year for partitioning
            month: Month for partitioning
            df: DataFrame to save
        """
        if df.empty:
            logger.warning("Attempting to save empty DataFrame")
            return
        
        try:
            # Create partitioned directory structure
            partition_path = os.path.join(
                self.processed_dir, 
                symbol, 
                f"year={year}", 
                f"month={month:02d}"
            )
            os.makedirs(partition_path, exist_ok=True)
            
            # Save to Parquet with optimized settings
            parquet_file = os.path.join(partition_path, "data.parquet")
            
            # Convert to PyArrow table for better performance
            table = pa.Table.from_pandas(df)
            
            # Write with compression and row group size optimization
            pq.write_table(
                table, 
                parquet_file, 
                compression='snappy',
                row_group_size=100000  # 100k rows per row group
            )
            
            logger.info(f"Saved {len(df)} ticks to {parquet_file}")
            
        except Exception as e:
            logger.error(f"Failed to save Parquet file: {e}")
            raise
    
    def get_available_symbols(self) -> List[str]:
        """
        Get list of available symbols in the storage.
        
        Returns:
            List of symbol names
        """
        if not os.path.exists(self.processed_dir):
            return []
        
        symbols = []
        for item in os.listdir(self.processed_dir):
            item_path = os.path.join(self.processed_dir, item)
            if os.path.isdir(item_path):
                symbols.append(item)
        
        return symbols
    
    def get_data_range(self, symbol: str) -> Optional[Tuple[datetime, datetime]]:
        """
        Get the date range of available data for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Tuple of (start_date, end_date) or None if no data
        """
        symbol_dir = os.path.join(self.processed_dir, symbol)
        if not os.path.exists(symbol_dir):
            return None
        
        start_dates = []
        end_dates = []
        
        # Walk through all Parquet files to find date range
        for root, dirs, files in os.walk(symbol_dir):
            for file in files:
                if file.endswith('.parquet'):
                    file_path = os.path.join(root, file)
                    try:
                        # Read just the timestamp column to get range
                        df = pd.read_parquet(file_path, columns=['timestamp'])
                        if not df.empty:
                            start_dates.append(df['timestamp'].min())
                            end_dates.append(df['timestamp'].max())
                    except Exception as e:
                        logger.error(f"Failed to read timestamp range from {file_path}: {e}")
                        continue
        
        if not start_dates or not end_dates:
            return None
        
        return (min(start_dates), max(end_dates))
    
    def get_cache_stats(self) -> Dict:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        return self.cache.get_stats()
    
    def clear_cache(self):
        """Clear the LRU cache."""
        self.cache.clear()
        logger.info("Cache cleared")
    
    def get_storage_stats(self) -> Dict:
        """
        Get storage statistics.
        
        Returns:
            Dictionary with storage statistics
        """
        stats = {
            'processed_dir': self.processed_dir,
            'cache_stats': self.cache.get_stats(),
            'available_symbols': self.get_available_symbols(),
            'total_files': 0,
            'total_size_mb': 0
        }
        
        # Calculate total files and size
        for symbol in stats['available_symbols']:
            symbol_dir = os.path.join(self.processed_dir, symbol)
            for root, dirs, files in os.walk(symbol_dir):
                for file in files:
                    if file.endswith('.parquet'):
                        file_path = os.path.join(root, file)
                        stats['total_files'] += 1
                        stats['total_size_mb'] += os.path.getsize(file_path) / (1024 * 1024)
        
        return stats 