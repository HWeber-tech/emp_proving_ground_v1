"""
DukascopyIngestor: Downloads, decompresses, and processes real .bi5 tick data.
"""
import os
import lzma
import struct
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, List
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DukascopyIngestor:
    """
    Downloads, decompresses, and processes real .bi5 tick data from Dukascopy.
    Raises RuntimeError if duka is unavailable or download fails.
    """
    
    def __init__(self, raw_dir: str, processed_dir: str, max_retries: int = 3):
        self.raw_dir = raw_dir
        self.processed_dir = processed_dir
        self.max_retries = max_retries
        
        # Create directories if they don't exist
        os.makedirs(raw_dir, exist_ok=True)
        os.makedirs(processed_dir, exist_ok=True)
        
        # Try to import duka - this will fail if not available
        try:
            import duka
            self.duka_available = True
        except ImportError:
            logger.error("Duka library not available. Cannot download real data.")
            self.duka_available = False
    
    def ingest_range(self, symbol: str, start_date: datetime, end_date: datetime) -> bool:
        """
        Download, decompress, clean, and store tick data for the given range.
        
        Args:
            symbol: Trading symbol (e.g., 'EURUSD')
            start_date: Start date for data download
            end_date: End date for data download
            
        Returns:
            bool: True if successful, False otherwise
            
        Raises:
            RuntimeError: If duka is not available or download fails
        """
        if not self.duka_available:
            raise RuntimeError("Duka library not available. Cannot download real data.")
        
        logger.info(f"Starting data ingestion for {symbol} from {start_date} to {end_date}")
        
        try:
            # Download raw .bi5 files
            raw_files = self._download_raw_data(symbol, start_date, end_date)
            if not raw_files:
                raise RuntimeError(f"Failed to download any data for {symbol}")
            
            # Process each downloaded file
            for raw_file in raw_files:
                self._process_bi5_file(symbol, raw_file)
            
            logger.info(f"Successfully ingested data for {symbol}")
            return True
            
        except Exception as e:
            logger.error(f"Data ingestion failed for {symbol}: {e}")
            raise RuntimeError(f"Data ingestion failed: {e}")
    
    def _download_raw_data(self, symbol: str, start_date: datetime, end_date: datetime) -> List[str]:
        """
        Download raw .bi5 files using duka library.
        
        Args:
            symbol: Trading symbol
            start_date: Start date
            end_date: End date
            
        Returns:
            List of downloaded file paths
        """
        try:
            import duka
            
            # Create symbol-specific directory
            symbol_dir = os.path.join(self.raw_dir, symbol)
            os.makedirs(symbol_dir, exist_ok=True)
            
            # Download data using duka
            duka.download_ticks(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                output_dir=symbol_dir
            )
            
            # Find downloaded files
            downloaded_files = []
            for root, dirs, files in os.walk(symbol_dir):
                for file in files:
                    if file.endswith('.bi5'):
                        downloaded_files.append(os.path.join(root, file))
            
            logger.info(f"Downloaded {len(downloaded_files)} .bi5 files for {symbol}")
            return downloaded_files
            
        except Exception as e:
            logger.error(f"Download failed for {symbol}: {e}")
            raise RuntimeError(f"Download failed: {e}")
    
    def _process_bi5_file(self, symbol: str, bi5_file_path: str):
        """
        Decompress and parse a .bi5 file.
        
        Args:
            symbol: Trading symbol
            bi5_file_path: Path to the .bi5 file
        """
        try:
            # Extract date from filename (format: YYYYMMDD_HHMMSS.bi5)
            filename = os.path.basename(bi5_file_path)
            date_str = filename.split('.')[0]
            
            # Parse timestamp
            if '_' in date_str:
                date_part, time_part = date_str.split('_')
                year = int(date_part[:4])
                month = int(date_part[4:6])
                day = int(date_part[6:8])
                hour = int(time_part[:2])
                minute = int(time_part[2:4])
                second = int(time_part[4:6])
                
                base_timestamp = datetime(year, month, day, hour, minute, second)
            else:
                # Fallback for different filename format
                year = int(date_str[:4])
                month = int(date_str[4:6])
                day = int(date_str[6:8])
                base_timestamp = datetime(year, month, day)
            
            # Decompress and parse the .bi5 file
            tick_data = self._parse_bi5_file(bi5_file_path, base_timestamp)
            
            if tick_data is not None and not tick_data.empty:
                # Save to partitioned Parquet storage
                self._save_to_parquet(symbol, year, month, tick_data)
                
        except Exception as e:
            logger.error(f"Failed to process {bi5_file_path}: {e}")
            raise
    
    def _parse_bi5_file(self, bi5_file_path: str, base_timestamp: datetime) -> Optional[pd.DataFrame]:
        """
        Parse a .bi5 file and extract tick data.
        
        Args:
            bi5_file_path: Path to the .bi5 file
            base_timestamp: Base timestamp for the file
            
        Returns:
            DataFrame with columns: timestamp, bid, ask, bid_volume, ask_volume
        """
        try:
            with open(bi5_file_path, 'rb') as f:
                compressed_data = f.read()
            
            # Decompress using LZMA
            decompressed_data = lzma.decompress(compressed_data)
            
            # Parse binary format: (timestamp_delta, ask, bid, ask_volume, bid_volume)
            # Each record is 20 bytes: 4 bytes for each field
            record_size = 20
            num_records = len(decompressed_data) // record_size
            
            timestamps = []
            asks = []
            bids = []
            ask_volumes = []
            bid_volumes = []
            
            current_timestamp = base_timestamp
            
            for i in range(num_records):
                offset = i * record_size
                record = decompressed_data[offset:offset + record_size]
                
                # Unpack binary data (little-endian)
                timestamp_delta, ask, bid, ask_volume, bid_volume = struct.unpack('<IIIII', record)
                
                # Convert timestamp delta to actual timestamp
                current_timestamp += timedelta(milliseconds=timestamp_delta)
                
                # Convert prices from integer format (multiply by 100000 for 5 decimal places)
                ask_price = ask / 100000.0
                bid_price = bid / 100000.0
                
                timestamps.append(current_timestamp)
                asks.append(ask_price)
                bids.append(bid_price)
                ask_volumes.append(ask_volume)
                bid_volumes.append(bid_volume)
            
            # Create DataFrame
            df = pd.DataFrame({
                'timestamp': timestamps,
                'ask': asks,
                'bid': bids,
                'ask_volume': ask_volumes,
                'bid_volume': bid_volumes
            })
            
            # Add mid price and spread
            df['mid_price'] = (df['ask'] + df['bid']) / 2
            df['spread'] = df['ask'] - df['bid']
            
            logger.info(f"Parsed {len(df)} ticks from {os.path.basename(bi5_file_path)}")
            return df
            
        except Exception as e:
            logger.error(f"Failed to parse {bi5_file_path}: {e}")
            return None
    
    def _save_to_parquet(self, symbol: str, year: int, month: int, df: pd.DataFrame):
        """
        Save tick data to partitioned Parquet storage.
        
        Args:
            symbol: Trading symbol
            year: Year for partitioning
            month: Month for partitioning
            df: DataFrame to save
        """
        try:
            # Create partitioned directory structure
            partition_path = os.path.join(
                self.processed_dir, 
                symbol, 
                f"year={year}", 
                f"month={month:02d}"
            )
            os.makedirs(partition_path, exist_ok=True)
            
            # Save to Parquet
            parquet_file = os.path.join(partition_path, "data.parquet")
            df.to_parquet(parquet_file, index=False, compression='snappy')
            
            logger.info(f"Saved {len(df)} ticks to {parquet_file}")
            
        except Exception as e:
            logger.error(f"Failed to save Parquet file: {e}")
            raise 