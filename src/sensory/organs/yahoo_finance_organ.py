"""
Yahoo Finance Organ - Ticket DATA-01
Real historical data organ using yfinance for development simulations
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


class YahooFinanceOrgan:
    """
    Sensory organ for fetching real historical market data from Yahoo Finance
    Provides high-fidelity data for meaningful evolutionary simulations
    """
    
    def __init__(self, data_dir: str = "data/historical"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"YahooFinanceOrgan initialized with data directory: {self.data_dir}")
    
    def _symbol_to_filename(self, symbol: str, interval: str) -> str:
        """Convert symbol and interval to standardized filename"""
        # Convert forex symbols (EURUSD=X -> EURUSD)
        clean_symbol = symbol.replace('=X', '')
        return f"{clean_symbol}_{interval}.parquet"
    
    def _symbol_to_yahoo_format(self, symbol: str) -> str:
        """Convert symbol to Yahoo Finance format"""
        # Handle forex pairs
        if len(symbol) == 6 and symbol[:3].isalpha() and symbol[3:6].isalpha():
            return f"{symbol[:3]}{symbol[3:6]}=X"
        return symbol
    
    def download_data(
        self,
        symbol: str,
        start_date: str,
        end_date: Optional[str] = None,
        interval: str = "1h"
    ) -> bool:
        """
        Download historical data from Yahoo Finance
        
        Args:
            symbol: Trading symbol (e.g., "EURUSD", "AAPL")
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format (default: today)
            interval: Data interval ("1m", "2m", "5m", "15m", "30m", "1h", "1d", "1wk", "1mo")
        
        Returns:
            bool: True if download successful, False otherwise
        """
        try:
            # Convert symbol to Yahoo format
            yahoo_symbol = self._symbol_to_yahoo_format(symbol)
            
            # Set end date to today if not provided
            if end_date is None:
                end_date = datetime.now().strftime("%Y-%m-%d")
            
            logger.info(f"Downloading {symbol} data from {start_date} to {end_date} ({interval})")
            
            # Download data
            ticker = yf.Ticker(yahoo_symbol)
            df = ticker.history(
                start=start_date,
                end=end_date,
                interval=interval
            )
            
            if df.empty:
                logger.warning(f"No data returned for {symbol}")
                return False
            
            # Clean and standardize the data
            df = df.reset_index()
            df = df.rename(columns={
                'Datetime': 'timestamp',
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            })
            
            # Ensure timestamp is datetime
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Add symbol column
            df['symbol'] = symbol
            
            # Save to parquet
            filename = self._symbol_to_filename(symbol, interval)
            filepath = self.data_dir / filename
            
            df.to_parquet(filepath, engine='pyarrow')
            
            logger.info(f"Successfully downloaded {len(df)} rows for {symbol} to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error downloading data for {symbol}: {e}")
            return False
    
    def get_available_data(self) -> Dict[str, Dict[str, Any]]:
        """Get list of available historical data files"""
        available = {}
        
        for file in self.data_dir.glob("*.parquet"):
            try:
                # Parse filename
                parts = file.stem.split('_')
                if len(parts) >= 2:
                    symbol = parts[0]
                    interval = '_'.join(parts[1:])
                    
                    # Get file info
                    stat = file.stat()
                    df = pd.read_parquet(file)
                    
                    available[file.stem] = {
                        'symbol': symbol,
                        'interval': interval,
                        'rows': len(df),
                        'start_date': df['timestamp'].min().strftime('%Y-%m-%d'),
                        'end_date': df['timestamp'].max().strftime('%Y-%m-%d'),
                        'file_size_mb': stat.st_size / (1024 * 1024),
                        'last_modified': datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
                    }
            except Exception as e:
                logger.warning(f"Error reading file info for {file}: {e}")
        
        return available
    
    def load_data(self, symbol: str, interval: str = "1h") -> Optional[pd.DataFrame]:
        """
        Load historical data from local storage
        
        Args:
            symbol: Trading symbol
            interval: Data interval
            
        Returns:
            DataFrame with historical data or None if not found
        """
        try:
            filename = self._symbol_to_filename(symbol, interval)
            filepath = self.data_dir / filename
            
            if not filepath.exists():
                logger.warning(f"No historical data found for {symbol} ({interval})")
                return None
            
            df = pd.read_parquet(filepath)
            logger.info(f"Loaded {len(df)} rows for {symbol} ({interval})")
            return df
            
        except Exception as e:
            logger.error(f"Error loading data for {symbol}: {e}")
            return None
    
    def fetch_data(self, symbol: str, period: str = "1d", interval: str = "1m") -> Optional[pd.DataFrame]:
        """
        Fetch real-time market data from Yahoo Finance
        
        Args:
            symbol: Trading symbol
            period: Data period ("1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max")
            interval: Data interval ("1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo")
            
        Returns:
            DataFrame with market data or None if fetch fails
        """
        try:
            # Convert symbol to Yahoo format
            yahoo_symbol = self._symbol_to_yahoo_format(symbol)
            
            # Create ticker object
            ticker = yf.Ticker(yahoo_symbol)
            
            # Fetch data
            data = ticker.history(period=period, interval=interval)
            
            if data.empty:
                logger.warning(f"No data received for {symbol}")
                return None
            
            # Clean and standardize the data
            data = data.reset_index()
            data = data.rename(columns={
                'Datetime': 'timestamp',
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            })
            
            # Ensure timestamp is datetime
            if 'timestamp' in data.columns:
                data['timestamp'] = pd.to_datetime(data['timestamp'])
            
            # Add symbol column
            data['symbol'] = symbol
            
            logger.info(f"Successfully fetched {len(data)} rows for {symbol}")
            return data
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return None
    
    def validate_data(self, symbol: str, interval: str = "1h") -> bool:
        """Validate the integrity of downloaded data"""
        try:
            df = self.load_data(symbol, interval)
            if df is None:
                return False
            
            # Basic validation checks
            required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                logger.error(f"Missing columns: {missing_columns}")
                return False
            
            # Check for NaN values
            nan_count = df[required_columns].isnull().sum().sum()
            if nan_count > 0:
                logger.warning(f"Found {nan_count} NaN values in data")
            
            # Check for duplicate timestamps
            duplicate_timestamps = df['timestamp'].duplicated().sum()
            if duplicate_timestamps > 0:
                logger.warning(f"Found {duplicate_timestamps} duplicate timestamps")
            
            # Check price consistency
            invalid_prices = ((df['high'] < df['low']) | 
                            (df['high'] < df['open']) | 
                            (df['high'] < df['close']) |
                            (df['low'] > df['open']) | 
                            (df['low'] > df['close'])).sum()
            
            if invalid_prices > 0:
                logger.warning(f"Found {invalid_prices} invalid price relationships")
            
            logger.info(f"Data validation passed for {symbol} ({interval})")
            return True
            
        except Exception as e:
            logger.error(f"Error validating data for {symbol}: {e}")
            return False


# Example usage
if __name__ == "__main__":
    organ = YahooFinanceOrgan()
    
    # Download EURUSD data
    success = organ.download_data(
        symbol="EURUSD",
        start_date="2024-01-01",
        end_date="2024-12-31",
        interval="1h"
    )
    
    if success:
        print("Data downloaded successfully")
        
        # List available data
        available = organ.get_available_data()
        for key, info in available.items():
            print(f"{key}: {info}")
        
        # Validate the data
        if organ.validate_data("EURUSD", "1h"):
            print("Data validation passed")
        
        # Load and display sample data
        df = organ.load_data("EURUSD", "1h")
        if df is not None:
            print(f"\nSample data:\n{df.head()}")
            print(f"\nData shape: {df.shape}")
            print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
