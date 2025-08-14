"""
Script to download and process historical tick data using DukascopyIngestor.
"""
import logging
import os
import sys
from datetime import datetime

import yaml

# Add the parent directory to the path so we can import emp modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from emp.data_pipeline.cleaner import TickDataCleaner
from emp.data_pipeline.ingestor import DukascopyIngestor
from emp.data_pipeline.storage import TickDataStorage

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_config(config_path: str = "config.yaml") -> dict:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config file
        
    Returns:
        Configuration dictionary
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        logger.error(f"Failed to load config from {config_path}: {e}")
        raise

def download_data_for_year(symbol: str, year: int, config: dict) -> bool:
    """
    Download and process data for a specific year.
    
    Args:
        symbol: Trading symbol
        year: Year to download
        config: Configuration dictionary
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Initialize components
        raw_dir = config['data']['raw_dir']
        processed_dir = config['data']['processed_dir']
        
        ingestor = DukascopyIngestor(raw_dir, processed_dir)
        cleaner = TickDataCleaner(symbol)
        storage = TickDataStorage(processed_dir)
        
        # Define date range for the year
        start_date = datetime(year, 1, 1)
        end_date = datetime(year, 12, 31, 23, 59, 59)
        
        logger.info(f"Starting download for {symbol} in {year}")
        
        # Download and process data
        success = ingestor.ingest_range(symbol, start_date, end_date)
        
        if success:
            logger.info(f"Successfully downloaded data for {symbol} in {year}")
            return True
        else:
            logger.error(f"Failed to download data for {symbol} in {year}")
            return False
            
    except Exception as e:
        logger.error(f"Error downloading data for {symbol} in {year}: {e}")
        return False

def main():
    """
    Main function to download historical data.
    """
    try:
        # Load configuration
        config = load_config()
        
        symbol = config['data']['symbol']
        start_year = config['data']['start_year']
        end_year = config['data']['end_year']
        
        logger.info(f"Starting data download for {symbol} from {start_year} to {end_year}")
        
        # Download data for each year
        successful_years = []
        failed_years = []
        
        for year in range(start_year, end_year + 1):
            logger.info(f"Processing year {year}...")
            
            if download_data_for_year(symbol, year, config):
                successful_years.append(year)
            else:
                failed_years.append(year)
        
        # Report results
        logger.info("=" * 50)
        logger.info("DOWNLOAD SUMMARY")
        logger.info("=" * 50)
        logger.info(f"Symbol: {symbol}")
        logger.info(f"Period: {start_year} - {end_year}")
        logger.info(f"Successful years: {successful_years}")
        logger.info(f"Failed years: {failed_years}")
        logger.info(f"Success rate: {len(successful_years)}/{end_year - start_year + 1}")
        
        if failed_years:
            logger.warning(f"Some years failed to download: {failed_years}")
            return 1
        else:
            logger.info("All years downloaded successfully!")
            return 0
            
    except Exception as e:
        logger.error(f"Download script failed: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 
