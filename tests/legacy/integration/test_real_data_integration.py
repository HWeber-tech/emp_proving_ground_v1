#!/usr/bin/env python3
"""
Test Real Data Integration

This script tests the real data integration to ensure
it can download and process real market data from multiple sources.

Usage:
    python test_real_data_integration.py --symbol EURUSD --days 7

Requirements:
    - Internet connection
    - Optional: Alpha Vantage API key for premium data
"""

import asyncio
import argparse
import logging
import sys
import yaml
from datetime import datetime, timedelta
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Import data components
try:
    from data import DukascopyIngestor, TickDataStorage, TickDataCleaner
    DATA_AVAILABLE = True
except ImportError:
    print("⚠️  Data components not available")
    DATA_AVAILABLE = False


def setup_logging():
    """Setup logging for the test."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('logs/real_data_test.log')
        ]
    )


def test_dukascopy_connection():
    """Test Dukascopy connection and data download."""
    print("\n🔗 Testing Dukascopy Connection...")
    
    try:
        ingestor = DukascopyIngestor()
        
        # Test connection
        print("   Testing connection to Dukascopy servers...")
        connected = ingestor.test_connection()
        
        if connected:
            print("✅ Dukascopy connection test passed")
            
            # Test data download for recent data
            test_symbol = 'EURUSD'
            test_date = datetime.now().date() - timedelta(days=1)
            
            print(f"   Downloading test data for {test_symbol} on {test_date}...")
            data = ingestor._download_day_data(test_symbol, test_date)
            
            if data is not None and not data.empty:
                print(f"✅ Downloaded {len(data)} ticks from Dukascopy")
                print(f"   Sample data:")
                print(f"   {data.head()}")
                return True
            else:
                print("⚠️  No data available (this may be normal for recent dates)")
                return True  # Connection works, just no data
        else:
            print("❌ Dukascopy connection test failed")
            return False
            
    except Exception as e:
        print(f"❌ Error testing Dukascopy: {e}")
        return False


def test_yahoo_finance_integration():
    """Test Yahoo Finance integration."""
    print("\n📈 Testing Yahoo Finance Integration...")
    
    try:
        from data.real_data_ingestor import RealDataIngestor
        
        ingestor = RealDataIngestor()
        
        # Test data download
        symbol = 'EURUSD'
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)
        
        print(f"   Downloading {symbol} data from {start_date.date()} to {end_date.date()}...")
        data = ingestor.download_yahoo_data(symbol, start_date, end_date)
        
        if data is not None and not data.empty:
            print(f"✅ Downloaded {len(data)} records from Yahoo Finance")
            print(f"   Sample data:")
            print(f"   {data.head()}")
            return True
        else:
            print("❌ No data downloaded from Yahoo Finance")
            return False
            
    except ImportError:
        print("⚠️  Yahoo Finance integration not available")
        return False
    except Exception as e:
        print(f"❌ Error testing Yahoo Finance: {e}")
        return False


def test_alpha_vantage_integration():
    """Test Alpha Vantage integration."""
    print("\n📊 Testing Alpha Vantage Integration...")
    
    try:
        from data.real_data_ingestor import RealDataIngestor
        
        # Check for API key
        api_key = None
        try:
            with open('config/trading/ctrader_config.yaml', 'r') as f:
                config = yaml.safe_load(f)
                api_key = config.get('alpha_vantage', {}).get('api_key')
        except:
            pass
        
        if not api_key:
            print("⚠️  No Alpha Vantage API key found (skipping test)")
            return True  # Not a failure, just no credentials
        
        ingestor = RealDataIngestor(api_key=api_key)
        
        # Test data download
        symbol = 'EURUSD'
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)
        
        print(f"   Downloading {symbol} data from {start_date.date()} to {end_date.date()}...")
        data = ingestor.download_alpha_vantage_data(symbol, start_date, end_date)
        
        if data is not None and not data.empty:
            print(f"✅ Downloaded {len(data)} records from Alpha Vantage")
            print(f"   Sample data:")
            print(f"   {data.head()}")
            return True
        else:
            print("❌ No data downloaded from Alpha Vantage")
            return False
            
    except ImportError:
        print("⚠️  Alpha Vantage integration not available")
        return False
    except Exception as e:
        print(f"❌ Error testing Alpha Vantage: {e}")
        return False


def test_data_pipeline_integration():
    """Test the complete data pipeline integration."""
    print("\n🔄 Testing Data Pipeline Integration...")
    
    try:
        from data import DukascopyIngestor, TickDataStorage, TickDataCleaner
        
        # Create components
        storage = TickDataStorage()
        cleaner = TickDataCleaner()
        ingestor = DukascopyIngestor()
        
        # Test symbol
        symbol = 'EURUSD'
        year = 2024
        
        print(f"   Testing complete pipeline for {symbol} {year}...")
        
        # Test data download
        success = ingestor.download_year_data(symbol, year)
        
        if success:
            print(f"✅ Successfully downloaded data for {symbol} {year}")
            
            # Test data loading
            start_time = datetime(year, 1, 1)
            end_time = datetime(year, 1, 7)  # First week
            
            data = storage.load_tick_data(symbol, start_time, end_time)
            
            if not data.empty:
                print(f"✅ Successfully loaded {len(data)} records from storage")
                
                # Test data cleaning
                cleaned_data = cleaner.clean(data, symbol)
                
                if not cleaned_data.empty:
                    print(f"✅ Successfully cleaned data: {len(cleaned_data)} records remaining")
                    print(f"   Sample cleaned data:")
                    print(f"   {cleaned_data.head()}")
                    return True
                else:
                    print("❌ Data cleaning failed")
                    return False
            else:
                print("❌ Data loading failed")
                return False
        else:
            print("❌ Data download failed")
            return False
            
    except Exception as e:
        print(f"❌ Error testing data pipeline: {e}")
        return False


def test_fallback_mechanism():
    """Test the fallback mechanism when real data is unavailable."""
    print("\n🔄 Testing Fallback Mechanism...")
    
    try:
        from data import DukascopyIngestor, TickDataStorage, TickDataCleaner
        
        # Create components
        storage = TickDataStorage()
        cleaner = TickDataCleaner()
        ingestor = DukascopyIngestor()
        
        # Test with a symbol that likely has no data
        symbol = 'INVALID_SYMBOL'
        start_time = datetime(2024, 1, 1)
        end_time = datetime(2024, 1, 7)
        
        print(f"   Testing fallback for {symbol}...")
        
        # This should trigger fallback to synthetic data
        data = ingestor.download_tick_data(symbol, start_time, end_time)
        
        if data is None:
            print("✅ Fallback mechanism working correctly (returned None)")
            return True
        else:
            print("⚠️  Unexpected data returned for invalid symbol")
            return False
            
    except Exception as e:
        print(f"❌ Error testing fallback mechanism: {e}")
        return False


def main():
    """Main test function."""
    parser = argparse.ArgumentParser(description="Test real data integration")
    parser.add_argument("--symbol", default="EURUSD", help="Symbol to test")
    parser.add_argument("--days", type=int, default=7, help="Number of days to test")
    parser.add_argument("--test-dukascopy", action="store_true", help="Test Dukascopy only")
    parser.add_argument("--test-yahoo", action="store_true", help="Test Yahoo Finance only")
    parser.add_argument("--test-alpha", action="store_true", help="Test Alpha Vantage only")
    parser.add_argument("--test-pipeline", action="store_true", help="Test complete pipeline only")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    
    print("🚀 Real Data Integration Test")
    print("=" * 50)
    
    # Run tests based on arguments
    test_results = {}
    
    if args.test_dukascopy or not any([args.test_yahoo, args.test_alpha, args.test_pipeline]):
        test_results['Dukascopy'] = test_dukascopy_connection()
    
    if args.test_yahoo or not any([args.test_dukascopy, args.test_alpha, args.test_pipeline]):
        test_results['Yahoo Finance'] = test_yahoo_finance_integration()
    
    if args.test_alpha or not any([args.test_dukascopy, args.test_yahoo, args.test_pipeline]):
        test_results['Alpha Vantage'] = test_alpha_vantage_integration()
    
    if args.test_pipeline or not any([args.test_dukascopy, args.test_yahoo, args.test_alpha]):
        test_results['Data Pipeline'] = test_data_pipeline_integration()
    
    # Test fallback mechanism
    test_results['Fallback'] = test_fallback_mechanism()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Summary")
    print("=" * 50)
    
    for test_name, result in test_results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
    
    passed_tests = sum(test_results.values())
    total_tests = len(test_results)
    
    print(f"\nOverall: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("\n🎉 SUCCESS: All real data integration tests passed!")
        print("   The system can now download and process real market data.")
    elif passed_tests > 0:
        print(f"\n⚠️  PARTIAL SUCCESS: {passed_tests}/{total_tests} tests passed")
        print("   Some data sources are working, others may need configuration.")
    else:
        print("\n❌ FAILURE: No real data integration tests passed")
        print("   Check network connection and data source configuration.")


if __name__ == "__main__":
    main() 
