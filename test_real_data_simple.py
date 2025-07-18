#!/usr/bin/env python3
"""
Simple Real Data Integration Test

This test directly verifies that the real data ingestor works
without complex import dependencies.
"""

import sys
import os
import logging
import pandas as pd
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_real_data_download():
    """Test downloading real market data."""
    print("🧪 Testing Real Data Download...")
    
    try:
        # Import the real data ingestor
        from src.data.real_data_ingestor import RealDataIngestor
        
        # Initialize ingestor
        ingestor = RealDataIngestor()
        
        # Test with a recent date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)
        
        # Test downloading real data
        symbol = "EURUSD"
        success = ingestor.download_symbol_data(symbol, start_date, end_date, 'yahoo')
        
        if success:
            print(f"✅ Successfully downloaded real data for {symbol}")
            
            # Test loading the data
            data = ingestor.load_symbol_data(symbol, start_date, end_date)
            if data is not None and not data.empty:
                print(f"✅ Successfully loaded {len(data)} real data records")
                print(f"   Data range: {data.index.min()} to {data.index.max()}")
                print(f"   Columns: {list(data.columns)}")
                print(f"   Sample data:")
                print(data.head())
                return True
            else:
                print("❌ Failed to load downloaded data")
                return False
        else:
            print("❌ Failed to download real data")
            return False
            
    except Exception as e:
        print(f"❌ Error testing real data download: {e}")
        return False


def test_realistic_synthetic_data():
    """Test realistic synthetic data generation."""
    print("\n🧪 Testing Realistic Synthetic Data Generation...")
    
    try:
        from src.data.real_data_ingestor import RealDataIngestor
        
        ingestor = RealDataIngestor()
        symbol = "EURUSD"
        
        # Test realistic synthetic data generation
        test_data = ingestor.create_test_data_from_real_patterns(symbol, 7)
        if test_data is not None and not test_data.empty:
            print(f"✅ Generated {len(test_data)} realistic synthetic records")
            print(f"   Data range: {test_data.index.min()} to {test_data.index.max()}")
            print(f"   Columns: {list(test_data.columns)}")
            print(f"   Sample data:")
            print(test_data.head())
            return True
        else:
            print("❌ Failed to generate realistic synthetic data")
            return False
            
    except Exception as e:
        print(f"❌ Error testing synthetic data generation: {e}")
        return False


def test_multiple_instruments():
    """Test with multiple instruments."""
    print("\n🧪 Testing Multiple Instruments...")
    
    try:
        from src.data.real_data_ingestor import RealDataIngestor
        
        ingestor = RealDataIngestor()
        instruments = ["EURUSD", "GBPUSD", "USDJPY"]
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=3)
        
        results = {}
        
        for instrument in instruments:
            print(f"   Testing {instrument}...")
            try:
                success = ingestor.download_symbol_data(instrument, start_date, end_date, 'yahoo')
                if success:
                    data = ingestor.load_symbol_data(instrument, start_date, end_date)
                    if data is not None and not data.empty:
                        results[instrument] = len(data)
                        print(f"   ✅ {instrument}: {len(data)} records")
                    else:
                        results[instrument] = 0
                        print(f"   ❌ {instrument}: No data")
                else:
                    results[instrument] = 0
                    print(f"   ❌ {instrument}: Download failed")
            except Exception as e:
                results[instrument] = 0
                print(f"   ❌ {instrument}: Error - {e}")
        
        successful_instruments = sum(1 for count in results.values() if count > 0)
        print(f"\n📊 Results: {successful_instruments}/{len(instruments)} instruments have data")
        
        return successful_instruments > 0
        
    except Exception as e:
        print(f"❌ Error testing multiple instruments: {e}")
        return False


def main():
    """Run all real data integration tests."""
    print("🚀 SIMPLE REAL DATA INTEGRATION TEST")
    print("=" * 50)
    
    # Test 1: Real data download
    test1_passed = test_real_data_download()
    
    # Test 2: Realistic synthetic data
    test2_passed = test_realistic_synthetic_data()
    
    # Test 3: Multiple instruments
    test3_passed = test_multiple_instruments()
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 TEST SUMMARY")
    print("=" * 50)
    print(f"Real Data Download: {'✅ PASSED' if test1_passed else '❌ FAILED'}")
    print(f"Realistic Synthetic Data: {'✅ PASSED' if test2_passed else '❌ FAILED'}")
    print(f"Multiple Instruments: {'✅ PASSED' if test3_passed else '❌ FAILED'}")
    
    total_passed = sum([test1_passed, test2_passed, test3_passed])
    print(f"\nOverall: {total_passed}/3 tests passed")
    
    if total_passed >= 2:
        print("🎉 Real data integration is working!")
        print("   The system can now load real market data instead of synthetic data.")
    else:
        print("⚠️  Real data integration needs improvement.")
        print("   The system is still relying on synthetic data generation.")
    
    return total_passed >= 2


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 