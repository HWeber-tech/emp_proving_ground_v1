#!/usr/bin/env python3
"""
Quick test to verify fraud removal and real data integration
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.sensory.organs.yahoo_finance_organ import YahooFinanceOrgan

def test_real_data():
    """Test that we can get real market data"""
    print("Testing fraud removal and real data integration...")
    
    try:
        # Test Yahoo Finance integration
        organ = YahooFinanceOrgan()
        
        # Fetch real EURUSD data
        data = organ.fetch_data('EURUSD=X', period="1d", interval="1h")
        
        if data is None:
            print("❌ FAILED: No data retrieved")
            return False
            
        print(f"✅ SUCCESS: Retrieved {len(data)} rows of real EURUSD data")
        print(f"   Date range: {data['timestamp'].min()} to {data['timestamp'].max()}")
        print(f"   Price range: {data['close'].min():.4f} to {data['close'].max():.4f}")
        
        # Check for synthetic data patterns
        price_changes = data['close'].pct_change().dropna()
        volatility = price_changes.std()
        
        if volatility < 0.0001:
            print("⚠️  WARNING: Data appears synthetic (too smooth)")
            return False
            
        print(f"✅ SUCCESS: Data volatility {volatility:.6f} indicates real market data")
        
        # Test data structure
        required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in data.columns]
        
        if missing_cols:
            print(f"❌ FAILED: Missing columns: {missing_cols}")
            return False
            
        print("✅ SUCCESS: All required columns present")
        
        # Test data quality
        null_count = data[required_cols].isnull().sum().sum()
        if null_count > 0:
            print(f"⚠️  WARNING: Found {null_count} null values")
        else:
            print("✅ SUCCESS: No null values found")
            
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False

if __name__ == "__main__":
    success = test_real_data()
    if success:
        print("\n🎉 FRAUD REMOVAL VERIFICATION: PASSED")
        print("   Real market data integration is working correctly")
    else:
        print("\n❌ FRAUD REMOVAL VERIFICATION: FAILED")
        print("   Issues detected in real data integration")
    
    sys.exit(0 if success else 1)
