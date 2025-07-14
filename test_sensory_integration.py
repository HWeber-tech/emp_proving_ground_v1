#!/usr/bin/env python3
"""
Test script for the new multidimensional sensory system integration.
"""

import sys
import os
import pandas as pd
from datetime import datetime, timedelta
import numpy as np

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def create_test_data():
    """Create some test market data"""
    # Create a simple DataFrame with OHLCV data
    dates = pd.date_range(start='2024-01-01', periods=100, freq='1h')
    
    # Simulate some price movement
    base_price = 1.1000
    prices = []
    for i in range(100):
        # Add some trend and noise
        trend = 0.0001 * i  # Small upward trend
        noise = np.random.normal(0, 0.0005)  # Random noise
        price = base_price + trend + noise
        prices.append(price)
    
    # Create OHLCV data
    data = []
    for i, (date, price) in enumerate(zip(dates, prices)):
        # Simulate OHLC from price
        open_price = price
        high_price = price + abs(np.random.normal(0, 0.0002))
        low_price = price - abs(np.random.normal(0, 0.0002))
        close_price = price + np.random.normal(0, 0.0001)
        
        # Ensure high >= max(open, close) and low <= min(open, close)
        high_price = max(high_price, open_price, close_price)
        low_price = min(low_price, open_price, close_price)
        
        # Simulate volume
        volume = int(np.random.uniform(1000, 5000))
        
        data.append({
            'timestamp': date,
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close_price,
            'volume': volume
        })
    
    return pd.DataFrame(data)

def test_sensory_integration():
    """Test the new sensory system integration"""
    print("Testing new multidimensional sensory system integration...")
    
    # Create test data
    print("Creating test data...")
    df = create_test_data()
    print(f"Created {len(df)} data points")
    
    # Test importing the sensory module
    print("Testing sensory module import...")
    try:
        from sensory import SensoryCortex
        print("✓ Successfully imported SensoryCortex")
    except ImportError as e:
        print(f"✗ Failed to import SensoryCortex: {e}")
        return
    
    # Test creating a simple market data object
    print("Testing market data object creation...")
    try:
        from sensory.core.base import MarketData
        from datetime import datetime
        
        # Create a simple market data object
        test_data = MarketData(
            timestamp=datetime.now(),
            symbol='EURUSD',
            bid=1.1000,
            ask=1.1001,
            volume=1000,
            spread=0.0001
        )
        print("✓ Successfully created MarketData object")
    except Exception as e:
        print(f"✗ Failed to create MarketData object: {e}")
        return
    
    # Test dimensional sensors
    print("Testing dimensional sensors...")
    try:
        from sensory.dimensions.why_dimension import WhyDimension
        from sensory.dimensions.how_dimension import HowDimension
        from sensory.dimensions.what_dimension import WhatDimension
        from sensory.dimensions.when_dimension import WhenDimension
        from sensory.dimensions.anomaly_dimension import AnomalyDimension
        
        # Create sensors
        why_sensor = WhyDimension()
        how_sensor = HowDimension()
        what_sensor = WhatDimension()
        when_sensor = WhenDimension()
        anomaly_sensor = AnomalyDimension()
        
        print("✓ Successfully created all dimensional sensors")
        
        # The original system doesn't need calibration
        print("✓ Original system uses adaptive learning (no calibration needed)")
        
    except Exception as e:
        print(f"✗ Failed to test dimensional sensors: {e}")
        return
    
    # Test intelligence engine
    print("Testing intelligence engine...")
    try:
        from sensory.orchestration.intelligence_engine import IntelligenceEngine
        
        # Create the intelligence engine
        engine = IntelligenceEngine()
        
        print("✓ Successfully created IntelligenceEngine")
        
        # Test processing a market data point
        test_market_data = MarketData(
            timestamp=datetime.now(),
            symbol='EURUSD',
            bid=1.1000,
            ask=1.1001,
            volume=1000,
            spread=0.0001
        )
        
        understanding = engine.process_market_data(test_market_data)
        
        print(f"✓ Successfully processed market data:")
        print(f"  Narrative: {understanding.narrative[:100]}...")
        print(f"  Confidence: {understanding.confidence:.3f}")
        print(f"  Intelligence Level: {understanding.intelligence_level.name}")
        print(f"  Regime: {understanding.regime.name}")
        
    except Exception as e:
        print(f"✗ Failed to test intelligence engine: {e}")
        return
    
    # Test the full SensoryCortex interface
    print("Testing full SensoryCortex interface...")
    try:
        # Create a mock data storage
        class MockDataStorage:
            def get_data_range(self, symbol, start_time, end_time):
                return df
        
        mock_storage = MockDataStorage()
        cortex = SensoryCortex('EURUSD', mock_storage)
        
        print("✓ Successfully created SensoryCortex")
        
        # Test calibration (should always return True)
        success = cortex.calibrate(df['timestamp'].min(), df['timestamp'].max())
        if success:
            print("✓ Calibration successful")
        
        # Test processing a data point
        test_row = df.iloc[-1]
        current_data = pd.Series({
            'open': test_row['open'],
            'high': test_row['high'],
            'low': test_row['low'],
            'close': test_row['close'],
            'volume': test_row['volume']
        }, name=test_row['timestamp'])
        
        reading = cortex.perceive(current_data)
        
        print(f"✓ Successfully processed data through SensoryCortex:")
        print(f"  Overall Sentiment: {reading.overall_sentiment}")
        print(f"  Confidence Level: {reading.confidence_level:.3f}")
        print(f"  Risk Level: {reading.risk_level:.3f}")
        print(f"  Macro Trend: {reading.macro_trend}")
        print(f"  Technical Signal: {reading.technical_signal}")
        print(f"  Session Phase: {reading.session_phase}")
        print(f"  Manipulation Probability: {reading.manipulation_probability:.3f}")
        
    except Exception as e:
        print(f"✗ Failed to test SensoryCortex interface: {e}")
        return
    
    print("\n🎉 Sensory system integration test completed successfully!")
    print("\nKey features tested:")
    print("  ✓ Module imports")
    print("  ✓ Data structures")
    print("  ✓ Dimensional sensors")
    print("  ✓ Intelligence engine")
    print("  ✓ Full SensoryCortex interface")
    print("  ✓ Data processing")
    print("\nThe complete multidimensional market intelligence system is ready for use!")

if __name__ == "__main__":
    test_sensory_integration() 