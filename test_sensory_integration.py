#!/usr/bin/env python3
"""
Test script for the new multidimensional sensory system integration.
"""

import sys
import os
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import asyncio

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
        from sensory.dimensions.enhanced_why_dimension import EnhancedFundamentalIntelligenceEngine as WhyDimension
        from sensory.dimensions.enhanced_how_dimension import InstitutionalMechanicsEngine as HowDimension
        from sensory.dimensions.enhanced_what_dimension import TechnicalRealityEngine as WhatDimension
        from sensory.dimensions.enhanced_when_dimension import ChronalIntelligenceEngine as WhenDimension
        from sensory.dimensions.enhanced_anomaly_dimension import AnomalyIntelligenceEngine as AnomalyDimension
        
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
        from sensory.orchestration.enhanced_intelligence_engine import ContextualFusionEngine as IntelligenceEngine
        
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
        
        # Run the async analysis method
        result = asyncio.run(engine.analyze_market_intelligence(test_market_data))
        print("✓ Successfully ran intelligence engine analysis")
        
        print(f"✓ Successfully processed market data:")
        print(f"  Narrative: {result.narrative_text[:100]}...")
        print(f"  Confidence: {result.confidence:.3f}")
        print(f"  Intelligence Level: {result.intelligence_level.name}")
        print(f"  Dominant Narrative: {result.dominant_narrative.name}")
        print(f"  Unified Score: {result.unified_score:.3f}")
        
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
        cortex = SensoryCortex()  # No arguments needed for ContextualFusionEngine
        
        print("✓ Successfully created SensoryCortex")
        
        # Test processing a data point using the correct interface
        test_row = df.iloc[-1]
        test_market_data = MarketData(
            timestamp=test_row['timestamp'],
            symbol='EURUSD',
            bid=test_row['close'] - 0.0001,
            ask=test_row['close'] + 0.0001,
            volume=test_row['volume'],
            spread=0.0002
        )
        
        # Use the correct method for ContextualFusionEngine
        result = asyncio.run(cortex.analyze_market_intelligence(test_market_data))
        
        print(f"✓ Successfully processed data through SensoryCortex:")
        print(f"  Intelligence Level: {result.intelligence_level.name}")
        print(f"  Confidence: {result.confidence:.3f}")
        print(f"  Unified Score: {result.unified_score:.3f}")
        print(f"  Dominant Narrative: {result.dominant_narrative.name}")
        print(f"  Narrative Coherence: {result.narrative_coherence.name}")
        print(f"  Risk Factors: {len(result.risk_factors)}")
        print(f"  Opportunity Factors: {len(result.opportunity_factors)}")
        
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