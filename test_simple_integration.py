#!/usr/bin/env python3
"""
Simple Sensory Integration Test

This test verifies that the market regime detection and pattern recognition have been
properly integrated into the sensory system.
"""

import sys
import os
import logging
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.sensory.core.base import MarketData, MarketRegime, InstrumentMeta
from src.sensory.dimensions.enhanced_when_dimension import ChronalIntelligenceEngine
from src.sensory.dimensions.enhanced_anomaly_dimension import AnomalyIntelligenceEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_simple_integration():
    """Test simple integration of market regime detection and pattern recognition"""
    
    print("Testing Sensory Integration...")
    
    # Test 1: Verify analysis folder is removed
    analysis_path = os.path.join("src", "analysis")
    if os.path.exists(analysis_path):
        print("❌ Analysis folder still exists")
        return False
    else:
        print("✅ Analysis folder successfully removed")
    
    # Test 2: Test market regime detection integration
    try:
        # Create instrument metadata
        instrument_meta = InstrumentMeta(
            symbol="EURUSD",
            pip_size=0.0001,
            lot_size=100000
        )
        
        # Create temporal analyzer (market regime detection)
        temporal_engine = ChronalIntelligenceEngine(instrument_meta)
        
        # Generate test market data
        test_data = []
        base_price = 1.1000
        
        for i in range(10):
            price_change = 0.001 * (i % 3 - 1)  # Simple pattern
            base_price += price_change
            
            market_data = MarketData(
                symbol="EURUSD",
                timestamp=datetime.now() + timedelta(minutes=i),
                open=base_price,
                high=base_price + 0.0005,
                low=base_price - 0.0005,
                close=base_price + price_change,
                volume=1000 + i * 100,
                bid=base_price - 0.0001,
                ask=base_price + 0.0001
            )
            test_data.append(market_data)
        
        # Test regime detection
        import asyncio
        for market_data in test_data:
            reading = asyncio.run(temporal_engine.update(market_data))
            print(f"✅ Temporal analysis: {reading.dimension} - Signal: {reading.signal_strength:.3f}, Confidence: {reading.confidence:.3f}")
        
        print("✅ Market regime detection integration working")
        
    except Exception as e:
        print(f"❌ Market regime detection integration failed: {e}")
        return False
    
    # Test 3: Test pattern recognition integration
    try:
        # Create anomaly engine (pattern recognition)
        anomaly_engine = AnomalyIntelligenceEngine(instrument_meta)
        
        # Test pattern detection
        for market_data in test_data:
            reading = asyncio.run(anomaly_engine.update(market_data))
            print(f"✅ Anomaly analysis: {reading.dimension} - Signal: {reading.signal_strength:.3f}, Confidence: {reading.confidence:.3f}")
        
        print("✅ Pattern recognition integration working")
        
    except Exception as e:
        print(f"❌ Pattern recognition integration failed: {e}")
        return False
    
    print("\n🎉 ALL TESTS PASSED - Sensory integration successful!")
    return True


if __name__ == "__main__":
    success = test_simple_integration()
    sys.exit(0 if success else 1)
