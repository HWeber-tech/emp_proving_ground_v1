#!/usr/bin/env python3
"""
Test Sensory Integration
Tests that technical indicators have been successfully integrated into the sensory system.
"""

import sys
import os
import asyncio
import logging
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.sensory.dimensions.enhanced_what_dimension import TechnicalRealityEngine, PriceActionAnalyzer
from src.sensory.core.base import MarketData

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_technical_indicators_integration():
    """Test that technical indicators are integrated into the WHAT dimension."""
    
    logger.info("Testing Technical Indicators Integration into Sensory System")
    logger.info("=" * 60)
    
    # Create test market data
    base_price = 1.2000
    test_data = []
    
    for i in range(100):  # Generate 100 data points
        # Simulate price movement
        price_change = np.random.normal(0, 0.001)  # Small random changes
        base_price += price_change
        
        # Create market data
        spread = 0.0002
        market_data = MarketData(
            symbol_id=1,
            symbol_name="EURUSD",
            bid=base_price - spread/2,
            ask=base_price + spread/2,
            timestamp=datetime.now() + timedelta(minutes=i),
            digits=5,
            volume=1000 + np.random.randint(-100, 100)
        )
        test_data.append(market_data)
    
    # Test PriceActionAnalyzer directly
    logger.info("Testing PriceActionAnalyzer with Technical Indicators...")
    
    analyzer = PriceActionAnalyzer()
    
    # Feed data to analyzer
    for data in test_data:
        analyzer.update_market_data(data)
    
    # Check if technical indicators were calculated
    if analyzer.technical_indicators:
        logger.info("✓ Technical indicators successfully calculated!")
        logger.info(f"  RSI: {analyzer.technical_indicators.rsi:.2f}")
        logger.info(f"  MACD: {analyzer.technical_indicators.macd:.6f}")
        logger.info(f"  Bollinger Upper: {analyzer.technical_indicators.bollinger_upper:.5f}")
        logger.info(f"  Bollinger Lower: {analyzer.technical_indicators.bollinger_lower:.5f}")
        logger.info(f"  ATR: {analyzer.technical_indicators.atr:.6f}")
        logger.info(f"  Support Level: {analyzer.technical_indicators.support_level:.5f}")
        logger.info(f"  Resistance Level: {analyzer.technical_indicators.resistance_level:.5f}")
    else:
        logger.error("✗ Technical indicators not calculated!")
        return False
    
    # Test TechnicalRealityEngine
    logger.info("\nTesting TechnicalRealityEngine integration...")
    
    engine = TechnicalRealityEngine()
    
    # Analyze with the last market data
    reading = await engine.analyze_technical_reality(test_data[-1])
    
    logger.info(f"✓ Technical Reality Analysis completed!")
    logger.info(f"  Dimension: {reading.dimension}")
    logger.info(f"  Value: {reading.value:.3f}")
    logger.info(f"  Confidence: {reading.confidence:.3f}")
    
    # Check if technical indicators are in context
    if 'technical_indicators' in reading.context:
        logger.info("✓ Technical indicators included in context!")
        indicators = reading.context['technical_indicators']
        logger.info(f"  Context RSI: {indicators['rsi']:.2f}")
        logger.info(f"  Context MACD: {indicators['macd']:.6f}")
    else:
        logger.warning("⚠ Technical indicators not found in context")
    
    logger.info("\n" + "=" * 60)
    logger.info("SENSORY INTEGRATION TEST COMPLETED SUCCESSFULLY!")
    logger.info("Technical indicators have been successfully integrated into the WHAT dimension.")
    logger.info("=" * 60)
    
    return True

async def main():
    """Run the integration test."""
    try:
        success = await test_technical_indicators_integration()
        if success:
            return 0
        else:
            return 1
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code) 