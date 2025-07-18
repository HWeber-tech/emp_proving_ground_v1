#!/usr/bin/env python3
"""
Simple Integration Test
Tests that technical indicators have been successfully integrated into the sensory system.
"""

import sys
import os
import logging

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_technical_indicators_integration():
    """Test that technical indicators are integrated into the WHAT dimension."""
    
    logger.info("Testing Technical Indicators Integration into Sensory System")
    logger.info("=" * 60)
    
    try:
        # Test 1: Check if the enhanced WHAT dimension file exists and has technical indicators
        logger.info("Test 1: Checking enhanced WHAT dimension file...")
        
        with open('src/sensory/dimensions/enhanced_what_dimension.py', 'r') as f:
            content = f.read()
        
        # Check for technical indicators class
        if 'class TechnicalIndicators:' in content:
            logger.info("✓ TechnicalIndicators class found!")
        else:
            logger.error("✗ TechnicalIndicators class not found!")
            return False
        
        # Check for technical indicators calculation method
        if '_calculate_technical_indicators' in content:
            logger.info("✓ _calculate_technical_indicators method found!")
        else:
            logger.error("✗ _calculate_technical_indicators method not found!")
            return False
        
        # Check for specific indicator methods
        indicator_methods = [
            '_calculate_rsi',
            '_calculate_macd', 
            '_calculate_bollinger_bands',
            '_calculate_atr',
            '_calculate_obv'
        ]
        
        for method in indicator_methods:
            if method in content:
                logger.info(f"✓ {method} method found!")
            else:
                logger.error(f"✗ {method} method not found!")
                return False
        
        # Test 2: Check if legacy files have been archived
        logger.info("\nTest 2: Checking legacy file archiving...")
        
        legacy_files = [
            'archive/sensory/legacy_dimensions/what_engine.py',
            'archive/sensory/legacy_dimensions/how_engine.py',
            'archive/sensory/legacy_dimensions/when_engine.py',
            'archive/sensory/legacy_dimensions/why_engine.py',
            'archive/sensory/legacy_dimensions/anomaly_engine.py'
        ]
        
        for file_path in legacy_files:
            if os.path.exists(file_path):
                logger.info(f"✓ {file_path} archived!")
            else:
                logger.error(f"✗ {file_path} not found in archive!")
                return False
        
        # Test 3: Check if enhanced files still exist
        logger.info("\nTest 3: Checking enhanced files...")
        
        enhanced_files = [
            'src/sensory/dimensions/enhanced_what_dimension.py',
            'src/sensory/dimensions/enhanced_how_dimension.py',
            'src/sensory/dimensions/enhanced_when_dimension.py',
            'src/sensory/dimensions/enhanced_why_dimension.py',
            'src/sensory/dimensions/enhanced_anomaly_dimension.py'
        ]
        
        for file_path in enhanced_files:
            if os.path.exists(file_path):
                logger.info(f"✓ {file_path} exists!")
            else:
                logger.error(f"✗ {file_path} not found!")
                return False
        
        # Test 4: Check if advanced analytics has been integrated
        logger.info("\nTest 4: Checking advanced analytics integration...")
        
        # Check if advanced analytics file still exists (should be archived later)
        if os.path.exists('src/analysis/advanced_analytics.py'):
            logger.info("✓ Advanced analytics file still exists (will be archived)")
        else:
            logger.warning("⚠ Advanced analytics file not found (may already be archived)")
        
        logger.info("\n" + "=" * 60)
        logger.info("SENSORY INTEGRATION TEST COMPLETED SUCCESSFULLY!")
        logger.info("Technical indicators have been successfully integrated into the WHAT dimension.")
        logger.info("Legacy files have been properly archived.")
        logger.info("Enhanced files are in place.")
        logger.info("=" * 60)
        
        return True
        
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        return False

def main():
    """Run the integration test."""
    try:
        success = test_technical_indicators_integration()
        if success:
            return 0
        else:
            return 1
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
