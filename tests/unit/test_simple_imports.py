#!/usr/bin/env python3
"""
Simple Import Test

This script tests that all components can be imported without errors.
"""

import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_basic_imports():
    """Test basic imports"""
    print("Testing basic imports...")
    
    try:
        # Test existing components
        from src.sensory import SensoryCortex
        print("✅ SensoryCortex imported")
        
        from src.sensory.core.base import MarketData, DimensionalReading
        print("✅ Core base components imported")
        
        # Test production components
        from src.sensory.core.production_validator import ProductionValidator
        print("✅ ProductionValidator imported")
        
        from src.sensory.core.real_data_providers import DataIntegrationOrchestrator
        print("✅ DataIntegrationOrchestrator imported")
        
        from src.sensory.infrastructure.streaming_pipeline import StreamingPipeline
        print("✅ StreamingPipeline imported")
        
        return True
        
    except Exception as e:
        print(f"❌ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_sensory_imports():
    """Test imports from main sensory module"""
    print("Testing sensory module imports...")
    
    try:
        from src.sensory import (
            SensoryCortex,
            ProductionValidator,
            DataIntegrationOrchestrator,
            StreamingPipeline
        )
        print("✅ All components imported from main sensory module")
        return True
        
    except Exception as e:
        print(f"❌ Sensory import failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Simple Import Test")
    print("=" * 40)
    
    success1 = test_basic_imports()
    print()
    success2 = test_sensory_imports()
    
    print("\n" + "=" * 40)
    if success1 and success2:
        print("🎉 ALL IMPORTS SUCCESSFUL")
    else:
        print("❌ SOME IMPORTS FAILED")
    
    sys.exit(0 if (success1 and success2) else 1) 
