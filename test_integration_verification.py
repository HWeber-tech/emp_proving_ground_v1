#!/usr/bin/env python3
"""
Integration Verification Test

This script verifies that all production components integrate correctly.
"""

import sys
import os
import time

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test that all production components can be imported"""
    print("🔍 Testing Production Component Imports...")
    
    try:
        # Test production validator
        from src.sensory.core.production_validator import ProductionValidator, ProductionError
        print("  ✅ Production validator imported successfully")
        
        # Test real data providers
        from src.sensory.core.real_data_providers import (
            DataIntegrationOrchestrator, 
            DataProviderError,
            RealFREDDataProvider
        )
        print("  ✅ Real data providers imported successfully")
        
        # Test streaming pipeline
        from src.sensory.infrastructure.streaming_pipeline import StreamingPipeline, StreamType
        print("  ✅ Streaming pipeline imported successfully")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Import test failed: {e}")
        return False

def test_production_validator_functionality():
    """Test production validator functionality"""
    print("🔍 Testing Production Validator Functionality...")
    
    try:
        from src.sensory.core.production_validator import ProductionValidator
        
        validator = ProductionValidator(strict_mode=False)  # Don't enforce in test
        
        # Test simulation detection
        def test_simulation_function():
            import random
            return random.uniform(0, 1)
        
        violations = validator.validate_function(test_simulation_function)
        
        if len(violations) > 0:
            print(f"  ✅ Simulation detection working - found {len(violations)} violations")
        else:
            print("  ❌ Simulation detection not working")
            return False
        
        return True
        
    except Exception as e:
        print(f"  ❌ Production validator test failed: {e}")
        return False

def test_real_data_providers_functionality():
    """Test real data providers functionality"""
    print("🔍 Testing Real Data Providers Functionality...")
    
    try:
        from src.sensory.core.real_data_providers import DataIntegrationOrchestrator, DataProviderError
        
        # Test configuration validation
        config = {
            'fred_api_key': 'demo',  # This should be rejected
            'exchange_api_key': 'test',
            'price_data_api_key': 'fake',
            'news_api_key': 'mock'
        }
        
        try:
            orchestrator = DataIntegrationOrchestrator(config)
            print("  ❌ Configuration validation failed - should have rejected demo keys")
            return False
        except DataProviderError:
            print("  ✅ Configuration validation working - correctly rejected demo keys")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Real data providers test failed: {e}")
        return False

def test_streaming_pipeline_functionality():
    """Test streaming pipeline functionality"""
    print("🔍 Testing Streaming Pipeline Functionality...")
    
    try:
        from src.sensory.infrastructure.streaming_pipeline import StreamingPipeline, StreamType
        
        config = {
            'kafka_bootstrap_servers': ['localhost:9092'],
            'redis_url': 'redis://localhost:6379'
        }
        
        pipeline = StreamingPipeline(config)
        print("  ✅ Streaming pipeline created successfully")
        
        # Test stream types
        stream_types = [StreamType.MARKET_DATA, StreamType.ORDER_FLOW, 
                       StreamType.ECONOMIC_DATA, StreamType.NEWS_EVENTS]
        print(f"  ✅ Stream types defined: {len(stream_types)} types")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Streaming pipeline test failed: {e}")
        return False

def test_integration_with_existing_system():
    """Test integration with existing sensory system"""
    print("🔍 Testing Integration with Existing System...")
    
    try:
        # Test that existing sensory system still works
        from src.sensory import SensoryCortex
        print("  ✅ Existing SensoryCortex still accessible")
        
        # Test that production components don't break existing imports
        from src.sensory.core.base import MarketData, DimensionalReading
        print("  ✅ Existing core components still accessible")
        
        # Test dimensional engines
        from src.sensory.dimensions.enhanced_why_dimension import EnhancedFundamentalIntelligenceEngine
        from src.sensory.dimensions.enhanced_how_dimension import InstitutionalMechanicsEngine
        print("  ✅ Enhanced dimensional engines still accessible")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Integration test failed: {e}")
        return False

def run_integration_verification():
    """Run complete integration verification"""
    print("🚀 Production System Integration Verification")
    print("=" * 60)
    
    tests = [
        ("Component Imports", test_imports),
        ("Production Validator", test_production_validator_functionality),
        ("Real Data Providers", test_real_data_providers_functionality),
        ("Streaming Pipeline", test_streaming_pipeline_functionality),
        ("Existing System Integration", test_integration_with_existing_system)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}")
        print("-" * 40)
        
        try:
            if test_func():
                passed += 1
                print(f"  ✅ {test_name} PASSED")
            else:
                failed += 1
                print(f"  ❌ {test_name} FAILED")
        except Exception as e:
            failed += 1
            print(f"  ❌ {test_name} ERROR: {e}")
    
    print("\n" + "=" * 60)
    print(f"📊 VERIFICATION RESULTS: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 ALL TESTS PASSED - INTEGRATION SUCCESSFUL")
        print("✅ Production system is fully integrated and ready")
        return True
    else:
        print("⚠️  SOME TESTS FAILED - INTEGRATION ISSUES DETECTED")
        print("❌ Address issues before proceeding")
        return False

if __name__ == "__main__":
    success = run_integration_verification()
    sys.exit(0 if success else 1) 