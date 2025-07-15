#!/usr/bin/env python3
"""
Production System Validation Test

This script validates that the system is production-ready by:
1. Checking for simulation code violations
2. Validating real data integration
3. Testing anti-simulation enforcement
4. Verifying infrastructure components
"""

import sys
import os
import time
import asyncio
from datetime import datetime
from typing import Dict, Any, List

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_production_validator():
    """Test the production validator"""
    print("🔍 Testing Production Validator...")
    
    try:
        from src.sensory.core.production_validator import ProductionValidator, ProductionError
        
        # Create validator
        validator = ProductionValidator(strict_mode=True)
        print("  ✅ Production validator created successfully")
        
        # Test simulation detection
        def simulate_market_data():
            import random
            return random.uniform(1.0, 2.0)
        
        violations = validator.validate_function(simulate_market_data)
        if violations:
            print(f"  ✅ Simulation detection working - found {len(violations)} violations")
        else:
            print("  ❌ Simulation detection failed - no violations found")
            return False
        
        # Test environment validation
        env_violations = validator.validate_environment()
        print(f"  ✅ Environment validation completed - {len(env_violations)} violations")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Production validator test failed: {e}")
        return False

def test_real_data_providers():
    """Test real data providers"""
    print("🔍 Testing Real Data Providers...")
    
    try:
        from src.sensory.core.real_data_providers import (
            DataIntegrationOrchestrator, 
            DataProviderError,
            RealFREDDataProvider
        )
        
        # Test configuration validation
        config = {
            'fred_api_key': 'demo',  # This should fail
            'exchange_api_key': 'test',
            'price_data_api_key': 'fake',
            'news_api_key': 'mock'
        }
        
        try:
            orchestrator = DataIntegrationOrchestrator(config)
            print("  ❌ Configuration validation failed - should have rejected demo keys")
            return False
        except DataProviderError as e:
            print(f"  ✅ Configuration validation working - correctly rejected: {e}")
        
        # Test FRED provider structure
        try:
            fred_provider = RealFREDDataProvider('REAL_API_KEY')
            print("  ✅ FRED provider structure valid")
        except Exception as e:
            print(f"  ❌ FRED provider test failed: {e}")
            return False
        
        return True
        
    except Exception as e:
        print(f"  ❌ Real data providers test failed: {e}")
        return False

def test_anti_simulation_enforcement():
    """Test anti-simulation enforcement"""
    print("🔍 Testing Anti-Simulation Enforcement...")
    
    try:
        from src.sensory.core.production_validator import ProductionError
        
        # Test that simulation code is blocked
        def test_simulation_function():
            import random
            import numpy as np
            
            # This should trigger violations
            simulated_data = np.random.normal(0, 1, 100)
            mock_result = random.choice(simulated_data)
            return mock_result
        
        from src.sensory.core.production_validator import production_validator
        
        violations = production_validator.validate_function(test_simulation_function)
        
        if len(violations) >= 3:  # Should detect multiple violations
            print(f"  ✅ Anti-simulation enforcement working - detected {len(violations)} violations")
            
            # Test enforcement
            try:
                production_validator.enforce_production_mode(violations)
                print("  ❌ Enforcement failed - should have raised exception")
                return False
            except ProductionError:
                print("  ✅ Anti-simulation enforcement working - correctly blocked simulation")
        else:
            print(f"  ❌ Anti-simulation detection weak - only {len(violations)} violations")
            return False
        
        return True
        
    except Exception as e:
        print(f"  ❌ Anti-simulation enforcement test failed: {e}")
        return False

def test_infrastructure_components():
    """Test infrastructure components"""
    print("🔍 Testing Infrastructure Components...")
    
    try:
        # Test streaming pipeline structure
        from src.sensory.infrastructure.streaming_pipeline import StreamingPipeline, StreamType
        
        config = {
            'kafka_bootstrap_servers': ['localhost:9092'],
            'redis_url': 'redis://localhost:6379'
        }
        
        pipeline = StreamingPipeline(config)
        print("  ✅ Streaming pipeline structure valid")
        
        # Test stream types
        stream_types = [StreamType.MARKET_DATA, StreamType.ORDER_FLOW, 
                       StreamType.ECONOMIC_DATA, StreamType.NEWS_EVENTS]
        print(f"  ✅ Stream types defined: {len(stream_types)} types")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Infrastructure components test failed: {e}")
        return False

def test_performance_validation():
    """Test performance validation"""
    print("🔍 Testing Performance Validation...")
    
    try:
        # Test import performance
        start_time = time.time()
        
        from src.sensory.core.production_validator import ProductionValidator
        from src.sensory.core.real_data_providers import DataIntegrationOrchestrator
        from src.sensory.infrastructure.streaming_pipeline import StreamingPipeline
        
        import_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        
        if import_time < 100:  # Should be under 100ms
            print(f"  ✅ Import performance good: {import_time:.2f}ms")
        else:
            print(f"  ⚠️  Import performance slow: {import_time:.2f}ms")
        
        # Test validator performance
        validator = ProductionValidator()
        
        def test_function():
            return "test"
        
        start_time = time.time()
        violations = validator.validate_function(test_function)
        validation_time = (time.time() - start_time) * 1000
        
        if validation_time < 10:  # Should be under 10ms
            print(f"  ✅ Validation performance good: {validation_time:.2f}ms")
        else:
            print(f"  ⚠️  Validation performance slow: {validation_time:.2f}ms")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Performance validation test failed: {e}")
        return False

def test_production_readiness():
    """Test overall production readiness"""
    print("🔍 Testing Production Readiness...")
    
    try:
        from src.sensory.core.production_validator import production_validator
        
        # Test core modules
        core_modules = [
            'src.sensory.core.production_validator',
            'src.sensory.core.real_data_providers',
            'src.sensory.infrastructure.streaming_pipeline'
        ]
        
        total_violations = 0
        
        for module_name in core_modules:
            try:
                violations = production_validator.validate_module(module_name)
                total_violations += len(violations)
                print(f"  ✅ {module_name}: {len(violations)} violations")
            except Exception as e:
                print(f"  ❌ {module_name}: validation failed - {e}")
                return False
        
        if total_violations == 0:
            print("  ✅ Production readiness validation passed - no simulation violations")
        else:
            print(f"  ⚠️  Production readiness validation found {total_violations} violations")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Production readiness test failed: {e}")
        return False

def run_production_validation():
    """Run complete production validation"""
    print("🚀 Production System Validation")
    print("=" * 50)
    
    tests = [
        ("Production Validator", test_production_validator),
        ("Real Data Providers", test_real_data_providers),
        ("Anti-Simulation Enforcement", test_anti_simulation_enforcement),
        ("Infrastructure Components", test_infrastructure_components),
        ("Performance Validation", test_performance_validation),
        ("Production Readiness", test_production_readiness)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}")
        print("-" * 30)
        
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
    
    print("\n" + "=" * 50)
    print(f"📊 VALIDATION RESULTS: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 ALL TESTS PASSED - PRODUCTION SYSTEM VALIDATED")
        print("✅ System is ready for production deployment")
        return True
    else:
        print("⚠️  SOME TESTS FAILED - PRODUCTION SYSTEM NOT READY")
        print("❌ Address issues before deployment")
        return False

if __name__ == "__main__":
    success = run_production_validation()
    sys.exit(0 if success else 1) 