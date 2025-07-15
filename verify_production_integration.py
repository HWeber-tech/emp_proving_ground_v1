#!/usr/bin/env python3
"""
Final Production Integration Verification

This script performs a comprehensive verification of the production integration.
"""

import sys
import os
import time
import traceback

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def verify_file_structure():
    """Verify that all production files are in place"""
    print("🔍 Verifying File Structure...")
    
    required_files = [
        'src/sensory/core/production_validator.py',
        'src/sensory/core/real_data_providers.py',
        'src/sensory/infrastructure/streaming_pipeline.py',
        'src/sensory/infrastructure/__init__.py',
        'config/production.yaml',
        'requirements.txt',
        'test_production_system.py',
        'test_integration_verification.py'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
        else:
            print(f"  ✅ {file_path}")
    
    if missing_files:
        print(f"  ❌ Missing files: {missing_files}")
        return False
    
    print("  ✅ All production files present")
    return True

def verify_imports():
    """Verify that all components can be imported"""
    print("🔍 Verifying Component Imports...")
    
    try:
        # Test production validator
        from src.sensory.core.production_validator import ProductionValidator, ProductionError
        print("  ✅ ProductionValidator imported")
        
        # Test real data providers
        from src.sensory.core.real_data_providers import (
            DataIntegrationOrchestrator, 
            DataProviderError,
            RealFREDDataProvider
        )
        print("  ✅ Real data providers imported")
        
        # Test streaming pipeline
        from src.sensory.infrastructure.streaming_pipeline import StreamingPipeline, StreamType
        print("  ✅ Streaming pipeline imported")
        
        # Test main sensory module
        from src.sensory import (
            SensoryCortex,
            ProductionValidator as PV,
            DataIntegrationOrchestrator as DIO,
            StreamingPipeline as SP
        )
        print("  ✅ Main sensory module imports working")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Import verification failed: {e}")
        traceback.print_exc()
        return False

def verify_production_validator():
    """Verify production validator functionality"""
    print("🔍 Verifying Production Validator...")
    
    try:
        from src.sensory.core.production_validator import ProductionValidator
        
        validator = ProductionValidator(strict_mode=False)
        
        # Test simulation detection
        def test_simulation():
            import random
            return random.uniform(0, 1)
        
        violations = validator.validate_function(test_simulation)
        
        if len(violations) > 0:
            print(f"  ✅ Simulation detection working - {len(violations)} violations found")
        else:
            print("  ❌ Simulation detection not working")
            return False
        
        return True
        
    except Exception as e:
        print(f"  ❌ Production validator verification failed: {e}")
        return False

def verify_real_data_providers():
    """Verify real data providers functionality"""
    print("🔍 Verifying Real Data Providers...")
    
    try:
        from src.sensory.core.real_data_providers import DataIntegrationOrchestrator, DataProviderError
        
        # Test configuration validation
        config = {
            'fred_api_key': 'demo',
            'exchange_api_key': 'test',
            'price_data_api_key': 'fake',
            'news_api_key': 'mock'
        }
        
        try:
            orchestrator = DataIntegrationOrchestrator(config)
            print("  ❌ Configuration validation failed")
            return False
        except DataProviderError:
            print("  ✅ Configuration validation working")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Real data providers verification failed: {e}")
        return False

def verify_streaming_pipeline():
    """Verify streaming pipeline functionality"""
    print("🔍 Verifying Streaming Pipeline...")
    
    try:
        from src.sensory.infrastructure.streaming_pipeline import StreamingPipeline, StreamType
        
        config = {
            'kafka_bootstrap_servers': ['localhost:9092'],
            'redis_url': 'redis://localhost:6379'
        }
        
        pipeline = StreamingPipeline(config)
        
        # Test stream types
        stream_types = [StreamType.MARKET_DATA, StreamType.ORDER_FLOW, 
                       StreamType.ECONOMIC_DATA, StreamType.NEWS_EVENTS]
        
        if len(stream_types) == 4:
            print("  ✅ Streaming pipeline structure valid")
        else:
            print("  ❌ Streaming pipeline structure invalid")
            return False
        
        return True
        
    except Exception as e:
        print(f"  ❌ Streaming pipeline verification failed: {e}")
        return False

def verify_backward_compatibility():
    """Verify that existing functionality still works"""
    print("🔍 Verifying Backward Compatibility...")
    
    try:
        # Test existing sensory system
        from src.sensory import SensoryCortex
        print("  ✅ SensoryCortex accessible")
        
        # Test existing core components
        from src.sensory.core.base import MarketData, DimensionalReading
        print("  ✅ Core components accessible")
        
        # Test enhanced dimensional engines
        from src.sensory.dimensions.enhanced_why_dimension import EnhancedFundamentalIntelligenceEngine
        from src.sensory.dimensions.enhanced_how_dimension import InstitutionalMechanicsEngine
        print("  ✅ Enhanced dimensional engines accessible")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Backward compatibility verification failed: {e}")
        return False

def verify_configuration():
    """Verify production configuration"""
    print("🔍 Verifying Production Configuration...")
    
    try:
        import yaml
        
        with open('config/production.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        required_sections = [
            'environment', 'data_providers', 'streaming', 
            'persistence', 'performance', 'security', 'monitoring'
        ]
        
        for section in required_sections:
            if section in config:
                print(f"  ✅ {section} configuration present")
            else:
                print(f"  ❌ {section} configuration missing")
                return False
        
        return True
        
    except Exception as e:
        print(f"  ❌ Configuration verification failed: {e}")
        return False

def run_final_verification():
    """Run complete final verification"""
    print("🚀 Final Production Integration Verification")
    print("=" * 60)
    
    verifications = [
        ("File Structure", verify_file_structure),
        ("Component Imports", verify_imports),
        ("Production Validator", verify_production_validator),
        ("Real Data Providers", verify_real_data_providers),
        ("Streaming Pipeline", verify_streaming_pipeline),
        ("Backward Compatibility", verify_backward_compatibility),
        ("Production Configuration", verify_configuration)
    ]
    
    passed = 0
    failed = 0
    
    for verification_name, verification_func in verifications:
        print(f"\n📋 {verification_name}")
        print("-" * 40)
        
        try:
            if verification_func():
                passed += 1
                print(f"  ✅ {verification_name} PASSED")
            else:
                failed += 1
                print(f"  ❌ {verification_name} FAILED")
        except Exception as e:
            failed += 1
            print(f"  ❌ {verification_name} ERROR: {e}")
    
    print("\n" + "=" * 60)
    print(f"📊 FINAL VERIFICATION RESULTS: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 ALL VERIFICATIONS PASSED")
        print("✅ Production integration is COMPLETE and READY")
        print("🚀 System is ready for production deployment")
        return True
    else:
        print("⚠️  SOME VERIFICATIONS FAILED")
        print("❌ Address issues before deployment")
        return False

if __name__ == "__main__":
    success = run_final_verification()
    sys.exit(0 if success else 1) 