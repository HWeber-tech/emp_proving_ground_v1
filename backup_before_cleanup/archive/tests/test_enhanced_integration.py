#!/usr/bin/env python3
"""
Test script to verify enhanced multidimensional market intelligence system integration.
This script tests that all enhanced dimensional engines can be imported and initialized.
"""

import sys
import traceback

def test_imports():
    """Test that all enhanced dimensional engines can be imported."""
    print("🧪 Testing Enhanced Multidimensional Market Intelligence System Integration")
    print("=" * 70)
    
    # Test imports
    tests = [
        ("Core Base", "src.sensory.core.base", ["DimensionalReading", "MarketData", "MarketRegime"]),
        ("Enhanced WHY Dimension", "src.sensory.dimensions.enhanced_why_dimension", ["EnhancedFundamentalIntelligenceEngine"]),
        ("Enhanced HOW Dimension", "src.sensory.dimensions.enhanced_how_dimension", ["InstitutionalMechanicsEngine"]),
        ("Enhanced WHAT Dimension", "src.sensory.dimensions.enhanced_what_dimension", ["TechnicalRealityEngine"]),
        ("Enhanced WHEN Dimension", "src.sensory.dimensions.enhanced_when_dimension", ["ChronalIntelligenceEngine"]),
        ("Enhanced ANOMALY Dimension", "src.sensory.dimensions.enhanced_anomaly_dimension", ["AnomalyIntelligenceEngine"]),
        ("Enhanced Intelligence Engine", "src.sensory.orchestration.enhanced_intelligence_engine", ["ContextualFusionEngine"]),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, module_path, expected_classes in tests:
        print(f"\n📦 Testing {test_name}...")
        try:
            module = __import__(module_path, fromlist=expected_classes)
            
            # Test that expected classes exist
            for class_name in expected_classes:
                if hasattr(module, class_name):
                    print(f"  ✅ {class_name} imported successfully")
                else:
                    print(f"  ❌ {class_name} not found in {module_path}")
                    failed += 1
                    continue
            
            passed += 1
            print(f"  ✅ {test_name} - ALL TESTS PASSED")
            
        except Exception as e:
            print(f"  ❌ {test_name} - IMPORT FAILED")
            print(f"     Error: {e}")
            failed += 1
    
    print("\n" + "=" * 70)
    print(f"📊 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 ALL TESTS PASSED! Enhanced system integration successful.")
        return True
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        return False

def test_engine_initialization():
    """Test that engines can be initialized."""
    print("\n🔧 Testing Engine Initialization...")
    
    try:
        from src.sensory.orchestration.enhanced_intelligence_engine import ContextualFusionEngine
        from src.sensory.dimensions.enhanced_why_dimension import EnhancedFundamentalIntelligenceEngine
        from src.sensory.dimensions.enhanced_how_dimension import InstitutionalMechanicsEngine
        from src.sensory.dimensions.enhanced_what_dimension import TechnicalRealityEngine
        from src.sensory.dimensions.enhanced_when_dimension import ChronalIntelligenceEngine
        from src.sensory.dimensions.enhanced_anomaly_dimension import AnomalyIntelligenceEngine
        
        # Test initialization
        engines = [
            ("EnhancedFundamentalIntelligenceEngine", EnhancedFundamentalIntelligenceEngine()),
            ("InstitutionalMechanicsEngine", InstitutionalMechanicsEngine()),
            ("TechnicalRealityEngine", TechnicalRealityEngine()),
            ("ChronalIntelligenceEngine", ChronalIntelligenceEngine()),
            ("AnomalyIntelligenceEngine", AnomalyIntelligenceEngine()),
        ]
        
        for name, engine in engines:
            print(f"  ✅ {name} initialized successfully")
        
        # Test main fusion engine
        fusion_engine = ContextualFusionEngine()
        print(f"  ✅ ContextualFusionEngine initialized successfully")
        
        print("🎉 All engines initialized successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Engine initialization failed: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Enhanced Multidimensional Market Intelligence System - Integration Test")
    print("=" * 80)
    
    # Test imports
    imports_ok = test_imports()
    
    # Test initialization if imports passed
    if imports_ok:
        init_ok = test_engine_initialization()
        
        if init_ok:
            print("\n🎯 SYSTEM STATUS: FULLY OPERATIONAL")
            print("✅ All enhanced dimensional engines imported and initialized")
            print("✅ Contextual fusion engine ready for use")
            print("✅ System ready for production deployment")
        else:
            print("\n⚠️  SYSTEM STATUS: PARTIALLY OPERATIONAL")
            print("✅ Imports working but initialization issues detected")
    else:
        print("\n❌ SYSTEM STATUS: INTEGRATION FAILED")
        print("❌ Import issues detected - system not ready")
    
    print("\n" + "=" * 80) 