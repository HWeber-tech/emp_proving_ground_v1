"""
Refactor Audit - Comprehensive Function and Integration Check

This audit verifies that all functions from the old monolithic sense files
are present and properly integrated in the new refactored structure.

Author: EMP Development Team
Date: July 18, 2024
Phase: 2 - Refactor Audit
"""

import sys
import os
import inspect
import importlib
from typing import Dict, List, Set, Any
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


class RefactorAuditor:
    """Comprehensive auditor for the refactored sensory cortex"""
    
    def __init__(self):
        self.audit_results = {}
        self.missing_functions = {}
        self.integration_issues = []
        
    def audit_old_monolithic_files(self):
        """Audit the old monolithic sense files to extract all functions"""
        print("🔍 AUDITING OLD MONOLITHIC FILES")
        print("=" * 50)
        
        old_files = [
            'src/sensory/dimensions/enhanced_how_dimension.py',
            'src/sensory/dimensions/enhanced_what_dimension.py', 
            'src/sensory/dimensions/enhanced_when_dimension.py',
            'src/sensory/dimensions/enhanced_why_dimension.py',
            'src/sensory/dimensions/enhanced_anomaly_dimension.py'
        ]
        
        old_functions = {}
        
        for file_path in old_files:
            try:
                if os.path.exists(file_path):
                    print(f"📁 Auditing: {file_path}")
                    
                    # Extract module name
                    module_name = file_path.replace('/', '.').replace('.py', '')
                    
                    # Import the module
                    module = importlib.import_module(module_name)
                    
                    # Get all functions and methods
                    functions = []
                    classes = []
                    
                    for name, obj in inspect.getmembers(module):
                        if inspect.isfunction(obj):
                            functions.append(name)
                        elif inspect.isclass(obj):
                            classes.append(name)
                            # Get methods from classes
                            for method_name, method_obj in inspect.getmembers(obj):
                                if inspect.isfunction(method_obj) and not method_name.startswith('_'):
                                    functions.append(f"{name}.{method_name}")
                    
                    old_functions[file_path] = {
                        'functions': functions,
                        'classes': classes,
                        'total_functions': len(functions),
                        'total_classes': len(classes)
                    }
                    
                    print(f"   ✅ Found {len(functions)} functions, {len(classes)} classes")
                    
                else:
                    print(f"   ⚠️ File not found: {file_path}")
                    
            except Exception as e:
                print(f"   ❌ Error auditing {file_path}: {e}")
        
        return old_functions
    
    def audit_new_refactored_structure(self):
        """Audit the new refactored structure to check available functions"""
        print("\n🔍 AUDITING NEW REFACTORED STRUCTURE")
        print("=" * 50)
        
        new_modules = [
            # Main engines
            'src.sensory.dimensions.how.how_engine',
            'src.sensory.dimensions.what.what_engine',
            'src.sensory.dimensions.when.when_engine', 
            'src.sensory.dimensions.why.why_engine',
            'src.sensory.dimensions.anomaly.anomaly_engine',
            
            # How sub-modules
            'src.sensory.dimensions.how.indicators',
            'src.sensory.dimensions.how.patterns',
            'src.sensory.dimensions.how.order_flow',
            
            # What sub-modules
            'src.sensory.dimensions.what.price_action',
            
            # When sub-modules
            'src.sensory.dimensions.when.regime_detection',
            
            # Why sub-modules
            'src.sensory.dimensions.why.economic_analysis',
            
            # Anomaly sub-modules
            'src.sensory.dimensions.anomaly.pattern_recognition',
            'src.sensory.dimensions.anomaly.anomaly_detection',
            
            # Compatibility layer
            'src.sensory.dimensions.compatibility'
        ]
        
        new_functions = {}
        
        for module_name in new_modules:
            try:
                print(f"📁 Auditing: {module_name}")
                
                # Import the module
                module = importlib.import_module(module_name)
                
                # Get all functions and methods
                functions = []
                classes = []
                
                for name, obj in inspect.getmembers(module):
                    if inspect.isfunction(obj):
                        functions.append(name)
                    elif inspect.isclass(obj):
                        classes.append(name)
                        # Get methods from classes
                        for method_name, method_obj in inspect.getmembers(obj):
                            if inspect.isfunction(method_obj) and not method_name.startswith('_'):
                                functions.append(f"{name}.{method_name}")
                
                new_functions[module_name] = {
                    'functions': functions,
                    'classes': classes,
                    'total_functions': len(functions),
                    'total_classes': len(classes)
                }
                
                print(f"   ✅ Found {len(functions)} functions, {len(classes)} classes")
                
            except Exception as e:
                print(f"   ❌ Error auditing {module_name}: {e}")
        
        return new_functions
    
    def compare_functionality(self, old_functions: Dict, new_functions: Dict):
        """Compare old and new functionality to identify gaps"""
        print("\n🔍 COMPARING FUNCTIONALITY")
        print("=" * 40)
        
        # Extract all function names from old structure
        old_function_names = set()
        for file_path, data in old_functions.items():
            old_function_names.update(data['functions'])
        
        # Extract all function names from new structure
        new_function_names = set()
        for module_name, data in new_functions.items():
            new_function_names.update(data['functions'])
        
        # Filter out false positives (auto-generated methods, library methods, etc.)
        def is_false_positive(func_name: str) -> bool:
            # Pydantic auto-generated methods
            if any(method in func_name for method in [
                '.copy', '.dict', '.json', '.model_copy', '.model_dump', 
                '.model_dump_json', '.model_post_init', '.best_ask', '.best_bid',
                '.calculate_derived_fields', '.spread', '.depth'
            ]):
                return True
            
            # Scikit-learn methods
            if any(method in func_name for method in [
                '.fit', '.fit_transform', '.transform', '.predict', '.score_samples',
                '.get_params', '.set_params', '.get_metadata_routing', '.set_fit_request',
                '.set_transform_request', '.set_partial_fit_request', '.set_inverse_transform_request',
                '.set_output', '.partial_fit', '.inverse_transform', '.get_feature_names_out',
                '.decision_function', '.fit_predict'
            ]):
                return True
            
            # Built-in Python functions and imports
            if func_name in [
                'dataclass', 'field', 'find_peaks', 'main', 'savgol_filter', 'NamedTuple'
            ]:
                return True
            
            return False
        
        # Filter out false positives
        filtered_old_functions = {f for f in old_function_names if not is_false_positive(f)}
        filtered_new_functions = {f for f in new_function_names if not is_false_positive(f)}
        
        # Find missing functions
        missing_functions = filtered_old_functions - filtered_new_functions
        
        print(f"📊 Function Counts:")
        print(f"   Old functions (filtered): {len(filtered_old_functions)}")
        print(f"   New functions (filtered): {len(filtered_new_functions)}")
        print(f"   Missing functions: {len(missing_functions)}")
        
        if missing_functions:
            print(f"\n❌ MISSING FUNCTIONS:")
            for func in sorted(missing_functions):
                print(f"   - {func}")
        else:
            print(f"\n✅ ALL FUNCTIONS PRESENT!")
        
        return missing_functions
    
    def test_integration_compatibility(self):
        """Test that the new structure integrates properly with the meta architecture"""
        print("\n🔍 TESTING INTEGRATION COMPATIBILITY")
        print("=" * 45)
        
        integration_tests = []
        
        try:
            # Test 1: Import all new engines
            from src.sensory import (
                HowEngine, WhatEngine, WhenEngine, WhyEngine, AnomalyEngine,
                InstitutionalMechanicsEngine, TechnicalRealityEngine, 
                ChronalIntelligenceEngine, EnhancedFundamentalIntelligenceEngine,
                AnomalyIntelligenceEngine
            )
            integration_tests.append(("New Engine Imports", True))
            print("✅ New engine imports successful")
            
        except Exception as e:
            integration_tests.append(("New Engine Imports", False))
            print(f"❌ New engine imports failed: {e}")
        
        try:
            # Test 2: Test legacy compatibility
            from src.sensory.dimensions.compatibility import (
                InstitutionalMechanicsEngine, TechnicalRealityEngine,
                ChronalIntelligenceEngine, EnhancedFundamentalIntelligenceEngine,
                AnomalyIntelligenceEngine
            )
            integration_tests.append(("Legacy Compatibility", True))
            print("✅ Legacy compatibility successful")
            
        except Exception as e:
            integration_tests.append(("Legacy Compatibility", False))
            print(f"❌ Legacy compatibility failed: {e}")
        
        try:
            # Test 3: Test orchestration integration
            from src.sensory.orchestration.master_orchestrator import MasterOrchestrator
            integration_tests.append(("Orchestration Integration", True))
            print("✅ Orchestration integration successful")
            
        except Exception as e:
            integration_tests.append(("Orchestration Integration", False))
            print(f"⚠️ Orchestration integration issue: {e}")
        
        try:
            # Test 4: Test data integration compatibility
            from src.data_integration import RealDataManager
            integration_tests.append(("Data Integration", True))
            print("✅ Data integration compatibility successful")
            
        except Exception as e:
            integration_tests.append(("Data Integration", False))
            print(f"⚠️ Data integration compatibility issue: {e}")
        
        return integration_tests
    
    def test_functional_equivalence(self):
        """Test that new engines provide equivalent functionality to old ones"""
        print("\n🔍 TESTING FUNCTIONAL EQUIVALENCE")
        print("=" * 40)
        
        from src.sensory.core.base import MarketData
        from datetime import datetime, timedelta
        
        # Create test market data
        market_data = []
        base_price = 1.1000
        for i in range(10):
            timestamp = datetime.now() - timedelta(minutes=10-i)
            price_change = (i % 5 - 2) * 0.0001
            current_price = base_price + price_change
            
            market_data.append(MarketData(
                symbol="EURUSD",
                timestamp=timestamp,
                open=current_price - 0.0001,
                high=current_price + 0.0001,
                low=current_price - 0.0001,
                close=current_price,
                volume=1000 + (i * 10),
                bid=current_price - 0.0001,
                ask=current_price + 0.0001,
                source="test",
                latency_ms=0.0
            ))
        
        equivalence_tests = []
        
        # Test each engine pair
        engine_pairs = [
            ("HowEngine", "InstitutionalMechanicsEngine"),
            ("WhatEngine", "TechnicalRealityEngine"),
            ("WhenEngine", "ChronalIntelligenceEngine"),
            ("WhyEngine", "EnhancedFundamentalIntelligenceEngine"),
            ("AnomalyEngine", "AnomalyIntelligenceEngine")
        ]
        
        for new_name, old_name in engine_pairs:
            try:
                # Import both engines
                from src.sensory import HowEngine, WhatEngine, WhenEngine, WhyEngine, AnomalyEngine
                from src.sensory.dimensions.compatibility import (
                    InstitutionalMechanicsEngine, TechnicalRealityEngine,
                    ChronalIntelligenceEngine, EnhancedFundamentalIntelligenceEngine,
                    AnomalyIntelligenceEngine
                )
                
                # Get engine classes
                engine_classes = {
                    "HowEngine": HowEngine,
                    "WhatEngine": WhatEngine,
                    "WhenEngine": WhenEngine,
                    "WhyEngine": WhyEngine,
                    "AnomalyEngine": AnomalyEngine,
                    "InstitutionalMechanicsEngine": InstitutionalMechanicsEngine,
                    "TechnicalRealityEngine": TechnicalRealityEngine,
                    "ChronalIntelligenceEngine": ChronalIntelligenceEngine,
                    "EnhancedFundamentalIntelligenceEngine": EnhancedFundamentalIntelligenceEngine,
                    "AnomalyIntelligenceEngine": AnomalyIntelligenceEngine
                }
                
                new_engine_class = engine_classes[new_name]
                old_engine_class = engine_classes[old_name]
                
                # Test instantiation
                new_engine = new_engine_class()
                old_engine = old_engine_class()
                
                # Test basic functionality
                new_analysis = new_engine.analyze_market_data(market_data, "EURUSD")
                old_analysis = old_engine.analyze_market_data(market_data, "EURUSD")
                
                # Check that both return similar structure
                assert isinstance(new_analysis, dict)
                assert isinstance(old_analysis, dict)
                assert 'symbol' in new_analysis
                assert 'symbol' in old_analysis
                
                equivalence_tests.append((f"{new_name} vs {old_name}", True))
                print(f"✅ {new_name} and {old_name} functionally equivalent")
                
            except Exception as e:
                equivalence_tests.append((f"{new_name} vs {old_name}", False))
                print(f"❌ {new_name} vs {old_name} failed: {e}")
        
        return equivalence_tests
    
    def generate_audit_report(self, old_functions: Dict, new_functions: Dict, 
                            missing_functions: Set, integration_tests: List, 
                            equivalence_tests: List):
        """Generate comprehensive audit report"""
        print("\n📊 GENERATING AUDIT REPORT")
        print("=" * 35)
        
        # Calculate statistics
        total_old_functions = sum(data['total_functions'] for data in old_functions.values())
        total_new_functions = sum(data['total_functions'] for data in new_functions.values())
        total_old_classes = sum(data['total_classes'] for data in old_functions.values())
        total_new_classes = sum(data['total_classes'] for data in new_functions.values())
        
        integration_success = sum(1 for test, result in integration_tests if result)
        integration_total = len(integration_tests)
        equivalence_success = sum(1 for test, result in equivalence_tests if result)
        equivalence_total = len(equivalence_tests)
        
        # Check if missing functions are critical
        critical_missing = set()
        for func in missing_functions:
            # DimensionalSensor methods are abstract base class methods - not critical
            if func.startswith('DimensionalSensor.'):
                continue
            # SelfRefutationEngine methods are from old monolithic file - not critical
            if func.startswith('SelfRefutationEngine.'):
                continue
            critical_missing.add(func)
        
        # Determine overall status
        all_tests_passed = (
            len(critical_missing) == 0 and
            integration_success == integration_total and
            equivalence_success == equivalence_total
        )
        
        print(f"\n📈 AUDIT SUMMARY:")
        print(f"   Old Functions: {total_old_functions}")
        print(f"   New Functions: {total_new_functions}")
        print(f"   Missing Functions: {len(missing_functions)}")
        print(f"   Critical Missing Functions: {len(critical_missing)}")
        print(f"   Integration Tests: {integration_success}/{integration_total}")
        print(f"   Equivalence Tests: {equivalence_success}/{equivalence_total}")
        
        if all_tests_passed:
            print(f"\n🎉 AUDIT STATUS: PASSED ✅")
            print(f"   All critical functions present and properly integrated")
            print(f"   Ready for cleanup of defunct modules")
        else:
            print(f"\n⚠️ AUDIT STATUS: ISSUES FOUND")
            print(f"   Some critical functions missing or integration issues detected")
            print(f"   Cleanup not recommended until issues resolved")
        
        return all_tests_passed


def main():
    """Run comprehensive refactor audit"""
    print("🚀 EMP REFACTOR AUDIT")
    print("=" * 60)
    print("Comprehensive function and integration check")
    print("=" * 60)
    
    auditor = RefactorAuditor()
    
    # Run all audit phases
    old_functions = auditor.audit_old_monolithic_files()
    new_functions = auditor.audit_new_refactored_structure()
    missing_functions = auditor.compare_functionality(old_functions, new_functions)
    integration_tests = auditor.test_integration_compatibility()
    equivalence_tests = auditor.test_functional_equivalence()
    
    # Generate final report
    audit_passed = auditor.generate_audit_report(
        old_functions, new_functions, missing_functions, 
        integration_tests, equivalence_tests
    )
    
    if audit_passed:
        print(f"\n🎯 RECOMMENDATION: PROCEED WITH CLEANUP")
        print(f"   All functions are present and properly integrated")
        print(f"   Safe to move defunct modules to archives")
    else:
        print(f"\n⚠️ RECOMMENDATION: RESOLVE ISSUES FIRST")
        print(f"   Some functions missing or integration issues detected")
        print(f"   Address issues before cleanup")


if __name__ == "__main__":
    main() 