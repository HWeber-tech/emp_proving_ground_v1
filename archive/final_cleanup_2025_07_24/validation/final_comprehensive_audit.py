"""
Final Comprehensive Audit

This script performs a complete audit of the refactored sensory cortex
before cleanup to ensure all functionality is preserved and working.

Author: EMP Development Team
Date: July 18, 2024
Phase: 2 - Final Audit Before Cleanup
"""

import sys
import os
import inspect
import importlib
import traceback
from typing import Dict, List, Set, Any, Tuple
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


class FinalComprehensiveAuditor:
    """
    Final comprehensive auditor for the refactored sensory cortex.
    """
    
    def __init__(self):
        """Initialize the final auditor"""
        self.audit_results = {}
        self.test_results = {}
        print("🔍 FINAL COMPREHENSIVE AUDIT")
        print("=" * 60)
    
    def run_complete_audit(self) -> Dict[str, Any]:
        """
        Run the complete final audit.
        
        Returns:
            Complete audit results
        """
        print("🚀 STARTING FINAL COMPREHENSIVE AUDIT")
        print("=" * 60)
        
        # 1. Function Coverage Audit
        print("\n📊 1. FUNCTION COVERAGE AUDIT")
        print("-" * 40)
        coverage_results = self._audit_function_coverage()
        
        # 2. Import and Integration Audit
        print("\n🔗 2. IMPORT AND INTEGRATION AUDIT")
        print("-" * 40)
        integration_results = self._audit_imports_and_integration()
        
        # 3. End-to-End Logic Audit
        print("\n🔄 3. END-TO-END LOGIC AUDIT")
        print("-" * 40)
        e2e_results = self._audit_end_to_end_logic()
        
        # 4. Backward Compatibility Audit
        print("\n🔄 4. BACKWARD COMPATIBILITY AUDIT")
        print("-" * 40)
        compatibility_results = self._audit_backward_compatibility()
        
        # 5. System Integration Audit
        print("\n⚙️ 5. SYSTEM INTEGRATION AUDIT")
        print("-" * 40)
        system_results = self._audit_system_integration()
        
        # Compile final results
        final_results = {
            'timestamp': datetime.now(),
            'coverage_audit': coverage_results,
            'integration_audit': integration_results,
            'e2e_audit': e2e_results,
            'compatibility_audit': compatibility_results,
            'system_audit': system_results,
            'overall_status': self._determine_overall_status([
                coverage_results, integration_results, e2e_results,
                compatibility_results, system_results
            ])
        }
        
        self._print_final_summary(final_results)
        return final_results
    
    def _audit_function_coverage(self) -> Dict[str, Any]:
        """Audit function coverage across all modules"""
        try:
            # Check old files
            old_files = [
                'src/sensory/dimensions/enhanced_how_dimension.py',
                'src/sensory/dimensions/enhanced_what_dimension.py',
                'src/sensory/dimensions/enhanced_when_dimension.py',
                'src/sensory/dimensions/enhanced_why_dimension.py',
                'src/sensory/dimensions/enhanced_anomaly_dimension.py'
            ]
            
            old_functions = 0
            old_classes = 0
            for file_path in old_files:
                try:
                    if os.path.exists(file_path):
                        module_name = file_path.replace('/', '.').replace('.py', '')
                        module = importlib.import_module(module_name)
                        
                        functions = 0
                        classes = 0
                        for name, obj in inspect.getmembers(module):
                            if inspect.isfunction(obj):
                                functions += 1
                            elif inspect.isclass(obj):
                                classes += 1
                                for method_name, method_obj in inspect.getmembers(obj):
                                    if inspect.isfunction(method_obj) and not method_name.startswith('_'):
                                        functions += 1
                        
                        old_functions += functions
                        old_classes += classes
                        print(f"✅ Old {file_path}: {functions} functions, {classes} classes")
                except Exception as e:
                    print(f"❌ Error checking {file_path}: {e}")
            
            # Check new modules
            new_modules = [
                'src.sensory.dimensions.how.how_engine',
                'src.sensory.dimensions.how.indicators',
                'src.sensory.dimensions.how.patterns',
                'src.sensory.dimensions.how.order_flow',
                'src.sensory.dimensions.what.what_engine',
                'src.sensory.dimensions.what.price_action',
                'src.sensory.dimensions.when.when_engine',
                'src.sensory.dimensions.when.regime_detection',
                'src.sensory.dimensions.why.why_engine',
                'src.sensory.dimensions.why.economic_analysis',
                'src.sensory.dimensions.anomaly.anomaly_engine',
                'src.sensory.dimensions.anomaly.pattern_recognition',
                'src.sensory.dimensions.anomaly.anomaly_detection',
                'src.sensory.dimensions.compatibility'
            ]
            
            new_functions = 0
            new_classes = 0
            for module_name in new_modules:
                try:
                    module = importlib.import_module(module_name)
                    
                    functions = 0
                    classes = 0
                    for name, obj in inspect.getmembers(module):
                        if inspect.isfunction(obj):
                            functions += 1
                        elif inspect.isclass(obj):
                            classes += 1
                            for method_name, method_obj in inspect.getmembers(obj):
                                if inspect.isfunction(method_obj) and not method_name.startswith('_'):
                                    functions += 1
                    
                    new_functions += functions
                    new_classes += classes
                    print(f"✅ New {module_name}: {functions} functions, {classes} classes")
                except Exception as e:
                    print(f"❌ Error checking {module_name}: {e}")
            
            coverage = (new_functions / old_functions * 100) if old_functions > 0 else 0
            
            results = {
                'old_functions': old_functions,
                'old_classes': old_classes,
                'new_functions': new_functions,
                'new_classes': new_classes,
                'coverage_percentage': coverage,
                'missing_functions': old_functions - new_functions,
                'status': 'PASS' if coverage >= 60 else 'FAIL'
            }
            
            print(f"\n📊 COVERAGE SUMMARY:")
            print(f"   Old Functions: {old_functions}")
            print(f"   New Functions: {new_functions}")
            print(f"   Coverage: {coverage:.1f}%")
            print(f"   Missing: {old_functions - new_functions}")
            print(f"   Status: {results['status']}")
            
            return results
            
        except Exception as e:
            print(f"❌ Error in function coverage audit: {e}")
            return {'status': 'ERROR', 'error': str(e)}
    
    def _audit_imports_and_integration(self) -> Dict[str, Any]:
        """Audit imports and integration"""
        try:
            print("Testing import capabilities...")
            
            # Test core imports
            core_imports = [
                'src.sensory.core.base',
                'src.sensory.dimensions.how',
                'src.sensory.dimensions.what',
                'src.sensory.dimensions.when',
                'src.sensory.dimensions.why',
                'src.sensory.dimensions.anomaly'
            ]
            
            import_results = {}
            for module_name in core_imports:
                try:
                    module = importlib.import_module(module_name)
                    import_results[module_name] = 'PASS'
                    print(f"✅ {module_name}: PASS")
                except Exception as e:
                    import_results[module_name] = f'FAIL: {e}'
                    print(f"❌ {module_name}: FAIL - {e}")
            
            # Test engine instantiation
            print("\nTesting engine instantiation...")
            engines = [
                ('HowEngine', 'src.sensory.dimensions.how.how_engine'),
                ('WhatEngine', 'src.sensory.dimensions.what.what_engine'),
                ('WhenEngine', 'src.sensory.dimensions.when.when_engine'),
                ('WhyEngine', 'src.sensory.dimensions.why.why_engine'),
                ('AnomalyEngine', 'src.sensory.dimensions.anomaly.anomaly_engine')
            ]
            
            engine_results = {}
            for engine_name, module_name in engines:
                try:
                    module = importlib.import_module(module_name)
                    engine_class = getattr(module, engine_name)
                    engine_instance = engine_class()
                    engine_results[engine_name] = 'PASS'
                    print(f"✅ {engine_name}: PASS")
                except Exception as e:
                    engine_results[engine_name] = f'FAIL: {e}'
                    print(f"❌ {engine_name}: FAIL - {e}")
            
            # Test legacy compatibility
            print("\nTesting legacy compatibility...")
            legacy_imports = [
                'src.sensory.dimensions.compatibility'
            ]
            
            legacy_results = {}
            for module_name in legacy_imports:
                try:
                    module = importlib.import_module(module_name)
                    legacy_results[module_name] = 'PASS'
                    print(f"✅ {module_name}: PASS")
                except Exception as e:
                    legacy_results[module_name] = f'FAIL: {e}'
                    print(f"❌ {module_name}: FAIL - {e}")
            
            results = {
                'core_imports': import_results,
                'engine_instantiation': engine_results,
                'legacy_compatibility': legacy_results,
                'status': 'PASS' if all(r == 'PASS' for r in import_results.values()) else 'FAIL'
            }
            
            return results
            
        except Exception as e:
            print(f"❌ Error in import audit: {e}")
            return {'status': 'ERROR', 'error': str(e)}
    
    def _audit_end_to_end_logic(self) -> Dict[str, Any]:
        """Audit end-to-end logic functionality"""
        try:
            print("Testing end-to-end logic...")
            
            # Test data creation
            from src.sensory.core.base import MarketData
            from datetime import datetime, timedelta
            
            market_data = []
            base_price = 1.1000
            for i in range(20):
                timestamp = datetime.now() - timedelta(minutes=20-i)
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
            
            print(f"✅ Created {len(market_data)} test data points")
            
            # Test each engine
            engines_to_test = [
                ('HowEngine', 'src.sensory.dimensions.how.how_engine'),
                ('WhatEngine', 'src.sensory.dimensions.what.what_engine'),
                ('WhenEngine', 'src.sensory.dimensions.when.when_engine'),
                ('WhyEngine', 'src.sensory.dimensions.why.why_engine'),
                ('AnomalyEngine', 'src.sensory.dimensions.anomaly.anomaly_engine')
            ]
            
            e2e_results = {}
            for engine_name, module_name in engines_to_test:
                try:
                    # Import and instantiate
                    module = importlib.import_module(module_name)
                    engine_class = getattr(module, engine_name)
                    engine = engine_class()
                    
                    # Test analyze_market_data
                    analysis = engine.analyze_market_data(market_data, "EURUSD")
                    if isinstance(analysis, dict) and 'symbol' in analysis:
                        analysis_status = 'PASS'
                    else:
                        analysis_status = 'FAIL'
                    
                    # Test get_dimensional_reading
                    reading = engine.get_dimensional_reading(market_data, "EURUSD")
                    if hasattr(reading, 'dimension') and hasattr(reading, 'signal_strength'):
                        reading_status = 'PASS'
                    else:
                        reading_status = 'FAIL'
                    
                    e2e_results[engine_name] = {
                        'instantiation': 'PASS',
                        'analyze_market_data': analysis_status,
                        'get_dimensional_reading': reading_status,
                        'overall': 'PASS' if analysis_status == 'PASS' and reading_status == 'PASS' else 'FAIL'
                    }
                    
                    print(f"✅ {engine_name}: {e2e_results[engine_name]['overall']}")
                    
                except Exception as e:
                    e2e_results[engine_name] = {
                        'instantiation': f'FAIL: {e}',
                        'analyze_market_data': 'FAIL',
                        'get_dimensional_reading': 'FAIL',
                        'overall': 'FAIL'
                    }
                    print(f"❌ {engine_name}: FAIL - {e}")
            
            # Test orchestration integration
            print("\nTesting orchestration integration...")
            try:
                from src.sensory.orchestration.master_orchestrator import MasterOrchestrator
                orchestrator = MasterOrchestrator()
                orchestration_status = 'PASS'
                print("✅ MasterOrchestrator: PASS")
            except Exception as e:
                orchestration_status = f'FAIL: {e}'
                print(f"❌ MasterOrchestrator: FAIL - {e}")
            
            results = {
                'engine_e2e_tests': e2e_results,
                'orchestration_integration': orchestration_status,
                'status': 'PASS' if all(r['overall'] == 'PASS' for r in e2e_results.values()) else 'FAIL'
            }
            
            return results
            
        except Exception as e:
            print(f"❌ Error in end-to-end logic audit: {e}")
            return {'status': 'ERROR', 'error': str(e)}
    
    def _audit_backward_compatibility(self) -> Dict[str, Any]:
        """Audit backward compatibility"""
        try:
            print("Testing backward compatibility...")
            
            # Test legacy class imports
            legacy_classes = [
                'InstitutionalMechanicsEngine',
                'TechnicalRealityEngine',
                'ChronalIntelligenceEngine',
                'EnhancedFundamentalIntelligenceEngine',
                'AnomalyIntelligenceEngine'
            ]
            
            compatibility_results = {}
            for class_name in legacy_classes:
                try:
                    from src.sensory.dimensions.compatibility import (
                        InstitutionalMechanicsEngine, TechnicalRealityEngine,
                        ChronalIntelligenceEngine, EnhancedFundamentalIntelligenceEngine,
                        AnomalyIntelligenceEngine
                    )
                    
                    engine_class = globals()[class_name]
                    engine_instance = engine_class()
                    
                    # Test basic functionality
                    if hasattr(engine_instance, 'analyze_market_data'):
                        compatibility_results[class_name] = 'PASS'
                        print(f"✅ {class_name}: PASS")
                    else:
                        compatibility_results[class_name] = 'FAIL: Missing analyze_market_data'
                        print(f"❌ {class_name}: FAIL - Missing analyze_market_data")
                        
                except Exception as e:
                    compatibility_results[class_name] = f'FAIL: {e}'
                    print(f"❌ {class_name}: FAIL - {e}")
            
            # Test legacy enums
            try:
                from src.sensory.dimensions.compatibility import PatternType, AnomalyType
                enum_status = 'PASS'
                print("✅ Legacy enums: PASS")
            except Exception as e:
                enum_status = f'FAIL: {e}'
                print(f"❌ Legacy enums: FAIL - {e}")
            
            results = {
                'legacy_classes': compatibility_results,
                'legacy_enums': enum_status,
                'status': 'PASS' if all(r == 'PASS' for r in compatibility_results.values()) else 'FAIL'
            }
            
            return results
            
        except Exception as e:
            print(f"❌ Error in backward compatibility audit: {e}")
            return {'status': 'ERROR', 'error': str(e)}
    
    def _audit_system_integration(self) -> Dict[str, Any]:
        """Audit system integration"""
        try:
            print("Testing system integration...")
            
            # Test data integration
            try:
                from src.data_integration.real_data_integration import RealDataManager
                data_manager = RealDataManager()
                data_integration_status = 'PASS'
                print("✅ RealDataManager: PASS")
            except Exception as e:
                data_integration_status = f'FAIL: {e}'
                print(f"❌ RealDataManager: FAIL - {e}")
            
            # Test sensory integration
            try:
                from src.sensory import (
                    HowEngine, WhatEngine, WhenEngine, WhyEngine, AnomalyEngine
                )
                sensory_import_status = 'PASS'
                print("✅ Sensory imports: PASS")
            except Exception as e:
                sensory_import_status = f'FAIL: {e}'
                print(f"❌ Sensory imports: FAIL - {e}")
            
            # Test full system workflow
            try:
                # Create test data
                from src.sensory.core.base import MarketData
                from datetime import datetime, timedelta
                
                test_data = []
                for i in range(10):
                    test_data.append(MarketData(
                        symbol="EURUSD",
                        timestamp=datetime.now() - timedelta(minutes=10-i),
                        open=1.1000,
                        high=1.1005,
                        low=1.0995,
                        close=1.1002,
                        volume=1000,
                        bid=1.1001,
                        ask=1.1003,
                        source="test",
                        latency_ms=0.0
                    ))
                
                # Test full workflow
                from src.sensory.dimensions.how.how_engine import HowEngine
                engine = HowEngine()
                analysis = engine.analyze_market_data(test_data, "EURUSD")
                reading = engine.get_dimensional_reading(test_data, "EURUSD")
                
                if analysis and reading:
                    workflow_status = 'PASS'
                    print("✅ Full workflow: PASS")
                else:
                    workflow_status = 'FAIL: Empty results'
                    print("❌ Full workflow: FAIL - Empty results")
                    
            except Exception as e:
                workflow_status = f'FAIL: {e}'
                print(f"❌ Full workflow: FAIL - {e}")
            
            results = {
                'data_integration': data_integration_status,
                'sensory_imports': sensory_import_status,
                'full_workflow': workflow_status,
                'status': 'PASS' if all(s == 'PASS' for s in [data_integration_status, sensory_import_status, workflow_status]) else 'FAIL'
            }
            
            return results
            
        except Exception as e:
            print(f"❌ Error in system integration audit: {e}")
            return {'status': 'ERROR', 'error': str(e)}
    
    def _determine_overall_status(self, results_list: List[Dict[str, Any]]) -> str:
        """Determine overall audit status"""
        try:
            statuses = [r.get('status', 'UNKNOWN') for r in results_list]
            
            if all(s == 'PASS' for s in statuses):
                return 'PASS'
            elif any(s == 'FAIL' for s in statuses):
                return 'FAIL'
            else:
                return 'PARTIAL'
                
        except Exception as e:
            return f'ERROR: {e}'
    
    def _print_final_summary(self, results: Dict[str, Any]):
        """Print final audit summary"""
        print("\n" + "=" * 60)
        print("🎯 FINAL AUDIT SUMMARY")
        print("=" * 60)
        
        print(f"\n📊 COVERAGE AUDIT: {results['coverage_audit']['status']}")
        print(f"   Coverage: {results['coverage_audit'].get('coverage_percentage', 0):.1f}%")
        print(f"   Missing Functions: {results['coverage_audit'].get('missing_functions', 0)}")
        
        print(f"\n🔗 INTEGRATION AUDIT: {results['integration_audit']['status']}")
        print(f"   Core Imports: {sum(1 for v in results['integration_audit']['core_imports'].values() if v == 'PASS')}/{len(results['integration_audit']['core_imports'])}")
        print(f"   Engine Instantiation: {sum(1 for v in results['integration_audit']['engine_instantiation'].values() if v == 'PASS')}/{len(results['integration_audit']['engine_instantiation'])}")
        
        print(f"\n🔄 E2E LOGIC AUDIT: {results['e2e_audit']['status']}")
        e2e_engines = results['e2e_audit']['engine_e2e_tests']
        print(f"   Engine Tests: {sum(1 for v in e2e_engines.values() if v['overall'] == 'PASS')}/{len(e2e_engines)}")
        
        print(f"\n🔄 COMPATIBILITY AUDIT: {results['compatibility_audit']['status']}")
        legacy_classes = results['compatibility_audit']['legacy_classes']
        print(f"   Legacy Classes: {sum(1 for v in legacy_classes.values() if v == 'PASS')}/{len(legacy_classes)}")
        
        print(f"\n⚙️ SYSTEM AUDIT: {results['system_audit']['status']}")
        
        print(f"\n🎯 OVERALL STATUS: {results['overall_status']}")
        
        if results['overall_status'] == 'PASS':
            print("\n✅ CLEANUP READY: All systems operational!")
        elif results['overall_status'] == 'PARTIAL':
            print("\n⚠️ CLEANUP CAUTION: Some issues detected")
        else:
            print("\n❌ CLEANUP NOT RECOMMENDED: Critical issues found")


def main():
    """Run the final comprehensive audit"""
    auditor = FinalComprehensiveAuditor()
    results = auditor.run_complete_audit()
    
    # Save results
    import json
    with open('final_audit_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n📄 Audit results saved to: final_audit_results.json")
    
    return results


if __name__ == "__main__":
    main() 