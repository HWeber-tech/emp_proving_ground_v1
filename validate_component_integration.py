#!/usr/bin/env python3
"""
Component Integration Validator - Phase 2B
==========================================

Validates all component dependencies and fixes integration issues.
"""

import os
import sys
import importlib
import inspect
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

class ComponentValidator:
    """Validates component integration and dependencies"""
    
    def __init__(self):
        self.missing_components = []
        self.broken_imports = []
        self.missing_methods = []
        
    def validate_imports(self):
        """Check all required imports"""
        required_modules = [
            'src.trading.risk.advanced_risk_manager',
            'src.trading.strategies.strategy_manager',
            'src.trading.risk.market_regime_detector',
            'src.data_integration.real_data_integration',
            'src.sensory.organs.yahoo_finance_organ',
            'src.decision_genome'
        ]
        
        for module_path in required_modules:
            try:
                module = importlib.import_module(module_path)
                print(f"✅ {module_path}")
            except ImportError as e:
                print(f"❌ {module_path}: {e}")
                self.broken_imports.append((module_path, str(e)))
    
    def validate_class_methods(self):
        """Validate required class methods exist"""
        class_specs = {
            'AdvancedRiskManager': {
                'module': 'src.trading.risk.advanced_risk_manager',
                'methods': ['validate_signal', 'calculate_position_size', 'update_dynamic_parameters']
            },
            'StrategyManager': {
                'module': 'src.trading.strategies.strategy_manager',
                'methods': ['add_strategy', 'evaluate_strategies', 'get_strategy_performance']
            },
            'MarketRegimeDetector': {
                'module': 'src.trading.risk.market_regime_detector',
                'methods': ['detect_regime', 'get_regime_probabilities']
            },
            'YahooFinanceOrgan': {
                'module': 'src.sensory.organs.yahoo_finance_organ',
                'methods': ['fetch_data', 'validate_data']
            }
        }
        
        for class_name, spec in class_specs.items():
            try:
                module = importlib.import_module(spec['module'])
                cls = getattr(module, class_name)
                
                for method in spec['methods']:
                    if hasattr(cls, method):
                        print(f"✅ {class_name}.{method}")
                    else:
                        print(f"❌ {class_name}.{method} - missing")
                        self.missing_methods.append(f"{class_name}.{method}")
                        
            except Exception as e:
                print(f"❌ Error validating {class_name}: {e}")
    
    def check_file_structure(self):
        """Check required files exist"""
        required_files = [
            'src/trading/risk/advanced_risk_manager.py',
            'src/trading/strategies/strategy_manager.py',
            'src/trading/risk/market_regime_detector.py',
            'src/data_integration/real_data_integration.py',
            'src/sensory/organs/yahoo_finance_organ.py',
            'src/decision_genome.py',
            'tests/integration/test_component_integration.py'
        ]
        
        for file_path in required_files:
            if os.path.exists(file_path):
                print(f"✅ {file_path}")
            else:
                print(f"❌ {file_path} - missing")
                self.missing_components.append(file_path)
    
    def generate_fix_report(self):
        """Generate report of issues to fix"""
        print("\n" + "="*60)
        print("COMPONENT INTEGRATION VALIDATION REPORT")
        print("="*60)
        
        if not self.missing_components and not self.broken_imports and not self.missing_methods:
            print("🎉 All components are properly integrated!")
            return True
        
        if self.missing_components:
            print("\nMissing Files:")
            for component in self.missing_components:
                print(f"  - {component}")
        
        if self.broken_imports:
            print("\nBroken Imports:")
            for module, error in self.broken_imports:
                print(f"  - {module}: {error}")
        
        if self.missing_methods:
            print("\nMissing Methods:")
            for method in self.missing_methods:
                print(f"  - {method}")
        
        return False
    
    def run_validation(self):
        """Run complete validation"""
        print("Validating component integration...")
        print("\n1. Checking file structure...")
        self.check_file_structure()
        
        print("\n2. Validating imports...")
        self.validate_imports()
        
        print("\n3. Validating class methods...")
        self.validate_class_methods()
        
        return self.generate_fix_report()


if __name__ == "__main__":
    validator = ComponentValidator()
    success = validator.run_validation()
    sys.exit(0 if success else 1)
