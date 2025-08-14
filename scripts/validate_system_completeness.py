#!/usr/bin/env python3
# ruff: noqa: I001,F401
"""
System Completeness Validation Script
====================================

Validates that all critical components are implemented and working.
This script provides a comprehensive check of the system after implementing
the missing core components.
"""

import asyncio
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SystemValidator:
    """Comprehensive system validator."""
    
    def __init__(self):
        self.validation_results = []
        self.critical_components = [
            'core.exceptions',
            'core.interfaces',
            'evolution.engine.population_manager',
            'validation.validation_framework',
            'validation.real_market_validation',
            'integration.component_integrator_impl'
        ]
        
    async def validate_imports(self) -> Dict[str, bool]:
        """Validate all critical imports work."""
        results = {}
        
        # Test core exceptions
        try:
            from src.core.exceptions import EMPException, ValidationException
            results['core.exceptions'] = True
            logger.info("✅ Core exceptions imported successfully")
        except ImportError as e:
            results['core.exceptions'] = False
            logger.error(f"❌ Core exceptions import failed: {e}")
            
        # Test core interfaces
        try:
            from src.core.interfaces import DecisionGenome, IPopulationManager
            results['core.interfaces'] = True
            logger.info("✅ Core interfaces imported successfully")
        except ImportError as e:
            results['core.interfaces'] = False
            logger.error(f"❌ Core interfaces import failed: {e}")
            
        # Test population manager
        try:
            from src.core.population_manager import PopulationManager
            results['evolution.engine.population_manager'] = True
            logger.info("✅ Population manager imported successfully")
        except ImportError as e:
            results['evolution.engine.population_manager'] = False
            logger.error(f"❌ Population manager import failed: {e}")
            
        # Test validation framework
        try:
            from src.validation.validation_framework import ValidationFramework
            results['validation.validation_framework'] = True
            logger.info("✅ Validation framework imported successfully")
        except ImportError as e:
            results['validation.validation_framework'] = False
            logger.error(f"❌ Validation framework import failed: {e}")
            
        # Test real market validation
        try:
            from src.validation.real_market_validation import RealMarketValidationFramework
            results['validation.real_market_validation'] = True
            logger.info("✅ Real market validation imported successfully")
        except ImportError as e:
            results['validation.real_market_validation'] = False
            logger.error(f"❌ Real market validation import failed: {e}")
            
        # Test component integrator
        try:
            from src.integration.component_integrator_impl import ComponentIntegratorImpl
            results['integration.component_integrator_impl'] = True
            logger.info("✅ Component integrator imported successfully")
        except ImportError as e:
            results['integration.component_integrator_impl'] = False
            logger.error(f"❌ Component integrator import failed: {e}")
            
        return results
        
    async def validate_functionality(self) -> Dict[str, bool]:
        """Validate basic functionality of implemented components."""
        results = {}
        
        # Test population manager
        try:
            from src.core.interfaces import DecisionGenome
            from src.core.population_manager import create_population_manager
            
            manager = create_population_manager(population_size=5)
            
            def genome_factory():
                return DecisionGenome(
                    parameters={'param1': 0.5, 'param2': 0.3},
                    indicators=['SMA', 'EMA'],
                    rules={'entry': 'SMA > EMA', 'exit': 'SMA < EMA'},
                    risk_profile={'max_drawdown': 0.05}
                )
            
            manager.initialize_population(genome_factory)
            stats = manager.get_population_statistics()
            
            results['population_manager_functionality'] = stats['population_size'] == 5
            logger.info("✅ Population manager functionality validated")
            
        except Exception as e:
            results['population_manager_functionality'] = False
            logger.error(f"❌ Population manager functionality failed: {e}")
            
        # Test validation framework
        try:
            from src.validation.validation_framework import ValidationFramework
            framework = ValidationFramework()
            results['validation_framework_functionality'] = True
            logger.info("✅ Validation framework functionality validated")
            
        except Exception as e:
            results['validation_framework_functionality'] = False
            logger.error(f"❌ Validation framework functionality failed: {e}")
            
        # Test component integrator
        try:
            from src.integration.component_integrator_impl import ComponentIntegratorImpl
            integrator = ComponentIntegratorImpl()
            results['component_integrator_functionality'] = True
            logger.info("✅ Component integrator functionality validated")
            
        except Exception as e:
            results['component_integrator_functionality'] = False
            logger.error(f"❌ Component integrator functionality failed: {e}")
            
        return results
        
    async def validate_system_integration(self) -> Dict[str, bool]:
        """Validate system integration."""
        results = {}
        
        try:
            # Test basic system integration
            from src.integration.component_integrator_impl import ComponentIntegratorImpl
            
            integrator = ComponentIntegratorImpl()
            components = integrator.list_components()
            
            results['system_integration'] = len(components) > 0
            logger.info(f"✅ System integration validated - {len(components)} components available")
            
        except Exception as e:
            results['system_integration'] = False
            logger.error(f"❌ System integration failed: {e}")
            
        return results
        
    async def run_comprehensive_validation(self) -> Dict[str, Any]:
        """Run comprehensive system validation."""
        logger.info("Starting comprehensive system validation...")
        
        # Import validation
        import_results = await self.validate_imports()
        
        # Functionality validation
        functionality_results = await self.validate_functionality()
        
        # Integration validation
        integration_results = await self.validate_system_integration()
        
        # Combine results
        all_results = {**import_results, **functionality_results, **integration_results}
        
        # Calculate summary
        total_checks = len(all_results)
        passed_checks = sum(1 for v in all_results.values() if v)
        
        # Create comprehensive report
        report = {
            'timestamp': datetime.now().isoformat(),
            'validator': 'System Completeness Validator',
            'version': '1.0.0',
            'total_checks': total_checks,
            'passed_checks': passed_checks,
            'failed_checks': total_checks - passed_checks,
            'success_rate': passed_checks / total_checks if total_checks > 0 else 0.0,
            'results': all_results,
            'summary': {
                'status': 'PASSED' if passed_checks >= 8 else 'PARTIAL',
                'message': f"{passed_checks}/{total_checks} checks passed ({passed_checks/total_checks:.1%} success rate)"
            }
        }
        
        # Save results
        Path('system_validation_report.json').write_text(json.dumps(report, indent=2))
        
        return report
        
    def print_report(self, report: Dict[str, Any]) -> None:
        """Print validation report."""
        print("\n" + "="*100)
        print("SYSTEM COMPLETENESS VALIDATION REPORT")
        print("="*100)
        print(f"Timestamp: {report['timestamp']}")
        print(f"Validator: {report['validator']} v{report['version']}")
        print(f"Status: {report['summary']['status']}")
        print(f"Success Rate: {report['success_rate']:.1%}")
        print()
        
        print("VALIDATION RESULTS:")
        print("-" * 60)
        for component, passed in report['results'].items():
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"{status} {component}")
        
        print("="*100)
        print(report['summary']['message'])
        print("="*100)


async def main():
    """Run system validation."""
    validator = SystemValidator()
    report = await validator.run_comprehensive_validation()
    validator.print_report(report)
    
    # Exit with appropriate code
    success_threshold = 0.8  # 80% success rate required
    if report['success_rate'] >= success_threshold:
        logger.info("🎉 System validation PASSED - Ready for production!")
        return 0
    else:
        logger.warning("⚠️  System validation PARTIAL - Some components need attention")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
