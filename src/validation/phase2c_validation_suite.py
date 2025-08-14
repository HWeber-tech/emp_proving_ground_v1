#!/usr/bin/env python3
"""
Phase 2C Validation Suite - Honest Real Data Testing
======================================================

Comprehensive validation suite for Phase 2C focusing on real market data testing,
actual performance measurement, and fraud-free validation.
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Any, Dict

from src.validation.honest_validation_framework import HonestValidationFramework
from src.validation.real_market_validation import RealMarketValidationFramework

logger = logging.getLogger(__name__)


class Phase2CValidationSuite:
    """
    Comprehensive Phase 2C validation suite with real market data testing.
    """
    
    def __init__(self):
        self.real_validator = RealMarketValidationFramework()
        self.honest_validator = HonestValidationFramework()
        
    async def run_phase2c_week3a(self) -> Dict[str, Any]:
        """Run Week 3A: Real Validation Framework (Days 1-4)"""
        logger.info("="*80)
        logger.info("PHASE 2C - WEEK 3A: REAL VALIDATION FRAMEWORK")
        logger.info("="*80)
        
        # Run honest validation framework
        honest_report = await self.honest_validator.run_all_validations()
        
        # Run real market validation
        real_report = await self.real_validator.run_comprehensive_validation()
        
        # Combine results
        combined_results = {
            'phase': '2C',
            'week': '3A',
            'description': 'Real Validation Framework',
            'timestamp': datetime.now().isoformat(),
            'honest_validation': honest_report,
            'real_market_validation': real_report,
            'summary': {
                'honest_success_rate': honest_report['success_rate'],
                'real_success_rate': real_report['success_rate'],
                'overall_success': (
                    honest_report['success_rate'] >= 0.8 and 
                    real_report['success_rate'] >= 0.8
                )
            }
        }
        
        # Save combined report
        with open('phase2c_week3a_report.json', 'w') as f:
            json.dump(combined_results, f, indent=2)
        
        return combined_results
    
    async def run_phase2c_week3b(self) -> Dict[str, Any]:
        """Run Week 3B: Accuracy Testing (Days 5-7)"""
        logger.info("="*80)
        logger.info("PHASE 2C - WEEK 3B: ACCURACY TESTING")
        logger.info("="*80)
        
        # Focus on accuracy testing with real data
        accuracy_results = {
            'phase': '2C',
            'week': '3B',
            'description': 'Accuracy Testing with Real Data',
            'timestamp': datetime.now().isoformat(),
            'tests': []
        }
        
        # Test anomaly detection accuracy
        anomaly_result = await self.real_validator.validate_anomaly_detection_accuracy()
        accuracy_results['tests'].append(anomaly_result.to_dict())
        
        # Test regime classification accuracy
        regime_result = await self.real_validator.validate_regime_classification_accuracy()
        accuracy_results['tests'].append(regime_result.to_dict())
        
        # Test performance metrics
        performance_result = await self.real_validator.validate_real_performance_metrics()
        accuracy_results['tests'].append(performance_result.to_dict())
        
        # Calculate summary
        passed = sum(1 for test in accuracy_results['tests'] if test['passed'])
        total = len(accuracy_results['tests'])
        
        accuracy_results['summary'] = {
            'total_tests': total,
            'passed_tests': passed,
            'failed_tests': total - passed,
            'success_rate': passed / total if total > 0 else 0.0,
            'status': 'PASSED' if passed >= 2 else 'FAILED'  # Allow 1 failure
        }
        
        # Save accuracy report
        with open('phase2c_week3b_accuracy_report.json', 'w') as f:
            json.dump(accuracy_results, f, indent=2)
        
        return accuracy_results
    
    async def run_comprehensive_phase2c(self) -> Dict[str, Any]:
        """Run complete Phase 2C validation"""
        logger.info("="*100)
        logger.info("COMPREHENSIVE PHASE 2C VALIDATION")
        logger.info("="*100)
        
        # Run Week 3A
        week3a_results = await self.run_phase2c_week3a()
        
        # Run Week 3B
        week3b_results = await self.run_phase2c_week3b()
        
        # Create comprehensive report
        comprehensive_report = {
            'phase': '2C',
            'description': 'Honest Validation Implementation',
            'timestamp': datetime.now().isoformat(),
            'weeks': {
                'week3a': week3a_results,
                'week3b': week3b_results
            },
            'summary': {
                'week3a_success': week3a_results['summary']['overall_success'],
                'week3b_success': week3b_results['summary']['success_rate'] >= 0.67,  # 2/3 tests
                'overall_success': (
                    week3a_results['summary']['overall_success'] and
                    week3b_results['summary']['success_rate'] >= 0.67
                )
            }
        }
        
        # Save comprehensive report
        with open('phase2c_comprehensive_report.json', 'w') as f:
            json.dump(comprehensive_report, f, indent=2)
        
        return comprehensive_report
    
    def print_comprehensive_report(self, report: Dict[str, Any]):
        """Print comprehensive Phase 2C report"""
        print("\n" + "="*120)
        print("PHASE 2C COMPREHENSIVE VALIDATION REPORT")
        print("="*120)
        print(f"Timestamp: {report['timestamp']}")
        print(f"Phase: {report['phase']}")
        print(f"Description: {report['description']}")
        print()
        
        # Week 3A Summary
        week3a = report['weeks']['week3a']
        print("WEEK 3A - REAL VALIDATION FRAMEWORK:")
        print(f"  Honest Validation Success: {week3a['honest_validation']['success_rate']:.1%}")
        print(f"  Real Market Validation Success: {week3a['real_market_validation']['success_rate']:.1%}")
        print(f"  Overall Week 3A Success: {'✅' if week3a['summary']['overall_success'] else '❌'}")
        print()
        
        # Week 3B Summary
        week3b = report['weeks']['week3b']
        print("WEEK 3B - ACCURACY TESTING:")
        print(f"  Tests Passed: {week3b['summary']['passed_tests']}/{week3b['summary']['total_tests']}")
        print(f"  Success Rate: {week3b['summary']['success_rate']:.1%}")
        print(f"  Week 3B Success: {'✅' if week3b['summary']['success_rate'] >= 0.67 else '❌'}")
        print()
        
        print("="*120)
        print(f"OVERALL PHASE 2C SUCCESS: {'✅ PASSED' if report['summary']['overall_success'] else '❌ FAILED'}")
        print("="*120)


async def main():
    """Run complete Phase 2C validation"""
    logging.basicConfig(level=logging.INFO)
    
    suite = Phase2CValidationSuite()
    report = await suite.run_comprehensive_phase2c()
    suite.print_comprehensive_report(report)
    
    # Exit with appropriate code
    import sys
    sys.exit(0 if report['summary']['overall_success'] else 1)


if __name__ == "__main__":
    asyncio.run(main())
