#!/usr/bin/env python3
"""
Integration Test Runner - Phase 2B
==================================

Runs comprehensive integration tests for component communication and data flow.
"""

import asyncio
import logging
import sys
import os
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from tests.integration.test_component_integration import ComponentIntegrationTester

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'integration_test_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


async def run_integration_tests():
    """Run all integration tests and generate report"""
    logger.info("="*80)
    logger.info("PHASE 2B COMPONENT INTEGRATION TESTS")
    logger.info("="*80)
    
    tester = ComponentIntegrationTester()
    
    try:
        results = await tester.run_all_tests()
        
        # Generate detailed report
        print("\n" + "="*80)
        print("INTEGRATION TEST RESULTS SUMMARY")
        print("="*80)
        
        passed = results['passed_tests']
        total = results['total_tests']
        success_rate = (passed / total * 100) if total > 0 else 0
        
        print(f"Total Tests: {total}")
        print(f"Passed: {passed}")
        print(f"Failed: {results['failed_tests']}")
        print(f"Success Rate: {success_rate:.1f}%")
        
        print("\n" + "-"*80)
        print("DETAILED RESULTS")
        print("-"*80)
        
        for result in results['results']:
            status = "✅ PASS" if result['success'] else "❌ FAIL"
            print(f"{status} {result['test']}")
            print(f"    {result['message']}")
            print()
        
        # Save results to file
        report_file = f"integration_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        import json
        with open(report_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"Detailed report saved to: {report_file}")
        
        return passed == total
        
    except Exception as e:
        logger.error(f"Integration test runner failed: {e}")
        return False


if __name__ == "__main__":
    success = asyncio.run(run_integration_tests())
    sys.exit(0 if success else 1)
