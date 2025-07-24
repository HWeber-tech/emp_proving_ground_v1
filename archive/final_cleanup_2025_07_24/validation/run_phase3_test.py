#!/usr/bin/env python3
"""
Phase 3 Test Suite - Advanced Intelligence Validation
====================================================

Comprehensive test suite for validating Phase 3 advanced intelligence features:
- Sentient Predator: Real-time self-improvement
- Paranoid Predator: Adversarial evolution
- Apex Ecosystem: Multi-agent intelligence
- Competitive Intelligence & Market warfare

Usage:
    python run_phase3_test.py --test-all
    python run_phase3_test.py --test-sentient
    python run_phase3_test.py --test-paranoid
    python run_phase3_test.py --test-ecosystem
    python run_phase3_test.py --test-competitive
"""

import asyncio
import logging
import argparse
import json
from datetime import datetime
from typing import Dict, List, Any

from src.phase3_integration import Phase3Integration

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Phase3TestSuite:
    """Comprehensive test suite for Phase 3 features."""
    
    def __init__(self):
        self.integration = Phase3Integration()
        self.test_results = {}
    
    async def setup(self):
        """Initialize test environment."""
        await self.integration.initialize()
        logger.info("Test suite initialized")
    
    async def test_sentient_predator(self) -> Dict[str, Any]:
        """Test sentient predator features."""
        logger.info("Testing sentient predator features...")
        
        test_data = {
            'symbol': 'EURUSD',
            'regime': 'trending_bull',
            'volatility': 0.025,
            'trend_strength': 0.75,
            'volume_anomaly': 1.3,
            'strategy_response': {'action': 'buy', 'confidence': 0.82},
            'outcome': {'pnl': 0.015, 'win': True}
        }
        
        result = await self.integration.run_sentient_predator(test_data)
        
        # Validate results
        validation = {
            'passed': True,
            'details': {
                'adaptation_signal': result['adaptation']['learning_signal'] > 0.5,
                'predictions_count': len(result['predictions']) >= 3,
                'confidence_high': all(p['confidence'] > 0.5 for p in result['predictions']),
                'timestamp_valid': isinstance(result['timestamp'], datetime)
            }
        }
        
        return {
            'test': 'sentient_predator',
            'result': result,
            'validation': validation,
            'score': sum(validation['details'].values()) / len(validation['details'])
        }
    
    async def test_paranoid_predator(self) -> Dict[str, Any]:
        """Test paranoid predator features."""
        logger.info("Testing paranoid predator features...")
        
        # Mock strategy population
        strategy_population = [
            {'id': f'strategy_{i}', 'fitness': 0.8 + i * 0.01}
            for i in range(10)
        ]
        
        result = await self.integration.run_paranoid_predator(strategy_population)
        
        # Validate results
        validation = {
            'passed': True,
            'details': {
                'strategies_trained': result['trained_strategies'] > 0,
                'attack_results': len(result['attack_results']) > 0,
                'survival_rate_valid': 0 <= result['survival_rate'] <= 1,
                'timestamp_valid': isinstance(result['timestamp'], datetime)
            }
        }
        
        return {
            'test': 'paranoid_predator',
            'result': result,
            'validation': validation,
            'score': sum(validation['details'].values()) / len(validation['details'])
        }
    
    async def test_apex_ecosystem(self) -> Dict[str, Any]:
        """Test apex ecosystem features."""
        logger.info("Testing apex ecosystem features...")
        
        # Mock species populations
        species_populations = {
            'stalker': [{'id': f'stalker_{i}'} for i in range(5)],
            'ambusher': [{'id': f'ambusher_{i}'} for i in range(4)],
            'pack_hunter': [{'id': f'pack_{i}'} for i in range(6)],
            'scavenger': [{'id': f'scavenger_{i}'} for i in range(3)],
            'alpha': [{'id': f'alpha_{i}'} for i in range(2)]
        }
        
        performance_history = {
            'returns': [0.01, 0.02, -0.01, 0.03, 0.01, 0.015, -0.005, 0.025],
            'sharpe_ratios': [1.5, 1.8, 1.2, 2.1, 1.6, 1.9, 1.4, 2.0]
        }
        
        result = await self.integration.run_apex_ecosystem(
            species_populations, performance_history
        )
        
        # Validate results
        validation = {
            'passed': True,
            'details': {
                'ecosystem_evolved': result['evolved_ecosystem']['species_count'] == 5,
                'populations_optimized': len(result['optimized_populations']) == 5,
                'metrics_available': result['ecosystem_summary']['best_metrics'] is not None,
                'timestamp_valid': isinstance(result['timestamp'], datetime)
            }
        }
        
        return {
            'test': 'apex_ecosystem',
            'result': result,
            'validation': validation,
            'score': sum(validation['details'].values()) / len(validation['details'])
        }
    
    async def test_competitive_intelligence(self) -> Dict[str, Any]:
        """Test competitive intelligence features."""
        logger.info("Testing competitive intelligence features...")
        
        market_data = {
            'symbol': 'EURUSD',
            'price_data': [1.1850, 1.1865, 1.1845, 1.1870, 1.1855],
            'volume_data': [1000, 1200, 800, 1500, 1100],
            'timestamp': datetime.now()
        }
        
        result = await self.integration.run_competitive_intelligence(market_data)
        
        # Validate results
        validation = {
            'passed': True,
            'details': {
                'competitors_found': result['competitors_identified'] > 0,
                'counter_strategies': result['counter_strategies'] > 0,
                'market_analysis': result['market_share_analysis'] is not None,
                'timestamp_valid': isinstance(result['timestamp'], datetime)
            }
        }
        
        return {
            'test': 'competitive_intelligence',
            'result': result,
            'validation': validation,
            'score': sum(validation['details'].values()) / len(validation['details'])
        }
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all Phase 3 tests."""
        logger.info("Running all Phase 3 tests...")
        
        tests = [
            self.test_sentient_predator,
            self.test_paranoid_predator,
            self.test_apex_ecosystem,
            self.test_competitive_intelligence
        ]
        
        results = {}
        total_score = 0
        
        for test in tests:
            try:
                result = await test()
                results[result['test']] = result
                total_score += result['score']
            except Exception as e:
                logger.error(f"Test {test.__name__} failed: {e}")
                results[test.__name__] = {
                    'test': test.__name__,
                    'error': str(e),
                    'score': 0
                }
        
        # Calculate overall metrics
        average_score = total_score / len(tests)
        
        return {
            'tests': results,
            'summary': {
                'total_tests': len(tests),
                'passed_tests': len([r for r in results.values() if r.get('score', 0) > 0.8]),
                'average_score': average_score,
                'phase3_ready': average_score >= 0.7,
                'timestamp': datetime.now()
            }
        }
    
    async def generate_test_report(self, results: Dict[str, Any]) -> str:
        """Generate comprehensive test report."""
        report = []
        report.append("=" * 60)
        report.append("PHASE 3 TEST REPORT")
        report.append("=" * 60)
        report.append(f"Test Date: {results['summary']['timestamp']}")
        report.append(f"Total Tests: {results['summary']['total_tests']}")
        report.append(f"Passed Tests: {results['summary']['passed_tests']}")
        report.append(f"Average Score: {results['summary']['average_score']:.2f}")
        report.append(f"Phase 3 Ready: {results['summary']['phase3_ready']}")
        report.append("")
        
        for test_name, test_result in results['tests'].items():
            report.append(f"{test_name.upper()}:")
            report.append(f"  Score: {test_result.get('score', 0):.2f}")
            if 'error' in test_result:
                report.append(f"  Error: {test_result['error']}")
            else:
                report.append(f"  Status: {'PASS' if test_result['score'] > 0.8 else 'FAIL'}")
            report.append("")
        
        return "\n".join(report)


async def main():
    """Main test runner."""
    parser = argparse.ArgumentParser(description="Phase 3 Test Suite")
    parser.add_argument("--test-all", action="store_true", help="Run all tests")
    parser.add_argument("--test-sentient", action="store_true", help="Test sentient predator")
    parser.add_argument("--test-paranoid", action="store_true", help="Test paranoid predator")
    parser.add_argument("--test-ecosystem", action="store_true", help="Test apex ecosystem")
    parser.add_argument("--test-competitive", action="store_true", help="Test competitive intelligence")
    parser.add_argument("--output", default="phase3_test_report.json", help="Output file")
    
    args = parser.parse_args()
    
    # Initialize test suite
    suite = Phase3TestSuite()
    await suite.setup()
    
    # Determine which tests to run
    tests_to_run = []
    if args.test_all:
        tests_to_run = ['all']
    else:
        if args.test_sentient:
            tests_to_run.append('sentient')
        if args.test_paranoid:
            tests_to_run.append('paranoid')
        if args.test_ecosystem:
            tests_to_run.append('ecosystem')
        if args.test_competitive:
            tests_to_run.append('competitive')
    
    if not tests_to_run:
        tests_to_run = ['all']
    
    # Run tests
    results = {}
    
    if 'all' in tests_to_run:
        results = await suite.run_all_tests()
    else:
        for test in tests_to_run:
            if test == 'sentient':
                results['sentient'] = await suite.test_sentient_predator()
            elif test == 'paranoid':
                results['paranoid'] = await suite.test_paranoid_predator()
            elif test == 'ecosystem':
                results['ecosystem'] = await suite.test_apex_ecosystem()
            elif test == 'competitive':
                results['competitive'] = await suite.test_competitive_intelligence()
    
    # Generate report
    if 'all' in tests_to_run:
        report = await suite.generate_test_report(results)
        print(report)
        
        # Save results
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\nResults saved to {args.output}")
    
    # Print summary
    if 'all' in tests_to_run:
        print(f"\nPhase 3 Status: {'READY' if results['summary']['phase3_ready'] else 'NEEDS WORK'}")
    else:
        for test_name, test_result in results.items():
            print(f"{test_name}: {test_result.get('score', 0):.2f}")


if __name__ == "__main__":
    asyncio.run(main())
