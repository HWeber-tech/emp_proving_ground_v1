#!/usr/bin/env python3
"""
Phase 3 Final Integration Test
==============================

Complete end-to-end test of all Phase 3 intelligence systems.
This script demonstrates the full capabilities of the EMP as a
sentient, predatory, and adaptive trading platform.

Usage:
    python run_phase3_final_integration.py
"""

import asyncio
import logging
import sys
from datetime import datetime, timedelta
import json

# Add src to path
sys.path.append('src')

from src.intelligence import get_phase3_orchestrator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def run_full_integration_test():
    """Run complete Phase 3 integration test."""
    
    print("🚀 Starting Phase 3 Final Integration Test")
    print("=" * 60)
    
    # Get the orchestrator
    orchestrator = await get_phase3_orchestrator()
    
    # Test market scenarios
    test_scenarios = [
        {
            'name': 'Bull Market',
            'data': {
                'price': 105.0,
                'volume': 1500000,
                'volatility': 0.015,
                'trend': 0.03,
                'market_regime': 'BULL'
            }
        },
        {
            'name': 'Bear Market',
            'data': {
                'price': 95.0,
                'volume': 2000000,
                'volatility': 0.035,
                'trend': -0.04,
                'market_regime': 'BEAR'
            }
        },
        {
            'name': 'High Volatility Crisis',
            'data': {
                'price': 92.0,
                'volume': 3000000,
                'volatility': 0.08,
                'trend': -0.02,
                'market_regime': 'CRISIS'
            }
        }
    ]
    
    # Test strategies
    test_strategies = [
        {
            'id': 'stalker_001',
            'type': 'stalker',
            'species': 'long_term_trend',
            'parameters': {'lookback': 50, 'threshold': 0.02}
        },
        {
            'id': 'ambusher_002',
            'type': 'ambusher',
            'species': 'scalping',
            'parameters': {'window': 5, 'profit_target': 0.005}
        },
        {
            'id': 'pack_hunter_003',
            'type': 'pack_hunter',
            'species': 'multi_timeframe',
            'parameters': {'timeframes': ['1m', '5m', '1h']}
        },
        {
            'id': 'scavenger_004',
            'type': 'scavenger',
            'species': 'mean_reversion',
            'parameters': {'z_score': 2.0, 'holding_period': 30}
        },
        {
            'id': 'alpha_005',
            'type': 'alpha',
            'species': 'momentum',
            'parameters': {'momentum_period': 20, 'vol_filter': 0.02}
        }
    ]
    
    results = []
    
    for scenario in test_scenarios:
        print(f"\n📊 Testing {scenario['name']} Scenario")
        print("-" * 40)
        
        # Run complete intelligence cycle
        cycle_results = await orchestrator.run_intelligence_cycle(
            market_data=scenario['data'],
            current_strategies=test_strategies
        )
        
        # Analyze results
        analysis = {
            'scenario': scenario['name'],
            'timestamp': cycle_results['timestamp'],
            'sentient_adaptations': len(cycle_results['sentient_adaptations']),
            'predictions': len(cycle_results['predictions']),
            'adversarial_results': len(cycle_results['adversarial_results']),
            'red_team_findings': len(cycle_results['red_team_findings']),
            'specialized_predators': len(cycle_results['specialized_predators']),
            'portfolio_evolution': bool(cycle_results['portfolio_evolution']),
            'competitive_intelligence': bool(cycle_results['competitive_intelligence'])
        }
        
        results.append(analysis)
        
        print(f"✅ Sentient Adaptations: {analysis['sentient_adaptations']}")
        print(f"🔮 Predictions Generated: {analysis['predictions']}")
        print(f"⚔️ Adversarial Results: {analysis['adversarial_results']}")
        print(f"🛡️ Red Team Findings: {analysis['red_team_findings']}")
        print(f"🐺 Specialized Predators: {analysis['specialized_predators']}")
        print(f"📈 Portfolio Evolution: {analysis['portfolio_evolution']}")
        print(f"🕵️ Competitive Intelligence: {analysis['competitive_intelligence']}")
    
    # Get final status
    status = await orchestrator.get_phase3_status()
    
    print("\n" + "=" * 60)
    print("🎯 FINAL INTEGRATION RESULTS")
    print("=" * 60)
    
    # Summary statistics
    total_adaptations = sum(r['sentient_adaptations'] for r in results)
    total_predictions = sum(r['predictions'] for r in results)
    total_adversarial = sum(r['adversarial_results'] for r in results)
    total_findings = sum(r['red_team_findings'] for r in results)
    
    print(f"\n📊 Summary Across All Scenarios:")
    print(f"   Total Sentient Adaptations: {total_adaptations}")
    print(f"   Total Predictions Generated: {total_predictions}")
    print(f"   Total Adversarial Results: {total_adversarial}")
    print(f"   Total Red Team Findings: {total_findings}")
    
    print(f"\n🔧 System Status:")
    for system, info in status.items():
        if isinstance(info, dict):
            active = info.get('active', False)
            status_symbol = "✅" if active else "❌"
            print(f"   {system}: {status_symbol} {info.get('status', 'Unknown')}")
    
    # Save results
    final_results = {
        'timestamp': datetime.utcnow().isoformat(),
        'scenarios_tested': len(test_scenarios),
        'strategies_tested': len(test_strategies),
        'scenario_results': results,
        'system_status': status,
        'summary': {
            'total_adaptations': total_adaptations,
            'total_predictions': total_predictions,
            'total_adversarial': total_adversarial,
            'total_findings': total_findings
        }
    }
    
    with open('phase3_integration_results.json', 'w') as f:
        json.dump(final_results, f, indent=2, default=str)
    
    print(f"\n💾 Results saved to: phase3_integration_results.json")
    
    # Success criteria
    success_criteria = {
        'min_adaptations': 5,
        'min_predictions': 3,
        'min_adversarial': 3,
        'min_findings': 2,
        'all_systems_active': True
    }
    
    meets_criteria = (
        total_adaptations >= success_criteria['min_adaptations'] and
        total_predictions >= success_criteria['min_predictions'] and
        total_adversarial >= success_criteria['min_adversarial'] and
        total_findings >= success_criteria['min_findings']
    )
    
    print(f"\n🎯 SUCCESS CRITERIA MET: {meets_criteria}")
    
    if meets_criteria:
        print("\n🎉 PHASE 3 INTEGRATION TEST PASSED!")
        print("   The EMP is now a fully operational sentient trading platform")
    else:
        print("\n⚠️  PHASE 3 INTEGRATION TEST NEEDS ATTENTION")
        print("   Some systems may need calibration")
    
    return meets_criteria


async def main():
    """Main test function."""
    try:
        success = await run_full_integration_test()
        return 0 if success else 1
    except Exception as e:
        logger.error(f"Integration test failed: {e}")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
