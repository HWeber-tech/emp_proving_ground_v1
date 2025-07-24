#!/usr/bin/env python3
"""
Simple Phase 3 Test
==================

Basic validation of Phase 3 orchestrator functionality.
"""

import asyncio
import logging
from datetime import datetime

from src.operational.state_store import get_state_store
from src.core.events import get_event_bus
from src.thinking.phase3_orchestrator import get_phase3_orchestrator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_phase3_simple():
    """Simple test of Phase 3 systems."""
    print("🎯 Starting Phase 3 Simple Test")
    
    # Initialize components
    state_store = await get_state_store()
    event_bus = await get_event_bus()
    
    # Get orchestrator
    orchestrator = await get_phase3_orchestrator(state_store, event_bus)
    
    # Run basic analysis
    print("📊 Running basic analysis...")
    results = await orchestrator.run_full_analysis()
    
    # Print results
    print(f"\n📈 Analysis Results:")
    print(f"   Analysis ID: {results.get('analysis_id')}")
    print(f"   Timestamp: {results.get('timestamp')}")
    
    if 'overall_metrics' in results:
        metrics = results['overall_metrics']
        print(f"\n📊 Overall Metrics:")
        print(f"   Intelligence Score: {metrics.get('intelligence_score', 0):.2f}")
        print(f"   Predatory Score: {metrics.get('predatory_score', 0):.2f}")
        print(f"   Overall Readiness: {metrics.get('overall_readiness', 0):.2f}")
    
    # Check system status
    print(f"\n🔍 System Status:")
    for system_name, system_results in results.get('systems', {}).items():
        if 'error' not in system_results:
            print(f"   ✅ {system_name.title()}: Active")
        else:
            print(f"   ❌ {system_name.title()}: {system_results['error']}")
    
    # Get orchestrator status
    status = await orchestrator.get_status()
    print(f"\n🎯 Orchestrator Status:")
    print(f"   Running: {status.get('is_running')}")
    print(f"   Systems: {len(status.get('systems', {}))}")
    
    return results


if __name__ == "__main__":
    asyncio.run(test_phase3_simple())
