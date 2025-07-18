#!/usr/bin/env python3
"""
Test script for Genesis Run
"""

import asyncio
import logging
from datetime import datetime
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from run_genesis import GenesisRunner, GenesisConfig

async def test_genesis():
    """Test the Genesis Run with minimal configuration"""
    logging.basicConfig(level=logging.INFO)
    
    # Create minimal test configuration
    config = GenesisConfig(
        symbol="EURUSD",
        population_size=10,  # Small population for testing
        generations=5,       # Few generations for testing
        start_date="2024-01-01",
        end_date="2024-01-31",  # One month of data
        max_risk_per_trade=0.02,
        max_drawdown=0.25
    )
    
    # Create runner
    runner = GenesisRunner(config)
    
    # Run test
    success = await runner.run()
    
    if success:
        print("✅ Genesis Run test completed successfully!")
        
        # Check results
        results_file = Path("genesis_results.json")
        if results_file.exists():
            import json
            with open(results_file) as f:
                results = json.load(f)
            
            print(f"\n📊 Test Results Summary:")
            print(f"   Generations: {len(results['generations'])}")
            print(f"   Final Best Fitness: {results['generations'][-1]['best_fitness']:.4f}")
            print(f"   System Health: {results['system_health']['sensory_cortex']['overall_health']:.2f}")
            print(f"   Data Quality: {results['system_health']['data_pipeline']['data_quality']:.2f}")
            
            return True
    else:
        print("❌ Genesis Run test failed")
        return False

if __name__ == "__main__":
    asyncio.run(test_genesis())
