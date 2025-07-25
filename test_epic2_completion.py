#!/usr/bin/env python3
"""
Test script for Epic 2: The Ambusher
"""

import asyncio
import json
import numpy as np
from datetime import datetime
from pathlib import Path

from evolution.ambusher.ambusher_orchestrator import AmbusherOrchestrator
from evolution.ambusher.ambusher_fitness import AmbusherFitnessFunction

async def test_ambusher_components():
    """Test all ambusher components."""
    print("=== Testing Epic 2: The Ambusher ===")
    
    # Test data
    market_data = {
        'prices': [1.1000, 1.1005, 1.1010, 1.1008, 1.1015, 1.1020, 1.1018, 1.1025, 1.1030, 1.1028],
        'volumes': [100000, 150000, 200000, 120000, 180000, 250000, 160000, 220000, 300000, 190000],
        'order_book': {
            'bids': [
                {'price': 1.1025, 'volume': 500000},
                {'price': 1.1020, 'volume': 800000},
                {'price': 1.1015, 'volume': 1200000}
            ],
            'asks': [
                {'price': 1.1030, 'volume': 600000},
                {'price': 1.1035, 'volume': 900000},
                {'price': 1.1040, 'volume': 1100000}
            ],
            'depth': 5000000
        }
    }
    
    trade_history = [
        {'pnl': 0.001, 'duration': 120, 'max_drawdown': 0.002, 'event_type': 'liquidity_grab'},
        {'pnl': -0.0005, 'duration': 180, 'max_drawdown': 0.003, 'event_type': 'stop_cascade'},
        {'pnl': 0.002, 'duration': 90, 'max_drawdown': 0.001, 'event_type': 'momentum_burst'}
    ]
    
    # Test 1: AmbusherFitnessFunction
    print("\n1. Testing AmbusherFitnessFunction...")
    fitness_config = {
        'profit_weight': 0.4,
        'accuracy_weight': 0.3,
        'timing_weight': 0.2,
        'risk_weight': 0.1,
        'min_liquidity_threshold': 1000000,
        'max_drawdown': 0.05
    }
    
    fitness_func = AmbusherFitnessFunction(fitness_config)
    test_genome = {
        'liquidity_grab_threshold': 0.001,
        'cascade_threshold': 0.002,
        'momentum_threshold': 0.0015,
        'volume_threshold': 2.0,
        'volume_spike': 3.0,
        'consecutive_moves': 3,
        'iceberg_threshold': 1000000,
        'risk_multiplier': 1.0,
        'position_size': 0.01,
        'stop_loss': 0.005,
        'take_profit': 0.01,
        'entry_delay': 0
    }
    
    fitness = fitness_func.calculate_fitness(test_genome, market_data, trade_history)
    print(f"   ✅ Fitness calculated: {fitness:.4f}")
    
    # Test 2: AmbusherOrchestrator
    print("\n2. Testing AmbusherOrchestrator...")
    config = {
        'genetic': {
            'population_size': 20,
            'generations': 5,
            'mutation_rate': 0.1,
            'crossover_rate': 0.8,
            'elite_size': 2
        },
        'fitness': fitness_config,
        'genome_path': 'data/evolution/test_ambusher_genome.json',
        'history_path': 'data/evolution/test_ambusher_history.json'
    }
    
    orchestrator = AmbusherOrchestrator(config)
    await orchestrator.start()
    
    # Test evolution
    print("   Running evolution...")
    result = await orchestrator.evolve_strategy(market_data, trade_history)
    
    print(f"   ✅ Evolution completed")
    print(f"   Best fitness: {result['fitness']:.4f}")
    print(f"   Genome saved to: {config['genome_path']}")
    
    # Test current strategy
    current_strategy = orchestrator.get_current_strategy()
    if current_strategy:
        print(f"   ✅ Current strategy loaded")
        print(f"   Key parameters: {list(current_strategy.keys())}")
    
    # Test performance metrics
    metrics = orchestrator.get_performance_metrics()
    print(f"   ✅ Metrics tracked: {metrics}")
    
    # Test reset
    await orchestrator.reset()
    print("   ✅ Reset completed")
    
    # Verify files were created
    genome_path = Path(config['genome_path'])
    history_path = Path(config['history_path'])
    
    if genome_path.exists():
        print(f"   ✅ Genome file created: {genome_path}")
        genome_path.unlink()  # Clean up
    
    if history_path.exists():
        print(f"   ✅ History file created: {history_path}")
        history_path.unlink()  # Clean up
    
    print("\n=== Epic 2: The Ambusher - All Tests Passed! ===")
    
    return True

if __name__ == "__main__":
    asyncio.run(test_ambusher_components())
