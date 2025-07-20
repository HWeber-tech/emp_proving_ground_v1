#!/usr/bin/env python3
"""
EMP Proving Ground v0.2 Feature Test Suite

This script tests the major v0.2 upgrade features:
1. Intelligent Adversarial Engine v0.2
2. Wise Arbiter Fitness Evaluator v0.2
3. Triathlon Evaluation Framework
4. Multi-objective Fitness Metrics
"""

import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

# Import the main system
from emp_proving_ground_unified import (
    TickDataStorage, TickDataCleaner, DukascopyIngestor,
    MarketSimulator, AdversarialEngine, SensoryCortex,
    DecisionGenome, FitnessEvaluator, FitnessScore
)


def test_intelligent_adversarial_engine():
    """Test v0.2 intelligent adversarial engine features"""
    print("="*60)
    print("TESTING: Intelligent Adversarial Engine v0.2")
    print("="*60)
    
    # Initialize components
    data_storage = TickDataStorage("test_data")
    cleaner = TickDataCleaner()
    ingestor = DukascopyIngestor(data_storage, cleaner)
    
    # Generate test data
    ingestor.ingest_year("EURUSD", 2022)
    
    # Initialize simulator and adversarial engine
    simulator = MarketSimulator(data_storage, initial_balance=100000.0)
    adversary = AdversarialEngine(difficulty_level=0.7, seed=42)
    
    # Load data
    end_time = datetime.now()
    start_time = end_time - timedelta(days=7)
    simulator.load_data("EURUSD", start_time, end_time)
    
    # Test 1: Liquidity Zone Detection
    print("\n1. Testing Liquidity Zone Detection...")
    
    # Run a few steps to populate data
    for i in range(100):
        market_state = simulator.step()
        if market_state is None:
            break
        
        # Update liquidity zones
        adversary._update_liquidity_zones(market_state, simulator)
        
        if i == 50:  # Check mid-simulation
            print(f"   - Liquidity zones detected: {len(adversary.liquidity_zones)}")
            if adversary.liquidity_zones:
                zone = adversary.liquidity_zones[0]
                print(f"   - Sample zone: {zone['type']} at {zone['price_level']:.5f}")
                print(f"   - Confluence score: {zone['confluence_score']:.3f}")
    
    # Test 2: Consolidation Detection
    print("\n2. Testing Consolidation Detection...")
    
    # Reset and test consolidation
    simulator = MarketSimulator(data_storage, initial_balance=100000.0)
    simulator.load_data("EURUSD", start_time, end_time)
    
    for i in range(200):
        market_state = simulator.step()
        if market_state is None:
            break
        
        # Update consolidation detection
        adversary._update_consolidation_detection(market_state, simulator)
        
        if i == 100:  # Check mid-simulation
            print(f"   - Consolidation periods detected: {len(adversary.consolidation_periods)}")
            if adversary.consolidation_periods:
                consolidation = adversary.consolidation_periods[0]
                print(f"   - Sample consolidation: {consolidation['low_boundary']:.5f} - {consolidation['high_boundary']:.5f}")
                print(f"   - Consolidation score: {consolidation['consolidation_score']:.3f}")
    
    # Test 3: Intelligent Stop Hunt Triggering
    print("\n3. Testing Intelligent Stop Hunt...")
    
    # Reset and test stop hunting
    simulator = MarketSimulator(data_storage, initial_balance=100000.0)
    simulator.load_data("EURUSD", start_time, end_time)
    
    stop_hunts_triggered = 0
    for i in range(300):
        market_state = simulator.step()
        if market_state is None:
            break
        
        # Update liquidity zones
        adversary._update_liquidity_zones(market_state, simulator)
        
        # Check for stop hunt triggers
        if adversary._should_trigger_intelligent_stop_hunt(market_state, simulator):
            stop_hunts_triggered += 1
            print(f"   - Stop hunt triggered at step {i}")
    
    print(f"   - Total intelligent stop hunts triggered: {stop_hunts_triggered}")
    
    # Test 4: Breakout Trap Triggering
    print("\n4. Testing Breakout Trap...")
    
    # Reset and test breakout traps
    simulator = MarketSimulator(data_storage, initial_balance=100000.0)
    simulator.load_data("EURUSD", start_time, end_time)
    
    breakout_traps_triggered = 0
    for i in range(300):
        market_state = simulator.step()
        if market_state is None:
            break
        
        # Update consolidation detection
        adversary._update_consolidation_detection(market_state, simulator)
        
        # Check for breakout trap triggers
        if adversary._should_trigger_breakout_trap(market_state, simulator):
            breakout_traps_triggered += 1
            print(f"   - Breakout trap triggered at step {i}")
    
    print(f"   - Total breakout traps triggered: {breakout_traps_triggered}")
    
    print("\n✓ Intelligent Adversarial Engine v0.2 tests completed!")


def test_triathlon_fitness_evaluation():
    """Test v0.2 triathlon fitness evaluation framework"""
    print("\n" + "="*60)
    print("TESTING: Triathlon Fitness Evaluation v0.2")
    print("="*60)
    
    # Initialize components
    data_storage = TickDataStorage("test_data")
    cleaner = TickDataCleaner()
    ingestor = DukascopyIngestor(data_storage, cleaner)
    
    # Generate test data for all three regimes
    print("\n1. Generating test data for triathlon evaluation...")
    ingestor.ingest_year("EURUSD", 2020)  # Volatile year
    ingestor.ingest_year("EURUSD", 2021)  # Ranging year
    ingestor.ingest_year("EURUSD", 2022)  # Trending year
    
    # Initialize fitness evaluator
    fitness_evaluator = FitnessEvaluator(
        data_storage=data_storage,
        evaluation_period_days=7,
        adversarial_intensity=0.7
    )
    
    # Test 1: Regime Dataset Identification
    print("\n2. Testing regime dataset identification...")
    fitness_evaluator._identify_regime_datasets()
    
    for regime_name, regime_config in fitness_evaluator.regime_datasets.items():
        print(f"   - {regime_name}: {regime_config['name']}")
        print(f"     Period: {regime_config['start_time'].date()} to {regime_config['end_time'].date()}")
        print(f"     Characteristics: {', '.join(regime_config['characteristics'])}")
    
    # Test 2: Create a test genome
    print("\n3. Testing genome evaluation...")
    test_genome = DecisionGenome(max_depth=5, max_nodes=20)
    
    # Test 3: Evaluate genome with triathlon testing
    print("   - Running triathlon evaluation...")
    fitness_score = fitness_evaluator.evaluate_genome(test_genome)
    
    print(f"   - Total fitness: {fitness_score.total_fitness:.4f}")
    print(f"   - Regime scores:")
    if fitness_score.regime_scores:
        for regime, score in fitness_score.regime_scores.items():
            if isinstance(score, float):
                print(f"     {regime}: {score:.4f}")
    
    # Test 4: Multi-objective metrics
    print("\n4. Testing multi-objective fitness metrics...")
    print(f"   - Sortino ratio: {fitness_score.sortino_ratio:.4f}")
    print(f"   - Calmar ratio: {fitness_score.calmar_ratio:.4f}")
    print(f"   - Profit factor: {fitness_score.profit_factor:.4f}")
    print(f"   - Consistency score: {fitness_score.consistency_score:.4f}")
    print(f"   - Robustness score: {fitness_score.robustness_score:.4f}")
    
    # Test 5: Component scores
    print("\n5. Testing component scores...")
    print(f"   - Returns score: {fitness_score.returns_score:.4f}")
    print(f"   - Robustness score: {fitness_score.robustness_score:.4f}")
    print(f"   - Adaptability score: {fitness_score.adaptability_score:.4f}")
    print(f"   - Efficiency score: {fitness_score.efficiency_score:.4f}")
    print(f"   - Antifragility score: {fitness_score.antifragility_score:.4f}")
    
    print("\n✓ Triathlon Fitness Evaluation v0.2 tests completed!")


def test_individual_metrics():
    """Test individual v0.2 fitness metrics"""
    print("\n" + "="*60)
    print("TESTING: Individual v0.2 Fitness Metrics")
    print("="*60)
    
    # Initialize components
    data_storage = TickDataStorage("test_data")
    fitness_evaluator = FitnessEvaluator(data_storage, evaluation_period_days=7)
    
    # Create mock simulation results for testing
    mock_results = {
        "equity_curve": [
            {"timestamp": datetime.now() - timedelta(days=7), "equity": 100000},
            {"timestamp": datetime.now() - timedelta(days=6), "equity": 101000},
            {"timestamp": datetime.now() - timedelta(days=5), "equity": 100500},
            {"timestamp": datetime.now() - timedelta(days=4), "equity": 102000},
            {"timestamp": datetime.now() - timedelta(days=3), "equity": 101500},
            {"timestamp": datetime.now() - timedelta(days=2), "equity": 103000},
            {"timestamp": datetime.now() - timedelta(days=1), "equity": 102500},
            {"timestamp": datetime.now(), "equity": 104000}
        ],
        "trades": [
            {"timestamp": datetime.now() - timedelta(days=6), "price": 1.1000},
            {"timestamp": datetime.now() - timedelta(days=5), "price": 1.1050},
            {"timestamp": datetime.now() - timedelta(days=4), "price": 1.1100},
            {"timestamp": datetime.now() - timedelta(days=3), "price": 1.1080},
            {"timestamp": datetime.now() - timedelta(days=2), "price": 1.1150},
            {"timestamp": datetime.now() - timedelta(days=1), "price": 1.1120},
            {"timestamp": datetime.now(), "price": 1.1200}
        ],
        "adversarial_events": []
    }
    
    # Test 1: Sortino Ratio
    print("\n1. Testing Sortino Ratio calculation...")
    sortino = fitness_evaluator._calculate_sortino_ratio(mock_results)
    print(f"   - Sortino ratio: {sortino:.4f}")
    
    # Test 2: Calmar Ratio
    print("\n2. Testing Calmar Ratio calculation...")
    calmar = fitness_evaluator._calculate_calmar_ratio(mock_results)
    print(f"   - Calmar ratio: {calmar:.4f}")
    
    # Test 3: Profit Factor
    print("\n3. Testing Profit Factor calculation...")
    profit_factor = fitness_evaluator._calculate_profit_factor(mock_results)
    print(f"   - Profit factor: {profit_factor:.4f}")
    
    # Test 4: Consistency Score
    print("\n4. Testing Consistency Score calculation...")
    consistency = fitness_evaluator._calculate_consistency_score(mock_results)
    print(f"   - Consistency score: {consistency:.4f}")
    
    # Test 5: Robustness Score
    print("\n5. Testing Robustness Score calculation...")
    robustness = fitness_evaluator._calculate_robustness_score(mock_results)
    print(f"   - Robustness score: {robustness:.4f}")
    
    print("\n✓ Individual fitness metrics tests completed!")


def test_integration():
    """Test integration of v0.2 features"""
    print("\n" + "="*60)
    print("TESTING: v0.2 Feature Integration")
    print("="*60)
    
    # Initialize components
    data_storage = TickDataStorage("test_data")
    cleaner = TickDataCleaner()
    ingestor = DukascopyIngestor(data_storage, cleaner)
    
    # Generate test data
    ingestor.ingest_year("EURUSD", 2022)
    
    # Initialize all components
    simulator = MarketSimulator(data_storage, initial_balance=100000.0)
    adversary = AdversarialEngine(difficulty_level=0.7, seed=42)
    sensory_cortex = SensoryCortex("EURUSD", data_storage)
    fitness_evaluator = FitnessEvaluator(data_storage, evaluation_period_days=7)
    
    # Load data and calibrate
    end_time = datetime.now()
    start_time = end_time - timedelta(days=7)
    simulator.load_data("EURUSD", start_time, end_time)
    
    calibration_start = start_time - timedelta(days=3)
    sensory_cortex.calibrate(calibration_start, start_time)
    
    # Add adversarial callback
    simulator.add_adversarial_callback(adversary.apply_adversarial_effects)
    
    # Test integrated simulation
    print("\n1. Running integrated v0.2 simulation...")
    
    simulation_data = {
        "trades": [],
        "equity_curve": [],
        "decisions": [],
        "adversarial_events": [],
        "sensory_readings": []
    }
    
    # Create a test genome
    test_genome = DecisionGenome(max_depth=5, max_nodes=20)
    
    step_count = 0
    max_steps = 100  # Short test
    
    while step_count < max_steps:
        # Step simulator
        market_state = simulator.step()
        
        if market_state is None:
            break
        
        # Get sensory reading
        sensory_reading = sensory_cortex.perceive(market_state)
        simulation_data["sensory_readings"].append(sensory_reading)
        
        # Make decision
        action = test_genome.decide(sensory_reading)
        
        if action:
            simulation_data["decisions"].append({
                "timestamp": market_state.timestamp,
                "action": action,
                "sensory_reading": sensory_reading,
                "market_state": market_state
            })
        
        # Record equity
        account_summary = simulator.get_account_summary()
        simulation_data["equity_curve"].append({
            "timestamp": market_state.timestamp,
            "equity": account_summary["equity"],
            "balance": account_summary["balance"],
            "positions": account_summary["positions"]
        })
        
        # Record adversarial events
        active_events = adversary.get_active_events()
        if active_events:
            simulation_data["adversarial_events"].extend(active_events)
        
        step_count += 1
    
    print(f"   - Simulation completed: {step_count} steps")
    print(f"   - Final equity: {simulation_data['equity_curve'][-1]['equity']:.2f}")
    print(f"   - Decisions made: {len(simulation_data['decisions'])}")
    print(f"   - Adversarial events: {len(simulation_data['adversarial_events'])}")
    print(f"   - Sensory readings: {len(simulation_data['sensory_readings'])}")
    
    # Test 2: Evaluate genome with integrated results
    print("\n2. Testing integrated fitness evaluation...")
    
    # Create fitness score from simulation data
    fitness_score = FitnessScore(genome_id=test_genome.genome_id)
    
    # Calculate metrics
    sortino = fitness_evaluator._calculate_sortino_ratio(simulation_data)
    calmar = fitness_evaluator._calculate_calmar_ratio(simulation_data)
    profit_factor = fitness_evaluator._calculate_profit_factor(simulation_data)
    consistency = fitness_evaluator._calculate_consistency_score(simulation_data)
    robustness = fitness_evaluator._calculate_robustness_score(simulation_data)
    
    print(f"   - Sortino ratio: {sortino:.4f}")
    print(f"   - Calmar ratio: {calmar:.4f}")
    print(f"   - Profit factor: {profit_factor:.4f}")
    print(f"   - Consistency score: {consistency:.4f}")
    print(f"   - Robustness score: {robustness:.4f}")
    
    print("\n✓ v0.2 Feature Integration tests completed!")


def main():
    """Run all v0.2 feature tests"""
    print("EMP PROVING GROUND v0.2 FEATURE TEST SUITE")
    print("="*60)
    print("Testing major v0.2 upgrade features...")
    
    try:
        # Test intelligent adversarial engine
        test_intelligent_adversarial_engine()
        
        # Test triathlon fitness evaluation
        test_triathlon_fitness_evaluation()
        
        # Test individual metrics
        test_individual_metrics()
        
        # Test integration
        test_integration()
        
        print("\n" + "="*60)
        print("ALL v0.2 FEATURE TESTS COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\nv0.2 Upgrade Summary:")
        print("✓ Intelligent Stop Hunt Module (Liquidity Zone Hunter)")
        print("✓ Intelligent Spoofing Module (Breakout Trap)")
        print("✓ Triathlon Evaluation Framework")
        print("✓ Multi-objective Fitness Metrics (Sortino, Calmar, Profit Factor)")
        print("✓ Robustness Testing with Dual Adversarial Levels")
        print("✓ Anti-overfitting Penalty for Regime Inconsistency")
        print("✓ Enhanced Consolidation Detection")
        print("✓ Comprehensive Component Scoring")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main()) 