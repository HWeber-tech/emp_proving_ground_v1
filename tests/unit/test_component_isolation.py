#!/usr/bin/env python3
"""
Component-Level Testing
Tests each major component of the EMP system in isolation.
"""

import sys
import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from decimal import Decimal

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_core_components():
    """Test core components in isolation."""
    print("Testing Core Components...")
    
    try:
        from src.core import Instrument
        
        # Test Instrument creation
        instrument = Instrument(
            symbol="EURUSD",
            pip_decimal_places=4,
            contract_size=Decimal('100000'),
            long_swap_rate=Decimal('-0.0001'),
            short_swap_rate=Decimal('0.0001'),
            margin_currency="USD",
            swap_time="22:00"
        )
        print("✅ Instrument creation successful")
        
        # Test PnL components
        from src.pnl import EnhancedPosition, TradeRecord
        
        position = EnhancedPosition(
            symbol="EURUSD",
            quantity=10000,
            avg_price=Decimal('1.1000'),
            entry_timestamp=datetime.now(),
            last_swap_time=datetime.now()
        )
        print("✅ EnhancedPosition creation successful")
        
        # Test Risk components
        from src.risk import RiskManager, RiskConfig
        from src.core import InstrumentProvider
        
        risk_config = RiskConfig(
            max_risk_per_trade_pct=Decimal('0.02'),
            max_leverage=Decimal('10.0'),
            max_total_exposure_pct=Decimal('0.5'),
            max_drawdown_pct=Decimal('0.25'),
            min_position_size=1000,
            max_position_size=1000000
        )
        
        instrument_provider = InstrumentProvider()
        risk_manager = RiskManager(risk_config, instrument_provider)
        print("✅ RiskManager creation successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Core component test failed: {e}")
        return False

def test_data_components():
    """Test data handling components."""
    print("\nTesting Data Components...")
    
    try:
        from src.data import TickDataStorage
        
        # Create mock data storage
        storage = TickDataStorage()
        print("✅ TickDataStorage creation successful")
        
        # Test synthetic data creation
        synthetic_data = create_synthetic_market_data()
        print("✅ Synthetic data creation successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Data component test failed: {e}")
        return False

def test_sensory_components():
    """Test sensory cortex components."""
    print("\nTesting Sensory Cortex Components...")
    
    try:
        from src.sensory.core.base import InstrumentMeta
        from src.sensory.dimensions.why_engine import WHYEngine
        from src.sensory.dimensions.how_engine import HOWEngine
        from src.sensory.dimensions.what_engine import WATEngine
        from src.sensory.dimensions.when_engine import WHENEngine
        from src.sensory.dimensions.anomaly_engine import ANOMALYEngine
        
        # Create instrument meta
        instrument_meta = InstrumentMeta(
            symbol="EURUSD",
            pip_size=0.0001,
            lot_size=100000,
            timezone="UTC",
            typical_spread=0.00015,
            avg_daily_range=0.01
        )
        print("✅ InstrumentMeta creation successful")
        
        # Test each engine
        why_engine = WHYEngine(instrument_meta)
        print("✅ WHYEngine creation successful")
        
        how_engine = HOWEngine(instrument_meta)
        print("✅ HOWEngine creation successful")
        
        what_engine = WATEngine(instrument_meta)
        print("✅ WATEngine creation successful")
        
        when_engine = WHENEngine(instrument_meta)
        print("✅ WHENEngine creation successful")
        
        anomaly_engine = ANOMALYEngine(instrument_meta)
        print("✅ ANOMALYEngine creation successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Sensory component test failed: {e}")
        return False

def test_evolution_components():
    """Test evolution system components."""
    print("\nTesting Evolution Components...")
    
    try:
        from src.evolution import (
            DecisionGenome, 
            EvolutionConfig, 
            FitnessEvaluator, 
            EvolutionEngine
        )
        
        # Test configuration
        config = EvolutionConfig(
            population_size=10,
            elite_ratio=0.2,
            crossover_ratio=0.5,
            mutation_ratio=0.3,
            tournament_size=2,
            max_generations=5,
            convergence_threshold=0.001,
            stagnation_limit=3
        )
        print("✅ EvolutionConfig creation successful")
        
        # Test genome creation
        genome = DecisionGenome(
            genome_id="test_genome",
            decision_tree={
                'type': 'test',
                'parameters': {
                    'buy_threshold': 0.5,
                    'sell_threshold': 0.5,
                    'momentum_weight': 0.3,
                    'trend_weight': 0.4,
                    'institutional_weight': 0.3
                }
            }
        )
        print("✅ DecisionGenome creation successful")
        
        # Test fitness evaluator (with mock storage)
        class MockStorage:
            def get_data_range(self, instrument, start, end):
                return create_synthetic_market_data()
        
        fitness_evaluator = FitnessEvaluator(MockStorage(), "EURUSD")
        print("✅ FitnessEvaluator creation successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Evolution component test failed: {e}")
        return False

def create_synthetic_market_data():
    """Create synthetic market data for testing."""
    np.random.seed(42)
    
    # Generate 100 data points
    dates = pd.date_range(start='2023-01-01', periods=100, freq='H')
    
    # Generate realistic price movements
    base_price = 1.1000
    prices = [base_price]
    
    for i in range(99):
        change = np.random.normal(0.0001, 0.001)
        new_price = prices[-1] + change
        prices.append(max(0.5, new_price))
    
    # Create OHLCV data
    data = []
    for i in range(0, len(prices), 5):
        if i + 5 <= len(prices):
            bar_prices = prices[i:i+5]
            data.append({
                'open': bar_prices[0],
                'high': max(bar_prices),
                'low': min(bar_prices),
                'close': bar_prices[-1],
                'volume': np.random.randint(1000, 10000)
            })
    
    df = pd.DataFrame(data)
    df.index = pd.date_range(start='2023-01-01', periods=len(df), freq='H')
    return df

def main():
    """Run all component tests."""
    print("=" * 60)
    print("EMP SYSTEM - COMPONENT-LEVEL TESTING")
    print("=" * 60)
    
    results = {}
    
    # Test each component
    results['core'] = test_core_components()
    results['data'] = test_data_components()
    results['sensory'] = test_sensory_components()
    results['evolution'] = test_evolution_components()
    
    # Summary
    print("\n" + "=" * 60)
    print("COMPONENT TESTING SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for component, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{component.upper():12} : {status}")
        if not passed:
            all_passed = False
    
    print(f"\nOverall Status: {'✅ ALL PASSED' if all_passed else '❌ SOME FAILED'}")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 