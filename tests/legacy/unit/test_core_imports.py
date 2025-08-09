#!/usr/bin/env python3
"""
Core Import Verification Test
Tests all major components of the EMP system for import compatibility.
"""

import sys
import os
import logging
from typing import Dict, List, Any

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_core_imports():
    """Test core module imports."""
    print("Testing Core Module Imports...")
    
    try:
        # Core modules
        from src.core import Instrument
        print("✅ src.core imported successfully")
        
        from src.data import TickDataStorage
        print("✅ src.data imported successfully")
        
        from src.risk import RiskManager
        print("✅ src.risk imported successfully")
        
        from src.pnl import EnhancedPosition, TradeRecord
        print("✅ src.pnl imported successfully")
        
        from src.simulation import MarketSimulator
        print("✅ src.simulation imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ Core import failed: {e}")
        return False

def test_sensory_imports():
    """Test sensory cortex imports."""
    print("\nTesting Sensory Cortex Imports...")
    
    try:
        # Sensory core
        from src.sensory.core.base import InstrumentMeta
        print("✅ src.sensory.core.base imported successfully")
        
        from src.sensory.core.data_integration import OrderFlowDataProvider, OrderBookSnapshot
        print("✅ src.sensory.core.data_integration imported successfully")
        
        # Sensory dimensions
        from src.sensory.dimensions.why_engine import WHYEngine
        print("✅ src.sensory.dimensions.why_engine imported successfully")
        
        from src.sensory.dimensions.how_engine import HOWEngine
        print("✅ src.sensory.dimensions.how_engine imported successfully")
        
        from src.sensory.dimensions.what_engine import WATEngine
        print("✅ src.sensory.dimensions.what_engine imported successfully")
        
        from src.sensory.dimensions.when_engine import WHENEngine
        print("✅ src.sensory.dimensions.when_engine imported successfully")
        
        from src.sensory.dimensions.anomaly_engine import ANOMALYEngine
        print("✅ src.sensory.dimensions.anomaly_engine imported successfully")
        
        # Sensory orchestration
        from src.sensory.orchestration.master_orchestrator import MasterOrchestrator
        print("✅ src.sensory.orchestration.master_orchestrator imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ Sensory import failed: {e}")
        return False

def test_evolution_imports():
    """Test evolution system imports."""
    print("\nTesting Evolution System Imports...")
    
    try:
        from src.evolution import (
            DecisionGenome,
            EvolutionConfig,
            FitnessEvaluator,
            EvolutionEngine
        )
        print("✅ src.evolution imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ Evolution import failed: {e}")
        return False

def test_utility_imports():
    """Test utility and configuration imports."""
    print("\nTesting Utility Imports...")
    
    try:
        # Check if config exists
        import yaml
        print("✅ yaml imported successfully")
        
        # Check if we can read config
        config_path = "config.yaml"
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            print("✅ config.yaml loaded successfully")
        else:
            print("⚠️  config.yaml not found")
        
        return True
        
    except ImportError as e:
        print(f"❌ Utility import failed: {e}")
        return False

def main():
    """Run all import tests."""
    print("=" * 60)
    print("EMP SYSTEM - CORE IMPORT VERIFICATION")
    print("=" * 60)
    
    results = {}
    
    # Test each component
    results['core'] = test_core_imports()
    results['sensory'] = test_sensory_imports()
    results['evolution'] = test_evolution_imports()
    results['utility'] = test_utility_imports()
    
    # Summary
    print("\n" + "=" * 60)
    print("IMPORT VERIFICATION SUMMARY")
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
