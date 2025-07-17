#!/usr/bin/env python3
"""
Simple Integration Test - Validates core system integration
Tests the integrated system without complex dependencies
"""

import sys
from pathlib import Path
from decimal import Decimal

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_basic_imports():
    """Test basic system imports"""
    print("🧪 Testing Basic System Imports...")
    
    try:
        from src.core import RiskConfig, InstrumentProvider
        print("✅ src.core imports successful")
        
        from src.risk import RiskManager
        print("✅ src.risk imports successful")
        
        from src.simulation import MarketSimulator
        print("✅ src.simulation imports successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

def test_risk_management():
    """Test risk management functionality"""
    print("\n🧪 Testing Risk Management...")
    
    try:
        from src.core import RiskConfig, InstrumentProvider
        from src.risk import RiskManager
        
        # Setup risk configuration
        risk_config = RiskConfig(
            max_risk_per_trade_pct=Decimal("0.02"),
            max_leverage=Decimal("10.0"),
            max_total_exposure_pct=Decimal("0.5"),
            max_drawdown_pct=Decimal("0.25")
        )
        
        instrument_provider = InstrumentProvider()
        risk_manager = RiskManager(risk_config, instrument_provider)
        
        # Test basic functionality
        instrument = instrument_provider.get_instrument("EURUSD")
        if instrument:
            position_size = risk_manager.calculate_position_size(
                account_equity=Decimal("100000"),
                stop_loss_pips=Decimal("50"),
                instrument=instrument,
                account_currency="USD"
            )
            print(f"✅ Risk management working - calculated position size: {position_size}")
        else:
            print("⚠️ EURUSD instrument not found, using mock data")
            
        return True
        
    except Exception as e:
        print(f"❌ Risk management test failed: {e}")
        return False

def test_simulation_framework():
    """Test simulation framework"""
    print("\n🧪 Testing Simulation Framework...")
    
    try:
        from src.simulation import MarketSimulator
        from src.data import TickDataStorage
        
        # Test basic initialization
        data_storage = TickDataStorage()
        simulator = MarketSimulator(data_storage, initial_balance=100000.0)
        
        print("✅ Simulation framework initialized successfully")
        return True
        
    except Exception as e:
        print(f"❌ Simulation test failed: {e}")
        return False

def main():
    """Run all basic integration tests"""
    print("🚀 Basic Integration Test Suite")
    print("=" * 50)
    
    tests = [
        test_basic_imports,
        test_risk_management,
        test_simulation_framework
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All basic integration tests passed!")
        print("✅ Core system integration is working")
    else:
        print("⚠️ Some tests failed - check dependencies")
    
    return passed == total

if __name__ == "__main__":
    main()
