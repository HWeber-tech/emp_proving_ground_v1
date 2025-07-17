#!/usr/bin/env python3
import sys
import os
sys.path.append('src')

def test_imports():
    try:
        from sensory.core.base import MarketRegime
        print("✓ MarketRegime import successful")
        
        required_members = ['UNKNOWN', 'VOLATILE', 'TRENDING_BULL', 'TRENDING_BEAR', 'RANGING_HIGH_VOL', 'RANGING_LOW_VOL', 'TRANSITION', 'CRISIS']
        for member in required_members:
            assert hasattr(MarketRegime, member), f"Missing MarketRegime.{member}"
        print("✓ All required MarketRegime members present")
        
        from sensory.orchestration.enhanced_intelligence_engine import ContextualFusionEngine
        print("✓ ContextualFusionEngine import successful")
        
        data_dir = 'src/sensory/data'
        required_files = ['yield_curve.csv', 'risk_indexes.csv', 'policy_rates.csv']
        for file in required_files:
            assert os.path.exists(os.path.join(data_dir, file)), f"Missing {file}"
        print("✓ All required CSV files present")
        
        print("\n🎉 All system stabilization fixes verified successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)
