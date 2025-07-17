#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, '/home/ubuntu/repos/emp_proving_ground_v1/src')

from sensory.core.base import MarketData, InstrumentMeta
from sensory.orchestration.enhanced_intelligence_engine import ContextualFusionEngine
from datetime import datetime

async def test_basic_functionality():
    print("Testing basic sensory system functionality...")
    
    market_data = MarketData(
        symbol='EURUSD',
        timestamp=datetime.now(),
        bid=1.0950,
        ask=1.0952,
        open=1.0951,
        high=1.0955,
        low=1.0948,
        close=1.0951,
        volume=1000.0,
        volatility=0.01
    )
    
    instrument_meta = InstrumentMeta(
        symbol='EURUSD',
        pip_size=0.0001,
        lot_size=100000,
        timezone='UTC',
        sessions={'london': ('07:00', '15:30')}
    )
    
    try:
        fusion_engine = ContextualFusionEngine()
        
        result = await fusion_engine.analyze_market_intelligence(market_data)
        print(f"✓ Fusion engine created and analyzed data successfully")
        print(f"  Unified score: {result.unified_score:.3f}")
        print(f"  Confidence: {result.confidence:.3f}")
        return True
        
    except Exception as e:
        print(f"✗ Error in fusion engine: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    import asyncio
    success = asyncio.run(test_basic_functionality())
    sys.exit(0 if success else 1)
