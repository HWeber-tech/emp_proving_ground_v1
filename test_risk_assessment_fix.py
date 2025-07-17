#!/usr/bin/env python3

import sys
import traceback
from datetime import datetime
import pandas as pd
import numpy as np

sys.path.append('src')

from sensory import SensoryCortex, InstrumentMeta, MarketData

async def test_risk_assessment_fix():
    """Test that risk_assessment KeyError is fixed"""
    
    print("✓ Creating synthetic market data for testing")
    
    base_time = datetime(2024, 5, 15, 10, 0, 0)
    synthetic_data = []
    
    for i in range(10):
        tick_data = {
            'timestamp': base_time + pd.Timedelta(seconds=i*60),
            'bid': 1.0850 + np.random.normal(0, 0.0001),
            'ask': 1.0852 + np.random.normal(0, 0.0001),
            'volume': 1000 + np.random.randint(0, 500),
            'open': 1.0851,
            'high': 1.0853,
            'low': 1.0849,
            'close': 1.0851
        }
        synthetic_data.append(tick_data)
    
    ticks = pd.DataFrame(synthetic_data)
    
    meta = InstrumentMeta(
        symbol='EURUSD',
        pip_size=0.0001,
        lot_size=100_000,
        timezone='UTC',
        sessions={
            'london': ('07:00', '15:30'),
            'ny': ('12:00', '21:00')
        }
    )
    
    try:
        cortex = SensoryCortex(instrument_meta=meta)
        print("✓ SensoryCortex created successfully")
    except Exception as e:
        print(f"✗ Error creating SensoryCortex: {e}")
        traceback.print_exc()
        return False
    
    success_count = 0
    for i, row in enumerate(ticks.to_dict('records'), 1):
        try:
            market_data = MarketData(
                symbol='EURUSD',
                timestamp=row['timestamp'],
                open=row['open'],
                high=row['high'],
                low=row['low'],
                close=row['close'],
                volume=row['volume'],
                bid=row['bid'],
                ask=row['ask']
            )
            understanding = await cortex.update(market_data)
            
            print(f"✓ Tick {i}: Success - Signal: {understanding.signal_strength:.3f}, Confidence: {understanding.confidence:.3f}")
            success_count += 1
            
        except KeyError as e:
            if 'risk_assessment' in str(e):
                print(f"✗ Tick {i}: risk_assessment KeyError still present - {e}")
                return False
            else:
                print(f"✗ Tick {i}: Other KeyError - {e}")
                return False
                
        except Exception as e:
            print(f"✗ Tick {i}: Other error - {e}")
            traceback.print_exc()
            return False
    
    print(f"\n✅ All {success_count} ticks processed successfully!")
    print("✅ risk_assessment KeyError fix confirmed working!")
    return True

if __name__ == "__main__":
    import asyncio
    success = asyncio.run(test_risk_assessment_fix())
    sys.exit(0 if success else 1)
