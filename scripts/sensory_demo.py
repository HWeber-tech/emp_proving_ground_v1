#!/usr/bin/env python3
"""
Sensory Demo Script - Smoke Run Test

Feeds 100,000 EURUSD ticks through the sensory fusion loop to verify end-to-end functionality.
This script validates that the sensory cortex can process high-volume tick data without crashes
and produces meaningful dimensional readings across all 5 dimensions (WHY, HOW, WHAT, WHEN, ANOMALY).

Usage:
    python scripts/sensory_demo.py

Expected Output:
    - Progress updates every 1000 ticks
    - Dimensional readings table showing WHY/HOW/WHAT/WHEN/ANOMALY values
    - Final statistics with >90% success rate
"""

import asyncio
import logging
import os
import sys
from datetime import datetime, timedelta
from typing import Any

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.core.base import DimensionalReading
from src.orchestration.enhanced_understanding_engine import ContextualFusionEngine
from src.sensory.organs.dimensions.base_organ import InstrumentMeta, MarketData

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class TickGenerator:
    """Generates realistic EURUSD tick data for testing"""

    def __init__(self, initial_price: float = 1.1000, spread_pips: float = 2.0):
        self.current_price = initial_price
        self.spread = spread_pips * 0.0001  # Convert pips to price units
        self.tick_count = 0

        self.volatility = 0.0001  # Base volatility per tick
        self.trend_strength = 0.0  # Current trend bias
        self.volume_base = 1000000  # Base volume

    def generate_tick(self, timestamp: datetime) -> MarketData:
        """Generate a single realistic tick"""

        price_change = (
            np.random.normal(0, self.volatility)  # Random walk
            + self.trend_strength * 0.00001  # Trend component
            + -0.1 * (self.current_price - 1.1000) * 0.0001  # Mean reversion to 1.1000
        )

        self.current_price += price_change

        self.current_price = max(1.0500, min(1.1500, self.current_price))

        mid_price = self.current_price
        bid = mid_price - self.spread / 2
        ask = mid_price + self.spread / 2

        volume = max(100000, np.random.lognormal(np.log(self.volume_base), 0.5))

        tick_range = self.volatility * 0.5
        high = mid_price + np.random.uniform(0, tick_range)
        low = mid_price - np.random.uniform(0, tick_range)
        open_price = mid_price + np.random.uniform(-tick_range / 2, tick_range / 2)
        close_price = mid_price

        if self.tick_count % 10000 == 0:
            self.trend_strength = np.random.uniform(-1, 1)

        self.tick_count += 1

        return MarketData(
            symbol="EURUSD",
            timestamp=timestamp,
            open=open_price,
            high=high,
            low=low,
            close=close_price,
            volume=volume,
            bid=bid,
            ask=ask,
            source="demo_generator",
            latency_ms=np.random.exponential(2.0),  # Realistic latency
        )


class SensoryDemo:
    """Main demo class that orchestrates the smoke run test"""

    def __init__(self):
        self.tick_generator = TickGenerator()
        self.fusion_engine = None
        self.stats = {
            "total_ticks": 0,
            "successful_updates": 0,
            "failed_updates": 0,
            "errors": [],
            "start_time": None,
            "end_time": None,
        }

    async def initialize_fusion_engine(self):
        """Initialize the contextual fusion engine"""
        try:
            instrument_meta = InstrumentMeta(
                symbol="EURUSD",
                pip_size=0.0001,
                lot_size=100000,
                timezone="UTC",
                sessions={
                    "ASIAN": ("00:00", "09:00"),
                    "LONDON": ("08:00", "17:00"),
                    "NEW_YORK": ("13:00", "22:00"),
                },
                typical_spread=0.0002,
                avg_daily_range=0.01,
            )

            self.fusion_engine = ContextualFusionEngine()
            logger.info("Contextual fusion engine initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize fusion engine: {e}")
            raise

    async def run_smoke_test(self, num_ticks: int = 100000):
        """Run the main smoke test with specified number of ticks"""

        logger.info(f"Starting sensory demo with {num_ticks:,} ticks")
        self.stats["start_time"] = datetime.utcnow()

        await self.initialize_fusion_engine()

        print("\n" + "=" * 80)
        print("SENSORY CORTEX SMOKE RUN TEST")
        print("=" * 80)
        print(f"Target: {num_ticks:,} EURUSD ticks")
        print(f"Started: {self.stats['start_time']}")
        print("=" * 80)

        base_time = datetime.utcnow()

        for i in range(num_ticks):
            try:
                tick_time = base_time + timedelta(milliseconds=i * 100)  # 10 ticks per second
                market_data = self.tick_generator.generate_tick(tick_time)

                market_understanding = await self.fusion_engine.analyze_market_understanding(
                    market_data
                )

                self.stats["successful_updates"] += 1

                if (i + 1) % 1000 == 0:
                    await self._print_progress_update(i + 1, market_understanding, market_data)

            except Exception as e:
                self.stats["failed_updates"] += 1
                self.stats["errors"].append(f"Tick {i + 1}: {str(e)}")
                logger.error(f"Error processing tick {i + 1}: {e}")

                if self.stats["failed_updates"] > num_ticks * 0.1:  # More than 10% failure rate
                    logger.error("Too many failures, stopping demo")
                    break

            self.stats["total_ticks"] = i + 1

        self.stats["end_time"] = datetime.utcnow()
        await self._print_final_statistics()

    async def _print_progress_update(
        self, tick_count: int, market_understanding: Any, market_data: MarketData
    ):
        """Print progress update with dimensional readings"""

        progress = (tick_count / 100000) * 100

        readings = {}
        if hasattr(market_understanding, "dimensional_readings"):
            for dim, reading in market_understanding.dimensional_readings.items():
                if isinstance(reading, DimensionalReading):
                    readings[dim] = {
                        "value": reading.signal_strength,
                        "confidence": reading.confidence,
                    }
                else:
                    readings[dim] = {"value": 0.0, "confidence": 0.0}

        print(
            f"\n[{progress:5.1f}%] Tick {tick_count:,}/100,000 | Price: {market_data.mid_price:.5f}"
        )
        print("┌─────────┬─────────┬────────────┐")
        print("│ Dimension│  Value  │ Confidence │")
        print("├─────────┼─────────┼────────────┤")

        for dim in ["WHY", "HOW", "WHAT", "WHEN", "ANOMALY"]:
            if dim in readings:
                value = readings[dim]["value"]
                confidence = readings[dim]["confidence"]
                print(f"│ {dim:8s}│ {value:+7.3f} │ {confidence:10.3f} │")
            else:
                print(f"│ {dim:8s}│   N/A   │    N/A     │")

        print("└─────────┴─────────┴────────────┘")

        success_rate = (self.stats["successful_updates"] / tick_count) * 100
        print(f"Success Rate: {success_rate:.1f}% | Errors: {self.stats['failed_updates']}")

    async def _print_final_statistics(self):
        """Print final test statistics"""

        duration = (self.stats["end_time"] - self.stats["start_time"]).total_seconds()
        success_rate = (self.stats["successful_updates"] / self.stats["total_ticks"]) * 100
        ticks_per_second = self.stats["total_ticks"] / duration if duration > 0 else 0

        print("\n" + "=" * 80)
        print("FINAL STATISTICS")
        print("=" * 80)
        print(f"Total Ticks Processed: {self.stats['total_ticks']:,}")
        print(f"Successful Updates:    {self.stats['successful_updates']:,}")
        print(f"Failed Updates:        {self.stats['failed_updates']:,}")
        print(f"Success Rate:          {success_rate:.2f}%")
        print(f"Processing Duration:   {duration:.2f} seconds")
        print(f"Processing Speed:      {ticks_per_second:.1f} ticks/second")
        print("=" * 80)

        if self.stats["errors"]:
            print("\nERROR SUMMARY:")
            print("-" * 40)
            error_counts = {}
            for error in self.stats["errors"][:10]:  # Show first 10 errors
                error_type = error.split(":")[1].strip() if ":" in error else error
                error_counts[error_type] = error_counts.get(error_type, 0) + 1

            for error_type, count in error_counts.items():
                print(f"  {error_type}: {count} occurrences")

            if len(self.stats["errors"]) > 10:
                print(f"  ... and {len(self.stats['errors']) - 10} more errors")

        print("\nVERDICT:")
        if success_rate >= 90:
            print("✅ SMOKE TEST PASSED - Sensory fusion loop is stable")
        elif success_rate >= 75:
            print("⚠️  SMOKE TEST MARGINAL - Some stability issues detected")
        else:
            print("❌ SMOKE TEST FAILED - Significant stability problems")

        print("=" * 80)


async def main():
    """Main entry point"""
    try:
        demo = SensoryDemo()
        await demo.run_smoke_test(num_ticks=100000)

    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user")

    except Exception as e:
        logger.error(f"Demo failed with error: {e}")
        print(f"\n❌ Demo failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
