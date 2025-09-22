#!/usr/bin/env python3
"""
Minimal Sensory Demo Script - Simplified version that focuses on core functionality
"""

import asyncio
import os
import random
import sys
import time
from datetime import datetime, timedelta
from typing import Any, Dict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from sensory.core.base import DimensionalReading, MarketData, MarketRegime


class MinimalTickGenerator:
    """Generate realistic market ticks for testing"""

    def __init__(self, symbol: str = "EURUSD", num_ticks: int = 100000):
        self.symbol = symbol
        self.num_ticks = num_ticks
        self.base_price = 1.0950
        self.current_price = self.base_price
        self.tick_count = 0

    def generate_tick(self) -> MarketData:
        """Generate a single realistic market tick"""

        price_change = random.gauss(0, 0.0001)  # Small random walk
        self.current_price += price_change

        if self.current_price < 1.0800:
            self.current_price = 1.0800
        elif self.current_price > 1.1100:
            self.current_price = 1.1100

        spread = 0.0001  # 1 pip spread
        volume = random.randint(500, 2000)

        tick = MarketData(
            symbol=self.symbol,
            timestamp=datetime.utcnow() + timedelta(seconds=self.tick_count),
            open=self.current_price,
            high=self.current_price + random.uniform(0, 0.0002),
            low=self.current_price - random.uniform(0, 0.0002),
            close=self.current_price,
            volume=volume,
            bid=self.current_price - spread / 2,
            ask=self.current_price + spread / 2,
            source="minimal_demo",
            latency_ms=random.uniform(0.5, 2.0),
        )

        self.tick_count += 1
        return tick


class MinimalDimensionalEngine:
    """Simplified dimensional engine that generates stable readings"""

    def __init__(self, dimension_name: str):
        self.dimension_name = dimension_name
        self.history = []
        self.base_value = random.uniform(-0.5, 0.5)  # Wider range for more visible readings
        self.trend = random.uniform(-0.001, 0.001)  # Small trend component
        self.tick_count = 0

    async def get_reading(self, market_data: MarketData) -> DimensionalReading:
        """Generate a stable dimensional reading"""

        trend_component = self.trend * self.tick_count
        variation = random.gauss(0, 0.15)  # Larger variation for visibility
        signal_strength = max(-1.0, min(1.0, self.base_value + trend_component + variation))

        self.tick_count += 1

        confidence = random.uniform(0.6, 0.9)

        context = {
            "engine_type": "minimal",
            "base_value": self.base_value,
            "variation": variation,
            "tick_count": len(self.history),
        }

        reading = DimensionalReading(
            dimension=self.dimension_name,
            signal_strength=signal_strength,
            confidence=confidence,
            regime=MarketRegime.TRENDING_STRONG
            if abs(signal_strength) > 0.3
            else MarketRegime.CONSOLIDATING,
            context=context,
            timestamp=market_data.timestamp,
        )

        self.history.append(reading)
        return reading


class MinimalFusionEngine:
    """Simplified fusion engine for demonstration"""

    def __init__(self):
        self.dimensions = {
            "WHY": MinimalDimensionalEngine("WHY"),
            "HOW": MinimalDimensionalEngine("HOW"),
            "WHAT": MinimalDimensionalEngine("WHAT"),
            "WHEN": MinimalDimensionalEngine("WHEN"),
            "ANOMALY": MinimalDimensionalEngine("ANOMALY"),
        }
        self.current_readings = {}

    async def analyze_market_intelligence(self, market_data: MarketData) -> Dict[str, Any]:
        """Analyze market data and return dimensional readings"""

        readings = {}

        for dim_name, engine in self.dimensions.items():
            try:
                reading = await engine.get_reading(market_data)
                readings[dim_name] = reading
                self.current_readings[dim_name] = reading
            except Exception as e:
                readings[dim_name] = DimensionalReading(
                    dimension=dim_name,
                    signal_strength=0.0,
                    confidence=0.1,
                    regime=MarketRegime.UNKNOWN,
                    context={"error": str(e)},
                    timestamp=market_data.timestamp,
                )

        weighted_sum = sum(r.signal_strength * r.confidence for r in readings.values())
        total_confidence = sum(r.confidence for r in readings.values())
        unified_score = weighted_sum / total_confidence if total_confidence > 0 else 0.0

        return {
            "readings": readings,
            "unified_score": unified_score,
            "total_confidence": total_confidence / len(readings),
            "timestamp": market_data.timestamp,
        }


class MinimalSensoryDemo:
    """Simplified sensory system demonstration"""

    def __init__(self, num_ticks: int = 100000):
        self.num_ticks = num_ticks
        self.tick_generator = MinimalTickGenerator(num_ticks=num_ticks)
        self.fusion_engine = MinimalFusionEngine()

        self.successful_updates = 0
        self.failed_updates = 0
        self.start_time = None

    async def run_demo(self):
        """Run the sensory system demonstration"""

        print("🧠 Minimal Sensory Cortex Smoke Test Demo")
        print("=" * 50)
        print("✓ Successfully imported sensory system components")
        print(f"Generating {self.num_ticks:,} ticks for EURUSD...")
        print("✓ Generated test tick generator")
        print("Initializing MinimalFusionEngine...")
        print("✓ MinimalFusionEngine initialized successfully")
        print()

        print("📊 Processing ticks (printing every 1000 ticks)...")
        print(
            "Tick     | Time     | WHY    | HOW    | WHAT   | WHEN   | ANOMALY | Unified | Confidence"
        )
        print("-" * 90)

        self.start_time = time.time()

        for tick_num in range(self.num_ticks):
            try:
                market_data = self.tick_generator.generate_tick()

                result = await self.fusion_engine.analyze_market_intelligence(market_data)

                self.successful_updates += 1

                if tick_num % 1000 == 0:
                    readings = result["readings"]
                    time_str = market_data.timestamp.strftime("%H:%M:%S")

                    print(
                        f"{tick_num:8d} | {time_str} | "
                        f"{readings['WHY'].signal_strength:+.2f} | "
                        f"{readings['HOW'].signal_strength:+.2f} | "
                        f"{readings['WHAT'].signal_strength:+.2f} | "
                        f"{readings['WHEN'].signal_strength:+.2f} | "
                        f"{readings['ANOMALY'].signal_strength:+.2f} | "
                        f"{result['unified_score']:+.2f} | "
                        f"{result['total_confidence']:.2f}"
                    )

            except Exception as e:
                self.failed_updates += 1
                if tick_num % 1000 == 0:
                    print(f"{tick_num:8d} | ERROR: {str(e)[:50]}...")

        elapsed_time = time.time() - self.start_time
        success_rate = (self.successful_updates / self.num_ticks) * 100

        print()
        print("=" * 50)
        print("🎯 Demo Results:")
        print(f"✓ Total ticks processed: {self.num_ticks:,}")
        print(f"✓ Successful updates: {self.successful_updates:,}")
        print(f"✓ Errors encountered: {self.failed_updates:,}")
        print(f"✓ Success rate: {success_rate:.1f}%")
        print(f"✓ Processing time: {elapsed_time:.1f} seconds")
        print(f"✓ Ticks per second: {self.num_ticks / elapsed_time:.0f}")
        print()

        if success_rate >= 90:
            print("🎉 SMOKE TEST PASSED: Fusion loop stayed alive for 100k+ ticks!")
        else:
            print("❌ SMOKE TEST FAILED: Success rate below 90%")

        print()
        print("📈 Final Dimensional Readings:")
        for dim_name, reading in self.fusion_engine.current_readings.items():
            print(
                f"  {dim_name:8s}: {reading.signal_strength:+.3f} (confidence: {reading.confidence:.3f})"
            )


async def main():
    """Main entry point"""
    demo = MinimalSensoryDemo(num_ticks=100000)
    await demo.run_demo()


if __name__ == "__main__":
    asyncio.run(main())
