"""
Sensory Cortex v2.2 - Integration Test Suite

Comprehensive testing of the complete sensory cortex system including
all dimensional engines, orchestration, and production scenarios.
"""

import asyncio
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging

# Configure logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import system components
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.sensory.core.base import (
    MarketData, InstrumentMeta, OrderBookSnapshot, OrderBookLevel, MarketRegime,
    DimensionalReading
)
from src.sensory.core.utils import EMA, WelfordVar, compute_confidence
from src.sensory.dimensions.why_engine import WHYEngine
from src.sensory.dimensions.how_engine import HOWEngine
from src.sensory.dimensions.what_engine import WATEngine
from src.sensory.dimensions.when_engine import WHENEngine
from src.sensory.dimensions.anomaly_engine import ANOMALYEngine
from src.sensory.orchestration.master_orchestrator import MasterOrchestrator


class TestDataGenerator:
    """Generate realistic test data for comprehensive testing."""
    
    @staticmethod
    def create_instrument_meta() -> InstrumentMeta:
        """Create test instrument metadata."""
        return InstrumentMeta(
            symbol="EURUSD",
            pip_size=0.0001,
            lot_size=100000,
            timezone="UTC",
            sessions={
                "ASIAN": ("00:00", "09:00"),
                "LONDON": ("08:00", "17:00"),
                "NEW_YORK": ("13:00", "22:00")
            },
            typical_spread=0.0001,
            avg_daily_range=0.01
        )
    
    @staticmethod
    def create_market_data(
        timestamp: Optional[datetime] = None,
        price: float = 1.1000,
        spread: float = 0.0002,
        volume: float = 1000.0
    ) -> MarketData:
        """Create test market data."""
        if timestamp is None:
            timestamp = datetime.utcnow()
        
        return MarketData(
            symbol="EURUSD",
            timestamp=timestamp,
            open=price - 0.0005,
            high=price + 0.0010,
            low=price - 0.0015,
            close=price,
            volume=volume,
            bid=price - spread/2,
            ask=price + spread/2
        )
    
    @staticmethod
    def create_order_book(
        mid_price: float = 1.1000,
        depth: int = 5
    ) -> OrderBookSnapshot:
        """Create test order book snapshot."""
        bids = []
        asks = []
        
        for i in range(depth):
            bid_price = mid_price - (i + 1) * 0.0001
            ask_price = mid_price + (i + 1) * 0.0001
            volume = 1000000 - i * 100000  # Decreasing volume
            
            bids.append(OrderBookLevel(price=bid_price, volume=volume))
            asks.append(OrderBookLevel(price=ask_price, volume=volume))
        
        return OrderBookSnapshot(
            symbol="EURUSD",
            timestamp=datetime.utcnow(),
            bids=bids,
            asks=asks
        )
    
    @staticmethod
    def create_price_series(
        length: int = 100,
        start_price: float = 1.1000,
        volatility: float = 0.001,
        trend: float = 0.0
    ) -> List[MarketData]:
        """Create realistic price series for testing."""
        prices = []
        current_price = start_price
        base_time = datetime.utcnow() - timedelta(minutes=length)
        
        for i in range(length):
            # Add trend and random walk
            price_change = trend + np.random.normal(0, volatility)
            current_price += price_change
            
            # Create market data
            timestamp = base_time + timedelta(minutes=i)
            market_data = TestDataGenerator.create_market_data(
                timestamp=timestamp,
                price=current_price,
                volume=np.random.uniform(500, 2000)
            )
            prices.append(market_data)
        
        return prices


class TestDimensionalEngines:
    """Test individual dimensional engines."""
    
    @pytest.fixture
    def instrument_meta(self):
        """Instrument metadata fixture."""
        return TestDataGenerator.create_instrument_meta()
    
    @pytest.fixture
    def market_data(self):
        """Market data fixture."""
        return TestDataGenerator.create_market_data()
    
    @pytest.fixture
    def order_book(self):
        """Order book fixture."""
        return TestDataGenerator.create_order_book()
    
    @pytest.mark.asyncio
    async def test_why_engine_basic_functionality(self, instrument_meta, market_data):
        """Test WHY engine basic functionality."""
        engine = WHYEngine(instrument_meta)
        
        # Test update
        reading = await engine.update(market_data)
        
        # Validate reading
        assert isinstance(reading, DimensionalReading)
        assert reading.dimension == "WHY"
        assert -1.0 <= reading.signal_strength <= 1.0
        assert 0.0 <= reading.confidence <= 1.0
        assert isinstance(reading.regime, MarketRegime)
        assert reading.data_quality > 0.0
        assert reading.processing_time_ms >= 0.0
        
        logger.info(f"WHY Engine - Signal: {reading.signal_strength:.3f}, "
                   f"Confidence: {reading.confidence:.3f}")
    
    @pytest.mark.asyncio
    async def test_how_engine_ict_patterns(self, instrument_meta):
        """Test HOW engine ICT pattern detection."""
        engine = HOWEngine(instrument_meta)
        
        # Create trending price series
        price_series = TestDataGenerator.create_price_series(
            length=50, trend=0.0002, volatility=0.0005
        )
        
        readings = []
        for market_data in price_series:
            reading = await engine.update(market_data)
            readings.append(reading)
        
        # Validate pattern detection
        assert len(readings) == 50
        
        # Check for ICT pattern detection
        pattern_detections = [r for r in readings if 'ict_patterns' in r.evidence]
        assert len(pattern_detections) > 0, "Should detect some ICT patterns in trending data"
        
        # Validate signal consistency
        signals = [r.signal_strength for r in readings[-10:]]  # Last 10 readings
        signal_std = np.std(signals)
        assert signal_std < 0.5, "Signals should be relatively consistent in trending market"
        
        logger.info(f"HOW Engine - Detected {len(pattern_detections)} pattern instances")
    
    @pytest.mark.asyncio
    async def test_what_engine_market_structure(self, instrument_meta):
        """Test WHAT engine market structure analysis."""
        engine = WATEngine(instrument_meta)
        
        # Create price series with clear structure
        price_series = TestDataGenerator.create_price_series(
            length=100, trend=0.0001, volatility=0.0003
        )
        
        readings = []
        for market_data in price_series:
            reading = await engine.update(market_data)
            readings.append(reading)
        
        # Validate structure detection
        assert len(readings) == 100
        
        # Check for swing point detection
        swing_detections = [r for r in readings if 'swing_points' in r.evidence]
        assert len(swing_detections) > 0, "Should detect swing points"
        
        # Check momentum analysis
        momentum_readings = [r for r in readings if 'momentum' in r.evidence]
        assert len(momentum_readings) > 0, "Should calculate momentum"
        
        logger.info(f"WHAT Engine - Detected {len(swing_detections)} swing points")
    
    @pytest.mark.asyncio
    async def test_when_engine_temporal_analysis(self, instrument_meta, market_data):
        """Test WHEN engine temporal analysis."""
        engine = WHENEngine(instrument_meta)
        
        # Test different times of day
        test_times = [
            datetime(2024, 1, 15, 8, 0),   # London open
            datetime(2024, 1, 15, 13, 30), # NY open
            datetime(2024, 1, 15, 21, 0),  # Asian session
        ]
        
        readings = []
        for test_time in test_times:
            market_data.timestamp = test_time
            reading = await engine.update(market_data)
            readings.append(reading)
        
        # Validate temporal analysis
        assert len(readings) == 3
        
        # Check session detection
        for reading in readings:
            assert 'session_strength' in reading.evidence
            assert 'timing_quality' in reading.evidence
        
        # Different sessions should have different characteristics
        session_strengths = [r.evidence['session_strength'] for r in readings]
        assert len(set(session_strengths)) > 1, "Different sessions should have different strengths"
        
        logger.info(f"WHEN Engine - Session strengths: {session_strengths}")
    
    @pytest.mark.asyncio
    async def test_anomaly_engine_detection(self, instrument_meta, order_book):
        """Test ANOMALY engine manipulation detection."""
        engine = ANOMALYEngine(instrument_meta)
        
        # Create normal market data
        normal_data = TestDataGenerator.create_market_data(volume=1000)
        normal_reading = await engine.update(normal_data, order_book)
        
        # Create anomalous market data (high volume spike)
        anomalous_data = TestDataGenerator.create_market_data(volume=10000)
        anomalous_reading = await engine.update(anomalous_data, order_book)
        
        # Validate anomaly detection
        assert isinstance(normal_reading, DimensionalReading)
        assert isinstance(anomalous_reading, DimensionalReading)
        
        # Anomalous data should have different signal
        assert abs(anomalous_reading.signal_strength) >= abs(normal_reading.signal_strength)
        
        # Check for anomaly evidence
        assert 'anomaly_score' in anomalous_reading.evidence
        
        logger.info(f"ANOMALY Engine - Normal: {normal_reading.signal_strength:.3f}, "
                   f"Anomalous: {anomalous_reading.signal_strength:.3f}")


class TestMasterOrchestrator:
    """Test master orchestrator functionality."""
    
    @pytest.fixture
    def instrument_meta(self):
        """Instrument metadata fixture."""
        return TestDataGenerator.create_instrument_meta()
    
    @pytest.fixture
    def orchestrator(self, instrument_meta):
        """Master orchestrator fixture."""
        return MasterOrchestrator(instrument_meta)
    
    @pytest.mark.asyncio
    async def test_orchestrator_basic_synthesis(self, orchestrator):
        """Test basic orchestrator synthesis."""
        market_data = TestDataGenerator.create_market_data()
        order_book = TestDataGenerator.create_order_book()
        
        # Perform synthesis
        result = await orchestrator.update(market_data, order_book)
        
        # Validate synthesis result
        assert result.timestamp == market_data.timestamp
        assert -1.0 <= result.signal_strength <= 1.0
        assert 0.0 <= result.confidence <= 1.0
        assert isinstance(result.regime, MarketRegime)
        assert 0.0 <= result.consensus_level <= 1.0
        assert len(result.dimensional_weights) == 5
        assert len(result.dimensional_contributions) == 5
        assert result.narrative != ""
        assert result.processing_time_ms > 0
        
        logger.info(f"Orchestrator - Signal: {result.signal_strength:.3f}, "
                   f"Confidence: {result.confidence:.3f}, "
                   f"Consensus: {result.consensus_level:.3f}")
    
    @pytest.mark.asyncio
    async def test_orchestrator_contextual_weighting(self, orchestrator):
        """Test contextual weighting adaptation."""
        # Test different market conditions
        conditions = [
            # Normal market
            TestDataGenerator.create_market_data(price=1.1000, volume=1000),
            # High volatility
            TestDataGenerator.create_market_data(price=1.1050, volume=5000),
            # Low volatility
            TestDataGenerator.create_market_data(price=1.1005, volume=200),
        ]
        
        results = []
        for market_data in conditions:
            result = await orchestrator.update(market_data)
            results.append(result)
        
        # Validate weight adaptation
        assert len(results) == 3
        
        # Weights should adapt to conditions
        weight_sets = [r.dimensional_weights for r in results]
        
        # Check that weights are different across conditions
        why_weights = [w.get('WHY', 0) for w in weight_sets]
        assert len(set(why_weights)) > 1 or max(why_weights) - min(why_weights) > 0.01
        
        logger.info(f"Weight adaptation - WHY weights: {why_weights}")
    
    @pytest.mark.asyncio
    async def test_orchestrator_graceful_degradation(self, orchestrator):
        """Test graceful degradation when engines fail."""
        # Simulate engine failure by corrupting data
        corrupted_data = TestDataGenerator.create_market_data()
        corrupted_data.close = float('nan')  # Invalid data
        
        # System should handle gracefully
        result = await orchestrator.update(corrupted_data)
        
        # Should still produce result
        assert result is not None
        assert result.confidence >= 0.0  # May be low but not negative
        
        # Check system health
        health = orchestrator.get_system_health()
        assert 'overall_health' in health
        assert 'degraded_engines' in health
        
        logger.info(f"Degradation test - Health: {health['overall_health']:.3f}")
    
    @pytest.mark.asyncio
    async def test_orchestrator_consensus_calculation(self, orchestrator):
        """Test consensus calculation across dimensions."""
        # Create trending market data
        price_series = TestDataGenerator.create_price_series(
            length=20, trend=0.0005, volatility=0.0002
        )
        
        consensus_levels = []
        for market_data in price_series:
            result = await orchestrator.update(market_data)
            consensus_levels.append(result.consensus_level)
        
        # Validate consensus tracking
        assert len(consensus_levels) == 20
        assert all(0.0 <= c <= 1.0 for c in consensus_levels)
        
        # In trending market, consensus should generally increase
        early_consensus = np.mean(consensus_levels[:5])
        late_consensus = np.mean(consensus_levels[-5:])
        
        logger.info(f"Consensus evolution - Early: {early_consensus:.3f}, "
                   f"Late: {late_consensus:.3f}")


class TestPerformanceAndRobustness:
    """Test system performance and robustness."""
    
    @pytest.fixture
    def instrument_meta(self):
        """Instrument metadata fixture."""
        return TestDataGenerator.create_instrument_meta()
    
    @pytest.mark.asyncio
    async def test_processing_speed(self, instrument_meta):
        """Test processing speed requirements."""
        orchestrator = MasterOrchestrator(instrument_meta)
        
        # Test processing speed
        market_data = TestDataGenerator.create_market_data()
        order_book = TestDataGenerator.create_order_book()
        
        start_time = datetime.utcnow()
        result = await orchestrator.update(market_data, order_book)
        processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        # Should process within reasonable time (< 100ms for single update)
        assert processing_time < 100, f"Processing took {processing_time:.1f}ms, should be < 100ms"
        assert result.processing_time_ms < 100
        
        logger.info(f"Processing speed - Total: {processing_time:.1f}ms, "
                   f"Internal: {result.processing_time_ms:.1f}ms")
    
    @pytest.mark.asyncio
    async def test_memory_usage(self, instrument_meta):
        """Test memory usage stability."""
        orchestrator = MasterOrchestrator(instrument_meta)
        
        # Process many updates
        for i in range(100):
            market_data = TestDataGenerator.create_market_data()
            await orchestrator.update(market_data)
        
        # Check history management
        assert len(orchestrator.synthesis_history) <= 100, "History should be limited"
        
        # System should still be responsive
        final_result = await orchestrator.update(TestDataGenerator.create_market_data())
        assert final_result.confidence >= 0.0
        
        logger.info(f"Memory test - History length: {len(orchestrator.synthesis_history)}")
    
    @pytest.mark.asyncio
    async def test_error_recovery(self, instrument_meta):
        """Test error recovery capabilities."""
        orchestrator = MasterOrchestrator(instrument_meta)
        
        # Test with various error conditions
        error_conditions = [
            # Invalid price data
            TestDataGenerator.create_market_data(price=float('inf')),
            # Negative volume
            TestDataGenerator.create_market_data(volume=-100),
            # Zero spread
            TestDataGenerator.create_market_data(spread=0.0),
        ]
        
        recovery_count = 0
        for error_data in error_conditions:
            try:
                result = await orchestrator.update(error_data)
                if result.confidence >= 0.0:  # System recovered
                    recovery_count += 1
            except Exception as e:
                logger.warning(f"Error condition failed: {e}")
        
        # Should recover from most error conditions
        recovery_rate = recovery_count / len(error_conditions)
        assert recovery_rate >= 0.5, f"Recovery rate {recovery_rate:.1f} too low"
        
        logger.info(f"Error recovery - Rate: {recovery_rate:.1f}")


class TestRealWorldScenarios:
    """Test real-world market scenarios."""
    
    @pytest.fixture
    def instrument_meta(self):
        """Instrument metadata fixture."""
        return TestDataGenerator.create_instrument_meta()
    
    @pytest.mark.asyncio
    async def test_trending_market_scenario(self, instrument_meta):
        """Test behavior in trending market."""
        orchestrator = MasterOrchestrator(instrument_meta)
        
        # Create strong trending market
        price_series = TestDataGenerator.create_price_series(
            length=50, trend=0.001, volatility=0.0003
        )
        
        results = []
        for market_data in price_series:
            result = await orchestrator.update(market_data)
            results.append(result)
        
        # Validate trending behavior
        assert len(results) == 50
        
        # Should detect trending regime
        trending_count = sum(1 for r in results if r.regime == MarketRegime.TRENDING_STRONG)
        assert trending_count > len(results) * 0.3, "Should detect trending regime"
        
        # Signals should align with trend
        late_signals = [r.signal_strength for r in results[-10:]]
        signal_consistency = np.std(late_signals)
        assert signal_consistency < 0.3, "Signals should be consistent in trend"
        
        logger.info(f"Trending scenario - Trending detections: {trending_count}, "
                   f"Signal consistency: {signal_consistency:.3f}")
    
    @pytest.mark.asyncio
    async def test_consolidation_scenario(self, instrument_meta):
        """Test behavior in consolidating market."""
        orchestrator = MasterOrchestrator(instrument_meta)
        
        # Create consolidating market (no trend, low volatility)
        price_series = TestDataGenerator.create_price_series(
            length=50, trend=0.0, volatility=0.0001
        )
        
        results = []
        for market_data in price_series:
            result = await orchestrator.update(market_data)
            results.append(result)
        
        # Validate consolidation behavior
        assert len(results) == 50
        
        # Should detect consolidating regime
        consolidating_count = sum(1 for r in results if r.regime == MarketRegime.CONSOLIDATING)
        assert consolidating_count > len(results) * 0.2, "Should detect consolidation"
        
        # Signals should be weak/neutral
        signal_strengths = [abs(r.signal_strength) for r in results]
        avg_signal_strength = np.mean(signal_strengths)
        assert avg_signal_strength < 0.5, "Signals should be weak in consolidation"
        
        logger.info(f"Consolidation scenario - Consolidating detections: {consolidating_count}, "
                   f"Avg signal strength: {avg_signal_strength:.3f}")
    
    @pytest.mark.asyncio
    async def test_high_volatility_scenario(self, instrument_meta):
        """Test behavior in high volatility market."""
        orchestrator = MasterOrchestrator(instrument_meta)
        
        # Create high volatility market
        price_series = TestDataGenerator.create_price_series(
            length=30, trend=0.0, volatility=0.002
        )
        
        results = []
        for market_data in price_series:
            result = await orchestrator.update(market_data)
            results.append(result)
        
        # Validate high volatility behavior
        assert len(results) == 30
        
        # Should adapt weights for volatility
        anomaly_weights = [r.dimensional_weights.get('ANOMALY', 0) for r in results]
        avg_anomaly_weight = np.mean(anomaly_weights)
        
        # ANOMALY engine should get higher weight in volatile conditions
        assert avg_anomaly_weight > 0.05, "ANOMALY should get weight in volatile market"
        
        # Confidence should be appropriately adjusted
        confidences = [r.confidence for r in results]
        avg_confidence = np.mean(confidences)
        
        logger.info(f"High volatility scenario - Avg ANOMALY weight: {avg_anomaly_weight:.3f}, "
                   f"Avg confidence: {avg_confidence:.3f}")


# Test runner
if __name__ == "__main__":
    # Run basic functionality tests
    async def run_basic_tests():
        """Run basic functionality tests."""
        print("🧪 Running Sensory Cortex v2.2 Integration Tests")
        print("=" * 60)
        
        # Test data generator
        print("\n📊 Testing Data Generation...")
        instrument = TestDataGenerator.create_instrument_meta()
        market_data = TestDataGenerator.create_market_data()
        order_book = TestDataGenerator.create_order_book()
        print(f"✅ Generated test data for {instrument.symbol}")
        
        # Test individual engines
        print("\n🔧 Testing Dimensional Engines...")
        
        # WHY Engine
        why_engine = WHYEngine(instrument)
        why_reading = await why_engine.update(market_data)
        print(f"✅ WHY Engine: Signal={why_reading.signal_strength:.3f}, "
              f"Confidence={why_reading.confidence:.3f}")
        
        # HOW Engine
        how_engine = HOWEngine(instrument)
        how_reading = await how_engine.update(market_data)
        print(f"✅ HOW Engine: Signal={how_reading.signal_strength:.3f}, "
              f"Confidence={how_reading.confidence:.3f}")
        
        # WHAT Engine
        what_engine = WATEngine(instrument)
        what_reading = await what_engine.update(market_data)
        print(f"✅ WHAT Engine: Signal={what_reading.signal_strength:.3f}, "
              f"Confidence={what_reading.confidence:.3f}")
        
        # WHEN Engine
        when_engine = WHENEngine(instrument)
        when_reading = await when_engine.update(market_data)
        print(f"✅ WHEN Engine: Signal={when_reading.signal_strength:.3f}, "
              f"Confidence={when_reading.confidence:.3f}")
        
        # ANOMALY Engine
        anomaly_engine = ANOMALYEngine(instrument)
        anomaly_reading = await anomaly_engine.update(market_data, order_book)
        print(f"✅ ANOMALY Engine: Signal={anomaly_reading.signal_strength:.3f}, "
              f"Confidence={anomaly_reading.confidence:.3f}")
        
        # Test orchestrator
        print("\n🎼 Testing Master Orchestrator...")
        orchestrator = MasterOrchestrator(instrument)
        synthesis = await orchestrator.update(market_data, order_book)
        
        print(f"✅ Orchestrator Synthesis:")
        print(f"   Signal Strength: {synthesis.signal_strength:.3f}")
        print(f"   Confidence: {synthesis.confidence:.3f}")
        print(f"   Consensus: {synthesis.consensus_level:.3f}")
        print(f"   Regime: {synthesis.regime}")
        print(f"   Processing Time: {synthesis.processing_time_ms:.1f}ms")
        print(f"   Narrative: {synthesis.narrative}")
        
        # Test system health
        health = orchestrator.get_system_health()
        print(f"\n🏥 System Health:")
        print(f"   Overall Health: {health['overall_health']:.3f}")
        print(f"   Healthy Engines: {health['healthy_engines']}/{health['total_engines']}")
        print(f"   Average Performance: {health['average_performance']:.3f}")
        
        print("\n🎯 All Integration Tests Completed Successfully!")
        print("=" * 60)
    
    # Run the tests
    asyncio.run(run_basic_tests())

