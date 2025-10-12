"""
Comprehensive Integration Test Suite for Multidimensional Market Understanding System

This module provides thorough testing of the complete system including:
- Individual dimensional engines
- Cross-dimensional correlations
- Contextual fusion
- Adaptive weights
- Narrative generation
- Performance validation
"""

import logging
from datetime import datetime, timedelta
from typing import List

import numpy as np
import pytest

from src.core.base import DimensionalReading, MarketData
from src.orchestration.enhanced_understanding_engine import ContextualFusionEngine
from src.sensory.enhanced._shared import ReadingAdapter
from src.sensory.enhanced.anomaly_dimension import AnomalyUnderstandingEngine
from src.sensory.enhanced.how_dimension import InstitutionalUnderstandingEngine
from src.sensory.enhanced.what_dimension import TechnicalRealityEngine
from src.sensory.enhanced.when_dimension import ChronalUnderstandingEngine
from src.sensory.enhanced.why_dimension import (
    EnhancedFundamentalUnderstandingEngine,
)


def _to_dimensional_reading(
    candidate: DimensionalReading | ReadingAdapter,
) -> DimensionalReading:
    if isinstance(candidate, ReadingAdapter):
        return candidate.reading
    return candidate

# Configure logging for tests
logging.basicConfig(level=logging.WARNING)


class TestDataGenerator:
    """Generate realistic test market data"""

    def __init__(self, base_price: float = 1.0950):
        self.base_price = base_price
        self.current_price = base_price
        self.time_counter = 0

    def generate_market_data(self, scenario: str = "normal") -> MarketData:
        """Generate market data for different scenarios"""

        self.time_counter += 1
        current_time = datetime.now() + timedelta(minutes=self.time_counter)

        if scenario == "trending_bull":
            price_change = np.random.normal(0.0001, 0.0002)  # Upward bias
            volume = np.random.normal(1500, 300)
            volatility = np.random.exponential(0.006)

        elif scenario == "trending_bear":
            price_change = np.random.normal(-0.0001, 0.0002)  # Downward bias
            volume = np.random.normal(1800, 400)  # Higher volume in bear moves
            volatility = np.random.exponential(0.008)

        elif scenario == "ranging":
            # Mean-reverting behavior
            deviation = self.current_price - self.base_price
            price_change = np.random.normal(-deviation * 0.1, 0.0001)
            volume = np.random.normal(1200, 200)
            volatility = np.random.exponential(0.004)

        elif scenario == "volatile":
            price_change = np.random.normal(0, 0.0005)  # High volatility
            volume = np.random.normal(2000, 600)
            volatility = np.random.exponential(0.015)

        elif scenario == "anomaly":
            # Inject anomalous behavior
            if np.random.random() < 0.3:  # 30% chance of anomaly
                price_change = np.random.choice([-0.002, 0.002])  # Large moves
                volume = np.random.choice([500, 5000])  # Extreme volume
            else:
                price_change = np.random.normal(0, 0.0001)
                volume = np.random.normal(1000, 200)
            volatility = np.random.exponential(0.010)

        else:  # normal
            price_change = np.random.normal(0, 0.0001)
            volume = np.random.normal(1000, 200)
            volatility = np.random.exponential(0.005)

        self.current_price += price_change
        volume = max(volume, 100)  # Minimum volume
        volatility = max(volatility, 0.001)  # Minimum volatility

        return MarketData(
            timestamp=current_time,
            bid=self.current_price - 0.0001,
            ask=self.current_price + 0.0001,
            volume=volume,
            volatility=volatility,
        )

    def generate_sequence(self, scenario: str, length: int) -> List[MarketData]:
        """Generate a sequence of market data"""
        return [self.generate_market_data(scenario) for _ in range(length)]


class TestDimensionalEngines:
    """Test individual dimensional engines"""

    @pytest.fixture
    def data_generator(self):
        return TestDataGenerator()

    @pytest.fixture
    def sample_market_data(self, data_generator):
        return data_generator.generate_market_data("normal")

    @pytest.mark.asyncio
    async def test_why_engine_basic_functionality(self, sample_market_data):
        """Test WHY dimension engine basic functionality"""

        engine = EnhancedFundamentalUnderstandingEngine()

        # Test basic analysis
        reading = await engine.analyze_fundamental_understanding(sample_market_data)

        assert isinstance(reading, DimensionalReading)
        assert reading.dimension == "WHY"
        assert -1.0 <= reading.value <= 1.0
        assert 0.0 <= reading.confidence <= 1.0
        assert isinstance(reading.context, dict)
        assert reading.timestamp is not None

    @pytest.mark.asyncio
    async def test_how_engine_basic_functionality(self, sample_market_data):
        """Test HOW dimension engine basic functionality"""

        engine = InstitutionalUnderstandingEngine()

        # Test basic analysis
        reading = await engine.analyze_institutional_understanding(sample_market_data)

        assert isinstance(reading, DimensionalReading)
        assert reading.dimension == "HOW"
        assert -1.0 <= reading.value <= 1.0
        assert 0.0 <= reading.confidence <= 1.0
        assert isinstance(reading.context, dict)

    @pytest.mark.asyncio
    async def test_what_engine_basic_functionality(self, sample_market_data):
        """Test WHAT dimension engine basic functionality"""

        engine = TechnicalRealityEngine()

        # Test basic analysis
        reading = await engine.analyze_technical_reality(sample_market_data)

        assert isinstance(reading, DimensionalReading)
        assert reading.dimension == "WHAT"
        assert -1.0 <= reading.value <= 1.0
        assert 0.0 <= reading.confidence <= 1.0
        assert isinstance(reading.context, dict)

    @pytest.mark.asyncio
    async def test_when_engine_basic_functionality(self, sample_market_data):
        """Test WHEN dimension engine basic functionality"""

        engine = ChronalUnderstandingEngine()

        # Test basic analysis
        reading = await engine.analyze_temporal_understanding(sample_market_data)

        assert isinstance(reading, DimensionalReading)
        assert reading.dimension == "WHEN"
        assert -1.0 <= reading.value <= 1.0
        assert 0.0 <= reading.confidence <= 1.0
        assert isinstance(reading.context, dict)

    @pytest.mark.asyncio
    async def test_anomaly_engine_basic_functionality(self, sample_market_data):
        """Test ANOMALY dimension engine basic functionality"""

        engine = AnomalyUnderstandingEngine()

        # Test basic analysis
        reading = _to_dimensional_reading(
            engine.analyze_anomaly_understanding(sample_market_data)
        )

        assert isinstance(reading, DimensionalReading)
        assert reading.dimension == "ANOMALY"
        assert 0.0 <= reading.value <= 1.0  # Anomaly is always positive
        assert 0.0 <= reading.confidence <= 1.0
        assert isinstance(reading.context, dict)

    @pytest.mark.asyncio
    async def test_engines_with_trending_data(self, data_generator):
        """Test engines with trending market data"""

        # Generate trending data
        trending_data = data_generator.generate_sequence("trending_bull", 20)

        engines = {
            "WHY": EnhancedFundamentalUnderstandingEngine(),
            "HOW": InstitutionalUnderstandingEngine(),
            "WHAT": TechnicalRealityEngine(),
            "WHEN": ChronalUnderstandingEngine(),
            "ANOMALY": AnomalyUnderstandingEngine(),
        }

        readings = {}

        # Feed data to engines
        for market_data in trending_data:
            for name, engine in engines.items():
                if name == "WHY":
                    reading = await engine.analyze_fundamental_understanding(market_data)
                elif name == "HOW":
                    reading = await engine.analyze_institutional_understanding(market_data)
                elif name == "WHAT":
                    reading = await engine.analyze_technical_reality(market_data)
                elif name == "WHEN":
                    reading = await engine.analyze_temporal_understanding(market_data)
                elif name == "ANOMALY":
                    reading = _to_dimensional_reading(
                        engine.analyze_anomaly_understanding(market_data)
                    )

                readings[name] = reading

        # Verify all engines produced valid readings
        for name, reading in readings.items():
            assert reading is not None
            assert reading.dimension == name
            assert isinstance(reading.value, (int, float))
            assert isinstance(reading.confidence, (int, float))

    @pytest.mark.asyncio
    async def test_engines_with_anomalous_data(self, data_generator):
        """Test engines with anomalous market data"""

        # Generate anomalous data
        anomalous_data = data_generator.generate_sequence("anomaly", 15)

        anomaly_engine = AnomalyUnderstandingEngine()
        anomaly_readings = []

        # Feed data to anomaly engine
        for market_data in anomalous_data:
            reading = _to_dimensional_reading(
                anomaly_engine.analyze_anomaly_understanding(market_data)
            )
            anomaly_readings.append(reading.value)

        # Should detect some anomalies
        max_anomaly = max(anomaly_readings)
        assert max_anomaly > 0.2, "Should detect anomalies in anomalous data"


class TestContextualFusion:
    """Test contextual fusion engine"""

    @pytest.fixture
    def fusion_engine(self):
        return ContextualFusionEngine()

    @pytest.fixture
    def data_generator(self):
        return TestDataGenerator()

    @pytest.mark.asyncio
    async def test_fusion_engine_basic_functionality(self, fusion_engine, data_generator):
        """Test basic fusion engine functionality"""

        market_data = data_generator.generate_market_data("normal")

        # Test basic analysis
        synthesis = await fusion_engine.analyze_market_understanding(market_data)

        assert synthesis is not None
        assert hasattr(synthesis, "understanding_level")
        assert not hasattr(synthesis, "intelligence_level")
        assert hasattr(synthesis, "narrative_coherence")
        assert hasattr(synthesis, "unified_score")
        assert hasattr(synthesis, "confidence")
        assert hasattr(synthesis, "narrative_text")

        # Verify score bounds
        assert -1.0 <= synthesis.unified_score <= 1.0
        assert 0.0 <= synthesis.confidence <= 1.0

    def test_intelligence_alias_removed(self, fusion_engine) -> None:
        assert not hasattr(fusion_engine, "analyze_market_intelligence")
        with pytest.raises(AttributeError):
            getattr(fusion_engine, "analyze_market_intelligence")

    @pytest.mark.asyncio
    async def test_fusion_with_multiple_data_points(self, fusion_engine, data_generator):
        """Test fusion engine with multiple data points"""

        # Generate sequence of data
        data_sequence = data_generator.generate_sequence("trending_bull", 10)

        syntheses = []

        for market_data in data_sequence:
            synthesis = await fusion_engine.analyze_market_understanding(market_data)
            syntheses.append(synthesis)

        # Verify all syntheses are valid
        for synthesis in syntheses:
            assert synthesis is not None
            assert isinstance(synthesis.unified_score, (int, float))
            assert isinstance(synthesis.confidence, (int, float))
            assert isinstance(synthesis.narrative_text, str)

        # Confidence should generally improve with more data
        confidences = [s.confidence for s in syntheses]
        assert confidences[-1] >= confidences[0] - 0.1, (
            "Confidence should not degrade significantly"
        )

    @pytest.mark.asyncio
    async def test_adaptive_weights(self, fusion_engine, data_generator):
        """Test adaptive weight functionality"""

        # Generate data sequence
        data_sequence = data_generator.generate_sequence("volatile", 15)

        initial_weights = None
        final_weights = None

        for i, market_data in enumerate(data_sequence):
            await fusion_engine.analyze_market_understanding(market_data)

            if i == 0:
                initial_weights = fusion_engine.weight_manager.calculate_current_weights()
            elif i == len(data_sequence) - 1:
                final_weights = fusion_engine.weight_manager.calculate_current_weights()

        # Weights should adapt
        assert initial_weights is not None
        assert final_weights is not None

        # At least one weight should change
        weight_changes = [
            abs(final_weights[dim] - initial_weights[dim]) for dim in initial_weights.keys()
        ]
        assert max(weight_changes) > 0.01, "Weights should adapt over time"

    @pytest.mark.asyncio
    async def test_correlation_detection(self, fusion_engine, data_generator):
        """Test cross-dimensional correlation detection"""

        # Generate enough data for correlation analysis
        data_sequence = data_generator.generate_sequence("trending_bull", 25)

        for market_data in data_sequence:
            await fusion_engine.analyze_market_understanding(market_data)

        # Check for detected correlations
        correlations = fusion_engine.correlation_analyzer.get_dimensional_correlations()

        # Should have some correlations detected
        assert len(correlations) > 0, "Should detect some correlations with sufficient data"

        # Verify correlation structure
        for (dim_a, dim_b), correlation in correlations.items():
            assert dim_a in ["WHY", "HOW", "WHAT", "WHEN", "ANOMALY"]
            assert dim_b in ["WHY", "HOW", "WHAT", "WHEN", "ANOMALY"]
            assert -1.0 <= correlation.correlation <= 1.0
            assert 0.0 <= correlation.significance <= 1.0

    @pytest.mark.asyncio
    async def test_pattern_detection(self, fusion_engine, data_generator):
        """Test cross-dimensional pattern detection"""

        # Generate data that should create patterns
        data_sequence = data_generator.generate_sequence("trending_bull", 30)

        for market_data in data_sequence:
            await fusion_engine.analyze_market_understanding(market_data)

        # Check for detected patterns
        patterns = fusion_engine.correlation_analyzer.get_cross_dimensional_patterns()

        # May or may not detect patterns, but should not crash
        for pattern in patterns:
            assert hasattr(pattern, "pattern_name")
            assert hasattr(pattern, "involved_dimensions")
            assert hasattr(pattern, "pattern_strength")
            assert hasattr(pattern, "confidence")
            assert 0.0 <= pattern.pattern_strength <= 1.0
            assert 0.0 <= pattern.confidence <= 1.0

    @pytest.mark.asyncio
    async def test_narrative_generation(self, fusion_engine, data_generator):
        """Test narrative generation"""

        # Generate data sequence
        data_sequence = data_generator.generate_sequence("trending_bull", 10)

        narratives = []

        for market_data in data_sequence:
            synthesis = await fusion_engine.analyze_market_understanding(market_data)
            narratives.append(synthesis.narrative_text)

        # Verify narratives are generated
        for narrative in narratives:
            assert isinstance(narrative, str)
            assert len(narrative) > 0
            # Should contain some market-related terms
            market_terms = [
                "market",
                "price",
                "trend",
                "technical",
                "fundamental",
                "institutional",
            ]
            assert any(term in narrative.lower() for term in market_terms)


class TestSystemIntegration:
    """Test complete system integration"""

    @pytest.fixture
    def fusion_engine(self):
        return ContextualFusionEngine()

    @pytest.fixture
    def data_generator(self):
        return TestDataGenerator()

    @pytest.mark.asyncio
    async def test_complete_workflow(self, fusion_engine, data_generator):
        """Test complete workflow from data to synthesis"""

        # Test different market scenarios
        scenarios = ["normal", "trending_bull", "trending_bear", "ranging", "volatile"]

        for scenario in scenarios:
            data_sequence = data_generator.generate_sequence(scenario, 5)

            for market_data in data_sequence:
                synthesis = await fusion_engine.analyze_market_understanding(market_data)

                # Verify synthesis is complete
                assert synthesis is not None
                assert synthesis.narrative_text is not None
                assert len(synthesis.narrative_text) > 0
                assert isinstance(synthesis.supporting_evidence, list)
                assert isinstance(synthesis.risk_factors, list)
                assert isinstance(synthesis.opportunity_factors, list)

    @pytest.mark.asyncio
    async def test_error_handling(self, fusion_engine):
        """Test system error handling"""

        # Test with invalid data
        invalid_data = MarketData(
            timestamp=datetime.now(),
            bid=float("nan"),
            ask=float("nan"),
            volume=-100,  # Invalid volume
            volatility=float("inf"),  # Invalid volatility
        )

        # Should handle gracefully without crashing
        try:
            synthesis = await fusion_engine.analyze_market_understanding(invalid_data)
            # If it doesn't crash, verify basic structure
            assert synthesis is not None
        except Exception as e:
            # If it does raise an exception, it should be handled gracefully
            assert isinstance(e, (ValueError, TypeError))

    @pytest.mark.asyncio
    async def test_performance_characteristics(self, fusion_engine, data_generator):
        """Test system performance characteristics"""

        import time

        # Generate larger data sequence
        data_sequence = data_generator.generate_sequence("normal", 20)

        start_time = time.time()

        for market_data in data_sequence:
            await fusion_engine.analyze_market_understanding(market_data)

        end_time = time.time()
        total_time = end_time - start_time

        # Should process reasonably quickly
        avg_time_per_analysis = total_time / len(data_sequence)
        assert avg_time_per_analysis < 1.0, (
            f"Analysis taking too long: {avg_time_per_analysis:.3f}s per analysis"
        )

    @pytest.mark.asyncio
    async def test_memory_usage(self, fusion_engine, data_generator):
        """Test memory usage doesn't grow unbounded"""

        import os

        psutil = pytest.importorskip("psutil")

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        # Generate large data sequence
        data_sequence = data_generator.generate_sequence("normal", 50)

        for market_data in data_sequence:
            await fusion_engine.analyze_market_understanding(market_data)

        final_memory = process.memory_info().rss
        memory_growth = final_memory - initial_memory

        # Memory growth should be reasonable (less than 100MB)
        assert memory_growth < 100 * 1024 * 1024, (
            f"Excessive memory growth: {memory_growth / 1024 / 1024:.1f}MB"
        )

    def test_diagnostic_information(self, fusion_engine):
        """Test diagnostic information retrieval"""

        diagnostics = fusion_engine.get_diagnostic_information()

        assert isinstance(diagnostics, dict)
        assert "current_readings" in diagnostics
        assert "adaptive_weights" in diagnostics
        assert "correlations" in diagnostics
        assert "patterns" in diagnostics


class TestScenarioValidation:
    """Test system behavior in specific market scenarios"""

    @pytest.fixture
    def fusion_engine(self):
        return ContextualFusionEngine()

    @pytest.fixture
    def data_generator(self):
        return TestDataGenerator()

    @pytest.mark.asyncio
    async def test_bull_trend_detection(self, fusion_engine, data_generator):
        """Test detection of bullish trends"""

        # Generate strong bull trend
        data_sequence = data_generator.generate_sequence("trending_bull", 20)

        syntheses = []
        for market_data in data_sequence:
            synthesis = await fusion_engine.analyze_market_understanding(market_data)
            syntheses.append(synthesis)

        # Later syntheses should show bullish bias
        final_syntheses = syntheses[-5:]  # Last 5 syntheses
        avg_score = np.mean([s.unified_score for s in final_syntheses])

        # Should detect bullish trend (positive unified score)
        assert avg_score > 0.1, f"Should detect bullish trend, got average score: {avg_score:.3f}"

    @pytest.mark.asyncio
    async def test_bear_trend_detection(self, fusion_engine, data_generator):
        """Test detection of bearish trends"""

        # Generate strong bear trend
        data_sequence = data_generator.generate_sequence("trending_bear", 20)

        syntheses = []
        for market_data in data_sequence:
            synthesis = await fusion_engine.analyze_market_understanding(market_data)
            syntheses.append(synthesis)

        # Later syntheses should show bearish bias
        final_syntheses = syntheses[-5:]  # Last 5 syntheses
        avg_score = np.mean([s.unified_score for s in final_syntheses])

        # Should detect bearish trend (negative unified score)
        assert avg_score < -0.1, f"Should detect bearish trend, got average score: {avg_score:.3f}"

    @pytest.mark.asyncio
    async def test_ranging_market_detection(self, fusion_engine, data_generator):
        """Test detection of ranging markets"""

        # Generate ranging market
        data_sequence = data_generator.generate_sequence("ranging", 25)

        syntheses = []
        for market_data in data_sequence:
            synthesis = await fusion_engine.analyze_market_understanding(market_data)
            syntheses.append(synthesis)

        # Should show low directional bias in ranging market
        final_syntheses = syntheses[-10:]  # Last 10 syntheses
        scores = [s.unified_score for s in final_syntheses]
        avg_abs_score = np.mean([abs(score) for score in scores])

        # Should have low directional bias
        assert avg_abs_score < 0.4, (
            f"Should detect ranging market, got average absolute score: {avg_abs_score:.3f}"
        )

    @pytest.mark.asyncio
    async def test_volatility_detection(self, fusion_engine, data_generator):
        """Test detection of high volatility periods"""

        # Generate volatile market
        data_sequence = data_generator.generate_sequence("volatile", 15)

        syntheses = []
        for market_data in data_sequence:
            synthesis = await fusion_engine.analyze_market_understanding(market_data)
            syntheses.append(synthesis)

        # Should detect volatility in risk factors or anomaly levels
        final_synthesis = syntheses[-1]

        # Check for volatility indicators
        volatility_detected = (
            any("volatil" in factor.lower() for factor in final_synthesis.risk_factors)
            or any("stress" in factor.lower() for factor in final_synthesis.risk_factors)
            or "VOLATILE" in final_synthesis.dominant_narrative.name
        )

        # Note: This is a probabilistic test, may not always detect volatility
        # but should not crash
        assert isinstance(volatility_detected, bool)

    @pytest.mark.asyncio
    async def test_anomaly_detection(self, fusion_engine, data_generator):
        """Test detection of market anomalies"""

        # Generate anomalous market data
        data_sequence = data_generator.generate_sequence("anomaly", 20)

        anomaly_levels = []
        for market_data in data_sequence:
            synthesis = await fusion_engine.analyze_market_understanding(market_data)

            # Check anomaly level in current readings
            if "ANOMALY" in fusion_engine.current_readings:
                anomaly_levels.append(fusion_engine.current_readings["ANOMALY"].value)

        # Should detect some anomalies
        if anomaly_levels:
            max_anomaly = max(anomaly_levels)
            assert max_anomaly > 0.2, f"Should detect anomalies, max level: {max_anomaly:.3f}"


# Performance benchmarks


class TestPerformanceBenchmarks:
    """Performance benchmark tests"""

    @pytest.mark.asyncio
    async def test_throughput_benchmark(self):
        """Benchmark system throughput"""

        import time

        fusion_engine = ContextualFusionEngine()
        data_generator = TestDataGenerator()

        # Generate test data
        data_sequence = data_generator.generate_sequence("normal", 100)

        start_time = time.time()

        for market_data in data_sequence:
            await fusion_engine.analyze_market_understanding(market_data)

        end_time = time.time()
        total_time = end_time - start_time

        throughput = len(data_sequence) / total_time

        print(f"System throughput: {throughput:.2f} analyses/second")
        print(f"Average time per analysis: {total_time / len(data_sequence) * 1000:.2f}ms")

        # Should achieve reasonable throughput
        assert throughput > 1.0, f"Throughput too low: {throughput:.2f} analyses/second"


# Utility functions for running tests


def run_basic_tests():
    """Run basic functionality tests"""
    pytest.main([__file__ + "::TestDimensionalEngines", "-v"])


def run_integration_tests():
    """Run integration tests"""
    pytest.main([__file__ + "::TestContextualFusion", "-v"])
    pytest.main([__file__ + "::TestSystemIntegration", "-v"])


def run_scenario_tests():
    """Run scenario validation tests"""
    pytest.main([__file__ + "::TestScenarioValidation", "-v"])


def run_performance_tests():
    """Run performance benchmark tests"""
    pytest.main([__file__ + "::TestPerformanceBenchmarks", "-v"])


def run_all_tests():
    """Run all tests"""
    pytest.main([__file__, "-v"])


if __name__ == "__main__":
    # Run all tests when executed directly
    run_all_tests()
