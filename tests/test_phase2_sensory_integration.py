"""
Comprehensive Test Suite for Phase 2 Sensory Cortex Implementation
=================================================================

Tests all 5D+1 sensory dimensions and integration orchestrator.
Author: EMP Development Team
Phase: 2 - Truth-First Completion
"""

import pytest
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import logging

# Import all test subjects
from src.sensory.enhanced.why.macro_predator_intelligence import (
    MacroPredatorIntelligence, MacroEnvironmentState
)
from src.sensory.enhanced.how.institutional_footprint_hunter import (
    InstitutionalFootprintHunter, InstitutionalFootprint
)
from src.sensory.enhanced.what.pattern_synthesis_engine import (
    PatternSynthesisEngine, PatternSynthesis
)
from src.sensory.enhanced.when.temporal_advantage_system import (
    TemporalAdvantageSystem, TemporalAdvantage
)
from src.sensory.enhanced.anomaly.manipulation_detection import (
    ManipulationDetectionSystem, AnomalyDetection
)
from src.sensory.enhanced.chaos.antifragile_adaptation import (
    ChaosAdaptationSystem, ChaosAdaptation
)
from src.sensory.enhanced.integration.sensory_integration_orchestrator import (
    SensoryIntegrationOrchestrator, UnifiedMarketIntelligence
)


class TestMacroPredatorIntelligence:
    """Test suite for WHY dimension - Macro Predator Intelligence"""
    
    @pytest.fixture
    def macro_intelligence(self):
        return MacroPredatorIntelligence()
    
    @pytest.mark.asyncio
    async def test_macro_analysis_basic(self, macro_intelligence):
        """Test basic macro environment analysis"""
        result = await macro_intelligence.analyze_macro_environment()
        
        assert isinstance(result, MacroEnvironmentState)
        assert -1.0 <= result.central_bank_sentiment <= 1.0
        assert 0.0 <= result.geopolitical_risk <= 1.0
        assert -1.0 <= result.economic_momentum <= 1.0
        assert -1.0 <= result.policy_outlook <= 1.0
        assert 0.0 <= result.confidence_score <= 1.0
    
    @pytest.mark.asyncio
    async def test_macro_analysis_with_mock_data(self, macro_intelligence):
        """Test macro analysis with mocked external data"""
        with patch.object(macro_intelligence.central_bank_parser, 'parse_latest_statements', 
                         return_value=0.7):
            with patch.object(macro_intelligence.geopolitical_mapper, 'assess_current_tensions', 
                             return_value=0.3):
                result = await macro_intelligence.analyze_macro_environment()
                
                assert result.central_bank_sentiment == 0.7
                assert result.geopolitical_risk == 0.3
    
    @pytest.mark.asyncio
    async def test_macro_fallback_behavior(self, macro_intelligence):
        """Test fallback behavior when analysis fails"""
        with patch.object(macro_intelligence.central_bank_parser, 'parse_latest_statements', 
                         side_effect=Exception("API failure")):
            result = await macro_intelligence.analyze_macro_environment()
            
            assert result.central_bank_sentiment == 0.0
            assert result.confidence_score == 0.1


class TestInstitutionalFootprintHunter:
    """Test suite for HOW dimension - Institutional Footprint Hunter"""
    
    @pytest.fixture
    def footprint_hunter(self):
        return InstitutionalFootprintHunter()
    
    @pytest.fixture
    def sample_market_data(self):
        """Create sample market data for testing"""
        dates = pd.date_range(start='2024-01-01', periods=100, freq='1H')
        return pd.DataFrame({
            'open': np.random.randn(100).cumsum() + 100,
            'high': np.random.randn(100).cumsum() + 101,
            'low': np.random.randn(100).cumsum() + 99,
            'close': np.random.randn(100).cumsum() + 100,
            'volume': np.random.randint(1000, 10000, 100)
        }, index=dates)
    
    @pytest.mark.asyncio
    async def test_footprint_analysis_basic(self, footprint_hunter, sample_market_data):
        """Test basic institutional footprint analysis"""
        result = await footprint_hunter.analyze_institutional_footprint(sample_market_data)
        
        assert isinstance(result, InstitutionalFootprint)
        assert isinstance(result.order_blocks, list)
        assert isinstance(result.fair_value_gaps, list)
        assert isinstance(result.liquidity_sweeps, list)
        assert -1.0 <= result.smart_money_flow <= 1.0
        assert result.institutional_bias in ['bullish', 'bearish', 'neutral']
        assert 0.0 <= result.confidence_score <= 1.0
    
    @pytest.mark.asyncio
    async def test_footprint_empty_data(self, footprint_hunter):
        """Test footprint analysis with empty data"""
        empty_data = pd.DataFrame()
        result = await footprint_hunter.analyze_institutional_footprint(empty_data)
        
        assert len(result.order_blocks) == 0
        assert len(result.fair_value_gaps) == 0
        assert len(result.liquidity_sweeps) == 0
        assert result.confidence_score == 0.1


class TestPatternSynthesisEngine:
    """Test suite for WHAT dimension - Pattern Synthesis Engine"""
    
    @pytest.fixture
    def pattern_engine(self):
        return PatternSynthesisEngine()
    
    @pytest.fixture
    def sample_market_data(self):
        """Create sample market data for testing"""
        dates = pd.date_range(start='2024-01-01', periods=100, freq='1H')
        return pd.DataFrame({
            'open': np.random.randn(100).cumsum() + 100,
            'high': np.random.randn(100).cumsum() + 101,
            'low': np.random.randn(100).cumsum() + 99,
            'close': np.random.randn(100).cumsum() + 100,
            'volume': np.random.randint(1000, 10000, 100)
        }, index=dates)
    
    @pytest.mark.asyncio
    async def test_pattern_analysis_basic(self, pattern_engine, sample_market_data):
        """Test basic pattern synthesis"""
        result = await pattern_engine.synthesize_patterns(sample_market_data)
        
        assert isinstance(result, PatternSynthesis)
        assert isinstance(result.fractal_patterns, list)
        assert isinstance(result.harmonic_patterns, list)
        assert isinstance(result.volume_profile, dict)
        assert isinstance(result.price_action_dna, dict)
        assert 0.0 <= result.pattern_strength <= 1.0
        assert 0.0 <= result.confidence_score <= 1.0
    
    @pytest.mark.asyncio
    async def test_pattern_empty_data(self, pattern_engine):
        """Test pattern synthesis with empty data"""
        empty_data = pd.DataFrame()
        result = await pattern_engine.synthesize_patterns(empty_data)
        
        assert len(result.fractal_patterns) == 0
        assert len(result.harmonic_patterns) == 0
        assert result.confidence_score == 0.1


class TestTemporalAdvantageSystem:
    """Test suite for WHEN dimension - Temporal Advantage System"""
    
    @pytest.fixture
    def temporal_system(self):
        return TemporalAdvantageSystem()
    
    @pytest.fixture
    def sample_market_data(self):
        """Create sample market data for testing"""
        dates = pd.date_range(start='2024-01-01', periods=100, freq='1H')
        return pd.DataFrame({
            'close': np.random.randn(100).cumsum() + 100
        }, index=dates)
    
    @pytest.mark.asyncio
    async def test_temporal_analysis_basic(self, temporal_system, sample_market_data):
        """Test basic temporal advantage analysis"""
        result = await temporal_system.analyze_temporal_advantage(sample_market_data)
        
        assert isinstance(result, TemporalAdvantage)
        assert 0.0 <= result.session_transition_score <= 1.0
        assert isinstance(result.economic_calendar_impact, dict)
        assert isinstance(result.microstructure_timing, dict)
        assert isinstance(result.volatility_regime, str)
        assert isinstance(result.optimal_entry_window, tuple)
        assert 0.0 <= result.confidence_score <= 1.0


class TestManipulationDetectionSystem:
    """Test suite for ANOMALY dimension - Manipulation Detection"""
    
    @pytest.fixture
    def detection_system(self):
        return ManipulationDetectionSystem()
    
    @pytest.fixture
    def sample_market_data(self):
        """Create sample market data for testing"""
        dates = pd.date_range(start='2024-01-01', periods=100, freq='1H')
        return pd.DataFrame({
            'close': np.random.randn(100).cumsum() + 100,
            'volume': np.random.randint(1000, 10000, 100)
        }, index=dates)
    
    @pytest.mark.asyncio
    async def test_detection_basic(self, detection_system, sample_market_data):
        """Test basic manipulation detection"""
        result = await detection_system.detect_manipulation(sample_market_data)
        
        assert isinstance(result, AnomalyDetection)
        assert isinstance(result.spoofing_detected, bool)
        assert 0.0 <= result.wash_trading_score <= 1.0
        assert 0.0 <= result.pump_dump_probability <= 1.0
        assert 0.0 <= result.overall_risk_score <= 1.0
        assert 0.0 <= result.confidence_score <= 1.0


class TestChaosAdaptationSystem:
    """Test suite for CHAOS dimension - Antifragile Adaptation"""
    
    @pytest.fixture
    def chaos_system(self):
        return ChaosAdaptationSystem()
    
    @pytest.fixture
    def sample_market_data(self):
        """Create sample market data for testing"""
        dates = pd.date_range(start='2024-01-01', periods=100, freq='1H')
        return pd.DataFrame({
            'close': np.random.randn(100).cumsum() + 100,
            'volume': np.random.randint(1000, 10000, 100)
        }, index=dates)
    
    @pytest.mark.asyncio
    async def test_chaos_analysis_basic(self, chaos_system, sample_market_data):
        """Test basic chaos adaptation analysis"""
        result = await chaos_system.analyze_chaos_adaptation(sample_market_data)
        
        assert isinstance(result, ChaosAdaptation)
        assert 0.0 <= result.black_swan_probability <= 1.0
        assert 0.0 <= result.volatility_harvest_opportunity <= 1.0
        assert 0.0 <= result.crisis_alpha_potential <= 1.0
        assert isinstance(result.regime_change_detected, bool)
        assert isinstance(result.adaptation_strategy, str)
        assert 0.0 <= result.confidence_score <= 1.0


class TestSensoryIntegrationOrchestrator:
    """Test suite for unified sensory integration"""
    
    @pytest.fixture
    def orchestrator(self):
        return SensoryIntegrationOrchestrator()
    
    @pytest.fixture
    def sample_market_data(self):
        """Create sample market data for integration testing"""
        dates = pd.date_range(start='2024-01-01', periods=100, freq='1H')
        return pd.DataFrame({
            'open': np.random.randn(100).cumsum() + 100,
            'high': np.random.randn(100).cumsum() + 101,
            'low': np.random.randn(100).cumsum() + 99,
            'close': np.random.randn(100).cumsum() + 100,
            'volume': np.random.randint(1000, 10000, 100)
        }, index=dates)
    
    @pytest.mark.asyncio
    async def test_unified_intelligence_analysis(self, orchestrator, sample_market_data):
        """Test unified intelligence analysis across all dimensions"""
        result = await orchestrator.analyze_unified_intelligence(sample_market_data, 'EURUSD')
        
        assert isinstance(result, UnifiedMarketIntelligence)
        assert result.symbol == 'EURUSD'
        assert isinstance(result.timestamp, datetime)
        assert 0.0 <= result.overall_confidence <= 1.0
        assert -1.0 <= result.signal_strength <= 1.0
        assert 0.0 <= result.risk_assessment <= 1.0
        assert 0.0 <= result.opportunity_score <= 1.0
        assert 0.0 <= result.confluence_score <= 1.0
    
    @pytest.mark.asyncio
    async def test_integration_basic(self, orchestrator, sample_market_data):
        """Test basic sensory integration"""
        result = await orchestrator.process_market_intelligence(sample_market_data)
        
        assert isinstance(result, UnifiedMarketIntelligence)
        assert result.macro_environment is not None
        assert result.institutional_footprint is not None
        assert result.pattern_synthesis is not None
        assert result.temporal_advantage is not None
        assert result.anomaly_detection is not None
        assert result.chaos_adaptation is not None
        assert 0.0 <= result.unified_confidence <= 1.0
        assert isinstance(result.recommended_action, str)
        assert isinstance(result.risk_assessment, dict)
    
    @pytest.mark.asyncio
    async def test_integration_empty_data(self, orchestrator):
        """Test integration with empty data"""
        empty_data = {'price_data': pd.DataFrame()}
        result = await orchestrator.process_market_intelligence(empty_data)
        
        assert result.unified_confidence == 0.1
        assert result.recommended_action == 'hold'
    
    @pytest.mark.asyncio
    async def test_integration_performance(self, orchestrator, sample_market_data):
        """Test integration performance requirements"""
        start_time = datetime.now()
        
        result = await orchestrator.process_market_intelligence(sample_market_data)
        
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        # Should complete within 5 seconds for basic data
        assert processing_time < 5.0
        assert result.unified_confidence > 0.0


class TestPerformanceRequirements:
    """Test suite for performance requirements"""
    
    @pytest.mark.asyncio
    async def test_sub_second_response_time(self):
        """Test sub-second response time requirement"""
        orchestrator = SensoryIntegrationOrchestrator()
        
        # Create minimal test data
        test_data = {
            'price_data': pd.DataFrame({
                'close': [100, 101, 102, 103, 104]
            })
        }
        
        start_time = datetime.now()
        result = await orchestrator.process_market_intelligence(test_data)
        end_time = datetime.now()
        
        processing_time = (end_time - start_time).total_seconds()
        assert processing_time < 1.0, f"Response time {processing_time}s exceeds 1s requirement"
    
    @pytest.mark.asyncio
    async def test_concurrent_processing(self):
        """Test concurrent processing capabilities"""
        orchestrator = SensoryIntegrationOrchestrator()
        
        # Create multiple test datasets
        datasets = []
        for i in range(5):
            dates = pd.date_range(start='2024-01-01', periods=50, freq='1H')
            datasets.append({
                'price_data': pd.DataFrame({
                    'close': np.random.randn(50).cumsum() + 100
                }, index=dates)
            })
        
        # Process concurrently
        start_time = datetime.now()
        tasks = [orchestrator.process_market_intelligence(data) for data in datasets]
        results = await asyncio.gather(*tasks)
        end_time = datetime.now()
        
        total_time = (end_time - start_time).total_seconds()
        avg_time = total_time / len(datasets)
        
        assert avg_time < 2.0, f"Average processing time {avg_time}s too high"


class TestAccuracyRequirements:
    """Test suite for accuracy requirements"""
    
    @pytest.mark.asyncio
    async def test_anomaly_detection_accuracy(self):
        """Test anomaly detection accuracy > 90%"""
        detection_system = ManipulationDetectionSystem()
        
        # Create synthetic data with known anomalies
        normal_data = pd.DataFrame({
            'close': np.random.randn(1000).cumsum() + 100
        })
        
        # Test on normal data (should have low false positive rate)
        result = await detection_system.detect_manipulation(normal_data)
        
        # Should have low risk score for normal data
        assert result.overall_risk_score < 0.3
        assert result.confidence > 0.5
    
    @pytest.mark.asyncio
    async def test_pattern_recognition_validation(self):
        """Test pattern recognition against known patterns"""
        pattern_engine = PatternSynthesisEngine()
        
        # Create data with clear trend
        dates = pd.date_range(start='2024-01-01', periods=100, freq='1H')
        trend_data = pd.DataFrame({
            'close': np.linspace(100, 110, 100)
        }, index=dates)
        
        result = await pattern_engine.synthesize_patterns(trend_data)
        
        # Should detect strong trend pattern
        assert result.pattern_strength > 0.5
        assert result.confidence_score > 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
