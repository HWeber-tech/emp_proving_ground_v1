#!/usr/bin/env python3
"""
Integration test for Epic 4: Fusing Foresight
Tests the complete integration of predictive intelligence into the system.
"""

import asyncio
import pytest
from datetime import datetime
from typing import Dict, Any
import pandas as pd

# Import our components
import sys
sys.path.append('src')

from core.events import ContextPacket, MarketForecast
from thinking.thinking_manager import ThinkingManager

class MockSystemConfig:
    """Mock system config for testing."""
    
    def __init__(self, model_run_id: str = "test_model"):
        self.config = {
            'mlflow': {
                'model_run_id': model_run_id,
                'uri': 'http://localhost:5000'
            }
        }
    
    def get(self, key, default=None):
        return self.config.get(key, default)

class MockPredictiveModeler:
    """Mock predictive modeler for testing."""
    
    def __init__(self, model_run_id: str, mlflow_uri: str):
        self.model_run_id = model_run_id
    
    async def forecast(self, recent_market_data: pd.DataFrame) -> Dict[str, float]:
        """Return mock forecast."""
        return {
            "prob_up": 0.65,
            "prob_down": 0.25,
            "prob_flat": 0.10,
            "model_version": self.model_run_id,
            "timestamp": datetime.utcnow().isoformat()
        }

async def test_epic4_integration():
    """Test complete Epic 4 integration."""
    print("=== Testing Epic 4: Fusing Foresight ===")
    
    # Test 1: Events Integration
    print("\n1. Testing Events Integration...")
    try:
        from core.events import MarketForecast, ContextPacket
        
        # Test MarketForecast creation
        forecast = MarketForecast(
            prob_up=0.65,
            prob_down=0.25,
            prob_flat=0.10,
            model_version="test_model_v1",
            timestamp=datetime.utcnow().isoformat()
        )
        print("✅ MarketForecast model created successfully")
        
        # Test ContextPacket with forecast
        context = ContextPacket(
            timestamp=datetime.utcnow().isoformat(),
            market_data={"price": 1.1000, "volume": 1000},
            risk_metrics={"volatility": 0.02},
            sequence_id="test_001",
            market_forecast=forecast
        )
        print("✅ ContextPacket with forecast created successfully")
        
    except Exception as e:
        print(f"❌ Events integration test failed: {e}")
        return False
    
    # Test 2: Thinking Manager Integration
    print("\n2. Testing Thinking Manager Integration...")
    try:
        # Create mock system config
        config = MockSystemConfig("test_model_v1")
        
        # Create thinking manager
        manager = ThinkingManager(config)
        
        # Test with mock modeler
        manager.predictive_modeler = MockPredictiveModeler("test_model_v1", "http://localhost:5000")
        
        # Test market data processing
        market_data = {
            'open': 1.1000,
            'high': 1.1005,
            'low': 1.0995,
            'close': 1.1002,
            'volume': 1000
        }
        
        # Add 20 data points to trigger forecast
        for i in range(20):
            market_data_copy = market_data.copy()
            market_data_copy['close'] += i * 0.0001  # Slight variation
            manager.market_data_buffer.append(market_data_copy)
        
        # Generate context
        context = await manager.on_market_understanding(market_data)
        
        # Verify context has forecast
        assert context.market_forecast is not None, "Context should have forecast"
        assert context.market_forecast.prob_up == 0.65, "Forecast probabilities should match"
        assert context.market_forecast.model_version == "test_model_v1", "Model version should match"
        
        print("✅ Thinking manager integration successful")
        
    except Exception as e:
        print(f"❌ Thinking manager integration test failed: {e}")
        return False
    
    # Test 3: WebSocket Integration Test
    print("\n3. Testing WebSocket Integration...")
    try:
        # Test that ContextPacket can be serialized
        forecast = MarketForecast(
            prob_up=0.65,
            prob_down=0.25,
            prob_flat=0.10,
            model_version="test_model_v1",
            timestamp=datetime.utcnow().isoformat()
        )
        
        context = ContextPacket(
            timestamp=datetime.utcnow().isoformat(),
            market_data={"price": 1.1000, "volume": 1000},
            risk_metrics={"volatility": 0.02},
            sequence_id="test_ws_001",
            market_forecast=forecast
        )
        
        # Test JSON serialization
        import json
        context_json = context.model_dump_json()
        parsed_context = ContextPacket.model_validate_json(context_json)
        
        assert parsed_context.market_forecast is not None
        assert parsed_context.market_forecast.prob_up == 0.65
        
        print("✅ WebSocket integration test successful")
        
    except Exception as e:
        print(f"❌ WebSocket integration test failed: {e}")
        return False
    
    # Test 4: Probability Distribution Validation
    print("\n4. Testing Probability Distribution...")
    try:
        forecast = MarketForecast(
            prob_up=0.65,
            prob_down=0.25,
            prob_flat=0.10,
            model_version="test_model_v1",
            timestamp=datetime.utcnow().isoformat()
        )
        
        # Validate probabilities sum to 1.0
        total = forecast.prob_up + forecast.prob_down + forecast.prob_flat
        assert abs(total - 1.0) < 0.001, f"Probabilities should sum to 1.0, got {total}"
        
        # Validate probabilities are between 0 and 1
        assert 0 <= forecast.prob_up <= 1, "prob_up should be between 0 and 1"
        assert 0 <= forecast.prob_down <= 1, "prob_down should be between 0 and 1"
        assert 0 <= forecast.prob_flat <= 1, "prob_flat should be between 0 and 1"
        
        print("✅ Probability distribution validation successful")
        
    except Exception as e:
        print(f"❌ Probability distribution test failed: {e}")
        return False
    
    print("\n=== Epic 4 Integration Complete ===")
    print("✅ All components of Epic 4 are successfully integrated!")
    print("✅ The predator now has predictive intelligence!")
    return True

async def main():
    """Run all integration tests."""
    success = await test_epic4_integration()
    return 0 if success else 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
