"""
End-to-End Startup Integration Test

Tests that the main Application can be instantiated and all its
core components can be wired up without raising dependency errors.
"""

import pytest
import asyncio
import sys
from pathlib import Path

# Add src to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Test basic imports
def test_basic_imports():
    """Test that basic components can be imported."""
    try:
        try:
            from src.core.models import InstrumentMeta  # legacy
        except Exception:  # pragma: no cover
            InstrumentMeta = None  # type: ignore
        from src.data_integration.real_data_integration import RealDataManager
        
        # Test InstrumentMeta creation
        instrument = InstrumentMeta(
            symbol="EURUSD",
            pip_size=0.0001,
            lot_size=100000
        )
        
        assert instrument.symbol == "EURUSD"
        assert instrument.pip_size == 0.0001
        
        # Test RealDataManager initialization
        data_manager = RealDataManager({})
        assert data_manager is not None
        
    except Exception as e:
        pytest.fail(f"Basic imports failed: {e}")


def test_instrument_meta_creation():
    """Test InstrumentMeta model creation."""
    try:
        from src.core.models import InstrumentMeta  # legacy
    except Exception:  # pragma: no cover
        InstrumentMeta = None  # type: ignore
    
    # Test basic creation
    instrument = InstrumentMeta(
        symbol="EURUSD",
        pip_size=0.0001,
        lot_size=100000
    )
    
    assert instrument.symbol == "EURUSD"
    assert instrument.pip_size == 0.0001
    assert instrument.lot_size == 100000
    assert instrument.commission == 0.0  # Default value
    
    # Test with all parameters
    instrument2 = InstrumentMeta(
        symbol="GBPUSD",
        pip_size=0.0001,
        lot_size=100000,
        commission=0.001,
        spread=0.0002,
        leverage=50
    )
    
    assert instrument2.symbol == "GBPUSD"
    assert instrument2.leverage == 50


def test_data_manager_initialization():
    """Test that data manager initializes correctly."""
    from src.data_integration.real_data_integration import RealDataManager
    
    # Test with empty config
    data_manager = RealDataManager({})
    assert data_manager is not None
    assert len(data_manager.get_available_sources()) >= 1  # At least Yahoo Finance
    
    # Test with mock config
    mock_config = {
        'fallback_to_mock': True,
        'cache_duration': 300
    }
    data_manager2 = RealDataManager(mock_config)
    assert data_manager2 is not None


@pytest.mark.asyncio
async def test_data_manager_functionality():
    """Test data manager basic functionality."""
    from src.data_integration.real_data_integration import RealDataManager
    
    data_manager = RealDataManager({})
    
    # Test available sources
    sources = data_manager.get_available_sources()
    assert isinstance(sources, list)
    
    # Test data quality report
    quality = data_manager.get_data_quality_report()
    assert isinstance(quality, dict)


def test_token_manager_import():
    """Test token manager can be imported."""
    try:
        from src.governance.token_manager import TokenManager
        try:
            from src.core.models import TokenData  # legacy
        except Exception:  # pragma: no cover
            TokenData = None  # type: ignore
        
        # Test TokenData creation
        token = TokenData(
            access_token="test_token",
            expires_in=3600
        )
        
        assert token.access_token == "test_token"
        assert token.expires_in == 3600
        
    except Exception as e:
        pytest.fail(f"Token manager import failed: {e}")


def test_governance_models_import():
    """Test governance models can be imported."""
    try:
        from src.governance.models import StrategyModel
        
        # Test StrategyModel creation
        strategy = StrategyModel(
            genome_id="test_genome_001",
            dna="sample_dna_data",
            fitness_score=0.85,
            generation=1,
            is_champion=True
        )
        
        assert strategy.genome_id == "test_genome_001"
        assert strategy.fitness_score == 0.85
        
    except Exception as e:
        pytest.fail(f"Governance models import failed: {e}")


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v"])
