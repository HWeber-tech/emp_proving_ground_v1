from __future__ import annotations

import importlib
import logging
import sys
from typing import Any, Dict

import pytest

from src.core.base import MarketData

# Forwarder vs canonical identity
from src.market_intelligence.dimensions.enhanced_what_dimension import (
    TechnicalRealityEngine as TE_MI,
)
from src.market_intelligence.dimensions.enhanced_why_dimension import (
    EnhancedFundamentalIntelligenceEngine as WHY_MI,
)
from src.sensory.enhanced.what_dimension import (
    TechnicalRealityEngine as TE_SENS,
)
from src.sensory.enhanced.why_dimension import (
    EnhancedFundamentalIntelligenceEngine as WHY_SENS,
)
from src.sensory.organs.dimensions.base_organ import MarketRegime


def test_forwarded_class_identities():
    # WHAT: MI shim lazily forwards to sensory canonical class
    assert TE_MI is TE_SENS
    # WHY: MI shim lazily forwards to sensory canonical class
    assert WHY_MI is WHY_SENS


@pytest.mark.asyncio
async def test_what_engine_behavior_and_source_tag():
    # Construct minimal MarketData (compat wrapper infers OHLC from bid/ask/price)
    md = MarketData(timestamp=None, bid=1.0, ask=1.0002, volume=1000.0, volatility=0.01)

    engine = TE_MI()  # forwarded to sensory
    reading = await engine.analyze_technical_reality(md)

    # Regime is a MarketRegime instance, and source should be tagged via context
    assert isinstance(reading.regime, MarketRegime)
    assert isinstance(getattr(reading, "context", {}), dict)
    assert reading.context.get("source") == "sensory.what"


def test_why_engine_behavior_and_source_tag():
    engine = WHY_MI()  # forwarded to sensory
    out: Dict[str, Any] = engine.analyze_fundamental_intelligence({})

    assert isinstance(out, dict)
    assert "meta" in out and isinstance(out["meta"], dict)
    assert out["meta"].get("source") == "sensory.why"


def test_lazy_cache_and_dir_exposure_for_what():
    import src.market_intelligence.dimensions.enhanced_what_dimension as miw

    # Repeated getattr should return identical cached object
    first = getattr(miw, "TechnicalRealityEngine")
    second = getattr(miw, "TechnicalRealityEngine")
    assert id(first) == id(second)

    # __dir__ exposes forwarded names
    names = dir(miw)
    assert "TechnicalRealityEngine" in names


def test_lazy_cache_and_dir_exposure_for_why():
    import src.market_intelligence.dimensions.enhanced_why_dimension as miy

    # Repeated getattr should return identical cached object
    first = getattr(miy, "EnhancedFundamentalIntelligenceEngine")
    second = getattr(miy, "EnhancedFundamentalIntelligenceEngine")
    assert id(first) == id(second)

    # __dir__ exposes forwarded names
    names = dir(miy)
    assert "EnhancedFundamentalIntelligenceEngine" in names


def test_no_import_time_logs_in_mi_modules(caplog: pytest.LogCaptureFixture):
    # Ensure MI modules have no side effects like logging on import
    caplog.set_level(logging.INFO)

    # Force fresh import of the modules in an isolated manner
    for mod in [
        "src.market_intelligence.dimensions.enhanced_what_dimension",
        "src.market_intelligence.dimensions.enhanced_why_dimension",
    ]:
        sys.modules.pop(mod, None)
        importlib.import_module(mod)

    # Check there are no indicative log messages
    forbidden = ("Starting", "Configured logging")
    assert all(not any(frag in rec.getMessage() for frag in forbidden) for rec in caplog.records), (
        "Import-time log side effects detected in MI modules"
    )
