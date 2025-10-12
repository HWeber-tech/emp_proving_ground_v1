from __future__ import annotations

import importlib
import logging
import sys
from typing import Any, Dict

import pytest

from src.core.base import MarketData
from src.sensory.enhanced.what_dimension import TechnicalRealityEngine
from src.sensory.enhanced.why_dimension import EnhancedFundamentalUnderstandingEngine
from src.sensory.organs.dimensions.base_organ import MarketRegime


def test_sensory_dimension_exports_are_canonical() -> None:
    why_module = importlib.import_module("src.sensory.enhanced.why_dimension")
    assert TechnicalRealityEngine.__module__ == "src.sensory.enhanced.what_dimension"
    assert (
        EnhancedFundamentalUnderstandingEngine.__module__
        == "src.sensory.enhanced.why_dimension"
    )
    with pytest.raises(AttributeError):
        getattr(why_module, "EnhancedFundamentalIntelligenceEngine")


@pytest.mark.asyncio
async def test_what_engine_behavior_and_source_tag() -> None:
    md = MarketData(timestamp=None, bid=1.0, ask=1.0002, volume=1000.0, volatility=0.01)

    engine = TechnicalRealityEngine()
    reading = await engine.analyze_technical_reality(md)

    assert isinstance(reading.regime, MarketRegime)
    assert isinstance(getattr(reading, "context", {}), dict)
    assert reading.context.get("source") == "sensory.what"


def test_why_engine_behavior_and_source_tag() -> None:
    engine = EnhancedFundamentalUnderstandingEngine()
    out: Dict[str, Any] = engine.analyze_fundamental_understanding({})

    assert isinstance(out, dict)
    assert "meta" in out and isinstance(out["meta"], dict)
    assert out["meta"].get("source") == "sensory.why"


def test_sensory_modules_do_not_log_on_import(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO)

    for mod in [
        "src.sensory.enhanced.what_dimension",
        "src.sensory.enhanced.why_dimension",
    ]:
        sys.modules.pop(mod, None)
        importlib.import_module(mod)

    forbidden = ("Starting", "Configured logging")
    assert all(not any(frag in rec.getMessage() for frag in forbidden) for rec in caplog.records), (
        "Import-time log side effects detected in sensory modules"
    )
