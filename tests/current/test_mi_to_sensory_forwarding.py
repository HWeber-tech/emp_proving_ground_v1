from __future__ import annotations

import importlib
import logging
import sys
from typing import Any, Dict

import pytest

from src.core.base import MarketData
from src.sensory.enhanced.what_dimension import TechnicalRealityEngine
from src.sensory.enhanced.why_dimension import EnhancedFundamentalIntelligenceEngine
from src.sensory.organs.dimensions.base_organ import MarketRegime


def test_canonical_classes_exposed_via_modules() -> None:
    import src.sensory.enhanced.what_dimension as what_mod
    import src.sensory.enhanced.why_dimension as why_mod

    assert getattr(what_mod, "TechnicalRealityEngine") is TechnicalRealityEngine
    assert getattr(why_mod, "EnhancedFundamentalIntelligenceEngine") is (
        EnhancedFundamentalIntelligenceEngine
    )


@pytest.mark.asyncio
async def test_what_engine_behavior_and_source_tag() -> None:
    md = MarketData(timestamp=None, bid=1.0, ask=1.0002, volume=1000.0, volatility=0.01)

    engine = TechnicalRealityEngine()
    reading = await engine.analyze_technical_reality(md)

    assert isinstance(reading.regime, MarketRegime)
    assert isinstance(getattr(reading, "context", {}), dict)
    assert reading.context.get("source") == "sensory.what"


def test_why_engine_behavior_and_source_tag() -> None:
    engine = EnhancedFundamentalIntelligenceEngine()
    out: Dict[str, Any] = engine.analyze_fundamental_intelligence({})

    assert isinstance(out, dict)
    assert "meta" in out and isinstance(out["meta"], dict)
    assert out["meta"].get("source") == "sensory.why"


def test_no_import_time_logs_in_sensory_modules(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO)

    for mod in [
        "src.sensory.enhanced.what_dimension",
        "src.sensory.enhanced.why_dimension",
    ]:
        sys.modules.pop(mod, None)
        importlib.import_module(mod)

    forbidden = ("Starting", "Configured logging")
    assert all(not any(frag in rec.getMessage() for frag in forbidden) for rec in caplog.records), (
        "Import-time log side effects detected in sensory enhanced modules"
    )
