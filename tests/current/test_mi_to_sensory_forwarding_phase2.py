from __future__ import annotations

import importlib
import logging
import sys
from typing import Any, Dict

import pytest

from src.core.base import MarketData
from src.orchestration.enhanced_understanding_engine import (
    ContextualFusionEngine as CanonicalFusionEngine,
)
from src.orchestration.enhanced_understanding_engine import (
    Synthesis as CanonicalSynthesis,
)
from src.sensory.enhanced.anomaly_dimension import AnomalyUnderstandingEngine
from src.sensory.enhanced.how_dimension import InstitutionalUnderstandingEngine
from src.sensory.enhanced.when_dimension import ChronalUnderstandingEngine
from src.sensory.organs.dimensions.base_organ import MarketRegime


def test_phase2_sensory_dimension_exports_are_canonical() -> None:
    assert ChronalUnderstandingEngine.__module__ == "src.sensory.enhanced.when_dimension"
    assert (
        InstitutionalUnderstandingEngine.__module__
        == "src.sensory.enhanced.how_dimension"
    )
    assert AnomalyUnderstandingEngine.__module__ == "src.sensory.enhanced.anomaly_dimension"


def test_when_engine_behavior_and_meta_tag() -> None:
    md = MarketData(timestamp=None, bid=1.0, ask=1.0002, volume=100.0)

    engine = ChronalUnderstandingEngine()
    reading = engine.analyze_temporal_understanding(md)

    assert isinstance(reading.regime, MarketRegime)
    assert isinstance(getattr(reading, "context", {}), dict)
    meta = reading.context.get("meta", {})
    if isinstance(meta, dict) and "source" in meta:
        assert meta.get("source") == "sensory.when"
    else:
        assert reading.context.get("source") == "sensory.when"


def test_how_engine_behavior_and_meta_tag() -> None:
    engine = InstitutionalUnderstandingEngine()
    out: Dict[str, Any] = engine.analyze_institutional_understanding({})
    assert isinstance(out, dict)
    assert "meta" in out and isinstance(out["meta"], dict)
    assert out["meta"].get("source") == "sensory.how"


def test_anomaly_engine_behavior_and_meta_tag() -> None:
    engine = AnomalyUnderstandingEngine()
    out: Dict[str, Any] = engine.analyze_anomaly_understanding([0.0, 0.1, -0.2])
    assert isinstance(out, dict)
    assert "meta" in out and isinstance(out["meta"], dict)
    assert out["meta"].get("source") == "sensory.anomaly"


def test_sensory_phase2_modules_do_not_log_on_import(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO)

    for mod in [
        "src.sensory.enhanced.when_dimension",
        "src.sensory.enhanced.how_dimension",
        "src.sensory.enhanced.anomaly_dimension",
    ]:
        sys.modules.pop(mod, None)
        importlib.import_module(mod)

    forbidden = ("Starting", "Configured logging")
    assert all(not any(frag in rec.getMessage() for frag in forbidden) for rec in caplog.records), (
        "Import-time log side effects detected in sensory modules (phase 2)"
    )


@pytest.mark.asyncio
async def test_contextual_fusion_engine_uses_canonical_dimensions() -> None:
    engine = CanonicalFusionEngine()
    market_data = MarketData(timestamp=None, bid=1.0, ask=1.0001, volume=250.0)

    synthesis = await engine.analyze_market_understanding(market_data)

    assert isinstance(synthesis, CanonicalSynthesis)
    assert engine.current_readings["WHEN"].context.get("source") == "sensory.when"
    assert engine.current_readings["HOW"].context.get("source") == "sensory.how"
    assert engine.current_readings["ANOMALY"].context.get("source") == "sensory.anomaly"
