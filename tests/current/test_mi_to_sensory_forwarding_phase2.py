from __future__ import annotations

import importlib
import logging
import sys
from typing import Any, Dict

import pytest

from src.core.base import MarketData
from src.sensory.enhanced.anomaly_dimension import AnomalyIntelligenceEngine
from src.sensory.enhanced.how_dimension import InstitutionalIntelligenceEngine
from src.sensory.enhanced.when_dimension import ChronalIntelligenceEngine
from src.sensory.organs.dimensions.base_organ import MarketRegime


def test_canonical_phase2_classes_exposed_via_modules() -> None:
    import src.sensory.enhanced.anomaly_dimension as anomaly_mod
    import src.sensory.enhanced.how_dimension as how_mod
    import src.sensory.enhanced.when_dimension as when_mod

    assert getattr(anomaly_mod, "AnomalyIntelligenceEngine") is AnomalyIntelligenceEngine
    assert getattr(how_mod, "InstitutionalIntelligenceEngine") is InstitutionalIntelligenceEngine
    assert getattr(when_mod, "ChronalIntelligenceEngine") is ChronalIntelligenceEngine


def test_when_engine_behavior_and_meta_tag() -> None:
    md = MarketData(timestamp=None, bid=1.0, ask=1.0002, volume=100.0)

    engine = ChronalIntelligenceEngine()
    reading = engine.analyze_temporal_intelligence(md)

    assert isinstance(reading.regime, MarketRegime)
    assert isinstance(getattr(reading, "context", {}), dict)
    meta = reading.context.get("meta", {})
    if isinstance(meta, dict) and "source" in meta:
        assert meta.get("source") == "sensory.when"
    else:
        assert reading.context.get("source") == "sensory.when"


def test_how_engine_behavior_and_meta_tag() -> None:
    engine = InstitutionalIntelligenceEngine()
    out: Dict[str, Any] = engine.analyze_institutional_intelligence({})

    assert isinstance(out, dict)
    assert "meta" in out and isinstance(out["meta"], dict)
    assert out["meta"].get("source") == "sensory.how"


def test_anomaly_engine_behavior_and_meta_tag() -> None:
    engine = AnomalyIntelligenceEngine()
    out: Dict[str, Any] = engine.analyze_anomaly_intelligence([0.0, 0.1, -0.2])

    assert isinstance(out, dict)
    assert "meta" in out and isinstance(out["meta"], dict)
    assert out["meta"].get("source") == "sensory.anomaly"


def test_no_import_time_logs_in_phase2_sensory_modules(caplog: pytest.LogCaptureFixture) -> None:
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
        "Import-time log side effects detected in sensory enhanced modules (phase 2)"
    )
