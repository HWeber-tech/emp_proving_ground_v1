from __future__ import annotations

import importlib
import logging
import sys
from typing import Any, Dict

import pytest

from src.core.base import MarketData

# ANOMALY
from src.market_intelligence.dimensions.enhanced_anomaly_dimension import (
    AnomalyIntelligenceEngine as ANOM_MI,
)

# HOW
from src.market_intelligence.dimensions.enhanced_how_dimension import (
    InstitutionalIntelligenceEngine as HOW_MI,
)

# WHEN
from src.market_intelligence.dimensions.enhanced_when_dimension import (
    ChronalIntelligenceEngine as WHEN_MI,
)
from src.sensory.enhanced.anomaly_dimension import (
    AnomalyIntelligenceEngine as ANOM_SENS,
)
from src.sensory.enhanced.how_dimension import (
    InstitutionalIntelligenceEngine as HOW_SENS,
)
from src.sensory.enhanced.when_dimension import (
    ChronalIntelligenceEngine as WHEN_SENS,
)
from src.sensory.organs.dimensions.base_organ import MarketRegime


def test_forwarded_class_identities_phase2():
    # Ensure MI shims lazily forward to sensory canonical classes
    assert WHEN_MI is WHEN_SENS
    assert HOW_MI is HOW_SENS
    assert ANOM_MI is ANOM_SENS


def test_when_engine_behavior_and_meta_tag():
    # Minimal MarketData; wrapper infers OHLC from bid/ask
    md = MarketData(timestamp=None, bid=1.0, ask=1.0002, volume=100.0)

    engine = WHEN_MI()
    reading = engine.analyze_temporal_intelligence(md)

    assert isinstance(reading.regime, MarketRegime)
    assert isinstance(getattr(reading, "context", {}), dict)
    # Robust check: support either nested meta or direct source in context
    meta = reading.context.get("meta", {})
    if isinstance(meta, dict) and "source" in meta:
        assert meta.get("source") == "sensory.when"
    else:
        assert reading.context.get("source") == "sensory.when"


def test_how_engine_behavior_and_meta_tag():
    engine = HOW_MI()
    out: Dict[str, Any] = engine.analyze_institutional_intelligence({})
    assert isinstance(out, dict)
    assert "meta" in out and isinstance(out["meta"], dict)
    assert out["meta"].get("source") == "sensory.how"


def test_anomaly_engine_behavior_and_meta_tag():
    engine = ANOM_MI()
    out: Dict[str, Any] = engine.analyze_anomaly_intelligence([0.0, 0.1, -0.2])
    assert isinstance(out, dict)
    assert "meta" in out and isinstance(out["meta"], dict)
    assert out["meta"].get("source") == "sensory.anomaly"


def test_lazy_cache_and_dir_exposure_for_when():
    import src.market_intelligence.dimensions.enhanced_when_dimension as miw

    first = getattr(miw, "ChronalIntelligenceEngine")
    second = getattr(miw, "ChronalIntelligenceEngine")
    assert id(first) == id(second)

    names = dir(miw)
    assert "ChronalIntelligenceEngine" in names


def test_lazy_cache_and_dir_exposure_for_how():
    import src.market_intelligence.dimensions.enhanced_how_dimension as mih

    first = getattr(mih, "InstitutionalIntelligenceEngine")
    second = getattr(mih, "InstitutionalIntelligenceEngine")
    assert id(first) == id(second)

    names = dir(mih)
    assert "InstitutionalIntelligenceEngine" in names


def test_lazy_cache_and_dir_exposure_for_anomaly():
    import src.market_intelligence.dimensions.enhanced_anomaly_dimension as mia

    first = getattr(mia, "AnomalyIntelligenceEngine")
    second = getattr(mia, "AnomalyIntelligenceEngine")
    assert id(first) == id(second)

    names = dir(mia)
    assert "AnomalyIntelligenceEngine" in names


def test_no_import_time_logs_in_mi_modules_phase2(caplog: pytest.LogCaptureFixture):
    # Ensure MI modules have no side effects like logging on import
    caplog.set_level(logging.INFO)

    for mod in [
        "src.market_intelligence.dimensions.enhanced_when_dimension",
        "src.market_intelligence.dimensions.enhanced_how_dimension",
        "src.market_intelligence.dimensions.enhanced_anomaly_dimension",
    ]:
        sys.modules.pop(mod, None)
        importlib.import_module(mod)

    forbidden = ("Starting", "Configured logging")
    assert all(not any(frag in rec.getMessage() for frag in forbidden) for rec in caplog.records), (
        "Import-time log side effects detected in MI modules (phase 2)"
    )
