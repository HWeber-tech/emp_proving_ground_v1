#!/usr/bin/env python3

from __future__ import annotations

from typing import Any, Dict

import pandas as pd
import pytest

from src.sensory.organs.dimensions.pattern_engine import PatternSynthesis
from src.sensory.what.patterns.orchestrator import PatternOrchestrator


@pytest.mark.asyncio
async def test_orchestrator_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    orchestrator = PatternOrchestrator()

    async def fake_synthesize_patterns(df: pd.DataFrame) -> PatternSynthesis:
        return PatternSynthesis(
            fractal_patterns=[{"pattern_type": "elliott_wave_5"}],
            harmonic_patterns=[{"pattern_name": "gartley"}],
            volume_profile={"point_of_control": 1.2345},
            price_action_dna={"dna_sequence": "Vm rABC"},
            pattern_strength=0.42,
            confidence_score=0.88,
        )

    # Patch the engine method to avoid heavy computations
    monkeypatch.setattr(orchestrator.engine, "synthesize_patterns", fake_synthesize_patterns)

    # Minimal but valid OHLCV frame
    idx = pd.date_range("2025-01-01", periods=60, freq="T")
    df = pd.DataFrame(
        {
            "open": 1.0,
            "high": 1.1,
            "low": 0.9,
            "close": 1.0,
            "volume": 1000,
        },
        index=idx,
    )

    result: Dict[str, Any] = await orchestrator.analyze(df)

    # Validate shape and key presence
    for key in [
        "fractal_patterns",
        "harmonic_patterns",
        "volume_profile",
        "price_action_dna",
        "pattern_strength",
        "confidence_score",
    ]:
        assert key in result

    assert result["pattern_strength"] == 0.42
    assert result["confidence_score"] == 0.88
    assert isinstance(result["fractal_patterns"], list)
    assert isinstance(result["harmonic_patterns"], list)
    assert isinstance(result["volume_profile"], dict)
    assert isinstance(result["price_action_dna"], dict)


@pytest.mark.asyncio
async def test_orchestrator_fallback_on_small_df() -> None:
    orchestrator = PatternOrchestrator()

    # Insufficient data should trigger engine fallback path
    idx = pd.date_range("2025-01-01", periods=10, freq="T")
    df = pd.DataFrame(
        {
            "open": 1.0,
            "high": 1.0,
            "low": 1.0,
            "close": 1.0,
            "volume": 1,
        },
        index=idx,
    )

    result: Dict[str, Any] = await orchestrator.analyze(df)

    # Expected keys present even in fallback
    for key in [
        "fractal_patterns",
        "harmonic_patterns",
        "volume_profile",
        "price_action_dna",
        "pattern_strength",
        "confidence_score",
    ]:
        assert key in result

    # Fallback strength should be zero or very small
    assert float(result["pattern_strength"]) == 0.0
    # Confidence score is low in fallback per engine contract
    assert float(result["confidence_score"]) <= 0.2
