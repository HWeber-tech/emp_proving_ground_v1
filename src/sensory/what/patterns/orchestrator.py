#!/usr/bin/env python3
"""
Pattern Orchestrator for WHAT dimension.

Thin wrapper around PatternSynthesisEngine that exposes a stable, minimal API
for higher layers. This isolates callers from internal reorganization of the
pattern engine and supports future decomposition of detectors.
"""

from __future__ import annotations

from typing import Any, Dict

import pandas as pd

from src.sensory.organs.dimensions.pattern_engine import PatternSynthesisEngine


class PatternOrchestrator:
    """Coordinates pattern synthesis for the WHAT dimension."""

    def __init__(self) -> None:
        self.engine = PatternSynthesisEngine()

    async def analyze(self, market_data: pd.DataFrame) -> dict[str, Any]:
        """
        Run the full pattern synthesis pipeline and return a dict payload.

        Returns a structure with keys:
          - fractal_patterns: list[dict]
          - harmonic_patterns: list[dict]
          - volume_profile: dict
          - price_action_dna: dict
          - pattern_strength: float
          - confidence_score: float
        """
        result = await self.engine.synthesize_patterns(market_data)
        return {
            "fractal_patterns": result.fractal_patterns,
            "harmonic_patterns": result.harmonic_patterns,
            "volume_profile": result.volume_profile,
            "price_action_dna": result.price_action_dna,
            "pattern_strength": result.pattern_strength,
            "confidence_score": result.confidence_score,
        }
