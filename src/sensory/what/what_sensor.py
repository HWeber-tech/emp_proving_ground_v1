#!/usr/bin/env python3

from __future__ import annotations

import asyncio
from typing import Any, List

import pandas as pd

from sensory.signals import SensorSignal
from sensory.what.patterns.orchestrator import PatternOrchestrator


class WhatSensor:
    """Pattern sensor (WHAT dimension)."""

    def __init__(self) -> None:
        self._orch = PatternOrchestrator()

    def process(self, df: pd.DataFrame) -> List[SensorSignal]:
        if df is None or df.empty or "close" not in df:
            return [
                SensorSignal(signal_type="WHAT", value={"pattern_strength": 0.0}, confidence=0.1)
            ]

        window = 20
        recent = df.tail(window)
        high = recent["close"].max()
        low = recent["close"].min()
        last = df["close"].iloc[-1]

        # Simple breakout as baseline
        base_strength = 0.0
        if last >= high:
            base_strength = 0.6
        elif last <= low:
            base_strength = -0.6

        # Attempt pattern synthesis (async engine) to compute strength/confidence
        patterns: dict[str, Any] = {}
        try:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop and loop.is_running():
                # In an async context, skip orchestration to avoid nested loops.
                patterns = {}
            else:
                patterns = asyncio.run(self._orch.analyze(df))
        except Exception:
            patterns = {}

        strength = float(patterns.get("pattern_strength", base_strength))
        confidence = float(patterns.get("confidence_score", 0.5))
        value = {"pattern_strength": strength, "pattern_details": patterns or {}}
        return [SensorSignal(signal_type="WHAT", value=value, confidence=confidence)]
