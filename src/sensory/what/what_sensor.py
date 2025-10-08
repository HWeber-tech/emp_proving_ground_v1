#!/usr/bin/env python3

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from collections.abc import Mapping, Sequence
from typing import Any, List

import pandas as pd

from src.sensory.lineage import build_lineage_record
from src.sensory.signals import SensorSignal
from src.sensory.what.patterns.orchestrator import PatternOrchestrator


def _coerce_float(value: object, *, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return default
    return default


def _coerce_mapping(payload: object) -> dict[str, object]:
    if isinstance(payload, Mapping):
        return {str(key): value for key, value in payload.items()}
    return {}


def _coerce_sequence_of_mappings(payload: object) -> list[dict[str, object]]:
    if not isinstance(payload, Sequence) or isinstance(payload, (bytes, bytearray, str)):
        return []
    results: list[dict[str, object]] = []
    for item in payload:
        mapping = _coerce_mapping(item)
        if mapping:
            results.append(mapping)
    return results


def _normalise_pattern_payload(payload: Mapping[str, object] | None) -> dict[str, object]:
    if payload is None:
        return {}

    pattern_payload: dict[str, object] = {}

    fractals = _coerce_sequence_of_mappings(payload.get("fractal_patterns"))
    if fractals:
        pattern_payload["fractal_patterns"] = fractals

    harmonics = _coerce_sequence_of_mappings(payload.get("harmonic_patterns"))
    if harmonics:
        pattern_payload["harmonic_patterns"] = harmonics

    volume_profile = _coerce_mapping(payload.get("volume_profile"))
    if volume_profile:
        pattern_payload["volume_profile"] = volume_profile

    price_action = _coerce_mapping(payload.get("price_action_dna"))
    if price_action:
        pattern_payload["price_action_dna"] = price_action

    for key, value in payload.items():
        if key in pattern_payload:
            continue
        if key in {"pattern_strength", "confidence_score"}:
            continue
        pattern_payload[str(key)] = value

    return pattern_payload


class WhatSensor:
    """Pattern sensor (WHAT dimension)."""

    def __init__(self) -> None:
        self._orch = PatternOrchestrator()

    def process(self, df: pd.DataFrame | None) -> List[SensorSignal]:
        if df is None or df.empty or "close" not in df:
            metadata: dict[str, object] = {
                "source": "sensory.what",
                "reason": "insufficient_market_data",
            }
            return [
                SensorSignal(
                    signal_type="WHAT",
                    value={"pattern_strength": 0.0},
                    confidence=0.1,
                    metadata=metadata,
                )
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
        patterns: dict[str, object] = {}
        try:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop and loop.is_running():
                # In an async context, skip orchestration to avoid nested loops.
                patterns = {}
            else:
                orchestrator_output = asyncio.run(self._orch.analyze(df))
                patterns = _coerce_mapping(orchestrator_output)
        except Exception:
            patterns = {}

        strength = _coerce_float(patterns.get("pattern_strength"), default=base_strength)
        confidence = _coerce_float(patterns.get("confidence_score"), default=0.5)
        details = _normalise_pattern_payload(patterns)

        timestamp = self._resolve_timestamp(df)
        quality = {
            "source": "sensory.what",
            "timestamp": timestamp.isoformat(),
            "confidence": float(confidence),
            "strength": float(strength),
        }
        data_quality = self._extract_data_quality(df)
        if data_quality is not None:
            quality["data_quality"] = data_quality

        lineage = build_lineage_record(
            "WHAT",
            "sensory.what",
            inputs={
                "window": window,
                "high": float(high),
                "low": float(low),
                "last_close": float(last),
                "base_strength": float(base_strength),
            },
            outputs={
                "pattern_strength": float(strength),
                "confidence": float(confidence),
            },
            telemetry={
                "pattern_strength": float(strength),
                "confidence": float(confidence),
            },
            metadata={
                "timestamp": timestamp.isoformat(),
                "mode": "pattern_analysis",
                "pattern_details": details,
            },
        )

        signal_metadata: dict[str, object] = {
            "source": "sensory.what",
            "window": window,
            "high": high,
            "low": low,
            "last_close": last,
            "base_strength": base_strength,
            "pattern_payload": details,
            "quality": quality,
            "lineage": lineage.as_dict(),
        }
        value: dict[str, object] = {
            "pattern_strength": strength,
            "confidence": confidence,
            "last_close": last,
        }
        if details:
            value["pattern_details"] = details

        return [
            SensorSignal(
                signal_type="WHAT",
                value=value,
                confidence=confidence,
                metadata=signal_metadata,
                lineage=lineage,
            )
        ]

    def _resolve_timestamp(self, df: pd.DataFrame) -> datetime:
        if not df.empty and "timestamp" in df:
            ts = pd.to_datetime(df["timestamp"].iloc[-1], utc=True, errors="coerce")
            if ts is not None and not pd.isna(ts):
                if ts.tzinfo is None:
                    ts = ts.tz_localize(timezone.utc)
                return ts.to_pydatetime()
        return datetime.now(timezone.utc)

    def _extract_data_quality(self, df: pd.DataFrame) -> float | None:
        if "data_quality" not in df or df["data_quality"].empty:
            return None
        try:
            return float(df["data_quality"].iloc[-1])
        except (TypeError, ValueError):
            return None
