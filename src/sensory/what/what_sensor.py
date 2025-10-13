#!/usr/bin/env python3

from __future__ import annotations

import asyncio
import threading
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

    def _compute_trend_strength(self, closes: pd.Series) -> float:
        """Compute directional strength from price deltas in ``closes``."""

        if closes.empty:
            return 0.0

        series = closes.astype(float).dropna()
        if series.size < 3:
            return 0.0

        deltas = series.diff().dropna()
        if deltas.empty:
            return 0.0

        positive = float(deltas[deltas > 0].sum())
        negative = float((-deltas[deltas < 0]).sum())
        total_movement = positive + negative
        if total_movement <= 0.0:
            return 0.0

        directional = (positive - negative) / total_movement
        return max(-1.0, min(1.0, directional))

    def process(self, df: pd.DataFrame | None) -> List[SensorSignal]:
        if df is None or df.empty or "close" not in df:
            return [self._default_signal(reason="insufficient_market_data")]

        window = 20
        recent = df.tail(window)
        high = recent["close"].max()
        low = recent["close"].min()
        last = df["close"].iloc[-1]

        trend_strength = self._compute_trend_strength(recent["close"])

        # Simple breakout as baseline
        base_strength = 0.0
        if last >= high:
            base_strength = 0.6
        elif last <= low:
            base_strength = -0.6

        if base_strength == 0.0:
            base_strength = 0.6 * trend_strength
        else:
            base_strength = 0.7 * base_strength + 0.3 * trend_strength

        # Attempt pattern synthesis (async engine) to compute strength/confidence
        patterns: dict[str, object] = {}
        try:
            patterns = self._run_pattern_orchestrator(df)
        except Exception:
            patterns = {}

        strength = _coerce_float(patterns.get("pattern_strength"), default=base_strength)
        strength = max(-1.0, min(1.0, strength))
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
                "trend_strength": float(trend_strength),
            },
            outputs={
                "pattern_strength": float(strength),
                "confidence": float(confidence),
            },
            telemetry={
                "pattern_strength": float(strength),
                "confidence": float(confidence),
                "trend_strength": float(trend_strength),
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
            "trend_strength": trend_strength,
            "pattern_payload": details,
            "quality": quality,
            "lineage": lineage.as_dict(),
        }
        value: dict[str, object] = {
            "pattern_strength": strength,
            "confidence": confidence,
            "last_close": last,
            "trend_strength": trend_strength,
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

    def _default_signal(self, *, reason: str) -> SensorSignal:
        timestamp = datetime.now(timezone.utc)
        confidence = 0.1
        lineage = build_lineage_record(
            "WHAT",
            "sensory.what",
            inputs={},
            outputs={"pattern_strength": 0.0, "confidence": confidence},
            telemetry={},
            metadata={
                "timestamp": timestamp.isoformat(),
                "mode": "default",
                "reason": reason,
            },
        )
        quality = {
            "source": "sensory.what",
            "timestamp": timestamp.isoformat(),
            "confidence": confidence,
            "strength": 0.0,
            "reason": reason,
        }
        metadata: dict[str, object] = {
            "source": "sensory.what",
            "reason": reason,
            "trend_strength": 0.0,
            "quality": quality,
            "lineage": lineage.as_dict(),
        }
        return SensorSignal(
            signal_type="WHAT",
            value={
                "pattern_strength": 0.0,
                "trend_strength": 0.0,
                "confidence": confidence,
            },
            confidence=confidence,
            metadata=metadata,
            lineage=lineage,
        )

    def _run_pattern_orchestrator(self, df: pd.DataFrame) -> dict[str, object]:
        if df.empty:
            return {}

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            result = asyncio.run(self._orch.analyze(df))
            return _coerce_mapping(result)

        return self._run_coroutine_in_worker_thread(df)

    def _run_coroutine_in_worker_thread(self, df: pd.DataFrame) -> dict[str, object]:
        result_holder: dict[str, object] = {}
        error_holder: dict[str, BaseException] = {}

        def _runner() -> None:
            try:
                outcome = asyncio.run(self._orch.analyze(df))
            except BaseException as exc:  # pragma: no cover - defensive
                error_holder["error"] = exc
                return
            result_holder.update(_coerce_mapping(outcome))

        thread = threading.Thread(
            target=_runner,
            name="what-sensor-pattern",
            daemon=True,
        )
        thread.start()
        thread.join()

        if error_holder:
            raise error_holder["error"]
        return result_holder

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
