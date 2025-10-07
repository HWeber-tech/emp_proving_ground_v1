"""Market manipulation detection with lineage telemetry for the ANOMALY organ."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

import pandas as pd

from src.sensory.lineage import SensorLineageRecord, SensorLineageRecorder, build_lineage_record
from src.sensory.organs.analyzers.anomaly_organ import ManipulationDetector

__all__ = ["ManipulationDetectionSystem", "ManipulationDetectionResult"]


def _extract_dataframe(payload: Any) -> pd.DataFrame | None:
    if isinstance(payload, pd.DataFrame):
        return payload
    if isinstance(payload, Mapping):
        for key in ("price_data", "market_data", "data", "df"):
            candidate = payload.get(key)
            if isinstance(candidate, pd.DataFrame):
                return candidate
    return None


def _pattern_confidence(pattern: Mapping[str, Any]) -> float:
    try:
        return float(pattern.get("confidence", 0.0))
    except Exception:  # pragma: no cover - defensive
        return 0.0


def _pattern_subtype(pattern: Mapping[str, Any]) -> str:
    subtype = pattern.get("subtype")
    if isinstance(subtype, str):
        return subtype
    return "unknown"


def _normalise_pattern(pattern: Mapping[str, Any]) -> dict[str, Any]:
    cleaned: dict[str, Any] = {}
    for key, value in pattern.items():
        try:
            cleaned[str(key)] = value
        except Exception:  # pragma: no cover - defensive
            continue
    return cleaned


def _volatility_factor(df: pd.DataFrame) -> float:
    if "close" not in df:
        return 0.0
    returns = df["close"].pct_change().dropna()
    if returns.empty:
        return 0.0
    vol = float(returns.std())
    return min(1.0, max(0.0, vol * 12.0))


def _volume_factor(df: pd.DataFrame) -> float:
    if "volume" not in df or df["volume"].empty:
        return 0.0
    volume = df["volume"].astype(float)
    baseline = float(volume.rolling(window=min(len(volume), 20)).median().iloc[-1])
    latest = float(volume.iloc[-1])
    spread = float(volume.rolling(window=min(len(volume), 20)).std().fillna(0.0).iloc[-1])
    if spread == 0.0:
        spread = max(1.0, baseline * 0.1)
    z_score = abs(latest - baseline) / spread
    return min(1.0, max(0.0, z_score / 6.0))


@dataclass(slots=True, frozen=True)
class ManipulationDetectionResult:
    """Structured result for manipulation detection suitable for orchestration."""

    confidence: float
    overall_risk_score: float
    status: str
    patterns: tuple[dict[str, Any], ...]
    summary: Mapping[str, Any]
    lineage: SensorLineageRecord
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        payload = {
            "confidence": self.confidence,
            "overall_risk_score": self.overall_risk_score,
            "status": self.status,
            "patterns": [dict(pattern) for pattern in self.patterns],
            "summary": dict(self.summary),
            "lineage": self.lineage.as_dict(),
        }
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload


class ManipulationDetectionSystem:
    """Detect market manipulation and emit lineage-aware telemetry."""

    WARN_THRESHOLD = 0.4
    ALERT_THRESHOLD = 0.7

    def __init__(
        self,
        *,
        detector: ManipulationDetector | None = None,
        lineage_recorder: SensorLineageRecorder | None = None,
        history_limit: int = 128,
    ) -> None:
        self._detector = detector or ManipulationDetector()
        self._lineage_recorder = lineage_recorder
        self._history_limit = max(1, history_limit)
        self._history: list[ManipulationDetectionResult] = []

    async def detect_manipulation(self, data: Any) -> ManipulationDetectionResult:
        df = _extract_dataframe(data)
        if df is None or df.empty:
            result = self._build_result(
                patterns=tuple(),
                confidence=0.2,
                risk_score=0.0,
                status="nominal",
                summary={"rows": 0, "pattern_count": 0},
                metadata={"reason": "no_data"},
            )
            self._record(result)
            return result

        try:
            patterns_raw = self._detector.detect_manipulation_patterns(df)
        except Exception:
            result = self._build_result(
                patterns=tuple(),
                confidence=0.2,
                risk_score=0.0,
                status="nominal",
                summary={"rows": len(df), "pattern_count": 0, "failure": True},
                metadata={"reason": "detector_failure"},
            )
            self._record(result)
            return result

        patterns = tuple(_normalise_pattern(pattern) for pattern in patterns_raw)
        confidence = self._calculate_confidence(patterns)
        risk_score = self._calculate_risk_score(patterns, df)
        status = self._determine_status(risk_score, confidence)
        summary = self._build_summary(patterns, df, risk_score, confidence)
        metadata = {"rows": len(df)}

        result = self._build_result(
            patterns=patterns,
            confidence=confidence,
            risk_score=risk_score,
            status=status,
            summary=summary,
            metadata=metadata,
        )
        self._record(result)
        return result

    def latest_result(self) -> ManipulationDetectionResult | None:
        if not self._history:
            return None
        return self._history[-1]

    def history(self, limit: int | None = None) -> list[ManipulationDetectionResult]:
        if limit is None or limit <= 0:
            return list(self._history)
        return list(self._history[-limit:])

    def _calculate_confidence(self, patterns: Iterable[Mapping[str, Any]]) -> float:
        pattern_list = list(patterns)
        confidences = [_pattern_confidence(pattern) for pattern in pattern_list]
        if not confidences:
            return 0.2
        top = max(confidences)
        weighted = self._weighted_confidence(pattern_list)
        combined = 0.6 * top + 0.4 * weighted
        return max(0.2, min(1.0, combined))

    def _calculate_risk_score(
        self,
        patterns: Iterable[Mapping[str, Any]],
        df: pd.DataFrame,
    ) -> float:
        pattern_list = list(patterns)
        weighted = self._weighted_confidence(pattern_list)
        volatility = _volatility_factor(df)
        volume_factor = _volume_factor(df)
        pattern_intensity = min(1.0, len(pattern_list) * 0.08)
        risk = weighted + 0.35 * volatility + 0.2 * volume_factor + pattern_intensity
        return float(min(1.0, max(0.0, risk)))

    def _weighted_confidence(self, patterns: Iterable[Mapping[str, Any]]) -> float:
        total_weight = 0.0
        weighted_sum = 0.0
        for pattern in patterns:
            confidence = _pattern_confidence(pattern)
            subtype = _pattern_subtype(pattern)
            weight = 1.0
            if subtype == "pump_and_dump":
                weight = 1.5
            elif subtype == "spoofing":
                weight = 1.3
            elif subtype == "layering":
                weight = 1.1
            weighted_sum += confidence * weight
            total_weight += weight
        if total_weight == 0.0:
            return 0.0
        return min(1.0, weighted_sum / total_weight)

    def _determine_status(self, risk_score: float, confidence: float) -> str:
        if risk_score >= self.ALERT_THRESHOLD or confidence >= 0.85:
            return "alert"
        if risk_score >= self.WARN_THRESHOLD or confidence >= 0.6:
            return "warn"
        return "nominal"

    def _build_summary(
        self,
        patterns: Iterable[Mapping[str, Any]],
        df: pd.DataFrame,
        risk_score: float,
        confidence: float,
    ) -> dict[str, Any]:
        patterns_list = list(patterns)
        types = Counter(_pattern_subtype(pattern) for pattern in patterns_list)
        volatility = _volatility_factor(df)
        volume_factor = _volume_factor(df)
        summary = {
            "pattern_count": len(patterns_list),
            "pattern_types": dict(types),
            "volatility_factor": volatility,
            "volume_factor": volume_factor,
            "risk_score": risk_score,
            "confidence": confidence,
        }
        if patterns_list:
            summary["max_pattern_confidence"] = max(_pattern_confidence(p) for p in patterns_list)
        return summary

    def _trim_patterns_for_metadata(self, patterns: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
        trimmed: list[dict[str, Any]] = []
        for pattern in list(patterns)[:3]:
            payload = {
                "type": pattern.get("type"),
                "subtype": pattern.get("subtype"),
                "confidence": _pattern_confidence(pattern),
            }
            if "description" in pattern:
                payload["description"] = pattern["description"]
            trimmed.append(payload)
        return trimmed

    def _build_result(
        self,
        *,
        patterns: tuple[dict[str, Any], ...],
        confidence: float,
        risk_score: float,
        status: str,
        summary: Mapping[str, Any],
        metadata: Mapping[str, Any],
    ) -> ManipulationDetectionResult:
        lineage_inputs = {
            "rows": metadata.get("rows", 0),
            "pattern_types": dict(summary.get("pattern_types", {})),
        }
        lineage_outputs = {
            "pattern_count": summary.get("pattern_count", 0),
            "risk_score": risk_score,
            "confidence": confidence,
            "status": status,
        }
        lineage = build_lineage_record(
            "ANOMALY",
            "sensory.anomaly.manipulation",
            inputs=lineage_inputs,
            outputs=lineage_outputs,
            telemetry={
                "volatility": summary.get("volatility_factor", 0.0),
                "volume_factor": summary.get("volume_factor", 0.0),
                "weighted_confidence": self._weighted_confidence(patterns),
            },
            metadata={
                "status_thresholds": {
                    "warn": self.WARN_THRESHOLD,
                    "alert": self.ALERT_THRESHOLD,
                },
                "pattern_examples": self._trim_patterns_for_metadata(patterns),
            },
        )

        return ManipulationDetectionResult(
            confidence=float(confidence),
            overall_risk_score=float(risk_score),
            status=status,
            patterns=patterns,
            summary=dict(summary),
            lineage=lineage,
            metadata=dict(metadata),
        )

    def _record(self, result: ManipulationDetectionResult) -> None:
        self._history.append(result)
        if len(self._history) > self._history_limit:
            self._history = self._history[-self._history_limit :]
        if self._lineage_recorder is not None:
            try:
                self._lineage_recorder.record(result.lineage)
            except Exception:  # pragma: no cover - defensive
                pass
