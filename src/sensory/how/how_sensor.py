from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import pandas as pd

from src.sensory.enhanced.how_dimension import InstitutionalIntelligenceEngine
from src.sensory.how.order_book_analytics import OrderBookAnalytics, OrderBookSnapshot
from src.sensory.lineage import (
    SensorLineageRecord,
    SensorLineageRecorder,
    build_lineage_record,
)
from src.sensory.signals import SensorSignal
from src.sensory.thresholds import ThresholdAssessment, evaluate_thresholds

__all__ = ["HowSensor", "HowSensorConfig"]


@dataclass(slots=True)
class HowSensorConfig:
    """Configuration for the HOW sensor calibration thresholds."""

    minimum_confidence: float = 0.2
    warn_threshold: float = 0.35
    alert_threshold: float = 0.65

    def clamp_confidence(self, confidence: float) -> float:
        return max(self.minimum_confidence, min(1.0, float(confidence)))


class HowSensor:
    """Bridge the institutional HOW engine into the legacy sensory pipeline."""

    def __init__(
        self,
        config: HowSensorConfig | None = None,
        *,
        order_book_analytics: OrderBookAnalytics | None = None,
        engine: InstitutionalIntelligenceEngine | None = None,
        lineage_recorder: SensorLineageRecorder | None = None,
    ) -> None:
        self._engine = engine or InstitutionalIntelligenceEngine()
        self._config = config or HowSensorConfig()
        self._order_book_analytics = order_book_analytics or OrderBookAnalytics()
        self._lineage_recorder = lineage_recorder

    def process(
        self,
        df: pd.DataFrame | None,
        *,
        order_book: pd.DataFrame | None = None,
    ) -> list[SensorSignal]:
        if df is None or df.empty:
            return [self._default_signal(confidence=0.05)]

        payload = self._build_market_payload(df)
        reading_adapter = self._engine.analyze_institutional_intelligence(payload)
        reading = reading_adapter.reading

        signal_strength = float(getattr(reading, "signal_strength", 0.0))
        confidence = self._config.clamp_confidence(getattr(reading, "confidence", 0.0))
        context = dict(getattr(reading, "context", {}) or {})

        telemetry: dict[str, float] = {
            "liquidity": float(reading_adapter.get("liquidity", 0.0)),
            "participation": float(reading_adapter.get("participation", 0.0)),
            "imbalance": float(reading_adapter.get("imbalance", 0.0)),
            "volatility_drag": float(reading_adapter.get("volatility_drag", 0.0)),
        }

        assessment = evaluate_thresholds(
            signal_strength,
            self._config.warn_threshold,
            self._config.alert_threshold,
            mode="absolute",
        )

        order_snapshot: OrderBookSnapshot | None = None
        if order_book is not None:
            order_snapshot = self._order_book_analytics.describe(order_book)
            if order_snapshot is not None:
                telemetry.update(order_snapshot.as_dict())

        audit: dict[str, object] = {
            "signal": signal_strength,
            "confidence": confidence,
            "context": context,
        }
        audit.update({name: metric_value for name, metric_value in telemetry.items()})

        thresholds: dict[str, float] = {
            "warn": self._config.warn_threshold,
            "alert": self._config.alert_threshold,
        }

        lineage = build_lineage_record(
            "HOW",
            "sensory.how",
            inputs=payload,
            outputs={"signal": signal_strength, "confidence": confidence},
            telemetry=telemetry,
            metadata={
                "mode": "market_data",
                "thresholds": thresholds,
                "order_book_sampled": order_snapshot is not None,
                "state": assessment.state,
                "breached_level": assessment.breached_level,
            },
        )
        self._record_lineage(lineage)

        metadata: dict[str, object] = {
            "source": "sensory.how",
            "regime": getattr(getattr(reading, "regime", None), "name", None),
            "thresholds": thresholds,
            "audit": audit,
            "lineage": lineage.as_dict(),
            "state": assessment.state,
            "threshold_assessment": assessment.as_dict(),
        }
        if order_snapshot is not None:
            metadata["order_book"] = order_snapshot.as_dict()

        value: dict[str, object] = {
            "strength": signal_strength,
            "confidence": confidence,
            "context": context,
            "state": assessment.state,
        }
        value.update({name: metric_value for name, metric_value in telemetry.items()})
        return [
            SensorSignal(
                signal_type="HOW",
                value=value,
                confidence=confidence,
                metadata=metadata,
            )
        ]

    def _build_market_payload(self, df: pd.DataFrame) -> Mapping[str, Any]:
        row = df.iloc[-1]
        payload = {
            "timestamp": row.get("timestamp"),
            "symbol": row.get("symbol", "UNKNOWN"),
            "open": float(row.get("open", row.get("close", 0.0) or 0.0)),
            "high": float(row.get("high", row.get("close", 0.0) or 0.0)),
            "low": float(row.get("low", row.get("close", 0.0) or 0.0)),
            "close": float(row.get("close", 0.0) or 0.0),
            "volume": float(row.get("volume", 0.0) or 0.0),
            "volatility": float(row.get("volatility", 0.0) or 0.0),
            "spread": float(row.get("spread", 0.0) or 0.0),
            "depth": float(row.get("depth", 0.0) or 0.0),
            "order_imbalance": float(row.get("order_imbalance", 0.0) or 0.0),
            "data_quality": float(row.get("data_quality", 0.8) or 0.8),
        }
        return payload

    def _default_signal(self, *, confidence: float) -> SensorSignal:
        thresholds: dict[str, float] = {
            "warn": self._config.warn_threshold,
            "alert": self._config.alert_threshold,
        }
        assessment = ThresholdAssessment(
            state="nominal",
            magnitude=0.0,
            thresholds=thresholds,
            breached_level=None,
            breach_ratio=0.0,
            distance_to_warn=thresholds["warn"],
            distance_to_alert=thresholds["alert"],
        )
        lineage = build_lineage_record(
            "HOW",
            "sensory.how",
            inputs={},
            outputs={"signal": 0.0, "confidence": confidence},
            telemetry={},
            metadata={
                "mode": "default",
                "thresholds": thresholds,
                "state": assessment.state,
            },
        )
        self._record_lineage(lineage)

        metadata: dict[str, object] = {
            "source": "sensory.how",
            "thresholds": thresholds,
            "audit": {"signal": 0.0, "confidence": confidence},
            "lineage": lineage.as_dict(),
            "state": assessment.state,
            "threshold_assessment": assessment.as_dict(),
        }
        return SensorSignal(
            signal_type="HOW",
            value={
                "strength": 0.0,
                "confidence": confidence,
                "state": assessment.state,
            },
            confidence=confidence,
            metadata=metadata,
        )

    def _record_lineage(self, lineage: SensorLineageRecord) -> None:
        if self._lineage_recorder is not None:
            self._lineage_recorder.record(lineage)
