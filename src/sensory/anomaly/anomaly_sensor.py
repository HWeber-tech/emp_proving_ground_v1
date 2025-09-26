from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import pandas as pd

from src.sensory.enhanced.anomaly_dimension import AnomalyIntelligenceEngine
from src.sensory.signals import SensorSignal

__all__ = ["AnomalySensor", "AnomalySensorConfig"]


@dataclass(slots=True)
class AnomalySensorConfig:
    """Configuration for anomaly detection thresholds."""

    window: int = 32
    warn_threshold: float = 0.4
    alert_threshold: float = 0.7


class AnomalySensor:
    """Detect displacements in price/volume behaviour for the ANOMALY dimension."""

    def __init__(self, config: AnomalySensorConfig | None = None) -> None:
        self._engine = AnomalyIntelligenceEngine()
        self._config = config or AnomalySensorConfig()

    def process(
        self, data: pd.DataFrame | Mapping[str, Any] | Sequence[float] | None
    ) -> list[SensorSignal]:
        if data is None:
            return [self._default_signal()]

        if isinstance(data, pd.DataFrame):
            return self._process_frame(data)

        reading = self._engine.analyze_anomaly_intelligence(data)
        return [self._build_signal(reading, mode_override=None)]

    # ------------------------------------------------------------------
    def _process_frame(self, df: pd.DataFrame) -> list[SensorSignal]:
        if df.empty or "close" not in df:
            return [self._default_signal()]

        sequence = df["close"].astype(float).tail(self._config.window).tolist()
        if len(sequence) >= 8:
            reading = self._engine.analyze_anomaly_intelligence(sequence)
            return [self._build_signal(reading, mode_override="sequence")]

        payload = self._build_market_payload(df)
        reading = self._engine.analyze_anomaly_intelligence(payload)
        return [self._build_signal(reading, mode_override="market_data")]

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
        }
        return payload

    def _build_signal(self, reading_adapter, *, mode_override: str | None) -> SensorSignal:
        reading = reading_adapter.reading
        signal_strength = float(getattr(reading, "signal_strength", 0.0))
        confidence = float(getattr(reading, "confidence", 0.0))
        context = dict(getattr(reading, "context", {}) or {})
        mode = mode_override or reading_adapter.get("mode", "sequence")

        metadata: dict[str, object] = {
            "source": "sensory.anomaly",
            "mode": mode,
            "thresholds": {
                "warn": self._config.warn_threshold,
                "alert": self._config.alert_threshold,
            },
            "audit": {
                "signal": signal_strength,
                "confidence": confidence,
                "context": context,
                "baseline": float(reading_adapter.get("baseline", 0.0)),
                "dispersion": float(reading_adapter.get("dispersion", 0.0)),
                "latest": float(reading_adapter.get("latest", 0.0)),
            },
        }

        value: dict[str, object] = {
            "strength": signal_strength,
            "confidence": confidence,
            "context": context,
        }
        return SensorSignal(
            signal_type="ANOMALY",
            value=value,
            confidence=confidence,
            metadata=metadata,
        )

    def _default_signal(self) -> SensorSignal:
        metadata: dict[str, object] = {
            "source": "sensory.anomaly",
            "thresholds": {
                "warn": self._config.warn_threshold,
                "alert": self._config.alert_threshold,
            },
            "mode": "unknown",
            "audit": {"signal": 0.0, "confidence": 0.0},
        }
        return SensorSignal(
            signal_type="ANOMALY",
            value={"strength": 0.0, "confidence": 0.0},
            confidence=0.0,
            metadata=metadata,
        )
