from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import pandas as pd

from src.sensory.enhanced.how_dimension import InstitutionalIntelligenceEngine
from src.sensory.how.order_book_imbalance import (
    OrderBookImbalanceMetrics,
    compute_order_book_imbalance,
)
from src.sensory.signals import SensorSignal

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

    def __init__(self, config: HowSensorConfig | None = None) -> None:
        self._engine = InstitutionalIntelligenceEngine()
        self._config = config or HowSensorConfig()

    def process(self, df: pd.DataFrame | None) -> list[SensorSignal]:
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

        order_book_metrics = payload.get("_order_book_metrics")
        if isinstance(order_book_metrics, OrderBookImbalanceMetrics):
            telemetry.update(
                {
                    "book_buy_volume": float(order_book_metrics.buy_volume),
                    "book_sell_volume": float(order_book_metrics.sell_volume),
                    "book_total_volume": float(order_book_metrics.total_volume),
                    "book_levels": float(order_book_metrics.levels_evaluated),
                }
            )

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

        metadata: dict[str, object] = {
            "source": "sensory.how",
            "regime": getattr(getattr(reading, "regime", None), "name", None),
            "thresholds": thresholds,
            "audit": audit,
        }

        value: dict[str, object] = {
            "strength": signal_strength,
            "confidence": confidence,
            "context": context,
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
        book_metrics = self._compute_order_book_metrics(row)
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
        if book_metrics is not None:
            existing_imbalance = row.get("order_imbalance")
            if existing_imbalance is None or pd.isna(existing_imbalance):
                payload["order_imbalance"] = float(book_metrics.imbalance)
            payload["_order_book_metrics"] = book_metrics
        return payload

    def _compute_order_book_metrics(
        self, row: pd.Series
    ) -> OrderBookImbalanceMetrics | None:
        order_book = row.get("order_book_snapshot") or row.get("order_book")
        if order_book is None:
            return None

        metrics = compute_order_book_imbalance(order_book, depth=5)
        return metrics if metrics.has_volume else None

    def _default_signal(self, *, confidence: float) -> SensorSignal:
        thresholds: dict[str, float] = {
            "warn": self._config.warn_threshold,
            "alert": self._config.alert_threshold,
        }
        metadata: dict[str, object] = {
            "source": "sensory.how",
            "thresholds": thresholds,
            "audit": {"signal": 0.0, "confidence": confidence},
        }
        return SensorSignal(
            signal_type="HOW",
            value={"strength": 0.0, "confidence": confidence},
            confidence=confidence,
            metadata=metadata,
        )
