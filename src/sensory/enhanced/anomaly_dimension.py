from __future__ import annotations

from math import tanh
from datetime import datetime
from statistics import fmean, pstdev
from typing import Any, Iterable, Mapping, Sequence

from src.core.base import DimensionalReading, MarketRegime
from src.sensory.enhanced._shared import (
    ReadingAdapter,
    build_legacy_payload,
    clamp,
    ensure_market_data,
    safe_timestamp,
)

__all__ = ["AnomalyUnderstandingEngine", "AnomalyIntelligenceEngine"]


class AnomalyUnderstandingEngine:
    def analyze_anomaly_understanding(
        self, data: Mapping[str, Any] | Sequence[float] | Iterable[float] | Any | None = None
    ) -> ReadingAdapter:
        """Detect displacement from typical behaviour across recent samples."""

        if isinstance(data, (Sequence, Iterable)) and not isinstance(data, (Mapping, str, bytes)):
            values = [float(x) for x in data]
            baseline = fmean(values) if values else 0.0
            dispersion = pstdev(values) if len(values) > 1 else 0.0
            latest = values[-1] if values else 0.0
            if dispersion <= 0.0:
                z_score = 0.0
            else:
                z_score = (latest - baseline) / dispersion
            abs_z = abs(z_score)
            # Scale the z-score into [0, 1] so the threshold evaluator can map warn/alert states.
            signal_strength = clamp(abs_z / 4.0, 0.0, 1.0)
            confidence = clamp(0.2 + 0.6 * min(1.0, len(values) / 25.0), 0.0, 1.0)
            context = {
                "source": "sensory.anomaly",
                "mode": "sequence",
                "baseline": baseline,
                "dispersion": dispersion,
                "latest": latest,
                "z_score": z_score,
            }
            timestamp = datetime.utcnow()
            reading = DimensionalReading(
                dimension="ANOMALY",
                signal_strength=float(signal_strength),
                confidence=float(confidence),
                regime=MarketRegime.UNKNOWN,
                context=context,
                data_quality=1.0,
                processing_time_ms=0.0,
                timestamp=timestamp,
            )
            extras = {
                "baseline": float(baseline),
                "dispersion": float(dispersion),
                "latest": float(latest),
                "z_score": float(z_score),
            }
            return build_legacy_payload(reading, source="sensory.anomaly", extras=extras)

        market_data = ensure_market_data(data)
        price_base = abs(float(getattr(market_data, "open", 0.0))) or 1.0
        move = abs(
            float(getattr(market_data, "close", price_base))
            - float(getattr(market_data, "open", price_base))
        )
        volume = float(getattr(market_data, "volume", 0.0))
        volatility = float(getattr(market_data, "volatility", 0.0))
        spread = float(getattr(market_data, "spread", 0.0))

        move_ratio = move / price_base
        volume_pressure = tanh(volume / max(1.0, price_base * 900.0))
        volatility_pressure = tanh(max(0.0, volatility) * 8.0)
        spread_penalty = tanh(max(0.0, spread) * 10000.0)

        raw_anomaly = (
            0.5 * tanh(move_ratio * 12.0)
            + 0.3 * volume_pressure
            + 0.2 * volatility_pressure
            + 0.15 * spread_penalty
        )
        signal_strength = clamp(raw_anomaly, 0.0, 1.0)

        confidence = clamp(
            0.25
            + 0.35 * (1.0 - spread_penalty)
            + 0.25 * min(1.0, volume_pressure + 0.1)
            + 0.15 * min(1.0, volatility_pressure + 0.1),
            0.0,
            1.0,
        )

        synthetic_z = raw_anomaly * 4.0
        context = {
            "source": "sensory.anomaly",
            "mode": "market_data",
            "move_ratio": float(move_ratio),
            "volume_pressure": float(volume_pressure),
            "volatility_pressure": float(volatility_pressure),
            "spread_penalty": float(spread_penalty),
            "z_score": float(synthetic_z),
        }

        reading = DimensionalReading(
            dimension="ANOMALY",
            signal_strength=float(signal_strength),
            confidence=float(confidence),
            regime=MarketRegime.UNKNOWN,
            context=context,
            data_quality=1.0,
            processing_time_ms=0.0,
            timestamp=safe_timestamp(market_data),
        )
        extras = {
            "move_ratio": float(move_ratio),
            "volume_pressure": float(volume_pressure),
            "volatility_pressure": float(volatility_pressure),
            "spread_penalty": float(spread_penalty),
            "z_score": float(synthetic_z),
        }
        return build_legacy_payload(reading, source="sensory.anomaly", extras=extras)

    def analyze_anomaly_intelligence(
        self, data: Mapping[str, Any] | Sequence[float] | Iterable[float] | Any | None = None
    ) -> ReadingAdapter:
        """Legacy alias maintained for backwards compatibility."""

        return self.analyze_anomaly_understanding(data)


AnomalyIntelligenceEngine = AnomalyUnderstandingEngine
