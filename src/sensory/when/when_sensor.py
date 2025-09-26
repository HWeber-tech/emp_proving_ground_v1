#!/usr/bin/env python3

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from math import exp
from typing import Iterable, Sequence

import pandas as pd

from src.sensory.signals import SensorSignal
from src.sensory.when.gamma_exposure import (
    GammaExposureAnalyzer,
    GammaExposureAnalyzerConfig,
    GammaExposureDataset,
    GammaExposureSummary,
)

__all__ = ["WhenSensor", "WhenSensorConfig"]


@dataclass(slots=True)
class WhenSensorConfig:
    """Configuration for the WHEN sensor scoring function."""

    minimum_confidence: float = 0.2
    session_weight: float = 0.4
    news_weight: float = 0.3
    gamma_weight: float = 0.3
    news_decay_minutes: int = 120
    gamma_near_fraction: float = 0.01
    gamma_pressure_normalizer: float = 5.0e4

    def _normalised_weights(self) -> tuple[float, float, float]:
        total = self.session_weight + self.news_weight + self.gamma_weight
        if total <= 0:
            return (1 / 3, 1 / 3, 1 / 3)
        return (
            self.session_weight / total,
            self.news_weight / total,
            self.gamma_weight / total,
        )


class WhenSensor:
    """Temporal/context sensor (WHEN dimension).

    Combines session intensity, macro event proximity, and option gamma posture to
    reflect the temporal edge of acting right now versus waiting for better
    conditions.
    """

    def __init__(
        self,
        config: WhenSensorConfig | None = None,
        *,
        gamma_dataset: GammaExposureDataset | None = None,
        gamma_analyzer: GammaExposureAnalyzer | None = None,
    ) -> None:
        self._config = config or WhenSensorConfig()
        analyzer_config = GammaExposureAnalyzerConfig(
            near_fraction=self._config.gamma_near_fraction,
            pressure_normalizer=self._config.gamma_pressure_normalizer,
        )
        self._gamma_analyzer = gamma_analyzer or GammaExposureAnalyzer(analyzer_config)
        self._gamma_dataset = gamma_dataset

    def process(
        self,
        df: pd.DataFrame | None,
        *,
        option_positions: pd.DataFrame | None = None,
        macro_events: Sequence[datetime] | None = None,
    ) -> list[SensorSignal]:
        if df is None or df.empty or "close" not in df or "timestamp" not in df:
            return [self._default_signal(confidence=0.05)]

        row = df.iloc[-1]
        timestamp = pd.to_datetime(row.get("timestamp"), utc=True, errors="coerce")
        if timestamp is None or pd.isna(timestamp):
            return [self._default_signal(confidence=0.05)]
        if timestamp.tzinfo is None:
            timestamp = timestamp.tz_localize(timezone.utc)

        symbol = str(row.get("symbol", "UNKNOWN"))
        spot_price = float(row.get("close", 0.0) or 0.0)

        session_intensity = self._calculate_session_intensity(timestamp)
        news_proximity = self._calculate_news_proximity(timestamp, macro_events or [])
        gamma_summary = self._summarise_gamma(
            symbol=symbol,
            as_of=timestamp,
            spot_price=spot_price,
            option_positions=option_positions,
        )
        gamma_impact = gamma_summary.impact_score
        primary_strike = gamma_summary.primary_strike
        dominant_profiles: list[dict[str, object]] = []
        for profile in gamma_summary.dominant_strikes:
            dominant_profiles.append(
                {
                    "strike": float(profile.strike),
                    "net_gamma": float(profile.net_gamma),
                    "abs_gamma": float(profile.abs_gamma),
                    "share_of_total": float(profile.share_of_total),
                    "distance": float(profile.distance),
                    "side": profile.side,
                }
            )

        w_session, w_news, w_gamma = self._config._normalised_weights()
        strength = w_session * session_intensity + w_news * news_proximity + w_gamma * gamma_impact

        confidence_components = [session_intensity, news_proximity, gamma_summary.gamma_pressure]
        confidence = max(
            self._config.minimum_confidence,
            min(1.0, sum(confidence_components) / (len(confidence_components) + 0.5)),
        )

        components: dict[str, object] = {
            "session_intensity": float(session_intensity),
            "news_proximity": float(news_proximity),
            "gamma_impact": float(gamma_impact),
            "gamma_pin_risk": float(gamma_summary.pin_risk_score),
            "gamma_pressure": float(gamma_summary.gamma_pressure),
            "gamma_flip_risk": bool(gamma_summary.flip_risk),
            "gamma_pin_strike": float(primary_strike.strike) if primary_strike else None,
        }
        gamma_snapshot: dict[str, object] = {
            "as_of": gamma_summary.as_of.isoformat(),
            "symbol": gamma_summary.symbol,
            "net_gamma": float(gamma_summary.net_gamma),
            "total_abs_gamma": float(gamma_summary.total_abs_gamma),
            "near_gamma": float(gamma_summary.near_gamma),
        }
        metadata: dict[str, object] = {
            "source": "sensory.when",
            "components": components,
            "gamma_snapshot": gamma_snapshot,
            "gamma_dominant_strikes": dominant_profiles,
        }

        value = {
            "strength": strength,
            "confidence": confidence,
            "session": session_intensity,
            "news": news_proximity,
            "gamma": gamma_impact,
        }

        return [
            SensorSignal(
                signal_type="WHEN",
                value=value,
                confidence=confidence,
                metadata=metadata,
            )
        ]

    def _calculate_session_intensity(self, timestamp: pd.Timestamp) -> float:
        hour = timestamp.astimezone(timezone.utc).hour
        # Rough session overlap windows (UTC)
        london = 7 <= hour < 16
        new_york = 12 <= hour < 21
        asia = hour >= 22 or hour < 7

        overlap_weight = 0.0
        if london and new_york:
            overlap_weight = 1.0
        elif london or new_york:
            overlap_weight = 0.7
        elif asia:
            overlap_weight = 0.4

        return float(overlap_weight)

    def _calculate_news_proximity(
        self, timestamp: pd.Timestamp, events: Sequence[datetime]
    ) -> float:
        if not events:
            return 0.1

        min_minutes: float | None = None
        ts = timestamp.astimezone(timezone.utc)
        for event in events:
            event_ts = event.astimezone(timezone.utc)
            minutes = (event_ts - ts).total_seconds() / 60.0
            if minutes < 0:
                continue
            if min_minutes is None or minutes < min_minutes:
                min_minutes = minutes

        if min_minutes is None:
            return 0.1

        decay = max(1.0, float(self._config.news_decay_minutes))
        score = exp(-min_minutes / decay)
        return float(max(0.0, min(1.0, score)))

    def _summarise_gamma(
        self,
        *,
        symbol: str,
        as_of: datetime,
        spot_price: float,
        option_positions: pd.DataFrame | None,
    ) -> GammaExposureSummary:
        if option_positions is not None:
            return self._gamma_analyzer.summarise(
                option_positions,
                spot_price=spot_price,
                as_of=as_of,
                symbol=symbol,
            )
        if self._gamma_dataset is not None:
            return self._gamma_dataset.summarise(
                symbol=symbol,
                as_of=as_of,
                spot_price=spot_price,
            )
        return GammaExposureSummary.empty(as_of=as_of, symbol=symbol, spot_price=spot_price)

    def _default_signal(self, *, confidence: float) -> SensorSignal:
        components: dict[str, object] = {
            "session_intensity": 0.0,
            "news_proximity": 0.0,
            "gamma_impact": 0.0,
        }
        metadata: dict[str, object] = {
            "source": "sensory.when",
            "components": components,
        }
        return SensorSignal(
            signal_type="WHEN",
            value={"strength": 0.0, "confidence": confidence},
            confidence=confidence,
            metadata=metadata,
        )
