"""Narrative hooks for the WHY sensory dimension."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable, Mapping

import pandas as pd

__all__ = ["NarrativeEvent", "NarrativeSummary", "NarrativeHookEngine"]


@dataclass(slots=True)
class NarrativeEvent:
    """Represents an economic calendar event or macro catalyst."""

    timestamp: datetime
    sentiment: float
    importance: float = 1.0
    description: str | None = None

    def weight(self, reference: pd.Timestamp) -> float:
        ts = pd.Timestamp(self.timestamp).astimezone(timezone.utc)
        delta_minutes = (ts - reference).total_seconds() / 60.0
        if delta_minutes < 0:
            # Already occurred; apply exponential decay
            decay = min(1.0, abs(delta_minutes) / 240.0)
            return max(0.0, self.importance * (1.0 - decay))
        anticipation = max(0.0, 1.0 - min(delta_minutes, 180.0) / 180.0)
        return self.importance * (0.6 + 0.4 * anticipation)


@dataclass(slots=True)
class NarrativeSummary:
    sentiment_score: float
    dominant_theme: str | None
    upcoming_event: dict[str, object] | None

    def as_dict(self) -> dict[str, object]:
        return {
            "sentiment_score": float(self.sentiment_score),
            "dominant_theme": self.dominant_theme,
            "upcoming_event": self.upcoming_event,
        }


class NarrativeHookEngine:
    """Blend narrative events and macro regime flags into a summary."""

    def summarise(
        self,
        *,
        as_of: datetime | pd.Timestamp,
        events: Iterable[NarrativeEvent] | None = None,
        macro_flags: Mapping[str, float] | None = None,
    ) -> NarrativeSummary:
        timestamp = pd.Timestamp(as_of).astimezone(timezone.utc)

        sentiment = 0.0
        total_weight = 0.0
        next_event_payload: dict[str, object] | None = None
        next_event_delta: float | None = None

        if events:
            for event in events:
                weight = max(0.0, float(event.weight(timestamp)))
                if weight <= 0:
                    continue
                total_weight += weight
                sentiment += weight * float(event.sentiment)

                event_ts = pd.Timestamp(event.timestamp).astimezone(timezone.utc)
                minutes = (event_ts - timestamp).total_seconds() / 60.0
                if minutes >= 0 and (next_event_delta is None or minutes < next_event_delta):
                    next_event_delta = minutes
                    next_event_payload = {
                        "timestamp": event_ts.isoformat(),
                        "description": event.description,
                        "sentiment": float(event.sentiment),
                        "importance": float(event.importance),
                    }

        sentiment_score = float(sentiment / total_weight) if total_weight > 0 else 0.0

        dominant_theme: str | None = None
        if macro_flags:
            dominant_theme = max(macro_flags.items(), key=lambda item: abs(float(item[1])))[0]
            sentiment_score = 0.7 * sentiment_score + 0.3 * float(macro_flags[dominant_theme])

        if next_event_payload is not None and next_event_delta is not None:
            next_event_payload["minutes_ahead"] = next_event_delta

        return NarrativeSummary(
            sentiment_score=float(max(-1.0, min(1.0, sentiment_score))),
            dominant_theme=dominant_theme,
            upcoming_event=next_event_payload,
        )

