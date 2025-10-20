from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.data_foundation.ingest.sentiment_behavior_feed import (
    SentimentBehaviorFeed,
    TaggedNlpItem,
)


def _now() -> datetime:
    return datetime.now(tz=timezone.utc)


def test_feed_integrates_sources_and_aggregates_tags() -> None:
    now = _now()
    feed = SentimentBehaviorFeed(recency_half_life_hours=None)
    news = [
        TaggedNlpItem(
            source="news",
            timestamp=now - timedelta(minutes=10),
            sentiment=0.6,
            tags=("earnings", "guidance"),
            confidence=0.9,
        ),
        TaggedNlpItem(
            source="news",
            timestamp=now - timedelta(hours=1),
            sentiment=-0.2,
            tags=("regulation", "risk"),
            confidence=0.7,
        ),
    ]
    social = [
        TaggedNlpItem(
            source="social",
            timestamp=now - timedelta(minutes=5),
            sentiment=0.2,
            tags=("earnings", "buzz"),
            confidence=0.4,
        ),
    ]
    filings = [
        TaggedNlpItem(
            source="filing",
            timestamp=now - timedelta(hours=2),
            sentiment=-0.5,
            tags=("guidance", "dividend"),
            confidence=1.0,
        ),
    ]

    snapshot = feed.build_snapshot(news=news, social=social, filings=filings, as_of=now)

    assert snapshot.sample_size == 4
    assert snapshot.coverage == {"news": 2, "social": 1, "filings": 1, "extras": 0}
    assert snapshot.overall_sentiment == pytest.approx(-0.02 / 3.0)

    pulses = {pulse.tag: pulse for pulse in snapshot.tags}
    assert set(pulses.keys()) == {"earnings", "guidance", "regulation", "risk", "buzz", "dividend"}

    earnings = pulses["earnings"]
    assert earnings.mention_count == 2
    assert set(earnings.sources) == {"news", "social"}
    assert earnings.sentiment == pytest.approx((0.9 * 0.6 + 0.4 * 0.2) / (0.9 + 0.4))

    guidance = pulses["guidance"]
    assert guidance.mention_count == 2
    assert set(guidance.sources) == {"news", "filing"}


def test_feed_deduplicates_by_reference_and_keeps_highest_confidence() -> None:
    now = _now()
    feed = SentimentBehaviorFeed(recency_half_life_hours=None)
    low_confidence = TaggedNlpItem(
        source="social",
        timestamp=now - timedelta(minutes=6),
        sentiment=0.1,
        tags=("buzz",),
        confidence=0.3,
        reference="tweet-1",
    )
    high_confidence = TaggedNlpItem(
        source="social",
        timestamp=now - timedelta(minutes=2),
        sentiment=0.8,
        tags=("buzz", "momentum"),
        confidence=0.9,
        reference="tweet-1",
    )

    snapshot = feed.build_snapshot(social=[low_confidence, high_confidence], as_of=now)

    assert snapshot.sample_size == 1
    assert snapshot.coverage == {"news": 0, "social": 1, "filings": 0, "extras": 0}

    pulses = {pulse.tag: pulse for pulse in snapshot.tags}
    assert set(pulses.keys()) == {"buzz", "momentum"}
    assert pulses["buzz"].sentiment == pytest.approx(0.8)
    assert pulses["buzz"].confidence == pytest.approx(1.0)
    assert pulses["buzz"].mention_count == 1
    assert pulses["momentum"].mention_count == 1


def test_feed_applies_recency_decay() -> None:
    now = _now()
    feed = SentimentBehaviorFeed(recency_half_life_hours=1.0)
    fresh = TaggedNlpItem(
        source="news",
        timestamp=now - timedelta(minutes=5),
        sentiment=0.8,
        tags=("growth",),
        confidence=0.8,
    )
    stale = TaggedNlpItem(
        source="news",
        timestamp=now - timedelta(hours=4),
        sentiment=-0.8,
        tags=("growth",),
        confidence=0.8,
    )

    snapshot = feed.build_snapshot(news=[fresh, stale], as_of=now)

    assert snapshot.sample_size == 2
    assert snapshot.coverage["news"] == 2
    pulse = snapshot.tags[0]
    assert pulse.tag == "growth"
    assert pulse.sentiment > 0
