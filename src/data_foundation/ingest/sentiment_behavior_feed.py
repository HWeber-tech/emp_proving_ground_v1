"""Sentiment & behaviour feed integrating NLP tags across news, social, and filings."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from math import exp, log
from typing import Iterable, Mapping, Sequence


UTC = timezone.utc


def _ensure_utc(timestamp: datetime) -> datetime:
    if timestamp.tzinfo is None:
        return timestamp.replace(tzinfo=UTC)
    return timestamp.astimezone(UTC)


def _normalise_source(value: str) -> str:
    return value.strip().lower() if value else "unknown"


def _clean_tag(tag: str) -> str:
    return tag.strip()


@dataclass(slots=True, frozen=True)
class TaggedNlpItem:
    """Canonical representation of a tagged NLP artefact."""

    source: str
    timestamp: datetime
    sentiment: float
    tags: tuple[str, ...]
    confidence: float = 1.0
    reference: str | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "source", _normalise_source(self.source))
        object.__setattr__(self, "timestamp", _ensure_utc(self.timestamp))

        sentiment = max(-1.0, min(1.0, float(self.sentiment)))
        object.__setattr__(self, "sentiment", sentiment)

        confidence = max(0.0, min(1.0, float(self.confidence)))
        object.__setattr__(self, "confidence", confidence)

        cleaned_tags = tuple(dict.fromkeys(_clean_tag(str(tag)) for tag in self.tags if str(tag).strip()))
        object.__setattr__(self, "tags", cleaned_tags)

        reference = self.reference.strip() if isinstance(self.reference, str) else None
        object.__setattr__(self, "reference", reference or None)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> "TaggedNlpItem":
        raw_timestamp = payload.get("timestamp")
        timestamp: datetime
        if isinstance(raw_timestamp, datetime):
            timestamp = raw_timestamp
        elif isinstance(raw_timestamp, str):
            cleaned = raw_timestamp.strip()
            if cleaned.endswith("Z"):
                cleaned = cleaned[:-1] + "+00:00"
            try:
                timestamp = datetime.fromisoformat(cleaned)
            except ValueError:
                timestamp = datetime.now(tz=UTC)
        else:
            timestamp = datetime.now(tz=UTC)
        return cls(
            source=str(payload.get("source", "unknown")),
            timestamp=timestamp,
            sentiment=float(payload.get("sentiment", 0.0)),
            tags=tuple(payload.get("tags", ())),
            confidence=float(payload.get("confidence", 1.0)),
            reference=str(payload.get("reference")) if payload.get("reference") is not None else None,
            metadata=dict(payload.get("metadata", {})),
        )


@dataclass(slots=True, frozen=True)
class TagPulse:
    """Aggregate sentiment pulse for a normalised tag."""

    tag: str
    sentiment: float
    confidence: float
    mention_count: int
    sources: tuple[str, ...]
    source_breakdown: Mapping[str, int]
    last_seen: datetime
    weight: float

    def as_dict(self) -> dict[str, object]:
        return {
            "tag": self.tag,
            "sentiment": self.sentiment,
            "confidence": self.confidence,
            "mention_count": self.mention_count,
            "sources": list(self.sources),
            "source_breakdown": dict(self.source_breakdown),
            "last_seen": self.last_seen.isoformat(),
            "weight": self.weight,
        }


@dataclass(slots=True, frozen=True)
class SentimentBehaviorSnapshot:
    """Snapshot of integrated sentiment & behaviour signals."""

    generated_at: datetime
    overall_sentiment: float
    sample_size: int
    coverage: Mapping[str, int]
    tags: tuple[TagPulse, ...] = ()
    metadata: Mapping[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "generated_at": self.generated_at.isoformat(),
            "overall_sentiment": self.overall_sentiment,
            "sample_size": self.sample_size,
            "coverage": dict(self.coverage),
            "tags": [pulse.as_dict() for pulse in self.tags],
            "metadata": dict(self.metadata),
        }

    def top_tags(self, limit: int = 5) -> tuple[TagPulse, ...]:
        if limit <= 0:
            return ()
        return self.tags[:limit]


@dataclass(slots=True)
class SentimentBehaviorFeed:
    """Integrates NLP tagging across news, social chatter, and regulatory filings."""

    minimum_confidence: float = 0.1
    recency_half_life_hours: float | None = 24.0

    def build_snapshot(
        self,
        *,
        news: Sequence[TaggedNlpItem] | None = None,
        social: Sequence[TaggedNlpItem] | None = None,
        filings: Sequence[TaggedNlpItem] | None = None,
        extras: Sequence[TaggedNlpItem] | None = None,
        as_of: datetime | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> SentimentBehaviorSnapshot:
        as_of_ts = _ensure_utc(as_of or datetime.now(tz=UTC))
        items = list(self._combine_inputs(news, social, filings, extras))
        filtered = self._filter_items(items)
        weighted = list(self._apply_weights(filtered, as_of_ts))
        return self._materialise_snapshot(weighted, as_of_ts, metadata)

    def _combine_inputs(
        self,
        news: Sequence[TaggedNlpItem] | None,
        social: Sequence[TaggedNlpItem] | None,
        filings: Sequence[TaggedNlpItem] | None,
        extras: Sequence[TaggedNlpItem] | None,
    ) -> Iterable[TaggedNlpItem]:
        for bucket in (news, social, filings, extras):
            if not bucket:
                continue
            for item in bucket:
                yield item

    def _filter_items(self, items: Sequence[TaggedNlpItem]) -> list[TaggedNlpItem]:
        dedup: dict[tuple[str, str], TaggedNlpItem] = {}
        filtered: list[TaggedNlpItem] = []
        for item in items:
            if not item.tags:
                continue
            if item.confidence < self.minimum_confidence:
                continue
            reference = item.reference
            if reference:
                key = (item.source, reference)
                existing = dedup.get(key)
                if existing is None or item.confidence > existing.confidence or item.timestamp > existing.timestamp:
                    dedup[key] = item
            else:
                filtered.append(item)
        filtered.extend(dedup.values())
        filtered.sort(key=lambda entry: entry.timestamp)
        return filtered

    def _apply_weights(
        self,
        items: Sequence[TaggedNlpItem],
        as_of: datetime,
    ) -> Iterable[tuple[TaggedNlpItem, float]]:
        if not items:
            return []
        decay_const: float | None = None
        if self.recency_half_life_hours and self.recency_half_life_hours > 0:
            decay_const = log(0.5) / (self.recency_half_life_hours * 3600.0)
        weighted: list[tuple[TaggedNlpItem, float]] = []
        for item in items:
            weight = item.confidence
            if decay_const is not None:
                delta_seconds = (as_of - item.timestamp).total_seconds()
                if delta_seconds > 0:
                    weight *= exp(decay_const * delta_seconds)
            if weight <= 0.0:
                continue
            weighted.append((item, weight))
        return weighted

    def _materialise_snapshot(
        self,
        weighted_items: Sequence[tuple[TaggedNlpItem, float]],
        as_of: datetime,
        metadata: Mapping[str, object] | None,
    ) -> SentimentBehaviorSnapshot:
        if not weighted_items:
            return SentimentBehaviorSnapshot(
                generated_at=as_of,
                overall_sentiment=0.0,
                sample_size=0,
                coverage={"news": 0, "social": 0, "filings": 0, "extras": 0},
                tags=(),
                metadata=dict(metadata or {}),
            )

        source_counts: Counter[str] = Counter()
        tag_stats: dict[str, dict[str, object]] = {}
        total_weight = 0.0
        sentiment_sum = 0.0

        for item, weight in weighted_items:
            total_weight += weight
            sentiment_sum += weight * item.sentiment
            source_counts[item.source] += 1

            unique_tags = set(item.tags)
            for tag in unique_tags:
                key = tag.lower()
                stats = tag_stats.setdefault(
                    key,
                    {
                        "tag": tag,
                        "weight": 0.0,
                        "sentiment": 0.0,
                        "mentions": 0,
                        "sources": Counter[str](),
                        "last_seen": item.timestamp,
                    },
                )
                stats["weight"] += weight
                stats["sentiment"] += weight * item.sentiment
                stats["mentions"] += 1
                stats_sources: Counter[str] = stats["sources"]
                stats_sources[item.source] += 1
                if item.timestamp > stats["last_seen"]:
                    stats["last_seen"] = item.timestamp

        overall_sentiment = sentiment_sum / total_weight if total_weight else 0.0
        overall_sentiment = max(-1.0, min(1.0, overall_sentiment))

        max_weight = max((stats["weight"] for stats in tag_stats.values()), default=1.0)
        pulses: list[TagPulse] = []
        for stats in tag_stats.values():
            weight = float(stats["weight"])
            sentiment = stats["sentiment"] / weight if weight else 0.0
            confidence = max(
                self.minimum_confidence,
                min(1.0, weight / max_weight if max_weight else 0.0),
            )
            sources_counter: Counter[str] = stats["sources"]
            pulse = TagPulse(
                tag=stats["tag"],
                sentiment=max(-1.0, min(1.0, sentiment)),
                confidence=confidence,
                mention_count=int(stats["mentions"]),
                sources=tuple(sorted(sources_counter)),
                source_breakdown=dict(sorted(sources_counter.items())),
                last_seen=stats["last_seen"],
                weight=weight,
            )
            pulses.append(pulse)

        pulses.sort(key=lambda pulse: (pulse.weight, pulse.mention_count, pulse.tag), reverse=True)

        coverage = {
            "news": source_counts.get("news", 0),
            "social": source_counts.get("social", 0),
            "filings": source_counts.get("filing", 0) + source_counts.get("filings", 0),
            "extras": source_counts.get("extras", 0),
        }

        return SentimentBehaviorSnapshot(
            generated_at=as_of,
            overall_sentiment=overall_sentiment,
            sample_size=len(weighted_items),
            coverage=coverage,
            tags=tuple(pulses),
            metadata=dict(metadata or {}),
        )
