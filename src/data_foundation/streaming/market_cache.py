"""Redis-backed market data cache with sliding window retention."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
import json
from typing import Callable, Mapping, MutableSequence, Protocol, Sequence

from src.data_foundation.monitoring.feed_anomaly import Tick

__all__ = ["MarketDataCache", "TickSnapshot", "RedisLike", "InMemoryRedis"]

TickInput = Tick | Mapping[str, object]
WarmSource = Mapping[str, Sequence[TickInput]] | Callable[[], Mapping[str, Sequence[TickInput]]]


class PipelineLike(Protocol):
    """Subset of the redis-py pipeline API used by the cache."""

    def lpush(self, key: str, *values: str) -> PipelineLike:  # pragma: no cover - protocol
        ...

    def delete(self, key: str) -> PipelineLike:  # pragma: no cover - protocol
        ...

    def ltrim(self, key: str, start: int, end: int) -> PipelineLike:  # pragma: no cover - protocol
        ...

    def expire(self, key: str, ttl_seconds: int) -> PipelineLike:  # pragma: no cover - protocol
        ...

    def execute(self) -> list[object]:  # pragma: no cover - protocol
        ...


class RedisLike(Protocol):
    """Minimal redis client protocol required by :class:`MarketDataCache`."""

    def lpush(self, key: str, *values: str) -> int:  # pragma: no cover - protocol
        ...

    def ltrim(self, key: str, start: int, end: int) -> int:  # pragma: no cover - protocol
        ...

    def lrange(self, key: str, start: int, end: int) -> Sequence[str]:  # pragma: no cover - protocol
        ...

    def delete(self, *keys: str) -> int:  # pragma: no cover - protocol
        ...

    def expire(self, key: str, ttl_seconds: int) -> bool:  # pragma: no cover - protocol
        ...

    def pipeline(self) -> PipelineLike:  # pragma: no cover - protocol
        ...


@dataclass(frozen=True, slots=True)
class TickSnapshot:
    """Serialised representation stored in Redis."""

    timestamp: datetime
    price: float
    volume: float | None = None
    seqno: int | None = None

    def to_json(self) -> str:
        payload = {
            "timestamp": self.timestamp.astimezone(UTC).isoformat(),
            "price": float(self.price),
        }
        if self.volume is not None:
            payload["volume"] = float(self.volume)
        if self.seqno is not None:
            payload["seqno"] = int(self.seqno)
        return json.dumps(payload, separators=(",", ":"))

    @staticmethod
    def from_json(raw: str) -> Tick:
        payload = json.loads(raw)
        timestamp = payload["timestamp"]
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        if isinstance(timestamp, datetime):
            if timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=UTC)
            else:
                timestamp = timestamp.astimezone(UTC)
        else:  # pragma: no cover - defensive
            raise TypeError("timestamp must decode to a datetime")
        return Tick(
            timestamp=timestamp,
            price=float(payload["price"]),
            volume=float(payload["volume"]) if payload.get("volume") is not None else None,
            seqno=int(payload["seqno"]) if payload.get("seqno") is not None else None,
        )


def _normalise_tick(value: TickInput) -> TickSnapshot:
    if isinstance(value, Tick):
        timestamp = value.timestamp
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=UTC)
        else:
            timestamp = timestamp.astimezone(UTC)
        return TickSnapshot(timestamp, float(value.price), value.volume, value.seqno)

    if not isinstance(value, Mapping):  # pragma: no cover - defensive
        raise TypeError("tick must be a Tick or mapping")

    raw_timestamp = value.get("timestamp")
    if isinstance(raw_timestamp, str):
        timestamp = datetime.fromisoformat(raw_timestamp)
    elif isinstance(raw_timestamp, datetime):
        timestamp = raw_timestamp
    else:  # pragma: no cover - defensive
        raise TypeError("timestamp must be str or datetime")

    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=UTC)
    else:
        timestamp = timestamp.astimezone(UTC)

    price = value.get("price")
    if price is None:
        raise ValueError("tick payload missing price")

    volume = value.get("volume")
    seqno = value.get("seqno")

    return TickSnapshot(
        timestamp=timestamp,
        price=float(price),
        volume=float(volume) if volume is not None else None,
        seqno=int(seqno) if seqno is not None else None,
    )


class InMemoryPipeline:
    """Naive in-memory pipeline emulating redis-py behaviour for tests."""

    def __init__(self, redis: InMemoryRedis) -> None:  # type: ignore[name-defined]
        self._redis = redis
        self._commands: MutableSequence[Callable[[], object]] = []

    def lpush(self, key: str, *values: str) -> InMemoryPipeline:
        self._commands.append(lambda: self._redis.lpush(key, *values))
        return self

    def delete(self, key: str) -> InMemoryPipeline:
        self._commands.append(lambda: self._redis.delete(key))
        return self

    def ltrim(self, key: str, start: int, end: int) -> InMemoryPipeline:
        self._commands.append(lambda: self._redis.ltrim(key, start, end))
        return self

    def expire(self, key: str, ttl_seconds: int) -> InMemoryPipeline:
        self._commands.append(lambda: self._redis.expire(key, ttl_seconds))
        return self

    def execute(self) -> list[object]:
        return [command() for command in self._commands]


class InMemoryRedis:
    """Test double implementing the subset of redis features we need."""

    def __init__(self, *, clock: Callable[[], datetime] | None = None) -> None:
        self._store: dict[str, list[str]] = {}
        self._expiry: dict[str, datetime] = {}
        self._clock = clock or (lambda: datetime.now(UTC))

    def pipeline(self) -> InMemoryPipeline:
        return InMemoryPipeline(self)

    def lpush(self, key: str, *values: str) -> int:
        self._evict_if_expired(key)
        bucket = self._store.setdefault(key, [])
        for value in values:
            bucket.insert(0, value)
        return len(bucket)

    def ltrim(self, key: str, start: int, end: int) -> int:
        self._evict_if_expired(key)
        bucket = self._store.get(key)
        if bucket is None:
            return 0
        # Redis ltrim is inclusive and supports negative indices. Re-implement subset we need.
        length = len(bucket)
        if start < 0:
            start = max(length + start, 0)
        if end < 0:
            end = length + end
        end = min(end, length - 1)
        if start > end:
            self._store[key] = []
        else:
            self._store[key] = bucket[start : end + 1]
        return len(self._store[key])

    def lrange(self, key: str, start: int, end: int) -> Sequence[str]:
        self._evict_if_expired(key)
        bucket = self._store.get(key, [])
        length = len(bucket)
        if not bucket:
            return []
        if start < 0:
            start = max(length + start, 0)
        if end < 0:
            end = length + end
        end = min(end, length - 1)
        if start > end:
            return []
        return bucket[start : end + 1]

    def delete(self, *keys: str) -> int:
        removed = 0
        for key in keys:
            self._evict_if_expired(key)
            if key in self._store:
                del self._store[key]
                removed += 1
            self._expiry.pop(key, None)
        return removed

    def expire(self, key: str, ttl_seconds: int) -> bool:
        self._evict_if_expired(key)
        if ttl_seconds <= 0:
            self.delete(key)
            return True
        self._expiry[key] = self._clock() + timedelta(seconds=ttl_seconds)
        return True

    def _evict_if_expired(self, key: str) -> None:
        deadline = self._expiry.get(key)
        if deadline is None:
            return
        if deadline <= self._clock():
            self._store.pop(key, None)
            self._expiry.pop(key, None)


class MarketDataCache:
    """High-level wrapper storing recent ticks per symbol in Redis."""

    def __init__(
        self,
        *,
        redis_client: RedisLike | None = None,
        window_size: int = 1000,
        namespace: str = "market",
        ttl_seconds: int | None = None,
        warm_start: WarmSource | None = None,
    ) -> None:
        if window_size < 1:
            raise ValueError("window_size must be positive")
        self._redis: RedisLike = redis_client or InMemoryRedis()
        self._window_size = window_size
        self._namespace = namespace
        self._ttl_seconds = ttl_seconds
        if warm_start:
            self.warm_cache(warm_start)

    def warm_cache(self, data_or_loader: WarmSource) -> None:
        data = data_or_loader() if callable(data_or_loader) else data_or_loader
        if not data:
            return
        pipe = self._redis.pipeline()
        for symbol, ticks in data.items():
            serialised = self._serialise_ticks(ticks)
            key = self._key(symbol)
            pipe.delete(key)
            if serialised:
                pipe.lpush(key, *serialised)
                pipe.ltrim(key, 0, self._window_size - 1)
                if self._ttl_seconds is not None:
                    pipe.expire(key, self._ttl_seconds)
        pipe.execute()

    def store_tick(self, symbol: str, tick: TickInput) -> None:
        self.store_ticks(symbol, [tick])

    def store_ticks(self, symbol: str, ticks: Sequence[TickInput]) -> None:
        if not ticks:
            return
        serialised = self._serialise_ticks(ticks)
        if not serialised:
            return
        key = self._key(symbol)
        pipe = self._redis.pipeline()
        pipe.lpush(key, *serialised)
        pipe.ltrim(key, 0, self._window_size - 1)
        if self._ttl_seconds is not None:
            pipe.expire(key, self._ttl_seconds)
        pipe.execute()

    def get_recent_ticks(self, symbol: str, limit: int | None = None) -> tuple[Tick, ...]:
        if limit is None or limit > self._window_size:
            limit = self._window_size
        if limit <= 0:
            return ()
        raw_ticks = self._redis.lrange(self._key(symbol), 0, limit - 1)
        return tuple(TickSnapshot.from_json(raw) for raw in raw_ticks)

    def get_latest_tick(self, symbol: str) -> Tick | None:
        ticks = self.get_recent_ticks(symbol, limit=1)
        return ticks[0] if ticks else None

    def clear(self, symbol: str | None = None) -> None:
        if symbol is None:
            keys = list(getattr(self._redis, "_store", {}).keys())  # type: ignore[attr-defined]
            if not keys:
                return
            self._redis.delete(*keys)
            return
        self._redis.delete(self._key(symbol))

    def _serialise_ticks(self, ticks: Sequence[TickInput]) -> tuple[str, ...]:
        snapshots = [_normalise_tick(tick) for tick in ticks]
        return tuple(snapshot.to_json() for snapshot in snapshots)

    def _key(self, symbol: str) -> str:
        return f"{self._namespace}:{symbol}"
