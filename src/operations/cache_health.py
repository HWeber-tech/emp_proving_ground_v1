"""Cache health evaluation and telemetry helpers."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Mapping

from src.core.event_bus import Event, EventBus, get_global_bus


logger = logging.getLogger(__name__)


class CacheHealthStatus(StrEnum):
    """Severity levels exposed by cache health telemetry."""

    ok = "ok"
    warn = "warn"
    fail = "fail"


_STATUS_ORDER: Mapping[CacheHealthStatus, int] = {
    CacheHealthStatus.ok: 0,
    CacheHealthStatus.warn: 1,
    CacheHealthStatus.fail: 2,
}


def _escalate(current: CacheHealthStatus, candidate: CacheHealthStatus) -> CacheHealthStatus:
    if _STATUS_ORDER[candidate] > _STATUS_ORDER[current]:
        return candidate
    return current


def _as_int(value: object | None) -> int:
    if value is None:
        return 0
    if isinstance(value, bool):  # bool is subclass of int – normalise explicitly
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return 0


def _as_float(value: object | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True)
class CacheHealthSnapshot:
    """Aggregated cache health telemetry."""

    service: str
    generated_at: datetime
    status: CacheHealthStatus
    configured: bool
    expected: bool
    namespace: str | None
    backing: str | None
    hit_rate: float | None
    hits: int
    misses: int
    evictions: int
    expirations: int
    invalidations: int
    metadata: dict[str, object] = field(default_factory=dict)
    issues: tuple[str, ...] = field(default_factory=tuple)

    def as_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "service": self.service,
            "generated_at": self.generated_at.isoformat(),
            "status": self.status.value,
            "configured": self.configured,
            "expected": self.expected,
            "namespace": self.namespace,
            "backing": self.backing,
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "expirations": self.expirations,
            "invalidations": self.invalidations,
            "issues": list(self.issues),
            "metadata": dict(self.metadata),
        }
        if self.hit_rate is not None:
            payload["hit_rate"] = self.hit_rate
        return payload

    def to_markdown(self) -> str:
        lines = [
            f"**Cache health – {self.service}**",
            f"- Status: {self.status.value}",
            f"- Configured: {'yes' if self.configured else 'no'} (expected: {'yes' if self.expected else 'no'})",
            f"- Namespace: {self.namespace or 'n/a'}",
            f"- Backing: {self.backing or 'n/a'}",
            f"- Hits: {self.hits}",
            f"- Misses: {self.misses}",
            f"- Hit rate: {self.hit_rate:.2%}" if self.hit_rate is not None else "- Hit rate: n/a",
            f"- Evictions: {self.evictions}",
            f"- Expirations: {self.expirations}",
            f"- Invalidations: {self.invalidations}",
        ]
        if self.metadata:
            detail = self.metadata.get("policy")
            if isinstance(detail, Mapping):
                ttl = detail.get("ttl_seconds")
                if ttl is not None:
                    lines.append(f"- TTL seconds: {ttl}")
                max_keys = detail.get("max_keys")
                if max_keys is not None:
                    lines.append(f"- Max keys: {max_keys}")
        if self.issues:
            lines.append("")
            lines.append("**Issues:**")
            for issue in self.issues:
                lines.append(f"- {issue}")
        return "\n".join(lines)


def format_cache_markdown(snapshot: CacheHealthSnapshot) -> str:
    """Convenience wrapper mirroring other operational formatters."""

    return snapshot.to_markdown()


def evaluate_cache_health(
    *,
    configured: bool,
    expected: bool,
    namespace: str | None,
    backing: str | None,
    metrics: Mapping[str, object] | None = None,
    policy: Mapping[str, object] | None = None,
    metadata: Mapping[str, object] | None = None,
    service: str = "redis_cache",
    now: datetime | None = None,
) -> CacheHealthSnapshot:
    """Assess cache health based on observed metrics and configuration."""

    metrics = metrics or {}
    moment = now or datetime.now(tz=UTC)
    issues: list[str] = []
    status = CacheHealthStatus.ok

    hits = max(_as_int(metrics.get("hits")), 0)
    misses = max(_as_int(metrics.get("misses")), 0)
    evictions = max(_as_int(metrics.get("evictions")), 0)
    expirations = max(_as_int(metrics.get("expirations")), 0)
    invalidations = max(_as_int(metrics.get("invalidations")), 0)

    total_requests = hits + misses
    hit_rate = None if total_requests <= 0 else hits / total_requests

    if expected and not configured:
        status = CacheHealthStatus.fail
        issues.append("Redis cache expected but not configured; falling back to in-memory store")
    elif not configured:
        status = CacheHealthStatus.warn
        issues.append("Redis cache not configured; cache metrics unavailable")

    if configured:
        if total_requests <= 0:
            status = _escalate(status, CacheHealthStatus.warn)
            issues.append("No cache traffic observed; hit/miss counters are zero")
        elif hit_rate is not None and hit_rate < 0.25 and misses >= 5:
            status = _escalate(status, CacheHealthStatus.warn)
            issues.append("Cache hit rate below 25% with sustained misses")
        elif hit_rate is not None and hit_rate < 0.1 and misses >= 20:
            status = _escalate(status, CacheHealthStatus.fail)
            issues.append("Cache hit rate below 10%; investigate key invalidation or TTL settings")

        if evictions > 0:
            status = _escalate(status, CacheHealthStatus.warn)
            issues.append(f"Cache evicted {evictions} keys; capacity may be too low")

    metadata_payload: dict[str, object] = {"policy": {}}
    if policy:
        metadata_payload["policy"] = dict(policy)
    else:
        metadata_payload.pop("policy", None)

    if metadata:
        metadata_payload.update(dict(metadata))

    snapshot = CacheHealthSnapshot(
        service=service,
        generated_at=moment,
        status=status,
        configured=configured,
        expected=expected,
        namespace=namespace,
        backing=backing,
        hit_rate=hit_rate,
        hits=hits,
        misses=misses,
        evictions=evictions,
        expirations=expirations,
        invalidations=invalidations,
        metadata=metadata_payload,
        issues=tuple(issues),
    )
    return snapshot


def publish_cache_health(event_bus: EventBus, snapshot: CacheHealthSnapshot) -> None:
    """Publish the cache health snapshot on the runtime event bus."""

    event = Event(
        type="telemetry.cache.health",
        payload=snapshot.as_dict(),
        source="operations.cache_health",
    )

    publish_from_sync = getattr(event_bus, "publish_from_sync", None)
    if callable(publish_from_sync) and event_bus.is_running():
        try:
            publish_result = publish_from_sync(event)
        except RuntimeError as exc:
            logger.warning(
                "Primary event bus publish_from_sync failed; falling back to global bus",
                exc_info=exc,
            )
        except Exception:
            logger.exception(
                "Unexpected error publishing cache health via runtime event bus",
            )
            raise
        else:
            if publish_result is not None:
                return
            logger.warning(
                "Primary event bus publish_from_sync returned None; falling back to global bus",
            )

    topic_bus = get_global_bus()
    try:
        topic_bus.publish_sync(event.type, event.payload, source=event.source)
    except RuntimeError as exc:
        logger.error(
            "Global event bus not running while publishing cache health snapshot",
            exc_info=exc,
        )
        raise
    except Exception:
        logger.exception(
            "Unexpected error publishing cache health snapshot via global bus",
        )
        raise


__all__ = [
    "CacheHealthStatus",
    "CacheHealthSnapshot",
    "evaluate_cache_health",
    "format_cache_markdown",
    "publish_cache_health",
]
