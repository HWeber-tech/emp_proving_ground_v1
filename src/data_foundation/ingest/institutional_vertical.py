"""Provisioning helpers for the institutional Timescale ingest vertical.

The roadmap calls for Tier-1 institutional runs to operate against managed
Timescale, Redis, and Kafka services while ensuring all background activity is
supervised.  This module ties together the existing configuration dataclasses
with runtime wiring so orchestration code can provision a fully supervised
ingest stack without re-implementing boilerplate in the runtime builder.

High-level responsibilities:

* Hydrate :class:`TimescaleIngestScheduler` instances with sensible defaults
  when the configuration does not request a bespoke schedule.
* Configure Redis caches using the institutional policy and expose a summary
  that can be surfaced in readiness dashboards.
* Instantiate Kafka ingest consumers and register their long-running coroutine
  with :class:`TaskSupervisor` so operators inherit lifecycle guarantees.
* Surface failover drill metadata so disaster-recovery automation can discover
  which scenarios are required for the active ingest slice.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Mapping, MutableMapping, Sequence

from sqlalchemy import text

try:  # pragma: no cover - optional dependency
    from sqlalchemy.exc import SQLAlchemyError as _SQLAlchemyError
except ModuleNotFoundError:  # pragma: no cover - test fallback when sqlalchemy absent
    class _SQLAlchemyError(Exception):
        """Fallback SQLAlchemyError used when SQLAlchemy is unavailable."""

try:  # pragma: no cover - optional dependency
    from redis.exceptions import RedisError as _RedisError
except ModuleNotFoundError:  # pragma: no cover - redis optional in tests
    class _RedisError(Exception):
        """Fallback RedisError used when redis is unavailable."""

from src.data_foundation.cache.redis_cache import (
    ManagedRedisCache,
    RedisCachePolicy,
    RedisConnectionSettings,
    configure_redis_client,
)
from src.data_foundation.ingest.configuration import InstitutionalIngestConfig
from src.data_foundation.ingest.scheduler import (
    IngestSchedule,
    RunCallback,
    TimescaleIngestScheduler,
)
from src.data_foundation.ingest.failover import IngestFailoverPolicy
from src.data_foundation.streaming.kafka_stream import (
    KafkaConnectionSettings,
    KafkaIngestEventConsumer,
    KafkaConsumerFactory,
    create_ingest_event_consumer,
    ingest_topic_config_from_mapping,
)
from src.data_foundation.persist.timescale import TimescaleIngestResult
from src.runtime.task_supervisor import TaskSupervisor
from src.operations.failover_drill import (
    FailoverDrillSnapshot,
    execute_failover_drill,
)


logger = logging.getLogger(__name__)


ProbeCallable = Callable[[], Awaitable[bool] | bool]


DEFAULT_INTERVAL_SECONDS = 3_600.0
DEFAULT_JITTER_SECONDS = 120.0


class ConnectivityProbeError(RuntimeError):
    """Raised when a managed connector health probe hits an expected failure."""


_EXPECTED_TIMESCALE_ERRORS: tuple[type[Exception], ...] = (
    _SQLAlchemyError,
    OSError,
    TimeoutError,
)
_EXPECTED_REDIS_ERRORS: tuple[type[Exception], ...] = (
    _RedisError,
    OSError,
    TimeoutError,
)
_EXPECTED_KAFKA_ERRORS: tuple[type[Exception], ...] = (
    RuntimeError,
    ValueError,
    TimeoutError,
    ConnectionError,
    AttributeError,
)


def _log_probe_failure(component: str, error: Exception, message: str | None = None) -> None:
    """Emit a warning with stack trace for expected probe failures."""

    summary = message or f"{component} connectivity probe failed"
    logger.warning("%s: %s", summary, error, exc_info=error)
DEFAULT_MAX_FAILURES = 3


def _extract_kafka_topics(mapping: Mapping[str, str]) -> tuple[str, ...]:
    topics: set[str] = set()
    topic_map, default_topic = ingest_topic_config_from_mapping(mapping)
    for topic in topic_map.values():
        cleaned = topic.strip()
        if cleaned:
            topics.add(cleaned)

    raw_consumer_topics = mapping.get("KAFKA_INGEST_CONSUMER_TOPICS")
    if raw_consumer_topics:
        for entry in str(raw_consumer_topics).split(","):
            cleaned = entry.strip()
            if cleaned:
                topics.add(cleaned)

    if default_topic:
        cleaned_default = default_topic.strip()
        if cleaned_default:
            topics.add(cleaned_default)

    return tuple(sorted(topics))


_TRUE_SENTINELS: frozenset[str] = frozenset({
    "1",
    "true",
    "yes",
    "y",
    "on",
})
_FALSE_SENTINELS: frozenset[str] = frozenset({
    "0",
    "false",
    "no",
    "n",
    "off",
})


def _coerce_bool(value: str | None, default: bool) -> bool:
    """Normalise boolean configuration values."""

    if value is None:
        return default
    normalized = str(value).strip().lower().replace("-", "_")
    if not normalized:
        return default
    if normalized in _TRUE_SENTINELS:
        return True
    if normalized in _FALSE_SENTINELS:
        return False
    return default


def _coerce_optional_float(value: str | None, *, default: float | None) -> float | None:
    """Normalise optional float configuration values with sentinel handling."""

    if value is None:
        return default

    normalized = str(value).strip().lower()
    if not normalized:
        return default
    if normalized in {"none", "null"}:
        return None
    if normalized in {"off", "disable", "disabled"}:
        return None
    try:
        return float(normalized)
    except (TypeError, ValueError):
        return default


def _build_kafka_metadata(
    mapping: Mapping[str, str] | None,
    *,
    dimensions: Sequence[str],
    kafka_topics: Sequence[str],
    provisioned: bool,
    streaming_enabled: bool,
) -> dict[str, object]:
    """Summarise Kafka ingest configuration for manifests and telemetry."""

    payload = {str(key): str(value) for key, value in (mapping or {}).items()}

    consumer_enabled = _coerce_bool(
        payload.get("KAFKA_INGEST_CONSUMER_ENABLED"),
        True,
    )

    consumer_group = (
        payload.get("KAFKA_INGEST_CONSUMER_GROUP") or "emp-ingest-bridge"
    ).strip()
    if not consumer_group:
        consumer_group = "emp-ingest-bridge"

    auto_reset = (
        payload.get("KAFKA_INGEST_CONSUMER_OFFSET_RESET")
        or payload.get("KAFKA_INGEST_CONSUMER_AUTO_OFFSET_RESET")
        or "latest"
    ).strip().lower()
    if not auto_reset:
        auto_reset = "latest"

    auto_commit = _coerce_bool(
        payload.get("KAFKA_INGEST_CONSUMER_AUTO_COMMIT"),
        True,
    )

    commit_on_publish = _coerce_bool(
        payload.get("KAFKA_INGEST_CONSUMER_COMMIT_ON_PUBLISH"),
        not auto_commit,
    )

    commit_async = _coerce_bool(
        payload.get("KAFKA_INGEST_CONSUMER_COMMIT_ASYNC"),
        False,
    )

    poll_timeout = _coerce_optional_float(
        payload.get("KAFKA_INGEST_CONSUMER_POLL_TIMEOUT"),
        default=1.0,
    )
    if poll_timeout is None or poll_timeout <= 0:
        poll_timeout = 1.0

    idle_sleep = _coerce_optional_float(
        payload.get("KAFKA_INGEST_CONSUMER_IDLE_SLEEP"),
        default=0.5,
    )
    if idle_sleep is None or idle_sleep < 0:
        idle_sleep = 0.5

    publish_consumer_lag = _coerce_bool(
        payload.get("KAFKA_INGEST_CONSUMER_PUBLISH_LAG"),
        False,
    )

    consumer_lag_event_type = (
        payload.get("KAFKA_INGEST_CONSUMER_LAG_EVENT_TYPE")
        or "telemetry.kafka.lag"
    ).strip()
    if not consumer_lag_event_type:
        consumer_lag_event_type = "telemetry.kafka.lag"

    consumer_lag_source = (
        payload.get("KAFKA_INGEST_CONSUMER_LAG_EVENT_SOURCE") or ""
    ).strip() or None

    consumer_lag_interval = _coerce_optional_float(
        payload.get("KAFKA_INGEST_CONSUMER_LAG_INTERVAL"),
        default=30.0,
    )

    event_type = (
        payload.get("KAFKA_INGEST_CONSUMER_EVENT_TYPE") or "telemetry.ingest"
    ).strip()
    if not event_type:
        event_type = "telemetry.ingest"

    event_source = (
        payload.get("KAFKA_INGEST_CONSUMER_EVENT_SOURCE")
        or "timescale_ingest.kafka"
    ).strip()
    if not event_source:
        event_source = "timescale_ingest.kafka"

    configured_topics = [topic for topic in kafka_topics if str(topic).strip()]

    return {
        "timescale_dimensions": list(dimensions),
        "consumer_enabled": consumer_enabled,
        "provisioned": bool(provisioned),
        "consumer_group": consumer_group,
        "auto_offset_reset": auto_reset,
        "auto_commit": auto_commit,
        "commit_on_publish": commit_on_publish,
        "commit_async": commit_async,
        "poll_timeout_seconds": float(poll_timeout),
        "idle_sleep_seconds": float(idle_sleep),
        "event_type": event_type,
        "event_source": event_source,
        "publish_consumer_lag": publish_consumer_lag,
        "consumer_lag_event_type": consumer_lag_event_type,
        "consumer_lag_event_source": consumer_lag_source,
        "consumer_lag_interval_seconds": consumer_lag_interval,
        "consumer_topics_configured": bool(configured_topics),
        "configured_topics": tuple(configured_topics),
        "topic_count": len(configured_topics),
        "streaming_enabled": bool(streaming_enabled),
        "streaming_active": bool(streaming_enabled and provisioned),
    }


def default_institutional_schedule() -> IngestSchedule:
    """Return the baseline ingest schedule for institutional runs."""

    return IngestSchedule(
        interval_seconds=DEFAULT_INTERVAL_SECONDS,
        jitter_seconds=DEFAULT_JITTER_SECONDS,
        max_failures=DEFAULT_MAX_FAILURES,
    )


def _plan_dimensions(config: InstitutionalIngestConfig) -> tuple[str, ...]:
    plan = config.plan
    dimensions: list[str] = []
    if plan.daily is not None:
        dimensions.append("daily")
    if plan.intraday is not None:
        dimensions.append("intraday")
    if plan.macro is not None:
        dimensions.append("macro")
    return tuple(dimensions)


def _redacted_url(url: str) -> str:
    if "@" not in url:
        return url
    credential, _, host = url.partition("@")
    if ":" in credential:
        prefix, _, suffix = credential.rpartition(":")
        prefix = prefix or "***"
        return f"{prefix}:***@{host}"
    return f"***@{host}"


def _policy_metadata(policy: RedisCachePolicy | None) -> dict[str, object] | None:
    if policy is None:
        return None
    return {
        "ttl_seconds": policy.ttl_seconds,
        "max_keys": policy.max_keys,
        "namespace": policy.namespace,
        "invalidate_prefixes": list(policy.invalidate_prefixes),
    }


def _prepare_redis_metrics(
    metrics: Mapping[str, object] | None,
) -> dict[str, object] | None:
    """Normalise Redis metrics and derive aggregate fields for summaries."""

    if metrics is None:
        return None

    payload: dict[str, object] = {str(key): value for key, value in metrics.items()}

    def _coerce_int(value: object) -> int | None:
        if value is None:
            return None
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, (int, float)):
            return int(value)
        try:
            return int(str(value).strip())
        except (TypeError, ValueError):
            return None

    hits = _coerce_int(payload.get("hits"))
    misses = _coerce_int(payload.get("misses"))

    if hits is not None and misses is not None:
        total_requests = hits + misses
        payload.setdefault("requests", total_requests)
        payload["hit_rate"] = (hits / total_requests) if total_requests > 0 else None

    return payload


@dataclass(slots=True)
class InstitutionalIngestServices:
    """Runtime objects bound to the managed ingest vertical."""

    config: InstitutionalIngestConfig
    scheduler: TimescaleIngestScheduler
    task_supervisor: TaskSupervisor
    redis_settings: RedisConnectionSettings | None = None
    redis_cache: ManagedRedisCache | None = None
    redis_policy: RedisCachePolicy | None = None
    kafka_settings: KafkaConnectionSettings | None = None
    kafka_consumer: KafkaIngestEventConsumer | None = None
    kafka_task_name: str = "timescale-ingest-kafka-bridge"
    kafka_metadata: Mapping[str, object] = field(default_factory=dict)

    _kafka_task: asyncio.Task[None] | None = field(init=False, default=None)
    _kafka_stop_event: asyncio.Event | None = field(init=False, default=None)
    _scheduler_task: asyncio.Task[None] | None = field(init=False, default=None)

    def start(self) -> None:
        """Start supervised components (scheduler plus Kafka bridge)."""

        self._scheduler_task = self.scheduler.start()
        if self.kafka_consumer is None or self._kafka_task is not None:
            return

        stop_event = asyncio.Event()
        self._kafka_stop_event = stop_event
        metadata: MutableMapping[str, object] = {
            "component": "timescale_ingest.kafka_consumer",
            "topics": list(self.kafka_consumer.topics),
        }
        metadata.update(self.kafka_metadata)

        self._kafka_task = self.task_supervisor.create(
            self.kafka_consumer.run_forever(stop_event),
            name=self.kafka_task_name,
            metadata=metadata,
        )

    async def stop(self) -> None:
        """Stop background components gracefully."""

        if self._kafka_stop_event is not None:
            self._kafka_stop_event.set()
        if self._kafka_task is not None:
            try:
                await asyncio.wait_for(self._kafka_task, timeout=2.0)
            except asyncio.TimeoutError:
                self._kafka_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self._kafka_task
            finally:
                self._kafka_task = None
                self._kafka_stop_event = None

        await self.scheduler.stop()
        self._scheduler_task = None

    def managed_manifest(self) -> tuple["ManagedConnectorSnapshot", ...]:
        """Return structured metadata for provisioned managed services."""

        kafka_topics: Sequence[str] | None = None
        if self.kafka_consumer is not None:
            kafka_topics = self.kafka_consumer.topics
        else:
            configured = self.kafka_metadata.get("configured_topics")
            if isinstance(configured, (list, tuple, set)):
                kafka_topics = tuple(str(topic) for topic in configured if str(topic).strip())

        return _build_managed_manifest(
            self.config,
            scheduler_running=self.scheduler.running,
            redis_settings=self.redis_settings,
            redis_cache=self.redis_cache,
            redis_policy=self.redis_policy,
            kafka_settings=self.kafka_settings,
            kafka_consumer=self.kafka_consumer,
            kafka_supervised=self._kafka_task is not None,
            kafka_metadata=self.kafka_metadata,
            kafka_topics=kafka_topics,
        )

    def summary(self) -> Mapping[str, object]:
        """Summarise managed connectors for observability dashboards."""

        schedule = self.scheduler
        schedule_state = schedule.state()

        redis_summary: str | None = None
        redis_backing: str | None = None
        redis_metrics: Mapping[str, object] | None = None
        if self.redis_settings is not None and self.redis_settings.configured:
            redis_summary = self.redis_settings.summary(redacted=True)
        if self.redis_cache is not None:
            raw_client = getattr(self.redis_cache, "raw_client", None)
            if raw_client is not None:
                redis_backing = raw_client.__class__.__name__
            try:
                redis_metrics = self.redis_cache.metrics()
            except Exception:  # pragma: no cover - metrics are best-effort
                logger.debug("Failed to collect Redis cache metrics", exc_info=True)

        prepared_redis_metrics = _prepare_redis_metrics(redis_metrics)

        kafka_summary: str | None = None
        if self.kafka_settings is not None and self.kafka_settings.configured:
            kafka_summary = self.kafka_settings.summary(redacted=True)

        kafka_topics: list[str] = []
        if self.kafka_consumer is not None:
            kafka_topics = list(self.kafka_consumer.topics)
        else:
            configured_topics = self.kafka_metadata.get("configured_topics")
            if isinstance(configured_topics, (list, tuple, set)):
                kafka_topics = [
                    str(topic)
                    for topic in configured_topics
                    if str(topic).strip()
                ]

        return {
            "timescale": {
                "configured": self.config.timescale_settings.configured,
                "application_name": self.config.timescale_settings.application_name,
                "url": _redacted_url(self.config.timescale_settings.url),
                "dimensions": list(_plan_dimensions(self.config)),
            },
            "schedule": {
                "running": schedule.running,
                "interval_seconds": schedule_state.interval_seconds,
                "jitter_seconds": schedule_state.jitter_seconds,
                "max_failures": schedule_state.max_failures,
                "next_run_at": schedule_state.next_run_at.isoformat()
                if schedule_state.next_run_at
                else None,
            },
            "redis": redis_summary,
            "redis_backing": redis_backing,
            "redis_policy": _policy_metadata(self.redis_policy),
            "redis_metrics": prepared_redis_metrics,
            "kafka": kafka_summary,
            "kafka_topics": kafka_topics,
            "kafka_metadata": dict(self.kafka_metadata) if self.kafka_metadata else {},
            "kafka_streaming_enabled": self.kafka_metadata.get(
                "streaming_enabled", self.config.enable_streaming
            ),
            "failover": self.failover_metadata(),
            "managed_manifest": [snapshot.as_dict() for snapshot in self.managed_manifest()],
        }

    def failover_metadata(self) -> Mapping[str, object] | None:
        """Expose configured failover drill requirements, if any."""

        settings = self.config.failover_drill
        if settings is None:
            return None
        return settings.to_metadata()

    async def connectivity_report(
        self,
        *,
        probes: Mapping[str, ProbeCallable] | None = None,
        timeout: float = 5.0,
    ) -> tuple["ManagedConnectorSnapshot", ...]:
        """Evaluate managed connector health using optional asynchronous probes."""

        manifest = {snapshot.name: snapshot for snapshot in self.managed_manifest()}
        probe_mapping = probes or self._default_connectivity_probes()

        async def _evaluate(
            name: str, snapshot: ManagedConnectorSnapshot
        ) -> ManagedConnectorSnapshot:
            probe = probe_mapping.get(name)
            if probe is None:
                return snapshot

            def _format_probe_error(error: ConnectivityProbeError) -> str:
                base_message = str(error) or f"{name} probe failed"
                cause = error.__cause__ or error.__context__
                if cause:
                    cause_text = str(cause).strip()
                    if cause_text:
                        return f"{base_message}: {cause_text}"
                return base_message

            try:
                result = probe()
            except ConnectivityProbeError as exc:
                message = _format_probe_error(exc)
                _log_probe_failure(name, exc, message=message)
                return snapshot.with_health(False, message)
            except Exception as exc:  # pragma: no cover - unexpected guardrail
                logger.exception("Unexpected connectivity probe failure for %s", name, exc_info=exc)
                raise

            if asyncio.iscoroutine(result):
                try:
                    awaited = await asyncio.wait_for(result, timeout=timeout)
                except ConnectivityProbeError as exc:
                    message = _format_probe_error(exc)
                    _log_probe_failure(name, exc, message=message)
                    return snapshot.with_health(False, message)
                except asyncio.TimeoutError as exc:
                    message = f"{name} connectivity probe timed out after {timeout:.1f}s"
                    _log_probe_failure(name, exc, message=message)
                    return snapshot.with_health(False, message)
                except Exception as exc:  # pragma: no cover - unexpected guardrail
                    logger.exception(
                        "Unexpected async connectivity probe failure for %s",
                        name,
                        exc_info=exc,
                    )
                    raise
                healthy = awaited
            else:
                healthy = result

            return snapshot.with_health(bool(healthy))

        evaluated: list[ManagedConnectorSnapshot] = []
        for name, snapshot in manifest.items():
            evaluated.append(await _evaluate(name, snapshot))

        return tuple(evaluated)

    async def run_failover_drill(
        self,
        results: Mapping[str, TimescaleIngestResult],
        *,
        fail_dimensions: Sequence[str] | None = None,
        scenario: str | None = None,
        failover_policy: IngestFailoverPolicy | None = None,
        fallback: Callable[[], Awaitable[None] | None] | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> FailoverDrillSnapshot:
        """Execute a Timescale failover drill using provisioned services."""

        if not results:
            raise ValueError("Timescale ingest results are required for a failover drill")

        settings = self.config.failover_drill
        if settings is None or not settings.enabled:
            raise RuntimeError("Timescale failover drills are not enabled for this ingest slice")

        def _normalise_dimensions(candidates: Sequence[str]) -> tuple[str, ...]:
            seen: set[str] = set()
            ordered: list[str] = []
            for entry in candidates:
                cleaned = str(entry).strip()
                if not cleaned or cleaned in seen:
                    continue
                seen.add(cleaned)
                ordered.append(cleaned)
            return tuple(ordered)

        dimensions = _normalise_dimensions(fail_dimensions or settings.dimensions)
        if not dimensions:
            plan_dimensions = _plan_dimensions(self.config)
            if plan_dimensions:
                dimensions = _normalise_dimensions(plan_dimensions)
        if not dimensions:
            dimensions = _normalise_dimensions(results.keys())
        if not dimensions:
            raise ValueError("Unable to determine failover drill dimensions")

        scenario_label = scenario or settings.label or "timescale_failover"

        manifest_snapshots = [snapshot.as_dict() for snapshot in self.managed_manifest()]
        summary_payload = self.summary()

        drill_metadata: dict[str, object] = {
            "requested_dimensions": list(dimensions),
            "managed_manifest": manifest_snapshots,
            "services_summary": dict(summary_payload),
            "failover_drill": settings.to_metadata(),
        }
        if self.redis_settings is not None and self.redis_settings.configured:
            drill_metadata.setdefault(
                "redis_summary",
                self.redis_settings.summary(redacted=True),
            )
        if self.kafka_settings is not None and self.kafka_settings.configured:
            drill_metadata.setdefault(
                "kafka_summary",
                self.kafka_settings.summary(redacted=True),
            )
        drill_metadata.setdefault(
            "timescale_summary",
            {
                "application_name": self.config.timescale_settings.application_name,
                "url": _redacted_url(self.config.timescale_settings.url),
            },
        )

        if metadata:
            drill_metadata.update({str(key): value for key, value in metadata.items()})

        fallback_callable = fallback if settings.run_fallback else None

        return await execute_failover_drill(
            plan=self.config.plan,
            results=results,
            fail_dimensions=dimensions,
            scenario=scenario_label,
            failover_policy=failover_policy,
            fallback=fallback_callable,
            metadata=drill_metadata,
        )

    def _default_connectivity_probes(self) -> Mapping[str, ProbeCallable]:
        """Build connectivity probes for provisioned managed services."""

        probes: dict[str, ProbeCallable] = {}

        def _timescale_probe() -> bool:
            settings = self.config.timescale_settings
            engine: object | None = None
            try:
                engine = settings.create_engine()
                with engine.connect() as conn:
                    conn.execute(text("SELECT 1"))
            except _EXPECTED_TIMESCALE_ERRORS as exc:
                _log_probe_failure("timescale", exc)
                raise ConnectivityProbeError("timescale probe failed") from exc
            finally:
                if engine is not None:
                    with contextlib.suppress(Exception):
                        engine.dispose()

            return True

        probes["timescale"] = _timescale_probe

        if self.redis_cache is not None:
            def _redis_probe() -> bool:
                client = getattr(self.redis_cache, "raw_client", None)
                if client is None:
                    return False
                ping = getattr(client, "ping", None)
                if not callable(ping):
                    logger.debug("Redis client does not expose ping(); assuming unhealthy")
                    return False
                try:
                    return bool(ping())
                except _EXPECTED_REDIS_ERRORS as exc:
                    _log_probe_failure("redis", exc)
                    raise ConnectivityProbeError("redis probe failed") from exc

            probes["redis"] = _redis_probe

        if self.kafka_consumer is not None:
            def _kafka_probe() -> bool:
                try:
                    return bool(self.kafka_consumer.ping())
                except _EXPECTED_KAFKA_ERRORS as exc:
                    _log_probe_failure("kafka", exc)
                    raise ConnectivityProbeError("kafka probe failed") from exc

            probes["kafka"] = _kafka_probe

        return probes


@dataclass(slots=True)
class InstitutionalIngestProvisioner:
    """Construct institutional ingest services from configuration objects."""

    ingest_config: InstitutionalIngestConfig
    redis_settings: RedisConnectionSettings | None = None
    redis_policy: RedisCachePolicy = field(
        default_factory=RedisCachePolicy.institutional_defaults
    )
    kafka_mapping: Mapping[str, str] | None = None

    def provision(
        self,
        *,
        run_ingest: RunCallback,
        event_bus: object,
        task_supervisor: TaskSupervisor,
        redis_client_factory: Callable[[RedisConnectionSettings], object] | None = None,
        kafka_consumer_factory: KafkaConsumerFactory | None = None,
        kafka_deserializer: Callable[[bytes | bytearray | str], Mapping[str, object]]
        | None = None,
    ) -> InstitutionalIngestServices:
        """Instantiate connectors and wrap them in a service bundle."""

        if task_supervisor is None:
            raise ValueError("task_supervisor must be provided for institutional ingest")

        schedule = self.ingest_config.schedule or default_institutional_schedule()

        scheduler_metadata: MutableMapping[str, object] = {
            "component": "timescale_ingest.scheduler",
            "timescale_configured": self.ingest_config.timescale_settings.configured,
            "dimensions": list(_plan_dimensions(self.ingest_config)),
            "interval_seconds": schedule.interval_seconds,
        }
        if self.ingest_config.failover_drill is not None:
            scheduler_metadata["failover_drill"] = (
                self.ingest_config.failover_drill.to_metadata()
            )

        scheduler = TimescaleIngestScheduler(
            schedule=schedule,
            run_callback=run_ingest,
            task_supervisor=task_supervisor,
            task_metadata=scheduler_metadata,
        )

        redis_cache: ManagedRedisCache | None = None
        redis_settings = self.redis_settings or self.ingest_config.redis_settings
        if redis_settings is not None and redis_settings.configured:
            client: object | None
            if redis_client_factory is not None:
                client = redis_client_factory(redis_settings)
            else:
                client = configure_redis_client(redis_settings)
                if client is None:
                    logger.warning(
                        "Redis configuration present (%s) but client could not be created; skipping managed cache",
                        redis_settings.summary(redacted=True),
                    )
            if client is not None:
                redis_cache = ManagedRedisCache(client, self.redis_policy)

        kafka_consumer: KafkaIngestEventConsumer | None = None
        kafka_settings = self.ingest_config.kafka_settings
        kafka_topics: tuple[str, ...] = tuple()
        kafka_mapping: Mapping[str, str] | None = None
        streaming_enabled = self.ingest_config.enable_streaming
        if kafka_settings and kafka_settings.configured:
            kafka_mapping = self._resolved_kafka_mapping()
            kafka_topics = _extract_kafka_topics(kafka_mapping)
            if streaming_enabled:
                kafka_consumer = create_ingest_event_consumer(
                    kafka_settings,
                    kafka_mapping or None,
                    event_bus=event_bus,
                    consumer_factory=kafka_consumer_factory,
                    deserializer=kafka_deserializer,
                )

        kafka_metadata = _build_kafka_metadata(
            kafka_mapping or {},
            dimensions=_plan_dimensions(self.ingest_config),
            kafka_topics=kafka_topics,
            provisioned=kafka_consumer is not None,
            streaming_enabled=streaming_enabled,
        )
        if kafka_consumer is None:
            # Ensure provisioned flag reflects the runtime reality even when
            # configuration requested a consumer but creation failed upstream.
            kafka_metadata["provisioned"] = False

        return InstitutionalIngestServices(
            config=self.ingest_config,
            scheduler=scheduler,
            task_supervisor=task_supervisor,
            redis_settings=redis_settings,
            redis_cache=redis_cache,
            redis_policy=self.redis_policy,
            kafka_settings=kafka_settings,
            kafka_consumer=kafka_consumer,
            kafka_metadata=kafka_metadata,
        )

    def _resolved_kafka_mapping(self) -> dict[str, str]:
        """Merge provided Kafka mapping with institutional defaults."""

        mapping = {str(key): str(value) for key, value in (self.kafka_mapping or {}).items()}
        consumer_topics_raw = mapping.get("KAFKA_INGEST_CONSUMER_TOPICS", "").strip()
        topic_map, default_topic = ingest_topic_config_from_mapping(mapping)
        has_topics = bool(consumer_topics_raw or topic_map or (default_topic and default_topic.strip()))
        if not has_topics:
            fallback_topic = mapping.get("KAFKA_INGEST_DEFAULT_TOPIC", "telemetry.ingest").strip()
            if not fallback_topic:
                fallback_topic = "telemetry.ingest"
            mapping["KAFKA_INGEST_CONSUMER_TOPICS"] = fallback_topic
        return mapping

    def resolved_kafka_mapping(self) -> dict[str, str]:
        """Expose the resolved Kafka mapping with institutional fallbacks applied."""

        return self._resolved_kafka_mapping()


__all__ = [
    "ConnectivityProbeError",
    "InstitutionalIngestProvisioner",
    "InstitutionalIngestServices",
    "default_institutional_schedule",
]


def _build_managed_manifest(
    config: InstitutionalIngestConfig,
    *,
    scheduler_running: bool,
    redis_settings: RedisConnectionSettings | None,
    redis_cache: ManagedRedisCache | None,
    redis_policy: RedisCachePolicy | None,
    kafka_settings: KafkaConnectionSettings | None,
    kafka_consumer: KafkaIngestEventConsumer | None,
    kafka_supervised: bool,
    kafka_metadata: Mapping[str, object] | None,
    kafka_topics: Sequence[str] | None,
) -> tuple["ManagedConnectorSnapshot", ...]:
    timescale_metadata = {
        "application_name": config.timescale_settings.application_name,
        "url": _redacted_url(config.timescale_settings.url),
        "dimensions": list(_plan_dimensions(config)),
    }

    redis_summary: str | None = None
    if redis_settings is not None:
        try:
            redis_summary = redis_settings.summary(redacted=True)
        except RuntimeError:  # pragma: no cover - redis optional dependency
            redis_summary = "Redis client unavailable"

    redis_backing: str | None = None
    if redis_cache is not None:
        raw_client = getattr(redis_cache, "raw_client", None)
        if raw_client is not None:
            redis_backing = raw_client.__class__.__name__

    redis_policy_metadata = _policy_metadata(redis_policy)
    redis_metrics: Mapping[str, object] | None = None
    if redis_cache is not None:
        try:
            redis_metrics = redis_cache.metrics()
        except Exception:  # pragma: no cover - metrics retrieval is best-effort
            logger.debug(
                "Failed to collect Redis cache metrics for manifest", exc_info=True
            )
    prepared_redis_metrics = _prepare_redis_metrics(redis_metrics)

    kafka_topics_list: list[str] = []
    if kafka_topics is not None:
        kafka_topics_list = [str(topic) for topic in kafka_topics if str(topic).strip()]
    elif kafka_consumer is not None:
        kafka_topics_list = list(kafka_consumer.topics)

    kafka_snapshot_metadata: dict[str, object] = {
        "summary": kafka_settings.summary(redacted=True)
        if kafka_settings is not None
        else None,
        "bootstrap_servers": (
            kafka_settings.bootstrap_servers if kafka_settings is not None else None
        ),
        "topics": kafka_topics_list,
    }
    if kafka_metadata:
        kafka_snapshot_metadata.update(dict(kafka_metadata))

    timescale_snapshot = ManagedConnectorSnapshot(
        name="timescale",
        configured=config.timescale_settings.configured,
        supervised=scheduler_running,
        metadata=timescale_metadata,
    )

    redis_snapshot = ManagedConnectorSnapshot(
        name="redis",
        configured=bool(redis_settings and redis_settings.configured),
        supervised=False,
        metadata={
            "summary": redis_summary,
            "backing": redis_backing,
            "policy": redis_policy_metadata,
            "metrics": prepared_redis_metrics,
        },
    )

    kafka_snapshot = ManagedConnectorSnapshot(
        name="kafka",
        configured=bool(kafka_settings and kafka_settings.configured),
        supervised=kafka_supervised,
        metadata=kafka_snapshot_metadata,
    )

    return (timescale_snapshot, redis_snapshot, kafka_snapshot)


def plan_managed_manifest(
    config: InstitutionalIngestConfig,
    *,
    redis_settings: RedisConnectionSettings | None = None,
    kafka_mapping: Mapping[str, str] | None = None,
) -> tuple["ManagedConnectorSnapshot", ...]:
    """Return a manifest snapshot without instantiating managed connectors."""

    mapping = {str(k): str(v) for k, v in (kafka_mapping or {}).items()}
    resolved_redis_settings = redis_settings or config.redis_settings
    provisioner = InstitutionalIngestProvisioner(
        config,
        redis_settings=resolved_redis_settings,
        redis_policy=config.redis_policy,
        kafka_mapping=mapping,
    )
    resolved_mapping = provisioner.resolved_kafka_mapping()
    kafka_topics = _extract_kafka_topics(resolved_mapping)

    kafka_metadata: dict[str, object] = {
        **_build_kafka_metadata(
            resolved_mapping,
            dimensions=_plan_dimensions(config),
            kafka_topics=kafka_topics,
            provisioned=False,
            streaming_enabled=config.enable_streaming,
        ),
    }

    return _build_managed_manifest(
        config,
        scheduler_running=False,
        redis_settings=resolved_redis_settings,
        redis_cache=None,
        redis_policy=provisioner.redis_policy,
        kafka_settings=config.kafka_settings,
        kafka_consumer=None,
        kafka_supervised=False,
        kafka_metadata=kafka_metadata,
        kafka_topics=kafka_topics,
    )


@dataclass(frozen=True, slots=True)
class ManagedConnectorSnapshot:
    """Snapshot of a managed ingest connector, including optional health state."""

    name: str
    configured: bool
    supervised: bool
    metadata: Mapping[str, object]
    healthy: bool | None = None
    error: str | None = None

    def as_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "name": self.name,
            "configured": self.configured,
            "supervised": self.supervised,
            "metadata": dict(self.metadata),
        }
        if self.healthy is not None:
            payload["healthy"] = self.healthy
        if self.error:
            payload["error"] = self.error
        return payload

    def with_health(
        self,
        healthy: bool | None,
        error: str | None = None,
    ) -> "ManagedConnectorSnapshot":
        return ManagedConnectorSnapshot(
            name=self.name,
            configured=self.configured,
            supervised=self.supervised,
            metadata=self.metadata,
            healthy=healthy,
            error=error,
        )


__all__.append("ManagedConnectorSnapshot")
__all__.append("plan_managed_manifest")
