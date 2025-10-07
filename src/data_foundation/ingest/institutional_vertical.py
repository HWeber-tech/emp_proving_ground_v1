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
from typing import Awaitable, Callable, Mapping, MutableMapping

from sqlalchemy import text

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
from src.data_foundation.streaming.kafka_stream import (
    KafkaConnectionSettings,
    KafkaIngestEventConsumer,
    KafkaConsumerFactory,
    create_ingest_event_consumer,
    ingest_topic_config_from_mapping,
)
from src.runtime.task_supervisor import TaskSupervisor


logger = logging.getLogger(__name__)


ProbeCallable = Callable[[], Awaitable[bool] | bool]


DEFAULT_INTERVAL_SECONDS = 3_600.0
DEFAULT_JITTER_SECONDS = 120.0
DEFAULT_MAX_FAILURES = 3


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


@dataclass(slots=True)
class InstitutionalIngestServices:
    """Runtime objects bound to the managed ingest vertical."""

    config: InstitutionalIngestConfig
    scheduler: TimescaleIngestScheduler
    task_supervisor: TaskSupervisor
    redis_settings: RedisConnectionSettings | None = None
    redis_cache: ManagedRedisCache | None = None
    kafka_settings: KafkaConnectionSettings | None = None
    kafka_consumer: KafkaIngestEventConsumer | None = None
    kafka_task_name: str = "timescale-ingest-kafka-bridge"
    kafka_metadata: Mapping[str, object] = field(default_factory=dict)

    _kafka_task: asyncio.Task[None] | None = field(init=False, default=None)
    _kafka_stop_event: asyncio.Event | None = field(init=False, default=None)

    def start(self) -> None:
        """Start supervised components (scheduler plus Kafka bridge)."""

        self.scheduler.start()
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

    def managed_manifest(self) -> tuple["ManagedConnectorSnapshot", ...]:
        """Return structured metadata for provisioned managed services."""

        timescale_snapshot = ManagedConnectorSnapshot(
            name="timescale",
            configured=self.config.timescale_settings.configured,
            supervised=self.scheduler.running,
            metadata={
                "application_name": self.config.timescale_settings.application_name,
                "url": _redacted_url(self.config.timescale_settings.url),
                "dimensions": list(_plan_dimensions(self.config)),
            },
        )

        redis_backing: str | None = None
        if self.redis_cache is not None:
            raw_client = getattr(self.redis_cache, "raw_client", None)
            if raw_client is not None:
                redis_backing = raw_client.__class__.__name__

        redis_snapshot = ManagedConnectorSnapshot(
            name="redis",
            configured=bool(self.redis_settings and self.redis_settings.configured),
            supervised=False,
            metadata={
                "summary": self.redis_settings.summary(redacted=True)
                if self.redis_settings and self.redis_settings.configured
                else None,
                "backing": redis_backing,
            },
        )

        kafka_snapshot = ManagedConnectorSnapshot(
            name="kafka",
            configured=bool(self.kafka_settings and self.kafka_settings.configured),
            supervised=self._kafka_task is not None,
            metadata={
                "bootstrap_servers": (
                    self.kafka_settings.bootstrap_servers if self.kafka_settings else None
                ),
                "topics": list(self.kafka_consumer.topics)
                if self.kafka_consumer is not None
                else [],
            },
        )

        return (timescale_snapshot, redis_snapshot, kafka_snapshot)

    def summary(self) -> Mapping[str, object]:
        """Summarise managed connectors for observability dashboards."""

        schedule = self.scheduler
        schedule_state = schedule.state()

        redis_summary: str | None = None
        redis_backing: str | None = None
        if self.redis_settings is not None and self.redis_settings.configured:
            redis_summary = self.redis_settings.summary(redacted=True)
        if self.redis_cache is not None:
            raw_client = getattr(self.redis_cache, "raw_client", None)
            if raw_client is not None:
                redis_backing = raw_client.__class__.__name__

        kafka_summary: str | None = None
        if self.kafka_settings is not None and self.kafka_settings.configured:
            kafka_summary = self.kafka_settings.summary(redacted=True)

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
            "kafka": kafka_summary,
            "kafka_topics": list(self.kafka_consumer.topics)
            if self.kafka_consumer is not None
            else [],
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

        async def _evaluate(name: str, snapshot: ManagedConnectorSnapshot) -> ManagedConnectorSnapshot:
            probe = probe_mapping.get(name)
            if probe is None:
                return snapshot
            try:
                result = probe()
                if asyncio.iscoroutine(result):
                    healthy = await asyncio.wait_for(result, timeout=timeout)
                else:
                    healthy = bool(result)
                return snapshot.with_health(bool(healthy))
            except Exception:  # pragma: no cover - defensive logging
                logger.exception("Connectivity probe for %s failed", name)
                return snapshot.with_health(False)

        evaluated: list[ManagedConnectorSnapshot] = []
        for name, snapshot in manifest.items():
            evaluated.append(await _evaluate(name, snapshot))

        return tuple(evaluated)

    def _default_connectivity_probes(self) -> Mapping[str, ProbeCallable]:
        """Build connectivity probes for provisioned managed services."""

        probes: dict[str, ProbeCallable] = {}

        def _timescale_probe() -> bool:
            settings = self.config.timescale_settings
            engine = settings.create_engine()
            try:
                with engine.connect() as conn:
                    conn.execute(text("SELECT 1"))
                return True
            except Exception:  # pragma: no cover - defensive logging
                logger.exception("Timescale connectivity probe failed")
                return False
            finally:
                engine.dispose()

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
                except Exception:  # pragma: no cover - defensive logging
                    logger.exception("Redis connectivity probe failed")
                    return False

            probes["redis"] = _redis_probe

        if self.kafka_consumer is not None:
            def _kafka_probe() -> bool:
                try:
                    return bool(self.kafka_consumer.ping())
                except Exception:  # pragma: no cover - defensive logging
                    logger.exception("Kafka connectivity probe failed")
                    return False

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
        redis_settings = self.redis_settings
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
        if kafka_settings and kafka_settings.configured:
            kafka_mapping = self._resolved_kafka_mapping()
            kafka_consumer = create_ingest_event_consumer(
                kafka_settings,
                kafka_mapping or None,
                event_bus=event_bus,
                consumer_factory=kafka_consumer_factory,
                deserializer=kafka_deserializer,
            )

        kafka_metadata: dict[str, object] = {
            "timescale_dimensions": list(_plan_dimensions(self.ingest_config)),
        }

        return InstitutionalIngestServices(
            config=self.ingest_config,
            scheduler=scheduler,
            task_supervisor=task_supervisor,
            redis_settings=redis_settings,
            redis_cache=redis_cache,
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


__all__ = [
    "InstitutionalIngestProvisioner",
    "InstitutionalIngestServices",
    "default_institutional_schedule",
]


@dataclass(frozen=True, slots=True)
class ManagedConnectorSnapshot:
    """Snapshot of a managed ingest connector, including optional health state."""

    name: str
    configured: bool
    supervised: bool
    metadata: Mapping[str, object]
    healthy: bool | None = None

    def as_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "name": self.name,
            "configured": self.configured,
            "supervised": self.supervised,
            "metadata": dict(self.metadata),
        }
        if self.healthy is not None:
            payload["healthy"] = self.healthy
        return payload

    def with_health(self, healthy: bool | None) -> "ManagedConnectorSnapshot":
        return ManagedConnectorSnapshot(
            name=self.name,
            configured=self.configured,
            supervised=self.supervised,
            metadata=self.metadata,
            healthy=healthy,
        )


__all__.append("ManagedConnectorSnapshot")
