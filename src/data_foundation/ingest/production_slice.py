"""Production-grade orchestration helpers for the institutional ingest slice."""

from __future__ import annotations

import asyncio
import logging
from uuid import uuid4
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Awaitable, Callable, Mapping

from src.data_foundation.cache.redis_cache import RedisConnectionSettings
from src.data_foundation.persist.timescale import (
    TimescaleConnectionSettings,
    TimescaleIngestJournal,
    TimescaleIngestResult,
    TimescaleIngestRunRecord,
)
from src.data_foundation.streaming.kafka_stream import KafkaConsumerFactory
from src.runtime.task_supervisor import TaskSupervisor

from .configuration import InstitutionalIngestConfig
from .institutional_vertical import (
    InstitutionalIngestProvisioner,
    InstitutionalIngestServices,
    ManagedConnectorSnapshot,
)
from .timescale_pipeline import (
    IngestResultPublisher,
    TimescaleBackboneOrchestrator,
)

logger = logging.getLogger(__name__)


OrchestratorFactory = Callable[
    [TimescaleConnectionSettings, IngestResultPublisher | None],
    TimescaleBackboneOrchestrator,
]

ProvisionerFactory = Callable[
    [InstitutionalIngestConfig, RedisConnectionSettings | None, Mapping[str, str] | None],
    InstitutionalIngestProvisioner,
]


def _default_orchestrator_factory(
    settings: TimescaleConnectionSettings,
    event_publisher: IngestResultPublisher | None,
) -> TimescaleBackboneOrchestrator:
    return TimescaleBackboneOrchestrator(settings, event_publisher=event_publisher)


def _default_provisioner_factory(
    config: InstitutionalIngestConfig,
    redis_settings: RedisConnectionSettings | None,
    kafka_mapping: Mapping[str, str] | None,
) -> InstitutionalIngestProvisioner:
    return InstitutionalIngestProvisioner(
        config,
        redis_settings=redis_settings,
        redis_policy=config.redis_policy,
        kafka_mapping=kafka_mapping,
    )


@dataclass(slots=True)
class ProductionIngestSlice:
    """Coordinate Timescale ingest runs with supervised async lifecycles."""

    ingest_config: InstitutionalIngestConfig
    event_bus: object
    task_supervisor: TaskSupervisor
    redis_settings: RedisConnectionSettings | None = None
    kafka_mapping: Mapping[str, str] | None = None
    redis_client_factory: Callable[[RedisConnectionSettings], object] | None = None
    kafka_consumer_factory: KafkaConsumerFactory | None = None
    kafka_deserializer: Callable[[bytes | bytearray | str], Mapping[str, object]] | None = None
    event_publisher: IngestResultPublisher | None = None
    orchestrator_factory: OrchestratorFactory = _default_orchestrator_factory
    provisioner_factory: ProvisionerFactory = _default_provisioner_factory
    _orchestrator: TimescaleBackboneOrchestrator = field(init=False)
    _provisioner: InstitutionalIngestProvisioner = field(init=False)
    _services: InstitutionalIngestServices | None = field(init=False, default=None)
    _last_results: dict[str, TimescaleIngestResult] | None = field(init=False, default=None)
    _last_run_at: datetime | None = field(init=False, default=None)
    _last_error: str | None = field(init=False, default=None)
    _run_lock: asyncio.Lock = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._run_lock = asyncio.Lock()
        if self.task_supervisor is None:
            raise ValueError("task_supervisor must be provided for production ingest")

        settings = self.ingest_config.timescale_settings
        if not isinstance(settings, TimescaleConnectionSettings):  # pragma: no cover - defensive
            raise TypeError("ingest_config.timescale_settings must be TimescaleConnectionSettings")

        self._orchestrator = self.orchestrator_factory(settings, self.event_publisher)
        self._provisioner = self.provisioner_factory(
            self.ingest_config,
            self.redis_settings or self.ingest_config.redis_settings,
            self.kafka_mapping,
        )

    async def run_once(self) -> bool:
        """Execute a single Timescale ingest run in a worker thread."""

        async with self._run_lock:
            services = self._ensure_services()
            if services is None:
                self._last_error = self.ingest_config.reason or "Timescale ingest disabled"
                return False

            async def _run() -> dict[str, TimescaleIngestResult]:
                return await asyncio.to_thread(
                    self._orchestrator.run,
                    plan=self.ingest_config.plan,
                )

            try:
                results = await _run()
            except Exception as exc:
                self._last_error = str(exc)
                logger.exception("Timescale ingest run failed")
                return False

            self._last_results = results
            self._last_run_at = datetime.now(tz=UTC)
            self._last_error = None
            if results:
                await asyncio.to_thread(
                    self._record_journal_entries,
                    results,
                    self._last_run_at,
                )
            self._invalidate_result_caches(services, results)
            return bool(results)

    def start(self) -> None:
        """Start supervised ingest components (scheduler, Kafka bridge)."""

        services = self._ensure_services()
        if services is None:
            return
        services.start()

    async def stop(self) -> None:
        """Stop background ingest components gracefully."""

        if self._services is None:
            return
        await self._services.stop()

    async def connectivity_report(
        self,
        *,
        probes: Mapping[str, Callable[[], Awaitable[bool] | bool]] | None = None,
        timeout: float = 5.0,
    ) -> tuple[ManagedConnectorSnapshot, ...]:
        """Evaluate managed connector health via the provisioned services."""

        services = self._ensure_services()
        if services is None:
            return tuple()
        return await services.connectivity_report(probes=probes, timeout=timeout)

    def summary(self) -> Mapping[str, object]:
        """Expose ingest execution and supervision metadata for dashboards."""

        services_summary: Mapping[str, object] | None = None
        if self._services is not None:
            services_summary = self._services.summary()

        results_summary: dict[str, Mapping[str, object]] = {}
        if self._last_results:
            results_summary = {
                key: result.as_dict() for key, result in self._last_results.items()
            }

        return {
            "should_run": self.ingest_config.should_run,
            "reason": self.ingest_config.reason,
            "services": services_summary,
            "last_results": results_summary,
            "last_run_at": self._last_run_at.isoformat() if self._last_run_at else None,
            "last_error": self._last_error,
        }

    def _ensure_services(self) -> InstitutionalIngestServices | None:
        if not self.ingest_config.should_run:
            return None

        if self._services is None:
            self._services = self._provisioner.provision(
                run_ingest=self._supervised_run,
                event_bus=self.event_bus,
                task_supervisor=self.task_supervisor,
                redis_client_factory=self.redis_client_factory,
                kafka_consumer_factory=self.kafka_consumer_factory,
                kafka_deserializer=self.kafka_deserializer,
            )
        return self._services

    async def _supervised_run(self) -> bool:
        try:
            return await self.run_once()
        except Exception:  # pragma: no cover - defensive logging handled in run_once
            logger.exception("Timescale ingest scheduler run raised an exception")
            return False

    def _invalidate_result_caches(
        self,
        services: InstitutionalIngestServices,
        results: Mapping[str, TimescaleIngestResult],
    ) -> None:
        """Purge Redis-backed Timescale query caches for refreshed dimensions."""

        cache = getattr(services, "redis_cache", None)
        if cache is None:
            return

        prefixes: list[str] = []
        for dimension, result in results.items():
            if not isinstance(result, TimescaleIngestResult):
                continue
            if result.rows_written <= 0:
                continue
            prefixes.append(f"timescale:{dimension}")

        if not prefixes:
            return

        invalidate = getattr(cache, "invalidate", None)
        if not callable(invalidate):  # pragma: no cover - ManagedRedisCache always implements
            logger.debug(
                "Redis cache for Timescale ingest does not expose invalidate(); skipping",
            )
            return

        try:
            removed = invalidate(prefixes)
        except Exception:  # pragma: no cover - defensive logging for redis failures
            logger.exception(
                "Timescale ingest failed to invalidate Redis cache for prefixes %s",
                prefixes,
            )
            return

        logger.debug(
            "Timescale ingest invalidated Redis cache prefixes %s (removed=%s)",
            prefixes,
            removed,
        )

    def _record_journal_entries(
        self,
        results: Mapping[str, TimescaleIngestResult],
        executed_at: datetime | None,
    ) -> None:
        """Persist ingest run metadata to the Timescale ingest journal."""

        if not results:
            return

        engine = None
        try:
            engine = self.ingest_config.timescale_settings.create_engine()
        except Exception:  # pragma: no cover - defensive logging
            logger.exception("Failed to create Timescale engine for ingest journal")
            return

        try:
            journal = TimescaleIngestJournal(engine)
            records: list[TimescaleIngestRunRecord] = []
            executed = executed_at or datetime.now(tz=UTC)
            for dimension, result in results.items():
                if not isinstance(result, TimescaleIngestResult):
                    continue
                metadata = self._build_journal_metadata(dimension)
                record = TimescaleIngestRunRecord(
                    run_id=str(uuid4()),
                    dimension=dimension,
                    status="ok" if result.rows_written else "skipped",
                    rows_written=result.rows_written,
                    freshness_seconds=result.freshness_seconds,
                    ingest_duration_seconds=result.ingest_duration_seconds,
                    executed_at=executed,
                    source=result.source,
                    symbols=result.symbols,
                    metadata=metadata,
                )
                records.append(record)

            if records:
                journal.record(records)
        except Exception:  # pragma: no cover - defensive logging
            logger.exception("Failed to record Timescale ingest journal from production slice")
        finally:
            if engine is not None:
                try:
                    engine.dispose()
                except Exception:  # pragma: no cover - defensive cleanup
                    logger.debug(
                        "Failed to dispose Timescale ingest journal engine",
                        exc_info=True,
                    )

    def _build_journal_metadata(self, dimension: str) -> Mapping[str, object]:
        metadata: dict[str, object] = {"trigger": "production_slice"}

        config_meta = self._normalise_metadata_mapping(self.ingest_config.metadata)
        if config_meta:
            metadata["config"] = config_meta

        plan_meta = self._plan_metadata_for_dimension(dimension)
        if plan_meta:
            metadata["plan"] = plan_meta

        schedule = self.ingest_config.schedule
        if schedule is not None:
            metadata["schedule"] = {
                "interval_seconds": schedule.interval_seconds,
                "jitter_seconds": schedule.jitter_seconds,
                "max_failures": schedule.max_failures,
            }

        if self.kafka_mapping:
            metadata["kafka_topics"] = sorted({str(topic) for topic in self.kafka_mapping.values()})

        return metadata

    def _plan_metadata_for_dimension(self, dimension: str) -> Mapping[str, object]:
        plan = self.ingest_config.plan
        if plan is None:
            return {}

        if dimension == "daily_bars" and plan.daily is not None:
            return {
                "source": plan.daily.source,
                "lookback_days": int(plan.daily.lookback_days),
                "symbols": plan.daily.normalised_symbols(),
            }

        if dimension == "intraday_trades" and plan.intraday is not None:
            return {
                "source": plan.intraday.source,
                "lookback_days": int(plan.intraday.lookback_days),
                "interval": plan.intraday.interval,
                "symbols": plan.intraday.normalised_symbols(),
            }

        if dimension == "macro_events" and plan.macro is not None:
            payload: dict[str, object] = {"source": plan.macro.source}
            if plan.macro.has_window():
                payload["window"] = {
                    "start": plan.macro.start,
                    "end": plan.macro.end,
                }
            events = plan.macro.events
            if events:
                payload["provided_events"] = len(events)
            return payload

        return {}

    @staticmethod
    def _normalise_metadata_mapping(
        mapping: Mapping[str, object] | None,
    ) -> Mapping[str, object]:
        if not mapping:
            return {}

        def _coerce(value: object) -> object:
            if value is None or isinstance(value, (str, int, float, bool)):
                return value
            if isinstance(value, Mapping):
                return {str(k): _coerce(v) for k, v in value.items()}
            if isinstance(value, (list, tuple, set)):
                return [_coerce(v) for v in value]
            return str(value)

        return {str(key): _coerce(val) for key, val in mapping.items()}


__all__ = ["ProductionIngestSlice"]
