"""Production-grade orchestration helpers for the institutional ingest slice."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Callable, Mapping

from src.data_foundation.cache.redis_cache import RedisConnectionSettings
from src.data_foundation.persist.timescale import (
    TimescaleConnectionSettings,
    TimescaleIngestResult,
)
from src.data_foundation.streaming.kafka_stream import KafkaConsumerFactory
from src.runtime.task_supervisor import TaskSupervisor

from .configuration import InstitutionalIngestConfig
from .institutional_vertical import (
    InstitutionalIngestProvisioner,
    InstitutionalIngestServices,
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

    def __post_init__(self) -> None:
        if self.task_supervisor is None:
            raise ValueError("task_supervisor must be provided for production ingest")

        settings = self.ingest_config.timescale_settings
        if not isinstance(settings, TimescaleConnectionSettings):  # pragma: no cover - defensive
            raise TypeError("ingest_config.timescale_settings must be TimescaleConnectionSettings")

        self._orchestrator = self.orchestrator_factory(settings, self.event_publisher)
        self._provisioner = self.provisioner_factory(
            self.ingest_config, self.redis_settings, self.kafka_mapping
        )

    async def run_once(self) -> bool:
        """Execute a single Timescale ingest run in a worker thread."""

        services = self._ensure_services()
        if services is None:
            self._last_error = self.ingest_config.reason or "Timescale ingest disabled"
            return False

        async def _run() -> dict[str, TimescaleIngestResult]:
            return await asyncio.to_thread(self._orchestrator.run, plan=self.ingest_config.plan)

        try:
            results = await _run()
        except Exception as exc:
            self._last_error = str(exc)
            logger.exception("Timescale ingest run failed")
            return False

        self._last_results = results
        self._last_run_at = datetime.now(tz=UTC)
        self._last_error = None
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


__all__ = ["ProductionIngestSlice"]
