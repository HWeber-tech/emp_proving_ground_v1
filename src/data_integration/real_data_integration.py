"""Institutional data backbone manager tying Timescale, Redis, and Kafka together."""

from __future__ import annotations

import asyncio
import logging
import re
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any, Callable, Mapping, Sequence

import pandas as pd
from sqlalchemy.engine import Engine

from src.core.market_data import MarketDataGateway
from src.data_foundation.cache.redis_cache import (
    ManagedRedisCache,
    RedisCachePolicy,
    RedisConnectionSettings,
    configure_redis_client,
    wrap_managed_cache,
)
from src.data_foundation.cache.timescale_query_cache import TimescaleQueryCache
from src.data_foundation.ingest.scheduler import IngestSchedule, TimescaleIngestScheduler
from src.data_foundation.ingest.timescale_pipeline import (
    TimescaleBackboneOrchestrator,
    TimescaleBackbonePlan,
)
from src.data_foundation.persist.timescale import TimescaleConnectionSettings, TimescaleIngestResult
from src.data_foundation.persist.timescale_reader import TimescaleQueryResult, TimescaleReader
from src.data_foundation.streaming.kafka_stream import (
    KafkaConnectionSettings,
    create_ingest_event_publisher,
)
from src.runtime.task_supervisor import TaskSupervisor

if TYPE_CHECKING:  # pragma: no cover - typing support only
    from src.data_foundation.streaming.kafka_stream import KafkaIngestEventPublisher
    from src.governance.system_config import SystemConfig

logger = logging.getLogger(__name__)

_PERIOD_PATTERN = re.compile(r"^(?P<value>\d+)(?P<unit>[smhdw])$", re.IGNORECASE)


def _now() -> datetime:
    return datetime.now(tz=UTC)


def _parse_period(period: str | None) -> timedelta | None:
    if not period:
        return None
    match = _PERIOD_PATTERN.match(period.strip())
    if match is None:
        return None
    value = int(match.group("value"))
    unit = match.group("unit").lower()
    mapping = {
        "s": timedelta(seconds=value),
        "m": timedelta(minutes=value),
        "h": timedelta(hours=value),
        "d": timedelta(days=value),
        "w": timedelta(weeks=value),
    }
    return mapping.get(unit)


def _coerce_datetime(value: object | None) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    try:
        converted = pd.to_datetime(value, utc=True)  # type: ignore[arg-type]
    except Exception:
        return None
    if isinstance(converted, pd.Series):
        if converted.empty:
            return None
        converted = converted.iloc[0]
    if isinstance(converted, pd.Timestamp):
        return converted.to_pydatetime()
    if isinstance(converted, datetime):
        return converted if converted.tzinfo else converted.replace(tzinfo=UTC)
    return None


def _normalise_dimension(interval: str | None, source: str | None) -> str:
    if source:
        src = source.strip().lower()
        if "intraday" in src or src in {"trade", "trades"}:
            return "intraday"
        if "daily" in src or src == "bars":
            return "daily"
    if interval is None:
        return "daily"
    normalised = interval.strip().lower()
    if normalised in {"", "1d", "1day", "1w", "1wk", "daily"}:
        return "daily"
    return "intraday"


class RealDataManager(MarketDataGateway):
    """Coordinate Timescale storage, Redis caching, and Kafka streaming."""

    def __init__(
        self,
        *,
        system_config: "SystemConfig" | None = None,
        extras: Mapping[str, str] | None = None,
        timescale_settings: TimescaleConnectionSettings | None = None,
        redis_settings: RedisConnectionSettings | None = None,
        kafka_settings: KafkaConnectionSettings | None = None,
        cache_policy: RedisCachePolicy | None = None,
        managed_cache: ManagedRedisCache | None = None,
        ingest_publisher: "KafkaIngestEventPublisher" | None = None,
        engine: Engine | None = None,
        task_supervisor: TaskSupervisor | None = None,
        auto_close_engine: bool | None = None,
    ) -> None:
        if system_config is not None and not extras and isinstance(system_config.extras, dict):
            extras = system_config.extras

        payload = {str(k): str(v) for k, v in (extras or {}).items()}

        self._timescale_settings = timescale_settings or TimescaleConnectionSettings.from_mapping(
            payload
        )
        self._redis_settings = redis_settings or RedisConnectionSettings.from_mapping(payload)
        self._cache_policy = cache_policy or RedisCachePolicy.from_mapping(payload)
        self._kafka_settings = kafka_settings or KafkaConnectionSettings.from_mapping(payload)
        self._task_supervisor = task_supervisor

        if engine is not None:
            self._engine = engine
            self._owns_engine = auto_close_engine if auto_close_engine is not None else False
        else:
            self._engine = self._timescale_settings.create_engine()
            self._owns_engine = True if auto_close_engine is None else auto_close_engine

        self._reader = TimescaleReader(self._engine)

        if managed_cache is not None:
            self._cache = managed_cache
            self._redis_client = managed_cache
            self._owns_redis = False
        else:
            redis_client = configure_redis_client(self._redis_settings, ping=False)
            self._redis_client = redis_client
            self._owns_redis = redis_client is not None
            self._cache = wrap_managed_cache(
                redis_client,
                policy=self._cache_policy,
                bootstrap=redis_client is None,
            )

        self._query_cache = TimescaleQueryCache(self._reader, self._cache)

        if ingest_publisher is not None:
            self._kafka_publisher = ingest_publisher
        else:
            self._kafka_publisher = create_ingest_event_publisher(
                self._kafka_settings,
                payload,
            )

        self._extras = payload
        self._ingest_scheduler: TimescaleIngestScheduler | None = None
        self._shutdown_lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # MarketDataGateway interface
    # ------------------------------------------------------------------

    def fetch_data(
        self,
        symbol: str,
        period: str | None = None,
        interval: str | None = None,
        start: str | datetime | None = None,
        end: str | datetime | None = None,
    ) -> pd.DataFrame:
        dimension = _normalise_dimension(interval, None)
        symbol = str(symbol).strip()
        if not symbol:
            raise ValueError("symbol must not be empty")

        end_dt = _coerce_datetime(end)
        start_dt = _coerce_datetime(start)
        if start_dt is None:
            delta = _parse_period(period)
            if delta is not None and end_dt is not None:
                start_dt = end_dt - delta

        result = self._fetch_timescale(dimension, symbol, start_dt, end_dt)
        frame = self._normalise_frame(result, dimension)
        if not frame.empty and "symbol" in frame.columns:
            frame = frame.loc[frame["symbol"].astype(str) == symbol]
        return frame.reset_index(drop=True)

    async def get_market_data(
        self,
        symbol: str,
        period: str | None = None,
        interval: str | None = None,
        start: str | datetime | None = None,
        end: str | datetime | None = None,
    ) -> pd.DataFrame:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            self.fetch_data,
            symbol,
            period,
            interval,
            start,
            end,
        )

    # ------------------------------------------------------------------
    # Ingest and caching helpers
    # ------------------------------------------------------------------

    def run_ingest_plan(
        self,
        plan: TimescaleBackbonePlan,
        *,
        fetch_daily: Callable[[list[str], int], pd.DataFrame] | None = None,
        fetch_intraday: Callable[[list[str], int, str], pd.DataFrame] | None = None,
        fetch_macro: Callable[[str, str], Sequence[Mapping[str, object]]] | None = None,
    ) -> dict[str, TimescaleIngestResult]:
        orchestrator = TimescaleBackboneOrchestrator(
            self._timescale_settings,
            event_publisher=self._kafka_publisher,
        )
        resolved_daily = fetch_daily or (lambda symbols, lookback: pd.DataFrame())
        resolved_intraday = (
            fetch_intraday
            or (lambda symbols, lookback, interval: pd.DataFrame())
        )
        resolved_macro = fetch_macro or (lambda start, end: [])

        results = orchestrator.run(
            plan=plan,
            fetch_daily=resolved_daily,
            fetch_intraday=resolved_intraday,
            fetch_macro=resolved_macro,
        )
        self._invalidate_cache()
        return results

    def cache_metrics(self, *, reset: bool = False) -> Mapping[str, int | str]:
        return self._cache.metrics(reset=reset)

    def start_ingest_scheduler(
        self,
        plan_factory: Callable[[], TimescaleBackbonePlan],
        schedule: IngestSchedule,
        *,
        fetch_daily: Callable[[list[str], int], pd.DataFrame] | None = None,
        fetch_intraday: Callable[[list[str], int, str], pd.DataFrame] | None = None,
        fetch_macro: Callable[[str, str], Sequence[Mapping[str, object]]] | None = None,
        metadata: Mapping[str, object] | None = None,
        task_supervisor: TaskSupervisor | None = None,
    ) -> TimescaleIngestScheduler:
        if self._ingest_scheduler is not None and self._ingest_scheduler.running:
            return self._ingest_scheduler

        supervisor = task_supervisor or self._task_supervisor
        if supervisor is None:
            raise RuntimeError("TaskSupervisor required to start ingest scheduler")

        async def _run_once() -> bool:
            plan = plan_factory()
            if plan.is_empty():
                logger.debug("Ingest scheduler invoked with empty plan; skipping run")
                return True

            results = await asyncio.to_thread(
                self.run_ingest_plan,
                plan,
                fetch_daily=fetch_daily,
                fetch_intraday=fetch_intraday,
                fetch_macro=fetch_macro,
            )
            return any(result.rows_written > 0 for result in results.values())

        scheduler = TimescaleIngestScheduler(
            schedule=schedule,
            run_callback=_run_once,
            task_supervisor=supervisor,
            task_metadata=metadata,
        )
        scheduler.start()
        self._ingest_scheduler = scheduler
        return scheduler

    async def stop_ingest_scheduler(self) -> None:
        if self._ingest_scheduler is None:
            return
        scheduler = self._ingest_scheduler
        self._ingest_scheduler = None
        await scheduler.stop()

    # ------------------------------------------------------------------
    # Resource management
    # ------------------------------------------------------------------

    async def shutdown(self) -> None:
        async with self._shutdown_lock:
            await self.stop_ingest_scheduler()
            if self._owns_engine and hasattr(self._engine, "dispose"):
                self._engine.dispose()
                self._owns_engine = False
            if self._owns_redis and self._redis_client is not None:
                close = getattr(self._redis_client, "close", None)
                if callable(close):
                    try:
                        close()
                    except Exception:  # pragma: no cover - best effort cleanup
                        logger.debug("Redis close failed during shutdown", exc_info=True)
                disconnect = getattr(self._redis_client, "disconnect", None)
                if callable(disconnect):
                    try:
                        disconnect()
                    except Exception:  # pragma: no cover
                        logger.debug("Redis disconnect failed during shutdown", exc_info=True)
                self._owns_redis = False

    def close(self) -> None:
        if self._ingest_scheduler is not None and self._ingest_scheduler.running:
            raise RuntimeError("Ingest scheduler still running; use await shutdown() instead")
        if self._owns_engine and hasattr(self._engine, "dispose"):
            self._engine.dispose()
            self._owns_engine = False
        if self._owns_redis and self._redis_client is not None:
            close = getattr(self._redis_client, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:  # pragma: no cover
                    logger.debug("Redis close failed during close()", exc_info=True)
            self._owns_redis = False

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def reader(self) -> TimescaleReader:
        return self._reader

    @property
    def cache(self) -> ManagedRedisCache:
        return self._cache

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fetch_timescale(
        self,
        dimension: str,
        symbol: str,
        start: datetime | None,
        end: datetime | None,
    ) -> TimescaleQueryResult:
        symbols = [symbol]
        limit: int | None = None
        if dimension == "daily":
            return self._query_cache.fetch_daily_bars(
                symbols=symbols,
                start=start,
                end=end,
                limit=limit,
            )
        if dimension == "intraday":
            return self._query_cache.fetch_intraday_trades(
                symbols=symbols,
                start=start,
                end=end,
                limit=limit,
            )
        raise ValueError(f"Unsupported dimension {dimension}")

    def _normalise_frame(
        self,
        result: TimescaleQueryResult,
        dimension: str,
    ) -> pd.DataFrame:
        frame = result.frame.copy()
        if frame.empty:
            return frame
        if "ts" in frame.columns:
            frame = frame.rename(columns={"ts": "timestamp"})
        frame.sort_values("timestamp", inplace=True)
        if "timestamp" in frame.columns:
            frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
        if dimension == "intraday" and "size" in frame.columns:
            frame = frame.rename(columns={"size": "volume"})
        if "volume" in frame.columns:
            frame["volume"] = pd.to_numeric(frame["volume"], errors="coerce").astype(float)
        if "price" in frame.columns:
            frame["price"] = pd.to_numeric(frame["price"], errors="coerce").astype(float)
        return frame

    def _invalidate_cache(self) -> None:
        try:
            self._cache.invalidate(("timescale:",))
        except Exception:  # pragma: no cover - cache failures should not break ingest
            logger.debug("Timescale cache invalidation failed", exc_info=True)
