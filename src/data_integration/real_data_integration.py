"""Institutional data backbone manager tying Timescale, Redis, and Kafka together."""

from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any, Callable, Mapping, Sequence

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine
from sqlalchemy.engine.url import make_url

from src.core.market_data import MarketDataGateway
from src.data_foundation.cache.redis_cache import (
    InMemoryRedis,
    ManagedRedisCache,
    RedisCachePolicy,
    RedisConnectionSettings,
    configure_redis_client,
    wrap_managed_cache,
)
from src.data_foundation.cache.timescale_query_cache import TimescaleQueryCache
from src.data_foundation.ingest import timescale_pipeline as ingest_pipeline
from src.data_foundation.ingest.fred_calendar import fetch_fred_calendar
from src.data_foundation.ingest.scheduler import IngestSchedule, TimescaleIngestScheduler
from src.data_foundation.ingest.timescale_pipeline import (
    DailyBarIngestPlan,
    IntradayTradeIngestPlan,
    MacroEventIngestPlan,
    TimescaleBackboneOrchestrator,
    TimescaleBackbonePlan,
)
from src.data_foundation.persist.timescale import TimescaleConnectionSettings, TimescaleIngestResult
from src.data_foundation.persist.timescale_reader import TimescaleQueryResult, TimescaleReader
from src.data_foundation.streaming.kafka_stream import (
    KafkaConnectionSettings,
    KafkaIngestEventPublisher,
    create_ingest_event_publisher,
)
from src.runtime.task_supervisor import TaskSupervisor

if TYPE_CHECKING:  # pragma: no cover - typing support only
    from src.governance.system_config import SystemConfig

logger = logging.getLogger(__name__)

_PERIOD_PATTERN = re.compile(r"^(?P<value>\d+)(?P<unit>[smhdw])$", re.IGNORECASE)
_DAILY_INTERVALS = {
    "",
    "1d",
    "1day",
    "day",
    "daily",
    "5d",
    "1w",
    "1wk",
    "1week",
    "weekly",
    "1mo",
    "1mth",
    "1month",
    "monthly",
    "3mo",
    "3month",
    "6mo",
    "6month",
    "1y",
    "1yr",
    "year",
    "yearly",
    "annual",
    "annually",
}


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
    if normalised in _DAILY_INTERVALS:
        return "daily"
    return "intraday"


def _normalise_calendars(values: Sequence[str] | None) -> tuple[str, ...]:
    if not values:
        return tuple()
    seen: list[str] = []
    for value in values:
        token = str(value).strip()
        if token and token not in seen:
            seen.append(token)
    return tuple(seen)


@dataclass(slots=True, frozen=True)
class ConnectivityProbeSnapshot:
    """Fine-grained result describing a single connectivity probe."""

    name: str
    healthy: bool
    status: str
    latency_ms: float | None = None
    error: str | None = None
    details: Mapping[str, object] | None = None

    def as_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "name": self.name,
            "healthy": self.healthy,
            "status": self.status,
        }
        if self.latency_ms is not None:
            payload["latency_ms"] = round(float(self.latency_ms), 3)
        if self.error:
            payload["error"] = self.error
        if self.details:
            payload["details"] = dict(self.details)
        return payload


@dataclass(slots=True, frozen=True)
class BackboneConnectivityReport:
    """Structured status for core data backbone services."""

    timescale: bool
    redis: bool
    kafka: bool
    probes: tuple[ConnectivityProbeSnapshot, ...] = field(default_factory=tuple)

    def as_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "timescale": self.timescale,
            "redis": self.redis,
            "kafka": self.kafka,
        }
        payload["status"] = self.overall_status()
        if self.probes:
            payload["probes"] = [probe.as_dict() for probe in self.probes]
        return payload

    def probe_map(self) -> dict[str, ConnectivityProbeSnapshot]:
        return {probe.name: probe for probe in self.probes}

    def overall_status(self) -> str:
        """Aggregate probe statuses into a single severity string."""

        statuses = [probe.status for probe in self.probes if probe.status]
        if not statuses:
            return "unknown"
        for candidate in ("error", "off", "degraded"):
            if candidate in statuses:
                return candidate
        return "ok"


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
        ingest_publisher: KafkaIngestEventPublisher | None = None,
        engine: Engine | None = None,
        task_supervisor: TaskSupervisor | None = None,
        auto_close_engine: bool | None = None,
        require_timescale: bool | None = None,
        require_redis: bool | None = None,
        require_kafka: bool | None = None,
    ) -> None:
        if system_config is not None and not extras and isinstance(system_config.extras, dict):
            extras = system_config.extras

        payload = {str(k): str(v) for k, v in (extras or {}).items()}

        def _coerce_bool_flag(value: object | None, default: bool) -> bool:
            if isinstance(value, bool):
                return value
            if value is None:
                return default
            text = str(value).strip().lower().replace("-", "_")
            if text in {"1", "true", "yes", "y", "on"}:
                return True
            if text in {"0", "false", "no", "n", "off"}:
                return False
            return default

        mode_indicator: str | None = None
        if system_config is not None:
            mode_attr = getattr(system_config, "data_backbone_mode", None)
            if mode_attr is not None:
                mode_indicator = getattr(mode_attr, "value", str(mode_attr))
        if not mode_indicator:
            raw_mode = payload.get("DATA_BACKBONE_MODE")
            if raw_mode:
                mode_indicator = str(raw_mode)
        normalized_mode = (mode_indicator or "").strip().lower()
        timescale_default_required = normalized_mode == "institutional"
        redis_default_required = False
        kafka_default_required = False

        timescale_required = _coerce_bool_flag(
            require_timescale
            if require_timescale is not None
            else payload.get("DATA_BACKBONE_REQUIRE_TIMESCALE"),
            timescale_default_required,
        )
        redis_required = _coerce_bool_flag(
            require_redis
            if require_redis is not None
            else payload.get("DATA_BACKBONE_REQUIRE_REDIS"),
            redis_default_required,
        )
        kafka_required = _coerce_bool_flag(
            require_kafka
            if require_kafka is not None
            else payload.get("DATA_BACKBONE_REQUIRE_KAFKA"),
            kafka_default_required,
        )

        self._timescale_settings = timescale_settings or TimescaleConnectionSettings.from_mapping(
            payload
        )
        if timescale_required and not self._timescale_settings.configured:
            raise RuntimeError(
                "Timescale connection required but DATA_BACKBONE_REQUIRE_TIMESCALE is enabled"
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
            redis_client = configure_redis_client(self._redis_settings, ping=True)
            if redis_client is None and redis_required:
                raise RuntimeError(
                    "Redis connection required but redis client could not be created"
                )
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
        if kafka_required and self._kafka_settings and not self._kafka_settings.configured:
            raise RuntimeError(
                "Kafka connection required but configuration is incomplete"
            )
        if kafka_required and self._kafka_publisher is None:
            raise RuntimeError(
                "Kafka connection required but ingest publisher could not be created"
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

        delta = _parse_period(period)
        if delta is not None:
            if end_dt is None:
                end_dt = _now()
            if start_dt is None and end_dt is not None:
                start_dt = end_dt - delta

        if start_dt is not None and end_dt is not None and start_dt > end_dt:
            start_dt, end_dt = end_dt, start_dt

        result = self._fetch_timescale(dimension, symbol, start_dt, end_dt)
        frame = self._normalise_frame(result, dimension)

        if frame.empty and delta is not None and end is None:
            fallback_result = self._fetch_timescale(dimension, symbol, None, None)
            fallback_frame = self._normalise_frame(fallback_result, dimension)
            if fallback_frame.empty:
                return fallback_frame.reset_index(drop=True)

            latest_ts = pd.to_datetime(
                fallback_frame["timestamp"].max(),
                utc=True,
                errors="coerce",
            )
            if latest_ts is not None and not pd.isna(latest_ts):
                end_dt = latest_ts.to_pydatetime()
            else:
                end_dt = _now()
            start_dt = end_dt - delta
            mask = fallback_frame["timestamp"] >= start_dt
            frame = fallback_frame.loc[mask]

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

    def fetch_macro_events(
        self,
        *,
        calendars: Sequence[str] | None = None,
        start: str | datetime | None = None,
        end: str | datetime | None = None,
        limit: int | None = None,
    ) -> pd.DataFrame:
        resolved_calendars = _normalise_calendars(calendars)
        if not resolved_calendars:
            resolved_calendars = self._configured_macro_calendars()

        start_dt = _coerce_datetime(start)
        end_dt = _coerce_datetime(end)

        if start_dt is None and end_dt is None:
            default_window = self._macro_default_window()
            end_dt = _now()
            start_dt = end_dt - default_window
        elif start_dt is None and end_dt is not None:
            start_dt = end_dt - self._macro_default_window()
        elif end_dt is None and start_dt is not None:
            end_dt = start_dt + self._macro_default_window()

        fetch_limit = limit if isinstance(limit, int) and limit > 0 else self._macro_fetch_limit()

        result = self._query_cache.fetch_macro_events(
            calendars=list(resolved_calendars) if resolved_calendars else None,
            start=start_dt,
            end=end_dt,
            limit=fetch_limit,
        )
        frame = self._normalise_frame(result, "macro_events")
        return frame.reset_index(drop=True)

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
        resolved_daily = fetch_daily or ingest_pipeline.fetch_daily_bars
        resolved_intraday = (
            fetch_intraday
            if fetch_intraday is not None
            else ingest_pipeline.fetch_intraday_trades
        )
        resolved_macro = fetch_macro or fetch_fred_calendar

        results = orchestrator.run(
            plan=plan,
            fetch_daily=resolved_daily,
            fetch_intraday=resolved_intraday,
            fetch_macro=resolved_macro,
        )
        if results:
            self._invalidate_cache()
        return results

    def ingest_market_slice(
        self,
        *,
        symbols: Sequence[str],
        daily_lookback_days: int | None = 60,
        intraday_lookback_days: int | None = 2,
        intraday_interval: str = "1m",
        macro_start: str | None = None,
        macro_end: str | None = None,
        macro_events: Sequence[Mapping[str, object]] | None = None,
        source: str = "yahoo",
        macro_source: str = "fred",
        fetch_daily: Callable[[list[str], int], pd.DataFrame] | None = None,
        fetch_intraday: Callable[[list[str], int, str], pd.DataFrame] | None = None,
        fetch_macro: Callable[[str, str], Sequence[Mapping[str, object]]] | None = None,
    ) -> dict[str, TimescaleIngestResult]:
        """Run a production-style ingest slice using configured connectors.

        Parameters mirror the roadmap deliverable: historical bars are hydrated
        from Timescale, live-style trades are captured for recent windows, and
        macro releases can be attached either via explicit events or by pulling
        from FRED when a window is provided.  At least one dimension must be
        requested otherwise the ingest is skipped.
        """

        normalised_symbols: list[str] = []
        seen: set[str] = set()
        for candidate in symbols:
            token = str(candidate).strip().upper()
            if not token or token in seen:
                continue
            seen.add(token)
            normalised_symbols.append(token)

        if not normalised_symbols:
            raise ValueError("symbols must contain at least one non-empty entry")

        daily_plan: DailyBarIngestPlan | None = None
        if daily_lookback_days and daily_lookback_days > 0:
            daily_plan = DailyBarIngestPlan(
                symbols=tuple(normalised_symbols),
                lookback_days=int(daily_lookback_days),
                source=source,
            )

        intraday_plan: IntradayTradeIngestPlan | None = None
        if intraday_lookback_days and intraday_lookback_days > 0:
            intraday_plan = IntradayTradeIngestPlan(
                symbols=tuple(normalised_symbols),
                lookback_days=int(intraday_lookback_days),
                interval=str(intraday_interval),
                source=source,
            )

        macro_plan: MacroEventIngestPlan | None = None
        if macro_events is not None:
            macro_plan = MacroEventIngestPlan(
                events=tuple(macro_events),
                source=macro_source,
            )
        elif macro_start or macro_end:
            if not (macro_start and macro_end):
                raise ValueError("macro_start and macro_end must both be provided")
            macro_plan = MacroEventIngestPlan(
                start=str(macro_start),
                end=str(macro_end),
                source=macro_source,
            )

        if not any((daily_plan, intraday_plan, macro_plan)):
            raise ValueError("At least one ingest dimension must be requested")

        plan = TimescaleBackbonePlan(
            daily=daily_plan,
            intraday=intraday_plan,
            macro=macro_plan,
        )

        return self.run_ingest_plan(
            plan,
            fetch_daily=fetch_daily,
            fetch_intraday=fetch_intraday,
            fetch_macro=fetch_macro,
        )

    def cache_metrics(self, *, reset: bool = False) -> Mapping[str, int | str]:
        return self._cache.metrics(reset=reset)

    def connectivity_report(self) -> BackboneConnectivityReport:
        """Expose health indicators for Timescale, Redis, and Kafka connectors."""

        timescale_probe = self._check_timescale()
        redis_probe = self._check_redis()
        kafka_probe = self._check_kafka()

        probes = (timescale_probe, redis_probe, kafka_probe)

        return BackboneConnectivityReport(
            timescale=timescale_probe.healthy,
            redis=redis_probe.healthy,
            kafka=kafka_probe.healthy,
            probes=probes,
        )

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

    @property
    def engine(self) -> Engine:
        """Expose the underlying SQLAlchemy engine used for Timescale access."""

        return self._engine

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
        if dimension == "macro_events":
            for column in ("actual", "forecast", "previous"):
                if column in frame.columns:
                    frame[column] = pd.to_numeric(frame[column], errors="coerce")
        return frame

    def _invalidate_cache(self) -> None:
        try:
            self._cache.invalidate(("timescale:",))
        except Exception:  # pragma: no cover - cache failures should not break ingest
            logger.debug("Timescale cache invalidation failed", exc_info=True)

    def _masked_timescale_url(self) -> str:
        try:
            url_obj = make_url(self._timescale_settings.url)
            return url_obj.render_as_string(hide_password=True)
        except Exception:
            return self._timescale_settings.url

    def _check_timescale(self) -> ConnectivityProbeSnapshot:
        start = time.perf_counter()
        healthy = False
        error: str | None = None
        backend: str | None = None
        try:
            with self._engine.connect() as conn:
                conn.execute(text("SELECT 1"))
                healthy = True
        except Exception:
            logger.exception("Timescale connectivity probe failed")
            error = "Timescale connection failed"
        latency_ms = (time.perf_counter() - start) * 1000
        details: dict[str, object] = {
            "url": self._masked_timescale_url(),
            "configured": self._timescale_settings.configured,
        }
        try:
            backend = make_url(self._timescale_settings.url).get_backend_name()
        except Exception:  # pragma: no cover - defensive parsing guard
            backend = None
        if backend:
            details["backend"] = backend
        status = "ok" if healthy else "error"
        if healthy and not self._timescale_settings.is_postgres():
            status = "degraded"
            if backend:
                error = f"Timescale backend running on {backend}"
            else:
                error = "Timescale backend not using PostgreSQL/Timescale"
        return ConnectivityProbeSnapshot(
            name="timescale",
            healthy=healthy,
            status=status,
            latency_ms=latency_ms,
            error=error,
            details=details,
        )

    def _check_redis(self) -> ConnectivityProbeSnapshot:
        start = time.perf_counter()
        client = getattr(self._cache, "raw_client", None)
        details: dict[str, object] = {
            "namespace": self._cache_policy.namespace,
            "ttl_seconds": self._cache_policy.ttl_seconds,
            "max_keys": self._cache_policy.max_keys,
            "endpoint": self._redis_settings.summary(redacted=True),
        }
        if client is not None:
            details["client_class"] = type(client).__name__
        if client is None:
            latency_ms = (time.perf_counter() - start) * 1000
            return ConnectivityProbeSnapshot(
                name="redis",
                healthy=False,
                status="off",
                latency_ms=latency_ms,
                error="redis client not configured",
                details=details,
            )
        if isinstance(client, InMemoryRedis):
            latency_ms = (time.perf_counter() - start) * 1000
            return ConnectivityProbeSnapshot(
                name="redis",
                healthy=False,
                status="degraded",
                latency_ms=latency_ms,
                error="in-memory redis fallback active",
                details=details,
            )
        ping = getattr(client, "ping", None)
        if not callable(ping):
            latency_ms = (time.perf_counter() - start) * 1000
            return ConnectivityProbeSnapshot(
                name="redis",
                healthy=False,
                status="error",
                latency_ms=latency_ms,
                error="redis client missing ping()",
                details=details,
            )
        try:
            ping()
        except Exception:
            logger.exception("Redis connectivity probe failed")
            error = "redis ping failed"
            healthy = False
        else:
            error = None
            healthy = True
        latency_ms = (time.perf_counter() - start) * 1000
        status = "ok" if healthy else "error"
        return ConnectivityProbeSnapshot(
            name="redis",
            healthy=healthy,
            status=status,
            latency_ms=latency_ms,
            error=error,
            details=details,
        )

    def _check_kafka(self) -> ConnectivityProbeSnapshot:
        start = time.perf_counter()
        details: dict[str, object] = {
            "endpoint": self._kafka_settings.summary(redacted=True),
        }
        publisher = self._kafka_publisher
        if publisher is None:
            latency_ms = (time.perf_counter() - start) * 1000
            return ConnectivityProbeSnapshot(
                name="kafka",
                healthy=False,
                status="off",
                latency_ms=latency_ms,
                error="kafka publisher not configured",
                details=details,
            )
        topic_map = getattr(publisher, "_topic_map", None)
        if isinstance(topic_map, Mapping):
            details["topics"] = sorted(str(topic) for topic in topic_map.values())
        default_topic = getattr(publisher, "_default_topic", None)
        if default_topic:
            details["default_topic"] = default_topic
        checkup = getattr(publisher, "checkup", None)
        error: str | None = None
        healthy = True
        if callable(checkup):
            try:
                healthy = bool(checkup())
                if not healthy:
                    error = "kafka checkup returned falsy"
            except Exception:
                logger.exception("Kafka connectivity probe failed")
                healthy = False
                error = "kafka checkup raised exception"
        latency_ms = (time.perf_counter() - start) * 1000
        status = "ok" if healthy else "error"
        return ConnectivityProbeSnapshot(
            name="kafka",
            healthy=healthy,
            status=status,
            latency_ms=latency_ms,
            error=error,
            details=details,
        )

    def _configured_macro_calendars(self) -> tuple[str, ...]:
        payload = self._extras.get("TIMESCALE_MACRO_CALENDARS")
        if not payload:
            return tuple()
        parts = [segment.strip() for segment in str(payload).split(",")]
        return _normalise_calendars(parts)

    def _macro_fetch_limit(self) -> int | None:
        raw = self._extras.get("TIMESCALE_MACRO_FETCH_LIMIT")
        if raw is None:
            return None
        try:
            candidate = int(str(raw).strip())
        except (TypeError, ValueError):
            return None
        return candidate if candidate > 0 else None

    def _macro_default_window(self) -> timedelta:
        raw = self._extras.get("TIMESCALE_MACRO_LOOKBACK_DAYS")
        try:
            days = int(str(raw).strip()) if raw is not None else 7
        except (TypeError, ValueError):
            days = 7
        if days <= 0:
            days = 7
        return timedelta(days=days)
