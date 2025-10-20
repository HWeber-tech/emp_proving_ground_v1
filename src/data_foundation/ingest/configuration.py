"""Configuration helpers for Timescale ingest orchestration.

These utilities translate ``SystemConfig`` extras (or plain mappings) into
``TimescaleBackbonePlan`` objects and connection settings so the runtime can
drive institutional ingest without open-coded parsing in the CLI. Centralising
the logic keeps execution aligned with the roadmap/context briefs and makes the
tests assertable.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, MutableMapping, Sequence

from src.governance.system_config import DataBackboneMode, EmpTier, SystemConfig
from ..reference.reference_data_loader import ReferenceDataLoader

from .timescale_pipeline import (
    DailyBarIngestPlan,
    IntradayTradeIngestPlan,
    MacroEventIngestPlan,
    TimescaleBackbonePlan,
)
from ..persist.timescale import TimescaleConnectionSettings
from ..schemas import MacroEvent
from ..streaming.kafka_stream import (
    KafkaConnectionSettings,
    resolve_ingest_topic_specs,
    should_auto_create_topics,
)
from .scheduler import IngestSchedule
from ..batch.spark_export import (
    SparkExportFormat,
    SparkExportJob,
    SparkExportPlan,
)
from ..cache.redis_cache import RedisCachePolicy, RedisConnectionSettings

logger = logging.getLogger(__name__)


CSV_SEPARATOR = ","


def _normalise_extras(extras: Mapping[str, str] | None) -> MutableMapping[str, str]:
    if not extras:
        return {}
    return {str(key): str(value) for key, value in extras.items()}


def _parse_csv(raw: str | None, fallback: Iterable[str] = ()) -> list[str]:
    if raw is None:
        return [symbol for symbol in fallback if str(symbol).strip()]
    tokens = [token.strip() for token in raw.split(CSV_SEPARATOR)]
    symbols = [token for token in tokens if token]
    if symbols:
        seen: set[str] = set()
        ordered: list[str] = []
        for symbol in symbols:
            upper = symbol.upper()
            if upper in seen:
                continue
            seen.add(upper)
            ordered.append(symbol)
        return ordered
    return [symbol for symbol in fallback if str(symbol).strip()]


_API_KEY_ENV_CANDIDATES: dict[str, tuple[str, ...]] = {
    "alpha_vantage": ("ALPHA_VANTAGE_API_KEY", "ALPHAVANTAGE_API_KEY"),
    "fred": ("FRED_API_KEY",),
    "news": ("NEWS_API_KEY", "NEWSAPI_KEY"),
}

_SESSION_CALENDAR_CACHE: tuple[dict[str, object], ...] | None = None
_SYMBOL_INVENTORY_CACHE: tuple[dict[str, object], ...] | None = None
_SYMBOL_LOOKUP_CACHE: dict[str, dict[str, object]] | None = None


def _api_key_metadata(extras: Mapping[str, str]) -> dict[str, dict[str, object]]:
    """Return redaction-safe metadata indicating which provider keys are configured."""

    summary: dict[str, dict[str, object]] = {}
    for provider, candidates in _API_KEY_ENV_CANDIDATES.items():
        configured_key: str | None = None
        for env_key in candidates:
            raw = extras.get(env_key)
            if raw is not None and raw.strip():
                configured_key = env_key
                break
            env_value = os.getenv(env_key)
            if env_value is not None and env_value.strip():
                configured_key = env_key
                break
        summary[provider] = {
            "configured": configured_key is not None,
            "source": configured_key,
        }
    return summary


def _configured_macro_calendars(extras: Mapping[str, str]) -> tuple[str, ...]:
    raw = extras.get("TIMESCALE_MACRO_CALENDARS")
    if not raw:
        return tuple()
    tokens = [segment.strip() for segment in raw.split(CSV_SEPARATOR)]
    calendars: list[str] = []
    for token in tokens:
        if token and token not in calendars:
            calendars.append(token)
    return tuple(calendars)


def _resolve_session_calendars() -> tuple[dict[str, object], ...]:
    global _SESSION_CALENDAR_CACHE
    if _SESSION_CALENDAR_CACHE is not None:
        return _SESSION_CALENDAR_CACHE

    try:
        loader = ReferenceDataLoader()
        sessions = loader.load_sessions()
    except Exception:  # pragma: no cover - defensive guard for optional config
        logger.debug("Failed to load trading session calendars", exc_info=True)
        _SESSION_CALENDAR_CACHE = tuple()
        return _SESSION_CALENDAR_CACHE

    payload: list[dict[str, object]] = []
    for identifier, session in sorted(sessions.items()):
        open_str = session.open_time.strftime("%H:%M") if session.open_time else None
        close_str = session.close_time.strftime("%H:%M") if session.close_time else None
        payload.append(
            {
                "id": identifier,
                "name": session.name,
                "timezone": session.timezone,
                "open_time": open_str,
                "close_time": close_str,
                "days": list(session.days),
                "venue": session.venue,
                "description": session.description,
            }
        )

    _SESSION_CALENDAR_CACHE = tuple(payload)
    return _SESSION_CALENDAR_CACHE


def _resolve_symbol_inventory() -> tuple[dict[str, object], ...]:
    global _SYMBOL_INVENTORY_CACHE
    global _SYMBOL_LOOKUP_CACHE
    if _SYMBOL_INVENTORY_CACHE is not None:
        return _SYMBOL_INVENTORY_CACHE

    try:
        loader = ReferenceDataLoader()
        instruments = loader.load_instruments()
    except Exception:  # pragma: no cover - defensive guard for optional config
        logger.debug("Failed to load instrument inventory", exc_info=True)
        _SYMBOL_INVENTORY_CACHE = tuple()
        _SYMBOL_LOOKUP_CACHE = {}
        return _SYMBOL_INVENTORY_CACHE

    inventory: list[dict[str, object]] = []
    lookup: dict[str, dict[str, object]] = {}
    for instrument in sorted(instruments.values(), key=lambda item: item.symbol):
        payload: dict[str, object] = {
            "symbol": instrument.symbol,
            "contract_size": str(instrument.contract_size),
            "pip_decimal_places": instrument.pip_decimal_places,
            "margin_currency": instrument.margin_currency,
        }

        if instrument.long_swap_rate is not None:
            payload["long_swap_rate"] = str(instrument.long_swap_rate)
        if instrument.short_swap_rate is not None:
            payload["short_swap_rate"] = str(instrument.short_swap_rate)
        if instrument.swap_time is not None:
            fmt = "%H:%M:%S" if instrument.swap_time.second else "%H:%M"
            payload["swap_time"] = instrument.swap_time.strftime(fmt)

        inventory.append(payload)
        lookup[instrument.symbol.upper()] = payload

    _SYMBOL_INVENTORY_CACHE = tuple(inventory)
    _SYMBOL_LOOKUP_CACHE = lookup
    return _SYMBOL_INVENTORY_CACHE


def _symbol_inventory_for_symbols(
    symbols: Sequence[str],
) -> tuple[dict[str, object], ...]:
    """Return inventory entries limited to the configured symbol list."""

    if not symbols:
        return tuple()

    _resolve_symbol_inventory()
    lookup = _SYMBOL_LOOKUP_CACHE or {}
    configured: list[dict[str, object]] = []
    seen: set[str] = set()
    for symbol in symbols:
        token = str(symbol).strip()
        if not token:
            continue
        upper = token.upper()
        if upper in seen:
            continue
        seen.add(upper)
        entry = lookup.get(upper)
        if entry is None:
            configured.append({"symbol": token})
            continue
        configured.append(dict(entry))

    return tuple(configured)


def _resolve_symbol_metadata(symbols: Sequence[str]) -> tuple[dict[str, object], ...]:
    if not symbols:
        return tuple()

    _resolve_symbol_inventory()
    lookup = _SYMBOL_LOOKUP_CACHE or {}

    metadata: list[dict[str, object]] = []
    seen: set[str] = set()
    for symbol in symbols:
        token = str(symbol).strip()
        if not token:
            continue
        upper = token.upper()
        if upper in seen:
            continue
        seen.add(upper)
        entry = lookup.get(upper)
        if entry is None:
            metadata.append({"symbol": token})
            continue
        metadata.append(dict(entry))

    return tuple(metadata)


def _metadata_base(
    api_keys: Mapping[str, Mapping[str, object]],
    session_calendars: Sequence[Mapping[str, object]],
    macro_calendars: Sequence[str],
    symbol_inventory: Sequence[Mapping[str, object]],
    symbol_metadata: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    return {
        "api_keys": {provider: dict(details) for provider, details in api_keys.items()},
        "session_calendars": [dict(entry) for entry in session_calendars],
        "macro_calendars": list(macro_calendars),
        "symbol_inventory": [dict(entry) for entry in symbol_inventory],
        "symbol_metadata": [dict(entry) for entry in symbol_metadata],
    }


def _parse_bool(extras: Mapping[str, str], key: str, default: bool = False) -> bool:
    raw = extras.get(key)
    if raw is None:
        return default
    normalised = raw.strip().lower()
    if normalised in {"1", "true", "yes", "y", "on"}:
        return True
    if normalised in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _parse_int(extras: Mapping[str, str], key: str, default: int) -> int:
    raw = extras.get(key)
    if raw is None:
        return default
    try:
        return int(raw.strip())
    except (TypeError, ValueError):
        return default


def _parse_float(extras: Mapping[str, str], key: str, default: float) -> float:
    raw = extras.get(key)
    if raw is None:
        return default
    try:
        return float(raw.strip())
    except (TypeError, ValueError):
        return default


def _parse_optional_int(extras: Mapping[str, str], key: str) -> int | None:
    raw = extras.get(key)
    if raw is None:
        return None
    try:
        return int(raw.strip())
    except (TypeError, ValueError):
        logger.warning("Invalid value for %s; expected integer", key)
        return None


def _parse_optional_float(extras: Mapping[str, str], key: str) -> float | None:
    raw = extras.get(key)
    if raw is None:
        return None
    try:
        return float(raw.strip())
    except (TypeError, ValueError):
        logger.warning("Invalid value for %s; expected float", key)
        return None


def _parse_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    payload = value.strip()
    if not payload:
        return None
    if payload.endswith("Z"):
        payload = f"{payload[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(payload)
    except ValueError:
        logger.warning("Invalid ISO timestamp '%s'", value)
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _load_macro_events_from_json(payload: str) -> list[MacroEvent]:
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        logger.warning("Invalid JSON payload for TIMESCALE_MACRO_EVENTS; skipping inline events")
        return []

    if not isinstance(data, list):
        logger.warning(
            "Expected TIMESCALE_MACRO_EVENTS to decode to a list; got %s", type(data).__name__
        )
        return []

    events: list[MacroEvent] = []
    for idx, item in enumerate(data):
        if not isinstance(item, Mapping):
            logger.warning(
                "Skipping macro event %s: expected mapping, got %s", idx, type(item).__name__
            )
            continue
        try:
            events.append(MacroEvent.parse_obj(item))
        except Exception:  # pragma: no cover - defensive logging; validation tested separately
            logger.exception("Failed to parse macro event %s from inline JSON", idx)
    return events


def _load_macro_events_from_file(path: str) -> list[MacroEvent]:
    try:
        text = Path(path).expanduser().read_text(encoding="utf-8")
    except FileNotFoundError:
        logger.warning("Macro events file %s not found; skipping", path)
        return []
    except OSError as exc:
        logger.warning("Failed to read macro events file %s: %s", path, exc)
        return []
    return _load_macro_events_from_json(text)


def _resolve_macro_plan(extras: Mapping[str, str]) -> tuple[MacroEventIngestPlan | None, str]:
    if not _parse_bool(extras, "TIMESCALE_ENABLE_MACRO", False):
        return None, "disabled"

    inline_payload = extras.get("TIMESCALE_MACRO_EVENTS")
    file_payload = extras.get("TIMESCALE_MACRO_EVENTS_FILE")
    if inline_payload:
        events = _load_macro_events_from_json(inline_payload)
        if events:
            return MacroEventIngestPlan(events=events, source="inline"), "events"
    elif file_payload:
        events = _load_macro_events_from_file(file_payload)
        if events:
            return MacroEventIngestPlan(events=events, source="file"), "events"

    start = extras.get("TIMESCALE_MACRO_START")
    end = extras.get("TIMESCALE_MACRO_END")
    if start and end:
        return MacroEventIngestPlan(start=start, end=end, source="fred"), "window"

    logger.warning(
        "Macro ingest enabled but neither events nor window configuration was provided; skipping macro plan",
    )
    return None, "disabled"


def _resolve_schedule(extras: Mapping[str, str]) -> IngestSchedule | None:
    if not _parse_bool(extras, "TIMESCALE_INGEST_SCHEDULE", False):
        return None

    interval = max(60, _parse_int(extras, "TIMESCALE_INGEST_INTERVAL_SECONDS", 3600))
    jitter = max(0, _parse_int(extras, "TIMESCALE_INGEST_JITTER_SECONDS", 120))
    max_failures = max(0, _parse_int(extras, "TIMESCALE_INGEST_MAX_FAILURES", 3))
    return IngestSchedule(
        interval_seconds=float(interval),
        jitter_seconds=float(jitter),
        max_failures=int(max_failures),
    )


def _parse_alert_routes(extras: Mapping[str, str]) -> dict[str, str]:
    payload = extras.get("OPERATIONS_ALERT_ROUTES") or extras.get("TIMESCALE_ALERT_ROUTES")
    if not payload:
        return {}

    try:
        decoded = json.loads(payload)
    except json.JSONDecodeError:
        logger.warning("Invalid OPERATIONS_ALERT_ROUTES payload; expected JSON mapping")
        return {}

    if not isinstance(decoded, Mapping):
        logger.warning(
            "OPERATIONS_ALERT_ROUTES should decode to a mapping; got %s",
            type(decoded).__name__,
        )
        return {}

    routes: dict[str, str] = {}
    for key, value in decoded.items():
        key_str = str(key).strip()
        value_str = str(value).strip()
        if key_str and value_str:
            routes[key_str] = value_str
    return routes


def _parse_retention_settings(extras: Mapping[str, str]) -> "TimescaleRetentionSettings":
    enabled = _parse_bool(extras, "TIMESCALE_RETENTION_ENABLED", True)

    def _policy(
        *,
        dimension: str,
        target_key: str,
        minimum_key: str,
        optional_key: str | None = None,
        cap_key: str | None = None,
        default_target: int,
        default_minimum: int,
        default_cap: int | None = None,
    ) -> "TimescaleRetentionPolicySettings":
        target = max(0, _parse_int(extras, target_key, default_target))
        minimum = max(0, _parse_int(extras, minimum_key, default_minimum))
        if minimum > target:
            minimum = target
        optional = _parse_bool(extras, optional_key, False) if optional_key else False
        cap_days: int | None = None
        if cap_key:
            cap_candidate = _parse_optional_int(extras, cap_key)
            if cap_candidate is not None:
                cap_days = max(0, cap_candidate)
        if cap_days is None and default_cap is not None:
            cap_days = max(0, default_cap)
        if cap_days is not None and cap_days < target:
            cap_days = target
        return TimescaleRetentionPolicySettings(
            dimension=dimension,
            target_days=target,
            minimum_days=minimum,
            optional=optional,
            cap_days=cap_days,
        )

    policies = (
        _policy(
            dimension="daily_bars",
            target_key="TIMESCALE_RETENTION_DAILY_TARGET_DAYS",
            minimum_key="TIMESCALE_RETENTION_DAILY_MIN_DAYS",
            cap_key="TIMESCALE_RETENTION_DAILY_CAP_DAYS",
            default_target=365,
            default_minimum=300,
            default_cap=365,
        ),
        _policy(
            dimension="intraday_trades",
            target_key="TIMESCALE_RETENTION_INTRADAY_TARGET_DAYS",
            minimum_key="TIMESCALE_RETENTION_INTRADAY_MIN_DAYS",
            optional_key="TIMESCALE_RETENTION_INTRADAY_OPTIONAL",
            cap_key="TIMESCALE_RETENTION_INTRADAY_CAP_DAYS",
            default_target=30,
            default_minimum=7,
            default_cap=30,
        ),
        _policy(
            dimension="macro_events",
            target_key="TIMESCALE_RETENTION_MACRO_TARGET_DAYS",
            minimum_key="TIMESCALE_RETENTION_MACRO_MIN_DAYS",
            cap_key="TIMESCALE_RETENTION_MACRO_CAP_DAYS",
            default_target=365,
            default_minimum=180,
            default_cap=365,
        ),
    )

    return TimescaleRetentionSettings(enabled=enabled, policies=policies)


def _parse_backup_settings(extras: Mapping[str, str]) -> TimescaleBackupSettings:
    enabled = _parse_bool(extras, "TIMESCALE_BACKUP_ENABLED", True)
    expected = max(60.0, _parse_float(extras, "TIMESCALE_BACKUP_FREQUENCY_SECONDS", 86_400.0))
    retention = max(1, _parse_int(extras, "TIMESCALE_BACKUP_RETENTION_DAYS", 7))
    minimum_retention = max(1, _parse_int(extras, "TIMESCALE_BACKUP_MIN_RETENTION_DAYS", retention))
    warn_after = _parse_optional_float(extras, "TIMESCALE_BACKUP_WARN_AFTER_SECONDS")
    fail_after = _parse_optional_float(extras, "TIMESCALE_BACKUP_FAIL_AFTER_SECONDS")
    restore_interval = max(0, _parse_int(extras, "TIMESCALE_BACKUP_RESTORE_INTERVAL_DAYS", 30))
    last_backup_at = _parse_timestamp(extras.get("TIMESCALE_BACKUP_LAST_SUCCESS"))
    last_backup_status = extras.get("TIMESCALE_BACKUP_LAST_STATUS")
    last_restore_at = _parse_timestamp(extras.get("TIMESCALE_BACKUP_LAST_RESTORE"))
    last_restore_status = extras.get("TIMESCALE_BACKUP_LAST_RESTORE_STATUS")
    providers = tuple(_parse_csv(extras.get("TIMESCALE_BACKUP_PROVIDERS")))
    storage = extras.get("TIMESCALE_BACKUP_STORAGE")
    failures = tuple(_parse_csv(extras.get("TIMESCALE_BACKUP_FAILURES")))

    return TimescaleBackupSettings(
        enabled=enabled,
        expected_frequency_seconds=float(expected),
        retention_days=int(retention),
        minimum_retention_days=int(minimum_retention),
        warn_after_seconds=warn_after,
        fail_after_seconds=fail_after,
        restore_test_interval_days=int(restore_interval),
        last_backup_at=last_backup_at,
        last_backup_status=last_backup_status,
        last_restore_test_at=last_restore_at,
        last_restore_status=last_restore_status,
        providers=providers,
        storage_location=storage,
        recorded_failures=failures,
    )


def _parse_failover_drill_settings(
    extras: Mapping[str, str],
) -> TimescaleFailoverDrillSettings | None:
    if not _parse_bool(extras, "TIMESCALE_FAILOVER_DRILL", False):
        return None

    dimensions = _parse_csv(extras.get("TIMESCALE_FAILOVER_DRILL_DIMENSIONS"), ("daily_bars",))
    label = extras.get("TIMESCALE_FAILOVER_DRILL_LABEL", "required_timescale_failover")
    run_fallback = _parse_bool(
        extras,
        "TIMESCALE_FAILOVER_DRILL_RUN_FALLBACK",
        True,
    )
    return TimescaleFailoverDrillSettings(
        enabled=True,
        dimensions=tuple(dimensions),
        label=label or "required_timescale_failover",
        run_fallback=run_fallback,
    )


def _parse_spark_stress_settings(extras: Mapping[str, str]) -> TimescaleSparkStressSettings | None:
    if not _parse_bool(extras, "TIMESCALE_SPARK_STRESS", False):
        return None

    cycles = max(1, _parse_int(extras, "TIMESCALE_SPARK_STRESS_CYCLES", 3))
    warn_after = _parse_optional_float(extras, "TIMESCALE_SPARK_STRESS_WARN_AFTER_SECONDS")
    fail_after = _parse_optional_float(extras, "TIMESCALE_SPARK_STRESS_FAIL_AFTER_SECONDS")
    label = extras.get("TIMESCALE_SPARK_STRESS_LABEL", "spark_export_stress")

    return TimescaleSparkStressSettings(
        enabled=True,
        cycles=cycles,
        warn_after_seconds=warn_after,
        fail_after_seconds=fail_after,
        label=label or "spark_export_stress",
    )


def _parse_cross_region_settings(extras: Mapping[str, str]) -> TimescaleCrossRegionSettings | None:
    if not _parse_bool(extras, "TIMESCALE_CROSS_REGION_ENABLED", False):
        return None

    warn_after = max(
        0.0,
        _parse_float(
            extras,
            "TIMESCALE_CROSS_REGION_WARN_AFTER_SECONDS",
            900.0,
        ),
    )
    fail_after = max(
        warn_after,
        _parse_float(
            extras,
            "TIMESCALE_CROSS_REGION_FAIL_AFTER_SECONDS",
            1_800.0,
        ),
    )
    ratio = _parse_float(
        extras,
        "TIMESCALE_CROSS_REGION_MAX_ROW_DIFFERENCE_RATIO",
        0.05,
    )
    ratio = max(0.0, min(ratio, 1.0))
    schedule_limit = _parse_optional_float(
        extras, "TIMESCALE_CROSS_REGION_MAX_SCHEDULE_INTERVAL_SECONDS"
    )
    dimensions = tuple(_parse_csv(extras.get("TIMESCALE_CROSS_REGION_DIMENSIONS")))

    replica_mapping: dict[str, str] = {}
    replica_url = extras.get("TIMESCALE_SECONDARY_URL") or extras.get("TIMESCALE_REPLICA_URL")
    if replica_url:
        replica_mapping["TIMESCALEDB_URL"] = replica_url
    secondary_app = extras.get("TIMESCALE_SECONDARY_APP")
    if secondary_app:
        replica_mapping["TIMESCALEDB_APP"] = secondary_app
    secondary_statement = extras.get("TIMESCALE_SECONDARY_STATEMENT_TIMEOUT_MS")
    if secondary_statement:
        replica_mapping["TIMESCALEDB_STATEMENT_TIMEOUT_MS"] = secondary_statement
    secondary_connect = extras.get("TIMESCALE_SECONDARY_CONNECT_TIMEOUT")
    if secondary_connect:
        replica_mapping["TIMESCALEDB_CONNECT_TIMEOUT"] = secondary_connect

    replica_settings = (
        TimescaleConnectionSettings.from_mapping(replica_mapping) if replica_mapping else None
    )

    return TimescaleCrossRegionSettings(
        enabled=True,
        primary_region=extras.get("TIMESCALE_CROSS_REGION_PRIMARY_REGION", "primary"),
        replica_region=extras.get("TIMESCALE_CROSS_REGION_REPLICA_REGION", "replica"),
        warn_after_seconds=warn_after,
        fail_after_seconds=fail_after,
        max_row_difference_ratio=ratio,
        max_schedule_interval_seconds=schedule_limit,
        dimensions=dimensions,
        replica_settings=replica_settings,
    )


def _resolve_macro_calendars(plan: MacroEventIngestPlan | None) -> tuple[str, ...]:
    if plan is None or not plan.events:
        return tuple()
    calendars: list[str] = []
    seen: set[str] = set()
    for event in plan.events:
        calendar: str | None = None
        if isinstance(event, MacroEvent):
            calendar = event.calendar
        elif isinstance(event, Mapping):
            value = event.get("calendar")
            if value is not None:
                calendar = str(value)
        if calendar:
            normalised = calendar.strip()
            if normalised and normalised not in seen:
                seen.add(normalised)
                calendars.append(normalised)
    return tuple(calendars)


def _parse_spark_export_plan(
    extras: Mapping[str, str],
    plan: TimescaleBackbonePlan,
) -> SparkExportPlan | None:
    if not _parse_bool(extras, "TIMESCALE_SPARK_EXPORT", False):
        return None

    root_raw = extras.get("TIMESCALE_SPARK_EXPORT_ROOT", "data/spark_exports")
    root_path = Path(str(root_raw)).expanduser()

    format_raw = extras.get("TIMESCALE_SPARK_EXPORT_FORMAT", SparkExportFormat.csv.value)
    try:
        export_format = SparkExportFormat(str(format_raw).strip().lower())
    except ValueError:
        logger.warning("Unknown TIMESCALE_SPARK_EXPORT_FORMAT '%s'; defaulting to csv", format_raw)
        export_format = SparkExportFormat.csv

    partition_columns = tuple(_parse_csv(extras.get("TIMESCALE_SPARK_EXPORT_PARTITION_BY")))
    include_metadata = _parse_bool(extras, "TIMESCALE_SPARK_EXPORT_INCLUDE_METADATA", True)
    publish = _parse_bool(extras, "TIMESCALE_SPARK_EXPORT_PUBLISH", True)
    filename_prefix = extras.get("TIMESCALE_SPARK_EXPORT_PREFIX", "timescale")

    limit_raw = _parse_int(extras, "TIMESCALE_SPARK_EXPORT_LIMIT", 0)
    limit = limit_raw if limit_raw > 0 else None

    requested_dimensions = _parse_csv(extras.get("TIMESCALE_SPARK_EXPORT_DIMENSIONS"))
    default_dimensions: list[str] = []
    if plan.daily is not None:
        default_dimensions.append("daily_bars")
    if plan.intraday is not None:
        default_dimensions.append("intraday_trades")
    if plan.macro is not None:
        default_dimensions.append("macro_events")

    dimensions = (
        [dim.lower() for dim in requested_dimensions]
        if requested_dimensions
        else default_dimensions
    )

    jobs: list[SparkExportJob] = []

    if "daily_bars" in dimensions and plan.daily is not None:
        jobs.append(
            SparkExportJob(
                dimension="daily_bars",
                symbols=tuple(plan.daily.normalised_symbols()),
                limit=limit,
                partition_by=partition_columns,
                filename=f"{filename_prefix}_daily.{export_format.extension}",
            )
        )

    if "intraday_trades" in dimensions and plan.intraday is not None:
        jobs.append(
            SparkExportJob(
                dimension="intraday_trades",
                symbols=tuple(plan.intraday.normalised_symbols()),
                limit=limit,
                partition_by=partition_columns,
                filename=f"{filename_prefix}_intraday.{export_format.extension}",
            )
        )

    if "macro_events" in dimensions and plan.macro is not None:
        macro_start = _parse_timestamp(plan.macro.start) if plan.macro.start else None
        macro_end = _parse_timestamp(plan.macro.end) if plan.macro.end else None
        jobs.append(
            SparkExportJob(
                dimension="macro_events",
                symbols=_resolve_macro_calendars(plan.macro),
                start=macro_start,
                end=macro_end,
                limit=limit,
                partition_by=partition_columns,
                filename=f"{filename_prefix}_macro.{export_format.extension}",
            )
        )

    if not jobs:
        logger.warning(
            "Spark export enabled but no matching ingest dimensions were configured; skipping plan"
        )
        return None

    return SparkExportPlan(
        root_path=root_path,
        format=export_format,
        jobs=tuple(jobs),
        partition_columns=partition_columns,
        include_metadata=include_metadata,
        publish_telemetry=publish,
    )


@dataclass(frozen=True)
class TimescaleIngestRecoverySettings:
    """Runtime knobs controlling Timescale ingest recovery behaviour."""

    enabled: bool = False
    max_attempts: int = 1
    lookback_multiplier: float = 2.0
    target_missing_symbols: bool = True

    def should_attempt(self) -> bool:
        return self.enabled and self.max_attempts > 0


@dataclass(frozen=True)
class TimescaleBackupSettings:
    """Resolved backup policy and state derived from extras."""

    enabled: bool = True
    expected_frequency_seconds: float = 86_400.0
    retention_days: int = 7
    minimum_retention_days: int = 7
    warn_after_seconds: float | None = None
    fail_after_seconds: float | None = None
    restore_test_interval_days: int = 30
    last_backup_at: datetime | None = None
    last_backup_status: str | None = None
    last_restore_test_at: datetime | None = None
    last_restore_status: str | None = None
    providers: tuple[str, ...] = ()
    storage_location: str | None = None
    recorded_failures: tuple[str, ...] = ()

    def to_metadata(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "enabled": self.enabled,
            "expected_frequency_seconds": self.expected_frequency_seconds,
            "retention_days": self.retention_days,
            "minimum_retention_days": self.minimum_retention_days,
            "restore_test_interval_days": self.restore_test_interval_days,
            "providers": list(self.providers),
            "storage_location": self.storage_location,
            "recorded_failures": list(self.recorded_failures),
            "last_backup_status": self.last_backup_status,
            "last_restore_status": self.last_restore_status,
        }
        if self.warn_after_seconds is not None:
            payload["warn_after_seconds"] = self.warn_after_seconds
        if self.fail_after_seconds is not None:
            payload["fail_after_seconds"] = self.fail_after_seconds
        if self.last_backup_at is not None:
            payload["last_backup_at"] = self.last_backup_at.isoformat()
        if self.last_restore_test_at is not None:
            payload["last_restore_test_at"] = self.last_restore_test_at.isoformat()
        return payload


@dataclass(frozen=True)
class TimescaleRetentionPolicySettings:
    """Retention expectations for a single Timescale dataset."""

    dimension: str
    target_days: int
    minimum_days: int
    optional: bool = False
    cap_days: int | None = None

    def to_metadata(self) -> Mapping[str, object]:
        payload: dict[str, object] = {
            "dimension": self.dimension,
            "target_days": self.target_days,
            "minimum_days": self.minimum_days,
        }
        if self.optional:
            payload["optional"] = True
        if self.cap_days is not None:
            payload["cap_days"] = self.cap_days
        return payload


@dataclass(frozen=True)
class TimescaleRetentionSettings:
    """Aggregate retention configuration derived from extras."""

    enabled: bool = True
    policies: tuple[TimescaleRetentionPolicySettings, ...] = ()

    def to_metadata(self) -> Mapping[str, object]:
        return {
            "enabled": self.enabled,
            "policies": [policy.to_metadata() for policy in self.policies],
        }


@dataclass(frozen=True)
class TimescaleFailoverDrillSettings:
    """Configuration for simulated ingest failover drills."""

    enabled: bool
    dimensions: tuple[str, ...] = field(default_factory=tuple)
    label: str = "required_timescale_failover"
    run_fallback: bool = True

    def to_metadata(self) -> dict[str, object]:
        return {
            "enabled": self.enabled,
            "dimensions": list(self.dimensions),
            "label": self.label,
            "run_fallback": self.run_fallback,
        }


@dataclass(frozen=True)
class TimescaleSparkStressSettings:
    """Configuration captured for Spark export stress drills."""

    enabled: bool
    cycles: int = 3
    warn_after_seconds: float | None = None
    fail_after_seconds: float | None = None
    label: str = "spark_export_stress"

    def to_metadata(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "enabled": self.enabled,
            "cycles": self.cycles,
            "label": self.label,
        }
        if self.warn_after_seconds is not None:
            payload["warn_after_seconds"] = self.warn_after_seconds
        if self.fail_after_seconds is not None:
            payload["fail_after_seconds"] = self.fail_after_seconds
        return payload


@dataclass(frozen=True)
class KafkaReadinessSettings:
    """Configuration for Kafka readiness telemetry thresholds."""

    enabled: bool = False
    warn_lag_messages: int = 1_000
    fail_lag_messages: int = 10_000
    warn_stale_seconds: float = 600.0
    fail_stale_seconds: float = 1_800.0
    min_publishers: int = 1
    require_topics: bool = True
    require_consumer: bool = False

    def to_metadata(self) -> Mapping[str, object]:
        return {
            "enabled": self.enabled,
            "warn_lag_messages": int(self.warn_lag_messages),
            "fail_lag_messages": int(self.fail_lag_messages),
            "warn_stale_seconds": float(self.warn_stale_seconds),
            "fail_stale_seconds": float(self.fail_stale_seconds),
            "min_publishers": int(self.min_publishers),
            "require_topics": self.require_topics,
            "require_consumer": self.require_consumer,
        }


@dataclass(frozen=True)
class TimescaleCrossRegionSettings:
    """Cross-region replication configuration for the ingest slice."""

    enabled: bool
    primary_region: str = "primary"
    replica_region: str = "replica"
    warn_after_seconds: float = 900.0
    fail_after_seconds: float = 1_800.0
    max_row_difference_ratio: float = 0.05
    max_schedule_interval_seconds: float | None = None
    dimensions: tuple[str, ...] = field(default_factory=tuple)
    replica_settings: TimescaleConnectionSettings | None = None

    def to_metadata(self) -> Mapping[str, object]:
        payload: dict[str, object] = {
            "enabled": self.enabled,
            "primary_region": self.primary_region,
            "replica_region": self.replica_region,
            "warn_after_seconds": self.warn_after_seconds,
            "fail_after_seconds": self.fail_after_seconds,
            "max_row_difference_ratio": self.max_row_difference_ratio,
            "max_schedule_interval_seconds": self.max_schedule_interval_seconds,
            "dimensions": list(self.dimensions),
            "replica_configured": bool(self.replica_settings and self.replica_settings.configured),
        }
        return payload


@dataclass(frozen=True)
class InstitutionalIngestConfig:
    """Resolved ingest configuration derived from ``SystemConfig`` extras."""

    should_run: bool
    reason: str | None
    plan: TimescaleBackbonePlan = field(default_factory=TimescaleBackbonePlan)
    timescale_settings: TimescaleConnectionSettings = field(
        default_factory=TimescaleConnectionSettings.from_env
    )
    kafka_settings: KafkaConnectionSettings = field(
        default_factory=KafkaConnectionSettings.from_env
    )
    redis_settings: RedisConnectionSettings = field(
        default_factory=RedisConnectionSettings.from_env
    )
    redis_policy: RedisCachePolicy = field(
        default_factory=RedisCachePolicy.institutional_defaults
    )
    enable_streaming: bool = True
    metadata: dict[str, object] = field(default_factory=dict)
    schedule: IngestSchedule | None = None
    recovery: TimescaleIngestRecoverySettings = field(
        default_factory=TimescaleIngestRecoverySettings
    )
    operational_alert_routes: dict[str, str] = field(default_factory=dict)
    backup: TimescaleBackupSettings = field(default_factory=TimescaleBackupSettings)
    retention: TimescaleRetentionSettings = field(default_factory=TimescaleRetentionSettings)
    spark_export: SparkExportPlan | None = None
    failover_drill: TimescaleFailoverDrillSettings | None = None
    spark_stress: TimescaleSparkStressSettings | None = None
    cross_region: TimescaleCrossRegionSettings | None = None
    kafka_readiness: KafkaReadinessSettings = field(default_factory=KafkaReadinessSettings)


def build_institutional_ingest_config(
    config: SystemConfig,
    *,
    fallback_symbols: Sequence[str] | None = None,
) -> InstitutionalIngestConfig:
    """Translate ``SystemConfig`` extras into a Timescale ingest execution plan."""

    extras = _normalise_extras(config.extras)
    fallback = fallback_symbols or ()
    redis_settings = RedisConnectionSettings.from_mapping(extras)
    streaming_enabled = _parse_bool(
        extras,
        "KAFKA_INGEST_ENABLE_STREAMING",
        True,
    )
    api_keys_summary = _api_key_metadata(extras)
    session_calendars = _resolve_session_calendars()
    macro_calendars = _configured_macro_calendars(extras)
    symbols = _parse_csv(extras.get("TIMESCALE_SYMBOLS"), fallback)
    symbol_inventory = _symbol_inventory_for_symbols(symbols)
    symbol_metadata = _resolve_symbol_metadata(symbols)

    def _base_metadata() -> dict[str, object]:
        metadata = _metadata_base(
            api_keys_summary,
            session_calendars,
            macro_calendars,
            symbol_inventory,
            symbol_metadata,
        )
        metadata["symbols"] = list(symbols)
        return metadata

    reason: str | None = None
    if config.data_backbone_mode is not DataBackboneMode.institutional:
        reason = "Data backbone mode is not institutional"
        metadata = _base_metadata()
        metadata["mode"] = config.data_backbone_mode.value
        return InstitutionalIngestConfig(
            should_run=False,
            reason=reason,
            redis_settings=redis_settings,
            metadata=metadata,
            enable_streaming=streaming_enabled,
        )

    if config.tier is not EmpTier.tier_1:
        reason = "Timescale ingest currently targets Tier-1 institutional runs"
        metadata = _base_metadata()
        metadata["tier"] = config.tier.value
        return InstitutionalIngestConfig(
            should_run=False,
            reason=reason,
            redis_settings=redis_settings,
            metadata=metadata,
            enable_streaming=streaming_enabled,
        )

    daily_plan = None
    if symbols:
        daily_plan = DailyBarIngestPlan(
            symbols=symbols,
            lookback_days=_parse_int(extras, "TIMESCALE_LOOKBACK_DAYS", 60),
        )

    intraday_plan = None
    if symbols and _parse_bool(extras, "TIMESCALE_ENABLE_INTRADAY", False):
        intraday_plan = IntradayTradeIngestPlan(
            symbols=symbols,
            lookback_days=_parse_int(extras, "TIMESCALE_INTRADAY_LOOKBACK_DAYS", 2),
            interval=extras.get("TIMESCALE_INTRADAY_INTERVAL", "1m"),
        )

    macro_plan, macro_mode = _resolve_macro_plan(extras)

    plan = TimescaleBackbonePlan(daily=daily_plan, intraday=intraday_plan, macro=macro_plan)
    schedule = _resolve_schedule(extras)

    metadata = _base_metadata()
    metadata.update(
        {
            "symbols": symbols,
            "daily": bool(daily_plan),
            "daily_lookback_days": daily_plan.lookback_days if daily_plan else None,
            "intraday": bool(intraday_plan),
            "intraday_lookback_days": intraday_plan.lookback_days if intraday_plan else None,
            "intraday_interval": intraday_plan.interval if intraday_plan else None,
            "macro_mode": macro_mode,
            "schedule_enabled": bool(schedule),
            "schedule_interval_seconds": schedule.interval_seconds if schedule else None,
            "schedule_jitter_seconds": schedule.jitter_seconds if schedule else None,
            "schedule_max_failures": schedule.max_failures if schedule else None,
        }
    )
    if macro_plan and macro_plan.events is not None:
        metadata["macro_events"] = len(macro_plan.events)
    if macro_plan and macro_plan.has_window():
        metadata["macro_window"] = {"start": macro_plan.start, "end": macro_plan.end}

    redis_policy = RedisCachePolicy.from_mapping(
        extras,
        fallback=RedisCachePolicy.institutional_defaults(),
    )
    metadata["redis_cache_policy"] = {
        "ttl_seconds": redis_policy.ttl_seconds,
        "max_keys": redis_policy.max_keys,
        "namespace": redis_policy.namespace,
        "invalidate_prefixes": list(redis_policy.invalidate_prefixes),
    }
    metadata["redis_configured"] = redis_settings.configured
    metadata["redis_client_name"] = redis_settings.client_name
    metadata["redis_ssl"] = redis_settings.ssl

    alert_routes = _parse_alert_routes(extras)
    if alert_routes:
        metadata["operational_alert_routes"] = dict(alert_routes)

    spark_plan = _parse_spark_export_plan(extras, plan)
    if spark_plan is not None:
        metadata["spark_export"] = {
            "root_path": str(spark_plan.root_path),
            "format": spark_plan.format.value,
            "dimensions": [job.dimension for job in spark_plan.jobs],
            "partition_by": list(spark_plan.partition_columns),
            "include_metadata": spark_plan.include_metadata,
            "publish": spark_plan.publish_telemetry,
        }

    should_run = not plan.is_empty()
    if not should_run:
        reason = "No ingest slices configured via extras or fallback symbols"

    timescale_settings = TimescaleConnectionSettings.from_mapping(extras)
    kafka_settings = KafkaConnectionSettings.from_mapping(extras)
    kafka_topics = resolve_ingest_topic_specs(extras)
    kafka_topic_names = [spec.name for spec in kafka_topics]
    metadata["kafka_configured"] = kafka_settings.configured
    metadata["kafka_topics"] = kafka_topic_names
    metadata["kafka_auto_create_topics"] = should_auto_create_topics(extras)
    metadata["kafka_streaming_enabled"] = streaming_enabled

    default_readiness_enabled = bool(kafka_settings.configured or kafka_topic_names)
    warn_lag = max(0, _parse_int(extras, "KAFKA_READINESS_WARN_LAG_MESSAGES", 1_000))
    fail_lag = max(
        warn_lag,
        _parse_int(extras, "KAFKA_READINESS_FAIL_LAG_MESSAGES", 10_000),
    )
    warn_stale = max(
        0.0,
        _parse_float(extras, "KAFKA_READINESS_WARN_STALE_SECONDS", 600.0),
    )
    fail_stale = max(
        warn_stale,
        _parse_float(extras, "KAFKA_READINESS_FAIL_STALE_SECONDS", 1_800.0),
    )
    min_publishers = max(0, _parse_int(extras, "KAFKA_READINESS_MIN_PUBLISHERS", 1))
    kafka_readiness_settings = KafkaReadinessSettings(
        enabled=_parse_bool(
            extras,
            "KAFKA_READINESS_ENABLED",
            default_readiness_enabled,
        ),
        warn_lag_messages=warn_lag,
        fail_lag_messages=fail_lag,
        warn_stale_seconds=warn_stale,
        fail_stale_seconds=fail_stale,
        min_publishers=min_publishers,
        require_topics=_parse_bool(
            extras,
            "KAFKA_READINESS_REQUIRE_TOPICS",
            True,
        ),
        require_consumer=_parse_bool(
            extras,
            "KAFKA_READINESS_REQUIRE_CONSUMER",
            False,
        ),
    )
    metadata["kafka_readiness"] = kafka_readiness_settings.to_metadata()

    recovery_settings = TimescaleIngestRecoverySettings(
        enabled=_parse_bool(extras, "TIMESCALE_INGEST_RECOVERY", False),
        max_attempts=max(0, _parse_int(extras, "TIMESCALE_INGEST_RECOVERY_MAX_ATTEMPTS", 1)),
        lookback_multiplier=max(
            1.0,
            _parse_float(extras, "TIMESCALE_INGEST_RECOVERY_LOOKBACK_MULTIPLIER", 2.0),
        ),
        target_missing_symbols=_parse_bool(
            extras,
            "TIMESCALE_INGEST_RECOVERY_TARGET_MISSING_SYMBOLS",
            True,
        ),
    )
    if recovery_settings.enabled:
        metadata["recovery"] = {
            "max_attempts": recovery_settings.max_attempts,
            "lookback_multiplier": recovery_settings.lookback_multiplier,
            "target_missing_symbols": recovery_settings.target_missing_symbols,
        }

    backup_settings = _parse_backup_settings(extras)
    metadata["backup"] = backup_settings.to_metadata()
    retention_settings = _parse_retention_settings(extras)
    metadata["retention"] = retention_settings.to_metadata()

    failover_drill_settings = _parse_failover_drill_settings(extras)
    if failover_drill_settings is not None:
        metadata["failover_drill"] = failover_drill_settings.to_metadata()

    spark_stress_settings = _parse_spark_stress_settings(extras)
    if spark_stress_settings is not None:
        metadata["spark_stress"] = spark_stress_settings.to_metadata()

    cross_region_settings = _parse_cross_region_settings(extras)
    if cross_region_settings is not None:
        metadata["cross_region"] = cross_region_settings.to_metadata()

    return InstitutionalIngestConfig(
        should_run=should_run,
        reason=reason,
        plan=plan,
        timescale_settings=timescale_settings,
        kafka_settings=kafka_settings,
        redis_settings=redis_settings,
        redis_policy=redis_policy,
        enable_streaming=streaming_enabled,
        metadata=metadata,
        schedule=schedule,
        recovery=recovery_settings,
        operational_alert_routes=alert_routes,
        backup=backup_settings,
        retention=retention_settings,
        spark_export=spark_plan,
        failover_drill=failover_drill_settings,
        spark_stress=spark_stress_settings,
        cross_region=cross_region_settings,
        kafka_readiness=kafka_readiness_settings,
    )


__all__ = [
    "InstitutionalIngestConfig",
    "TimescaleBackupSettings",
    "TimescaleRetentionPolicySettings",
    "TimescaleRetentionSettings",
    "TimescaleFailoverDrillSettings",
    "TimescaleSparkStressSettings",
    "TimescaleCrossRegionSettings",
    "build_institutional_ingest_config",
]
