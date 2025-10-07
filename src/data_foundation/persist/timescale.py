"""TimescaleDB connection, migrations, and ingest helpers.

This module translates the roadmap's "TimescaleDB ingest vertical" objective into
executable code:

* ``TimescaleConnectionSettings`` wires configuration sourced from ``SystemConfig``
  extras or environment variables into a SQLAlchemy engine.
* ``TimescaleMigrator`` sets up the core schemas (daily bars, intraday trades,
  macro events) and promotes them to hypertables when running against a real
  Timescale/PostgreSQL backend.
* ``TimescaleIngestor`` provides idempotent upsert logic so the existing Yahoo
  bootstrap downloader can seed institutional storage without duplicate rows.

The helpers are written to fall back to generic SQL when running in test
environments (e.g. SQLite) so CI can exercise the round-trip behaviour without a
Timescale daemon.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Iterable, Mapping, Sequence, cast
from uuid import uuid4

import json
import re

import pandas as pd
from sqlalchemy import MetaData, Table, case, create_engine, func, select, text
from sqlalchemy.engine import Connection, Engine, Row, RowMapping

from src.core.coercion import coerce_float, coerce_int

logger = logging.getLogger(__name__)

_JSON_INPUT_TYPES = (str, bytes, bytearray)

_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _validate_identifier(value: str, *, label: str) -> str:
    """Ensure SQL identifiers only contain safe characters."""

    if not value:
        raise ValueError(f"{label} identifier must not be empty")
    if not _IDENTIFIER_PATTERN.fullmatch(value):
        raise ValueError(
            f"{label} identifier must contain only letters, numbers, or underscores: {value!r}"
        )
    return value


def _ensure_mapping(
    row: Mapping[str, Any] | RowMapping | Row[Any],
) -> dict[str, Any]:
    if isinstance(row, Mapping):
        return {str(key): row[key] for key in row}
    if isinstance(row, RowMapping):
        return {str(key): row[key] for key in row}
    if isinstance(row, Row):
        return {str(key): row._mapping[key] for key in row._mapping}
    raise TypeError("Row must provide mapping access")


def _load_json_payload(raw: object) -> object | None:
    if isinstance(raw, _JSON_INPUT_TYPES):
        text_value = raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else raw
        try:
            loaded: object = json.loads(text_value)
            return loaded
        except json.JSONDecodeError:
            logger.debug("Failed to decode JSON payload", exc_info=True)
            return None
    return None


def _load_json_mapping(raw: object) -> dict[str, Any]:
    payload = _load_json_payload(raw)
    if isinstance(payload, Mapping):
        return {str(key): payload[key] for key in payload}
    return {}


def _load_json_sequence(raw: object) -> list[Any]:
    payload = _load_json_payload(raw)
    if isinstance(payload, Sequence) and not isinstance(payload, _JSON_INPUT_TYPES):
        return list(payload)
    return []


# ---------------------------------------------------------------------------
# Connection settings
# ---------------------------------------------------------------------------


DEFAULT_SQLITE_FALLBACK = "sqlite:///data/timescale_sim.db"


def _normalise_mapping(env: Mapping[str, str] | None) -> Mapping[str, str]:
    if env is None:
        return {k: v for k, v in os.environ.items() if isinstance(v, str)}
    return env


@dataclass(frozen=True)
class TimescaleConnectionSettings:
    """Settings required to connect to TimescaleDB via SQLAlchemy."""

    url: str
    application_name: str = "emp-timescale-ingest"
    statement_timeout_ms: int = 10_000
    connect_timeout: int = 5

    @classmethod
    def from_mapping(
        cls, mapping: Mapping[str, str] | None = None
    ) -> "TimescaleConnectionSettings":
        """Build settings from a mapping (SystemConfig extras or environment)."""

        data = _normalise_mapping(mapping)

        url = (
            data.get("TIMESCALEDB_URL")
            or data.get("TIMESCALE_URL")
            or data.get("DATABASE_URL")
            or DEFAULT_SQLITE_FALLBACK
        )

        app_name = (
            data.get("TIMESCALEDB_APP") or data.get("TIMESCALE_APP") or "emp-timescale-ingest"
        )

        def _coerce_int(key: str, default: int) -> int:
            raw = data.get(key)
            if raw is None:
                return default
            try:
                return int(str(raw))
            except (TypeError, ValueError):
                return default

        statement_timeout = _coerce_int("TIMESCALEDB_STATEMENT_TIMEOUT_MS", 10_000)
        connect_timeout = _coerce_int("TIMESCALEDB_CONNECT_TIMEOUT", 5)

        return cls(
            url=url,
            application_name=app_name,
            statement_timeout_ms=statement_timeout,
            connect_timeout=connect_timeout,
        )

    @classmethod
    def from_env(cls) -> "TimescaleConnectionSettings":
        return cls.from_mapping(os.environ)

    @property
    def configured(self) -> bool:
        """Return ``True`` when pointing at a non-simulated Timescale service."""

        return self.url != DEFAULT_SQLITE_FALLBACK

    def is_postgres(self) -> bool:
        return self.url.startswith("postgresql") or ".postgres" in self.url

    def create_engine(self) -> Engine:
        """Instantiate a SQLAlchemy engine with sensible defaults."""

        connect_args: dict[str, object] = {}
        if self.is_postgres():
            connect_args["application_name"] = self.application_name
            connect_args["connect_timeout"] = self.connect_timeout
            # Encode statement timeout using the libpq options flag
            opts = f"-c statement_timeout={self.statement_timeout_ms}"
            connect_args["options"] = opts

        kwargs: dict[str, object] = {"pool_pre_ping": True}
        if connect_args:
            kwargs["connect_args"] = connect_args

        engine = create_engine(self.url, **kwargs)
        logger.debug("Created Timescale engine for %s", self.url)
        return engine


# ---------------------------------------------------------------------------
# Migrations
# ---------------------------------------------------------------------------


def _table_name(schema: str, table: str, dialect_name: str) -> str:
    safe_schema = _validate_identifier(schema, label="schema")
    safe_table = _validate_identifier(table, label="table")
    if dialect_name == "postgresql":
        return f"{safe_schema}.{safe_table}"
    return f"{safe_schema}_{safe_table}"


def _table_identity(
    schema: str, table: str, dialect_name: str
) -> tuple[str | None, str]:
    """Return schema/name pair suitable for SQLAlchemy table reflection."""

    safe_schema = _validate_identifier(schema, label="schema")
    safe_table = _validate_identifier(table, label="table")
    if dialect_name == "postgresql":
        return safe_schema, safe_table
    return None, f"{safe_schema}_{safe_table}"


def _reflect_table(conn: Connection, identifier: tuple[str, str]) -> Table:
    """Reflect an allow-listed Timescale table using the current dialect."""

    schema, table_name = _table_identity(*identifier, conn.dialect.name)
    metadata = MetaData()
    return Table(table_name, metadata, schema=schema, autoload_with=conn)


def _timestamp_type(dialect: str) -> str:
    return "TIMESTAMPTZ" if dialect == "postgresql" else "TIMESTAMP"


def _float_type(dialect: str) -> str:
    return "DOUBLE PRECISION" if dialect == "postgresql" else "REAL"


def _now_function(dialect: str) -> str:
    return "NOW()" if dialect == "postgresql" else "CURRENT_TIMESTAMP"


def _json_default(value: object) -> str:
    return str(value)


def _normalise_timestamp(value: object) -> str | None:
    """Return an ISO timestamp for JSON-friendly payloads."""

    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=UTC)
        return value.astimezone(UTC).isoformat()
    return str(value)


_SEVERITY_RANK: dict[str, int] = {
    "critical": 4,
    "error": 3,
    "warning": 2,
    "warn": 2,
    "info": 1,
}


class TimescaleMigrator:
    """Create schemas and hypertables required by the institutional ingest slice."""

    _DAILY_BARS = ("market_data", "daily_bars")
    _INTRADAY_TRADES = ("market_data", "intraday_trades")
    _MACRO_EVENTS = ("macro_data", "events")
    _INGEST_RUNS = ("telemetry", "ingest_runs")
    _COMPLIANCE_AUDIT = ("telemetry", "compliance_audit")
    _COMPLIANCE_KYC = ("telemetry", "compliance_kyc")
    _EXECUTION_SNAPSHOTS = ("telemetry", "execution_snapshots")
    _CONFIGURATION_AUDIT = ("telemetry", "configuration_audit")

    def __init__(self, engine: Engine) -> None:
        self._engine = engine
        self._logger = logging.getLogger(f"{__name__}.TimescaleMigrator")

    def apply(self) -> None:
        with self._engine.begin() as conn:
            dialect = conn.dialect.name
            self._logger.info("Running Timescale migrations using %s dialect", dialect)
            if dialect == "postgresql":
                self._bootstrap_timescale(conn)
            self._create_daily_bars(conn, dialect)
            self._create_intraday_trades(conn, dialect)
            self._create_macro_events(conn, dialect)
            self._create_ingest_journal(conn, dialect)
            self._create_compliance_audit(conn, dialect)
            self._create_compliance_kyc(conn, dialect)
            self._create_execution_snapshots(conn, dialect)
            self._create_configuration_audit(conn, dialect)

    def ensure_compliance_tables(self) -> None:
        """Ensure compliance audit tables exist without mutating other schemas."""

        with self._engine.begin() as conn:
            dialect = conn.dialect.name
            self._create_compliance_audit(conn, dialect)
            self._create_compliance_kyc(conn, dialect)

    def ensure_execution_tables(self) -> None:
        """Ensure execution telemetry tables exist."""

        with self._engine.begin() as conn:
            dialect = conn.dialect.name
            self._create_execution_snapshots(conn, dialect)

    def ensure_configuration_tables(self) -> None:
        """Ensure configuration audit tables exist."""

        with self._engine.begin() as conn:
            dialect = conn.dialect.name
            self._create_configuration_audit(conn, dialect)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _bootstrap_timescale(self, conn: Connection) -> None:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS timescaledb"))
        conn.execute(text("CREATE SCHEMA IF NOT EXISTS market_data"))
        conn.execute(text("CREATE SCHEMA IF NOT EXISTS macro_data"))

    def _create_daily_bars(self, conn: Connection, dialect: str) -> None:
        table = _table_name(*self._DAILY_BARS, dialect)
        ts_type = _timestamp_type(dialect)
        float_type = _float_type(dialect)
        now_fn = _now_function(dialect)
        ddl = f"""
        CREATE TABLE IF NOT EXISTS {table} (
            ts {ts_type} NOT NULL,
            symbol TEXT NOT NULL,
            open {float_type},
            high {float_type},
            low {float_type},
            close {float_type},
            adj_close {float_type},
            volume {float_type},
            source TEXT NOT NULL DEFAULT 'yahoo',
            ingested_at {ts_type} NOT NULL DEFAULT {now_fn},
            PRIMARY KEY (symbol, ts)
        )
        """
        conn.execute(text(ddl))

        if dialect == "postgresql":
            conn.execute(
                text(
                    "SELECT create_hypertable(:table, 'ts', if_not_exists => TRUE, migrate_data => TRUE)",
                ),
                {"table": table},
            )

    def _create_intraday_trades(self, conn: Connection, dialect: str) -> None:
        table = _table_name(*self._INTRADAY_TRADES, dialect)
        ts_type = _timestamp_type(dialect)
        float_type = _float_type(dialect)
        now_fn = _now_function(dialect)
        ddl = f"""
        CREATE TABLE IF NOT EXISTS {table} (
            ts {ts_type} NOT NULL,
            symbol TEXT NOT NULL,
            price {float_type} NOT NULL,
            size {float_type},
            exchange TEXT,
            conditions TEXT,
            source TEXT NOT NULL DEFAULT 'yahoo',
            ingested_at {ts_type} NOT NULL DEFAULT {now_fn},
            PRIMARY KEY (symbol, ts, price)
        )
        """
        conn.execute(text(ddl))
        if dialect == "postgresql":
            conn.execute(
                text(
                    "SELECT create_hypertable(:table, 'ts', if_not_exists => TRUE, migrate_data => TRUE)",
                ),
                {"table": table},
            )

    def _create_macro_events(self, conn: Connection, dialect: str) -> None:
        table = _table_name(*self._MACRO_EVENTS, dialect)
        ts_type = _timestamp_type(dialect)
        float_type = _float_type(dialect)
        now_fn = _now_function(dialect)
        ddl = f"""
        CREATE TABLE IF NOT EXISTS {table} (
            ts {ts_type} NOT NULL,
            event_name TEXT NOT NULL,
            calendar TEXT,
            currency TEXT,
            actual {float_type},
            forecast {float_type},
            previous {float_type},
            importance TEXT,
            source TEXT NOT NULL DEFAULT 'fred',
            ingested_at {ts_type} NOT NULL DEFAULT {now_fn},
            PRIMARY KEY (event_name, ts)
        )
        """
        conn.execute(text(ddl))
        if dialect == "postgresql":
            conn.execute(
                text(
                    "SELECT create_hypertable(:table, 'ts', if_not_exists => TRUE, migrate_data => TRUE)",
                ),
                {"table": table},
            )

    def _create_ingest_journal(self, conn: Connection, dialect: str) -> None:
        if dialect == "postgresql":
            conn.execute(text("CREATE SCHEMA IF NOT EXISTS telemetry"))

        table = _table_name(*self._INGEST_RUNS, dialect)
        ts_type = _timestamp_type(dialect)
        float_type = _float_type(dialect)
        ddl = f"""
        CREATE TABLE IF NOT EXISTS {table} (
            run_id TEXT PRIMARY KEY,
            dimension TEXT NOT NULL,
            status TEXT NOT NULL,
            rows_written INTEGER NOT NULL,
            freshness_seconds {float_type},
            ingest_duration_seconds {float_type},
            executed_at {ts_type} NOT NULL,
            source TEXT,
            symbols TEXT,
            metadata TEXT
        )
        """
        conn.execute(text(ddl))

        if dialect == "postgresql":
            index_stmt = text(
                "CREATE INDEX IF NOT EXISTS telemetry_ingest_runs_dimension_idx "
                "ON telemetry.ingest_runs (dimension, executed_at DESC)"
            )
        else:
            index_stmt = text(
                f"CREATE INDEX IF NOT EXISTS telemetry_ingest_runs_dimension_idx "
                f"ON {table} (dimension, executed_at DESC)"
            )
        conn.execute(index_stmt)

    def _create_compliance_audit(self, conn: Connection, dialect: str) -> None:
        if dialect == "postgresql":
            conn.execute(text("CREATE SCHEMA IF NOT EXISTS telemetry"))

        table = _table_name(*self._COMPLIANCE_AUDIT, dialect)
        ts_type = _timestamp_type(dialect)
        float_type = _float_type(dialect)
        ddl = f"""
        CREATE TABLE IF NOT EXISTS {table} (
            event_id TEXT PRIMARY KEY,
            strategy_id TEXT NOT NULL,
            trade_id TEXT NOT NULL,
            intent_id TEXT,
            symbol TEXT,
            side TEXT,
            status TEXT,
            policy_name TEXT,
            quantity {float_type},
            price {float_type},
            notional {float_type},
            passed INTEGER NOT NULL,
            worst_severity TEXT,
            violations TEXT,
            checks TEXT,
            totals TEXT,
            snapshot TEXT,
            recorded_at {ts_type} NOT NULL,
            execution_ts {ts_type} NOT NULL
        )
        """
        conn.execute(text(ddl))

        if dialect == "postgresql":
            index_stmt = text(
                "CREATE INDEX IF NOT EXISTS telemetry_compliance_audit_strategy_idx "
                "ON telemetry.compliance_audit (strategy_id, recorded_at DESC)"
            )
        else:
            index_stmt = text(
                f"CREATE INDEX IF NOT EXISTS telemetry_compliance_audit_strategy_idx "
                f"ON {table} (strategy_id, recorded_at DESC)"
            )
        conn.execute(index_stmt)

    def _create_compliance_kyc(self, conn: Connection, dialect: str) -> None:
        if dialect == "postgresql":
            conn.execute(text("CREATE SCHEMA IF NOT EXISTS telemetry"))

        table = _table_name(*self._COMPLIANCE_KYC, dialect)
        ts_type = _timestamp_type(dialect)
        float_type = _float_type(dialect)
        ddl = f"""
        CREATE TABLE IF NOT EXISTS {table} (
            event_id TEXT PRIMARY KEY,
            strategy_id TEXT NOT NULL,
            case_id TEXT NOT NULL,
            entity_id TEXT NOT NULL,
            entity_type TEXT,
            status TEXT,
            risk_rating TEXT,
            risk_score {float_type},
            watchlist_hits TEXT,
            outstanding_items TEXT,
            checklist TEXT,
            alerts TEXT,
            metadata TEXT,
            recorded_at {ts_type} NOT NULL,
            last_review_at {ts_type},
            next_review_due {ts_type},
            evaluated_at {ts_type} NOT NULL,
            assigned_to TEXT,
            channel TEXT,
            snapshot TEXT
        )
        """
        conn.execute(text(ddl))

        if dialect == "postgresql":
            idx_strategy = text(
                "CREATE INDEX IF NOT EXISTS telemetry_compliance_kyc_strategy_idx "
                "ON telemetry.compliance_kyc (strategy_id, recorded_at DESC)"
            )
            idx_entity = text(
                "CREATE INDEX IF NOT EXISTS telemetry_compliance_kyc_entity_idx "
                "ON telemetry.compliance_kyc (entity_id, recorded_at DESC)"
            )
        else:
            idx_strategy = text(
                f"CREATE INDEX IF NOT EXISTS telemetry_compliance_kyc_strategy_idx "
                f"ON {table} (strategy_id, recorded_at DESC)"
            )
            idx_entity = text(
                f"CREATE INDEX IF NOT EXISTS telemetry_compliance_kyc_entity_idx "
                f"ON {table} (entity_id, recorded_at DESC)"
            )
        conn.execute(idx_strategy)
        conn.execute(idx_entity)

    def _create_execution_snapshots(self, conn: Connection, dialect: str) -> None:
        if dialect == "postgresql":
            conn.execute(text("CREATE SCHEMA IF NOT EXISTS telemetry"))

        table = _table_name(*self._EXECUTION_SNAPSHOTS, dialect)
        ts_type = _timestamp_type(dialect)
        float_type = _float_type(dialect)
        json_type = "JSONB" if dialect == "postgresql" else "TEXT"
        ddl = f"""
        CREATE TABLE IF NOT EXISTS {table} (
            snapshot_id TEXT PRIMARY KEY,
            service TEXT NOT NULL,
            strategy_id TEXT,
            status TEXT NOT NULL,
            fill_rate {float_type},
            failure_rate {float_type},
            orders_submitted INTEGER,
            orders_executed INTEGER,
            orders_failed INTEGER,
            pending_orders INTEGER,
            avg_latency_ms {float_type},
            max_latency_ms {float_type},
            drop_copy_lag_seconds {float_type},
            drop_copy_active BOOLEAN,
            connection_healthy BOOLEAN,
            sessions_active {json_type},
            issues {json_type},
            policy {json_type},
            state {json_type},
            metadata {json_type},
            recorded_at {ts_type} NOT NULL DEFAULT {_now_function(dialect)}
        )
        """
        conn.execute(text(ddl))

        if dialect == "postgresql":
            service_idx = text(
                "CREATE INDEX IF NOT EXISTS telemetry_execution_snapshots_service_idx "
                "ON telemetry.execution_snapshots (service, recorded_at DESC)"
            )
            recorded_idx = text(
                "CREATE INDEX IF NOT EXISTS telemetry_execution_snapshots_recorded_idx "
                "ON telemetry.execution_snapshots (recorded_at DESC)"
            )
        else:
            service_idx = text(
                f"CREATE INDEX IF NOT EXISTS telemetry_execution_snapshots_service_idx "
                f"ON {table} (service, recorded_at DESC)"
            )
            recorded_idx = text(
                f"CREATE INDEX IF NOT EXISTS telemetry_execution_snapshots_recorded_idx "
                f"ON {table} (recorded_at DESC)"
            )
        conn.execute(service_idx)
        conn.execute(recorded_idx)

    def _create_configuration_audit(self, conn: Connection, dialect: str) -> None:
        if dialect == "postgresql":
            conn.execute(text("CREATE SCHEMA IF NOT EXISTS telemetry"))

        table = _table_name(*self._CONFIGURATION_AUDIT, dialect)
        ts_type = _timestamp_type(dialect)
        ddl = f"""
        CREATE TABLE IF NOT EXISTS {table} (
            snapshot_id TEXT PRIMARY KEY,
            applied_at {ts_type} NOT NULL,
            severity TEXT NOT NULL,
            changes TEXT,
            current_config TEXT NOT NULL,
            previous_config TEXT,
            metadata TEXT
        )
        """
        conn.execute(text(ddl))

        index_stmt = text(
            f"CREATE INDEX IF NOT EXISTS telemetry_configuration_audit_applied_idx "
            f"ON {table} (applied_at DESC)"
        )
        conn.execute(index_stmt)


# ---------------------------------------------------------------------------
# Ingest pipeline
# ---------------------------------------------------------------------------


def _chunk_records(
    records: Sequence[dict[str, object]], chunk_size: int
) -> Iterable[Sequence[dict[str, object]]]:
    size = max(int(chunk_size), 1)
    for start in range(0, len(records), size):
        yield records[start : start + size]


def _normalize_sqlite_records(
    records: Sequence[dict[str, object]], datetime_keys: Sequence[str]
) -> None:
    for record in records:
        for key in datetime_keys:
            value = record.get(key)
            if isinstance(value, datetime) and value.tzinfo is not None:
                record[key] = value.astimezone(UTC).replace(tzinfo=None)


def _prepare_daily_bar_records(
    df: pd.DataFrame, source: str, ingest_ts: datetime
) -> list[dict[str, object]]:
    frame = df.copy()
    frame = frame.rename(columns={"date": "ts", "time": "ts", "timestamp": "ts"})
    if "ts" not in frame.columns:
        raise ValueError("DataFrame must contain a 'date' or 'timestamp' column")

    frame["ts"] = pd.to_datetime(frame["ts"], utc=True)
    frame["symbol"] = frame["symbol"].astype(str)

    numeric_columns = ["open", "high", "low", "close", "adj_close", "volume"]
    for column in numeric_columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        else:
            frame[column] = pd.NA

    frame["source"] = source
    frame["ingested_at"] = ingest_ts

    ordered = [
        "ts",
        "symbol",
        "open",
        "high",
        "low",
        "close",
        "adj_close",
        "volume",
        "source",
        "ingested_at",
    ]
    frame = frame[ordered]
    frame = frame.where(pd.notnull(frame), None)

    records = cast(list[dict[str, object]], frame.to_dict("records"))
    for record in records:
        ts_value = record.get("ts")
        if isinstance(ts_value, pd.Timestamp):
            record["ts"] = ts_value.to_pydatetime()
        ingested_value = record.get("ingested_at")
        if isinstance(ingested_value, pd.Timestamp):
            record["ingested_at"] = ingested_value.to_pydatetime()
    return records


def _prepare_intraday_trade_records(
    df: pd.DataFrame, source: str, ingest_ts: datetime
) -> list[dict[str, object]]:
    frame = df.copy()
    frame = frame.rename(columns={"timestamp": "ts", "time": "ts"})
    if "ts" not in frame.columns:
        raise ValueError("DataFrame must contain a 'timestamp' column")
    if "symbol" not in frame.columns:
        raise ValueError("DataFrame must contain a 'symbol' column")
    if "price" not in frame.columns:
        raise ValueError("DataFrame must contain a 'price' column")

    frame["ts"] = pd.to_datetime(frame["ts"], utc=True)
    frame["symbol"] = frame["symbol"].astype(str)
    frame["price"] = pd.to_numeric(frame["price"], errors="coerce")
    if "size" in frame.columns:
        frame["size"] = pd.to_numeric(frame["size"], errors="coerce")
    else:
        frame["size"] = pd.NA
    if "exchange" not in frame.columns:
        frame["exchange"] = None
    if "conditions" not in frame.columns:
        frame["conditions"] = None

    frame["source"] = source
    frame["ingested_at"] = ingest_ts

    ordered = [
        "ts",
        "symbol",
        "price",
        "size",
        "exchange",
        "conditions",
        "source",
        "ingested_at",
    ]
    frame = frame[ordered]
    frame = frame.where(pd.notnull(frame), None)

    records = cast(list[dict[str, object]], frame.to_dict("records"))
    for record in records:
        ts_value = record.get("ts")
        if isinstance(ts_value, pd.Timestamp):
            record["ts"] = ts_value.to_pydatetime()
        ingested_value = record.get("ingested_at")
        if isinstance(ingested_value, pd.Timestamp):
            record["ingested_at"] = ingested_value.to_pydatetime()
    return records


def _prepare_macro_event_records(
    df: pd.DataFrame, source: str, ingest_ts: datetime
) -> list[dict[str, object]]:
    frame = df.copy()
    frame = frame.rename(columns={"timestamp": "ts", "datetime": "ts"})
    if "ts" not in frame.columns:
        raise ValueError("DataFrame must contain a 'timestamp' column")

    frame["ts"] = pd.to_datetime(frame["ts"], utc=True)
    if "event_name" not in frame.columns:
        if "event" in frame.columns:
            frame["event_name"] = frame["event"]
        else:
            raise ValueError("DataFrame must contain an 'event' or 'event_name' column")

    if "calendar" not in frame.columns:
        frame["calendar"] = None
    if "currency" not in frame.columns:
        frame["currency"] = None
    for column in ("actual", "forecast", "previous"):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        else:
            frame[column] = pd.NA
    if "importance" not in frame.columns:
        frame["importance"] = None

    frame["source"] = source
    frame["ingested_at"] = ingest_ts

    ordered = [
        "ts",
        "event_name",
        "calendar",
        "currency",
        "actual",
        "forecast",
        "previous",
        "importance",
        "source",
        "ingested_at",
    ]
    frame = frame[ordered]
    frame = frame.where(pd.notnull(frame), None)

    records = cast(list[dict[str, object]], frame.to_dict("records"))
    for record in records:
        ts_value = record.get("ts")
        if isinstance(ts_value, pd.Timestamp):
            record["ts"] = ts_value.to_pydatetime()
        ingested_value = record.get("ingested_at")
        if isinstance(ingested_value, pd.Timestamp):
            record["ingested_at"] = ingested_value.to_pydatetime()
    return records


@dataclass(frozen=True)
class TimescaleIngestResult:
    rows_written: int
    symbols: tuple[str, ...]
    start_ts: datetime | None
    end_ts: datetime | None
    ingest_duration_seconds: float
    freshness_seconds: float | None
    dimension: str = "daily_bars"
    source: str | None = None

    @classmethod
    def empty(
        cls,
        *,
        dimension: str = "daily_bars",
        source: str | None = None,
    ) -> "TimescaleIngestResult":
        return cls(0, tuple(), None, None, 0.0, None, dimension, source)

    def as_dict(self) -> dict[str, object]:
        return {
            "dimension": self.dimension,
            "rows_written": self.rows_written,
            "symbols": list(self.symbols),
            "start_ts": self.start_ts.isoformat() if self.start_ts else None,
            "end_ts": self.end_ts.isoformat() if self.end_ts else None,
            "ingest_duration_seconds": self.ingest_duration_seconds,
            "freshness_seconds": self.freshness_seconds,
            "source": self.source,
        }


def _coerce_timestamp(value: object) -> datetime:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError:
            return datetime.fromtimestamp(0, tz=UTC)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        return parsed.astimezone(UTC)
    raise TypeError(f"Unsupported timestamp value: {value!r}")


@dataclass(frozen=True)
class TimescaleIngestRunRecord:
    """Structured record describing a Timescale ingest attempt."""

    run_id: str
    dimension: str
    status: str
    rows_written: int
    freshness_seconds: float | None
    ingest_duration_seconds: float | None
    executed_at: datetime
    source: str | None = None
    symbols: tuple[str, ...] = field(default_factory=tuple)
    metadata: Mapping[str, object] = field(default_factory=dict)

    def _serialise(self, value: Mapping[str, object]) -> str | None:
        if not value:
            return None
        try:
            return json.dumps(value)
        except (TypeError, ValueError):
            logger.debug("Failed to serialise ingest run metadata", exc_info=True)
            safe = {str(k): str(v) for k, v in value.items()}
            return json.dumps(safe)

    def _serialise_symbols(self) -> str | None:
        if not self.symbols:
            return None
        try:
            return json.dumps(list(self.symbols))
        except (TypeError, ValueError):
            logger.debug("Failed to serialise ingest run symbols", exc_info=True)
            return json.dumps([str(symbol) for symbol in self.symbols])

    def as_row(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "dimension": self.dimension,
            "status": self.status,
            "rows_written": int(self.rows_written),
            "freshness_seconds": float(self.freshness_seconds)
            if self.freshness_seconds is not None
            else None,
            "ingest_duration_seconds": float(self.ingest_duration_seconds)
            if self.ingest_duration_seconds is not None
            else None,
            "executed_at": self.executed_at,
            "source": self.source,
            "symbols": self._serialise_symbols(),
            "metadata": self._serialise(self.metadata),
        }

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable mapping of the ingest run record."""

        payload: dict[str, object] = {
            "run_id": self.run_id,
            "dimension": self.dimension,
            "status": self.status,
            "rows_written": int(self.rows_written),
            "freshness_seconds": (
                float(self.freshness_seconds) if self.freshness_seconds is not None else None
            ),
            "ingest_duration_seconds": (
                float(self.ingest_duration_seconds)
                if self.ingest_duration_seconds is not None
                else None
            ),
            "executed_at": self.executed_at.astimezone(UTC).isoformat(),
            "source": self.source,
            "symbols": list(self.symbols),
            "metadata": dict(self.metadata),
        }
        return payload

    @classmethod
    def from_row(
        cls, row: Mapping[str, Any] | Row[Any] | RowMapping
    ) -> "TimescaleIngestRunRecord":  # pragma: no cover - simple glue
        mapping = _ensure_mapping(row)

        metadata = _load_json_mapping(mapping.get("metadata"))
        symbols_payload = _load_json_sequence(mapping.get("symbols"))
        symbols = tuple(str(item) for item in symbols_payload if item is not None)

        freshness = coerce_float(mapping.get("freshness_seconds"))
        duration = coerce_float(mapping.get("ingest_duration_seconds"))

        source_raw = mapping.get("source")
        source = str(source_raw) if source_raw is not None else None

        return cls(
            run_id=str(mapping["run_id"]),
            dimension=str(mapping["dimension"]),
            status=str(mapping["status"]),
            rows_written=coerce_int(mapping.get("rows_written"), default=0),
            freshness_seconds=freshness,
            ingest_duration_seconds=duration,
            executed_at=_coerce_timestamp(mapping["executed_at"]),
            source=source,
            symbols=symbols,
            metadata=metadata,
        )


@dataclass(frozen=True)
class TimescaleComplianceAuditRecord:
    """Structured compliance journal entry persisted to Timescale."""

    event_id: str
    strategy_id: str
    trade_id: str
    intent_id: str | None
    symbol: str
    side: str
    status: str
    policy_name: str
    quantity: float | None
    price: float | None
    notional: float | None
    recorded_at: datetime
    execution_ts: datetime
    passed: bool
    worst_severity: str | None
    violations: tuple[str, ...]
    checks: tuple[Mapping[str, Any], ...]
    totals: Mapping[str, Any]
    snapshot: Mapping[str, Any]

    @staticmethod
    def _as_float(value: Any) -> float | None:
        if value is None:
            return None
        if isinstance(value, str) and not value.strip():
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @classmethod
    def from_snapshot(
        cls,
        snapshot: Mapping[str, Any],
        *,
        strategy_id: str,
        recorded_at: datetime | None = None,
        event_id: str | None = None,
    ) -> "TimescaleComplianceAuditRecord":
        payload = json.loads(json.dumps(snapshot, default=_json_default))

        trade_id = str(payload.get("trade_id") or uuid4())
        intent_raw = payload.get("intent_id")
        intent_id = str(intent_raw) if intent_raw is not None else None
        symbol = str(payload.get("symbol") or "UNKNOWN").upper()
        side = str(payload.get("side") or "UNKNOWN").upper()
        status = str(payload.get("status") or "UNKNOWN").upper()
        policy_name = str(payload.get("policy_name") or "default")

        quantity = cls._as_float(payload.get("quantity"))
        price = cls._as_float(payload.get("price"))
        notional = cls._as_float(payload.get("notional"))

        execution_raw = payload.get("timestamp")
        execution_ts = (
            _coerce_timestamp(execution_raw) if execution_raw is not None else datetime.now(tz=UTC)
        )
        recorded = recorded_at or datetime.now(tz=UTC)

        checks_raw = payload.get("checks")
        checks: list[Mapping[str, Any]] = []
        if isinstance(checks_raw, list):
            for item in checks_raw:
                if isinstance(item, dict):
                    checks.append(item)

        totals_raw = payload.get("totals")
        totals: Mapping[str, Any]
        if isinstance(totals_raw, dict):
            totals = totals_raw
        else:
            totals = {}

        violations = tuple(
            str(check.get("message", "")).strip()
            for check in checks
            if not bool(check.get("passed"))
        )
        passed = all(bool(check.get("passed")) for check in checks) if checks else True

        worst = None
        worst_rank = -1
        for check in checks:
            if bool(check.get("passed")):
                continue
            severity = str(check.get("severity", "")).lower()
            rank = _SEVERITY_RANK.get(severity, 0)
            if rank > worst_rank:
                worst = severity
                worst_rank = rank

        event_identifier = event_id or f"{trade_id}-{uuid4()}"

        return cls(
            event_id=event_identifier,
            strategy_id=str(strategy_id),
            trade_id=trade_id,
            intent_id=intent_id,
            symbol=symbol,
            side=side,
            status=status,
            policy_name=policy_name,
            quantity=quantity,
            price=price,
            notional=notional,
            recorded_at=recorded,
            execution_ts=execution_ts,
            passed=passed,
            worst_severity=worst,
            violations=violations,
            checks=tuple(checks),
            totals=totals,
            snapshot=payload,
        )

    def as_row(self) -> dict[str, object]:
        return {
            "event_id": self.event_id,
            "strategy_id": self.strategy_id,
            "trade_id": self.trade_id,
            "intent_id": self.intent_id,
            "symbol": self.symbol,
            "side": self.side,
            "status": self.status,
            "policy_name": self.policy_name,
            "quantity": float(self.quantity) if self.quantity is not None else None,
            "price": float(self.price) if self.price is not None else None,
            "notional": float(self.notional) if self.notional is not None else None,
            "passed": 1 if self.passed else 0,
            "worst_severity": self.worst_severity,
            "violations": json.dumps(list(self.violations), default=_json_default)
            if self.violations
            else None,
            "checks": json.dumps([dict(check) for check in self.checks], default=_json_default),
            "totals": json.dumps(dict(self.totals), default=_json_default) if self.totals else None,
            "snapshot": json.dumps(dict(self.snapshot), default=_json_default),
            "recorded_at": self.recorded_at,
            "execution_ts": self.execution_ts,
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "strategy_id": self.strategy_id,
            "trade_id": self.trade_id,
            "intent_id": self.intent_id,
            "symbol": self.symbol,
            "side": self.side,
            "status": self.status,
            "policy_name": self.policy_name,
            "quantity": self.quantity,
            "price": self.price,
            "notional": self.notional,
            "passed": self.passed,
            "worst_severity": self.worst_severity,
            "violations": list(self.violations),
            "checks": [dict(check) for check in self.checks],
            "totals": dict(self.totals),
            "snapshot": dict(self.snapshot),
            "recorded_at": self.recorded_at.astimezone(UTC).isoformat(),
            "execution_ts": self.execution_ts.astimezone(UTC).isoformat(),
        }

    @classmethod
    def from_row(
        cls, row: Mapping[str, Any] | Row[Any] | RowMapping
    ) -> "TimescaleComplianceAuditRecord":  # pragma: no cover - simple glue
        mapping = _ensure_mapping(row)

        violations_raw = _load_json_sequence(mapping.get("violations"))
        violations = tuple(str(item) for item in violations_raw if item is not None)

        checks_raw = _load_json_sequence(mapping.get("checks"))
        checks = [dict(item) for item in checks_raw if isinstance(item, Mapping)]

        totals = _load_json_mapping(mapping.get("totals"))

        snapshot = _load_json_mapping(mapping.get("snapshot"))

        intent_raw = mapping.get("intent_id")
        passed_raw = mapping.get("passed")
        if isinstance(passed_raw, str):
            passed = passed_raw.strip().lower() in {"1", "true", "t", "yes"}
        else:
            passed = bool(passed_raw)

        worst = mapping.get("worst_severity")
        worst_severity = str(worst) if worst is not None else None

        return cls(
            event_id=str(mapping["event_id"]),
            strategy_id=str(mapping["strategy_id"]),
            trade_id=str(mapping["trade_id"]),
            intent_id=str(intent_raw) if intent_raw is not None else None,
            symbol=str(mapping.get("symbol") or "UNKNOWN"),
            side=str(mapping.get("side") or "UNKNOWN"),
            status=str(mapping.get("status") or "UNKNOWN"),
            policy_name=str(mapping.get("policy_name") or "default"),
            quantity=cls._as_float(mapping.get("quantity")),
            price=cls._as_float(mapping.get("price")),
            notional=cls._as_float(mapping.get("notional")),
            recorded_at=_coerce_timestamp(mapping.get("recorded_at")),
            execution_ts=_coerce_timestamp(mapping.get("execution_ts")),
            passed=passed,
            worst_severity=worst_severity,
            violations=violations,
            checks=tuple(checks),
            totals=totals,
            snapshot=snapshot,
        )


class TimescaleComplianceJournal:
    """Persist trade compliance snapshots for downstream telemetry consumers."""

    def __init__(self, engine: Engine) -> None:
        self._engine = engine
        self._logger = logging.getLogger(f"{__name__}.TimescaleComplianceJournal")

    def record_snapshot(
        self,
        snapshot: Mapping[str, Any],
        *,
        strategy_id: str,
    ) -> dict[str, Any]:
        record = TimescaleComplianceAuditRecord.from_snapshot(snapshot, strategy_id=strategy_id)
        row = record.as_row()

        with self._engine.begin() as conn:
            table = _reflect_table(conn, TimescaleMigrator._COMPLIANCE_AUDIT)
            conn.execute(table.insert().values(**row))

        return record.as_dict()

    def fetch_recent(
        self,
        *,
        limit: int = 5,
        strategy_id: str | None = None,
    ) -> list[dict[str, Any]]:
        limit = max(1, int(limit))
        with self._engine.connect() as conn:
            table = _reflect_table(conn, TimescaleMigrator._COMPLIANCE_AUDIT)
            columns = [
                table.c.event_id,
                table.c.strategy_id,
                table.c.trade_id,
                table.c.intent_id,
                table.c.symbol,
                table.c.side,
                table.c.status,
                table.c.policy_name,
                table.c.quantity,
                table.c.price,
                table.c.notional,
                table.c.passed,
                table.c.worst_severity,
                table.c.violations,
                table.c.checks,
                table.c.totals,
                table.c.snapshot,
                table.c.recorded_at,
                table.c.execution_ts,
            ]

            stmt = select(*columns).order_by(table.c.recorded_at.desc()).limit(limit)
            if strategy_id is not None:
                stmt = stmt.where(table.c.strategy_id == strategy_id)

            rows = conn.execute(stmt).fetchall()

        return [TimescaleComplianceAuditRecord.from_row(row).as_dict() for row in rows]

    def summarise(
        self,
        *,
        strategy_id: str | None = None,
    ) -> dict[str, Any]:
        """Aggregate compliance audit journal statistics."""

        with self._engine.connect() as conn:
            table = _reflect_table(conn, TimescaleMigrator._COMPLIANCE_AUDIT)
            conditions = []
            if strategy_id is not None:
                conditions.append(table.c.strategy_id == strategy_id)

            summary_stmt = select(
                func.count().label("total_records"),
                func.sum(
                    case((table.c.passed == 1, 1), else_=0)
                ).label("passed_records"),
                func.max(table.c.recorded_at).label("last_recorded_at"),
            )
            for condition in conditions:
                summary_stmt = summary_stmt.where(condition)

            row = conn.execute(summary_stmt).first()
            mapping = row._mapping if row is not None else {}

            total = int(mapping.get("total_records") or 0)
            passed = int(mapping.get("passed_records") or 0)
            failed = max(total - passed, 0)
            last_recorded = _normalise_timestamp(mapping.get("last_recorded_at"))

            severity_stmt = select(
                table.c.worst_severity,
                func.count().label("count"),
            ).group_by(table.c.worst_severity)
            for condition in conditions:
                severity_stmt = severity_stmt.where(condition)

            severity_rows = conn.execute(severity_stmt).fetchall()
            severity_counts = {}
            for row in severity_rows:
                severity = (
                    row._mapping.get("worst_severity") if hasattr(row, "_mapping") else row[0]
                )
                key = str(severity) if severity not in (None, "") else "unknown"
                severity_counts[key] = int(
                    (row._mapping.get("count") if hasattr(row, "_mapping") else row[1]) or 0
                )

            status_stmt = select(
                table.c.status,
                func.count().label("count"),
            ).group_by(table.c.status)
            for condition in conditions:
                status_stmt = status_stmt.where(condition)

            status_rows = conn.execute(status_stmt).fetchall()
            status_counts = {}
            for row in status_rows:
                status = row._mapping.get("status") if hasattr(row, "_mapping") else row[0]
                key = str(status) if status not in (None, "") else "UNKNOWN"
                status_counts[key] = int(
                    (row._mapping.get("count") if hasattr(row, "_mapping") else row[1]) or 0
                )

        return {
            "total_records": total,
            "passed_records": passed,
            "failed_records": failed,
            "last_recorded_at": last_recorded,
            "severity_counts": severity_counts,
            "status_counts": status_counts,
        }

    def close(self) -> None:
        try:
            self._engine.dispose()
        except Exception:  # pragma: no cover - defensive cleanup
            self._logger.debug("Failed to dispose compliance journal engine", exc_info=True)


@dataclass(frozen=True)
class TimescaleKycCaseRecord:
    """Structured KYC/AML snapshot stored in Timescale."""

    event_id: str
    strategy_id: str
    case_id: str
    entity_id: str
    entity_type: str
    status: str
    risk_rating: str
    risk_score: float | None
    watchlist_hits: tuple[str, ...]
    outstanding_items: tuple[str, ...]
    checklist: tuple[Mapping[str, Any], ...]
    alerts: tuple[str, ...]
    metadata: Mapping[str, Any]
    recorded_at: datetime
    last_review_at: datetime | None
    next_review_due: datetime | None
    evaluated_at: datetime
    assigned_to: str | None
    channel: str | None
    snapshot: Mapping[str, Any]

    @staticmethod
    def _as_float(value: Any) -> float | None:
        if value is None:
            return None
        if isinstance(value, str) and not value.strip():
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @classmethod
    def from_snapshot(
        cls,
        snapshot: Mapping[str, Any],
        *,
        strategy_id: str,
        recorded_at: datetime | None = None,
        event_id: str | None = None,
    ) -> "TimescaleKycCaseRecord":
        payload = json.loads(json.dumps(snapshot, default=_json_default))

        case_id = str(payload.get("case_id") or uuid4())
        entity_id = str(payload.get("entity_id") or "UNKNOWN")
        entity_type = str(payload.get("entity_type") or "client")
        status = str(payload.get("status") or "UNKNOWN").upper()
        risk_rating = str(payload.get("risk_rating") or "UNKNOWN").upper()
        risk_score = cls._as_float(payload.get("risk_score"))

        def _parse_list(key: str) -> tuple[str, ...]:
            raw = payload.get(key)
            if isinstance(raw, (list, tuple, set)):
                return tuple(str(item).strip() for item in raw if str(item).strip())
            if raw in (None, ""):
                return tuple()
            return tuple(token.strip() for token in str(raw).split(",") if token.strip())

        watchlist_hits = _parse_list("watchlist_hits")
        outstanding_items = _parse_list("outstanding_items")
        alerts = _parse_list("alerts")

        checklist_entries: list[Mapping[str, Any]] = []
        checklist_raw = payload.get("checklist")
        if isinstance(checklist_raw, list):
            for entry in checklist_raw:
                if isinstance(entry, dict):
                    checklist_entries.append(entry)

        metadata_raw = payload.get("metadata")
        metadata = metadata_raw if isinstance(metadata_raw, dict) else {}

        recorded = recorded_at or datetime.now(tz=UTC)
        last_review = payload.get("last_reviewed_at")
        next_review = payload.get("next_review_due")
        evaluated = (
            payload.get("evaluated_at") or payload.get("observed_at") or recorded.isoformat()
        )

        assigned = payload.get("assigned_to")
        assigned_to = str(assigned).strip() if assigned not in (None, "") else None
        channel = payload.get("report_channel")
        channel_str = str(channel) if channel not in (None, "") else None

        event_identifier = event_id or f"{case_id}-{uuid4()}"

        return cls(
            event_id=event_identifier,
            strategy_id=str(strategy_id),
            case_id=case_id,
            entity_id=entity_id,
            entity_type=entity_type,
            status=status,
            risk_rating=risk_rating,
            risk_score=risk_score,
            watchlist_hits=watchlist_hits,
            outstanding_items=outstanding_items,
            checklist=tuple(checklist_entries),
            alerts=alerts,
            metadata=metadata,
            recorded_at=recorded,
            last_review_at=_coerce_timestamp(last_review) if last_review is not None else None,
            next_review_due=_coerce_timestamp(next_review) if next_review is not None else None,
            evaluated_at=_coerce_timestamp(evaluated),
            assigned_to=assigned_to,
            channel=channel_str,
            snapshot=payload,
        )

    def as_row(self) -> dict[str, object]:
        return {
            "event_id": self.event_id,
            "strategy_id": self.strategy_id,
            "case_id": self.case_id,
            "entity_id": self.entity_id,
            "entity_type": self.entity_type,
            "status": self.status,
            "risk_rating": self.risk_rating,
            "risk_score": float(self.risk_score) if self.risk_score is not None else None,
            "watchlist_hits": json.dumps(list(self.watchlist_hits), default=_json_default)
            if self.watchlist_hits
            else None,
            "outstanding_items": json.dumps(list(self.outstanding_items), default=_json_default)
            if self.outstanding_items
            else None,
            "checklist": json.dumps(
                [dict(entry) for entry in self.checklist], default=_json_default
            )
            if self.checklist
            else None,
            "alerts": json.dumps(list(self.alerts), default=_json_default) if self.alerts else None,
            "metadata": json.dumps(dict(self.metadata), default=_json_default)
            if self.metadata
            else None,
            "recorded_at": self.recorded_at,
            "last_review_at": self.last_review_at,
            "next_review_due": self.next_review_due,
            "evaluated_at": self.evaluated_at,
            "assigned_to": self.assigned_to,
            "channel": self.channel,
            "snapshot": json.dumps(dict(self.snapshot), default=_json_default),
        }

    def as_dict(self) -> dict[str, Any]:
        payload = {
            "event_id": self.event_id,
            "strategy_id": self.strategy_id,
            "case_id": self.case_id,
            "entity_id": self.entity_id,
            "entity_type": self.entity_type,
            "status": self.status,
            "risk_rating": self.risk_rating,
            "risk_score": self.risk_score,
            "watchlist_hits": list(self.watchlist_hits),
            "outstanding_items": list(self.outstanding_items),
            "checklist": [dict(entry) for entry in self.checklist],
            "alerts": list(self.alerts),
            "metadata": dict(self.metadata),
            "recorded_at": self.recorded_at.astimezone(UTC).isoformat(),
            "last_reviewed_at": self.last_review_at.astimezone(UTC).isoformat()
            if self.last_review_at is not None
            else None,
            "next_review_due": self.next_review_due.astimezone(UTC).isoformat()
            if self.next_review_due is not None
            else None,
            "evaluated_at": self.evaluated_at.astimezone(UTC).isoformat(),
            "assigned_to": self.assigned_to,
            "channel": self.channel,
            "snapshot": dict(self.snapshot),
        }
        return payload

    @classmethod
    def from_row(
        cls, row: Mapping[str, Any] | Row[Any] | RowMapping
    ) -> "TimescaleKycCaseRecord":  # pragma: no cover - glue
        mapping = _ensure_mapping(row)

        watchlist_raw = _load_json_sequence(mapping.get("watchlist_hits"))
        outstanding_raw = _load_json_sequence(mapping.get("outstanding_items"))
        checklist_raw = _load_json_sequence(mapping.get("checklist"))
        alerts_raw = _load_json_sequence(mapping.get("alerts"))
        metadata_raw = _load_json_mapping(mapping.get("metadata"))
        snapshot_raw = _load_json_mapping(mapping.get("snapshot"))

        risk_score = coerce_float(mapping.get("risk_score"))

        return cls(
            event_id=str(mapping["event_id"]),
            strategy_id=str(mapping["strategy_id"]),
            case_id=str(mapping["case_id"]),
            entity_id=str(mapping["entity_id"]),
            entity_type=str(mapping.get("entity_type") or "client"),
            status=str(mapping.get("status") or "UNKNOWN"),
            risk_rating=str(mapping.get("risk_rating") or "UNKNOWN"),
            risk_score=risk_score,
            watchlist_hits=tuple(str(item) for item in watchlist_raw if item is not None),
            outstanding_items=tuple(str(item) for item in outstanding_raw if item is not None),
            checklist=tuple(dict(item) for item in checklist_raw if isinstance(item, Mapping)),
            alerts=tuple(str(item) for item in alerts_raw if item is not None),
            metadata=metadata_raw,
            recorded_at=_coerce_timestamp(mapping.get("recorded_at")),
            last_review_at=_coerce_timestamp(mapping.get("last_review_at"))
            if mapping.get("last_review_at") is not None
            else None,
            next_review_due=_coerce_timestamp(mapping.get("next_review_due"))
            if mapping.get("next_review_due") is not None
            else None,
            evaluated_at=_coerce_timestamp(mapping.get("evaluated_at")),
            assigned_to=str(mapping.get("assigned_to"))
            if mapping.get("assigned_to") is not None
            else None,
            channel=str(mapping.get("channel")) if mapping.get("channel") else None,
            snapshot=snapshot_raw if isinstance(snapshot_raw, dict) else {},
        )


class TimescaleKycJournal:
    """Persist KYC/AML case snapshots for institutional audits."""

    def __init__(self, engine: Engine) -> None:
        self._engine = engine
        self._logger = logging.getLogger(f"{__name__}.TimescaleKycJournal")

    def record_case(
        self,
        snapshot: Mapping[str, Any],
        *,
        strategy_id: str,
    ) -> dict[str, Any]:
        record = TimescaleKycCaseRecord.from_snapshot(snapshot, strategy_id=strategy_id)
        row = record.as_row()

        with self._engine.begin() as conn:
            table = _reflect_table(conn, TimescaleMigrator._COMPLIANCE_KYC)
            conn.execute(table.insert().values(**row))

        return record.as_dict()

    def fetch_recent(
        self,
        *,
        limit: int = 5,
        strategy_id: str | None = None,
        entity_id: str | None = None,
    ) -> list[dict[str, Any]]:
        limit = max(1, int(limit))
        with self._engine.connect() as conn:
            table = _reflect_table(conn, TimescaleMigrator._COMPLIANCE_KYC)
            columns = [
                table.c.event_id,
                table.c.strategy_id,
                table.c.case_id,
                table.c.entity_id,
                table.c.entity_type,
                table.c.status,
                table.c.risk_rating,
                table.c.risk_score,
                table.c.watchlist_hits,
                table.c.outstanding_items,
                table.c.checklist,
                table.c.alerts,
                table.c.metadata,
                table.c.recorded_at,
                table.c.last_review_at,
                table.c.next_review_due,
                table.c.evaluated_at,
                table.c.assigned_to,
                table.c.channel,
                table.c.snapshot,
            ]
            stmt = select(*columns).order_by(table.c.recorded_at.desc()).limit(limit)
            if strategy_id is not None:
                stmt = stmt.where(table.c.strategy_id == strategy_id)
            if entity_id is not None:
                stmt = stmt.where(table.c.entity_id == entity_id)

            rows = conn.execute(stmt).fetchall()

        return [TimescaleKycCaseRecord.from_row(row).as_dict() for row in rows]

    def summarise(
        self,
        *,
        strategy_id: str | None = None,
        entity_id: str | None = None,
    ) -> dict[str, Any]:
        """Aggregate KYC case statistics for governance evidence."""

        with self._engine.connect() as conn:
            table = _reflect_table(conn, TimescaleMigrator._COMPLIANCE_KYC)
            conditions = []
            if strategy_id is not None:
                conditions.append(table.c.strategy_id == strategy_id)
            if entity_id is not None:
                conditions.append(table.c.entity_id == entity_id)

            summary_stmt = select(
                func.count().label("total_cases"),
                func.max(table.c.recorded_at).label("last_recorded_at"),
            )
            for condition in conditions:
                summary_stmt = summary_stmt.where(condition)

            row = conn.execute(summary_stmt).first()
            mapping = row._mapping if row is not None else {}

            total_cases = int(mapping.get("total_cases") or 0)
            last_recorded = _normalise_timestamp(mapping.get("last_recorded_at"))

            status_stmt = select(
                table.c.status,
                func.count().label("count"),
            ).group_by(table.c.status)
            for condition in conditions:
                status_stmt = status_stmt.where(condition)

            status_rows = conn.execute(status_stmt).fetchall()
            status_counts: dict[str, int] = {}
            for row in status_rows:
                status = row._mapping.get("status") if hasattr(row, "_mapping") else row[0]
                key = str(status) if status not in (None, "") else "UNKNOWN"
                status_counts[key] = int(
                    (row._mapping.get("count") if hasattr(row, "_mapping") else row[1]) or 0
                )

            rating_stmt = select(
                table.c.risk_rating,
                func.count().label("count"),
            ).group_by(table.c.risk_rating)
            for condition in conditions:
                rating_stmt = rating_stmt.where(condition)

            rating_rows = conn.execute(rating_stmt).fetchall()
            risk_counts: dict[str, int] = {}
            for row in rating_rows:
                rating = row._mapping.get("risk_rating") if hasattr(row, "_mapping") else row[0]
                key = str(rating) if rating not in (None, "") else "UNKNOWN"
                risk_counts[key] = int(
                    (row._mapping.get("count") if hasattr(row, "_mapping") else row[1]) or 0
                )

        return {
            "total_cases": total_cases,
            "last_recorded_at": last_recorded,
            "status_counts": status_counts,
            "risk_rating_counts": risk_counts,
        }

    def close(self) -> None:
        try:
            self._engine.dispose()
        except Exception:  # pragma: no cover - defensive cleanup
            self._logger.debug("Failed to dispose KYC journal engine", exc_info=True)


@dataclass(frozen=True)
class TimescaleExecutionSnapshotRecord:
    """Persisted execution readiness snapshot."""

    snapshot_id: str
    service: str
    status: str
    recorded_at: datetime
    orders_submitted: int
    orders_executed: int
    orders_failed: int
    pending_orders: int
    avg_latency_ms: float | None
    max_latency_ms: float | None
    fill_rate: float | None
    failure_rate: float | None
    drop_copy_lag_seconds: float | None
    drop_copy_active: bool | None
    connection_healthy: bool | None
    sessions_active: tuple[str, ...]
    issues: tuple[Mapping[str, Any], ...]
    policy: Mapping[str, Any]
    state: Mapping[str, Any]
    metadata: Mapping[str, Any]
    strategy_id: str | None = None

    @classmethod
    def from_snapshot(
        cls,
        snapshot: Any,
        *,
        strategy_id: str | None = None,
    ) -> "TimescaleExecutionSnapshotRecord":
        payload_candidate = getattr(snapshot, "as_dict", None)
        if callable(payload_candidate):
            raw_payload = payload_candidate()
            if isinstance(raw_payload, Mapping):
                payload = dict(raw_payload)
            else:  # pragma: no cover - defensive
                raise TypeError("Execution snapshot as_dict() must return a Mapping")
        elif isinstance(snapshot, Mapping):
            payload = dict(snapshot)
        else:  # pragma: no cover - defensive
            raise TypeError("Execution snapshot must provide as_dict() or Mapping interface")

        service = str(payload.get("service") or getattr(snapshot, "service", "execution"))
        status_obj = payload.get("status") or getattr(snapshot, "status", "unknown")
        status = str(getattr(status_obj, "value", status_obj))

        generated_at = getattr(snapshot, "generated_at", None)
        if isinstance(generated_at, datetime):
            recorded = (
                generated_at
                if generated_at.tzinfo is not None
                else generated_at.replace(tzinfo=UTC)
            )
        else:
            recorded = datetime.now(tz=UTC)

        state_payload = payload.get("state")
        if isinstance(state_payload, Mapping):
            state = dict(state_payload)
        else:
            state_attr = getattr(snapshot, "state", None)
            as_dict = getattr(state_attr, "as_dict", None)
            if callable(as_dict):
                state_candidate = as_dict()
                if isinstance(state_candidate, Mapping):
                    state = dict(state_candidate)
                else:
                    state = {}
            else:
                state = {}

        metadata_raw = payload.get("metadata")
        metadata: Mapping[str, Any]
        if isinstance(metadata_raw, Mapping):
            metadata = dict(metadata_raw)
        else:
            metadata = {}

        issues_raw = payload.get("issues")
        issues: tuple[Mapping[str, Any], ...] = tuple()
        if isinstance(issues_raw, Sequence):
            issues = tuple(
                dict(issue) if isinstance(issue, Mapping) else {"message": str(issue)}
                for issue in issues_raw
            )

        sessions_raw = state.get("sessions_active")
        if isinstance(sessions_raw, (list, tuple)):
            sessions_active = tuple(str(session) for session in sessions_raw)
        else:
            sessions_active = tuple()

        def _as_float(key: str) -> float | None:
            value = payload.get(key) if key in payload else state.get(key)
            if value is None:
                return None
            try:
                return float(value)
            except (TypeError, ValueError):
                return None

        fill_rate = _as_float("fill_rate")
        failure_rate = _as_float("failure_rate")
        avg_latency_ms = _as_float("avg_latency_ms")
        max_latency_ms = _as_float("max_latency_ms")
        drop_copy_lag_seconds = _as_float("drop_copy_lag_seconds")

        def _as_int(key: str) -> int:
            value = state.get(key) if key in state else payload.get(key)
            return coerce_int(value, default=0)

        orders_submitted = _as_int("orders_submitted")
        orders_executed = _as_int("orders_executed")
        orders_failed = _as_int("orders_failed")
        pending_orders = _as_int("pending_orders")

        def _as_bool(value: object) -> bool | None:
            if value is None:
                return None
            if isinstance(value, bool):
                return value
            text = str(value).strip().lower()
            if text in {"1", "true", "yes", "on", "y"}:
                return True
            if text in {"0", "false", "no", "off", "n"}:
                return False
            return None

        drop_copy_active = _as_bool(state.get("drop_copy_active"))
        if drop_copy_active is None:
            drop_copy_active = _as_bool(payload.get("drop_copy_active"))

        connection_healthy = _as_bool(state.get("connection_healthy"))
        if connection_healthy is None:
            connection_healthy = _as_bool(payload.get("connection_healthy"))

        policy_payload = payload.get("policy")
        if isinstance(policy_payload, Mapping):
            policy = dict(policy_payload)
        else:
            policy = {}

        snapshot_id = str(payload.get("snapshot_id") or uuid4())

        return cls(
            snapshot_id=snapshot_id,
            service=service,
            status=status,
            recorded_at=recorded,
            orders_submitted=orders_submitted,
            orders_executed=orders_executed,
            orders_failed=orders_failed,
            pending_orders=pending_orders,
            avg_latency_ms=avg_latency_ms,
            max_latency_ms=max_latency_ms,
            fill_rate=fill_rate,
            failure_rate=failure_rate,
            drop_copy_lag_seconds=drop_copy_lag_seconds,
            drop_copy_active=drop_copy_active,
            connection_healthy=connection_healthy,
            sessions_active=sessions_active,
            issues=issues,
            policy=policy,
            state=state,
            metadata=metadata,
            strategy_id=strategy_id,
        )

    def _serialise(self, value: Mapping[str, Any]) -> str | None:
        if not value:
            return None
        try:
            return json.dumps(value, default=_json_default)
        except (TypeError, ValueError):  # pragma: no cover - defensive
            safe = {str(k): str(v) for k, v in value.items()}
            return json.dumps(safe)

    def as_row(self) -> dict[str, object]:
        return {
            "snapshot_id": self.snapshot_id,
            "service": self.service,
            "strategy_id": self.strategy_id,
            "status": self.status,
            "fill_rate": float(self.fill_rate) if self.fill_rate is not None else None,
            "failure_rate": float(self.failure_rate) if self.failure_rate is not None else None,
            "orders_submitted": int(self.orders_submitted),
            "orders_executed": int(self.orders_executed),
            "orders_failed": int(self.orders_failed),
            "pending_orders": int(self.pending_orders),
            "avg_latency_ms": float(self.avg_latency_ms)
            if self.avg_latency_ms is not None
            else None,
            "max_latency_ms": float(self.max_latency_ms)
            if self.max_latency_ms is not None
            else None,
            "drop_copy_lag_seconds": float(self.drop_copy_lag_seconds)
            if self.drop_copy_lag_seconds is not None
            else None,
            "drop_copy_active": self.drop_copy_active,
            "connection_healthy": self.connection_healthy,
            "sessions_active": json.dumps(list(self.sessions_active), default=_json_default)
            if self.sessions_active
            else None,
            "issues": json.dumps([dict(issue) for issue in self.issues], default=_json_default)
            if self.issues
            else None,
            "policy": self._serialise(self.policy),
            "state": self._serialise(self.state),
            "metadata": self._serialise(self.metadata),
            "recorded_at": self.recorded_at.astimezone(UTC),
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            "snapshot_id": self.snapshot_id,
            "service": self.service,
            "strategy_id": self.strategy_id,
            "status": self.status,
            "recorded_at": self.recorded_at.astimezone(UTC).isoformat(),
            "orders_submitted": int(self.orders_submitted),
            "orders_executed": int(self.orders_executed),
            "orders_failed": int(self.orders_failed),
            "pending_orders": int(self.pending_orders),
            "avg_latency_ms": self.avg_latency_ms,
            "max_latency_ms": self.max_latency_ms,
            "fill_rate": self.fill_rate,
            "failure_rate": self.failure_rate,
            "drop_copy_lag_seconds": self.drop_copy_lag_seconds,
            "drop_copy_active": self.drop_copy_active,
            "connection_healthy": self.connection_healthy,
            "sessions_active": list(self.sessions_active),
            "issues": [dict(issue) for issue in self.issues],
            "policy": dict(self.policy),
            "state": dict(self.state),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_row(
        cls, row: Mapping[str, Any] | Row[Any] | RowMapping
    ) -> "TimescaleExecutionSnapshotRecord":  # pragma: no cover - glue
        mapping = _ensure_mapping(row)

        sessions_raw = _load_json_sequence(mapping.get("sessions_active"))
        issues_raw = _load_json_sequence(mapping.get("issues"))
        policy_raw = _load_json_mapping(mapping.get("policy"))
        state_raw = _load_json_mapping(mapping.get("state"))
        metadata_raw = _load_json_mapping(mapping.get("metadata"))

        return cls(
            snapshot_id=str(mapping["snapshot_id"]),
            service=str(mapping.get("service") or "execution"),
            status=str(mapping.get("status") or "unknown"),
            recorded_at=_coerce_timestamp(mapping.get("recorded_at")),
            orders_submitted=coerce_int(mapping.get("orders_submitted"), default=0),
            orders_executed=coerce_int(mapping.get("orders_executed"), default=0),
            orders_failed=coerce_int(mapping.get("orders_failed"), default=0),
            pending_orders=coerce_int(mapping.get("pending_orders"), default=0),
            avg_latency_ms=coerce_float(mapping.get("avg_latency_ms")),
            max_latency_ms=coerce_float(mapping.get("max_latency_ms")),
            fill_rate=coerce_float(mapping.get("fill_rate")),
            failure_rate=coerce_float(mapping.get("failure_rate")),
            drop_copy_lag_seconds=coerce_float(mapping.get("drop_copy_lag_seconds")),
            drop_copy_active=bool(mapping["drop_copy_active"])
            if mapping.get("drop_copy_active") is not None
            else None,
            connection_healthy=bool(mapping["connection_healthy"])
            if mapping.get("connection_healthy") is not None
            else None,
            sessions_active=tuple(str(item) for item in sessions_raw if item is not None),
            issues=tuple(dict(issue) for issue in issues_raw if isinstance(issue, Mapping)),
            policy=policy_raw if isinstance(policy_raw, Mapping) else {},
            state=state_raw if isinstance(state_raw, Mapping) else {},
            metadata=metadata_raw if isinstance(metadata_raw, Mapping) else {},
            strategy_id=str(mapping.get("strategy_id"))
            if mapping.get("strategy_id") is not None
            else None,
        )


class TimescaleExecutionJournal:
    """Persist execution readiness snapshots for institutional audits."""

    def __init__(self, engine: Engine) -> None:
        self._engine = engine
        self._logger = logging.getLogger(f"{__name__}.TimescaleExecutionJournal")

    def record_snapshot(
        self,
        snapshot: Any,
        *,
        strategy_id: str | None = None,
    ) -> dict[str, Any]:
        record = TimescaleExecutionSnapshotRecord.from_snapshot(snapshot, strategy_id=strategy_id)
        row = record.as_row()

        with self._engine.begin() as conn:
            table = _table_name(*TimescaleMigrator._EXECUTION_SNAPSHOTS, conn.dialect.name)
            statement = text(
                f"""
                INSERT INTO {table} (
                    snapshot_id,
                    service,
                    strategy_id,
                    status,
                    fill_rate,
                    failure_rate,
                    orders_submitted,
                    orders_executed,
                    orders_failed,
                    pending_orders,
                    avg_latency_ms,
                    max_latency_ms,
                    drop_copy_lag_seconds,
                    drop_copy_active,
                    connection_healthy,
                    sessions_active,
                    issues,
                    policy,
                    state,
                    metadata,
                    recorded_at
                ) VALUES (
                    :snapshot_id,
                    :service,
                    :strategy_id,
                    :status,
                    :fill_rate,
                    :failure_rate,
                    :orders_submitted,
                    :orders_executed,
                    :orders_failed,
                    :pending_orders,
                    :avg_latency_ms,
                    :max_latency_ms,
                    :drop_copy_lag_seconds,
                    :drop_copy_active,
                    :connection_healthy,
                    :sessions_active,
                    :issues,
                    :policy,
                    :state,
                    :metadata,
                    :recorded_at
                )
                """
            )
            conn.execute(statement, row)

        return record.as_dict()

    def fetch_recent(
        self,
        *,
        limit: int = 5,
        service: str | None = None,
        strategy_id: str | None = None,
    ) -> list[TimescaleExecutionSnapshotRecord]:
        if limit <= 0:
            return []

        with self._engine.begin() as conn:
            table = _table_name(*TimescaleMigrator._EXECUTION_SNAPSHOTS, conn.dialect.name)
            clauses: list[str] = []
            params: dict[str, object] = {}
            if service:
                clauses.append("service = :service")
                params["service"] = service
            if strategy_id:
                clauses.append("strategy_id = :strategy_id")
                params["strategy_id"] = strategy_id
            where = " WHERE " + " AND ".join(clauses) if clauses else ""
            statement = text(f"SELECT * FROM {table}{where} ORDER BY recorded_at DESC LIMIT :limit")
            params["limit"] = int(limit)
            rows = conn.execute(statement, params).fetchall()
        return [TimescaleExecutionSnapshotRecord.from_row(row) for row in rows]

    def summarise(
        self,
        *,
        service: str | None = None,
        strategy_id: str | None = None,
    ) -> dict[str, Any]:
        """Aggregate execution readiness journal statistics."""

        with self._engine.connect() as conn:
            table = _table_name(*TimescaleMigrator._EXECUTION_SNAPSHOTS, conn.dialect.name)
            clauses: list[str] = []
            params: dict[str, object] = {}
            if service:
                clauses.append("service = :service")
                params["service"] = service
            if strategy_id:
                clauses.append("strategy_id = :strategy_id")
                params["strategy_id"] = strategy_id
            where = " WHERE " + " AND ".join(clauses) if clauses else ""

            summary_stmt = text(
                f"SELECT COUNT(*) AS total_snapshots, MAX(recorded_at) AS last_recorded_at "
                f"FROM {table}{where}"
            )
            row = conn.execute(summary_stmt, params).first()
            mapping = row._mapping if row is not None else {}

            total = int(mapping.get("total_snapshots") or 0)
            last_recorded = _normalise_timestamp(mapping.get("last_recorded_at"))

            status_rows = conn.execute(
                text(f"SELECT status, COUNT(*) AS count FROM {table}{where} GROUP BY status"),
                params,
            ).fetchall()
            status_counts: dict[str, int] = {}
            for row in status_rows:
                status_value = row._mapping.get("status") if hasattr(row, "_mapping") else row[0]
                key = str(status_value) if status_value not in (None, "") else "unknown"
                status_counts[key] = int(
                    (row._mapping.get("count") if hasattr(row, "_mapping") else row[1]) or 0
                )

            service_rows = conn.execute(
                text(f"SELECT service, COUNT(*) AS count FROM {table}{where} GROUP BY service"),
                params,
            ).fetchall()
            service_counts: dict[str, int] = {}
            for row in service_rows:
                service_value = row._mapping.get("service") if hasattr(row, "_mapping") else row[0]
                key = str(service_value) if service_value not in (None, "") else "execution"
                service_counts[key] = int(
                    (row._mapping.get("count") if hasattr(row, "_mapping") else row[1]) or 0
                )

        return {
            "total_snapshots": total,
            "last_recorded_at": last_recorded,
            "status_counts": status_counts,
            "service_counts": service_counts,
        }

    def fetch_latest(
        self,
        *,
        service: str | None = None,
        strategy_id: str | None = None,
    ) -> TimescaleExecutionSnapshotRecord | None:
        rows = self.fetch_recent(limit=1, service=service, strategy_id=strategy_id)
        return rows[0] if rows else None

    def close(self) -> None:
        try:
            self._engine.dispose()
        except Exception:  # pragma: no cover - defensive cleanup
            self._logger.debug("Failed to dispose KYC journal engine", exc_info=True)


class TimescaleIngestJournal:
    """Persist ingest run metadata for audit and telemetry consumers."""

    def __init__(self, engine: Engine) -> None:
        self._engine = engine
        self._logger = logging.getLogger(f"{__name__}.TimescaleIngestJournal")

    def record(self, records: Sequence[TimescaleIngestRunRecord]) -> None:
        if not records:
            return

        rows = [record.as_row() for record in records]
        with self._engine.begin() as conn:
            table = _table_name(*TimescaleMigrator._INGEST_RUNS, conn.dialect.name)
            statement = text(
                f"""
                INSERT INTO {table} (
                    run_id,
                    dimension,
                    status,
                    rows_written,
                    freshness_seconds,
                    ingest_duration_seconds,
                    executed_at,
                    source,
                    symbols,
                    metadata
                ) VALUES (
                    :run_id,
                    :dimension,
                    :status,
                    :rows_written,
                    :freshness_seconds,
                    :ingest_duration_seconds,
                    :executed_at,
                    :source,
                    :symbols,
                    :metadata
                )
                """
            )
            for chunk in _chunk_records(rows, 100):
                conn.execute(statement, chunk)

    def fetch_recent(
        self,
        *,
        limit: int = 20,
        dimension: str | None = None,
    ) -> list[TimescaleIngestRunRecord]:
        limit = max(1, int(limit))
        with self._engine.connect() as conn:
            table = _table_name(*TimescaleMigrator._INGEST_RUNS, conn.dialect.name)
            if dimension:
                stmt = text(
                    f"SELECT run_id, dimension, status, rows_written, freshness_seconds, "
                    "ingest_duration_seconds, executed_at, source, symbols, metadata "
                    f"FROM {table} WHERE dimension = :dimension "
                    "ORDER BY executed_at DESC LIMIT :limit"
                )
                rows = conn.execute(stmt, {"dimension": dimension, "limit": limit}).fetchall()
            else:
                stmt = text(
                    f"SELECT run_id, dimension, status, rows_written, freshness_seconds, "
                    "ingest_duration_seconds, executed_at, source, symbols, metadata "
                    f"FROM {table} ORDER BY executed_at DESC LIMIT :limit"
                )
                rows = conn.execute(stmt, {"limit": limit}).fetchall()

        return [TimescaleIngestRunRecord.from_row(row) for row in rows]

    def fetch_latest_by_dimension(
        self, dimensions: Sequence[str] | None = None
    ) -> dict[str, TimescaleIngestRunRecord]:
        """Return the most recent record for each dimension."""

        with self._engine.connect() as conn:
            table = _table_name(*TimescaleMigrator._INGEST_RUNS, conn.dialect.name)
            params: dict[str, object] = {}
            filter_clause = ""
            if dimensions:
                unique_dimensions = tuple(dict.fromkeys(dimensions))
                if unique_dimensions:
                    placeholders = ", ".join(
                        f":dimension_{idx}" for idx, _ in enumerate(unique_dimensions)
                    )
                    filter_clause = f"WHERE dimension IN ({placeholders})"
                    params = {f"dimension_{idx}": dim for idx, dim in enumerate(unique_dimensions)}

            stmt = text(
                f"""
                SELECT run_id, dimension, status, rows_written, freshness_seconds,
                       ingest_duration_seconds, executed_at, source, symbols, metadata
                FROM (
                    SELECT run_id, dimension, status, rows_written, freshness_seconds,
                           ingest_duration_seconds, executed_at, source, symbols, metadata,
                           ROW_NUMBER() OVER (
                               PARTITION BY dimension
                               ORDER BY executed_at DESC, run_id DESC
                           ) AS rn
                    FROM {table}
                    {filter_clause}
                ) AS ranked
                WHERE rn = 1
                """
            )
            rows = conn.execute(stmt, params).fetchall()

        results: dict[str, TimescaleIngestRunRecord] = {}
        for row in rows:
            record = TimescaleIngestRunRecord.from_row(row)
            results[record.dimension] = record
        return results


# ---------------------------------------------------------------------------
# Configuration audit journal
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TimescaleConfigurationAuditRecord:
    """Structured configuration audit snapshot stored in Timescale."""

    snapshot_id: str
    applied_at: datetime
    severity: str
    changes: tuple[Mapping[str, Any], ...]
    current_config: Mapping[str, Any]
    previous_config: Mapping[str, Any] | None
    metadata: Mapping[str, Any]

    @classmethod
    def from_snapshot(cls, snapshot: Mapping[str, Any]) -> "TimescaleConfigurationAuditRecord":
        payload = json.loads(json.dumps(snapshot, default=_json_default))

        snapshot_id = str(payload.get("snapshot_id") or uuid4())
        applied_raw = payload.get("applied_at")
        applied_at = (
            _coerce_timestamp(applied_raw) if applied_raw is not None else datetime.now(tz=UTC)
        )
        severity = str(payload.get("status") or payload.get("severity") or "pass")

        changes_raw = payload.get("changes")
        changes: list[Mapping[str, Any]] = []
        if isinstance(changes_raw, list):
            for item in changes_raw:
                if isinstance(item, Mapping):
                    changes.append(dict(item))

        current_raw = payload.get("current_config")
        if isinstance(current_raw, Mapping):
            current_config = dict(current_raw)
        else:
            current_config = {}

        previous_raw = payload.get("previous_config")
        if isinstance(previous_raw, Mapping):
            previous_config: Mapping[str, Any] | None = dict(previous_raw)
        else:
            previous_config = None

        metadata_raw = payload.get("metadata")
        metadata = dict(metadata_raw) if isinstance(metadata_raw, Mapping) else {}

        return cls(
            snapshot_id=snapshot_id,
            applied_at=applied_at,
            severity=severity,
            changes=tuple(changes),
            current_config=current_config,
            previous_config=previous_config,
            metadata=metadata,
        )

    @classmethod
    def from_row(
        cls, row: Mapping[str, Any] | Row[Any] | RowMapping
    ) -> "TimescaleConfigurationAuditRecord":  # pragma: no cover - simple glue
        mapping = _ensure_mapping(row)

        applied_at = _coerce_timestamp(mapping.get("applied_at"))
        severity = str(mapping.get("severity") or "pass")

        changes_raw = _load_json_sequence(mapping.get("changes"))
        change_entries = [dict(item) for item in changes_raw if isinstance(item, Mapping)]

        current_config = _load_json_mapping(mapping.get("current_config"))
        previous_candidate = _load_json_payload(mapping.get("previous_config"))
        previous_config = (
            dict(previous_candidate)
            if isinstance(previous_candidate, Mapping)
            else None
        )

        metadata = _load_json_mapping(mapping.get("metadata"))

        return cls(
            snapshot_id=str(mapping.get("snapshot_id") or uuid4()),
            applied_at=applied_at,
            severity=severity,
            changes=tuple(change_entries),
            current_config=current_config,
            previous_config=previous_config,
            metadata=metadata,
        )

    def as_row(self) -> dict[str, object]:
        return {
            "snapshot_id": self.snapshot_id,
            "applied_at": self.applied_at.astimezone(UTC),
            "severity": self.severity,
            "changes": json.dumps(
                [dict(change) for change in self.changes],
                default=_json_default,
            ),
            "current_config": json.dumps(dict(self.current_config), default=_json_default),
            "previous_config": (
                json.dumps(dict(self.previous_config), default=_json_default)
                if self.previous_config is not None
                else None
            ),
            "metadata": json.dumps(dict(self.metadata), default=_json_default),
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            "snapshot_id": self.snapshot_id,
            "applied_at": self.applied_at.astimezone(UTC).isoformat(),
            "severity": self.severity,
            "changes": [dict(change) for change in self.changes],
            "current_config": dict(self.current_config),
            "previous_config": (dict(self.previous_config) if self.previous_config else None),
            "metadata": dict(self.metadata),
        }


class TimescaleConfigurationAuditJournal:
    """Persist configuration audit snapshots for downstream telemetry."""

    def __init__(self, engine: Engine) -> None:
        self._engine = engine
        self._logger = logging.getLogger(f"{__name__}.TimescaleConfigurationAuditJournal")

    def record_snapshot(self, snapshot: Mapping[str, Any] | Any) -> dict[str, Any]:
        as_dict = getattr(snapshot, "as_dict", None)
        if callable(as_dict):
            candidate = as_dict()
            if isinstance(candidate, Mapping):
                payload = dict(candidate)
            else:  # pragma: no cover - defensive
                raise TypeError("snapshot as_dict() must return a mapping")
        elif isinstance(snapshot, Mapping):
            payload = dict(snapshot)
        else:
            raise TypeError("snapshot must be a mapping or expose as_dict()")

        record = TimescaleConfigurationAuditRecord.from_snapshot(payload)
        row = record.as_row()

        with self._engine.begin() as conn:
            table = _table_name(*TimescaleMigrator._CONFIGURATION_AUDIT, conn.dialect.name)
            statement = text(
                f"""
                INSERT INTO {table} (
                    snapshot_id,
                    applied_at,
                    severity,
                    changes,
                    current_config,
                    previous_config,
                    metadata
                ) VALUES (
                    :snapshot_id,
                    :applied_at,
                    :severity,
                    :changes,
                    :current_config,
                    :previous_config,
                    :metadata
                )
                """
            )
            conn.execute(statement, row)

        return record.as_dict()

    def fetch_latest(self) -> TimescaleConfigurationAuditRecord | None:
        with self._engine.connect() as conn:
            table = _table_name(*TimescaleMigrator._CONFIGURATION_AUDIT, conn.dialect.name)
            stmt = text(
                f"""
                SELECT snapshot_id, applied_at, severity, changes,
                       current_config, previous_config, metadata
                FROM {table}
                ORDER BY applied_at DESC
                LIMIT 1
                """
            )
            row = conn.execute(stmt).fetchone()

        if row is None:
            return None
        return TimescaleConfigurationAuditRecord.from_row(row)

    def fetch_recent(self, *, limit: int = 5) -> list[TimescaleConfigurationAuditRecord]:
        limit = max(1, int(limit))
        with self._engine.connect() as conn:
            table = _table_name(*TimescaleMigrator._CONFIGURATION_AUDIT, conn.dialect.name)
            stmt = text(
                f"""
                SELECT snapshot_id, applied_at, severity, changes,
                       current_config, previous_config, metadata
                FROM {table}
                ORDER BY applied_at DESC
                LIMIT :limit
                """
            )
            rows = conn.execute(stmt, {"limit": limit}).fetchall()

        return [TimescaleConfigurationAuditRecord.from_row(row) for row in rows]

    def close(self) -> None:
        try:
            self._engine.dispose()
        except Exception:  # pragma: no cover - defensive cleanup
            self._logger.debug("Failed to dispose configuration audit engine", exc_info=True)


class TimescaleIngestor:
    """Persist market data frames into TimescaleDB with idempotent upserts."""

    def __init__(self, engine: Engine, *, chunk_size: int = 500) -> None:
        self._engine = engine
        self._chunk_size = max(chunk_size, 1)
        self._logger = logging.getLogger(f"{__name__}.TimescaleIngestor")

    def _ingest_records(
        self,
        *,
        records: list[dict[str, object]],
        schema: str,
        table: str,
        key_columns: Sequence[str],
        update_columns: Sequence[str],
        all_columns: Sequence[str],
        dimension: str,
        entity_key: str,
        timestamp_key: str,
        ingest_ts: datetime,
        source: str | None,
    ) -> TimescaleIngestResult:
        if not records:
            self._logger.info("No rows to ingest into Timescale for dimension %s", dimension)
            return TimescaleIngestResult.empty(dimension=dimension, source=source)

        ingest_start = time.perf_counter()

        entities = sorted(
            {str(rec[entity_key]) for rec in records if rec.get(entity_key) is not None}
        )
        timestamps: list[datetime] = []
        for rec in records:
            ts_value = rec.get(timestamp_key)
            if isinstance(ts_value, datetime):
                if ts_value.tzinfo is None:
                    timestamps.append(ts_value.replace(tzinfo=UTC))
                else:
                    timestamps.append(ts_value.astimezone(UTC))

        with self._engine.begin() as conn:
            dialect = conn.dialect.name
            table_name = _table_name(schema, table, dialect)
            if dialect != "postgresql":
                _normalize_sqlite_records(records, ("ts", "ingested_at"))

            if dialect == "postgresql":
                cols = ", ".join(all_columns)
                placeholders = ", ".join(f":{col}" for col in all_columns)
                conflict = ", ".join(key_columns)
                updates = ", ".join(f"{col} = EXCLUDED.{col}" for col in update_columns)
                statement = text(
                    f"INSERT INTO {table_name} ({cols}) VALUES ({placeholders}) "
                    f"ON CONFLICT ({conflict}) DO UPDATE SET {updates}"
                )
                for chunk in _chunk_records(records, self._chunk_size):
                    conn.execute(statement, chunk)
            else:
                predicate = " AND ".join(f"{col} = :{col}" for col in key_columns)
                delete_stmt = text(f"DELETE FROM {table_name} WHERE {predicate}")
                cols = ", ".join(all_columns)
                placeholders = ", ".join(f":{col}" for col in all_columns)
                insert_stmt = text(f"INSERT INTO {table_name} ({cols}) VALUES ({placeholders})")
                for chunk in _chunk_records(records, self._chunk_size):
                    for record in chunk:
                        conn.execute(delete_stmt, record)
                        conn.execute(insert_stmt, record)

        ingest_end = time.perf_counter()
        start_ts = min(timestamps) if timestamps else None
        end_ts = max(timestamps) if timestamps else None
        freshness = None
        if end_ts is not None:
            freshness = max((ingest_ts - end_ts).total_seconds(), 0.0)

        result = TimescaleIngestResult(
            rows_written=len(records),
            symbols=tuple(entities),
            start_ts=start_ts,
            end_ts=end_ts,
            ingest_duration_seconds=ingest_end - ingest_start,
            freshness_seconds=freshness,
            dimension=dimension,
            source=source,
        )

        self._logger.info(
            "Timescale ingest %s[%s]: rows=%s entities=%s freshness=%.2fs",
            dimension,
            source or "unknown",
            result.rows_written,
            ",".join(result.symbols),
            result.freshness_seconds or 0.0,
        )
        return result

    def upsert_daily_bars(
        self, df: pd.DataFrame, *, source: str = "yahoo"
    ) -> TimescaleIngestResult:
        if df.empty:
            self._logger.info("No rows to ingest into Timescale for daily bars")
            return TimescaleIngestResult.empty(dimension="daily_bars", source=source)

        ingest_ts = datetime.now(tz=UTC)
        records = _prepare_daily_bar_records(df, source, ingest_ts)
        return self._ingest_records(
            records=records,
            schema="market_data",
            table="daily_bars",
            key_columns=("symbol", "ts"),
            update_columns=(
                "open",
                "high",
                "low",
                "close",
                "adj_close",
                "volume",
                "source",
                "ingested_at",
            ),
            all_columns=(
                "ts",
                "symbol",
                "open",
                "high",
                "low",
                "close",
                "adj_close",
                "volume",
                "source",
                "ingested_at",
            ),
            dimension="daily_bars",
            entity_key="symbol",
            timestamp_key="ts",
            ingest_ts=ingest_ts,
            source=source,
        )

    def upsert_intraday_trades(
        self, df: pd.DataFrame, *, source: str = "yahoo"
    ) -> TimescaleIngestResult:
        if df.empty:
            self._logger.info("No rows to ingest into Timescale for intraday trades")
            return TimescaleIngestResult.empty(dimension="intraday_trades", source=source)

        ingest_ts = datetime.now(tz=UTC)
        records = _prepare_intraday_trade_records(df, source, ingest_ts)
        return self._ingest_records(
            records=records,
            schema="market_data",
            table="intraday_trades",
            key_columns=("symbol", "ts", "price"),
            update_columns=("size", "exchange", "conditions", "source", "ingested_at"),
            all_columns=(
                "ts",
                "symbol",
                "price",
                "size",
                "exchange",
                "conditions",
                "source",
                "ingested_at",
            ),
            dimension="intraday_trades",
            entity_key="symbol",
            timestamp_key="ts",
            ingest_ts=ingest_ts,
            source=source,
        )

    def upsert_macro_events(
        self, df: pd.DataFrame, *, source: str = "fred"
    ) -> TimescaleIngestResult:
        if df.empty:
            self._logger.info("No rows to ingest into Timescale for macro events")
            return TimescaleIngestResult.empty(dimension="macro_events", source=source)

        ingest_ts = datetime.now(tz=UTC)
        records = _prepare_macro_event_records(df, source, ingest_ts)
        return self._ingest_records(
            records=records,
            schema="macro_data",
            table="events",
            key_columns=("event_name", "ts"),
            update_columns=(
                "calendar",
                "currency",
                "actual",
                "forecast",
                "previous",
                "importance",
                "source",
                "ingested_at",
            ),
            all_columns=(
                "ts",
                "event_name",
                "calendar",
                "currency",
                "actual",
                "forecast",
                "previous",
                "importance",
                "source",
                "ingested_at",
            ),
            dimension="macro_events",
            entity_key="event_name",
            timestamp_key="ts",
            ingest_ts=ingest_ts,
            source=source,
        )


def ensure_timescale_daily_bars(
    engine: Engine,
    frame: pd.DataFrame,
    *,
    source: str = "yahoo",
) -> TimescaleIngestResult:
    """Convenience helper that migrates and ingests a frame of daily bars."""

    migrator = TimescaleMigrator(engine)
    migrator.apply()
    ingestor = TimescaleIngestor(engine)
    return ingestor.upsert_daily_bars(frame, source=source)


__all__ = [
    "DEFAULT_SQLITE_FALLBACK",
    "TimescaleConnectionSettings",
    "TimescaleMigrator",
    "TimescaleComplianceAuditRecord",
    "TimescaleComplianceJournal",
    "TimescaleKycCaseRecord",
    "TimescaleKycJournal",
    "TimescaleExecutionSnapshotRecord",
    "TimescaleExecutionJournal",
    "TimescaleIngestJournal",
    "TimescaleIngestor",
    "TimescaleIngestResult",
    "TimescaleIngestRunRecord",
    "ensure_timescale_daily_bars",
]
