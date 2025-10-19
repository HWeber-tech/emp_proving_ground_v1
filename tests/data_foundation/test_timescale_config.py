from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.data_foundation.batch.spark_export import SparkExportFormat
from src.data_foundation.ingest.configuration import build_institutional_ingest_config
from src.governance.system_config import (
    ConnectionProtocol,
    DataBackboneMode,
    EmpEnvironment,
    EmpTier,
    RunMode,
    SystemConfig,
)


pytestmark = pytest.mark.guardrail


def _base_config(**extras: str) -> SystemConfig:
    return SystemConfig(
        run_mode=RunMode.paper,
        environment=EmpEnvironment.demo,
        tier=EmpTier.tier_1,
        connection_protocol=ConnectionProtocol.bootstrap,
        data_backbone_mode=DataBackboneMode.institutional,
        extras=dict(extras),
    )


def test_config_builder_requires_institutional_mode() -> None:
    cfg = SystemConfig(
        run_mode=RunMode.paper,
        environment=EmpEnvironment.demo,
        tier=EmpTier.tier_1,
        connection_protocol=ConnectionProtocol.bootstrap,
        data_backbone_mode=DataBackboneMode.bootstrap,
    )

    resolved = build_institutional_ingest_config(cfg, fallback_symbols=("EURUSD",))
    assert not resolved.should_run
    assert resolved.reason == "Data backbone mode is not institutional"
    assert resolved.metadata["mode"] == DataBackboneMode.bootstrap.value


def test_config_builder_requires_tier_one() -> None:
    cfg = SystemConfig(
        run_mode=RunMode.paper,
        environment=EmpEnvironment.demo,
        tier=EmpTier.tier_0,
        connection_protocol=ConnectionProtocol.bootstrap,
        data_backbone_mode=DataBackboneMode.institutional,
    )

    resolved = build_institutional_ingest_config(cfg, fallback_symbols=("EURUSD",))
    assert not resolved.should_run
    assert resolved.reason == "Timescale ingest currently targets Tier-1 institutional runs"
    assert resolved.metadata["tier"] == EmpTier.tier_0.value


def test_config_builder_uses_fallback_symbols_when_not_overridden() -> None:
    cfg = _base_config()

    resolved = build_institutional_ingest_config(cfg, fallback_symbols=("EURUSD", "GBPUSD"))
    assert resolved.should_run is True
    assert resolved.plan.daily is not None
    assert resolved.plan.daily.normalised_symbols() == ["EURUSD", "GBPUSD"]
    assert resolved.plan.intraday is None
    assert resolved.metadata["symbols"] == ["EURUSD", "GBPUSD"]


def test_config_builder_creates_intraday_and_macro_event_plan() -> None:
    events = [
        {
            "timestamp": datetime(2024, 1, 2, 13, 30, tzinfo=timezone.utc).isoformat(),
            "calendar": "FOMC",
            "event": "Rate Decision",
            "currency": "USD",
            "source": "fred",
        },
        {
            "timestamp": datetime(2024, 1, 3, 9, 0, tzinfo=timezone.utc).isoformat(),
            "calendar": "ECB",
            "event": "Press Conference",
            "currency": "EUR",
            "source": "fred",
        },
    ]

    cfg = _base_config(
        TIMESCALE_SYMBOLS="EURUSD, GBPUSD",
        TIMESCALE_ENABLE_INTRADAY="true",
        TIMESCALE_INTRADAY_INTERVAL="5m",
        TIMESCALE_INTRADAY_LOOKBACK_DAYS="5",
        TIMESCALE_LOOKBACK_DAYS="120",
        TIMESCALE_ENABLE_MACRO="1",
        TIMESCALE_MACRO_EVENTS=str(events).replace("'", '"'),
    )

    resolved = build_institutional_ingest_config(cfg)
    assert resolved.should_run is True
    assert resolved.plan.daily is not None
    assert resolved.plan.daily.lookback_days == 120
    assert resolved.plan.intraday is not None
    assert resolved.plan.intraday.interval == "5m"
    assert resolved.plan.intraday.lookback_days == 5
    assert resolved.plan.macro is not None
    assert resolved.metadata["macro_mode"] == "events"
    assert resolved.metadata["macro_events"] == 2
    assert resolved.metadata["kafka_configured"] is False
    assert resolved.metadata["kafka_topics"] == []
    assert resolved.metadata["kafka_auto_create_topics"] is False
    readiness = resolved.metadata["kafka_readiness"]
    assert readiness["enabled"] is False
    assert readiness["require_topics"] is True
    assert readiness["min_publishers"] == 1


def test_config_builder_resolves_redis_cache_policy_overrides() -> None:
    cfg = _base_config(
        TIMESCALE_SYMBOLS="EURUSD",
        REDIS_CACHE_TTL_SECONDS="45",
        REDIS_CACHE_MAX_KEYS="128",
        REDIS_CACHE_NAMESPACE="emp:test",
        REDIS_CACHE_INVALIDATE_PREFIXES="timescale:daily,timescale:intraday",
    )

    resolved = build_institutional_ingest_config(cfg)

    assert resolved.redis_policy.ttl_seconds == 45
    assert resolved.redis_policy.max_keys == 128
    assert resolved.redis_policy.namespace == "emp:test"
    assert resolved.redis_policy.invalidate_prefixes == (
        "timescale:daily",
        "timescale:intraday",
    )

    policy_metadata = resolved.metadata["redis_cache_policy"]
    assert policy_metadata["ttl_seconds"] == 45
    assert policy_metadata["max_keys"] == 128
    assert policy_metadata["namespace"] == "emp:test"
    assert policy_metadata["invalidate_prefixes"] == [
        "timescale:daily",
        "timescale:intraday",
    ]


def test_config_builder_populates_redis_settings_metadata() -> None:
    cfg = _base_config(
        TIMESCALE_SYMBOLS="EURUSD",
        REDIS_URL="redis://cache.example.com:6380/2",
        REDIS_CLIENT_NAME="emp-ingest",
        REDIS_SSL="1",
    )

    resolved = build_institutional_ingest_config(cfg)

    settings = resolved.redis_settings
    assert settings.configured is True
    assert settings.host == "cache.example.com"
    assert settings.port == 6380
    assert settings.db == 2
    assert settings.client_name == "emp-ingest"
    assert settings.ssl is True

    metadata = resolved.metadata
    assert metadata["redis_configured"] is True
    assert metadata["redis_client_name"] == "emp-ingest"
    assert metadata["redis_ssl"] is True


def test_config_builder_falls_back_to_macro_window_when_events_missing() -> None:
    cfg = _base_config(
        TIMESCALE_SYMBOLS="EURUSD",
        TIMESCALE_ENABLE_MACRO="true",
        TIMESCALE_MACRO_START="2024-01-01",
        TIMESCALE_MACRO_END="2024-01-31",
    )

    resolved = build_institutional_ingest_config(cfg)
    assert resolved.should_run is True
    assert resolved.plan.macro is not None
    assert resolved.plan.macro.has_window() is True
    assert resolved.metadata["macro_mode"] == "window"
    assert resolved.metadata["macro_window"] == {
        "start": "2024-01-01",
        "end": "2024-01-31",
    }
    assert resolved.metadata["kafka_topics"] == []
    assert resolved.metadata["kafka_auto_create_topics"] is False
    readiness = resolved.metadata["kafka_readiness"]
    assert readiness["enabled"] is False
    assert readiness["require_topics"] is True
    assert readiness["min_publishers"] == 1


def test_config_builder_handles_missing_plan_by_disabling_ingest() -> None:
    cfg = _base_config()

    resolved = build_institutional_ingest_config(cfg, fallback_symbols=())
    assert resolved.should_run is False
    assert resolved.reason == "No ingest slices configured via extras or fallback symbols"


@pytest.mark.parametrize(
    "payload",
    [
        "{not-json}",
        "{}",
        "[1, 2, 3]",
        "[{'timestamp': '2024-01-02T00:00:00Z'}]".replace("'", '"'),
    ],
)
def test_config_builder_gracefully_ignores_invalid_macro_event_payload(payload: str) -> None:
    cfg = _base_config(
        TIMESCALE_SYMBOLS="EURUSD",
        TIMESCALE_ENABLE_MACRO="true",
        TIMESCALE_MACRO_EVENTS=payload,
    )

    resolved = build_institutional_ingest_config(cfg)
    assert resolved.should_run is True
    assert resolved.plan.macro is None
    assert resolved.metadata["macro_mode"] == "disabled"
    assert resolved.metadata["kafka_topics"] == []
    assert resolved.metadata["kafka_auto_create_topics"] is False


def test_config_builder_includes_kafka_topic_metadata() -> None:
    cfg = _base_config(
        TIMESCALE_SYMBOLS="EURUSD",
        KAFKA_BROKERS="localhost:9092",
        KAFKA_INGEST_TOPICS="daily_bars:timescale.daily",
        KAFKA_INGEST_AUTO_CREATE_TOPICS="true",
    )

    resolved = build_institutional_ingest_config(cfg)
    assert resolved.should_run is True
    assert resolved.metadata["kafka_configured"] is True
    assert resolved.metadata["kafka_topics"] == ["timescale.daily"]
    assert resolved.metadata["kafka_auto_create_topics"] is True
    readiness = resolved.kafka_readiness
    assert readiness.enabled is True
    assert readiness.min_publishers == 1
    assert readiness.require_topics is True
    readiness_metadata = resolved.metadata["kafka_readiness"]
    assert readiness_metadata["enabled"] is True
    assert readiness_metadata["require_topics"] is True
    assert readiness_metadata["min_publishers"] == 1


def test_config_builder_parses_kafka_readiness_settings() -> None:
    cfg = _base_config(
        TIMESCALE_SYMBOLS="EURUSD",
        KAFKA_BROKERS="localhost:9092",
        KAFKA_INGEST_TOPICS="daily_bars:timescale.daily",
        KAFKA_READINESS_ENABLED="false",
        KAFKA_READINESS_WARN_LAG_MESSAGES="500",
        KAFKA_READINESS_FAIL_LAG_MESSAGES="2000",
        KAFKA_READINESS_WARN_STALE_SECONDS="30",
        KAFKA_READINESS_FAIL_STALE_SECONDS="60",
        KAFKA_READINESS_MIN_PUBLISHERS="2",
        KAFKA_READINESS_REQUIRE_TOPICS="false",
        KAFKA_READINESS_REQUIRE_CONSUMER="true",
    )

    resolved = build_institutional_ingest_config(cfg)

    readiness = resolved.kafka_readiness
    assert readiness.enabled is False
    assert readiness.warn_lag_messages == 500
    assert readiness.fail_lag_messages == 2000
    assert readiness.warn_stale_seconds == 30.0
    assert readiness.fail_stale_seconds == 60.0
    assert readiness.min_publishers == 2
    assert readiness.require_topics is False
    assert readiness.require_consumer is True

    metadata = resolved.metadata["kafka_readiness"]
    assert metadata["enabled"] is False
    assert metadata["warn_lag_messages"] == 500
    assert metadata["fail_lag_messages"] == 2000
    assert metadata["warn_stale_seconds"] == 30.0
    assert metadata["fail_stale_seconds"] == 60.0
    assert metadata["min_publishers"] == 2
    assert metadata["require_topics"] is False
    assert metadata["require_consumer"] is True


def test_config_builder_records_api_keys_and_session_calendars() -> None:
    cfg = _base_config(
        TIMESCALE_SYMBOLS="EURUSD",
        ALPHA_VANTAGE_API_KEY="demo",
        FRED_API_KEY="fred-key",
        TIMESCALE_MACRO_CALENDARS="FOMC, NFP, FOMC",
    )

    resolved = build_institutional_ingest_config(cfg)
    metadata = resolved.metadata

    api_keys = metadata["api_keys"]
    assert api_keys["alpha_vantage"]["configured"] is True
    assert api_keys["alpha_vantage"]["source"] == "ALPHA_VANTAGE_API_KEY"
    assert api_keys["fred"]["configured"] is True
    assert api_keys["news"]["configured"] is False

    assert metadata["macro_calendars"] == ["FOMC", "NFP"]

    sessions = metadata["session_calendars"]
    assert isinstance(sessions, list)
    assert any(entry["id"] == "london_fx" for entry in sessions)
    london = next(entry for entry in sessions if entry["id"] == "london_fx")
    assert london["timezone"] == "Europe/London"
    assert london["open_time"] == "07:00"
    assert london["close_time"] == "16:30"
    assert "Mon" in london["days"]

    inventory = metadata["symbol_inventory"]
    assert isinstance(inventory, list)
    assert any(entry["symbol"] == "EURUSD" for entry in inventory)
    eurusd = next(entry for entry in inventory if entry["symbol"] == "EURUSD")
    assert eurusd["margin_currency"] == "USD"
    assert eurusd["pip_decimal_places"] == 4
    assert eurusd["contract_size"] == "100000"
    assert eurusd["swap_time"] == "22:00"

    symbol_metadata = metadata["symbol_metadata"]
    assert isinstance(symbol_metadata, list)
    assert any(entry["symbol"] == "EURUSD" for entry in symbol_metadata)
    eurusd_metadata = next(entry for entry in symbol_metadata if entry["symbol"] == "EURUSD")
    assert eurusd_metadata["margin_currency"] == "USD"
    assert eurusd_metadata["pip_decimal_places"] == 4
    assert eurusd_metadata["contract_size"] == "100000"
    assert eurusd_metadata["swap_time"] == "22:00"


def test_config_builder_detects_api_keys_from_environment(monkeypatch) -> None:
    monkeypatch.setenv("FRED_API_KEY", "from-env")
    cfg = _base_config(
        TIMESCALE_SYMBOLS="EURUSD",
    )

    resolved = build_institutional_ingest_config(cfg)
    api_keys = resolved.metadata["api_keys"]

    assert api_keys["fred"]["configured"] is True
    assert api_keys["fred"]["source"] == "FRED_API_KEY"
    assert api_keys["alpha_vantage"]["configured"] is False


def test_config_builder_parses_schedule() -> None:
    cfg = _base_config(
        TIMESCALE_SYMBOLS="EURUSD",
        TIMESCALE_INGEST_SCHEDULE="true",
        TIMESCALE_INGEST_INTERVAL_SECONDS="900",
        TIMESCALE_INGEST_JITTER_SECONDS="45",
        TIMESCALE_INGEST_MAX_FAILURES="0",
    )

    resolved = build_institutional_ingest_config(cfg)
    assert resolved.should_run is True
    assert resolved.schedule is not None
    assert resolved.schedule.interval_seconds == 900.0
    assert resolved.schedule.jitter_seconds == 45.0
    assert resolved.schedule.max_failures == 0
    assert resolved.metadata["schedule_enabled"] is True


def test_config_builder_parses_operational_alert_routes() -> None:
    cfg = _base_config(
        TIMESCALE_SYMBOLS="EURUSD",
        OPERATIONS_ALERT_ROUTES='{"timescale_ingest": "pagerduty:data", "timescale_ingest.daily_bars": "slack:#ops"}',
    )

    resolved = build_institutional_ingest_config(cfg)
    assert resolved.should_run is True
    assert resolved.operational_alert_routes["timescale_ingest"] == "pagerduty:data"
    assert (
        resolved.metadata["operational_alert_routes"]["timescale_ingest.daily_bars"] == "slack:#ops"
    )


def test_config_builder_parses_backup_settings() -> None:
    cfg = _base_config(
        TIMESCALE_SYMBOLS="EURUSD",
        TIMESCALE_BACKUP_ENABLED="true",
        TIMESCALE_BACKUP_FREQUENCY_SECONDS="7200",
        TIMESCALE_BACKUP_RETENTION_DAYS="14",
        TIMESCALE_BACKUP_MIN_RETENTION_DAYS="10",
        TIMESCALE_BACKUP_WARN_AFTER_SECONDS="10800",
        TIMESCALE_BACKUP_FAIL_AFTER_SECONDS="21600",
        TIMESCALE_BACKUP_RESTORE_INTERVAL_DAYS="5",
        TIMESCALE_BACKUP_LAST_SUCCESS="2024-01-03T10:00:00Z",
        TIMESCALE_BACKUP_LAST_STATUS="success",
        TIMESCALE_BACKUP_LAST_RESTORE="2023-12-30T10:00:00+00:00",
        TIMESCALE_BACKUP_LAST_RESTORE_STATUS="warn",
        TIMESCALE_BACKUP_PROVIDERS="s3, glacier",
        TIMESCALE_BACKUP_STORAGE="s3://timescale/backups",
        TIMESCALE_BACKUP_FAILURES="2024-01-01T00:00:00Z,2024-01-02T00:00:00Z",
    )

    resolved = build_institutional_ingest_config(cfg)

    backup = resolved.backup
    assert backup.enabled is True
    assert backup.expected_frequency_seconds == 7200.0
    assert backup.retention_days == 14
    assert backup.minimum_retention_days == 10
    assert backup.warn_after_seconds == 10_800.0
    assert backup.fail_after_seconds == 21_600.0
    assert backup.restore_test_interval_days == 5
    assert backup.last_backup_at is not None
    assert backup.last_backup_status == "success"
    assert backup.last_restore_test_at is not None
    assert backup.last_restore_status == "warn"
    assert backup.providers == ("s3", "glacier")
    assert backup.storage_location == "s3://timescale/backups"
    assert backup.recorded_failures == ("2024-01-01T00:00:00Z", "2024-01-02T00:00:00Z")

    metadata = resolved.metadata["backup"]
    assert metadata["enabled"] is True
    assert metadata["last_backup_status"] == "success"
    assert metadata["providers"] == ["s3", "glacier"]


def test_config_builder_parses_retention_settings() -> None:
    cfg = _base_config(
        TIMESCALE_SYMBOLS="EURUSD",
        TIMESCALE_RETENTION_ENABLED="true",
        TIMESCALE_RETENTION_DAILY_TARGET_DAYS="400",
        TIMESCALE_RETENTION_DAILY_MIN_DAYS="365",
        TIMESCALE_RETENTION_INTRADAY_TARGET_DAYS="45",
        TIMESCALE_RETENTION_INTRADAY_MIN_DAYS="10",
        TIMESCALE_RETENTION_INTRADAY_OPTIONAL="true",
        TIMESCALE_RETENTION_MACRO_TARGET_DAYS="720",
        TIMESCALE_RETENTION_MACRO_MIN_DAYS="540",
    )

    resolved = build_institutional_ingest_config(cfg)

    retention = resolved.retention
    assert retention.enabled is True
    dimensions = {policy.dimension: policy for policy in retention.policies}
    assert dimensions["daily_bars"].target_days == 400
    assert dimensions["daily_bars"].minimum_days == 365
    assert dimensions["intraday_trades"].optional is True
    assert dimensions["macro_events"].target_days == 720

    metadata = resolved.metadata["retention"]
    assert metadata["enabled"] is True
    names = {policy["dimension"] for policy in metadata["policies"]}
    assert {"daily_bars", "intraday_trades", "macro_events"} <= names


def test_config_builder_parses_recovery_settings() -> None:
    cfg = _base_config(
        TIMESCALE_SYMBOLS="EURUSD,GBPUSD",
        TIMESCALE_INGEST_RECOVERY="true",
        TIMESCALE_INGEST_RECOVERY_MAX_ATTEMPTS="4",
        TIMESCALE_INGEST_RECOVERY_LOOKBACK_MULTIPLIER="3.5",
        TIMESCALE_INGEST_RECOVERY_TARGET_MISSING_SYMBOLS="false",
    )

    resolved = build_institutional_ingest_config(cfg)
    assert resolved.should_run is True
    assert resolved.recovery.enabled is True
    assert resolved.recovery.max_attempts == 4
    assert resolved.recovery.lookback_multiplier == 3.5
    assert resolved.recovery.target_missing_symbols is False
    assert resolved.metadata["recovery"]["max_attempts"] == 4


def test_config_builder_parses_failover_drill_settings() -> None:
    cfg = SystemConfig().with_updated(
        data_backbone_mode=DataBackboneMode.institutional,
        tier=EmpTier.tier_1,
        extras={
            "TIMESCALE_SYMBOLS": "EURUSD,GBPUSD",
            "TIMESCALE_FAILOVER_DRILL": "true",
            "TIMESCALE_FAILOVER_DRILL_DIMENSIONS": "daily_bars,intraday_trades",
            "TIMESCALE_FAILOVER_DRILL_LABEL": "daily-intraday-drill",
            "TIMESCALE_FAILOVER_DRILL_RUN_FALLBACK": "false",
        },
    )

    resolved = build_institutional_ingest_config(cfg)

    assert resolved.failover_drill is not None
    assert resolved.failover_drill.enabled is True
    assert resolved.failover_drill.dimensions == (
        "daily_bars",
        "intraday_trades",
    )
    assert resolved.failover_drill.run_fallback is False
    assert resolved.metadata["failover_drill"]["label"] == "daily-intraday-drill"


def test_config_builder_parses_spark_stress_settings() -> None:
    cfg = SystemConfig().with_updated(
        data_backbone_mode=DataBackboneMode.institutional,
        tier=EmpTier.tier_1,
        extras={
            "TIMESCALE_SYMBOLS": "EURUSD",
            "TIMESCALE_SPARK_STRESS": "true",
            "TIMESCALE_SPARK_STRESS_CYCLES": "4",
            "TIMESCALE_SPARK_STRESS_WARN_AFTER_SECONDS": "0.5",
            "TIMESCALE_SPARK_STRESS_FAIL_AFTER_SECONDS": "2.0",
            "TIMESCALE_SPARK_STRESS_LABEL": "resilience-drill",
        },
    )

    resolved = build_institutional_ingest_config(cfg)

    assert resolved.spark_stress is not None
    assert resolved.spark_stress.enabled is True
    assert resolved.spark_stress.cycles == 4
    assert resolved.spark_stress.warn_after_seconds == 0.5
    assert resolved.spark_stress.fail_after_seconds == 2.0
    assert resolved.metadata["spark_stress"]["label"] == "resilience-drill"


def test_config_builder_parses_spark_export_plan(tmp_path) -> None:
    cfg = _base_config(
        TIMESCALE_SYMBOLS="EURUSD,GBPUSD",
        TIMESCALE_SPARK_EXPORT="true",
        TIMESCALE_SPARK_EXPORT_ROOT=str(tmp_path / "spark"),
        TIMESCALE_SPARK_EXPORT_FORMAT="jsonl",
        TIMESCALE_SPARK_EXPORT_DIMENSIONS="daily_bars,intraday_trades",
        TIMESCALE_SPARK_EXPORT_PARTITION_BY="symbol",
        TIMESCALE_SPARK_EXPORT_PUBLISH="false",
        TIMESCALE_ENABLE_INTRADAY="true",
    )

    resolved = build_institutional_ingest_config(cfg)
    assert resolved.spark_export is not None
    plan = resolved.spark_export
    assert plan.format is SparkExportFormat.jsonl
    assert plan.partition_columns == ("symbol",)
    assert len(plan.jobs) == 2
    metadata = resolved.metadata.get("spark_export")
    assert metadata["root_path"] == str(tmp_path / "spark")
    assert metadata["format"] == "jsonl"
    assert metadata["dimensions"] == ["daily_bars", "intraday_trades"]
    assert metadata["publish"] is False


def test_config_builder_parses_cross_region_settings() -> None:
    cfg = _base_config(
        TIMESCALE_SYMBOLS="EURUSD",
        TIMESCALE_CROSS_REGION_ENABLED="true",
        TIMESCALE_CROSS_REGION_PRIMARY_REGION="eu-west",
        TIMESCALE_CROSS_REGION_REPLICA_REGION="us-east",
        TIMESCALE_CROSS_REGION_WARN_AFTER_SECONDS="30",
        TIMESCALE_CROSS_REGION_FAIL_AFTER_SECONDS="120",
        TIMESCALE_CROSS_REGION_MAX_ROW_DIFFERENCE_RATIO="0.2",
        TIMESCALE_CROSS_REGION_MAX_SCHEDULE_INTERVAL_SECONDS="600",
        TIMESCALE_CROSS_REGION_DIMENSIONS="daily_bars,intraday_trades",
        TIMESCALE_SECONDARY_URL="sqlite:///replica.db",
        TIMESCALE_SECONDARY_APP="replica-app",
        TIMESCALE_SECONDARY_STATEMENT_TIMEOUT_MS="2500",
        TIMESCALE_SECONDARY_CONNECT_TIMEOUT="10",
    )

    resolved = build_institutional_ingest_config(cfg)
    settings = resolved.cross_region
    assert settings is not None
    assert settings.enabled is True
    assert settings.primary_region == "eu-west"
    assert settings.replica_region == "us-east"
    assert settings.warn_after_seconds == 30.0
    assert settings.fail_after_seconds == 120.0
    assert settings.max_row_difference_ratio == 0.2
    assert settings.max_schedule_interval_seconds == 600.0
    assert settings.dimensions == ("daily_bars", "intraday_trades")
    assert settings.replica_settings is not None
    assert settings.replica_settings.configured is True

    metadata = resolved.metadata.get("cross_region")
    assert metadata is not None
    assert metadata["primary_region"] == "eu-west"
    assert metadata["replica_region"] == "us-east"
    assert metadata["replica_configured"] is True
