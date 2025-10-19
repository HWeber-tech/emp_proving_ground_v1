import json
from pathlib import Path

import pytest

from src.data_foundation.cache.redis_cache import InMemoryRedis
from src.governance.system_config import DataBackboneMode, EmpTier, SystemConfig

from src.data_foundation.streaming.kafka_stream import KafkaTopicProvisioningSummary

from tools.operations import managed_ingest_connectors as mic


def _build_config(extras: dict[str, str] | None = None) -> SystemConfig:
    payload = {
        "TIMESCALE_URL": "sqlite:///:memory:",
        "TIMESCALE_SYMBOLS": "EURUSD",
        "REDIS_URL": "redis://localhost:6379/0",
        "KAFKA_BOOTSTRAP_SERVERS": "localhost:9092",
    }
    if extras:
        payload.update(extras)
    return SystemConfig(
        tier=EmpTier.tier_1,
        data_backbone_mode=DataBackboneMode.institutional,
        extras=payload,
    )


def _patch_config(monkeypatch: pytest.MonkeyPatch, config: SystemConfig) -> None:
    monkeypatch.setattr(mic, "_load_system_config", lambda *_args: config)


def _patch_redis_success(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        mic,
        "_prepare_redis_client",
        lambda _settings: (InMemoryRedis(), None),
    )


def test_managed_connectors_cli_reports_success(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    config = _build_config()
    _patch_config(monkeypatch, config)
    _patch_redis_success(monkeypatch)

    exit_code = mic.main(["--connectivity", "--format", "json"])
    assert exit_code == 0

    captured = capsys.readouterr().out.strip()
    assert captured
    report = json.loads(captured)

    assert report["should_run"] is True

    configuration = report.get("configuration")
    assert configuration is not None
    assert configuration.get("symbols") == ["EURUSD"]

    inventory = configuration.get("symbol_inventory")
    assert isinstance(inventory, list)
    assert {entry.get("symbol") for entry in inventory} == {"EURUSD"}
    eurusd_inventory = next(entry for entry in inventory if entry.get("symbol") == "EURUSD")
    assert eurusd_inventory.get("margin_currency") == "USD"
    assert eurusd_inventory.get("contract_size") == "100000"
    assert eurusd_inventory.get("pip_decimal_places") == 4
    assert eurusd_inventory.get("long_swap_rate") == "-0.0001"
    assert eurusd_inventory.get("short_swap_rate") == "0.0001"
    assert eurusd_inventory.get("swap_time") == "22:00"

    symbol_metadata = configuration.get("symbol_metadata")
    assert isinstance(symbol_metadata, list)
    assert any(entry.get("symbol") == "EURUSD" for entry in symbol_metadata)
    eurusd_metadata = next(entry for entry in symbol_metadata if entry.get("symbol") == "EURUSD")
    assert eurusd_metadata.get("margin_currency") == "USD"
    assert eurusd_metadata.get("contract_size") == "100000"
    assert eurusd_metadata.get("pip_decimal_places") == 4
    assert eurusd_metadata.get("long_swap_rate") == "-0.0001"
    assert eurusd_metadata.get("short_swap_rate") == "0.0001"
    assert eurusd_metadata.get("swap_time") == "22:00"

    api_keys = configuration.get("api_keys")
    assert isinstance(api_keys, dict)
    assert api_keys["alpha_vantage"]["configured"] is False

    sessions = configuration.get("session_calendars")
    assert isinstance(sessions, list)
    assert any(entry.get("id") == "london_fx" for entry in sessions)

    manifest = {entry["name"]: entry for entry in report["manifest"]}
    assert manifest["timescale"]["configured"] is True
    assert manifest["redis"]["configured"] is True
    assert manifest["kafka"]["configured"] is True

    timescale_manifest = manifest["timescale"]["metadata"]
    assert timescale_manifest.get("symbols") == ["EURUSD"]

    manifest_inventory = timescale_manifest.get("symbol_inventory")
    assert isinstance(manifest_inventory, list)
    manifest_eurusd = next(entry for entry in manifest_inventory if entry.get("symbol") == "EURUSD")
    assert manifest_eurusd.get("margin_currency") == "USD"
    assert manifest_eurusd.get("contract_size") == "100000"
    assert manifest_eurusd.get("pip_decimal_places") == 4
    assert manifest_eurusd.get("long_swap_rate") == "-0.0001"
    assert manifest_eurusd.get("short_swap_rate") == "0.0001"
    assert manifest_eurusd.get("swap_time") == "22:00"

    manifest_symbol_metadata = timescale_manifest.get("symbol_metadata")
    assert isinstance(manifest_symbol_metadata, list)
    manifest_metadata_eurusd = next(
        entry for entry in manifest_symbol_metadata if entry.get("symbol") == "EURUSD"
    )
    assert manifest_metadata_eurusd.get("margin_currency") == "USD"
    assert manifest_metadata_eurusd.get("contract_size") == "100000"
    assert manifest_metadata_eurusd.get("pip_decimal_places") == 4
    assert manifest_metadata_eurusd.get("long_swap_rate") == "-0.0001"
    assert manifest_metadata_eurusd.get("short_swap_rate") == "0.0001"
    assert manifest_metadata_eurusd.get("swap_time") == "22:00"

    connectivity = {entry["name"]: entry for entry in report["connectivity"]}
    assert connectivity["timescale"]["healthy"] is True
    assert "error" not in connectivity["timescale"]
    assert connectivity["redis"]["healthy"] is True
    assert "error" not in connectivity["redis"]


def test_managed_connectors_cli_reports_probe_failure(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = _build_config()
    _patch_config(monkeypatch, config)
    _patch_redis_success(monkeypatch)

    def failing_engine(self, *_args, **_kwargs):
        raise OSError("timescale unreachable")

    monkeypatch.setattr(
        "src.data_foundation.persist.timescale.TimescaleConnectionSettings.create_engine",
        failing_engine,
    )

    exit_code = mic.main(["--connectivity", "--format", "json"])
    assert exit_code == 0

    report = json.loads(capsys.readouterr().out.strip())
    connectivity = {entry["name"]: entry for entry in report["connectivity"]}
    timescale_snapshot = connectivity["timescale"]
    assert timescale_snapshot["healthy"] is False
    assert "timescale unreachable" in timescale_snapshot["error"]


def test_managed_connectors_cli_reports_redis_failure(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = _build_config()
    _patch_config(monkeypatch, config)

    monkeypatch.setattr(
        mic,
        "_prepare_redis_client",
        lambda _settings: (None, "redis offline"),
    )

    exit_code = mic.main(["--connectivity", "--format", "json"])
    assert exit_code == 0

    payload = json.loads(capsys.readouterr().out.strip())
    connectivity = {entry["name"]: entry for entry in payload["connectivity"]}

    redis_snapshot = connectivity["redis"]
    assert redis_snapshot["healthy"] is False
    assert "redis offline" in redis_snapshot["error"]


def test_managed_connectors_cli_can_provision_kafka_topics(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = _build_config({"KAFKA_INGEST_TOPICS": "daily:telemetry.daily"})
    _patch_config(monkeypatch, config)

    captured: dict[str, object] = {}

    class _FakeProvisioner:
        def __init__(self, settings) -> None:  # pragma: no cover - trivial wiring
            captured["settings"] = settings

        def ensure_topics(self, specs, *, dry_run: bool):
            captured["specs"] = tuple(spec.name for spec in specs)
            captured["dry_run"] = dry_run
            return KafkaTopicProvisioningSummary(
                requested=tuple(spec.name for spec in specs),
                existing=(),
                created=tuple(spec.name for spec in specs),
                dry_run=dry_run,
            )

    monkeypatch.setattr(mic, "KafkaTopicProvisioner", _FakeProvisioner)

    exit_code = mic.main([
        "--ensure-topics",
        "--topics-dry-run",
        "--format",
        "json",
    ])
    assert exit_code == 0

    payload = json.loads(capsys.readouterr().out.strip())
    summary = payload["kafka_topic_provisioning"]

    assert summary["dry_run"] is True
    assert summary["created"] == ["telemetry.daily"]
    assert captured["specs"] == ("telemetry.daily",)
    assert captured["dry_run"] is True


def test_managed_connectors_cli_topic_provisioning_handles_missing_topics(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = _build_config()
    _patch_config(monkeypatch, config)

    def _should_not_instantiate(*_args, **_kwargs):
        raise AssertionError("should not instantiate")

    monkeypatch.setattr(mic, "KafkaTopicProvisioner", _should_not_instantiate)

    exit_code = mic.main(["--ensure-topics", "--format", "json"])
    assert exit_code == 0

    payload = json.loads(capsys.readouterr().out.strip())
    summary = payload["kafka_topic_provisioning"]

    assert summary["requested"] == []
    assert "no_topics_configured" in summary["notes"]
