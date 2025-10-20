"""Regression tests for container deployment SystemConfig profiles."""

from __future__ import annotations

from pathlib import Path

from src.governance.system_config import (
    ConnectionProtocol,
    DataBackboneMode,
    EmpEnvironment,
    EmpTier,
    RunMode,
    SystemConfig,
)


def _load_profile(name: str) -> SystemConfig:
    root = Path(__file__).resolve().parents[2]
    path = root / "config" / "deployment" / name
    return SystemConfig.from_yaml(path, env={})


def test_runtime_dev_profile_defaults() -> None:
    config = _load_profile("runtime_dev.yaml")

    assert config.run_mode is RunMode.mock
    assert config.environment is EmpEnvironment.demo
    assert config.tier is EmpTier.tier_0
    assert config.connection_protocol is ConnectionProtocol.bootstrap
    assert config.data_backbone_mode is DataBackboneMode.institutional

    extras = config.extras
    assert extras["TIMESCALEDB_URL"].startswith("postgresql+psycopg2://emp_user:emp_password@timescale")
    assert extras["REDIS_URL"] == "redis://redis:6379/0"
    assert extras["KAFKA_BOOTSTRAP_SERVERS"] == "kafka:9092"
    assert extras["RUNTIME_HEALTHCHECK_PORT"] == "8000"
    assert extras["RUNTIME_HEALTHCHECK_ENABLED"] == "true"
    assert extras["RUNTIME_HEALTHCHECK_AUTH_SECRET"] == '${RUNTIME_HEALTHCHECK_AUTH_SECRET:-change-me}'
    assert extras["RUNTIME_LOG_CONTEXT"] == '{"deployment":"dev-container"}'


def test_runtime_paper_profile_defaults() -> None:
    config = _load_profile("runtime_paper.yaml")

    assert config.run_mode is RunMode.paper
    assert config.environment is EmpEnvironment.demo
    assert config.tier is EmpTier.tier_1
    assert config.connection_protocol is ConnectionProtocol.paper
    assert config.data_backbone_mode is DataBackboneMode.institutional

    extras = config.extras
    assert extras["TIMESCALEDB_URL"].startswith("postgresql+psycopg2://emp_user:emp_password@timescale")
    assert extras["REDIS_URL"] == "redis://redis:6379/0"
    assert extras["KAFKA_BOOTSTRAP_SERVERS"] == "kafka:9092"
    assert extras["RUNTIME_HEALTHCHECK_PORT"] == "8000"
    assert extras["RUNTIME_HEALTHCHECK_ENABLED"] == "true"
    assert extras["RUNTIME_HEALTHCHECK_AUTH_SECRET"] == '${RUNTIME_HEALTHCHECK_AUTH_SECRET:-change-me}'
    assert extras["RUNTIME_LOG_STRUCTURED"] == "true"
    assert extras["RUNTIME_LOG_CONTEXT"] == '{"deployment":"paper-container"}'
