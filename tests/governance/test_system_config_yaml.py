"""Tests for SystemConfig YAML loading helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.governance.system_config import (
    DataBackboneMode,
    EmpEnvironment,
    RunMode,
    SystemConfig,
    SystemConfigLoadError,
)


def test_from_yaml_nested_section_overrides_env(tmp_path: Path) -> None:
    config_path = tmp_path / "system_config.yaml"
    config_path.write_text(
        """
system_config:
  run_mode: live
  environment: production
  data_backbone_mode: institutional
  extras:
    KAFKA_BROKERS: kafka-1:9092
    REDIS_URL: redis://cache
        """.strip(),
        encoding="utf-8",
    )

    config = SystemConfig.from_yaml(config_path, env={})

    assert config.run_mode is RunMode.live
    assert config.environment is EmpEnvironment.production
    assert config.data_backbone_mode is DataBackboneMode.institutional
    assert config.extras == {
        "KAFKA_BROKERS": "kafka-1:9092",
        "REDIS_URL": "redis://cache",
    }


def test_from_yaml_respects_env_when_not_overridden(tmp_path: Path) -> None:
    config_path = tmp_path / "system_config.yaml"
    config_path.write_text(
        """
extras:
  CUSTOM_FLAG: enabled
        """.strip(),
        encoding="utf-8",
    )

    env_mapping = {
        "RUN_MODE": "paper",
        "EMP_ENVIRONMENT": "staging",
        "DATA_BACKBONE_MODE": "institutional",
    }

    config = SystemConfig.from_yaml(config_path, env=env_mapping)

    assert config.run_mode is RunMode.paper
    assert config.environment is EmpEnvironment.staging
    assert config.data_backbone_mode is DataBackboneMode.institutional
    assert config.extras == {"CUSTOM_FLAG": "enabled"}


def test_from_yaml_overrides_env_values_when_provided(tmp_path: Path) -> None:
    config_path = tmp_path / "system_config.yaml"
    config_path.write_text(
        """
run_mode: live
system_config:
  environment: demo
  extras:
    CUSTOM_FLAG: disabled
        """.strip(),
        encoding="utf-8",
    )

    env_mapping = {
        "RUN_MODE": "paper",
        "EMP_ENVIRONMENT": "production",
    }

    config = SystemConfig.from_yaml(config_path, env=env_mapping)

    assert config.run_mode is RunMode.live  # YAML overrides env
    assert config.environment is EmpEnvironment.demo
    # Extras merged from top-level and nested with nested winning when keys collide
    assert config.extras == {"CUSTOM_FLAG": "disabled"}


def test_from_yaml_missing_file(tmp_path: Path) -> None:
    missing = tmp_path / "missing.yaml"

    with pytest.raises(SystemConfigLoadError):
        SystemConfig.from_yaml(missing, env={})
