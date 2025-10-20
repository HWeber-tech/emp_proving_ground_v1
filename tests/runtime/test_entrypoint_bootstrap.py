"""Tests for deterministic boot helpers in the runtime entrypoint."""

import json
from pathlib import Path

from src.governance.system_config import SystemConfig

from main import (
    DEFAULT_CONFIG_SNAPSHOT_PATH,
    _capture_configuration_snapshot,
    _resolve_config_snapshot_path,
)


def test_resolve_config_snapshot_path_defaults_when_missing() -> None:
    path = _resolve_config_snapshot_path({})

    assert path == DEFAULT_CONFIG_SNAPSHOT_PATH


def test_resolve_config_snapshot_path_allows_disabling() -> None:
    assert _resolve_config_snapshot_path({"CONFIG_SNAPSHOT_PATH": "off"}) is None


def test_capture_configuration_snapshot_persists(tmp_path) -> None:
    target = tmp_path / "snapshots" / "config_snapshot.json"
    extras = {"CONFIG_SNAPSHOT_PATH": str(target)}
    config = SystemConfig()

    persisted, error, attempted = _capture_configuration_snapshot(config, extras, rng_seed=99)

    assert error is None
    assert attempted == target
    assert persisted == target

    payload = json.loads(target.read_text(encoding="utf-8"))
    metadata = payload.get("metadata")
    assert metadata is not None
    assert metadata.get("rng_seed") == 99
    assert metadata.get("source") == "runtime_boot"

