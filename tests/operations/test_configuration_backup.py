from __future__ import annotations

import json
from datetime import UTC, datetime
from zipfile import ZipFile

from src.operations.configuration_backup import (
    ConfigurationBackupPlan,
    ConfigurationBackupSource,
    build_configuration_backup,
    default_configuration_backup_plan,
)


def test_build_configuration_backup_creates_archive(tmp_path) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "settings.yaml").write_text("alpha: 1\n", encoding="utf-8")
    (config_dir / ".secret.env").write_text("TOKEN=abc\n", encoding="utf-8")

    env_templates = tmp_path / "env_templates"
    env_templates.mkdir()
    (env_templates / "runtime.env").write_text("MODE=paper\n", encoding="utf-8")

    plan = ConfigurationBackupPlan(
        sources=(
            ConfigurationBackupSource(name="config", path=config_dir),
            ConfigurationBackupSource(
                name="env_templates",
                path=env_templates,
                include_patterns=("*.env",),
            ),
        ),
        metadata={"tier": "paper"},
    )

    result = build_configuration_backup(
        plan,
        tmp_path / "backups",
        now=datetime(2024, 1, 1, 12, 0, tzinfo=UTC),
    )

    assert result.archive_path.exists()
    assert result.manifest_path.exists()

    with ZipFile(result.archive_path) as archive:
        names = sorted(archive.namelist())

    assert names[0] == "config/.secret.env"
    assert "config/settings.yaml" in names
    assert "env_templates/runtime.env" in names
    assert "manifest.json" in names

    manifest_payload = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest_payload["metadata"]["tier"] == "paper"
    assert len(manifest_payload["entries"]) == 3


def test_configuration_backup_handles_missing_sources(tmp_path) -> None:
    plan = ConfigurationBackupPlan(
        sources=(
            ConfigurationBackupSource(name="config", path=tmp_path / "missing"),
        )
    )

    result = build_configuration_backup(
        plan,
        tmp_path / "backups",
        now=datetime(2024, 1, 1, tzinfo=UTC),
    )

    manifest_payload = result.manifest.as_dict()
    assert manifest_payload["missing_sources"] == ["config"]

    with ZipFile(result.archive_path) as archive:
        assert archive.namelist() == ["manifest.json"]


def test_default_configuration_backup_plan_detects_sources(tmp_path) -> None:
    (tmp_path / "config").mkdir()
    (tmp_path / "env_templates").mkdir()
    (tmp_path / "config.yaml").write_text("run_mode: paper\n", encoding="utf-8")

    plan = default_configuration_backup_plan(tmp_path)

    names = {source.name for source in plan.sources}
    assert {"config", "env_templates"}.issubset(names)
    assert any(name.endswith("config.yaml") for name in names)
    assert plan.metadata["root"] == str(tmp_path)
