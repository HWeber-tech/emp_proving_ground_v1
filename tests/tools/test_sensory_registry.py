"""Tests for the sensory registry CLI."""

from __future__ import annotations

import json
from pathlib import Path

from tools.sensory import registry


def test_build_registry_includes_all_dimensions() -> None:
    entries = registry.build_registry()

    dimensions = {entry.dimension for entry in entries}
    assert dimensions == {"HOW", "WHAT", "WHEN", "WHY", "ANOMALY"}


def test_format_markdown_lists_config_fields() -> None:
    entries = registry.build_registry()
    markdown = registry.format_markdown(entries)

    assert "Sensory registry" in markdown
    assert "sensory.how.how_sensor.HowSensor" in markdown
    assert "`minimum_confidence`" in markdown
    assert "does not expose configuration" in markdown


def test_cli_supports_json_and_markdown(tmp_path: Path) -> None:
    destination = tmp_path / "registry.md"
    exit_code = registry.main(["--output", str(destination)])

    assert exit_code == 0
    assert destination.exists()
    content = destination.read_text(encoding="utf-8")
    assert "Sensory registry" in content

    json_exit = registry.main(["--format", "json"])
    assert json_exit == 0

    payload = json.loads(registry.format_json(registry.build_registry()))
    assert payload[0]["qualified_name"].startswith("sensory")
