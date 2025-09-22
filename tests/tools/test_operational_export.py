from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.telemetry import export_operational_snapshots as exporter


class _DummyApp:
    def __init__(self, summary: dict[str, object]) -> None:
        self._summary = summary
        self.shutdown_calls = 0

    async def __aenter__(self) -> "_DummyApp":  # pragma: no cover - trivial
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:  # pragma: no cover - trivial
        return None

    async def shutdown(self) -> None:
        self.shutdown_calls += 1

    def summary(self) -> dict[str, object]:
        return self._summary


def _configure_builder(monkeypatch: pytest.MonkeyPatch, summary: dict[str, object]) -> None:
    async def _build(**_ignored) -> _DummyApp:
        return _DummyApp(summary)

    monkeypatch.setattr(exporter, "_load_builder", lambda: _build)


@pytest.fixture(autouse=True)
def _stub_config(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.governance.system_config import SystemConfig

    def _from_env(cls) -> SystemConfig:  # type: ignore[override]
        return SystemConfig()

    monkeypatch.setattr(exporter.SystemConfig, "from_env", classmethod(_from_env))


def test_exporter_writes_requested_sections(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    summary = {
        "professional_readiness": {"snapshot": {"status": "ok"}},
        "security": {"snapshot": {"status": "warn"}},
        "incident_response": {"snapshot": {"status": "ok"}},
        "system_validation": {"snapshot": {"status": "pass"}},
    }
    _configure_builder(monkeypatch, summary)

    output_path = tmp_path / "telemetry.json"
    exit_code = exporter.main(["--output", str(output_path)])

    assert exit_code == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert payload["sections_requested"] == list(exporter.DEFAULT_SECTIONS)
    assert payload["missing_sections"] == []
    assert payload["snapshots"]["security"]["snapshot"]["status"] == "warn"


def test_exporter_errors_when_section_missing(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    summary = {"professional_readiness": {"snapshot": {"status": "ok"}}}
    _configure_builder(monkeypatch, summary)

    exit_code = exporter.main(["--section", "professional_readiness", "--section", "security"])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "warning: missing sections: security" in captured.err


def test_exporter_allows_missing_when_flag_set(monkeypatch: pytest.MonkeyPatch) -> None:
    summary = {"professional_readiness": {"snapshot": {"status": "ok"}}}
    _configure_builder(monkeypatch, summary)

    exit_code = exporter.main(
        ["--section", "professional_readiness", "--section", "security", "--allow-missing"]
    )

    assert exit_code == 0
