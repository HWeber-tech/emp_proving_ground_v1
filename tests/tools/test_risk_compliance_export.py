import json
from pathlib import Path

import pytest

from tools.telemetry import export_risk_compliance_snapshots as exporter


class _DummyApp:
    def __init__(self, summary: dict[str, object]) -> None:
        self._summary = summary
        self.shutdown_calls = 0

    async def __aenter__(self) -> "_DummyApp":  # pragma: no cover - trivial glue
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:  # pragma: no cover - trivial glue
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


def test_exporter_includes_journal_summary(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    summary = {
        "risk": {"snapshot": {"status": "ok"}},
        "execution": {"snapshot": {"status": "warn"}},
        "compliance_readiness": {"snapshot": {"status": "pass"}},
        "compliance_workflows": {"snapshot": {"status": "ok"}},
        "compliance": {"status": "GREEN"},
        "kyc": {"status": "APPROVED"},
    }
    _configure_builder(monkeypatch, summary)

    captured: dict[str, object] = {}

    def _fake_summarise(config, *, strategy_id, execution_service, recent):
        captured.update(strategy_id=strategy_id, execution_service=execution_service, recent=recent)
        return {
            "metadata": {"configured": True, "recent_limit": recent},
            "compliance": {"stats": {"total_records": 1}, "recent": []},
            "kyc": {"stats": {"total_cases": 0}, "recent": []},
            "execution": {"stats": {"total_snapshots": 2}, "recent": []},
        }

    monkeypatch.setattr(exporter, "_summarise_journals", _fake_summarise)

    output_path = tmp_path / "governance.json"
    exit_code = exporter.main(
        [
            "--output",
            str(output_path),
            "--recent",
            "2",
            "--strategy-id",
            "alpha",
            "--execution-service",
            "primary",
        ]
    )

    assert exit_code == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert payload["snapshots"]["risk"]["snapshot"]["status"] == "ok"
    assert payload["journal_summary"]["metadata"]["configured"] is True
    assert captured == {"strategy_id": "alpha", "execution_service": "primary", "recent": 2}


def test_exporter_errors_when_section_missing(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    summary = {"risk": {"snapshot": {"status": "ok"}}}
    _configure_builder(monkeypatch, summary)
    monkeypatch.setattr(exporter, "_summarise_journals", lambda *a, **k: {})

    exit_code = exporter.main(["--section", "risk", "--section", "execution"])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "warning: missing sections: execution" in captured.err


def test_exporter_allows_missing_when_flag_set(monkeypatch: pytest.MonkeyPatch) -> None:
    summary = {"risk": {"snapshot": {"status": "ok"}}}
    _configure_builder(monkeypatch, summary)
    monkeypatch.setattr(exporter, "_summarise_journals", lambda *a, **k: {})

    exit_code = exporter.main(["--section", "risk", "--section", "execution", "--allow-missing"])

    assert exit_code == 0
