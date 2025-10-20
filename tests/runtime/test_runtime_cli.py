import json
from pathlib import Path
from typing import Callable

import pytest

from types import SimpleNamespace

from src.governance.system_config import (
    ConnectionProtocol,
    DataBackboneMode,
    RunMode,
    SystemConfig,
)
from src.runtime import run_cli


def _bootstrap_config() -> SystemConfig:
    return SystemConfig().with_updated(
        connection_protocol=ConnectionProtocol.bootstrap,
        data_backbone_mode=DataBackboneMode.bootstrap,
        extras={
            "BOOTSTRAP_SYMBOLS": "EURUSD",
            "RUNTIME_HEALTHCHECK_AUTH_SECRET": "cli-runtime-health-secret",
        },
    )


@pytest.fixture()
def cli_env(monkeypatch) -> None:
    from src.runtime import cli as runtime_cli

    def _from_env(cls) -> SystemConfig:  # type: ignore[override]
        return _bootstrap_config()

    monkeypatch.setattr(runtime_cli.SystemConfig, "from_env", classmethod(_from_env))


@pytest.mark.asyncio()
async def test_runtime_cli_summary_json(cli_env, capsys, tmp_path: Path) -> None:
    _ = cli_env
    exit_code = await run_cli(
        [
            "summary",
            "--json",
            "--skip-ingest",
            "--symbols",
            "EURUSD",
            "--duckdb-path",
            str(tmp_path / "tier0.duckdb"),
        ]
    )

    assert exit_code == 0
    output = capsys.readouterr().out
    payload = json.loads(output)

    config_payload = payload["config"]
    assert config_payload["tier"] in {"tier-0", "tier_0"}
    assert config_payload["mode"] == "paper"
    assert config_payload["connection_protocol"] == "bootstrap"
    assert config_payload["confirm_live"] is False
    assert payload["runtime"]["ingestion"]["name"] == "skip-ingest"


@pytest.mark.asyncio()
async def test_runtime_cli_summary_paper_mode(cli_env, capsys, tmp_path: Path) -> None:
    _ = cli_env
    exit_code = await run_cli(
        [
            "summary",
            "--json",
            "--paper-mode",
            "--skip-ingest",
            "--symbols",
            "EURUSD",
            "--duckdb-path",
            str(tmp_path / "tier0.duckdb"),
        ]
    )

    assert exit_code == 0
    output = capsys.readouterr().out
    payload = json.loads(output)

    config_payload = payload["config"]
    assert config_payload["mode"] == "paper"
    assert config_payload["connection_protocol"] == "paper"
    assert config_payload["confirm_live"] is False


@pytest.mark.asyncio()
async def test_runtime_cli_summary_live_mode(
    cli_env, capsys, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _ = cli_env
    from src.runtime import cli as runtime_cli

    captured_config: dict[str, SystemConfig] = {}

    class _StubApp:
        def __init__(self, config: SystemConfig) -> None:
            self.config = config
            self.event_bus = SimpleNamespace(is_running=lambda: False)

        async def __aenter__(self) -> "_StubApp":
            return self

        async def __aexit__(self, exc_type, exc, tb) -> bool:
            return False

        def summary(self) -> dict[str, str]:
            return {"app": "stub"}

        async def shutdown(self) -> None:  # pragma: no cover - simple stub
            return None

    class _StubRuntime:
        def __init__(self) -> None:
            self.trading = object()
            self.ingestion = SimpleNamespace(factory=lambda: None)
            self.startup_callbacks: list[Callable[[], object]] = []

        def summary(self) -> dict[str, object]:
            return {"ingestion": {"name": "stub"}}

    async def _stub_build_app(*, config: SystemConfig) -> _StubApp:
        captured_config["value"] = config
        return _StubApp(config)

    def _stub_build_runtime(
        app: _StubApp, *, skip_ingest: bool, symbols_csv: str, duckdb_path: str
    ) -> _StubRuntime:
        assert app.config is captured_config["value"]
        return _StubRuntime()

    monkeypatch.setattr(runtime_cli, "build_professional_predator_app", _stub_build_app)
    monkeypatch.setattr(
        runtime_cli,
        "build_professional_runtime_application",
        _stub_build_runtime,
    )

    exit_code = await run_cli(
        [
            "summary",
            "--json",
            "--live-mode",
            "--no-trading",
            "--skip-ingest",
            "--symbols",
            "EURUSD",
            "--duckdb-path",
            str(tmp_path / "tier0.duckdb"),
        ]
    )

    assert exit_code == 0
    output = capsys.readouterr().out
    payload = json.loads(output)

    config_payload = payload["config"]
    assert config_payload["mode"] == "live"
    assert config_payload["connection_protocol"] == "fix"
    assert config_payload["confirm_live"] is True
    assert captured_config["value"].run_mode is RunMode.live


@pytest.mark.asyncio()
async def test_runtime_cli_run_without_trading(cli_env, tmp_path: Path) -> None:
    _ = cli_env
    exit_code = await run_cli(
        [
            "run",
            "--skip-ingest",
            "--no-trading",
            "--timeout",
            "0.1",
            "--symbols",
            "EURUSD",
            "--duckdb-path",
            str(tmp_path / "tier0.duckdb"),
        ]
    )

    assert exit_code == 0


@pytest.mark.asyncio()
async def test_runtime_cli_restart_cycles(cli_env, capsys, tmp_path: Path) -> None:
    _ = cli_env
    exit_code = await run_cli(
        [
            "restart",
            "--cycles",
            "2",
            "--skip-ingest",
            "--no-trading",
            "--timeout",
            "0.1",
            "--symbols",
            "EURUSD",
            "--duckdb-path",
            str(tmp_path / "tier0.duckdb"),
        ]
    )

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "Running cycle 1/2" in output
    assert "Completed cycle 2/2" in output
