import json
from pathlib import Path

import pytest

from src.governance.system_config import ConnectionProtocol, DataBackboneMode, SystemConfig
from src.runtime import run_cli


def _bootstrap_config() -> SystemConfig:
    return SystemConfig().with_updated(
        connection_protocol=ConnectionProtocol.bootstrap,
        data_backbone_mode=DataBackboneMode.bootstrap,
        extras={"BOOTSTRAP_SYMBOLS": "EURUSD"},
    )


@pytest.fixture()
def cli_env(monkeypatch) -> None:
    from src.runtime import cli as runtime_cli

    def _from_env(cls) -> SystemConfig:  # type: ignore[override]
        return _bootstrap_config()

    monkeypatch.setattr(runtime_cli.SystemConfig, "from_env", classmethod(_from_env))


@pytest.mark.asyncio()
async def test_runtime_cli_summary_json(cli_env, capsys, tmp_path: Path) -> None:
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

    assert payload["config"]["tier"] in {"tier-0", "tier_0"}
    assert payload["runtime"]["ingestion"]["name"] == "skip-ingest"


@pytest.mark.asyncio()
async def test_runtime_cli_run_without_trading(cli_env, tmp_path: Path) -> None:
    exit_code = await run_cli(
        [
            "run",
            "--skip-ingest",
            "--no-trading",
            "--symbols",
            "EURUSD",
            "--duckdb-path",
            str(tmp_path / "tier0.duckdb"),
        ]
    )

    assert exit_code == 0


@pytest.mark.asyncio()
async def test_runtime_cli_restart_cycles(cli_env, capsys, tmp_path: Path) -> None:
    exit_code = await run_cli(
        [
            "restart",
            "--cycles",
            "2",
            "--skip-ingest",
            "--no-trading",
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
