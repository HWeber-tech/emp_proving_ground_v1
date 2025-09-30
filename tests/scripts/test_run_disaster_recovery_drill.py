from __future__ import annotations

from pathlib import Path

import json

from src.operations.disaster_recovery import DisasterRecoveryStatus
import scripts.run_disaster_recovery_drill as cli


def test_cli_writes_outputs(tmp_path: Path, monkeypatch) -> None:
    called = {}

    class DummyReport:
        def __init__(self) -> None:
            self.status = DisasterRecoveryStatus.ready

        def to_markdown(self) -> str:
            return "# mock report"

        def as_dict(self) -> dict[str, object]:
            return {"status": "ready"}

    def fake_simulator(*args, **kwargs):
        called["args"] = (args, kwargs)
        return DummyReport()

    monkeypatch.setattr(cli, "simulate_default_disaster_recovery", fake_simulator)
    monkeypatch.setattr(cli, "format_disaster_recovery_markdown", lambda report: report.to_markdown())

    output = tmp_path / "drill.md"
    json_output = tmp_path / "drill.json"

    exit_code = cli.main(
        [
            "--output",
            str(output),
            "--json-output",
            str(json_output),
            "--scenario",
            "test-scenario",
            "--fail-dimension",
            "daily_bars",
            "--fail-dimension",
            "intraday_trades",
        ]
    )

    assert exit_code == 0
    assert output.read_text(encoding="utf-8") == "# mock report"
    payload = json.loads(json_output.read_text(encoding="utf-8"))
    assert payload == {"status": "ready"}
    assert "args" in called


def test_cli_exit_code_on_degraded(tmp_path: Path, monkeypatch) -> None:
    class DummyReport:
        def __init__(self) -> None:
            self.status = DisasterRecoveryStatus.degraded

        def to_markdown(self) -> str:
            return "# degraded"

        def as_dict(self) -> dict[str, object]:
            return {"status": "degraded"}

    monkeypatch.setattr(cli, "simulate_default_disaster_recovery", lambda **_: DummyReport())
    monkeypatch.setattr(cli, "format_disaster_recovery_markdown", lambda report: report.to_markdown())

    output = tmp_path / "drill.md"

    exit_code = cli.main(["--output", str(output)])

    assert exit_code == 1
