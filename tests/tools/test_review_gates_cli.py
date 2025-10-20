from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.governance.review_gates import main


def _write_definitions(path: Path) -> None:
    path.write_text(
        """
        gates:
          - gate_id: alpha_gate
            title: Alpha Gate
            description: Example gate
            severity: high
            criteria:
              - id: requirement_a
                description: Requirement A
                mandatory: true
              - id: optional_b
                description: Optional B
                mandatory: false
        """.strip()
    )


def test_review_gates_status_cli(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    definitions_path = tmp_path / "defs.yaml"
    _write_definitions(definitions_path)

    result = main(
        [
            "--definitions",
            str(definitions_path),
            "--state",
            str(tmp_path / "state.json"),
            "status",
        ]
    )
    assert result == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["gates"]
    assert payload["gates"][0]["status"] == "todo"
    assert payload["patch_proposals"] == []


def test_review_gates_decide_cli(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    definitions_path = tmp_path / "defs.yaml"
    state_path = tmp_path / "custom_state.json"
    _write_definitions(definitions_path)

    result = main(
        [
            "--definitions",
            str(definitions_path),
            "--state",
            str(tmp_path / "unused.json"),
            "decide",
            "--gate",
            "alpha_gate",
            "--verdict",
            "pass",
            "--decided-at",
            "2025-01-02T03:04:05+00:00",
            "--decided-by",
            "Ops Lead",
            "--criterion",
            "requirement_a=met",
            "--persist",
            str(state_path),
        ]
    )
    assert result == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["gate"] == "alpha_gate"
    assert payload["status"] == "completed"
    assert payload["criteria"]["requirement_a"] == "met"
    assert Path(payload["state_path"]).exists()

    state_payload = json.loads(state_path.read_text(encoding="utf-8"))
    assert state_payload["gates"][0]["gate_id"] == "alpha_gate"
    assert state_payload["gates"][0]["verdict"] == "pass"

    # Status command should now report completed
    result = main(
        [
            "--definitions",
            str(definitions_path),
            "--state",
            str(state_path),
            "status",
        ]
    )
    assert result == 0
    payload = json.loads(capsys.readouterr().out)
    gate_summary = payload["gates"][0]
    assert gate_summary["status"] == "completed"
    assert gate_summary["verdict"] == "pass"
    assert payload["patch_proposals"] == []
