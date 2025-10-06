from __future__ import annotations

import json

from tools.understanding import graph_diagnostics


def test_understanding_graph_cli_json_output(capsys) -> None:
    exit_code = graph_diagnostics.main(["--format", "json", "--indent", "0"])
    assert exit_code == 0

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["graph"]["status"] == "ok"
    assert payload["snapshot"]["metadata"]["decision_id"]


def test_understanding_graph_cli_dot_output(capsys) -> None:
    exit_code = graph_diagnostics.main(["--format", "dot"])
    assert exit_code == 0

    captured = capsys.readouterr()
    assert "digraph UnderstandingLoop" in captured.out
