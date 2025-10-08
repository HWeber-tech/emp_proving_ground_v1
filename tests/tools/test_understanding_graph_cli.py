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
    health_metrics = payload["graph"]["metadata"].get("health_metrics")
    assert health_metrics is not None
    assert health_metrics["fast_weight"]["graph_sparsity"] == 0.5
    assert "momentum_breakout" in health_metrics["strategy_activation"]["dominant_strategies"]


def test_understanding_graph_cli_dot_output(capsys) -> None:
    exit_code = graph_diagnostics.main(["--format", "dot"])
    assert exit_code == 0

    captured = capsys.readouterr()
    assert "digraph UnderstandingLoop" in captured.out
