from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.thinking.evaluation import (
    AblationMetrics,
    AblationResult,
    evaluate_ablation_gates,
    render_ablation_markdown,
    run_ablation_suite,
)
from tools.run_ablation_suite import main as run_ablation_cli


def test_ablation_suite_returns_expected_scenarios() -> None:
    results = run_ablation_suite()
    scenario_order = [result.scenario for result in results]
    assert scenario_order == [
        "lora-head",
        "no-depth",
        "no-ofi",
        "per-domain-head",
        "k=0.3",
        "k=0.5",
        "k=0.7",
    ]

    gates = evaluate_ablation_gates(results)
    assert all(gate.passed for gate in gates)

    markdown = render_ablation_markdown(results, gates)
    assert "Scenario" in markdown
    assert "Gate" in markdown


def test_ablation_gates_detect_accuracy_degradation() -> None:
    baseline_results = list(run_ablation_suite())
    degraded: list[AblationResult] = []
    for result in baseline_results:
        if result.scenario == "no-depth":
            degraded.append(
                AblationResult(
                    scenario=result.scenario,
                    parameters=dict(result.parameters),
                    metrics=AblationMetrics(
                        accuracy=0.60,
                        sharpe=result.metrics.sharpe,
                        alpha_bps=result.metrics.alpha_bps,
                        max_drawdown_bps=result.metrics.max_drawdown_bps,
                        latency_ms=result.metrics.latency_ms,
                    ),
                )
            )
        else:
            degraded.append(result)

    gates = evaluate_ablation_gates(degraded)
    failing = {gate.gate_id for gate in gates if not gate.passed}
    assert "no_depth_degradation" in failing
    assert "accuracy_floor" in failing


def test_ablation_gates_detect_k_trend_regression() -> None:
    baseline_results = list(run_ablation_suite())
    regressed: list[AblationResult] = []
    for result in baseline_results:
        if result.scenario == "k=0.7":
            regressed.append(
                AblationResult(
                    scenario=result.scenario,
                    parameters=dict(result.parameters),
                    metrics=AblationMetrics(
                        accuracy=result.metrics.accuracy,
                        sharpe=result.metrics.sharpe,
                        alpha_bps=result.metrics.alpha_bps - 2.2,
                        max_drawdown_bps=result.metrics.max_drawdown_bps,
                        latency_ms=result.metrics.latency_ms,
                    ),
                )
            )
        else:
            regressed.append(result)

    gates = evaluate_ablation_gates(regressed)
    failing = {gate.gate_id for gate in gates if not gate.passed}
    assert "k_alpha_trend" in failing
    assert "alpha_floor" in failing


def test_ablation_cli_writes_artifacts(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    output_path = tmp_path / "results.json"
    summary_path = tmp_path / "results.md"

    exit_code = run_ablation_cli(
        [
            "--output",
            str(output_path),
            "--summary",
            str(summary_path),
            "--enforce-gates",
        ]
    )
    assert exit_code == 0

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["scenarios"]
    assert payload["gates"]

    assert output_path.exists()
    written_payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert written_payload == payload

    summary_contents = summary_path.read_text(encoding="utf-8")
    assert "Scenario" in summary_contents
    assert "Gate" in summary_contents
