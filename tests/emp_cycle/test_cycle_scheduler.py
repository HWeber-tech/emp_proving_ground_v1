from __future__ import annotations

import json
from pathlib import Path

import pytest

from emp.cli import emp_cycle_scheduler
from emp.core import findings_memory


@pytest.fixture(autouse=True)
def _strategy_factory(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EMP_STRATEGY_FACTORY", "tests.emp_cycle.strategy_stub:make_strategy")


def _setup_idea(conn, params):
    artefacts = findings_memory.compute_params_artifacts(params)
    novelty = findings_memory.nearest_novelty(conn, params, artefacts=artefacts)
    return findings_memory.add_idea(conn, params, novelty, artefacts=artefacts).id


def test_scheduler_promotes_candidate(tmp_path: Path) -> None:
    db_path = tmp_path / "experiments.sqlite"
    baseline_path = tmp_path / "baseline.json"
    conn = findings_memory.connect(db_path)

    candidate_params = {"weight": 1.0}
    fid = _setup_idea(conn, candidate_params)

    args = [
        "--db-path",
        str(db_path),
        "--baseline-json",
        str(baseline_path),
        "--quick-threshold",
        "0.3",
        "--ucb-c",
        "0.2",
        "--full-timeout-secs",
        "5",
        "--max-full",
        "1",
        "--note",
        "scheduler:test",
    ]

    exit_code = emp_cycle_scheduler.main(args)
    assert exit_code == 0

    row = conn.execute("SELECT stage, tested_at, notes FROM findings WHERE id = ?", (fid,)).fetchone()
    assert row is not None
    assert row["stage"] in {"tested", "progress"}
    assert row["tested_at"] is not None
    assert "scheduler:test" in (row["notes"] or "")

    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    assert "sharpe" in baseline


def test_scheduler_handles_failed_quick(tmp_path: Path) -> None:
    db_path = tmp_path / "experiments.sqlite"
    baseline_path = tmp_path / "baseline.json"
    conn = findings_memory.connect(db_path)

    failing_params = {"weight": 0.1}
    fid = _setup_idea(conn, failing_params)

    args = [
        "--db-path",
        str(db_path),
        "--baseline-json",
        str(baseline_path),
        "--quick-threshold",
        "3.0",
        "--max-full",
        "0",
    ]

    exit_code = emp_cycle_scheduler.main(args)
    assert exit_code == 0

    row = conn.execute("SELECT stage, notes FROM findings WHERE id = ?", (fid,)).fetchone()
    assert row is not None
    assert row["stage"] == "idea"
    assert "failed_screen" in (row["notes"] or "")
