from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from emp.core import findings_memory


def _run_cycle(tmp_path, idea, *, timeout: float | None = None):
    ideas_path = tmp_path / "ideas.json"
    ideas_path.write_text(json.dumps([idea]), encoding="utf-8")

    db_path = tmp_path / "experiments.sqlite"
    baseline_path = tmp_path / "baseline.json"

    env = os.environ.copy()
    env["EMP_STRATEGY_FACTORY"] = "tests.emp_cycle.strategy_stub:make_strategy"

    cmd = [
        sys.executable,
        "-m",
        "emp.cli.emp_cycle",
        "--ideas-json",
        str(ideas_path),
        "--db-path",
        str(db_path),
        "--baseline-json",
        str(baseline_path),
        "--quick-threshold",
        "0.3",
        "--ucb-c",
        "0.2",
        "--seed",
        "42",
        "--git-sha",
        "timeouttest",
    ]
    if timeout is not None:
        cmd.extend(["--full-timeout-secs", str(timeout)])
    return subprocess.run(cmd, env=env, cwd=Path.cwd(), check=True, capture_output=True, text=True), db_path


def test_full_backtest_timeout(tmp_path):
    result, db_path = _run_cycle(
        tmp_path,
        {"name": "timeout", "weight": 1.0, "mode": "full_timeout", "bias": 0.2},
        timeout=0.1,
    )
    assert result.returncode == 0

    conn = findings_memory.connect(db_path)
    row = conn.execute("SELECT stage, notes FROM findings ORDER BY id DESC LIMIT 1").fetchone()
    assert row["stage"] == "tested"
    assert "full_eval_error:timeout" in (row["notes"] or "")


def test_full_backtest_exception(tmp_path):
    cmd_result, db_path = _run_cycle(
        tmp_path,
        {"name": "fail", "weight": 1.0, "mode": "full_error"},
    )
    assert cmd_result.returncode == 0

    conn = findings_memory.connect(db_path)
    row = conn.execute("SELECT stage, notes FROM findings ORDER BY id DESC LIMIT 1").fetchone()
    assert row["stage"] == "tested"
    assert "full_eval_error:" in (row["notes"] or "")
