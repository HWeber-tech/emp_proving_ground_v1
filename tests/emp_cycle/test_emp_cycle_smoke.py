from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from emp.core import findings_memory


def test_emp_cycle_smoke(tmp_path):
    ideas = [
        {"name": "alpha", "weight": 0.6},
        {"name": "beta", "weight": 1.4},
        {"name": "gamma", "weight": 0.3, "bias": 0.1},
    ]

    ideas_path = tmp_path / "ideas.json"
    ideas_path.write_text(json.dumps(ideas), encoding="utf-8")

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
    ]

    result = subprocess.run(cmd, env=env, cwd=Path.cwd(), check=True, capture_output=True, text=True)
    assert result.returncode == 0

    conn = findings_memory.connect(db_path)
    stages = dict(conn.execute("SELECT id, stage FROM findings").fetchall())

    assert "progress" in stages.values() or "tested" in stages.values()
    assert "screened" in stages.values()

    with open(baseline_path, "r", encoding="utf-8") as handle:
        baseline_data = json.load(handle)
    assert baseline_data["sharpe"] > 0
