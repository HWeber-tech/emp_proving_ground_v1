from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_run_simulation_creates_summary(tmp_path: Path) -> None:
    summary_path = tmp_path / "summary.json"
    diary_path = tmp_path / "diary.jsonl"

    env = os.environ.copy()
    env.pop("RUN_MODE", None)
    env.pop("EMP_ENVIRONMENT", None)
    env.pop("EMP_TIER", None)
    env.pop("DATA_BACKBONE_MODE", None)

    cmd = [
        sys.executable,
        "tools/runtime/run_simulation.py",
        "--timeout",
        "2",
        "--tick-interval",
        "0.05",
        "--max-ticks",
        "5",
        "--summary-path",
        str(summary_path),
        "--diary-path",
        str(diary_path),
    ]

    result = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        env=env,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    assert result.returncode == 0
    assert summary_path.exists(), result.stdout + result.stderr
    payload = json.loads(summary_path.read_text())
    assert payload["summary"]["status"] == "STOPPED"
    diary_info = payload["diary"]
    assert diary_info["path"] == str(diary_path.resolve())
    assert diary_info["exists"] is True
    assert diary_path.exists()
    assert diary_path.stat().st_size > 0
