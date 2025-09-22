#!/usr/bin/env python3
"""
Capture a full-repo mypy snapshot and a summary, timestamped, and write them to:
- ./mypy_snapshots/
- ./docs/development/mypy_snapshots/ (if this directory exists)

This script assumes:
- mypy is installed and configured via mypy.ini
- repository root is the current working directory
"""

from __future__ import annotations

import datetime as dt
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
OUT_DIR_A = REPO_ROOT / "mypy_snapshots"
OUT_DIR_B = REPO_ROOT / "docs" / "development" / "mypy_snapshots"
SUMMARY_SCRIPT = REPO_ROOT / "scripts" / "analysis" / "mypy_summary.py"
MYPY_INI = REPO_ROOT / "mypy.ini"


def run(cmd: list[str]) -> int:
    return subprocess.call(cmd)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def main() -> int:
    ts = dt.datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")
    ensure_dir(OUT_DIR_A)
    if OUT_DIR_B.parent.parent.exists():
        ensure_dir(OUT_DIR_B)

    snapshot_name = f"mypy_snapshot_{ts}.txt"
    summary_name = f"mypy_summary_{ts}.txt"

    snap_a = OUT_DIR_A / snapshot_name
    sum_a = OUT_DIR_A / summary_name
    snap_b = OUT_DIR_B / snapshot_name
    sum_b = OUT_DIR_B / summary_name

    # Run mypy snapshot (non-gating)
    print(f"[mypy-capture] Running mypy over {SRC_DIR} ...")
    with snap_a.open("w", encoding="utf-8") as fout:
        # Allow non-zero exit (errors present) without failing the script
        rc = subprocess.call(
            ["mypy", "--config-file", str(MYPY_INI), str(SRC_DIR)],
            stdout=fout,
            stderr=subprocess.STDOUT,
        )
        if rc != 0:
            print(f"[mypy-capture] mypy exited with {rc} (expected if errors exist)")

    # Copy snapshot to docs directory if present
    if OUT_DIR_B.exists():
        snap_b.write_text(snap_a.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"[mypy-capture] Snapshot also written to {snap_b.as_posix()}")

    # Generate summary using the companion script
    if not SUMMARY_SCRIPT.exists():
        print(f"[mypy-capture] Summary script not found: {SUMMARY_SCRIPT}", file=sys.stderr)
        return 0

    print(f"[mypy-capture] Generating summary for {snap_a.name} ...")
    rc2 = run(
        [
            sys.executable,
            str(SUMMARY_SCRIPT),
            "--snapshot",
            str(snap_a),
            "--summary-out",
            str(sum_a),
        ]
    )
    if rc2 != 0:
        print(f"[mypy-capture] Summary generation returned {rc2}", file=sys.stderr)

    if OUT_DIR_B.exists():
        sum_b.write_text(sum_a.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"[mypy-capture] Summary also written to {sum_b.as_posix()}")

    print("[mypy-capture] Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
