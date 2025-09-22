#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Automated audit and trimming of stub units under stubs/src.

Workflow (implements Steps 1–4 from the task):
1) Baseline snapshot
2) Inventory stubs
3) Automated trial removals (one unit at a time)
4) Finalize with post snapshot and ranked offenders

Constraints:
- Only modifies under stubs/src and stubs/.quarantine
- Writes artifacts under mypy_snapshots
- Uses repo mypy.ini
- Idempotent: restores any already-quarantined units before starting trials

Notes:
- We do not open PRs or touch CI/docs here
"""

from __future__ import annotations

import csv
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

# Patterns to parse mypy summary lines
RE_FOUND_ERRORS = re.compile(r"Found (?P<errors>\\d+) errors? in (?P<files>\\d+) files?")
RE_SUCCESS = re.compile(r"Success: no issues found in (?P<files>\\d+) source files")

# Repository root
REPO_ROOT = Path(__file__).resolve().parents[2]
STUBS_SRC = REPO_ROOT / "stubs" / "src"
QUARANTINE = REPO_ROOT / "stubs" / ".quarantine"
MYPY_SNAPSHOTS = REPO_ROOT / "mypy_snapshots"
MYPY_INI = REPO_ROOT / "mypy.ini"


# Dataclass for units
@dataclass
class Unit:
    kind: str  # 'dir' or 'file'
    name: str  # top-level name for dirs, or filename for files at root
    src_path: Path  # current location under stubs/src or file path
    rel_id: str  # identifier used in artifacts, e.g., 'core' or 'example.pyi'


# Timestamp function
def ts_utc() -> str:
    # Example: 2025-08-27T14-41-55Z (no colons for filesystem safety)
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")


# Ensure directories function
def ensure_dirs() -> None:
    MYPY_SNAPSHOTS.mkdir(parents=True, exist_ok=True)
    QUARANTINE.mkdir(parents=True, exist_ok=True)


# Shell quote function
def sh_quote(p: Path | str) -> str:
    s = str(p)
    if re.search(r"[^\\w@%+=:,./-]", s):
        return "'" + s.replace("'", "'\\\\''") + "'"
    return s


# Run mypy and capture function
def run_mypy_and_capture(snapshot_txt: Path) -> Tuple[int, str]:
    """
    Run mypy with --config-file mypy.ini, capture combined stdout+stderr,
    write to snapshot_txt, return (exit_code, combined_output).
    """
    if not MYPY_INI.exists():
        raise FileNotFoundError(f"mypy.ini not found at {MYPY_INI}")

    # Use bash -lc to honor set -o pipefail semantics if we pipe in future
    cmd = ["bash", "-lc", f"mypy --config-file {sh_quote(MYPY_INI)}"]
    proc = subprocess.run(cmd, cwd=REPO_ROOT, text=True, capture_output=True)
    combined = (proc.stdout or "") + (proc.stderr or "")
    snapshot_txt.write_text(combined, encoding="utf-8")
    return proc.returncode, combined


# Parse mypy summary function
def parse_mypy_summary(text: str) -> Tuple[Optional[int], str]:
    """
    Try to parse mypy final error count N from output.
    Return (N, summary_line). If N cannot be parsed, return (None, fallback).
    """
    n: Optional[int] = None
    summary_line = "mypy completed"
    lines = text.strip().splitlines()
    for line in reversed(lines[-100:]):  # search tail for summary
        m1 = RE_FOUND_ERRORS.search(line)
        if m1:
            n = int(m1.group("errors"))
            summary_line = line.strip()
            break
        m2 = RE_SUCCESS.search(line)
        if m2:
            n = 0
            summary_line = line.strip()
            break
    if n is None:
        summary_line = "mypy completed, exit_code=unknown"
    return n, summary_line


# Write summary function
def write_summary(
    summary_path: Path, exit_code: int, summary_line: str, parsed_errors: Optional[int]
) -> None:
    """
    Write small summary file.
    """
    if parsed_errors is not None:
        text = f"{summary_line}\\nexit_code={exit_code}\\nparsed_errors={parsed_errors}\\n"
    else:
        text = f"{summary_line}\\nexit_code={exit_code}\\nparsed_errors=unparsed\\n"
    summary_path.write_text(text, encoding="utf-8")


# Discover units function
def discover_units() -> List[Unit]:
    """
    Detect units:
    - Each top-level directory under stubs/src is a unit.
    - Each .pyi file directly under stubs/src is a unit.
    """
    if not STUBS_SRC.exists():
        return []

    units: List[Unit] = []

    for child in sorted(STUBS_SRC.iterdir()):
        if child.is_dir():
            units.append(Unit(kind="dir", name=child.name, src_path=child, rel_id=child.name))
        elif child.is_file() and child.suffix == ".pyi":
            units.append(Unit(kind="file", name=child.name, src_path=child, rel_id=child.name))

    return units


# Count files in path function
def count_files_in_path(path: Path) -> int:
    if path.is_file():
        return 1
    total = 0
    for fp in path.rglob("*"):
        if fp.is_file():
            total += 1
    return total


# Write inventory function
def write_inventory(units: List[Unit], ts: str) -> None:
    inv_txt = MYPY_SNAPSHOTS / f"stubs_inventory_{ts}.txt"
    inv_csv = MYPY_SNAPSHOTS / f"stubs_inventory_{ts}.csv"
    with (
        inv_txt.open("w", encoding="utf-8") as txt,
        inv_csv.open("w", encoding="utf-8", newline="") as csvf,
    ):
        writer = csv.writer(csvf)
        writer.writerow(["unit_path", "files", "count"])
        for u in units:
            cnt = count_files_in_path(u.src_path)
            unit_path = f"{STUBS_SRC.relative_to(REPO_ROOT)}/{u.rel_id}"
            txt.write(f"{unit_path}\\tfiles={cnt}\\n")
            writer.writerow([unit_path, "files", cnt])


# Restore if quarantined function
def restore_if_quarantined(units: List[Unit]) -> None:
    """
    If quarantine already contains a unit (idempotency), restore it to stubs/src before trials.
    """
    for u in units:
        q_src = QUARANTINE / u.rel_id
        src_dest = STUBS_SRC / u.rel_id
        if u.kind == "dir":
            if q_src.is_dir() and not src_dest.exists():
                src_dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(q_src), str(src_dest))
        else:
            q_file = QUARANTINE / u.rel_id
            if q_file.is_file() and not src_dest.exists():
                src_dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(q_file), str(src_dest))


# Move unit to quarantine function
def move_unit_to_quarantine(u: Unit) -> None:
    if u.kind == "dir":
        src = STUBS_SRC / u.rel_id
        dst = QUARANTINE / u.rel_id
        if dst.exists():
            if dst.is_dir():
                shutil.rmtree(dst)
            else:
                dst.unlink()
        if src.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))
    else:
        src = STUBS_SRC / u.rel_id
        dst = QUARANTINE / u.rel_id
        if dst.exists():
            if dst.is_dir():
                shutil.rmtree(dst)
            else:
                dst.unlink()
        if src.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))


# Restore unit from quarantine function
def restore_unit_from_quarantine(u: Unit) -> None:
    src = QUARANTINE / u.rel_id
    dst = STUBS_SRC / u.rel_id
    if src.exists() and not dst.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))


# Ranked offenders CSV from output function
def ranked_offenders_csv_from_output(text: str, path: Path) -> None:
    """
    Best-effort: count errors by file path in mypy output.
    Typical lines: path:line: col: error: message [code]
    """
    counts: dict[str, int] = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        first = line.split(":", 1)[0]
        # Normalize relative display; only consider paths under repo
        candidate = (REPO_ROOT / first).resolve()
        try:
            candidate.relative_to(REPO_ROOT)
        except Exception:
            continue
        if "error:" in line:
            counts[first] = counts.get(first, 0) + 1
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "count"])
        for p, c in sorted(counts.items(), key=lambda kv: kv[1], reverse=True):
            w.writerow([p, c])


# Append line function
def append_line(path: Path, line: str) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(line + "\\n")


# Sanitize unit function
def sanitize_unit(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s)


# Main function
def main() -> int:
    ensure_dirs()
    ts = ts_utc()

    # Step 1: Baseline
    baseline_txt = MYPY_SNAPSHOTS / f"stubs_audit_baseline_{ts}.txt"
    exit_code, out = run_mypy_and_capture(baseline_txt)
    n0, baseline_summary_line = parse_mypy_summary(out)
    baseline_summary = MYPY_SNAPSHOTS / f"stubs_audit_baseline_summary_{ts}.txt"
    write_summary(baseline_summary, exit_code, baseline_summary_line, n0)
    if n0 is None:
        print(
            "Warning: Could not parse baseline error count; using exit code as proxy.",
            file=sys.stderr,
        )
        n0 = exit_code if exit_code >= 0 else 0

    # Step 2: Inventory
    units = discover_units()
    write_inventory(units, ts)

    # Idempotency: restore any quarantined units
    restore_if_quarantined(units)

    decisions_csv = MYPY_SNAPSHOTS / f"stubs_audit_decisions_{ts}.csv"
    removed_txt = MYPY_SNAPSHOTS / f"stubs_removed_{ts}.txt"
    kept_txt = MYPY_SNAPSHOTS / f"stubs_kept_{ts}.txt"
    with decisions_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            ["unit", "action", "baseline_errors", "trial_errors", "delta", "trial_artifact_path"]
        )

    removed_count = 0
    kept_count = 0

    # Step 3: Automated trials
    for u in units:
        # Move unit to quarantine
        move_unit_to_quarantine(u)

        # Trial mypy
        trial_artifact = MYPY_SNAPSHOTS / f"stubs_trial_{sanitize_unit(u.rel_id)}_{ts}.txt"
        exit_code_t, out_t = run_mypy_and_capture(trial_artifact)
        n1, trial_summary_line = parse_mypy_summary(out_t)
        if n1 is None:
            n1 = exit_code_t if exit_code_t >= 0 else 0
        delta = n1 - (n0 or 0)

        # Decision
        if delta <= 0:
            append_line(removed_txt, u.rel_id)
            action = "removed"
            removed_count += 1
        else:
            restore_unit_from_quarantine(u)
            append_line(kept_txt, u.rel_id)
            action = "kept"
            kept_count += 1

        # Record decision row
        with decisions_csv.open("a", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(
                [u.rel_id, action, n0, n1, delta, str(trial_artifact.relative_to(REPO_ROOT))]
            )

    # Step 4: Finalize
    post_txt = MYPY_SNAPSHOTS / f"stubs_audit_post_{ts}.txt"
    exit_code_post, out_post = run_mypy_and_capture(post_txt)
    n_post, post_summary_line = parse_mypy_summary(out_post)
    post_summary = MYPY_SNAPSHOTS / f"stubs_audit_post_summary_{ts}.txt"
    write_summary(post_summary, exit_code_post, post_summary_line, n_post)

    ranked_csv = MYPY_SNAPSHOTS / f"stubs_audit_ranked_offenders_{ts}.csv"
    ranked_offenders_csv_from_output(out_post, ranked_csv)

    # Final console summary
    total_units = len(units)
    delta_total = (n_post or 0) - (n0 or 0)
    print(
        f"Units evaluated={total_units}, removed={removed_count}, kept={kept_count}, baseline_errors={n0}, final_errors={n_post}, delta_total={delta_total}"
    )
    return 0


# Main execution
if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        sys.exit(130)
