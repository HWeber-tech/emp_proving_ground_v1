#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Remove unused 'type: ignore' comments safely and generate artifacts.

Workflow:
- Resolve tools (mypy, ruff, black, isort) from local venvs or PATH
- Create UTC timestamp and mypy_snapshots dir
- Run pre mypy --warn-unused-ignores -> save .txt
- Parse unused ignore lines -> .csv and changed-files list
- For each file/line: remove only the 'type: ignore[...]' token; run mypy on file; revert if errors
- Run post mypy --warn-unused-ignores -> save .txt/.csv
- Format only edited files via ruff/black/isort
- Run final mypy (standard) -> save snapshot and summary
- Produce edits CSV with path,line_before,line_after,status (removed|kept|reverted)

Notes:
- Behavior-preserving: only remove the token and immediate extra whitespace, keep other comment text.
- Limit edits to lines flagged by mypy unused-ignores.
"""

from __future__ import annotations

import csv
import os
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

# ... existing code ...
VENV_DIRS = [
    ".venv_mypy/bin",
    ".venv_tools/bin",
    ".venv_ci/bin",
    ".venv311/bin",
    ".venv_cleanup/bin",
    ".venv_types_tmp/bin",
    ".venv_importlinter/bin",
]

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SNAP_DIR = PROJECT_ROOT / "mypy_snapshots"
MYPY_CONFIG = str(PROJECT_ROOT / "mypy.ini")

# Regex to capture mypy unused-ignores messages from pre/post runs
UNUSED_IGNORE_RE = re.compile(
    r'^(?P<path>[^:\n]+):(?P<line>\d+):.*error:\s*(?P<msg>Unused ["\']type: ignore["\'] comment.*)$',
    re.IGNORECASE,
)

# Regex to remove the token "# type: ignore" optionally with bracketed codes
IGNORE_TOKEN_RE = re.compile(r"(?:\s*#\s*type:\s*ignore(?:\[[^\]]*\])?)")


# ... existing code ...
def find_cmd(name: str) -> str:
    for d in VENV_DIRS:
        p = PROJECT_ROOT / d / name
        if p.exists() and os.access(p, os.X_OK):
            return str(p)
    # Fallback to PATH
    from shutil import which

    found = which(name)
    if not found:
        raise FileNotFoundError(f"Required tool not found: {name}")
    return found


# ... existing code ...
def run_cmd(
    cmd: List[str], cwd: Path | None = None, allow_error: bool = False
) -> subprocess.CompletedProcess:
    proc = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if not allow_error and proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {shlex.join(cmd)}\n{proc.stdout}")
    return proc


# ... existing code ...
def timestamp_utc() -> str:
    # Use Python to generate the same TS as `date -u +'%Y-%m-%dT%H-%M-%SZ'`
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")


@dataclass
class Artifacts:
    ts: str
    pre_txt: Path
    pre_csv: Path
    post_txt: Path
    post_csv: Path
    final_txt: Path
    final_sum: Path
    changed_list: Path
    edits_csv: Path


# ... existing code ...
def build_artifacts(ts: str) -> Artifacts:
    SNAP_DIR.mkdir(parents=True, exist_ok=True)
    return Artifacts(
        ts=ts,
        pre_txt=SNAP_DIR / f"mypy_unused_ignores_{ts}.txt",
        pre_csv=SNAP_DIR / f"mypy_unused_ignores_{ts}.csv",
        post_txt=SNAP_DIR / f"mypy_unused_ignores_postfix_{ts}.txt",
        post_csv=SNAP_DIR / f"mypy_unused_ignores_postfix_{ts}.csv",
        final_txt=SNAP_DIR / f"mypy_snapshot_{ts}.txt",
        final_sum=SNAP_DIR / f"mypy_summary_{ts}.txt",
        changed_list=PROJECT_ROOT / f"changed_files_unused_ignores_{ts}.txt",
        edits_csv=SNAP_DIR / f"unused_ignores_edits_{ts}.csv",
    )


# ... existing code ...
def write_csv_header(path: Path, header: Iterable[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(list(header))


# ... existing code ...
def append_edits_rows(edits_csv: Path, rows: List[Tuple[str, str, str, str]]) -> None:
    header_needed = not edits_csv.exists() or edits_csv.stat().st_size == 0
    with edits_csv.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if header_needed:
            w.writerow(["path", "line_before", "line_after", "status"])
        for row in rows:
            w.writerow(row)


# ... existing code ...
def parse_unused_ignores(txt: str) -> List[Tuple[str, int, str]]:
    results: List[Tuple[str, int, str]] = []
    for line in txt.splitlines():
        m = UNUSED_IGNORE_RE.match(line.strip())
        if m:
            p = str(Path(m.group("path")))
            ln = int(m.group("line"))
            msg = m.group("msg")
            results.append((p, ln, msg))
    return results


# ... existing code ...
def group_by_file(lines: List[Tuple[str, int, str]]) -> Dict[str, List[int]]:
    files: Dict[str, List[int]] = {}
    for path, ln, _ in lines:
        files.setdefault(path, []).append(ln)
    for p in list(files):
        files[p] = sorted(set(files[p]))
    return files


# ... existing code ...
def remove_ignore_token_from_line(line: str) -> Tuple[str, bool]:
    m = IGNORE_TOKEN_RE.search(line)
    if not m:
        return line, False
    start, end = m.span()
    before = line[:start]
    after = line[end:]
    new_line = before.rstrip()
    if after and not after.startswith(("#", " ", "\t")):
        new_line += " "
    new_line = new_line + after
    # Collapse excessive spaces before a comment
    new_line = re.sub(r"\s+#", " #", new_line)
    return new_line, True


# ... existing code ...
def run_mypy_on_file(mypy_bin: str, path: Path) -> int:
    proc = run_cmd([mypy_bin, "--config-file", MYPY_CONFIG, str(path)], allow_error=True)
    return proc.returncode


# ... existing code ...
def process_file(
    mypy_bin: str,
    file_path: Path,
    lines_to_edit: List[int],
    edits_csv: Path,
) -> bool:
    """
    Returns True if the file content was changed (any removal kept), False otherwise.
    """
    if not file_path.exists():
        return False
    content = file_path.read_text(encoding="utf-8")
    lines = content.splitlines(keepends=True)
    original_lines = lines[:]
    changed_any = False
    edits_rows: List[Tuple[str, str, str, str]] = []

    for ln in lines_to_edit:
        if ln < 1 or ln > len(lines):
            before = original_lines[ln - 1].rstrip("\n") if 1 <= ln <= len(original_lines) else ""
            edits_rows.append((str(file_path), before, before, "kept"))
            continue
        idx = ln - 1
        before = lines[idx].rstrip("\n")
        new_line, changed = remove_ignore_token_from_line(lines[idx])
        if not changed or new_line == lines[idx]:
            after = lines[idx].rstrip("\n")
            edits_rows.append((str(file_path), before, after, "kept"))
            continue

        # Tentatively apply and validate
        lines[idx] = new_line
        file_path.write_text("".join(lines), encoding="utf-8")
        rc = run_mypy_on_file(mypy_bin, file_path)
        if rc != 0:
            # revert this single line
            lines[idx] = original_lines[idx]
            file_path.write_text("".join(lines), encoding="utf-8")
            edits_rows.append((str(file_path), before, before, "reverted"))
        else:
            after = lines[idx].rstrip("\n")
            edits_rows.append((str(file_path), before, after, "removed"))
            changed_any = True

    append_edits_rows(edits_csv, edits_rows)
    return changed_any


# ... existing code ...
def main() -> int:
    # Tool resolution
    try:
        mypy_bin = find_cmd("mypy")
        ruff_bin = find_cmd("ruff")
        black_bin = find_cmd("black")
        isort_bin = find_cmd("isort")
    except FileNotFoundError as e:
        sys.stderr.write(f"{e}\nInstall dependencies from requirements/dev.txt and retry.\n")
        return 1

    ts = timestamp_utc()
    artifacts = build_artifacts(ts)

    # Step 2: pre-run mypy with --warn-unused-ignores
    pre_proc = run_cmd(
        [mypy_bin, "--config-file", MYPY_CONFIG, "--warn-unused-ignores"], allow_error=True
    )
    artifacts.pre_txt.write_text(pre_proc.stdout, encoding="utf-8")

    unused = parse_unused_ignores(pre_proc.stdout)
    # Write pre CSV
    with artifacts.pre_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "line", "message"])
        for path, line, msg in unused:
            w.writerow([path, line, msg])

    # Write unique changed files list
    changed_files: List[str] = sorted({p for p, _, _ in unused})
    artifacts.changed_list.write_text(
        "\n".join(changed_files) + ("\n" if changed_files else ""), encoding="utf-8"
    )

    # Step 3-4: apply minimal edits with per-file validation
    files_to_lines = group_by_file(unused)
    actually_touched: Set[Path] = set()
    # Ensure edits CSV has header if it will be used
    if unused:
        write_csv_header(artifacts.edits_csv, ["path", "line_before", "line_after", "status"])
    else:
        write_csv_header(artifacts.edits_csv, ["path", "line_before", "line_after", "status"])

    for file_str, line_list in files_to_lines.items():
        fp = (PROJECT_ROOT / file_str).resolve()
        if not fp.exists():
            continue
        try:
            changed = process_file(mypy_bin, fp, line_list, artifacts.edits_csv)
            if changed:
                actually_touched.add(fp)
        except Exception as e:
            # Defensive: on processing error, skip file
            sys.stderr.write(f"Skipping {fp} due to error: {e}\n")

    # Step 5: post-run mypy with --warn-unused-ignores
    post_proc = run_cmd(
        [mypy_bin, "--config-file", MYPY_CONFIG, "--warn-unused-ignores"], allow_error=True
    )
    artifacts.post_txt.write_text(post_proc.stdout, encoding="utf-8")

    post_unused = parse_unused_ignores(post_proc.stdout)
    with artifacts.post_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "line", "message"])
        for path, line, msg in post_unused:
            w.writerow([path, line, msg])

    # Format only edited files (even if reverted lines, formatting is safe but we restrict to touched set)
    if actually_touched:
        files = [str(p) for p in sorted(actually_touched)]
        try:
            run_cmd([ruff_bin, "--fix", *files], allow_error=True)
        except Exception:
            pass
        try:
            run_cmd([black_bin, *files], allow_error=True)
        except Exception:
            pass
        try:
            run_cmd([isort_bin, "--profile=black", *files], allow_error=True)
        except Exception:
            pass

    # Final mypy (standard)
    final_proc = run_cmd([mypy_bin, "--config-file", MYPY_CONFIG], allow_error=True)
    artifacts.final_txt.write_text(final_proc.stdout, encoding="utf-8")

    # Extract summary line
    summary_line = ""
    for line in final_proc.stdout.splitlines():
        if re.search(r"Found \d+ errors? in \d+ files?", line):
            summary_line = line.strip()
    artifacts.final_sum.write_text(
        (summary_line or "No mypy summary line detected.") + "\n", encoding="utf-8"
    )

    # Emit artifacts list to stdout for convenience
    print("ARTIFACTS:")
    print(str(artifacts.pre_txt))
    print(str(artifacts.pre_csv))
    print(str(artifacts.changed_list))
    print(str(artifacts.post_txt))
    print(str(artifacts.post_csv))
    print(str(artifacts.final_txt))
    print(str(artifacts.final_sum))
    print(str(artifacts.edits_csv))

    return 0


# ... existing code ...
if __name__ == "__main__":
    sys.exit(main())
