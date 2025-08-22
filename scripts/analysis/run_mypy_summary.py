#!/usr/bin/env python3
"""
Reporter: Runs mypy and generates machine-readable artifacts and a concise console summary.

- Invokes: ./.venv_mypy/bin/mypy src --config-file mypy.ini --show-error-codes --no-color-output
- Writes:
  * scripts/analysis/out/mypy_report.json
  * scripts/analysis/out/mypy_report.csv  (columns: file,line,code,message)
  * scripts/analysis/out/last_mypy.txt    (raw mypy output; mirrors mypy_quick.sh behavior)
- Prints console summary:
  * total errors
  * top 10 error codes
  * top 15 files by error count
  * earliest 20 errors (stable order: as emitted by mypy)
- Always exits with code 0 (reporting tool).
"""

from __future__ import annotations

import csv
import json
import os
import re
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "scripts" / "analysis" / "out"
MYPY_BIN = ROOT / ".venv_mypy" / "bin" / "mypy"
CONFIG_FILE = ROOT / "mypy.ini"
SRC_DIR = ROOT / "src"

# mypy default text output line format (strict-ish):
# file:line: [column: ]? error|note: message [error-code]
# examples:
# src/pkg/mod.py:123: error: Something bad [misc]
# src/pkg/mod.py:123:45: error: Something bad [misc]
# src/pkg/mod.py:124: note: Revealed type is "builtins.int"
LINE_RE = re.compile(
    r"""
    ^
    (?P<file>[^\n:]+)
    :
    (?P<line>\d+)
    (?:
        :
        (?P<column>\d+)
    )?
    :
    \s*
    (?P<level>error|note|warning)
    :
    \s*
    (?P<message>.*?)
    (?:\s\[(?P<code>[a-zA-Z0-9\-_]+)\])?
    $
    """,
    re.VERBOSE,
)


@dataclass
class MypyItem:
    file: str
    line: int
    column: int | None
    level: str  # "error" | "note" | "warning"
    message: str
    code: str | None

    def is_error(self) -> bool:
        return self.level == "error"


def run_mypy() -> Tuple[int, List[str]]:
    cmd = [
        str(MYPY_BIN),
        str(SRC_DIR),
        "--config-file",
        str(CONFIG_FILE),
        "--show-error-codes",
        "--no-color-output",
    ]
    env = os.environ.copy()
    # Respect MYPYPATH only for stubs to avoid dual module identities
    candidate_paths = []
    stubs_dir = ROOT / "stubs"
    if stubs_dir.exists():
        candidate_paths.append(str(stubs_dir))
    env["MYPYPATH"] = os.pathsep.join(candidate_paths + ([env["MYPYPATH"]] if "MYPYPATH" in env and env["MYPYPATH"] else []))

    try:
        proc = subprocess.run(
            cmd,
            cwd=str(ROOT),
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
    except FileNotFoundError:
        # Provide actionable guidance if mypy venv isn't present.
        guidance = (
            f"Unable to execute {MYPY_BIN}.\n"
            "Ensure the mypy virtualenv exists and mypy is installed:\n"
            f"  python -m venv {ROOT/'.venv_mypy'}\n"
            f"  {ROOT/'.venv_mypy/bin/pip'} install mypy\n"
        )
        print(guidance, file=sys.stderr)
        return 127, []

    stdout_lines = proc.stdout.splitlines()
    stderr_lines = proc.stderr.splitlines()
    all_lines = stdout_lines + stderr_lines  # mypy may print to both

    # Persist raw output for snapshotting
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "last_mypy.txt").write_text("\n".join(all_lines) + "\n", encoding="utf-8")

    return proc.returncode, all_lines


def parse_mypy_lines(lines: Iterable[str]) -> List[MypyItem]:
    items: List[MypyItem] = []
    for line in lines:
        m = LINE_RE.match(line)
        if not m:
            continue
        gd = m.groupdict()
        file = gd["file"]
        line_no = int(gd["line"])
        col = int(gd["column"]) if gd.get("column") else None
        level = gd["level"]
        message = gd["message"].strip()
        code = gd.get("code")
        items.append(MypyItem(file=file, line=line_no, column=col, level=level, message=message, code=code))
    return items


def write_csv(items: List[MypyItem], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "line", "code", "message"])
        for it in items:
            if it.level != "error":
                continue
            writer.writerow([it.file, it.line, it.code or "", it.message])


def write_json(items: List[MypyItem], path: Path) -> None:
    # Store both raw items and summaries for convenience
    serializable = {
        "items": [
            {
                "file": it.file,
                "line": it.line,
                "column": it.column,
                "level": it.level,
                "message": it.message,
                "code": it.code,
            }
            for it in items
        ],
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, sort_keys=False)


def summarize(items: List[MypyItem]) -> str:
    errors = [it for it in items if it.level == "error"]
    total_errors = len(errors)

    code_counts = Counter(it.code or "unknown" for it in errors)
    top_codes = code_counts.most_common(10)

    file_counts: dict[str, int] = defaultdict(int)
    for it in errors:
        file_counts[it.file] += 1
    top_files = sorted(file_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:15]

    earliest_20 = errors[:20]

    lines: List[str] = []
    lines.append("MyPy Summary")
    lines.append("============")
    lines.append(f"Total errors: {total_errors}")
    lines.append("")
    lines.append("Top 10 error codes:")
    for code, count in top_codes:
        lines.append(f"  {code:20s} {count}")
    lines.append("")
    lines.append("Top 15 files by error count:")
    for fname, count in top_files:
        lines.append(f"  {count:5d}  {fname}")
    lines.append("")
    lines.append("Earliest 20 errors:")
    for it in earliest_20:
        code = f" [{it.code}]" if it.code else ""
        col = f":{it.column}" if it.column is not None else ""
        lines.append(f"  {it.file}:{it.line}{col}: {it.message}{code}")
    return "\n".join(lines)


def main() -> int:
    # Ensure output directory
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    retcode, lines = run_mypy()
    items = parse_mypy_lines(lines)

    # Write artifacts
    write_json(items, OUT_DIR / "mypy_report.json")
    write_csv(items, OUT_DIR / "mypy_report.csv")

    # Print summary
    print(summarize(items))

    # Always succeed; this is a reporter.
    return 0


if __name__ == "__main__":
    sys.exit(main())