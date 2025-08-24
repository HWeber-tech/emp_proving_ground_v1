#!/usr/bin/env python3
"""
Summarize a mypy snapshot produced with show_error_codes=True.

Inputs:
  --snapshot PATH         Path to snapshot text file (stdout of mypy run)
  --summary-out PATH      Where to write a human-readable summary

Output:
  Writes a text summary including:
    - Timestamp and source snapshot path
    - Total error count
    - Top error codes by frequency
    - Errors by top-level package (derived from src/<pkg>/...)
    - Optional: errors by module prefix (first two path components)

Notes:
  - We only count lines that contain ': error:' to avoid counting 'note' lines.
  - Error code is extracted from trailing [code] at end of line.
  - Package is derived from file path under src/; otherwise 'external/unknown'.
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path


ERROR_LINE_RE = re.compile(r":\s*error:\s")
CODE_RE = re.compile(r"\[([A-Za-z0-9_-]+)\]\s*$")


def derive_package(p: Path) -> str:
    """
    Derive top-level package from a file path.
    Examples:
      src/sensory/utils/foo.py -> sensory
      src/core/performance/x/y.py -> core
      tools/x.py -> external/unknown
    """
    parts = p.as_posix().split("/")
    try:
        idx = parts.index("src")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    except ValueError:
        pass
    return "external/unknown"


def derive_module_prefix(p: Path) -> str:
    """
    Derive a module prefix (first two components under src) for coarser grouping.
    Examples:
      src/sensory/utils/foo.py -> sensory.utils
      src/core/performance/x/y.py -> core.performance
    """
    parts = p.as_posix().split("/")
    try:
        idx = parts.index("src")
        after = parts[idx + 1 :]
        if len(after) >= 2:
            return ".".join(after[:2])
        if len(after) == 1:
            return after[0]
    except ValueError:
        pass
    return "external/unknown"


def parse_snapshot(snapshot_path: Path):
    total_errors = 0
    codes = Counter()
    by_package = Counter()
    by_prefix = Counter()

    with snapshot_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if not ERROR_LINE_RE.search(line):
                continue

            total_errors += 1

            mcode = CODE_RE.search(line)
            code = mcode.group(1) if mcode else "unknown"

            # Attempt to parse the file path (prefix up to first ':')
            file_part = line.split(":", 1)[0].strip()
            path = Path(file_part)

            pkg = derive_package(path)
            pref = derive_module_prefix(path)

            codes[code] += 1
            by_package[pkg] += 1
            by_prefix[pref] += 1

    return total_errors, codes, by_package, by_prefix


def write_summary(
    summary_path: Path,
    snapshot_path: Path,
    total_errors: int,
    codes: Counter,
    by_package: Counter,
    by_prefix: Counter,
    top_n: int = 15,
):
    ts = dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    lines = []
    lines.append(f"Mypy Summary")
    lines.append(f"Generated: {ts} UTC")
    lines.append(f"Snapshot: {snapshot_path.as_posix()}")
    lines.append("")
    lines.append(f"Total errors: {total_errors}")
    lines.append("")

    def fmt_counter(title: str, counter: Counter, n: int):
        lines.append(title)
        if not counter:
            lines.append("  (none)")
            return
        for item, count in counter.most_common(n):
            lines.append(f"  {item:30s} {count}")
        lines.append("")

    fmt_counter(f"Top {top_n} error codes:", codes, top_n)
    fmt_counter("Errors by top-level package:", by_package, len(by_package))
    fmt_counter("Errors by module prefix:", by_prefix, min(len(by_prefix), 30))

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", required=True, help="Path to mypy snapshot file")
    parser.add_argument(
        "--summary-out", required=True, help="Path to write summary text file"
    )
    args = parser.parse_args(argv)

    snapshot_path = Path(args.snapshot)
    summary_path = Path(args.summary_out)

    if not snapshot_path.exists():
        print(f"Snapshot not found: {snapshot_path}", file=sys.stderr)
        return 2

    totals = parse_snapshot(snapshot_path)
    write_summary(summary_path, snapshot_path, *totals)
    print(f"Wrote summary to {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())