#!/usr/bin/env python3
"""
Basic repository status metrics for transparency.
- Counts files with 'pass'
- Counts 'NotImplementedError'
- Reports test coverage if .coverage file exists (placeholder)
"""

import os
import re


def count_pass_statements(root: str = ".") -> int:
    total = 0
    for base, _, files in os.walk(root):
        if ".venv" in base or "node_modules" in base or base.startswith(".git"):
            continue
        for f in files:
            if f.endswith(".py"):
                try:
                    with open(os.path.join(base, f), "r", encoding="utf-8", errors="ignore") as fh:
                        total += sum(1 for line in fh if re.match(r"^\s*pass\b", line))
                except Exception:
                    continue
    return total


def count_not_implemented(root: str = ".") -> int:
    total = 0
    pattern = re.compile(r"NotImplementedError")
    for base, _, files in os.walk(root):
        if ".venv" in base or "node_modules" in base or base.startswith(".git"):
            continue
        for f in files:
            if f.endswith(".py"):
                try:
                    with open(os.path.join(base, f), "r", encoding="utf-8", errors="ignore") as fh:
                        for line in fh:
                            if pattern.search(line):
                                total += 1
                except Exception:
                    continue
    return total


def main() -> int:
    p = count_pass_statements()
    n = count_not_implemented()
    print("Repository Status Metrics")
    print("- pass statements:", p)
    print("- NotImplementedError occurrences:", n)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


