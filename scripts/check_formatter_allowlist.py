#!/usr/bin/env python3
"""Validate the staged Ruff formatter rollout.

Reads the allowlist in ``config/formatter/ruff_format_allowlist.txt`` and runs
``ruff format --check`` against each listed path. The script succeeds when every
allowlisted path passes the formatter check. It exits non-zero when Ruff reports
formatting drift or when a configured path is missing.

This guard lets the team opt directories into formatter enforcement
incrementally without blocking CI on the unformatted remainder of the repo.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ALLOWLIST_PATH = Path("config/formatter/ruff_format_allowlist.txt")


def _load_allowlisted_paths() -> list[Path]:
    if not ALLOWLIST_PATH.exists():
        print(
            f"Allowlist file '{ALLOWLIST_PATH}' not found; "
            "did you remove it from the repo?",
            file=sys.stderr,
        )
        sys.exit(1)

    entries: list[Path] = []
    for raw_line in ALLOWLIST_PATH.read_text().splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        path = Path(line)
        if not path.exists():
            print(f"Configured path '{line}' does not exist", file=sys.stderr)
            sys.exit(1)
        entries.append(path)
    return entries


def _run_ruff_check(targets: list[Path]) -> int:
    """Execute ``ruff format --check`` for the given paths."""

    if not targets:
        print("No paths are allowlisted for formatter enforcement yet; skipping.")
        return 0

    failures = False
    for target in targets:
        rel_target = target.as_posix()
        print(f"Checking formatting for {rel_target}...")
        result = subprocess.run(
            ["ruff", "format", "--check", rel_target],
            check=False,
        )
        if result.returncode != 0:
            failures = True
    return 1 if failures else 0


def main() -> int:
    targets = _load_allowlisted_paths()
    return _run_ruff_check(targets)


if __name__ == "__main__":
    raise SystemExit(main())
