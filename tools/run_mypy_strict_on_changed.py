#!/usr/bin/env python3
"""Run mypy in strict-on-touch mode for changed Python files."""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parent.parent


def _iter_changed_files(raw: str) -> list[Path]:
    candidates: list[Path] = []
    for entry in raw.splitlines():
        entry = entry.strip()
        if not entry:
            continue
        if not entry.endswith(".py"):
            continue
        path = (REPO_ROOT / entry).resolve()
        if not path.is_file():
            continue
        try:
            path.relative_to(REPO_ROOT)
        except ValueError:
            # Outside repository scope; skip defensively.
            continue
        candidates.append(path)
    return candidates


def _run_mypy(paths: Iterable[Path]) -> None:
    path_args = [str(p.relative_to(REPO_ROOT)) for p in paths]
    cmd = [
        sys.executable,
        "-m",
        "mypy",
        "--config-file",
        "mypy.ini",
        "--check-untyped-defs",
        "--warn-unused-ignores",
        *path_args,
    ]
    print("Running:", " ".join(shlex.quote(part) for part in cmd))
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)


def main() -> int:
    raw = os.environ.get("CHANGED_PYTHON_FILES", "").strip()
    if not raw:
        print("No changed Python files detected; skipping strict mypy gate.")
        return 0

    changed_paths = _iter_changed_files(raw)
    if not changed_paths:
        print("Changed files do not include tracked Python sources; skipping strict mypy gate.")
        return 0

    _run_mypy(changed_paths)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
