#!/usr/bin/env python3
"""Scan the repository for forbidden integration references."""
from __future__ import annotations

import argparse
import pathlib
import re
import sys
from typing import Iterable, Iterator, Sequence, Tuple

# Patterns that indicate the presence of the deprecated cTrader/OpenAPI stack.
FORBIDDEN_PATTERNS: Tuple[re.Pattern[str], ...] = (
    re.compile(r"ctrader_open_api"),
    re.compile(r"swagger"),
    re.compile(r"spotware"),
    re.compile(r"real_ctrader_interface"),
    re.compile(r"\bfrom\s+fastapi\b"),
    re.compile(r"\bimport\s+fastapi\b"),
    re.compile(r"\bimport\s+uvicorn\b"),
)

DEFAULT_TARGETS: Tuple[str, ...] = ("src", "tests/current")


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Search the repository for references to forbidden integrations.")
    parser.add_argument(
        "targets",
        nargs="*",
        help="Paths to scan. Defaults to src and tests/current if omitted.",
    )
    return parser.parse_args(argv)


def existing_targets(root: pathlib.Path, targets: Sequence[str]) -> Iterator[pathlib.Path]:
    for raw in targets:
        path = pathlib.Path(raw)
        if not path.is_absolute():
            path = root / path
        if path.exists():
            yield path


def walk_files(paths: Iterable[pathlib.Path]) -> Iterator[pathlib.Path]:
    for path in paths:
        if path.is_file():
            yield path
            continue
        if path.is_dir():
            for child in path.rglob("*"):
                if child.is_file():
                    yield child


def scan_file(path: pathlib.Path) -> Iterator[Tuple[int, str]]:
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            for lineno, line in enumerate(handle, start=1):
                if any(pattern.search(line) for pattern in FORBIDDEN_PATTERNS):
                    yield lineno, line.rstrip("\n")
    except OSError:
        return


def format_path(root: pathlib.Path, path: pathlib.Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def main(argv: Sequence[str]) -> int:
    repo_root = pathlib.Path(__file__).resolve().parent.parent
    args = parse_args(argv)
    targets = args.targets or list(DEFAULT_TARGETS)

    valid_targets = list(existing_targets(repo_root, targets))
    if not valid_targets:
        print("No scan targets exist; skipping forbidden integration check.")
        return 0

    pretty_targets = " ".join(format_path(repo_root, t) for t in valid_targets)
    print(f"Scanning {pretty_targets} for forbidden integrations...")

    matches = []
    for file_path in walk_files(valid_targets):
        relative = format_path(repo_root, file_path)
        for lineno, line in scan_file(file_path):
            matches.append(f"{relative}:{lineno}:{line}")

    if matches:
        print("Forbidden references detected:", file=sys.stderr)
        for entry in matches:
            print(entry, file=sys.stderr)
        return 1

    print("No forbidden references found.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
