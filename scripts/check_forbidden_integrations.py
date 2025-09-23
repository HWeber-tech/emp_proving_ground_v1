#!/usr/bin/env python3
"""Scan the repository for forbidden integration references."""
from __future__ import annotations

import argparse
import pathlib
import re
import sys
from dataclasses import dataclass
from typing import Iterable, Iterator, Sequence

# Patterns that indicate the presence of the deprecated cTrader/OpenAPI stack.
FORBIDDEN_PATTERNS = (
    re.compile(r"ctrader_open_api", re.IGNORECASE),
    re.compile(r"swagger", re.IGNORECASE),
    re.compile(r"spotware", re.IGNORECASE),
    re.compile(r"real_ctrader_interface", re.IGNORECASE),
    re.compile(r"\bfrom\s+fastapi\b"),
    re.compile(r"\bimport\s+fastapi\b"),
    re.compile(r"\bimport\s+uvicorn\b"),
)

DEFAULT_TARGETS = (
    "src",
    "tests/current",
)

SKIP_DIR_NAMES = {
    ".git",
    ".hg",
    ".svn",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".nox",
    ".tox",
    "node_modules",
    "venv",
    ".venv",
    "dist",
    "build",
    "__pypackages__",
}

SKIP_FILE_SUFFIXES = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".bmp",
    ".ico",
    ".pdf",
    ".svg",
    ".zip",
    ".gz",
    ".bz2",
    ".xz",
    ".7z",
    ".tar",
    ".whl",
    ".jar",
    ".parquet",
}

SCANNER_BASENAME = pathlib.Path(__file__).name
ALLOWLIST_BASENAME = "check_forbidden_integrations.allowlist"


@dataclass(frozen=True)
class Match:
    path: str
    lineno: int
    line: str

    def format(self) -> str:
        return f"{self.path}:{self.lineno}:{self.line}"


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Search the repository for references to forbidden integrations.")
    parser.add_argument(
        "targets",
        nargs="*",
        help="Paths to scan. Defaults to src and tests/current if omitted.",
    )
    parser.add_argument(
        "--allowlist",
        default=None,
        help="Optional path to an allowlist file that suppresses known matches.",
    )
    return parser.parse_args(argv)


def existing_targets(root: pathlib.Path, targets: Sequence[str]) -> Iterator[pathlib.Path]:
    for raw in targets:
        path = pathlib.Path(raw)
        if not path.is_absolute():
            path = root / path
        if path.exists():
            yield path


def should_skip_file(path: pathlib.Path) -> bool:
    if path.name == SCANNER_BASENAME:
        return True
    if path.name.endswith(".pyc"):
        return True
    if any(part in SKIP_DIR_NAMES for part in path.parts):
        return True
    if path.suffix.lower() in SKIP_FILE_SUFFIXES:
        return True
    return False


def walk_files(paths: Iterable[pathlib.Path]) -> Iterator[pathlib.Path]:
    for path in paths:
        if path.is_file():
            if not should_skip_file(path):
                yield path
            continue
        if path.is_dir():
            for child in path.rglob("*"):
                if not child.is_file():
                    continue
                if should_skip_file(child):
                    continue
                yield child


def scan_file(path: pathlib.Path) -> Iterator[Match]:
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            for lineno, line in enumerate(handle, start=1):
                text = line.rstrip("\n")
                if any(pattern.search(text) for pattern in FORBIDDEN_PATTERNS):
                    yield Match(path=str(path), lineno=lineno, line=text)
    except OSError:
        return


def format_path(root: pathlib.Path, path: pathlib.Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def load_allowlist(path: pathlib.Path | None) -> set[str]:
    if path is None:
        return set()
    try:
        entries = set()
        with path.open("r", encoding="utf-8") as handle:
            for raw in handle:
                candidate = raw.split("#", 1)[0].strip()
                if candidate:
                    entries.add(candidate)
        return entries
    except OSError:
        return set()


def is_allowed(match: Match, allowlist: set[str]) -> bool:
    if not allowlist:
        return False
    relative = match.path
    keys = {
        match.format(),
        f"{relative}:{match.lineno}",
        relative,
    }
    return any(key in allowlist for key in keys)


def main(argv: Sequence[str]) -> int:
    repo_root = pathlib.Path(__file__).resolve().parent.parent
    args = parse_args(argv)

    allowlist_path: pathlib.Path | None
    if args.allowlist is not None:
        allowlist_path = pathlib.Path(args.allowlist)
    else:
        allowlist_path = pathlib.Path(__file__).with_name(ALLOWLIST_BASENAME)

    targets = list(args.targets) if args.targets else list(DEFAULT_TARGETS)
    valid_targets = list(existing_targets(repo_root, targets))
    if not valid_targets:
        print("No scan targets exist; skipping forbidden integration check.")
        return 0

    allowlist = load_allowlist(allowlist_path)

    pretty_targets = " ".join(format_path(repo_root, t) for t in valid_targets)
    print(f"Scanning {pretty_targets} for forbidden integrations...")

    matches: list[Match] = []
    for file_path in walk_files(valid_targets):
        relative = format_path(repo_root, file_path)
        for match in scan_file(file_path):
            # Update match path to use repository-relative formatting for reporting
            match = Match(path=relative, lineno=match.lineno, line=match.line)
            if is_allowed(match, allowlist):
                continue
            matches.append(match)

    if matches:
        print("Forbidden references detected:", file=sys.stderr)
        for entry in matches:
            print(entry.format(), file=sys.stderr)
        return 1

    print("No forbidden references found.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
