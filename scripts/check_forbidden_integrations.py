#!/usr/bin/env python3
"""Scan the repository for forbidden integration references."""

from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

DEFAULT_DIRECTORIES: tuple[str, ...] = (
    "src",
    "tests/current",
    "tests",
    "scripts",
    "docs",
    "tools",
)

DEFAULT_PATTERN = re.compile(
    r"(ctrader[-_]?open[-_]?api|ctraderapi\.com|connect\.icmarkets\.com|swagger|spotware|"
    r"real_ctrader_interface|from\s+fastapi|import\s+fastapi|import\s+uvicorn)",
    re.IGNORECASE,
)

DEFAULT_ALLOWLIST: tuple[Path, ...] = (
    Path("scripts/check_forbidden_integrations.sh"),
    Path("scripts/check_forbidden_integrations.py"),
    Path("scripts/phase1_deduplication.py"),
)

DEFAULT_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".cfg",
        ".ini",
        ".ipynb",
        ".md",
        ".mdx",
        ".py",
        ".pyi",
        ".rst",
        ".sh",
        ".toml",
        ".txt",
        ".yaml",
        ".yml",
    }
)

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Match:
    path: Path
    line_number: int
    line: str


def iter_files(base: Path) -> Iterator[Path]:
    if base.is_file():
        yield base
        return
    if not base.exists():
        return
    for candidate in base.rglob("*"):
        if candidate.is_file():
            yield candidate


def normalise_allowlist(entries: Iterable[Path | str], root: Path) -> set[Path]:
    resolved: set[Path] = set()
    for entry in entries:
        if not entry:
            continue
        candidate = Path(entry)
        resolved.add((candidate if candidate.is_absolute() else (root / candidate)).resolve())
    return resolved


def gather_allowlist(root: Path) -> set[Path]:
    values = list(DEFAULT_ALLOWLIST)

    env_payload = os.environ.get("FORBIDDEN_INTEGRATION_ALLOWLIST")
    if env_payload:
        values.extend(Path(line.strip()) for line in env_payload.splitlines() if line.strip())

    legacy_env = os.environ.get("ALLOWLIST")
    if legacy_env:
        values.extend(Path(line.strip()) for line in legacy_env.splitlines() if line.strip())

    return normalise_allowlist(values, root)


def scan_paths(
    paths: Sequence[str],
    *,
    root: Path,
    pattern: re.Pattern[str] = DEFAULT_PATTERN,
    allowlist: set[Path] | None = None,
    extensions: frozenset[str] = DEFAULT_EXTENSIONS,
) -> list[Match]:
    if allowlist is None:
        allowlist = gather_allowlist(root)
    else:
        allowlist = {Path(path).resolve() for path in allowlist}
    matches: list[Match] = []

    for raw in paths:
        base = (root / raw).resolve() if not os.path.isabs(raw) else Path(raw).resolve()
        if not base.exists():
            continue
        for candidate in iter_files(base):
            if candidate.suffix.lower() not in extensions:
                continue
            if candidate.resolve() in allowlist:
                continue
            try:
                with candidate.open("r", encoding="utf-8", errors="ignore") as handle:
                    for index, line in enumerate(handle, start=1):
                        if pattern.search(line):
                            matches.append(
                                Match(
                                    path=candidate.resolve(),
                                    line_number=index,
                                    line=line.rstrip(),
                                )
                            )
            except OSError:
                continue
    return matches


def format_matches(matches: Sequence[Match], *, root: Path) -> str:
    lines: list[str] = ["Forbidden integration references detected:"]
    for match in matches:
        try:
            rel_path = match.path.relative_to(root)
        except ValueError:
            rel_path = match.path
        lines.append(f"{rel_path}:{match.line_number}:{match.line}".rstrip())
    return "\n".join(lines)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="*",
        default=DEFAULT_DIRECTORIES,
        help="Directories or files to scan. Defaults to the standard project directories.",
    )
    parser.add_argument(
        "--pattern",
        help="Override the default forbidden-integration regular expression.",
    )
    parser.add_argument(
        "--allow",
        action="append",
        default=[],
        help="Additional files to add to the allowlist (relative to the repository root).",
    )
    parser.add_argument(
        "--extension",
        action="append",
        default=[],
        help="Additional file extensions to scan (include the leading dot).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    pattern = DEFAULT_PATTERN
    if args.pattern:
        pattern = re.compile(args.pattern, re.IGNORECASE)

    allowlist = gather_allowlist(ROOT)
    if args.allow:
        allowlist |= normalise_allowlist((Path(item) for item in args.allow), ROOT)

    extensions = DEFAULT_EXTENSIONS
    if args.extension:
        extensions = frozenset({*extensions, *(ext.lower() for ext in args.extension)})

    matches = scan_paths(
        args.paths,
        root=ROOT,
        pattern=pattern,
        allowlist=allowlist,
        extensions=extensions,
    )

    if matches:
        print(format_matches(matches, root=ROOT))
        return 1

    print("No forbidden integration references found.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
