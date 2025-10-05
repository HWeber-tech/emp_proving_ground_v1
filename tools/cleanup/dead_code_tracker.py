"""Utilities for triaging the roadmap dead-code backlog."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

__all__ = [
    "DeadCodeSummary",
    "parse_cleanup_report",
    "summarise_candidates",
    "main",
]


_REPORT_PATTERN = re.compile(r"src\\[\\\\\w./]+")


@dataclass(frozen=True)
class DeadCodeSummary:
    """Structured view of the dead-code candidates declared in the cleanup report."""

    total_candidates: int
    present: tuple[str, ...]
    missing: tuple[str, ...]
    shim_exports: tuple[str, ...]

    def as_dict(self) -> Mapping[str, object]:
        return {
            "total_candidates": self.total_candidates,
            "present": list(self.present),
            "missing": list(self.missing),
            "shim_exports": list(self.shim_exports),
        }

    def to_json(self) -> str:
        return json.dumps(self.as_dict(), indent=2, sort_keys=True)


def parse_cleanup_report(report_path: Path) -> list[str]:
    """Return normalised candidate paths from ``CLEANUP_REPORT.md``."""

    text = report_path.read_text(encoding="utf-8")
    matches = _REPORT_PATTERN.findall(text)
    candidates: list[str] = []
    for match in matches:
        normalised = match.replace("\\", "/")
        normalised = normalised.strip("~ ")
        if normalised.endswith("."):
            normalised = normalised[:-1]
        if normalised not in candidates:
            candidates.append(normalised)
    return candidates


def summarise_candidates(candidates: Sequence[str], *, repo_root: Path) -> DeadCodeSummary:
    """Classify candidate paths and highlight shim exports for triage."""

    present: list[str] = []
    missing: list[str] = []
    shim_exports: list[str] = []

    for candidate in sorted(dict.fromkeys(candidates)):
        path = repo_root / candidate
        if path.exists():
            present.append(candidate)
            if path.is_file() and path.suffix == ".py" and _looks_like_shim(path):
                shim_exports.append(candidate)
        else:
            missing.append(candidate)

    return DeadCodeSummary(
        total_candidates=len(candidates),
        present=tuple(present),
        missing=tuple(missing),
        shim_exports=tuple(shim_exports),
    )


def _looks_like_shim(path: Path) -> bool:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return False
    lower = text.lower()
    if "shim" in lower:
        return True
    if "__getattr__" in text and "importlib" in text:
        return True
    return False


def _format_summary(summary: DeadCodeSummary) -> str:
    lines = [
        f"Total candidates: {summary.total_candidates}",
        f"Present: {len(summary.present)}",
        f"Missing: {len(summary.missing)}",
        f"Shim exports: {len(summary.shim_exports)}",
    ]
    if summary.shim_exports:
        lines.append("Shim modules:")
        lines.extend(f"  - {path}" for path in summary.shim_exports)
    if summary.missing:
        lines.append("Missing modules:")
        lines.extend(f"  - {path}" for path in summary.missing)
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--format",
        choices={"text", "json"},
        default="text",
        help="Output format (text or json).",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("docs/reports/CLEANUP_REPORT.md"),
        help="Path to the cleanup report markdown file.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Repository root containing the candidate modules.",
    )
    args = parser.parse_args(argv)

    candidates = parse_cleanup_report(args.report)
    summary = summarise_candidates(candidates, repo_root=args.root)

    if args.format == "json":
        print(summary.to_json())
    else:
        print(_format_summary(summary))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
