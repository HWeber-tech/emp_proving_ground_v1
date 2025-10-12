"""Coverage domain breakdown helper for CI telemetry."""

from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone

try:
    from datetime import UTC  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover - Python < 3.11 fallback
    UTC = timezone.utc
from pathlib import Path
from typing import Iterable, Sequence

_TRACKED_DOMAINS: tuple[str, ...] = (
    "core",
    "data_foundation",
    "sensory",
    "trading",
    "risk",
    "operations",
    "operational",
    "observability",
    "runtime",
    "evolution",
    "compliance",
    "strategies",
    "understanding",
    "governance",
    "data_integration",
    "data_sources",
    "ecosystem",
    "performance",
    "portfolio",
    "simulation",
    "orchestration",
    "validation",
    "deployment",
    "integration",
    "domain",
    "thinking",
    "testing",
    "structlog",
    "system",
    "ui",
)

_DOMAIN_PREFIXES: dict[tuple[str, ...], str] = {
    ("src", domain): domain for domain in _TRACKED_DOMAINS
}

# Legacy namespace aliases that are still part of the repository but should map
# onto canonical domains when coverage is reported.
_DOMAIN_PREFIXES[("src", "intelligence")] = "understanding"

# Namespaces that have been retired entirely.  Coverage entries pointing at
# these paths indicate drift and should be surfaced explicitly instead of being
# silently reclassified.
_DEPRECATED_PREFIXES: tuple[tuple[str, ...], ...] = (("src", "market_intelligence"),)


@dataclass
class _DomainAccumulator:
    files: set[str]
    covered: int
    missed: int


@dataclass(frozen=True)
class CoverageDomain:
    name: str
    files: int
    covered: int
    missed: int

    @property
    def total(self) -> int:
        return self.covered + self.missed

    @property
    def percent(self) -> float:
        total = self.total
        if total == 0:
            return 0.0
        return round((self.covered / total) * 100.0, 2)

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "files": self.files,
            "covered": self.covered,
            "missed": self.missed,
            "coverage_percent": self.percent,
        }


@dataclass(frozen=True)
class CoverageMatrix:
    generated_at: str
    totals: CoverageDomain
    domains: tuple[CoverageDomain, ...]
    source_files: tuple[str, ...] = ()
    deprecated_files: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, object]:
        payload = {
            "generated_at": self.generated_at,
            "totals": self.totals.as_dict(),
            "domains": [domain.as_dict() for domain in self.domains],
            "source_files": list(self.source_files),
        }
        if self.deprecated_files:
            payload["deprecated_files"] = list(self.deprecated_files)
        return payload


def _normalise_parts(filename: str) -> tuple[str, ...]:
    parts: list[str] = []
    for part in Path(filename).parts:
        if part in {"", "."}:
            continue
        if part == "..":
            if parts:
                parts.pop()
            continue
        parts.append(part)
    return tuple(parts)


def _classify_domain(parts: tuple[str, ...]) -> str:
    for prefix, domain in _DOMAIN_PREFIXES.items():
        if parts[: len(prefix)] == prefix:
            return domain
    return "other"


def _iter_class_nodes(root: ET.Element) -> Iterable[ET.Element]:
    yield from root.iter("class")


def _count_line_coverage(node: ET.Element) -> tuple[int, int]:
    covered = 0
    missed = 0
    for line in node.iter("line"):
        hits = int(line.attrib.get("hits", "0") or "0")
        if hits > 0:
            covered += 1
        else:
            missed += 1
    return covered, missed


def build_coverage_matrix(coverage_report: Path) -> CoverageMatrix:
    tree = ET.parse(coverage_report)
    root = tree.getroot()

    domain_totals: dict[str, _DomainAccumulator] = {}
    all_files: set[str] = set()
    total_covered = 0
    total_missed = 0

    deprecated_files: set[str] = set()

    for class_node in _iter_class_nodes(root):
        filename = class_node.attrib.get("filename")
        if not filename:
            continue
        covered, missed = _count_line_coverage(class_node)
        if covered == 0 and missed == 0:
            continue
        parts = _normalise_parts(filename)
        normalised_filename = "/".join(parts)
        if not normalised_filename:
            continue
        for prefix in _DEPRECATED_PREFIXES:
            if parts[: len(prefix)] == prefix:
                deprecated_files.add(normalised_filename)
                break

        domain = _classify_domain(parts)
        totals = domain_totals.setdefault(
            domain,
            _DomainAccumulator(files=set(), covered=0, missed=0),
        )
        totals.files.add(normalised_filename)
        totals.covered += covered
        totals.missed += missed
        total_covered += covered
        total_missed += missed
        all_files.add(normalised_filename)

    domains = [
        CoverageDomain(
            name=domain,
            files=len(info.files),
            covered=info.covered,
            missed=info.missed,
        )
        for domain, info in domain_totals.items()
    ]
    domains.sort(key=lambda item: item.percent)

    totals_domain = CoverageDomain(
        name="total",
        files=len(all_files),
        covered=total_covered,
        missed=total_missed,
    )

    return CoverageMatrix(
        generated_at=datetime.now(tz=UTC).isoformat(timespec="seconds"),
        totals=totals_domain,
        domains=tuple(domains),
        source_files=tuple(sorted(all_files)),
        deprecated_files=tuple(sorted(deprecated_files)),
    )


def identify_laggards(matrix: CoverageMatrix, *, threshold: float) -> tuple[CoverageDomain, ...]:
    return tuple(
        domain for domain in matrix.domains if domain.percent < threshold
    )


def render_markdown(matrix: CoverageMatrix, *, threshold: float = 80.0) -> str:
    lines: list[str] = []
    lines.append(f"Generated at {matrix.generated_at}")
    lines.append("")
    lines.append("| Domain | Files | Covered | Missed | Coverage |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for domain in matrix.domains:
        lines.append(
            "| {name} | {files} | {covered} | {missed} | {percent:.2f}% |".format(
                name=domain.name,
                files=domain.files,
                covered=domain.covered,
                missed=domain.missed,
                percent=domain.percent,
            )
        )
    lines.append(
        "| **Total** | {files} | {covered} | {missed} | {percent:.2f}% |".format(
            files=matrix.totals.files,
            covered=matrix.totals.covered,
            missed=matrix.totals.missed,
            percent=matrix.totals.percent,
        )
    )
    lines.append("")

    laggards = identify_laggards(matrix, threshold=threshold)
    if laggards:
        labels = ", ".join(
            f"{domain.name} ({domain.percent:.2f}%)" for domain in laggards
        )
        lines.append(
            f"Domains below the {threshold:.2f}% threshold: {labels}."
        )
    else:
        lines.append(
            f"All tracked domains meet the {threshold:.2f}% coverage threshold."
        )
    lines.append("")

    if matrix.deprecated_files:
        lines.append("Deprecated namespace files detected:")
        for path in matrix.deprecated_files:
            lines.append(f"- {path}")
        lines.append("")

    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarise coverage by domain for quality telemetry.",
    )
    parser.add_argument(
        "--coverage-report",
        type=Path,
        required=True,
        help="Path to a coverage XML report generated by coverage.py/pytest --cov.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path to write the summary (stdout used when omitted).",
    )
    parser.add_argument(
        "--format",
        choices=("markdown", "json"),
        default="markdown",
        help="Output format for the summary (defaults to markdown).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=80.0,
        help="Highlight domains below this coverage percentage threshold.",
    )
    parser.add_argument(
        "--fail-below-threshold",
        action="store_true",
        help=(
            "Exit with status 1 when any tracked domain falls below the provided "
            "threshold."
        ),
    )
    parser.add_argument(
        "--require-file",
        dest="require_files",
        action="append",
        default=[],
        metavar="PATH",
        help=(
            "Require that the coverage report includes the given file. Provide "
            "multiple --require-file flags to guardrail critical regression "
            "suites."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    matrix = build_coverage_matrix(args.coverage_report)
    laggards = identify_laggards(matrix, threshold=args.threshold)

    required_files = tuple(args.require_files)
    missing_required: list[str] = []
    if required_files:
        available_files = set(matrix.source_files)
        for required in required_files:
            normalised = "/".join(_normalise_parts(required))
            if normalised not in available_files:
                missing_required.append(normalised or required)

    if args.format == "json":
        payload = matrix.as_dict()
        payload["threshold"] = args.threshold
        payload["laggards"] = [domain.name for domain in laggards]
        payload["lagging_count"] = len(laggards)
        payload["worst_domain"] = (
            laggards[0].as_dict()
            if laggards
            else (matrix.domains[0].as_dict() if matrix.domains else None)
        )
        if required_files:
            payload["required_files"] = [
                "/".join(_normalise_parts(path)) for path in required_files
            ]
            payload["missing_required_files"] = missing_required
        output_text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    else:
        output_text = render_markdown(matrix, threshold=args.threshold)
        if required_files:
            output_text = output_text.rstrip() + "\n"
            if missing_required:
                output_text += (
                    "Missing required coverage for: "
                    + ", ".join(sorted(missing_required))
                    + "\n"
                )
            else:
                normalised_required: list[str] = [
                    "/".join(_normalise_parts(path)) for path in required_files
                ]
                output_text += (
                    "All required files present in coverage: "
                    + ", ".join(sorted(normalised_required))
                    + "\n"
                )

    if args.output is None:
        print(output_text, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output_text)

    exit_code = 0
    if laggards and args.fail_below_threshold:
        exit_code = 1
    if missing_required:
        if exit_code == 0:
            exit_code = 1
        print(
            "Required files missing from coverage: "
            + ", ".join(sorted(missing_required)),
            file=sys.stderr,
        )

    return exit_code


__all__ = [
    "CoverageDomain",
    "CoverageMatrix",
    "identify_laggards",
    "build_coverage_matrix",
    "render_markdown",
    "main",
]


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
