"""Guardrail checks that ensure critical domains retain test coverage."""

from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

__all__ = [
    "CoverageGuardrail",
    "CoverageGuardrailReport",
    "evaluate_guardrails",
    "render_report",
    "main",
]

_DEFAULT_TARGETS: dict[str, str] = {
    "ingest_production_slice": "src/data_foundation/ingest/production_slice.py",
    "timescale_pipeline": "src/data_foundation/ingest/timescale_pipeline.py",
    "risk_policy": "src/trading/risk/risk_policy.py",
}


@dataclass(frozen=True)
class CoverageGuardrail:
    """Coverage summary for a single guardrail target."""

    label: str
    path: str
    covered: int
    missed: int
    missing: bool = False

    @property
    def total(self) -> int:
        return self.covered + self.missed

    @property
    def percent(self) -> float:
        total = self.total
        if total == 0:
            return 0.0
        return round((self.covered / total) * 100.0, 2)

    def as_dict(self) -> dict[str, Any]:
        payload = {
            "label": self.label,
            "path": self.path,
            "covered": self.covered,
            "missed": self.missed,
            "coverage_percent": self.percent,
        }
        if self.missing:
            payload["missing"] = True
        return payload


@dataclass(frozen=True)
class CoverageGuardrailReport:
    """Evaluation outcome for a set of guardrail targets."""

    generated_at: str
    minimum_percent: float
    targets: tuple[CoverageGuardrail, ...]
    failing: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "minimum_percent": self.minimum_percent,
            "targets": [target.as_dict() for target in self.targets],
            "failing": list(self.failing),
        }

    @property
    def has_failures(self) -> bool:
        return bool(self.failing)


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


def _build_coverage_index(root: ET.Element) -> dict[str, tuple[int, int]]:
    coverage: dict[str, tuple[int, int]] = {}
    for class_node in _iter_class_nodes(root):
        filename = class_node.attrib.get("filename")
        if not filename:
            continue
        normalised = "/".join(_normalise_parts(filename))
        if not normalised:
            continue
        covered, missed = _count_line_coverage(class_node)
        if covered == 0 and missed == 0:
            continue
        existing = coverage.get(normalised)
        if existing:
            covered += existing[0]
            missed += existing[1]
        coverage[normalised] = (covered, missed)
    return coverage


def _resolve_targets(targets: Mapping[str, str]) -> tuple[tuple[str, str], ...]:
    resolved: list[tuple[str, str]] = []
    for label, path in targets.items():
        resolved.append((str(label), "/".join(_normalise_parts(path))))
    return tuple(resolved)


def evaluate_guardrails(
    coverage_report: Path,
    targets: Mapping[str, str] | None = None,
    *,
    minimum_percent: float = 80.0,
) -> CoverageGuardrailReport:
    """Evaluate guardrail targets against a Cobertura XML report."""

    if targets is None:
        targets = _DEFAULT_TARGETS

    tree = ET.parse(coverage_report)
    root = tree.getroot()
    coverage_index = _build_coverage_index(root)
    resolved_targets = _resolve_targets(targets)

    guardrails: list[CoverageGuardrail] = []
    failing: list[str] = []

    for label, path in resolved_targets:
        stats = coverage_index.get(path)
        if stats is None:
            guardrails.append(
                CoverageGuardrail(label=label, path=path, covered=0, missed=0, missing=True)
            )
            failing.append(label)
            continue
        covered, missed = stats
        guardrail = CoverageGuardrail(
            label=label,
            path=path,
            covered=covered,
            missed=missed,
        )
        guardrails.append(guardrail)
        if guardrail.percent < minimum_percent:
            failing.append(label)

    report = CoverageGuardrailReport(
        generated_at=datetime.now(tz=UTC).isoformat(timespec="seconds"),
        minimum_percent=minimum_percent,
        targets=tuple(guardrails),
        failing=tuple(failing),
    )
    return report


def render_report(report: CoverageGuardrailReport) -> str:
    lines = [
        f"Guardrail coverage (minimum {report.minimum_percent:.2f}%):",
    ]
    for target in report.targets:
        status: str
        if target.missing:
            status = "missing"
        elif target.percent < report.minimum_percent:
            status = "fail"
        else:
            status = "ok"
        lines.append(
            f"- {target.label}: {target.percent:.2f}% ({status}) [{target.path}]"
        )
    if report.has_failures:
        lines.append("Guardrail coverage failed")
    else:
        lines.append("All guardrail targets met minimum coverage")
    return "\n".join(lines)


def _parse_target_overrides(entries: Sequence[str] | None) -> dict[str, str]:
    if not entries:
        return {}
    overrides: dict[str, str] = {}
    for entry in entries:
        if "=" not in entry:
            raise ValueError(f"Invalid target specification: {entry!r}")
        label, path = entry.split("=", 1)
        overrides[label.strip()] = path.strip()
    return overrides


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate coverage guardrails")
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("coverage.xml"),
        help="Path to the Cobertura XML report",
    )
    parser.add_argument(
        "--min-percent",
        type=float,
        default=80.0,
        help="Minimum required coverage percentage",
    )
    parser.add_argument(
        "--target",
        action="append",
        dest="targets",
        help="Override default targets using label=path",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Render the report as JSON",
    )

    args = parser.parse_args(argv)

    overrides = _parse_target_overrides(args.targets)
    if overrides:
        targets = overrides
    else:
        targets = _DEFAULT_TARGETS

    report = evaluate_guardrails(
        args.report,
        targets,
        minimum_percent=args.min_percent,
    )

    if args.json:
        print(json.dumps(report.as_dict(), indent=2, sort_keys=True))
    else:
        print(render_report(report))

    return 1 if report.has_failures else 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    sys.exit(main())
