"""Validate that critical pytest modules run and pass in CI guardrails."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence
import xml.etree.ElementTree as ET

try:
    from datetime import UTC  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover - Python < 3.11 fallback
    UTC = timezone.utc

__all__ = [
    "PytestCase",
    "RequirementResult",
    "analyse_report",
    "evaluate_requirements",
    "render_text_report",
    "main",
]


@dataclass(frozen=True)
class PytestCase:
    """Minimal representation of a pytest test case from a JUnit XML report."""

    classname: str
    name: str
    outcome: str  # one of "passed", "failed", "error", "skipped"

    @property
    def node_id(self) -> str:
        if self.classname:
            return f"{self.classname}::{self.name}" if self.name else self.classname
        return self.name


@dataclass(frozen=True)
class RequirementResult:
    """Outcome for a single requirement."""

    requirement: str
    satisfied: bool
    matches: tuple[PytestCase, ...]
    reason: str | None = None

    def as_dict(self) -> dict[str, object]:
        payload = {
            "requirement": self.requirement,
            "satisfied": self.satisfied,
            "matches": [case.node_id for case in self.matches],
        }
        if self.reason:
            payload["reason"] = self.reason
        return payload


def _normalise(value: str | None) -> str:
    return (value or "").strip()


def analyse_report(report: Path) -> tuple[PytestCase, ...]:
    """Parse a pytest JUnit XML report into simplified test case records."""

    tree = ET.parse(report)
    root = tree.getroot()
    cases: list[PytestCase] = []
    for testcase in root.iter("testcase"):
        classname = _normalise(testcase.attrib.get("classname"))
        name = _normalise(testcase.attrib.get("name"))
        outcome = "passed"
        if testcase.find("failure") is not None:
            outcome = "failed"
        elif testcase.find("error") is not None:
            outcome = "error"
        elif testcase.find("skipped") is not None:
            outcome = "skipped"
        cases.append(PytestCase(classname=classname, name=name, outcome=outcome))
    return tuple(cases)


def _matches(requirement: str, case: PytestCase) -> bool:
    if "::" in requirement:
        return case.node_id == requirement
    if requirement and case.classname:
        if case.classname == requirement:
            return True
        if case.classname.startswith(f"{requirement}."):
            return True
    return False


def evaluate_requirements(
    cases: Sequence[PytestCase],
    requirements: Sequence[str],
) -> tuple[RequirementResult, ...]:
    """Evaluate whether required modules/tests executed successfully."""

    results: list[RequirementResult] = []
    for requirement in requirements:
        matches = tuple(case for case in cases if _matches(requirement, case))
        if not matches:
            results.append(
                RequirementResult(
                    requirement=requirement,
                    satisfied=False,
                    matches=(),
                    reason="no matching tests found",
                )
            )
            continue

        passing = tuple(case for case in matches if case.outcome == "passed")
        if passing:
            results.append(
                RequirementResult(
                    requirement=requirement,
                    satisfied=True,
                    matches=passing,
                )
            )
            continue

        status_summary = ", ".join(sorted({case.outcome for case in matches}))
        results.append(
            RequirementResult(
                requirement=requirement,
                satisfied=False,
                matches=matches,
                reason=f"matching tests present but not passing (statuses: {status_summary})",
            )
        )
    return tuple(results)


def render_text_report(results: Sequence[RequirementResult]) -> str:
    lines = ["Pytest guardrail results:"]
    for result in results:
        status = "PASS" if result.satisfied else "FAIL"
        details = f"matches={len(result.matches)}"
        if result.reason:
            details = f"{details}; {result.reason}"
        lines.append(f"- {status} {result.requirement}: {details}")
    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate pytest guardrail execution")
    parser.add_argument("--report", required=True, type=Path, help="Path to pytest JUnit XML report")
    parser.add_argument(
        "--require",
        dest="requirements",
        action="append",
        default=[],
        help="Module prefix or fully-qualified node id that must have passing tests",
    )
    parser.add_argument(
        "--format",
        choices={"text", "json"},
        default="text",
        help="Output format",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    report_path: Path = args.report
    requirements: list[str] = list(args.requirements)
    if not requirements:
        parser.error("At least one --require entry must be provided")

    cases = analyse_report(report_path)
    results = evaluate_requirements(cases, requirements)
    failing = [result for result in results if not result.satisfied]

    payload = {
        "generated_at": datetime.now(tz=UTC).isoformat(timespec="seconds"),
        "report": str(report_path),
        "results": [result.as_dict() for result in results],
    }

    if args.format == "json":
        output = json.dumps(payload, indent=2, sort_keys=True)
    else:
        output = render_text_report(results)
    print(output)

    return 0 if not failing else 1


if __name__ == "__main__":
    raise SystemExit(main())
