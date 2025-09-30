"""Thin wrapper around ``pip-audit`` with reporting helpers."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Mapping, Sequence

DEFAULT_REQUIREMENT_CANDIDATES: tuple[str, ...] = (
    "requirements.txt",
    "requirements/requirements.txt",
    "requirements-freeze.txt",
)


class PipAuditExecutionError(RuntimeError):
    """Raised when ``pip-audit`` cannot be executed successfully."""

    def __init__(self, message: str, *, returncode: int, stderr: str | None = None) -> None:
        super().__init__(message)
        self.returncode = returncode
        self.stderr = stderr or ""


@dataclass(frozen=True)
class AuditFinding:
    """Represents a single vulnerability discovered by ``pip-audit``."""

    package: str
    version: str
    vulnerability_id: str
    severity: str | None
    fix_versions: tuple[str, ...]
    summary: str | None = None
    aliases: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, object]:
        return {
            "package": self.package,
            "version": self.version,
            "vulnerability_id": self.vulnerability_id,
            "severity": self.severity,
            "fix_versions": list(self.fix_versions),
            "summary": self.summary,
            "aliases": list(self.aliases),
        }


@dataclass(frozen=True)
class AuditResult:
    """Structured representation of a ``pip-audit`` execution."""

    command: tuple[str, ...]
    findings: tuple[AuditFinding, ...]
    ignored: tuple[AuditFinding, ...]
    returncode: int
    raw_report: tuple[Mapping[str, object], ...]

    @property
    def ok(self) -> bool:
        """Return ``True`` when no actionable findings remain."""

        return not self.findings

    def as_dict(self) -> dict[str, object]:
        return {
            "command": list(self.command),
            "returncode": self.returncode,
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "findings": [finding.as_dict() for finding in self.findings],
            "ignored": [finding.as_dict() for finding in self.ignored],
            "report": [dict(item) for item in self.raw_report],
        }


def _default_requirement_paths() -> list[Path]:
    paths: list[Path] = []
    for candidate in DEFAULT_REQUIREMENT_CANDIDATES:
        path = Path(candidate)
        if path.exists():
            paths.append(path)
    return paths


def load_ignore_list(path: Path) -> set[str]:
    """Return vulnerability identifiers listed in ``path``."""

    ignore: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        entry = line.strip()
        if not entry or entry.startswith("#"):
            continue
        ignore.add(entry)
    return ignore


def _build_command(requirements: Sequence[Path], extra_args: Sequence[str]) -> list[str]:
    command = ["pip-audit", "--format", "json"]
    for requirement in requirements:
        command.extend(["--requirement", str(requirement)])
    command.extend(extra_args)
    return command


def _parse_report(payload: object) -> tuple[Mapping[str, object], ...]:
    if payload is None:
        return tuple()
    if isinstance(payload, list):
        normalised: list[Mapping[str, object]] = []
        for entry in payload:
            if isinstance(entry, Mapping):
                normalised.append(entry)
        return tuple(normalised)
    raise ValueError("Unexpected pip-audit payload")


def _collect_findings(
    report: Sequence[Mapping[str, object]],
    *,
    ignore_ids: set[str],
) -> tuple[tuple[AuditFinding, ...], tuple[AuditFinding, ...]]:
    actionable: list[AuditFinding] = []
    ignored: list[AuditFinding] = []

    for item in report:
        package = str(item.get("name", ""))
        version = str(item.get("version", ""))
        vulnerabilities = item.get("vulns")
        if not isinstance(vulnerabilities, Sequence):
            continue
        for vuln in vulnerabilities:
            if not isinstance(vuln, Mapping):
                continue
            identifier = str(vuln.get("id", "")) or "UNKNOWN"
            severity = str(vuln.get("severity")) if vuln.get("severity") else None
            fix_versions_raw = vuln.get("fix_versions") or []
            if not isinstance(fix_versions_raw, Sequence):
                fix_versions_raw = []
            fix_versions = tuple(str(version) for version in fix_versions_raw if version)
            summary = vuln.get("description")
            if summary is not None:
                summary = str(summary)
            aliases_raw = vuln.get("aliases") or []
            aliases: tuple[str, ...]
            if isinstance(aliases_raw, Sequence):
                aliases = tuple(str(alias) for alias in aliases_raw if alias)
            else:
                aliases = tuple()

            finding = AuditFinding(
                package=package,
                version=version,
                vulnerability_id=identifier,
                severity=severity,
                fix_versions=fix_versions,
                summary=summary,
                aliases=aliases,
            )

            identifiers = {identifier, *aliases}
            if identifiers & ignore_ids:
                ignored.append(finding)
            else:
                actionable.append(finding)

    return tuple(actionable), tuple(ignored)


def run_audit(
    requirements: Sequence[Path],
    *,
    ignore_ids: Iterable[str] | None = None,
    extra_args: Sequence[str] | None = None,
) -> AuditResult:
    """Execute ``pip-audit`` and return a structured result."""

    if not requirements:
        raise ValueError("At least one requirement file must be supplied")

    ignore_set = set(ignore_ids or ())
    command = _build_command(requirements, extra_args or ())

    try:
        process = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:  # pragma: no cover - depends on environment
        raise PipAuditExecutionError(
            "pip-audit executable not found", returncode=127
        ) from exc

    if process.returncode not in {0, 1}:
        message = "pip-audit failed"
        if process.stderr:
            message = f"{message}: {process.stderr.strip()}"
        raise PipAuditExecutionError(
            message,
            returncode=process.returncode,
            stderr=process.stderr,
        )

    stdout = process.stdout.strip()
    if not stdout:
        payload: object = []
    else:
        try:
            payload = json.loads(stdout)
        except json.JSONDecodeError as exc:
            raise PipAuditExecutionError(
                "pip-audit returned malformed JSON",
                returncode=process.returncode,
                stderr=process.stderr,
            ) from exc

    report = _parse_report(payload)
    findings, ignored = _collect_findings(report, ignore_ids=ignore_set)

    return AuditResult(
        command=tuple(command),
        findings=findings,
        ignored=ignored,
        returncode=process.returncode,
        raw_report=report,
    )


def format_markdown(result: AuditResult) -> str:
    """Render an ``AuditResult`` as a Markdown report."""

    command_repr = " ".join(shlex.quote(part) for part in result.command)
    lines = [
        "# Dependency vulnerability scan",
        "",
        f"- Command: `{command_repr}`",
        f"- Outstanding vulnerabilities: {len(result.findings)}",
        f"- Ignored vulnerabilities: {len(result.ignored)}",
        "",
    ]

    if result.findings:
        lines.append("## Outstanding findings")
        lines.append("| Package | Version | Vulnerability | Severity | Fix versions |")
        lines.append("| --- | --- | --- | --- | --- |")
        for finding in result.findings:
            fixes = ", ".join(finding.fix_versions) or "-"
            severity = finding.severity or "unknown"
            lines.append(
                f"| {finding.package} | {finding.version} | {finding.vulnerability_id} | {severity} | {fixes} |"
            )
        lines.append("")
    else:
        lines.append("No actionable vulnerabilities detected.")
        lines.append("")

    if result.ignored:
        lines.append("## Ignored vulnerabilities")
        lines.append("| Package | Version | Vulnerability | Severity | Fix versions |")
        lines.append("| --- | --- | --- | --- | --- |")
        for finding in result.ignored:
            fixes = ", ".join(finding.fix_versions) or "-"
            severity = finding.severity or "unknown"
            lines.append(
                f"| {finding.package} | {finding.version} | {finding.vulnerability_id} | {severity} | {fixes} |"
            )
    else:
        lines.append("No ignored vulnerabilities recorded.")

    return "\n".join(lines)


def format_json(result: AuditResult) -> str:
    """Serialise an ``AuditResult`` as JSON."""

    return json.dumps(result.as_dict(), indent=2)


def _write_output(content: str, destination: Path | None) -> None:
    if destination is None:
        print(content)
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(content, encoding="utf-8")


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run pip-audit with CI-friendly output")
    parser.add_argument(
        "-r",
        "--requirement",
        dest="requirements",
        action="append",
        type=Path,
        help="Path to a requirements file (defaults cover common locations).",
    )
    parser.add_argument(
        "--ignore-file",
        type=Path,
        help="Path to a newline-delimited list of vulnerability IDs to ignore.",
    )
    parser.add_argument(
        "--format",
        choices=("markdown", "json"),
        default="markdown",
        help="Output format.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path to write the report. Defaults to stdout.",
    )
    parser.add_argument(
        "--extra-arg",
        dest="extra_args",
        action="append",
        default=[],
        help="Additional arguments forwarded to pip-audit.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""

    args = _parse_args(argv)
    requirements = args.requirements
    if not requirements:
        requirements = _default_requirement_paths()
    if not requirements:
        print("No requirement files supplied and defaults were not found.", file=sys.stderr)
        return 2

    ignore_ids: set[str] = set()
    if args.ignore_file is not None:
        ignore_ids = load_ignore_list(args.ignore_file)

    try:
        result = run_audit(requirements, ignore_ids=ignore_ids, extra_args=args.extra_args)
    except PipAuditExecutionError as exc:
        print(str(exc), file=sys.stderr)
        if exc.stderr:
            print(exc.stderr, file=sys.stderr)
        return exc.returncode or 2
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    formatter = format_markdown if args.format == "markdown" else format_json
    content = formatter(result)
    _write_output(content, args.output)

    return 0 if result.ok else 1


if __name__ == "__main__":  # pragma: no cover - CLI
    sys.exit(main())
