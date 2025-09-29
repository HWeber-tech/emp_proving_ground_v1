"""Helper for running pip-audit with repository-specific guardrails."""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Iterable, Mapping, MutableMapping, Sequence

import yaml

__all__ = [
    "AllowlistEntry",
    "AuditSummary",
    "load_allowlist",
    "invoke_pip_audit",
    "summarise_audit",
    "write_report",
    "run_audit",
    "render_summary",
    "main",
]


@dataclass(frozen=True, slots=True)
class AllowlistEntry:
    """Represents an allowed vulnerability exception."""

    vuln_id: str
    reason: str
    expires: date | None = None

    def is_expired(self, today: date | None = None) -> bool:
        """Return ``True`` when the exception has expired."""

        if self.expires is None:
            return False
        today = today or date.today()
        return today > self.expires


@dataclass(slots=True)
class AuditSummary:
    """Aggregated results returned from :func:`run_audit`."""

    requirements: tuple[str, ...]
    actionable: list[MutableMapping[str, object]]
    suppressed: list[MutableMapping[str, object]]
    expired_allowlist: tuple[str, ...]
    unused_allowlist: tuple[str, ...]

    @property
    def actionable_count(self) -> int:
        """Return the number of actionable vulnerabilities."""

        return sum(len(entry.get("vulns", [])) for entry in self.actionable)

    @property
    def suppressed_count(self) -> int:
        """Return the number of suppressed vulnerabilities."""

        return sum(len(entry.get("vulns", [])) for entry in self.suppressed)

    @property
    def is_clean(self) -> bool:
        """Return ``True`` when no actionable findings remain."""

        return not self.actionable and not self.expired_allowlist


def _parse_date(value: object) -> date | None:
    if not value:
        return None
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, str):
        return date.fromisoformat(value)
    raise TypeError(f"Unsupported expiry type: {value!r}")


def load_allowlist(path: Path | None) -> Mapping[str, AllowlistEntry]:
    """Load vulnerability allowlist entries from ``path``."""

    if path is None or not path.exists():
        return {}

    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    entries: dict[str, AllowlistEntry] = {}

    for item in payload.get("ignored_vulnerabilities", []) or []:
        vuln_id = str(item.get("id", "")).strip()
        if not vuln_id:
            continue
        reason = str(item.get("reason", "No reason provided")).strip()
        expires = _parse_date(item.get("expires")) if item.get("expires") else None
        entries[vuln_id] = AllowlistEntry(vuln_id=vuln_id, reason=reason, expires=expires)

    return entries


def invoke_pip_audit(
    requirements: Sequence[Path],
    *,
    pip_audit_bin: str = "pip-audit",
) -> list[MutableMapping[str, object]]:
    """Invoke the pip-audit CLI and return the JSON payload."""

    command = [pip_audit_bin, "--format", "json"]
    if requirements:
        for requirement in requirements:
            command.extend(["-r", str(requirement)])
    else:
        command.append("--local")

    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:  # pragma: no cover - environment guard
        raise RuntimeError(
            f"Unable to execute {pip_audit_bin!r}; ensure pip-audit is installed"
        ) from exc

    if completed.returncode not in {0, 1}:
        raise RuntimeError(
            "pip-audit exited with unexpected status "
            f"{completed.returncode}: {completed.stderr.strip()}"
        )

    output = completed.stdout.strip()
    if not output:
        return []

    try:
        payload = json.loads(output)
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive guard
        raise RuntimeError("pip-audit returned invalid JSON output") from exc

    if not isinstance(payload, list):
        raise RuntimeError("pip-audit JSON output must be a list")

    return [entry for entry in payload if isinstance(entry, MutableMapping)]


def _annotate_vuln(
    vuln: MutableMapping[str, object],
    allowlist: AllowlistEntry,
) -> MutableMapping[str, object]:
    copy = dict(vuln)
    copy["allowlist_reason"] = allowlist.reason
    if allowlist.expires:
        copy["allowlist_expires"] = allowlist.expires.isoformat()
    return copy


def summarise_audit(
    audit_payload: Iterable[MutableMapping[str, object]],
    allowlist: Mapping[str, AllowlistEntry],
    *,
    requirements: Sequence[Path] = (),
    today: date | None = None,
) -> AuditSummary:
    """Return :class:`AuditSummary` for the given audit payload."""

    actionable: list[MutableMapping[str, object]] = []
    suppressed: list[MutableMapping[str, object]] = []
    expired: set[str] = set()
    used_allowlist: set[str] = set()
    today = today or date.today()

    for package in audit_payload:
        vulns = package.get("vulns")
        if not isinstance(vulns, list):
            continue

        actionable_vulns: list[MutableMapping[str, object]] = []
        suppressed_vulns: list[MutableMapping[str, object]] = []

        for vuln in vulns:
            if not isinstance(vuln, MutableMapping):
                continue
            vuln_id = str(vuln.get("id", "")).strip()
            if not vuln_id:
                actionable_vulns.append(vuln)
                continue

            entry = allowlist.get(vuln_id)
            if entry is None:
                actionable_vulns.append(vuln)
                continue

            if entry.is_expired(today):
                expired.add(entry.vuln_id)
                used_allowlist.add(entry.vuln_id)
                actionable_vulns.append(_annotate_vuln(vuln, entry))
                continue

            used_allowlist.add(entry.vuln_id)
            suppressed_vulns.append(_annotate_vuln(vuln, entry))

        package_descriptor = {
            "name": package.get("name"),
            "version": package.get("version"),
        }

        if actionable_vulns:
            actionable.append({**package_descriptor, "vulns": actionable_vulns})
        if suppressed_vulns:
            suppressed.append({**package_descriptor, "vulns": suppressed_vulns})

    unused = sorted(id_ for id_ in allowlist.keys() if id_ not in used_allowlist)

    return AuditSummary(
        requirements=tuple(str(path) for path in requirements),
        actionable=actionable,
        suppressed=suppressed,
        expired_allowlist=tuple(sorted(expired)),
        unused_allowlist=tuple(unused),
    )


def write_report(summary: AuditSummary, destination: Path) -> None:
    """Persist a machine-readable summary to ``destination``."""

    report = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "requirements": list(summary.requirements),
        "actionable": summary.actionable,
        "suppressed": summary.suppressed,
        "expired_allowlist": list(summary.expired_allowlist),
        "unused_allowlist": list(summary.unused_allowlist),
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")


def run_audit(
    requirements: Sequence[Path],
    allowlist_path: Path | None,
    *,
    report_path: Path | None = None,
    pip_audit_bin: str = "pip-audit",
) -> AuditSummary:
    """Execute pip-audit and return a structured summary."""

    requirement_paths = tuple(Path(path) for path in requirements)
    payload = invoke_pip_audit(requirement_paths, pip_audit_bin=pip_audit_bin)
    allowlist = load_allowlist(allowlist_path)
    summary = summarise_audit(payload, allowlist, requirements=requirement_paths)

    if report_path is not None:
        write_report(summary, report_path)

    return summary


def render_summary(summary: AuditSummary) -> str:
    """Render a human-readable summary for CI logs."""

    lines = [
        "Dependency vulnerability scan",
        f"- Requirements scanned: {', '.join(summary.requirements) or 'environment'}",
        f"- Actionable vulnerabilities: {summary.actionable_count}",
        f"- Suppressed vulnerabilities: {summary.suppressed_count}",
    ]

    if summary.expired_allowlist:
        lines.append(
            "- Expired allowlist entries: " + ", ".join(summary.expired_allowlist)
        )
    if summary.unused_allowlist:
        lines.append(
            "- Unused allowlist entries: " + ", ".join(summary.unused_allowlist)
        )

    if summary.actionable:
        lines.append("")
        lines.append("Actionable findings:")
        for package in summary.actionable:
            name = package.get("name", "<unknown>")
            version = package.get("version", "?")
            for vuln in package.get("vulns", []):
                vuln_id = vuln.get("id", "unknown")
                fix_versions = ", ".join(vuln.get("fix_versions", []) or ["n/a"])
                lines.append(
                    f"- {name} {version}: {vuln_id} (fix: {fix_versions})"
                )

    if summary.suppressed:
        lines.append("")
        lines.append("Suppressed findings:")
        for package in summary.suppressed:
            name = package.get("name", "<unknown>")
            version = package.get("version", "?")
            for vuln in package.get("vulns", []):
                vuln_id = vuln.get("id", "unknown")
                reason = vuln.get("allowlist_reason", "")
                expires = vuln.get("allowlist_expires")
                expiry_note = f" (expires {expires})" if expires else ""
                lines.append(
                    f"- {name} {version}: {vuln_id} — {reason}{expiry_note}"
                )

    return "\n".join(lines)


def _default_requirements() -> tuple[Path, ...]:
    candidates = [
        Path("requirements/base.txt"),
        Path("requirements/dev.txt"),
    ]
    return tuple(path for path in candidates if path.exists())


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point used by ``scripts/security/run_pip_audit.py``."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-r",
        "--requirement",
        action="append",
        type=Path,
        dest="requirements",
        help="Path to a requirements file (defaults to requirements/base.txt and requirements/dev.txt)",
    )
    parser.add_argument(
        "--allowlist",
        type=Path,
        default=Path("config/security/pip_audit_allowlist.yaml"),
        help="Path to the vulnerability allowlist file",
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="Optional destination for the JSON summary report",
    )
    parser.add_argument(
        "--pip-audit-bin",
        type=str,
        default="pip-audit",
        help="Override the pip-audit executable path",
    )
    parser.add_argument(
        "--ignore-expired-allowlist",
        action="store_true",
        help="Exit successfully even when allowlist entries have expired",
    )
    args = parser.parse_args(argv)

    requirements = tuple(args.requirements) if args.requirements else _default_requirements()

    summary = run_audit(
        requirements,
        args.allowlist,
        report_path=args.report,
        pip_audit_bin=args.pip_audit_bin,
    )

    print(render_summary(summary))

    if summary.actionable:
        return 1
    if summary.expired_allowlist and not args.ignore_expired_allowlist:
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI guard
    raise SystemExit(main())
