"""Run pip-audit with repository-specific allowlisting."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--allowlist",
        type=Path,
        default=Path("config/governance/dependency_allowlist.json"),
        help="JSON file containing ignored vulnerability IDs.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("artifacts/dependency_audit/report.json"),
        help="Path to write the audit JSON report.",
    )
    parser.add_argument(
        "--exit-on-vuln",
        action="store_true",
        help="Exit with status 1 when unapproved vulnerabilities are present.",
    )
    return parser.parse_args()


def _load_allowlist(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"ignored_vulnerabilities": []}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _run_pip_audit() -> List[Dict[str, Any]]:
    command = [sys.executable, "-m", "pip_audit", "--format", "json", "--progress-spinner", "off"]
    proc = subprocess.run(command, check=False, capture_output=True, text=True)
    if proc.returncode not in (0, 1):
        print(proc.stdout)
        print(proc.stderr, file=sys.stderr)
        raise SystemExit(proc.returncode)
    if not proc.stdout.strip():
        return []
    return json.loads(proc.stdout)


def _filter_vulnerabilities(
    audit_results: Sequence[Dict[str, Any]],
    allowlist: Dict[str, Any],
) -> Dict[str, Any]:
    ignored = {
        entry["id"]: entry
        for entry in allowlist.get("ignored_vulnerabilities", [])
        if "id" in entry
    }

    filtered_results = []
    for package in audit_results:
        remaining_vulns = [v for v in package.get("vulns", []) if v.get("id") not in ignored]
        if remaining_vulns:
            filtered = dict(package)
            filtered["vulns"] = remaining_vulns
            filtered_results.append(filtered)

    return {
        "ignored": list(ignored.values()),
        "unapproved": filtered_results,
    }


def main() -> int:
    args = _parse_args()
    allowlist = _load_allowlist(args.allowlist)
    audit_results = _run_pip_audit()
    filtered = _filter_vulnerabilities(audit_results, allowlist)

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(filtered, indent=2, sort_keys=True))

    unapproved = filtered["unapproved"]
    if unapproved:
        message = ["Unapproved dependency vulnerabilities detected:"]
        for package in unapproved:
            pkg_name = package["name"]
            version = package.get("version", "?")
            for vuln in package.get("vulns", []):
                message.append(f"- {pkg_name} {version}: {vuln.get('id')} ({vuln.get('fix_versions')})")
        print("\n".join(message), file=sys.stderr)
        if args.exit_on_vuln:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
