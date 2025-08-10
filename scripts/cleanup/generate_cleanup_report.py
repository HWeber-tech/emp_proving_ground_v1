#!/usr/bin/env python3
"""
Aggregate cleanup signals into a single report:
- Duplicate summary (from audit_duplicates.py)
- Dependency cycles/orphans (from analyze_dependencies.py)
- Dead code candidates (from identify_dead_code.py)
"""

import subprocess


def run(cmd: str) -> str:
    try:
        out = subprocess.check_output(cmd, shell=True, text=True)
    except subprocess.CalledProcessError as e:
        out = e.output
    return out


def main() -> int:
    sections = []
    sections.append("# Cleanup Report\n")
    sections.append("## Duplicates\n\n")
    sections.append(run("python scripts/cleanup/audit_duplicates.py"))
    sections.append("\n## Dependencies\n\n")
    sections.append(run("python scripts/cleanup/analyze_dependencies.py"))
    sections.append("\n## Dead Code\n\n")
    sections.append(run("python scripts/cleanup/identify_dead_code.py"))

    report = "\n".join(sections)
    print(report)
    with open("docs/reports/CLEANUP_REPORT.md", "w", encoding="utf-8") as fh:
        fh.write(report)
    print("\nWrote docs/reports/CLEANUP_REPORT.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


