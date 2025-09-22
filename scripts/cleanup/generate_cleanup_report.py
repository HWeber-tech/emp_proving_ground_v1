#!/usr/bin/env python3
"""
Aggregate cleanup signals into a single report:
- Duplicate summary (from audit_duplicates.py)
- Dependency cycles/orphans (from analyze_dependencies.py)
- Dead code candidates (from identify_dead_code.py)
"""

import subprocess
import sys


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
    sections.append(run(f"{sys.executable} scripts/cleanup/audit_duplicates.py"))
    sections.append("\n## Dependencies\n\n")
    sections.append(run(f"{sys.executable} scripts/cleanup/analyze_dependencies.py"))
    sections.append("\n## Dead Code\n\n")
    sections.append(run(f"{sys.executable} scripts/cleanup/identify_dead_code.py"))

    report = "\n".join(sections)
    print(report)
    # Write to docs for human-readable docs site
    with open("docs/reports/CLEANUP_REPORT.md", "w", encoding="utf-8") as fh:
        fh.write(report)
    print("\nWrote docs/reports/CLEANUP_REPORT.md")
    # Also write to root reports directory per project policy
    try:
        import os

        os.makedirs("reports", exist_ok=True)
        with open("reports/CLEANUP_REPORT.md", "w", encoding="utf-8") as fh2:
            fh2.write(report)
        print("Wrote reports/CLEANUP_REPORT.md")
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
