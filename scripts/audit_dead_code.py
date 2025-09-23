#!/usr/bin/env python3
"""Generate a dead-code audit report using vulture."""

from __future__ import annotations

import re
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from textwrap import indent


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "docs" / "reports"
REPORT_PATH = REPORT_DIR / "dead_code_audit.md"
ALLOWED_RETURN_CODES = {0, 1, 2, 3}


UNUSED_PATTERN = re.compile(
    r"^(?P<path>[^:]+):(?P<line>\d+): unused (?P<symbol>[^']+) '"
    r"(?P<name>[^']+)' \((?P<confidence>\d+)% confidence, (?P<size>\d+) lines?\)$"
)
UNREACHABLE_PATTERN = re.compile(
    r"^(?P<path>[^:]+):(?P<line>\d+): unreachable code after '(?P<trigger>[^']+)' "
    r"\((?P<confidence>\d+)% confidence, (?P<size>\d+) lines?\)$"
)


def run_vulture() -> tuple[list[dict[str, object]], int, str]:
    """Execute vulture and return parsed findings along with metadata."""

    cmd = [
        "vulture",
        str(ROOT / "src"),
        str(ROOT / "tests"),
        "--min-confidence",
        "80",
        "--sort-by-size",
    ]

    completed = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
    )

    if completed.returncode not in ALLOWED_RETURN_CODES:
        sys.stderr.write(completed.stderr)
        raise SystemExit(f"vulture exited with unexpected status {completed.returncode}.")

    stdout = completed.stdout.strip()
    if not stdout:
        return [], completed.returncode, completed.stderr

    findings: list[dict[str, object]] = []
    for line in stdout.splitlines():
        text = line.strip()
        if not text:
            continue
        match_unused = UNUSED_PATTERN.match(text)
        match_unreachable = UNREACHABLE_PATTERN.match(text)
        path, parsed_line, confidence, size, symbol_type, name = (None,) * 6
        if match_unused:
            path = Path(match_unused.group("path"))
            if not path.is_absolute():
                path = (ROOT / path).resolve()
            parsed_line = int(match_unused.group("line"))
            confidence = int(match_unused.group("confidence"))
            size = int(match_unused.group("size"))
            symbol_type = match_unused.group("symbol").strip()
            name = match_unused.group("name").strip()
        elif match_unreachable:
            path = Path(match_unreachable.group("path"))
            if not path.is_absolute():
                path = (ROOT / path).resolve()
            parsed_line = int(match_unreachable.group("line"))
            confidence = int(match_unreachable.group("confidence"))
            size = int(match_unreachable.group("size"))
            trigger = match_unreachable.group("trigger").strip()
            symbol_type = "unreachable"
            name = f"after {trigger}"
        else:
            parts = text.split(":", 2)
            if len(parts) >= 3:
                path = Path(parts[0])
                if not path.is_absolute():
                    path = (ROOT / path).resolve()
                try:
                    parsed_line = int(parts[1])
                except ValueError:
                    parsed_line = None
            else:
                path = None
            symbol_type = "unknown"
            name = text
            confidence = 0
            size = 0

        if path is not None and path.suffix == ".py":
            try:
                module = (
                    path.with_suffix("").resolve().relative_to(ROOT).as_posix().replace("/", ".")
                )
            except ValueError:
                module = path.with_suffix("").name
        elif path is not None:
            module = path.stem
        else:
            module = "unknown"

        if path is not None:
            try:
                rel_path = path.resolve().relative_to(ROOT)
            except ValueError:
                rel_path = path
            path_str = rel_path.as_posix()
        else:
            path_str = "<unknown>"

        findings.append(
            {
                "path": path_str,
                "module": module,
                "line": parsed_line if parsed_line is not None else 0,
                "confidence": confidence,
                "size": size,
                "type": symbol_type,
                "name": name,
            }
        )

    return findings, completed.returncode, completed.stderr


def format_table(findings: list[dict[str, object]]) -> str:
    """Render a Markdown table for the top findings."""

    header = "| Confidence | Size | Type | Module | Object | Line |\n"
    divider = "| --- | --- | --- | --- | --- | --- |\n"

    rows = []
    for entry in findings:
        confidence = int(entry.get("confidence", 0))
        size = int(entry.get("size", 0))
        entry_type = str(entry.get("type", "?"))
        module = str(entry.get("module", "?"))
        name = str(entry.get("name", "?"))
        line = int(entry.get("line", 0))
        rows.append(f"| {confidence} | {size} | {entry_type} | {module} | {name} | {line} |")

    if not rows:
        rows.append("| – | – | – | – | – | – |")

    return header + divider + "\n".join(rows)


def build_report(
    findings: list[dict[str, object]],
    exit_code: int,
    stderr: str,
) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M %Z")
    cmd_repr = "vulture src tests --min-confidence 80 --sort-by-size"

    total = len(findings)
    by_type = Counter(str(item.get("type", "unknown")) for item in findings)
    by_module = Counter(str(item.get("module", "unknown")) for item in findings)

    summary_lines = [
        f"- **Total candidates**: {total}",
    ]
    if exit_code not in {0, 1}:
        summary_lines.append(
            f"- **Command exit status**: vulture exited with status {exit_code}; review stderr below."
        )
    if by_type:
        type_breakdown = ", ".join(
            f"{count} {item_type}(s)" for item_type, count in by_type.most_common()
        )
        summary_lines.append(f"- **By symbol type**: {type_breakdown}")

    if by_module:
        top_modules = ", ".join(
            f"`{module}` ({count})" for module, count in by_module.most_common(5)
        )
        summary_lines.append(f"- **Top modules**: {top_modules}")

    top_findings = findings[:20]

    lines = [
        f"# Dead code audit – {timestamp}",
        "",
        f"*Generated via `{cmd_repr}`.*",
        "",
        "## Summary",
        "",
        *summary_lines,
        "",
        "## Top candidates",
        "",
        format_table(top_findings),
        "",
        "## Observations",
        "",
        "- Vulture heuristics can report false positives, especially for dynamic imports, registry lookups, or symbols referenced exclusively via strings.",
        "- Review candidates with module owners before deleting code; prioritize entries with high confidence, large size, and no associated tests.",
        "",
        "## Next steps",
        "",
        "1. Convert high-confidence findings into cleanup tickets, starting with modules that are otherwise unreferenced.",
        "2. Update or silence legitimate dynamic hooks using `# noqa: Vulture` comments to keep future scans focused on actionable debt.",
        "3. Re-run this script after each cleanup pass to monitor progress.",
    ]

    if stderr.strip():
        lines.extend(
            [
                "",
                "## Command stderr",
                "",
                "```",
                stderr.strip(),
                "```",
            ]
        )

    lines.append("")
    return "\n".join(lines)


def main() -> None:
    findings, exit_code, stderr = run_vulture()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report = build_report(findings, exit_code, stderr)
    REPORT_PATH.write_text(report, encoding="utf-8")

    sys.stdout.write(
        indent(
            "\n".join(
                [
                    f"Report written to {REPORT_PATH.relative_to(ROOT)}",
                    "Re-run the script after addressing findings to refresh the snapshot.",
                ]
            ),
            "",
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
