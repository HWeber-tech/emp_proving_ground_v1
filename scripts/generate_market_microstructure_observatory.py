#!/usr/bin/env python3
"""Generate the Market Microstructure Observatory report.

The high-impact roadmap calls for notebooks showcasing liquidity and volume
profiling studies.  This script converts the canonical JSON fixture into a
Markdown observatory report that can be opened in any notebook viewer or static
site.  Analysts can run it locally or as part of CI to refresh the artefact
stored under ``artifacts/reports``.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input",
        type=Path,
        nargs="?",
        default=Path("docs/microstructure_raw_data.json"),
        help="Path to the canonical microstructure JSON payload",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/reports/market_microstructure_observatory.md"),
        help="Path to the generated Markdown report",
    )
    return parser.parse_args()


def load_payload(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_markdown(payload: dict) -> str:
    summary = payload.get("summary", {})
    latency = payload.get("latency_analysis", {})
    depth = payload.get("depth_analysis", {})
    frequency = payload.get("frequency_analysis", {})
    generated_at = datetime.utcnow().isoformat()

    def fmt_section(title: str, content: dict) -> list[str]:
        if not content:
            return [f"### {title}", "No data available."]
        rows = [
            f"- **{key.replace('_', ' ').title()}**: {value}"
            for key, value in content.items()
        ]
        return [f"### {title}"] + rows

    lines: list[str] = [
        "# Market Microstructure Observatory",
        "",
        f"_Generated at {generated_at} UTC_",
        "",
        "## Session Summary",
        f"- **Symbol**: {summary.get('symbol', 'N/A')}",
        f"- **Duration (minutes)**: {summary.get('duration_minutes', 'N/A')}",
        f"- **Total Updates**: {summary.get('total_updates', 'N/A')}",
        f"- **Capture Timestamp**: {summary.get('test_timestamp', 'N/A')}",
        "",
    ]

    for section in (
        fmt_section("Latency Profile", latency),
        fmt_section("Depth Snapshot", depth),
        fmt_section("Update Frequency", frequency),
    ):
        lines.extend(section)
        lines.append("")

    lines.extend(
        [
            "## Observability Notes",
            "- Latency metrics are sourced from FIX round-trip measurements.",
            "- Depth statistics summarise top-of-book levels captured in the raw sample.",
            "- Frequency section highlights heartbeat cadence for monitoring drift.",
            "",
            "## Next Actions",
            "- Attach the rendered Markdown to the ops dashboard release notes.",
            "- Feed latency anomalies into `scripts/status_metrics.py` for alerting.",
            "- Enrich this report with venue-specific liquidity sweeps when new data arrives.",
        ]
    )

    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    payload = load_payload(args.input)
    markdown = build_markdown(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(markdown, encoding="utf-8")
    print(f"Report written to {args.output}")


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
