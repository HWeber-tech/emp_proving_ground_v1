"""Generate a weekly capital efficiency memo from a risk report JSON artefact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.risk import RiskReport, generate_capital_efficiency_memo


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "risk_report",
        type=Path,
        help="Path to a JSON file produced by scripts/generate_risk_report.py",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path to write the memo Markdown output",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    try:
        payload = json.loads(args.risk_report.read_text(encoding="utf-8"))
    except FileNotFoundError:
        parser.error(f"risk report file not found: {args.risk_report}")
    except json.JSONDecodeError as exc:
        parser.error(f"risk report file is not valid JSON: {exc}")

    try:
        report = RiskReport.from_mapping(payload)
    except ValueError as exc:
        parser.error(str(exc))

    memo = generate_capital_efficiency_memo(report)

    if args.output:
        args.output.write_text(memo, encoding="utf-8")

    print(memo)


if __name__ == "__main__":
    main()
