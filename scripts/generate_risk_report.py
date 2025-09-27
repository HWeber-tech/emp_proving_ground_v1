"""CLI for generating high-impact risk reports."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

from src.risk import (
    PortfolioRiskLimits,
    generate_risk_report,
    load_portfolio_limits,
    render_risk_report_json,
    render_risk_report_markdown,
)
from src.risk.reporting.report_generator import parse_returns_file


def _parse_exposure(value: str) -> tuple[str, float]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("exposures must be provided as SYMBOL=value")
    symbol, raw = value.split("=", 1)
    try:
        return symbol.strip(), float(raw)
    except ValueError as exc:  # pragma: no cover - defensive guard
        raise argparse.ArgumentTypeError(str(exc)) from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "returns_file",
        type=Path,
        help="Path to a text file containing delimited return series",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.99,
        help="Confidence level for VaR/ES calculations (default: 0.99)",
    )
    parser.add_argument(
        "--simulations",
        type=int,
        default=10_000,
        help="Number of Monte Carlo simulations (default: 10000)",
    )
    parser.add_argument(
        "--output-markdown",
        type=Path,
        help="Optional path to write the Markdown report",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        help="Optional path to write the JSON report",
    )
    parser.add_argument(
        "--limits-file",
        type=Path,
        help="Optional path to an alternative portfolio risk limits YAML file",
    )
    parser.add_argument(
        "--exposure",
        action="append",
        type=_parse_exposure,
        help="Repeatable SYMBOL=value exposure inputs",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    returns = parse_returns_file(args.returns_file)
    exposures: Dict[str, float] | None = None
    if args.exposure:
        exposures = {symbol: value for symbol, value in args.exposure}

    limits: PortfolioRiskLimits | None = None
    try:
        limits = load_portfolio_limits(args.limits_file) if args.limits_file else load_portfolio_limits()
    except FileNotFoundError:
        parser.error("Unable to locate portfolio limits file")
    except ValueError as exc:
        parser.error(str(exc))

    report = generate_risk_report(
        returns,
        confidence=args.confidence,
        simulations=args.simulations,
        exposures=exposures,
        limits=limits,
    )

    markdown = render_risk_report_markdown(report)
    json_payload = render_risk_report_json(report)

    if args.output_markdown:
        args.output_markdown.write_text(markdown, encoding="utf-8")
    if args.output_json:
        args.output_json.write_text(json_payload, encoding="utf-8")

    print(markdown)


if __name__ == "__main__":
    main()
