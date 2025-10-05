from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .ci_metrics import (
    DEFAULT_METRICS_PATH,
    load_metrics,
    summarise_trend_staleness,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate CI telemetry freshness by checking coverage, formatter, and remediation trend ages."
        ),
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=DEFAULT_METRICS_PATH,
        help="Path to the CI metrics JSON file (defaults to tests/.telemetry/ci_metrics.json)",
    )
    parser.add_argument(
        "--max-age-hours",
        type=float,
        default=24.0,
        help="Maximum allowed age (in hours) for the most recent entry in each telemetry trend",
    )
    parser.add_argument(
        "--require-trend",
        action="append",
        dest="required_trends",
        metavar="TREND",
        help=(
            "Limit validation to the specified trend (may be passed multiple times); "
            "defaults to validating all known trends"
        ),
    )
    parser.add_argument(
        "--format",
        choices=("human", "json"),
        default="human",
        help="Output format for the summary (defaults to human-readable text)",
    )
    return parser


def _selected_trends(
    trends: Mapping[str, Mapping[str, object]], required: Iterable[str] | None
) -> list[tuple[str, Mapping[str, object]]]:
    if required:
        selected: list[tuple[str, Mapping[str, object]]] = []
        for name in required:
            if name in trends:
                selected.append((name, trends[name]))
            else:
                selected.append((name, {}))
        return selected
    return sorted(trends.items(), key=lambda item: item[0])


def _render_human(summary: Mapping[str, object], required: Iterable[str] | None) -> str:
    evaluated_at = summary.get("evaluated_at", "(unknown)")
    threshold = summary.get("threshold_hours", "?")
    lines = [f"Evaluated at: {evaluated_at} (threshold={threshold}h)"]
    trends_obj = summary.get("trends", {})
    if not isinstance(trends_obj, Mapping):
        return "\n".join(lines + ["No trend data available"])

    for name, stats in _selected_trends(trends_obj, required):
        if not stats:
            lines.append(f"! {name}: missing trend telemetry")
            continue
        entry_count = stats.get("entry_count", "?")
        last_ts = stats.get("last_timestamp") or "never"
        age_hours = stats.get("age_hours")
        if isinstance(age_hours, (int, float)):
            age_repr = f"{age_hours:.2f}h"
        else:
            age_repr = "n/a"
        is_stale = bool(stats.get("is_stale"))
        status = "STALE" if is_stale else "fresh"
        prefix = "!" if is_stale else "-"
        lines.append(
            f"{prefix} {name}: {status} (entries={entry_count}, last={last_ts}, age={age_repr})"
        )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    metrics = load_metrics(args.metrics)
    summary = summarise_trend_staleness(metrics, max_age_hours=args.max_age_hours)

    trends = summary.get("trends", {})
    if not isinstance(trends, Mapping):
        parser.error("Unexpected trends payload in metrics summary")

    required = tuple(args.required_trends) if args.required_trends else None
    selected = _selected_trends(trends, required)

    stale_trends: list[str] = []
    missing_trends: list[str] = []
    for name, stats in selected:
        if not stats:
            missing_trends.append(name)
            continue
        if bool(stats.get("is_stale")):
            stale_trends.append(name)

    if args.format == "json":
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(_render_human(summary, required))

    exit_code = 0
    if missing_trends:
        print(
            f"Missing telemetry trends: {', '.join(sorted(missing_trends))}",
            file=sys.stderr,
        )
        exit_code = 1

    if stale_trends:
        print(
            f"Stale telemetry trends: {', '.join(sorted(stale_trends))}",
            file=sys.stderr,
        )
        exit_code = 1

    return exit_code


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
