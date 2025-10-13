#!/usr/bin/env python3
"""Produce backlog artefacts from AlphaTrade final dry run evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.operations.dry_run_audit import DryRunStatus
from src.operations.final_dry_run_backlog import (
    collect_backlog_items,
    format_backlog_markdown,
    items_to_json_serialisable,
)


_SEVERITY_RANK = {
    DryRunStatus.pass_: 0,
    DryRunStatus.warn: 1,
    DryRunStatus.fail: 2,
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Transform final dry run summaries into backlog-ready artefacts."
        )
    )
    parser.add_argument(
        "--summary",
        type=Path,
        required=True,
        help="Path to the dry run summary JSON (or orchestrator bundle).",
    )
    parser.add_argument(
        "--sign-off",
        type=Path,
        help="Optional path to a sign-off JSON file when not embedded in the summary.",
    )
    parser.add_argument(
        "--review",
        type=Path,
        help="Optional path to a review JSON file when not embedded in the summary.",
    )
    parser.add_argument(
        "--format",
        choices=("markdown", "json"),
        default="markdown",
        help="Output format (default: markdown).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Write the backlog to this path instead of stdout.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Final Dry Run Backlog",
        help="Heading to use when emitting Markdown output.",
    )
    parser.add_argument(
        "--min-severity",
        choices=(DryRunStatus.pass_.value, DryRunStatus.warn.value, DryRunStatus.fail.value),
        default=DryRunStatus.warn.value,
        help="Ignore items below this severity (default: warn).",
    )
    parser.add_argument(
        "--include-pass",
        action="store_true",
        help="Include PASS items even when a higher min severity is requested.",
    )
    parser.add_argument(
        "--fail-on-warn",
        action="store_true",
        help="Exit with status 1 when WARN items are present.",
    )
    parser.add_argument(
        "--fail-on-fail",
        action="store_true",
        help="Exit with status 2 when FAIL items are present.",
    )
    return parser


def _load_json(path: Path | None) -> Mapping[str, Any] | None:
    if path is None:
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _extract_embedded(payload: Mapping[str, Any], key: str) -> Mapping[str, Any] | None:
    value = payload.get(key)
    return value if isinstance(value, Mapping) else None


def _filter_by_severity(
    items,
    *,
    minimum: DryRunStatus,
    include_pass: bool,
):
    threshold = _SEVERITY_RANK[minimum]
    result = []
    for item in items:
        rank = _SEVERITY_RANK.get(item.severity, 0)
        if rank < threshold and not (include_pass and item.severity is DryRunStatus.pass_):
            continue
        result.append(item)
    return tuple(result)


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    summary_payload = _load_json(args.summary)
    if summary_payload is None:
        parser.error("Summary file could not be read")

    if not isinstance(summary_payload, Mapping):
        parser.error("Summary JSON must contain an object")

    embedded_summary = _extract_embedded(summary_payload, "summary") or summary_payload
    embedded_sign_off = (
        _load_json(args.sign_off)
        if args.sign_off
        else _extract_embedded(summary_payload, "sign_off")
    )
    embedded_review = (
        _load_json(args.review)
        if args.review
        else _extract_embedded(summary_payload, "review")
    )

    include_pass = args.include_pass or args.min_severity == DryRunStatus.pass_.value

    items = collect_backlog_items(
        embedded_summary,
        sign_off=embedded_sign_off,
        review=embedded_review,
        include_pass=include_pass,
    )

    minimum_status = DryRunStatus(args.min_severity)
    filtered_items = _filter_by_severity(
        items,
        minimum=minimum_status,
        include_pass=include_pass,
    )

    if args.format == "json":
        payload = json.dumps(items_to_json_serialisable(filtered_items), indent=2)
    else:
        payload = format_backlog_markdown(filtered_items, title=args.title)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")

    exit_code = 0
    if args.fail_on_warn and any(
        item.severity is DryRunStatus.warn for item in filtered_items
    ):
        exit_code = max(exit_code, 1)
    if args.fail_on_fail and any(
        item.severity is DryRunStatus.fail for item in filtered_items
    ):
        exit_code = max(exit_code, 2)
    return exit_code


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())

