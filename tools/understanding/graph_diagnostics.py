"""Render understanding loop diagnostics as JSON, Markdown, or GraphViz dot."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Sequence

from src.understanding import UnderstandingDiagnosticsBuilder


logger = logging.getLogger(__name__)
_FORMATS = ("json", "dot", "markdown")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a sensory→belief→router→policy graph diagnostic aligned with "
            "the understanding loop roadmap deliverables."
        )
    )
    parser.add_argument(
        "--format",
        choices=_FORMATS,
        default="json",
        help="Output representation (json, dot, markdown). Defaults to json.",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="JSON indentation level (ignored for other formats).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional file path for the rendered diagnostic payload.",
    )
    return parser


def _render_payload(format_: str, indent: int) -> str:
    builder = UnderstandingDiagnosticsBuilder()
    artifacts = builder.build()
    if format_ == "json":
        payload = {
            "graph": artifacts.graph.as_dict(),
            "snapshot": artifacts.to_snapshot().as_dict(),
        }
        return json.dumps(payload, indent=indent, sort_keys=True)
    if format_ == "dot":
        return artifacts.graph.to_dot()
    if format_ == "markdown":
        return artifacts.graph.to_markdown()
    raise ValueError(f"unsupported format: {format_}")


def _emit(text: str, output: Path | None) -> None:
    if output is None:
        print(text)
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(f"{text}\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        text = _render_payload(args.format, args.indent)
    except Exception:  # pragma: no cover - surfaced via non-zero exit for CLI consumers
        logger.exception("Failed to build understanding loop diagnostics")
        return 1
    _emit(text, args.output)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
