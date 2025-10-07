"""CLI helpers for decision diary exports and probe registry governance hooks."""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from src.understanding import DecisionDiaryStore, ProbeRegistry

logger = logging.getLogger(__name__)

try:  # Python < 3.11 compatibility
    from datetime import UTC  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover - compatibility fallback
    UTC = timezone.utc  # type: ignore[assignment]


def _parse_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    candidate = value.strip()
    if not candidate:
        return None
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid ISO timestamp: {value}") from exc
    return parsed.astimezone(UTC) if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _emit(text: str, output: Path | None) -> None:
    if output is None:
        print(text)
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(f"{text}\n", encoding="utf-8")


def _load_probe_registry(path: Path | None) -> ProbeRegistry | None:
    if path is None:
        return None
    if not path.exists():
        raise FileNotFoundError(f"probe registry not found at {path}")
    return ProbeRegistry.from_file(path)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Export AlphaTrade understanding loop decision diaries and probe registry"
            " metadata for governance workflows."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    diary_parser = subparsers.add_parser(
        "export-diary",
        help="Render decision diary entries as JSON or Markdown.",
    )
    diary_parser.add_argument(
        "--diary",
        required=True,
        type=Path,
        help="Path to the decision diary JSON store.",
    )
    diary_parser.add_argument(
        "--probe-registry",
        type=Path,
        help=(
            "Optional path to a probe registry JSON file. Definitions found"
            " there augment the store before exporting entries."
        ),
    )
    diary_parser.add_argument(
        "--format",
        choices=("json", "markdown"),
        default="json",
        help="Output format (defaults to json).",
    )
    diary_parser.add_argument(
        "--since",
        type=_parse_timestamp,
        help="Optional ISO timestamp filter; entries older than this are excluded.",
    )
    diary_parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="JSON indentation level (ignored for markdown).",
    )
    diary_parser.add_argument(
        "--output",
        type=Path,
        help="Optional file path for the exported payload.",
    )

    registry_parser = subparsers.add_parser(
        "export-probes",
        help="Render the probe registry for governance playbooks.",
    )
    registry_parser.add_argument(
        "--registry",
        required=True,
        type=Path,
        help="Path to the probe registry JSON file.",
    )
    registry_parser.add_argument(
        "--format",
        choices=("json", "markdown"),
        default="markdown",
        help="Output format (defaults to markdown).",
    )
    registry_parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="JSON indentation level (ignored for markdown).",
    )
    registry_parser.add_argument(
        "--output",
        type=Path,
        help="Optional path for the rendered registry payload.",
    )

    return parser


def _handle_export_diary(args: argparse.Namespace) -> int:
    diary_path: Path = args.diary
    if not diary_path.exists():
        logger.error("Decision diary not found at %s", diary_path)
        return 1
    try:
        registry = _load_probe_registry(args.probe_registry)
    except FileNotFoundError as exc:
        logger.error(str(exc))
        return 1
    try:
        store = DecisionDiaryStore(diary_path, probe_registry=registry)
    except Exception:
        logger.exception("Failed to load decision diary from %s", diary_path)
        return 1
    try:
        if args.format == "json":
            payload = store.export_json(since=args.since, indent=args.indent)
        else:
            payload = store.export_markdown(since=args.since)
    except Exception:
        logger.exception("Failed to export decision diary")
        return 1
    _emit(payload, args.output)
    return 0


def _handle_export_probes(args: argparse.Namespace) -> int:
    registry_path: Path = args.registry
    if not registry_path.exists():
        logger.error("Probe registry not found at %s", registry_path)
        return 1
    try:
        registry = ProbeRegistry.from_file(registry_path)
    except Exception:
        logger.exception("Failed to load probe registry from %s", registry_path)
        return 1
    try:
        if args.format == "json":
            payload = json.dumps(registry.as_dict(), indent=args.indent, sort_keys=True)
        else:
            payload = registry.to_markdown()
    except Exception:
        logger.exception("Failed to render probe registry")
        return 1
    _emit(payload, args.output)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "export-diary":
        return _handle_export_diary(args)
    if args.command == "export-probes":
        return _handle_export_probes(args)
    parser.error(f"unknown command: {args.command}")
    return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
