"""Run the TRM milestone exit drill and emit documentation artefacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from src.operations.trm_exit_drill import (
    TRMDrillStatus,
    run_trm_exit_drill,
)


DEFAULT_DIARIES = Path("docs/examples/trm_exit_drill_diaries.jsonl")
DEFAULT_SCHEMA = Path("interfaces/rim_types.json")
DEFAULT_OUTPUT = Path("docs/reflection/trm_exit_drill.md")


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Execute the TRM exit drill and record the results.",
    )
    parser.add_argument(
        "--diaries",
        type=Path,
        default=DEFAULT_DIARIES,
        help="Decision diary JSONL file to feed into the drill.",
    )
    parser.add_argument(
        "--schema",
        type=Path,
        default=DEFAULT_SCHEMA,
        help="Path to the RIM suggestion schema.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Optional override for the RIM configuration file.",
    )
    parser.add_argument(
        "--publish-dir",
        type=Path,
        help="Directory where suggestion artefacts should be written.",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        help="Directory where telemetry logs should be written.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Destination Markdown report path.",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        help="Optional JSON payload destination for automation.",
    )
    parser.add_argument(
        "--artifact-output",
        type=Path,
        help="Optional path to copy the generated suggestion artefact.",
    )
    return parser.parse_args(argv)


def _write_markdown(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)

    report = run_trm_exit_drill(
        diaries_path=args.diaries,
        schema_path=args.schema,
        config_path=args.config,
        publish_dir=args.publish_dir,
        log_dir=args.log_dir,
    )

    _write_markdown(args.output, report.to_markdown())

    if args.artifact_output and report.suggestion_artifact and report.suggestion_artifact.exists():
        args.artifact_output.parent.mkdir(parents=True, exist_ok=True)
        args.artifact_output.write_text(
            report.suggestion_artifact.read_text(encoding="utf-8"),
            encoding="utf-8",
        )

    if args.json_output:
        payload: dict[str, object] = {
            "status": report.status.value,
            "generated_at": report.generated_at.isoformat(),
            "components": [component.as_dict() for component in report.components],
            "metrics": report.metrics.as_dict(),
            "diaries_path": report.diaries_path.as_posix(),
            "suggestion_artifact": (
                report.suggestion_artifact.as_posix()
                if report.suggestion_artifact is not None
                else None
            ),
            "telemetry_log": (
                report.telemetry_log.as_posix()
                if report.telemetry_log is not None
                else None
            ),
            "config_path": report.config_path.as_posix(),
            "schema_path": report.schema_path.as_posix(),
        }
        _write_json(args.json_output, payload)

    print(
        "[TRM] Exit drill complete:",
        report.status.value,
        "suggestions",
        report.metrics.suggestion_count,
    )

    return 0 if report.status is not TRMDrillStatus.FAIL else 1


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())

