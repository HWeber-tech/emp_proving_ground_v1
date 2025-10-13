"""CLI wrapper for the compliance artifact pack builder."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

from src.operations.compliance_artifact_pack import build_compliance_artifact_pack


def _load_mapping(path: str | None) -> Mapping[str, Any] | None:
    if not path:
        return None
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {file_path}")
    with file_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError(f"Expected mapping at {file_path}, received {type(payload).__name__}")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export audit logs and compliance telemetry into an evidence pack.",
    )
    parser.add_argument(
        "--audit-log",
        type=Path,
        default=Path("data/audit_log.jsonl"),
        help="Path to the source audit log JSONL file.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/compliance"),
        help="Directory where the pack will be created (timestamped subdirectory).",
    )
    parser.add_argument(
        "--timestamp",
        help="Optional timestamp identifier (defaults to current UTC in YYYYMMDDTHHMMSSZ).",
    )
    parser.add_argument(
        "--archive",
        action="store_true",
        help="Compress the pack into a .tar.gz alongside the output directory.",
    )
    parser.add_argument(
        "--archive-path",
        type=Path,
        help="Explicit path for the generated archive (implies --archive).",
    )
    parser.add_argument(
        "--compliance-json",
        help="Optional path to a compliance readiness snapshot JSON payload to include.",
    )
    parser.add_argument(
        "--regulatory-json",
        help="Optional path to a regulatory telemetry snapshot JSON payload to include.",
    )
    parser.add_argument(
        "--metadata-json",
        help="Optional path to additional metadata merged into the manifest.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    timestamp = args.timestamp or datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    output_dir = args.output_root / timestamp

    archive_path: Path | None = None
    if args.archive_path:
        archive_path = args.archive_path
    elif args.archive:
        archive_path = output_dir.parent / f"{output_dir.name}.tar.gz"

    compliance_payload = _load_mapping(args.compliance_json)
    regulatory_payload = _load_mapping(args.regulatory_json)
    metadata_payload = _load_mapping(args.metadata_json)

    pack = build_compliance_artifact_pack(
        audit_log_path=args.audit_log,
        output_dir=output_dir,
        compliance_snapshot=compliance_payload,
        regulatory_snapshot=regulatory_payload,
        metadata=metadata_payload,
        archive_path=archive_path,
    )

    print(json.dumps(pack.as_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
