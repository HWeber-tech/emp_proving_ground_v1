#!/usr/bin/env python3
"""Validate RIM JSONL artifacts against interfaces/rim_types.json."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, List

from jsonschema import Draft7Validator, RefResolver

SCHEMA_PATH = Path("interfaces/rim_types.json")
DEFAULT_TARGETS = [Path("docs/examples"), Path("artifacts/rim_suggestions")]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate RIM JSONL artifacts")
    parser.add_argument("--schema", type=Path, default=SCHEMA_PATH, help="Path to schema file")
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        default=DEFAULT_TARGETS,
        help="Paths or files to validate",
    )
    return parser.parse_args()


def load_validator(schema_path: Path) -> Draft7Validator:
    schema_doc = json.loads(schema_path.read_text())
    resolver = RefResolver.from_schema(schema_doc)
    suggestion_schema = schema_doc["definitions"]["RIMSuggestion"]
    return Draft7Validator(suggestion_schema, resolver=resolver)


def iter_jsonl_files(paths: Iterable[Path]) -> Iterable[Path]:
    for path in paths:
        if not path.exists():
            continue
        if path.is_dir():
            for candidate in sorted(path.glob("*.jsonl")):
                if candidate.is_file():
                    yield candidate
        elif path.suffix == ".jsonl":
            yield path


def validate_file(path: Path, validator: Draft7Validator) -> List[str]:
    errors: List[str] = []
    with path.open() as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"{path}:{line_no} invalid json: {exc}")
                continue
            for err in validator.iter_errors(data):
                errors.append(f"{path}:{line_no} {err.message}")
    return errors


def main() -> int:
    args = parse_args()
    validator = load_validator(args.schema)
    all_errors: List[str] = []
    for jsonl_path in iter_jsonl_files(args.paths):
        all_errors.extend(validate_file(jsonl_path, validator))
    if all_errors:
        print("\n".join(all_errors), file=sys.stderr)
        return 1
    print("All RIM artifacts validated successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
