#!/usr/bin/env python3
"""
Mapping Hit-Count Logger

- Runs the import rewriter in strict dry-run mode to capture per-mapping hit counts.
- Joins mapping ids with entries from docs/development/import_rewrite_map.yaml.
- Appends a CSV row per mapping id with hits to docs/reports/imports_mapping_hits.csv.
- Emits a synthetic zero row when no hits are present to show a timestamped heartbeat.

CSV columns:
  timestamp_iso, mapping_id, sources_json, target_module, symbols_json, hit_count
"""

from __future__ import annotations

import csv
import datetime as dt
import json
import os
import re
import subprocess
import sys
from typing import Any, Dict, List, Tuple

REWRITER_CMD = [
    sys.executable,
    "scripts/cleanup/rewrite_imports.py",
    "--root",
    "src",
    "--map",
    "docs/development/import_rewrite_map.yaml",
    "--dry-run",
    "--strict",
    "--verbose",
]
MAP_PATH = os.path.join("docs", "development", "import_rewrite_map.yaml")
CSV_DIR = os.path.join("docs", "reports")
CSV_PATH = os.path.join(CSV_DIR, "imports_mapping_hits.csv")


def load_mapping(path: str) -> List[Dict[str, Any]]:
    """Load mapping list from map file (JSON-first, YAML fallback)."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
    except FileNotFoundError:
        return []
    # JSON-first
    try:
        data = json.loads(content)
        if isinstance(data, dict) and isinstance(data.get("mappings"), list):
            return data["mappings"]  # type: ignore[return-value]
    except json.JSONDecodeError:
        pass

    # YAML fallback
    try:
        import yaml  # type: ignore
    except Exception:
        return []
    try:
        data = yaml.safe_load(content)  # type: ignore
        if isinstance(data, dict) and isinstance(data.get("mappings"), list):
            return data["mappings"]  # type: ignore[return-value]
    except Exception:
        return []
    return []


def run_rewriter() -> Tuple[int, str, str]:
    """Execute the rewriter; always return (returncode, stdout, stderr)."""
    try:
        proc = subprocess.run(
            REWRITER_CMD,
            capture_output=True,
            text=True,
            check=False,
        )
        return proc.returncode, proc.stdout, proc.stderr
    except Exception as ex:
        # Return a synthetic failure with empty output; script remains non-fatal.
        return 1, "", f"Failed to run rewriter: {ex}"


def parse_hits(stdout: str) -> Dict[int, int]:
    """
    Parse rewriter stdout for per-mapping hit counts.

    Expected section:
      Per-mapping hit counts:
        mapping[5]: 3
        mapping[12]: 1
    Or the line:
      Per-mapping hit counts: none
    """
    hits: Dict[int, int] = {}
    lines = stdout.splitlines()
    # Locate the "Per-mapping hit counts:" section
    section_start = None
    for i, line in enumerate(lines):
        if line.strip().startswith("Per-mapping hit counts:"):
            section_start = i
            break
    if section_start is None:
        return hits

    # If explicitly "none", return empty
    if "none" in lines[section_start].lower():
        return hits

    # Parse subsequent mapping lines until hitting a blank line or a new section
    mapping_re = re.compile(r"^\s*mapping\[(\d+)\]:\s+(\d+)\s*$")
    for j in range(section_start + 1, len(lines)):
        ln = lines[j]
        if not ln.strip():
            break
        m = mapping_re.match(ln)
        if not m:
            # stop when the section ends
            if ln.strip().endswith(":") and "Unresolved" in ln:
                break
            continue
        mid = int(m.group(1))
        cnt = int(m.group(2))
        hits[mid] = hits.get(mid, 0) + cnt
    return hits


def ensure_csv_header(path: str) -> None:
    if not os.path.exists(CSV_DIR):
        os.makedirs(CSV_DIR, exist_ok=True)
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "timestamp_iso",
                    "mapping_id",
                    "sources_json",
                    "target_module",
                    "symbols_json",
                    "hit_count",
                ]
            )


def to_json(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    except Exception:
        return "null"


def main() -> int:
    rc, out, err = run_rewriter()

    if err:
        # Emit stderr to help diagnosis in CI logs (non-fatal)
        print(err, file=sys.stderr)

    hits = parse_hits(out)
    mappings = load_mapping(MAP_PATH)
    ensure_csv_header(CSV_PATH)

    timestamp = dt.datetime.utcnow().isoformat(timespec="seconds") + "Z"

    rows: List[List[str]] = []
    if hits:
        for mid, cnt in sorted(hits.items()):
            entry: Dict[str, Any] = {}
            if 0 <= mid < len(mappings):
                raw = mappings[mid]
                if isinstance(raw, dict):
                    entry = raw
            sources = entry.get("sources", [])
            target = entry.get("target_module", "") or ""
            symbols = entry.get("symbols", [])
            rows.append(
                [
                    timestamp,
                    str(mid),
                    to_json(sources),
                    target,
                    to_json(symbols),
                    str(cnt),
                ]
            )
    else:
        # Synthetic zero-row heartbeat when no hits present
        rows.append(
            [
                timestamp,
                str(-1),
                "[]",
                "",
                "[]",
                str(0),
            ]
        )

    with open(CSV_PATH, "a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    # Do not fail CI even if the rewriter returned non-zero; we still logged data.
    return 0


if __name__ == "__main__":
    sys.exit(main())