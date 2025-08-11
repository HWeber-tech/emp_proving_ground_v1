#!/usr/bin/env python3
"""
CI guard: fail if new duplicate classes/functions are introduced.

- Runs the AST duplicate scanner to refresh the report at docs/reports/duplicate_map.json
- Fails if any duplicate class groups exist
- Fails if any duplicate function groups exist excluding an allowed set (default: {"main"})
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Set


def run_scanner(root: Path, out_dir: Path, min_count: int) -> None:
    cmd = [
        sys.executable,
        "scripts/cleanup/duplicate_map.py",
        "--root",
        str(root),
        "--out",
        str(out_dir),
        "--min-count",
        str(min_count),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        if proc.stdout:
            print(proc.stdout)
        if proc.stderr:
            print(proc.stderr, file=sys.stderr)
        raise SystemExit(proc.returncode)


def load_report(out_dir: Path) -> Dict[str, Any]:
    json_path = out_dir / "duplicate_map.json"
    if not json_path.exists():
        print(f"[ERROR] Report not found: {json_path}", file=sys.stderr)
        raise SystemExit(2)
    try:
        return json.loads(json_path.read_text(encoding="utf-8"))
    except Exception as ex:
        print(f"[ERROR] Failed reading JSON report: {ex}", file=sys.stderr)
        raise SystemExit(2)


def count_class_groups(report: Dict[str, Any]) -> int:
    classes = report.get("classes", {})
    return len(classes)


def disallowed_function_groups(report: Dict[str, Any], allowed: Set[str]) -> Dict[str, Any]:
    functions = report.get("functions", {})
    return {name: meta for name, meta in functions.items() if name not in allowed}


def summarize_failures(classes: Dict[str, Any], disallowed_funcs: Dict[str, Any]) -> None:
    print("== Duplicate Summary ==", file=sys.stderr)
    if classes:
        print(f"- Duplicate class groups: {len(classes)}", file=sys.stderr)
        # classes is mapping of name->{...}
        for name, data in sorted(classes.items(), key=lambda kv: -kv[1].get("count", 0))[:20]:
            print(f"    * {name}: {data.get('count', 0)}", file=sys.stderr)
    if disallowed_funcs:
        print(f"- Duplicate function groups (disallowed): {len(disallowed_funcs)}", file=sys.stderr)
        for name, data in sorted(disallowed_funcs.items(), key=lambda kv: -kv[1].get("count", 0))[:20]:
            print(f"    * {name}: {data.get('count', 0)}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fail CI if new duplicate classes/functions are introduced.")
    parser.add_argument("--root", default="src", help="Root directory to scan (default: src)")
    parser.add_argument("--out", default="docs/reports", help="Output directory for reports (default: docs/reports)")
    parser.add_argument("--min-count", type=int, default=2, help="Duplicates threshold for grouping (default: 2)")
    parser.add_argument(
        "--allow-func",
        action="append",
        default=["main"],
        help="Allow-listed duplicate function names (can be provided multiple times). Default: main",
    )
    parser.add_argument("--skip-run", action="store_true", help="Skip running the scanner (use existing report)")
    args = parser.parse_args(argv)

    root = Path(args.root).resolve()
    out_dir = Path(args.out)

    if not args.skip_run:
        run_scanner(root, out_dir, args.min_count)

    report = load_report(out_dir)
    classes_map = report.get("classes", {})
    class_groups = len(classes_map)
    disallowed_funcs = disallowed_function_groups(report, allowed=set(args.allow_func))

    if class_groups > 0 or disallowed_funcs:
        summarize_failures(classes_map, disallowed_funcs)
        print(
            f"[FAIL] Duplicate groups detected: classes={class_groups}, disallowed_functions={len(disallowed_funcs)}",
            file=sys.stderr,
        )
        return 1

    print("[PASS] No duplicate class groups. Function duplicates only in allow-list.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())