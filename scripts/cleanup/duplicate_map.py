#!/usr/bin/env python3
"""
AST-based duplicate scanner.
Scans a source tree for duplicate class and function names across files,
groups occurrences, classifies by domain inferred from path segments,
and emits CSV and JSON reports suitable for planning canonicalization.

Usage:
  python scripts/cleanup/duplicate_map.py --root src --out docs/reports --min-count 2
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


@dataclass(frozen=True)
class Occurrence:
    name: str  # definition name
    kind: str  # "class" or "function"
    path: str  # POSIX-style relative path
    line: int  # 1-based line number
    domain: str  # inferred domain (e.g., "risk", "evolution", "strategy")


DEFAULT_EXCLUDE_DIRS: Set[str] = {
    ".git",
    ".idea",
    ".vscode",
    ".venv",
    "venv",
    "env",
    "__pycache__",
    "node_modules",
    "dist",
    "build",
    "backup",
    "reports",
    "mlruns",
}

DOMAIN_KEYWORDS_ORDERED = [
    "core",
    "risk",
    "strategy",
    "evolution",
    "sensory",
    "operational",
    "governance",
    "trading",
    "validation",
    "ecosystem",
    "genome",
    "integration",
    "understanding",
    "performance",
    "portfolio",
    "data_foundation",
    "data_integration",
    "data_sources",
    "simulation",
    "thinking",
    "sentient",
    "ui",
]

DOMAIN_KEYWORD_ALIASES: Dict[str, str] = {
    "intelligence": "understanding",
    "market_intelligence": "sensory",
}


def to_posix(path: Path) -> str:
    try:
        rel = path.relative_to(Path.cwd())
    except Exception:
        rel = path
    return rel.as_posix()


def classify_domain(file_path: Path) -> str:
    parts = [p.lower() for p in file_path.parts]
    parts_set = set(parts)
    # Prefer explicit keywords anywhere in the path
    for kw in DOMAIN_KEYWORDS_ORDERED:
        if kw in parts_set:
            return kw
    for legacy, target in DOMAIN_KEYWORD_ALIASES.items():
        if legacy in parts_set:
            return target
    # Fallback: use first segment under "src"
    if "src" in parts:
        try:
            idx = parts.index("src")
            if idx + 1 < len(parts):
                candidate = parts[idx + 1]
                return DOMAIN_KEYWORD_ALIASES.get(candidate, candidate)
        except ValueError:
            pass
    return "other"


def should_skip(path: Path, include_tests: bool, exclude_dirs: Set[str]) -> bool:
    parts_lower = [p.lower() for p in path.parts]
    for seg in parts_lower:
        if seg in exclude_dirs:
            return True
    if not include_tests:
        if "tests" in parts_lower or any(seg.startswith("test") for seg in parts_lower):
            return True
    return False


def parse_python(source: str, filename: str) -> Optional[ast.AST]:
    try:
        return ast.parse(source, filename=filename)
    except SyntaxError as ex:
        print(f"[WARN] SyntaxError in {filename}: {ex}", file=sys.stderr)
        return None


def extract_defs(
    tree: ast.AST, file_path: Path, domain: str, ignore_private: bool
) -> Tuple[List[Occurrence], List[Occurrence]]:
    classes: List[Occurrence] = []
    functions: List[Occurrence] = []
    # Only module-level defs (no nested methods/functions)
    for node in getattr(tree, "body", []):
        if isinstance(node, ast.ClassDef):
            name = node.name
            if ignore_private and name.startswith("_"):
                continue
            classes.append(
                Occurrence(
                    name=name,
                    kind="class",
                    path=to_posix(file_path),
                    line=getattr(node, "lineno", 1),
                    domain=domain,
                )
            )
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            name = node.name
            if ignore_private and name.startswith("_"):
                continue
            functions.append(
                Occurrence(
                    name=name,
                    kind="function",
                    path=to_posix(file_path),
                    line=getattr(node, "lineno", 1),
                    domain=domain,
                )
            )
    return classes, functions


def scan_root(
    root: Path,
    include_tests: bool = False,
    exclude_dirs: Optional[Set[str]] = None,
    ignore_private: bool = True,
) -> Tuple[Dict[str, List[Occurrence]], Dict[str, List[Occurrence]]]:
    exclude = set(d.lower() for d in (exclude_dirs or set())) | set(
        d.lower() for d in DEFAULT_EXCLUDE_DIRS
    )
    class_map: Dict[str, List[Occurrence]] = defaultdict(list)
    func_map: Dict[str, List[Occurrence]] = defaultdict(list)

    for pyfile in root.rglob("*.py"):
        if should_skip(pyfile, include_tests=include_tests, exclude_dirs=exclude):
            continue
        try:
            text = pyfile.read_text(encoding="utf-8", errors="ignore")
        except Exception as ex:
            print(f"[WARN] Failed reading {pyfile}: {ex}", file=sys.stderr)
            continue

        tree = parse_python(text, filename=str(pyfile))
        if not tree:
            continue

        domain = classify_domain(pyfile)
        classes, functions = extract_defs(tree, pyfile, domain, ignore_private)
        for occ in classes:
            class_map[occ.name].append(occ)
        for occ in functions:
            func_map[occ.name].append(occ)

    return class_map, func_map


def filter_duplicates(
    map_: Dict[str, List[Occurrence]], min_count: int
) -> Dict[str, List[Occurrence]]:
    return {name: occs for name, occs in map_.items() if len(occs) >= min_count}


def ensure_out_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def unique_domains(occurrences: List[Occurrence]) -> List[str]:
    return sorted({o.domain for o in occurrences})


def primary_domain(occurrences: List[Occurrence]) -> str:
    cnt = Counter(o.domain for o in occurrences)
    if not cnt:
        return "other"
    # tie-breaker: alphabetical on domain name
    max_count = max(cnt.values())
    candidates = sorted([d for d, c in cnt.items() if c == max_count])
    return candidates[0]


def write_csv(path: Path, duplicates: Dict[str, List[Occurrence]]) -> None:
    rows = []
    for name, occs in duplicates.items():
        rows.append(
            (
                name,
                len(occs),
                ";".join(unique_domains(occs)),
                ";".join(o.path for o in occs),
                ";".join(str(o.line) for o in occs),
            )
        )
    # Sort by count desc, then name asc
    rows.sort(key=lambda r: (-r[1], r[0]))
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "count", "domains", "files", "lines"])
        for row in rows:
            writer.writerow(row)


def top_n(duplicates: Dict[str, List[Occurrence]], n: int) -> List[Tuple[str, int]]:
    pairs = [(name, len(occs)) for name, occs in duplicates.items()]
    pairs.sort(key=lambda p: (-p[1], p[0]))
    return pairs[:n]


def build_summary(duplicates: Dict[str, List[Occurrence]]) -> Dict[str, object]:
    # Count each duplicate group against its primary domain
    domain_counter: Counter[str] = Counter()
    for name, occs in duplicates.items():
        domain_counter[primary_domain(occs)] += 1
    return {
        "groups": len(duplicates),
        "by_domain": dict(sorted(domain_counter.items(), key=lambda kv: (-kv[1], kv[0]))),
    }


def write_json(
    path: Path,
    classes_dup: Dict[str, List[Occurrence]],
    functions_dup: Dict[str, List[Occurrence]],
    root: Path,
    min_count: int,
    top: int,
) -> None:
    payload = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "root": to_posix(root),
        "min_count": min_count,
        "summary": {
            "classes": {
                "total_duplicate_groups": len(classes_dup),
                **build_summary(classes_dup),
                "top": [{"name": n, "count": c} for n, c in top_n(classes_dup, top)],
            },
            "functions": {
                "total_duplicate_groups": len(functions_dup),
                **build_summary(functions_dup),
                "top": [{"name": n, "count": c} for n, c in top_n(functions_dup, top)],
            },
        },
        "classes": {
            name: {
                "count": len(occs),
                "domains": unique_domains(occs),
                "occurrences": [{"path": o.path, "line": o.line, "domain": o.domain} for o in occs],
            }
            for name, occs in classes_dup.items()
        },
        "functions": {
            name: {
                "count": len(occs),
                "domains": unique_domains(occs),
                "occurrences": [{"path": o.path, "line": o.line, "domain": o.domain} for o in occs],
            }
            for name, occs in functions_dup.items()
        },
    }
    # Avoid file churn: if only the timestamp differs, do not rewrite the file
    try:
        if path.exists():
            existing = json.loads(path.read_text(encoding="utf-8"))
            comparable = dict(payload)
            comparable["generated_at"] = existing.get("generated_at", comparable["generated_at"])
            if comparable == existing:
                # Content equal (ignoring timestamp) -> keep existing file unchanged
                return
    except Exception:
        # If we can't read/parse the existing file, proceed to write a fresh one
        pass
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Scan for duplicate class/function names and emit CSV/JSON reports."
    )
    parser.add_argument("--root", default="src", help="Root directory to scan (default: src)")
    parser.add_argument(
        "--out", default="docs/reports", help="Output directory for reports (default: docs/reports)"
    )
    parser.add_argument(
        "--min-count", type=int, default=2, help="Minimum duplicates threshold (default: 2)"
    )
    parser.add_argument(
        "--include-tests", action="store_true", help="Include tests directories (default: False)"
    )
    parser.add_argument(
        "--include-private",
        action="store_true",
        help="Include private names (starting with _) (default: False)",
    )
    parser.add_argument(
        "--top", type=int, default=20, help="Top N names to include in JSON summary (default: 20)"
    )
    args = parser.parse_args(argv)

    root = Path(args.root).resolve()
    out_dir = Path(args.out)
    ensure_out_dir(out_dir)

    ignore_private = not args.include_private

    class_map, func_map = scan_root(
        root=root,
        include_tests=args.include_tests,
        exclude_dirs=DEFAULT_EXCLUDE_DIRS,
        ignore_private=ignore_private,
    )

    classes_dup = filter_duplicates(class_map, args.min_count)
    functions_dup = filter_duplicates(func_map, args.min_count)

    classes_csv = out_dir / "duplicate_map_classes.csv"
    functions_csv = out_dir / "duplicate_map_functions.csv"
    json_path = out_dir / "duplicate_map.json"

    write_csv(classes_csv, classes_dup)
    write_csv(functions_csv, functions_dup)
    write_json(json_path, classes_dup, functions_dup, root, args.min_count, args.top)

    # Console summary
    print("Duplicate scan complete.")
    print(f"  Root: {to_posix(root)}")
    print(f"  Min count: {args.min_count}")
    print(f"  Classes duplicate groups: {len(classes_dup)}")
    print(f"  Functions duplicate groups: {len(functions_dup)}")
    topc = top_n(classes_dup, min(args.top, 10))
    topf = top_n(functions_dup, min(args.top, 10))
    if topc:
        print("  Top class duplicates:")
        for name, cnt in topc:
            print(f"    - {name}: {cnt}")
    if topf:
        print("  Top function duplicates:")
        for name, cnt in topf:
            print(f"    - {name}: {cnt}")
    print(f"  Wrote: {to_posix(classes_csv)}, {to_posix(functions_csv)}, {to_posix(json_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
