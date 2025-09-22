#!/usr/bin/env python3
"""
Phase 0.1 inventory: duplicate functionality audit.

Find likely duplicate implementations by scanning class/function names and
grouping them by normalized names and domain-specific keywords.
"""

from __future__ import annotations

import ast
import os
import re
from collections import defaultdict
from typing import Dict, List, Set, Tuple

ROOTS_TO_SCAN = ["src", "tests"]
NAME_NORMALIZER = re.compile(r"[^a-z0-9]+")


def list_python_files(roots: List[str]) -> List[str]:
    files: List[str] = []
    for root in roots:
        for base, _, fns in os.walk(root):
            if ".git" in base or ".venv" in base:
                continue
            for fn in fns:
                if fn.endswith(".py"):
                    files.append(os.path.join(base, fn))
    return files


def normalize_name(name: str) -> str:
    return NAME_NORMALIZER.sub("_", name.lower()).strip("_")


def extract_defs(path: str) -> Tuple[Set[str], Set[str]]:
    classes: Set[str] = set()
    functions: Set[str] = set()
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as fh:
            tree = ast.parse(fh.read(), filename=path)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                classes.add(node.name)
            elif isinstance(node, ast.FunctionDef):
                functions.add(node.name)
    except Exception:
        pass
    return classes, functions


def categorize(name: str) -> str:
    n = name.lower()
    if any(k in n for k in ["strategy", "engine", "executor"]):
        return "strategy"
    if any(k in n for k in ["risk", "var", "sizing"]):
        return "risk"
    if any(k in n for k in ["evolution", "genetic", "ga", "population", "fitness"]):
        return "evolution"
    if any(k in n for k in ["fitness", "score", "evaluator"]):
        return "fitness"
    return "other"


def find_duplicates(files: List[str]) -> Dict[str, Dict[str, List[str]]]:
    class_to_files: Dict[str, List[str]] = defaultdict(list)
    func_to_files: Dict[str, List[str]] = defaultdict(list)
    for path in files:
        classes, functions = extract_defs(path)
        for c in classes:
            class_to_files[normalize_name(c)].append(path)
        for f in functions:
            func_to_files[normalize_name(f)].append(path)

    dups: Dict[str, Dict[str, List[str]]] = {
        "classes": {k: v for k, v in class_to_files.items() if len(set(v)) > 1},
        "functions": {k: v for k, v in func_to_files.items() if len(set(v)) > 1},
    }
    return dups


def summarize_by_category(dups: Dict[str, Dict[str, List[str]]]) -> Dict[str, Dict[str, int]]:
    summary: Dict[str, Dict[str, int]] = {
        "classes": defaultdict(int),
        "functions": defaultdict(int),
    }
    for name, paths in dups.get("classes", {}).items():
        summary["classes"][categorize(name)] += 1
    for name, paths in dups.get("functions", {}).items():
        summary["functions"][categorize(name)] += 1
    return {k: dict(v) for k, v in summary.items()}


def main() -> int:
    files = list_python_files(ROOTS_TO_SCAN)
    dups = find_duplicates(files)
    summary = summarize_by_category(dups)

    print("Duplicate Definitions Summary")
    for kind in ("classes", "functions"):
        print(f"- {kind}:")
        for cat, count in sorted(summary[kind].items()):
            print(f"  - {cat}: {count}")

    print("\nTop duplicates (classes):")
    for name, paths in list(dups["classes"].items())[:20]:
        unique_paths = sorted(set(paths))
        print(f"- {name}: {len(unique_paths)} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
