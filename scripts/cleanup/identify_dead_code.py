#!/usr/bin/env python3
"""
Phase 0.1 inventory: identify dead code (never imported/executed).
Heuristic: files not imported by any other module and not entry points.
"""

from __future__ import annotations

import os
import ast
from typing import Dict, List, Set


ROOT = "src"


def list_python_files(root: str) -> List[str]:
    files: List[str] = []
    for base, _, fns in os.walk(root):
        if ".git" in base or ".venv" in base:
            continue
        for fn in fns:
            if fn.endswith(".py"):
                files.append(os.path.join(base, fn))
    return files


def module_name(path: str) -> str:
    if path.startswith(ROOT + os.sep):
        path = path[len(ROOT) + 1 :]
    return path[:-3].replace(os.sep, ".")


def extract_imports(path: str) -> Set[str]:
    imports: Set[str] = set()
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as fh:
            tree = ast.parse(fh.read(), filename=path)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for n in node.names:
                    imports.add(n.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.add(node.module)
    except Exception:
        pass
    return imports


def find_dead(files: List[str]) -> List[str]:
    mods = {module_name(p): p for p in files}
    referenced: Set[str] = set()
    for p in files:
        for imp in extract_imports(p):
            if imp.startswith("src."):
                referenced.add(imp)
    dead: List[str] = []
    for m, p in mods.items():
        # treat __init__ as not dead, and main entry points as not dead
        if m.endswith("__init__") or os.path.basename(p) in {"main.py", "main_icmarkets.py"}:
            continue
        if m not in referenced:
            dead.append(p)
    return dead


def main() -> int:
    files = list_python_files(ROOT)
    dead = find_dead(files)
    print("Dead code candidates (first 100):")
    for p in dead[:100]:
        print("- ", p)
    print(f"Total candidates: {len(dead)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


