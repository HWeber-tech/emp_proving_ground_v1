#!/usr/bin/env python3
"""
Phase 0.1 inventory: dependency analysis (imports graph, circulars, orphans).
"""

from __future__ import annotations

import ast
import os
from collections import defaultdict
from typing import Dict, List, Set

ROOT = "src"

# Exclude legacy/compat shims from orphan detection counts to reduce noise
EXCLUDE_PREFIXES = {
    "src.trading.strategy_engine",
    "src.trading.risk_management",
    "src.trading.risk",
    "src.evolution.engine",
}


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


def build_graph(files: List[str]) -> Dict[str, Set[str]]:
    graph: Dict[str, Set[str]] = defaultdict(set)
    name_for_path: Dict[str, str] = {}
    for p in files:
        name_for_path[p] = module_name(p)
    for p in files:
        mod = name_for_path[p]
        for imp in extract_imports(p):
            if imp.startswith("src."):
                graph[mod].add(imp)
    return graph


def find_circular(graph: Dict[str, Set[str]]) -> List[List[str]]:
    cycles: List[List[str]] = []
    temp: Set[str] = set()
    perm: Set[str] = set()
    stack: List[str] = []

    def visit(n: str):
        if n in perm:
            return
        if n in temp:
            try:
                i = stack.index(n)
                cycles.append(stack[i:] + [n])
            except ValueError:
                pass
            return
        temp.add(n)
        stack.append(n)
        for m in graph.get(n, set()):
            visit(m)
        temp.remove(n)
        perm.add(n)
        stack.pop()

    for n in list(graph.keys()):
        if n not in perm:
            visit(n)
    return cycles


def find_orphans(graph: Dict[str, Set[str]]) -> Set[str]:
    referenced: Set[str] = set()
    for n, outs in graph.items():
        referenced.update(outs)
    all_nodes = set(graph.keys()) | referenced
    orphans = {n for n in all_nodes if all(n not in outs for outs in graph.values())}
    # Keep top-level packages
    filtered = {o for o in orphans if not o.endswith("__init__")}

    # Filter legacy shims
    def _excluded(name: str) -> bool:
        try:
            n = name if name.startswith("src.") else f"src.{name}"
            return any(n.startswith(prefix) for prefix in EXCLUDE_PREFIXES)
        except Exception:
            return False

    return {o for o in filtered if not _excluded(o)}


def main() -> int:
    files = list_python_files(ROOT)
    graph = build_graph(files)
    cycles = find_circular(graph)
    orphans = find_orphans(graph)

    print("Dependency Analysis")
    print(f"- modules: {len(graph)}")
    print(f"- circulars: {len(cycles)}")
    for c in cycles[:10]:
        print("  * ", " -> ".join(c))
    print(f"- orphans: {len(orphans)} (first 20)")
    for o in list(orphans)[:20]:
        print("  * ", o)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
