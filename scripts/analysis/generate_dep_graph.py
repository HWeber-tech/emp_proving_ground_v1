#!/usr/bin/env python3
"""
Generate a project dependency DOT graph and fanin/fanout CSV by statically parsing imports.

- Walks the src/ tree
- Parses Python AST to collect import edges between local modules
- Normalizes mixed import styles (e.g., "core.x" -> "src.core.x") if core is a local top-level package
- Writes:
  * docs/reports/dependency_graph.dot
  * docs/reports/fanin_fanout.csv

Graphviz 'dot' binary is NOT required; we only emit DOT text.
"""

from __future__ import annotations

import ast
import csv
import sys
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple


@dataclass(frozen=True)
class ModuleFile:
    path: Path
    module: str  # e.g., "src.core.event_bus"


def list_local_top_packages(src_dir: Path) -> Set[str]:
    """Top-level directories in src that contain .py files (considered project packages)."""
    top: Set[str] = set()
    if not src_dir.exists():
        return top
    for entry in src_dir.iterdir():
        if entry.is_dir():
            # consider a dir a package if it contains at least one .py
            has_py = any(p.suffix == ".py" for p in entry.rglob("*.py"))
            if has_py:
                top.add(entry.name)
    return top


def to_module_name(src_dir: Path, py_file: Path) -> Optional[str]:
    """
    Convert a file path under src_dir to a module name rooted at 'src.'.
    Examples:
      src/core/event_bus.py -> src.core.event_bus
      src/core/__init__.py  -> src.core
    """
    try:
        rel = py_file.relative_to(src_dir)
    except Exception:
        return None
    parts = list(rel.parts)
    if not parts:
        return None
    if parts[-1] == "__init__.py":
        parts = parts[:-1]
    else:
        # strip .py
        stem = Path(parts[-1]).stem
        parts = parts[:-1] + [stem]
    # prefix with "src"
    return ".".join(["src"] + parts)


def resolve_relative(curr_mod: str, module: Optional[str], level: int) -> Optional[str]:
    """
    Resolve a relative import to an absolute module.
    curr_mod: e.g., "src.sensory.organs.price_organ"
    module: e.g., "dimensions.utils" or None
    level: number of leading dots in 'from ... import'
    """
    if not curr_mod.startswith("src."):
        return None
    # base package path (exclude the current module leaf)
    base_parts = curr_mod.split(".")[:-1]  # drop leaf
    # go up "level-1" (level includes the base package)
    up = max(0, level)
    # In Python, level=1 means relative to current package (no pop), level=2 pops one, etc.
    # But mypy/CPython semantics: level indicates how many parents to traverse (1 - only current package).
    # We'll approximate: pop (level-1)
    pops = max(0, up - 1)
    base_parts = base_parts[: max(0, len(base_parts) - pops)]
    tail: List[str] = []
    if module:
        tail = [p for p in module.split(".") if p]
    resolved = ".".join(base_parts + tail)
    if not resolved:
        return None
    # ensure prefix "src."
    if not resolved.startswith("src."):
        resolved = "src." + resolved
    return resolved


def normalize_import(name: Optional[str], local_tops: Set[str]) -> Optional[str]:
    """
    Normalize an import to 'src.<top>...' if it's a local package, else return None.
    name may be None for 'from . import x'
    """
    if not name:
        return None
    n = name.replace("/", ".").replace("\\", ".")
    if n.startswith("src."):
        # ensure second token is a local top package
        parts = n.split(".")
        if len(parts) >= 2 and parts[1] in local_tops:
            return n
        # If not in local tops, consider it external
        return None
    # Handle imports like "core.x", "trading.y"
    first = n.split(".")[0]
    if first in local_tops:
        return "src." + n
    # Not a local import
    return None


def iter_python_files(src_dir: Path) -> Iterable[Path]:
    for p in src_dir.rglob("*.py"):
        # Skip __pycache__ or hidden
        if any(part == "__pycache__" for part in p.parts):
            continue
        yield p


def collect_modules(src_dir: Path) -> List[ModuleFile]:
    mods: List[ModuleFile] = []
    for f in iter_python_files(src_dir):
        m = to_module_name(src_dir, f)
        if m:
            mods.append(ModuleFile(path=f, module=m))
    return mods


def parse_imports_for_module(mf: ModuleFile, local_tops: Set[str]) -> Set[str]:
    """
    Return set of normalized imported modules (as module strings) used by this file.
    Only returns local project imports (normalized to 'src.<top>...')
    """
    imports: Set[str] = set()
    try:
        src = mf.path.read_text(encoding="utf-8", errors="ignore")
        tree = ast.parse(src)
    except Exception:
        return imports

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                nm = normalize_import(alias.name, local_tops)
                if nm:
                    imports.add(nm)
        elif isinstance(node, ast.ImportFrom):
            base: Optional[str]
            if node.level and node.level > 0:
                base = resolve_relative(mf.module, node.module, node.level)
            else:
                base = node.module
                if base and not base.startswith("src."):
                    # Try to normalize if it's local without 'src.'
                    first = base.split(".")[0] if base else ""
                    if first in local_tops:
                        base = "src." + base
            nm = normalize_import(base, local_tops)
            if nm:
                imports.add(nm)
    # Do not include self-imports
    if mf.module in imports:
        imports.discard(mf.module)
    return imports


def build_dependency_graph(src_dir: Path) -> Tuple[Dict[str, Set[str]], Set[str]]:
    """
    Build edges: module -> set(imported local modules)
    Returns (edges, nodes)
    """
    local_tops = list_local_top_packages(src_dir)
    modules = collect_modules(src_dir)
    edges: Dict[str, Set[str]] = {}
    nodes: Set[str] = set()
    for mf in modules:
        nodes.add(mf.module)
        imports = parse_imports_for_module(mf, local_tops)
        if imports:
            edges.setdefault(mf.module, set()).update(imports)
            nodes.update(imports)
        else:
            edges.setdefault(mf.module, set())
    return edges, nodes


def compute_fanin_fanout(edges: Dict[str, Set[str]]) -> Tuple[Dict[str, int], Dict[str, int]]:
    fanout: Dict[str, int] = {m: len(dsts) for m, dsts in edges.items()}
    fanin: Dict[str, int] = {m: 0 for m in edges}
    # ensure imported-only modules appear
    for dsts in edges.values():
        for d in dsts:
            if d not in fanin:
                fanin[d] = 0
    for src, dsts in edges.items():
        for d in dsts:
            fanin[d] = fanin.get(d, 0) + 1
    return fanin, fanout


def write_fanin_fanout_csv(out_csv: Path, fanin: Dict[str, int], fanout: Dict[str, int]) -> None:
    rows = []
    modules = sorted(set(fanin.keys()) | set(fanout.keys()))
    for m in modules:
        rows.append((m, fanin.get(m, 0), fanout.get(m, 0)))
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["module", "fanin", "fanout"])
        w.writerows(rows)


def write_dot(out_dot: Path, edges: Dict[str, Set[str]], nodes: Optional[Set[str]] = None) -> None:
    """
    Emit a DOT file for the dependency graph.
    """
    out_dot.parent.mkdir(parents=True, exist_ok=True)
    with out_dot.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write("digraph emp_deps {\n")
        fh.write('  rankdir=LR;\n')
        fh.write('  node [shape=box, fontsize=10];\n')
        # emit nodes explicitly (helps with isolated nodes)
        all_nodes: Set[str] = set(nodes or set(edges.keys()))
        for src, dsts in edges.items():
            all_nodes.add(src)
            all_nodes.update(dsts)
        for n in sorted(all_nodes):
            safe = n.replace('"', '\\"')
            fh.write(f'  "{safe}";\n')
        # edges
        for src, dsts in edges.items():
            s = src.replace('"', '\\"')
            for d in sorted(dsts):
                dd = d.replace('"', '\\"')
                fh.write(f'  "{s}" -> "{dd}";\n')
        fh.write("}\n")


def main(argv: Optional[List[str]] = None) -> int:
    ap = ArgumentParser(description="Generate dependency DOT and fanin/fanout CSV.")
    ap.add_argument("--src", default="src", help="Source root directory")
    ap.add_argument("--out-dot", default=str(Path("docs") / "reports" / "dependency_graph.dot"))
    ap.add_argument("--out-csv", default=str(Path("docs") / "reports" / "fanin_fanout.csv"))
    args = ap.parse_args(argv)

    src_dir = Path(args.src).resolve()
    out_dot = Path(args.out_dot).resolve()
    out_csv = Path(args.out_csv).resolve()

    edges, nodes = build_dependency_graph(src_dir)
    fanin, fanout = compute_fanin_fanout(edges)

    write_fanin_fanout_csv(out_csv, fanin, fanout)
    write_dot(out_dot, edges, nodes)

    print(f"Wrote DOT: {out_dot}")
    print(f"Wrote CSV: {out_csv}")
    print(f"Modules: {len(nodes)}, Edges: {sum(len(d) for d in edges.values())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())