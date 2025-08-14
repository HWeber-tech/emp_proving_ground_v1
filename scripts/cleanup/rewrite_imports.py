#!/usr/bin/env python3
"""
AST-based Import Rewrite Utility

Behavior summary:
- Walk *.py under a configurable root (default: src).
- Skip dirs: .git, .idea, .vscode, .venv, venv, env, __pycache__, node_modules, dist, build, backup, reports, mlruns, docs (unless --include-docs).
- Parse files with ast to locate Import and ImportFrom nodes.
- Load mapping from docs/development/import_rewrite_map.yaml (JSON-first, fallback to YAML if PyYAML installed).
- Apply declarative mapping rules:
  * ImportFrom:
      - If module matches mapping "sources":
          - If symbols == ["*"], rewrite module to target_module (respect aliasing).
          - Else rewrite only matched symbols; support rename old→new; if preserve_alias and no explicit asname, rewrite to "new as old".
          - If mixed mapped/unmapped, split into multiple ImportFrom lines per target_module and one for any remaining unmapped under original module.
      - Star-import from a legacy module without wildcard mapping: warn and honor --strict (report and exit non-zero).
  * Import:
      - If alias.name matches mapping with symbols ["*"], rewrite to "import target_module as alias" (preserve alias if present).
      - If mapping is symbol-level only (e.g., performance:get_global_cache), do NOT rewrite plain "import performance"; warn and honor --strict.
- Preserve aliasing (as x).
- Atomic writes with sibling .orig backup unless --no-backup. Use temp file then replace original.
- Emit summary: changed files, per-mapping hit counts, unresolved star-imports and unresolved legacy imports.
- Exit code: 0 on success; non-zero if --strict and unresolved items or on errors.

CLI:
  python scripts/cleanup/rewrite_imports.py --root src --map docs/development/import_rewrite_map.yaml --dry-run --strict --exclude "src/**/generated/*.py" --include-docs --no-backup --verbose
"""

from __future__ import annotations

import argparse
import ast
import fnmatch
import io
import json
import os
import re
import shutil
import sys
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

SKIP_DIRS: Set[str] = {
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

DEFAULT_ROOT = "src"
DEFAULT_MAP = os.path.join("docs", "development", "import_rewrite_map.yaml")


@dataclass(frozen=True)
class MappingEntry:
    id: int
    sources: Tuple[str, ...]
    target_module: str
    symbols: Tuple[str, ...]
    rename: Dict[str, str]
    preserve_alias: bool
    label: str  # human-friendly label for reporting


@dataclass
class MappingIndex:
    by_source: Dict[str, List[MappingEntry]]
    star_by_source: Dict[str, MappingEntry]
    all_sources: Set[str]
    entries: List[MappingEntry]


@dataclass
class FileEdit:
    start_line: int  # 1-based inclusive
    end_line: int    # 1-based inclusive
    new_text: str    # text to replace the [start_line-1:end_line] block


@dataclass
class Summary:
    changed_files: List[str]
    per_mapping_hits: Dict[int, int]
    unresolved_star_imports: List[str]          # messages "file:line module"
    unresolved_legacy_plain_imports: List[str]  # messages "file:line module"
    warnings: List[str]


def eprint(*args: Any, **kwargs: Any) -> None:
    print(*args, file=sys.stderr, **kwargs)


def load_mapping(map_path: str) -> Dict[str, Any]:
    """
    Load mapping file with JSON-first strategy; fallback to YAML if available.
    If both fail, prints a helpful error and exits non-zero.
    """
    try:
        with io.open(map_path, "r", encoding="utf-8") as f:
            content = f.read()
    except FileNotFoundError:
        eprint(f"[ERROR] Mapping file not found: {map_path}")
        sys.exit(2)
    except OSError as ex:
        eprint(f"[ERROR] Failed to read mapping file '{map_path}': {ex}")
        sys.exit(2)

    # JSON-first
    try:
        data = json.loads(content)
        return data
    except json.JSONDecodeError:
        pass

    # YAML fallback (PyYAML optional)
    try:
        import yaml  # type: ignore
    except Exception:
        eprint(
            "[ERROR] Mapping file is not valid JSON. "
            "Either provide valid JSON content or install PyYAML to parse YAML.\n"
            "Tip: The repo uses JSON-as-YAML, so valid JSON is also valid YAML."
        )
        sys.exit(2)

    try:
        data = yaml.safe_load(content)  # type: ignore
        return data
    except Exception as ex:
        eprint(f"[ERROR] Failed to parse mapping as YAML: {ex}")
        sys.exit(2)


def build_mapping_index(mapping: Dict[str, Any]) -> MappingIndex:
    """
    Build index structures for fast lookup:
      - by_source: source module -> list of MappingEntry
      - star_by_source: source module -> MappingEntry whose symbols == ["*"]
      - all_sources: set of all source module names
    """
    if not isinstance(mapping, dict) or "mappings" not in mapping:
        eprint("[ERROR] Mapping file missing required 'mappings' key.")
        sys.exit(2)

    entries: List[MappingEntry] = []
    by_source: Dict[str, List[MappingEntry]] = {}
    star_by_source: Dict[str, MappingEntry] = {}
    all_sources: Set[str] = set()

    raw_entries = mapping.get("mappings", [])
    if not isinstance(raw_entries, list):
        eprint("[ERROR] Mapping 'mappings' must be a list.")
        sys.exit(2)

    for idx, m in enumerate(raw_entries):
        if not isinstance(m, dict):
            eprint(f"[ERROR] Invalid mapping entry at index {idx}: expected object.")
            sys.exit(2)
        sources = tuple(m.get("sources", []) or [])
        target_module = m.get("target_module")
        symbols = tuple(m.get("symbols", []) or [])
        rename = dict(m.get("rename", {}) or {})
        preserve_alias = bool(m.get("preserve_alias", False))

        if not sources or not isinstance(sources, tuple):
            eprint(f"[ERROR] Mapping entry {idx} missing/invalid 'sources'.")
            sys.exit(2)
        if not target_module or not isinstance(target_module, str):
            eprint(f"[ERROR] Mapping entry {idx} missing/invalid 'target_module'.")
            sys.exit(2)
        if not symbols or not isinstance(symbols, tuple):
            eprint(f"[ERROR] Mapping entry {idx} missing/invalid 'symbols'.")
            sys.exit(2)

        label = f"{'|'.join(sources)} -> {target_module} [{','.join(symbols)}]"
        entry = MappingEntry(
            id=idx,
            sources=sources,
            target_module=target_module,
            symbols=symbols,
            rename=rename,
            preserve_alias=preserve_alias,
            label=label,
        )
        entries.append(entry)

        for src in sources:
            all_sources.add(src)
            by_source.setdefault(src, []).append(entry)
            if len(symbols) == 1 and symbols[0] == "*":
                # Prefer first-declared wildcard if duplicates exist
                star_by_source.setdefault(src, entry)

    return MappingIndex(by_source=by_source, star_by_source=star_by_source, all_sources=all_sources, entries=entries)


def should_skip_dir(dirname: str, include_docs: bool) -> bool:
    if dirname in SKIP_DIRS:
        return True
    if dirname == "docs" and not include_docs:
        return True
    return False


def is_python_file(path: str) -> bool:
    return path.endswith(".py")


def to_posix(path: str) -> str:
    return path.replace("\\", "/")


def rel_to_repo(path: str) -> str:
    try:
        return to_posix(os.path.relpath(path, start=os.getcwd()))
    except Exception:
        return to_posix(path)


def match_any_exclude(rel_path: str, excludes: Sequence[str]) -> bool:
    # Use POSIX-style matching for globs
    posix = to_posix(rel_path)
    for pat in excludes:
        if fnmatch.fnmatch(posix, pat):
            return True
    return False


def get_indent(line: str) -> str:
    m = re.match(r"[ \t]*", line)
    return m.group(0) if m else ""


def compose_import_from(module: str, items: List[Tuple[str, Optional[str]]], indent: str) -> str:
    # items: list of (name, asname)
    parts = []
    for name, asname in items:
        if asname:
            parts.append(f"{name} as {asname}")
        else:
            parts.append(name)
    return f"{indent}from {module} import {', '.join(parts)}"


def compose_import(module: str, asname: Optional[str], indent: str) -> str:
    if asname:
        return f"{indent}import {module} as {asname}"
    return f"{indent}import {module}"


def analyze_and_rewrite_file(
    file_path: str,
    index: MappingIndex,
    per_mapping_hits: Dict[int, int],
    strict: bool,
    excludes: Sequence[str],
    verbose: bool,
) -> Tuple[Optional[str], List[FileEdit], List[str], List[str], List[str]]:
    """
    Returns:
      - new_text or None if unchanged
      - edits list (applied bottom-to-top)
      - unresolved_star_warnings
      - unresolved_plain_warnings
      - general_warnings
    """
    unresolved_star_warnings: List[str] = []
    unresolved_plain_warnings: List[str] = []
    general_warnings: List[str] = []

    try:
        with io.open(file_path, "r", encoding="utf-8") as f:
            original_text = f.read()
    except Exception as ex:
        general_warnings.append(f"[WARN] Failed to read {rel_to_repo(file_path)}: {ex}")
        return None, [], unresolved_star_warnings, unresolved_plain_warnings, general_warnings

    try:
        module_ast = ast.parse(original_text)
    except SyntaxError as ex:
        general_warnings.append(
            f"[WARN] SyntaxError in {rel_to_repo(file_path)} at line {getattr(ex, 'lineno', '?')}: {ex.msg}"
        )
        return None, [], unresolved_star_warnings, unresolved_plain_warnings, general_warnings
    except Exception as ex:
        general_warnings.append(
            f"[WARN] Failed to parse AST for {rel_to_repo(file_path)}: {ex}"
        )
        return None, [], unresolved_star_warnings, unresolved_plain_warnings, general_warnings

    lines = original_text.splitlines(keepends=True)
    edits: List[FileEdit] = []

    # Helper to derive file:line label
    def loc_label(node: ast.AST) -> str:
        return f"{rel_to_repo(file_path)}:{getattr(node, 'lineno', '?')}"

    def is_legacy_module(mod: str) -> bool:
        for src in index.all_sources:
            if mod == src or mod.startswith(src + "."):
                return True
        return False

    # Iterate nodes and plan rewrites
    for node in ast.walk(module_ast):
        if isinstance(node, ast.ImportFrom):
            # Ignore relative imports
            if getattr(node, "module", None) is None:
                continue
            module_name = node.module
            # Determine indentation from the first line of the node
            try:
                indent = get_indent(lines[node.lineno - 1])
            except Exception:
                indent = ""

            # Star import case
            if len(node.names) == 1 and node.names[0].name == "*":
                star_entry = index.star_by_source.get(module_name)
                if star_entry:
                    # Rewrite module path only
                    new_line = compose_import_from(star_entry.target_module, [("*", None)], indent)
                    edits.append(
                        FileEdit(
                            start_line=node.lineno,
                            end_line=node.end_lineno or node.lineno,
                            new_text=new_line + "\n",
                        )
                    )
                    per_mapping_hits[star_entry.id] = per_mapping_hits.get(star_entry.id, 0) + 1
                    if verbose:
                        print(f"[REWRITE] {loc_label(node)}: from {module_name} import * -> from {star_entry.target_module} import *")
                else:
                    # Warn only when star-import targets a legacy module per mapping sources
                    if is_legacy_module(module_name):
                        unresolved_star_warnings.append(f"{loc_label(node)} star-import from legacy module '{module_name}'")
                continue

            # Symbol imports
            entries = index.by_source.get(module_name)
            if not entries:
                # No mapping for this module; nothing to do
                continue

            # If a wildcard mapping exists for this source, apply it to all symbols (keep names)
            wildcard_entry = None
            for e in entries:
                if len(e.symbols) == 1 and e.symbols[0] == "*":
                    wildcard_entry = e
                    break

            # Group items per target module or original
            grouped: Dict[str, List[Tuple[str, Optional[str]]]] = {}
            changed = False

            if wildcard_entry:
                # All symbols from this module go to wildcard target, names unchanged/asname preserved
                for a in node.names:
                    new_name = a.name
                    asname = a.asname
                    grouped.setdefault(wildcard_entry.target_module, []).append((new_name, asname))
                per_mapping_hits[wildcard_entry.id] = per_mapping_hits.get(wildcard_entry.id, 0) + len(node.names)
                changed = True
            else:
                # Symbol-level mappings
                remaining: List[Tuple[str, Optional[str]]] = []
                for a in node.names:
                    applied = False
                    for e in entries:
                        if a.name in e.symbols:
                            # Rename if specified
                            new_name = e.rename.get(a.name, a.name)
                            # If rename happened and preserve_alias and no explicit asname, use "new as old"
                            if new_name != a.name and e.preserve_alias and a.asname is None:
                                asname = a.name
                            else:
                                asname = a.asname
                            grouped.setdefault(e.target_module, []).append((new_name, asname))
                            per_mapping_hits[e.id] = per_mapping_hits.get(e.id, 0) + 1
                            applied = True
                            changed = True
                            break
                    if not applied:
                        remaining.append((a.name, a.asname))
                if remaining:
                    grouped.setdefault(module_name, []).extend(remaining)

            # Build replacement text
            # Prefer deterministic ordering: mapped target modules sorted by name; original module last if present
            mod_keys = [k for k in grouped.keys() if k != module_name]
            mod_keys.sort()
            if module_name in grouped:
                mod_keys.append(module_name)

            new_lines: List[str] = []
            for mod in mod_keys:
                items = grouped[mod]
                new_lines.append(compose_import_from(mod, items, indent))

            new_block = "\n".join(new_lines) + "\n"

            # If not changed and only original module present, skip
            if not changed and set(grouped.keys()) == {module_name}:
                continue

            edits.append(
                FileEdit(
                    start_line=node.lineno,
                    end_line=node.end_lineno or node.lineno,
                    new_text=new_block,
                )
            )
            if verbose:
                print(f"[REWRITE] {loc_label(node)}: from {module_name} import ... ->")
                for ln in new_lines:
                    print(f"          {ln.strip()}")

        elif isinstance(node, ast.Import):
            # For "import a, b as c" -> split into separate lines for simplicity
            # Determine indentation
            try:
                indent = get_indent(lines[node.lineno - 1])
            except Exception:
                indent = ""

            new_lines: List[str] = []
            any_changed = False
            for a in node.names:
                mod = a.name
                asname = a.asname

                star_entry = index.star_by_source.get(mod)
                if star_entry:
                    # Rewrite to target module, preserve aliasing
                    new_lines.append(compose_import(star_entry.target_module, asname, indent))
                    per_mapping_hits[star_entry.id] = per_mapping_hits.get(star_entry.id, 0) + 1
                    any_changed = True
                    if verbose:
                        print(f"[REWRITE] {rel_to_repo(file_path)}:{node.lineno} import {mod}{' as '+asname if asname else ''} "
                              f"-> import {star_entry.target_module}{' as '+asname if asname else ''}")
                else:
                    # If there is any mapping for this module but not wildcard, warn (symbol-level mapping)
                    if mod in index.by_source and mod not in index.star_by_source:
                        unresolved_plain_warnings.append(
                            f"{rel_to_repo(file_path)}:{node.lineno} plain import of legacy module '{mod}'"
                        )
                        # Keep as-is for now
                        new_lines.append(compose_import(mod, asname, indent))
                    else:
                        # If importing submodule, only rewrite if exact star mapping exists for that submodule
                        # Otherwise keep and potentially warn if parent is legacy? Spec: warn if there is no star mapping for pkg.sub
                        parent_candidates = [mod]
                        if "." in mod:
                            parent_candidates.append(mod.rsplit(".", 1)[0])
                        # Only warn if exact module in sources but no star mapping
                        if mod in index.all_sources and mod not in index.star_by_source:
                            unresolved_plain_warnings.append(
                                f"{rel_to_repo(file_path)}:{node.lineno} plain import of legacy module '{mod}'"
                            )
                        new_lines.append(compose_import(mod, asname, indent))

            # Replace node with composed lines if anything changed or if line was split
            original_line_span = (node.end_lineno or node.lineno) - node.lineno + 1
            # We always replace to ensure consistency when splitting multiple imports into separate lines
            edits.append(
                FileEdit(
                    start_line=node.lineno,
                    end_line=node.end_lineno or node.lineno,
                    new_text="\n".join(new_lines) + "\n",
                )
            )

    if not edits:
        return None, [], unresolved_star_warnings, unresolved_plain_warnings, general_warnings

    # Apply edits bottom-to-top
    edits_sorted = sorted(edits, key=lambda e: e.start_line, reverse=True)
    new_lines = lines[:]
    for ed in edits_sorted:
        start_idx = ed.start_line - 1
        end_idx = ed.end_line  # slice end is exclusive
        new_lines[start_idx:end_idx] = [ed.new_text]

    new_text = "".join(new_lines)
    if new_text == original_text:
        # No effective change in content
        return None, [], unresolved_star_warnings, unresolved_plain_warnings, general_warnings

    return new_text, edits_sorted, unresolved_star_warnings, unresolved_plain_warnings, general_warnings


def iter_python_files(root: str, include_docs: bool, excludes: Sequence[str]) -> Iterable[str]:
    root = os.path.abspath(root)
    for dirpath, dirnames, filenames in os.walk(root):
        # Mutate dirnames in-place to skip unwanted directories
        pruned: List[str] = []
        for d in list(dirnames):
            if should_skip_dir(d, include_docs):
                pruned.append(d)
        for d in pruned:
            dirnames.remove(d)

        for fname in filenames:
            if not is_python_file(fname):
                continue
            fpath = os.path.join(dirpath, fname)
            rel = rel_to_repo(fpath)
            if match_any_exclude(rel, excludes):
                continue
            yield fpath


def safe_write_atomic(path: str, content: str, make_backup: bool) -> None:
    # Backup original as .orig (only if not already exists)
    if make_backup and os.path.exists(path):
        orig = path + ".orig"
        if not os.path.exists(orig):
            try:
                shutil.copy2(path, orig)
            except Exception as ex:
                eprint(f"[WARN] Failed to create backup '{orig}': {ex}")

    # Write to a temp file next to target, then replace
    dir_name = os.path.dirname(path)
    base = os.path.basename(path)
    tmp_path = os.path.join(dir_name, f".{base}.tmp_rewrite")
    with io.open(tmp_path, "w", encoding="utf-8", newline="") as f:
        f.write(content)
    os.replace(tmp_path, path)


def print_summary(sumr: Summary, strict: bool) -> None:
    print("=== Import Rewrite Summary ===")
    if sumr.changed_files:
        print("Changed files:")
        for p in sumr.changed_files:
            print(f"  - {rel_to_repo(p)}")
    else:
        print("No files changed.")

    if sumr.per_mapping_hits:
        print("Per-mapping hit counts:")
        # We don't have labels here; we will print id and count. Labels are emitted inline during verbose rewrites.
        for mid, cnt in sorted(sumr.per_mapping_hits.items(), key=lambda x: x[0]):
            print(f"  mapping[{mid}]: {cnt}")
    else:
        print("Per-mapping hit counts: none")

    if sumr.unresolved_star_imports:
        print(f"Unresolved star-imports: {len(sumr.unresolved_star_imports)}")
        for w in sumr.unresolved_star_imports:
            print(f"  - {w}")
    else:
        print("Unresolved star-imports: 0")

    if sumr.unresolved_legacy_plain_imports:
        print(f"Unresolved legacy plain imports: {len(sumr.unresolved_legacy_plain_imports)}")
        for w in sumr.unresolved_legacy_plain_imports:
            print(f"  - {w}")
    else:
        print("Unresolved legacy plain imports: 0")

    if sumr.warnings:
        print("Warnings:")
        for w in sumr.warnings:
            print(f"  - {w}")

    if strict and (sumr.unresolved_star_imports or sumr.unresolved_legacy_plain_imports):
        print("[STRICT] Unresolved legacy imports detected.", file=sys.stderr)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Automated AST-based import rewrite tool. See module docstring for behavior details."
    )
    parser.add_argument("--root", default=DEFAULT_ROOT, help="Root directory to search (default: src)")
    parser.add_argument("--map", dest="map_path", default=DEFAULT_MAP, help="Path to mapping file (default: docs/development/import_rewrite_map.yaml)")
    parser.add_argument("--dry-run", action="store_true", help="Don't write changes; print intended edits and summary")
    parser.add_argument("--strict", action="store_true", help="Fail (non-zero) if unresolved legacy imports/star-imports remain")
    parser.add_argument("--exclude", action="append", default=[], help="Glob pattern to exclude (repeatable)")
    parser.add_argument("--include-docs", action="store_true", help="Include docs/**/*.py (code examples) in the search")
    backup_group = parser.add_mutually_exclusive_group()
    backup_group.add_argument("--backup", dest="backup", action="store_true", default=True, help="Create .orig backups (default)")
    backup_group.add_argument("--no-backup", dest="backup", action="store_false", help="Disable .orig backups")
    parser.add_argument("--verbose", action="store_true", help="Print per-file details")

    args = parser.parse_args()

    mapping = load_mapping(args.map_path)
    index = build_mapping_index(mapping)

    changed_files: List[str] = []
    per_mapping_hits: Dict[int, int] = {}
    unresolved_star: List[str] = []
    unresolved_plain: List[str] = []
    warnings: List[str] = []

    # Iterate files
    files_iter = list(iter_python_files(args.root, args.include_docs, args.exclude))
    for fpath in files_iter:
        new_text, edits, unresolved_star_w, unresolved_plain_w, general_w = analyze_and_rewrite_file(
            fpath,
            index,
            per_mapping_hits,
            args.strict,
            args.exclude,
            args.verbose,
        )
        unresolved_star.extend(unresolved_star_w)
        unresolved_plain.extend(unresolved_plain_w)
        warnings.extend(general_w)

        if new_text is None:
            continue

        changed_files.append(fpath)
        if args.dry_run:
            # Print intended edits
            rel = rel_to_repo(fpath)
            print(f"[DRY-RUN] Would rewrite: {rel}")
            for ed in edits:
                print(
                    f"  - Replace lines {ed.start_line}-{ed.end_line} with:\n"
                    + "\n".join(f"    {ln}" for ln in ed.new_text.rstrip('\n').split('\n'))
                )
        else:
            try:
                safe_write_atomic(fpath, new_text, make_backup=args.backup)
            except Exception as ex:
                eprint(f"[ERROR] Failed to write changes to {rel_to_repo(fpath)}: {ex}")
                return 2

    # Print summary
    sumr = Summary(
        changed_files=changed_files,
        per_mapping_hits=per_mapping_hits,
        unresolved_star_imports=unresolved_star,
        unresolved_legacy_plain_imports=unresolved_plain,
        warnings=warnings,
    )
    print_summary(sumr, strict=args.strict)

    if args.strict and (unresolved_star or unresolved_plain):
        return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())