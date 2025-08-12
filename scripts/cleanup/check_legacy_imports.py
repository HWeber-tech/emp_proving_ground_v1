#!/usr/bin/env python3
"""
Legacy Import Guard

- Loads mapping from docs/development/import_rewrite_map.yaml (JSON-first, fallback to YAML if PyYAML is installed).
- Builds a blacklist of legacy modules from all "sources" entries.
- Walks *.py under --root (default: src), skipping common build/cache/docs dirs.
- Flags any Import/ImportFrom referencing a blacklisted module (exact match or startswith "name.").
- Allows exceptions via --allow-file for specific façade/shim files (defaults below).
- Outputs concise summary with file:line and offending module.
- Exit non-zero with --fail when violations are present.

CLI:
  python scripts/cleanup/check_legacy_imports.py --root src --map docs/development/import_rewrite_map.yaml --fail --verbose
"""

from __future__ import annotations

import argparse
import ast
import io
import json
import os
import sys
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
    "docs",  # docs excluded by default for guard
}

DEFAULT_ROOT = "src"
DEFAULT_MAP = os.path.join("docs", "development", "import_rewrite_map.yaml")

DEFAULT_ALLOWED_FILES = [
    "src/core/sensory_organ.py",
    "src/phase2d_integration_validator.py",
    "src/intelligence/red_team_ai.py",
    "src/sensory/models/__init__.py",
]


def eprint(*args: Any, **kwargs: Any) -> None:
    print(*args, file=sys.stderr, **kwargs)


def to_posix(path: str) -> str:
    return path.replace("\\", "/")


def rel_to_repo(path: str) -> str:
    try:
        return to_posix(os.path.relpath(path, start=os.getcwd()))
    except Exception:
        return to_posix(path)


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


def build_blacklist(mapping: Dict[str, Any]) -> Set[str]:
    """
    Collect all legacy module names from 'sources' fields across mappings.
    """
    if not isinstance(mapping, dict) or "mappings" not in mapping:
        eprint("[ERROR] Mapping file missing required 'mappings' key.")
        sys.exit(2)
    raw_entries = mapping.get("mappings", [])
    if not isinstance(raw_entries, list):
        eprint("[ERROR] Mapping 'mappings' must be a list.")
        sys.exit(2)

    blacklist: Set[str] = set()
    for idx, m in enumerate(raw_entries):
        if not isinstance(m, dict):
            eprint(f"[ERROR] Invalid mapping entry at index {idx}: expected object.")
            sys.exit(2)
        sources = m.get("sources", [])
        if not isinstance(sources, list) or not sources:
            eprint(f"[ERROR] Mapping entry {idx} missing/invalid 'sources'.")
            sys.exit(2)
        for s in sources:
            if isinstance(s, str) and s:
                blacklist.add(s)
    return blacklist


def should_skip_dir(dirname: str) -> bool:
    return dirname in SKIP_DIRS


def is_python_file(path: str) -> bool:
    return path.endswith(".py")


def iter_python_files(root: str) -> Iterable[str]:
    root = os.path.abspath(root)
    for dirpath, dirnames, filenames in os.walk(root):
        # prune directories
        for d in list(dirnames):
            if should_skip_dir(d):
                dirnames.remove(d)
        for fname in filenames:
            if not is_python_file(fname):
                continue
            yield os.path.join(dirpath, fname)


def module_matches_blacklist(module: str, blacklist: Set[str]) -> Optional[str]:
    """
    Return the blacklisted root that matches this module, if any.
    Match if module == blacklisted or module startswith blacklisted + '.'
    """
    for root in blacklist:
        if module == root or module.startswith(root + "."):
            return root
    return None


def analyze_file_for_legacy_imports(
    file_path: str,
    blacklist: Set[str],
    whitelist: Set[str],
) -> List[Tuple[int, str]]:
    """
    Return list of (lineno, offending_module) violations.
    """
    try:
        with io.open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
    except Exception as ex:
        eprint(f"[WARN] Failed to read {rel_to_repo(file_path)}: {ex}")
        return []

    try:
        tree = ast.parse(text)
    except SyntaxError as ex:
        eprint(f"[WARN] SyntaxError in {rel_to_repo(file_path)} at line {getattr(ex, 'lineno', '?')}: {ex.msg}")
        return []
    except Exception as ex:
        eprint(f"[WARN] Failed to parse AST for {rel_to_repo(file_path)}: {ex}")
        return []

    violations: List[Tuple[int, str]] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                mod = alias.name
                hit = module_matches_blacklist(mod, blacklist)
                if hit and not any(mod == t or mod.startswith(t + ".") for t in whitelist):
                    violations.append((node.lineno, mod))
        elif isinstance(node, ast.ImportFrom):
            # skip relative imports with no module (e.g., from . import x)
            if getattr(node, "module", None) is None:
                continue
            mod = node.module
            hit = module_matches_blacklist(mod, blacklist)
            if hit and not any(mod == t or mod.startswith(t + ".") for t in whitelist):
                violations.append((node.lineno, mod))

    return violations


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Guard that flags legacy imports based on declarative mapping 'sources'."
    )
    parser.add_argument("--root", default=DEFAULT_ROOT, help="Root directory to search (default: src)")
    parser.add_argument("--map", dest="map_path", default=DEFAULT_MAP, help="Path to mapping file (default: docs/development/import_rewrite_map.yaml)")
    parser.add_argument("--allow-file", action="append", default=[], help="Repo-relative path to allow legacy imports (repeatable)")
    parser.add_argument("--fail", action="store_true", help="Exit non-zero if violations are found")
    parser.add_argument("--verbose", action="store_true", help="Print per-file details")

    args = parser.parse_args()

    mapping = load_mapping(args.map_path)
    blacklist = build_blacklist(mapping)
    whitelist: Set[str] = set()
    raw_entries = mapping.get("mappings", [])
    if isinstance(raw_entries, list):
        for m in raw_entries:
            if isinstance(m, dict):
                tgt = m.get("target_module")
                if isinstance(tgt, str) and tgt:
                    whitelist.add(tgt)

    # Prepare allowed files (normalize to posix repo-relative)
    allowed: Set[str] = set(to_posix(p) for p in DEFAULT_ALLOWED_FILES)
    for p in args.allow_file or []:
        allowed.add(to_posix(p))

    all_violations: List[Tuple[str, int, str]] = []  # (file, lineno, module)

    files = list(iter_python_files(args.root))
    for fpath in files:
        rel = rel_to_repo(fpath)
        rel_posix = to_posix(rel)

        if rel_posix in allowed:
            if args.verbose:
                print(f"[ALLOW] Skipping allowed file: {rel_posix}")
            continue

        violations = analyze_file_for_legacy_imports(fpath, blacklist, whitelist)
        if violations and args.verbose:
            print(f"[SCAN] {rel_posix}: {len(violations)} violation(s)")

        for lineno, mod in violations:
            all_violations.append((rel_posix, lineno, mod))

    # Print concise summary
    if all_violations:
        print("Legacy import violations:")
        for rel, lineno, mod in sorted(all_violations):
            print(f"  - {rel}:{lineno} uses legacy module '{mod}'")
        print(f"Total violations: {len(all_violations)}")
    else:
        print("No legacy import violations detected.")

    if args.fail and all_violations:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())