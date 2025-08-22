#!/usr/bin/env python3
"""
Explicit Any Rewriter (dry-run by default)

Rewrites trivial, behavior-preserving typing patterns across code under src/:
  - Dict[str, Any] → dict[str, object]
  - List[Any] → list[object]
  - Sequence[Any] → Sequence[object]
  - Mapping[str, Any] → Mapping[str, object]
  - Optional[Dict[str, Any]] → Optional[dict[str, object]]
  - Callable[..., Any] → Callable[..., None] when the callable is a handler/callback signature
      Heuristic: the annotated name (variable/parameter) contains "handler" or "callback"
  - kwargs: Any → kwargs: object (for **kwargs)
  - args: Any → args: object (for *args)

Safeguards:
  - Only touches type annotations via LibCST (no strings/comments).
  - Uses include/exclude regex filters. Default excludes tests and stubs.

CLI:
  explicit_any_rewriter.py [--apply] [--paths PATH ...]
                           [--include REGEX] [--exclude REGEX]
                           [--backup]

Default:
  - Dry-run (prints unified diffs)
  - paths: ["src"]
  - exclude: ^stubs/|/tests?/|^tests/
"""

from __future__ import annotations

import argparse
import difflib
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import libcst as cst
import libcst.matchers as m

# ---------------------------
# Helpers
# ---------------------------

# Broaden default excludes to avoid stubs, tests, docs, build artifacts, and venvs.
DEFAULT_EXCLUDE = r"(?:^|/)(stubs/|tests?/|docs/|migrations/|__pycache__/|build/|dist/|site-packages/|\.venv(?:_.+)?/)"
TYPING_ALIASES = {
    # typing to builtin
    "typing.Dict": "dict",
    "typing.List": "list",
    # PEP 585 builtins remain identical name
    "dict": "dict",
    "list": "list",
    # Leave these from typing (capitalized)
    "typing.Sequence": "Sequence",
    "typing.Mapping": "Mapping",
    "typing.Optional": "Optional",
    "typing.Callable": "Callable",
    # Support from collections.abc
    "collections.abc.Sequence": "Sequence",
    "collections.abc.Mapping": "Mapping",
    "collections.abc.Callable": "Callable",
}

# Names that indicate a handler/callback for Callable[..., Any] → Callable[..., None]
CALLBACK_HINTS = ("handler", "callback")


def dotted_name(node: cst.BaseExpression) -> str | None:
    if isinstance(node, cst.Name):
        return node.value
    if isinstance(node, cst.Attribute):
        parts = []
        cur: cst.BaseExpression | None = node
        while isinstance(cur, cst.Attribute):
            parts.append(cur.attr.value)
            cur = cur.value
        if isinstance(cur, cst.Name):
            parts.append(cur.value)
        parts.reverse()
        return ".".join(parts)
    return None


def qual_to_target(qual: str) -> str | None:
    if qual in TYPING_ALIASES:
        return TYPING_ALIASES[qual]
    # If not fully-qualified, try to map bare names
    if qual in ("Dict", "List", "Set", "Tuple"):
        return qual.lower()
    if qual in ("Sequence", "Mapping", "Optional", "Callable", "Iterable"):
        return qual
    return None


def is_name_like_any(node: cst.CSTNode) -> bool:
    return m.matches(node, m.Name("Any")) or m.matches(node, m.Attribute(value=m.Name("typing"), attr=m.Name("Any")))


def make_name_or_attr(name: str) -> cst.BaseExpression:
    if "." in name:
        parts = name.split(".")
        expr: cst.BaseExpression = cst.Name(parts[0])
        for part in parts[1:]:
            expr = cst.Attribute(value=expr, attr=cst.Name(part))
        return expr
    return cst.Name(name)


def make_subscript(name: str, args: list[cst.BaseExpression]) -> cst.Subscript:
    return cst.Subscript(
        value=make_name_or_attr(name),
        slice=[cst.SubscriptElement(slice=cst.Index(value=a)) for a in args],
    )


def to_object_node() -> cst.BaseExpression:
    # prefer builtin 'object'
    return cst.Name("object")


@dataclass
class Stats:
    files_changed: int = 0
    replacements: dict[str, int] = field(default_factory=lambda: {})

    def inc(self, key: str, n: int = 1) -> None:
        self.replacements[key] = self.replacements.get(key, 0) + n


# ---------------------------
# Transformer
# ---------------------------

class AnyRewriter(cst.CSTTransformer):
    def __init__(self, stats: Stats):
        self.stats = stats
        # Track current param target name to apply Callable callback heuristic and args/kwargs handling
        self._param_target_name: str | None = None
        self._in_param: bool = False
        self._param_star: cst.ParamStar | None = None  # kind: * or **
        # Track usage for import hygiene
        self._uses_mapping = False
        self._uses_sequence = False
        self._uses_iterable = False

    # ---- Function parameters context to detect args/kwargs and callback names ----

    def visit_Param(self, node: cst.Param) -> None:
        self._in_param = True
        self._param_target_name = node.name.value if node.name else None
        self._param_star = node.star

    def leave_Param(self, original_node: cst.Param, updated_node: cst.Param) -> cst.Param:
        self._in_param = False
        self._param_star = None
        self._param_target_name = None
        return updated_node

    # ---- Module-level import hygiene ----

    def leave_Module(self, original_node: cst.Module, updated_node: cst.Module) -> cst.Module:
        # Operate on imports: ensure collections.abc for Mapping/Sequence/Iterable; prune unused typing imports for Any/Dict/List/Set/Tuple
        code = updated_node.code

        needed_cabc: set[str] = set()
        if re.search(r"\bMapping\b", code):
            needed_cabc.add("Mapping")
        if re.search(r"\bSequence\b", code):
            needed_cabc.add("Sequence")
        if re.search(r"\bIterable\b", code):
            needed_cabc.add("Iterable")

        # Determine which typing names are still used after rewrite
        typing_maybe_remove = {"Any", "Dict", "List", "Set", "Tuple"}
        still_used_typing: set[str] = set()
        for name in typing_maybe_remove:
            if re.search(rf"\b{name}\b", code):
                still_used_typing.add(name)

        # Walk body, adjust ImportFrom nodes
        new_body: list[cst.CSTNode] = []
        has_cabc_import = False
        cabc_existing: set[str] = set()

        for stmt in updated_node.body:
            if isinstance(stmt, cst.SimpleStatementLine) and stmt.body and isinstance(stmt.body[0], cst.ImportFrom):
                imp: cst.ImportFrom = stmt.body[0]
                # from typing import ...
                if isinstance(imp.module, cst.Name) and imp.module.value == "typing":
                    if isinstance(imp.names, cst.ImportStar):
                        new_body.append(stmt)
                        continue
                    new_aliases: list[cst.ImportAlias] = []
                    for alias in imp.names:  # type: ignore[assignment]
                        if isinstance(alias, cst.ImportAlias) and isinstance(alias.name, cst.Name):
                            nm = alias.evaluated_name
                            if nm in typing_maybe_remove and nm not in still_used_typing:
                                # drop it
                                continue
                        new_aliases.append(alias)  # type: ignore[arg-type]
                    if new_aliases:
                        new_imp = imp.with_changes(names=tuple(new_aliases))  # type: ignore[arg-type]
                        new_body.append(cst.SimpleStatementLine([new_imp]))
                    else:
                        # remove entire import if emptied
                        continue
                    continue
                # from collections.abc import ...
                if isinstance(imp.module, cst.Attribute) and dotted_name(imp.module) == "collections.abc":
                    has_cabc_import = True
                    if isinstance(imp.names, cst.ImportStar):
                        new_body.append(stmt)
                        continue
                    kept_aliases: list[cst.ImportAlias] = []
                    for alias in imp.names:  # type: ignore[assignment]
                        if isinstance(alias, cst.ImportAlias) and isinstance(alias.name, cst.Name):
                            cabc_existing.add(alias.name.value)
                            kept_aliases.append(alias)  # keep existing
                    # We'll possibly add missing names later
                    new_body.append(cst.SimpleStatementLine([imp.with_changes(names=tuple(kept_aliases))]))  # type: ignore[arg-type]
                    continue
            new_body.append(stmt)

        # If we need collections.abc names, ensure they are imported
        to_add = sorted([n for n in needed_cabc if n not in cabc_existing])
        if to_add:
            cabc_import = cst.SimpleStatementLine(
                [
                    cst.ImportFrom(
                        module=cst.Attribute(value=cst.Name("collections"), attr=cst.Name("abc")),
                        names=tuple(cst.ImportAlias(name=cst.Name(n)) for n in to_add),
                    )
                ]
            )
            # Insert after any future-imports and before other code; naive: put at top
            new_body = [cabc_import] + new_body

        return updated_node.with_changes(body=tuple(new_body))

    # ---- Core: rewrite annotations only ----

    def leave_Annotation(self, original_node: cst.Annotation, updated_node: cst.Annotation) -> cst.Annotation:
        # Bare Any -> object in any annotation context
        if is_name_like_any(updated_node.annotation):
            self.stats.inc("Any -> object")
            return updated_node.with_changes(annotation=to_object_node())

        new_ann, touched = self._rewrite_type_expr(updated_node.annotation)
        if touched:
            return updated_node.with_changes(annotation=new_ann)
        return updated_node

    def leave_Arg(self, original_node: cst.Arg, updated_node: cst.Arg) -> cst.Arg:
        # Type in type aliases like: X: TypeAlias = Dict[str, Any] is within value, not annotation.
        # We do not rewrite runtime values; skip.
        return updated_node

    # ---- Helpers ----

    def _rewrite_type_expr(self, expr: cst.BaseExpression) -> tuple[cst.BaseExpression, bool]:
        """
        Returns possibly-updated expr and whether change was made.
        Only affects valid typing constructs.
        """
        # Subscripted types like Dict[str, Any], List[Any], Optional[T], Callable[..., R], etc.
        if isinstance(expr, cst.Subscript):
            base_name = dotted_name(expr.value)
            if base_name is None:
                # Recurse into subscript args anyway
                changed = False
                new_slices = []
                for s in expr.slice:
                    if isinstance(s.slice, cst.Index):
                        new_val, ch = self._rewrite_type_expr(s.slice.value)
                        changed = changed or ch
                        new_slices.append(s.with_changes(slice=cst.Index(value=new_val)))
                    else:
                        new_slices.append(s)
                return expr.with_changes(slice=new_slices), changed

            target = qual_to_target(base_name) or base_name

            # Normalize slice list
            slices: list[cst.BaseExpression] = []
            changed_inner = False
            for s in expr.slice:
                if isinstance(s.slice, cst.Index):
                    new_v, ch = self._rewrite_type_expr(s.slice.value)
                    slices.append(new_v)
                    changed_inner = changed_inner or ch
                else:
                    # Unexpected - keep as is
                    slices.append(s.slice.value)  # type: ignore[attr-defined]
            # Pattern handlers
            # 1) Dict[str, Any] -> dict[str, object]
            if target in ("dict",) and len(slices) == 2:
                key_t, val_t = slices
                if self._is_str_type(key_t) and is_name_like_any(val_t):
                    self.stats.inc("Dict[str, Any] -> dict[str, object]")
                    return make_subscript("dict", [cst.Name("str"), to_object_node()]), True or changed_inner

            if target in ("Dict",) and len(slices) == 2:
                key_t, val_t = slices
                if self._is_str_type(key_t) and is_name_like_any(val_t):
                    self.stats.inc("Dict[str, Any] -> dict[str, object]")
                    return make_subscript("dict", [cst.Name("str"), to_object_node()]), True

            # 2) List[Any] -> list[object]
            if target in ("list",) and len(slices) == 1:
                el = slices[0]
                if is_name_like_any(el):
                    self.stats.inc("List[Any] -> list[object]")
                    return make_subscript("list", [to_object_node()]), True

            if target in ("List",) and len(slices) == 1:
                el = slices[0]
                if is_name_like_any(el):
                    self.stats.inc("List[Any] -> list[object]")
                    return make_subscript("list", [to_object_node()]), True

            # 3) Sequence[Any] -> Sequence[object]
            if target in ("Sequence",) and len(slices) == 1:
                el = slices[0]
                if is_name_like_any(el):
                    self._uses_sequence = True
                    self.stats.inc("Sequence[Any] -> Sequence[object]")
                    return make_subscript("Sequence", [to_object_node()]), True

            # 4) Mapping[str, Any] -> Mapping[str, object]
            if target in ("Mapping",) and len(slices) == 2:
                k, v = slices
                if self._is_str_type(k) and is_name_like_any(v):
                    self._uses_mapping = True
                    self.stats.inc("Mapping[str, Any] -> Mapping[str, object]")
                    return make_subscript("Mapping", [cst.Name("str"), to_object_node()]), True

            # 4b) Iterable[Any] -> Iterable[object]
            if target in ("Iterable",) and len(slices) == 1:
                el = slices[0]
                if is_name_like_any(el):
                    self._uses_iterable = True
                    self.stats.inc("Iterable[Any] -> Iterable[object]")
                    return make_subscript("Iterable", [to_object_node()]), True

            # 5) Optional[Dict[str, Any]] -> Optional[dict[str, object]]
            if target in ("Optional",) and len(slices) == 1:
                inner = slices[0]
                if isinstance(inner, cst.Subscript):
                    inner_base = dotted_name(inner.value)
                    inner_target = qual_to_target(inner_base) if inner_base else None
                    if inner_target in ("dict", "Dict"):
                        inner_slices = []
                        for s in inner.slice:
                            assert isinstance(s.slice, cst.Index)
                            inner_slices.append(s.slice.value)
                        if len(inner_slices) == 2 and self._is_str_type(inner_slices[0]) and is_name_like_any(inner_slices[1]):
                            self.stats.inc("Optional[Dict[str, Any]] -> Optional[dict[str, object]]")
                            new_inner = make_subscript("dict", [cst.Name("str"), to_object_node()])
                            return make_subscript("Optional", [new_inner]), True

            # 6) Callable[..., Any] -> Callable[..., None] for handler/callback
            if target in ("Callable",) and len(slices) == 2:
                args_node, ret_node = slices
                if is_name_like_any(ret_node):
                    if self._is_ellipsis(args_node) and self._is_callback_context():
                        self.stats.inc("Callable[..., Any] -> Callable[..., None] (callback)")
                        return make_subscript("Callable", [cst.Ellipsis(), cst.Name("None")]), True
                    # General case: Callable[..., Any] -> Callable[..., object]
                    self.stats.inc("Callable[..., Any] -> Callable[..., object]")
                    # keep args_node as-is
                    return make_subscript("Callable", [args_node, to_object_node()]), True

            # 7) Set[Any] -> set[object]
            if target in ("set",) and len(slices) == 1:
                el = slices[0]
                if is_name_like_any(el):
                    self.stats.inc("Set[Any] -> set[object]")
                    return make_subscript("set", [to_object_node()]), True

            if target in ("Set",) and len(slices) == 1:
                el = slices[0]
                if is_name_like_any(el):
                    self.stats.inc("Set[Any] -> set[object]")
                    return make_subscript("set", [to_object_node()]), True

            # 8) Tuple[Any, ...] -> tuple[object, ...]
            if target in ("tuple",) and len(slices) == 2:
                first, second = slices
                if is_name_like_any(first) and self._is_ellipsis(second):
                    self.stats.inc("Tuple[Any, ...] -> tuple[object, ...]")
                    return make_subscript("tuple", [to_object_node(), cst.Ellipsis()]), True
            if target in ("Tuple",) and len(slices) == 2:
                first, second = slices
                if is_name_like_any(first) and self._is_ellipsis(second):
                    self.stats.inc("Tuple[Any, ...] -> tuple[object, ...]")
                    return make_subscript("tuple", [to_object_node(), cst.Ellipsis()]), True

            # For any other subscript, ensure inner changes propagate
            if changed_inner:
                return expr.with_changes(slice=[cst.SubscriptElement(slice=cst.Index(value=s)) for s in slices]), True
            return expr, False

        # Bare Any → object in *args/**kwargs only
        if self._in_param and is_name_like_any(expr):
            if self._param_star is not None:
                # *args
                if isinstance(self._param_star, cst.ParamStar) and self._param_star.kind == "*":
                    self.stats.inc("args: Any -> args: object")
                    return to_object_node(), True
                # **kwargs
                if isinstance(self._param_star, cst.ParamStar) and self._param_star.kind == "**":
                    self.stats.inc("kwargs: Any -> kwargs: object")
                    return to_object_node(), True

        # No change
        return expr, False

    def _is_callback_context(self) -> bool:
        if not self._param_target_name:
            return False
        name = self._param_target_name.lower()
        return any(h in name for h in CALLBACK_HINTS)

    @staticmethod
    def _is_str_type(node: cst.BaseExpression) -> bool:
        # Accept 'str' or 'builtins.str' or typing_extensions.Literal['...']? Only str is requested.
        if isinstance(node, cst.Name) and node.value == "str":
            return True
        if isinstance(node, cst.Attribute):
            return dotted_name(node) in ("builtins.str",)
        return False

    @staticmethod
    def _is_ellipsis(node: cst.BaseExpression) -> bool:
        return isinstance(node, cst.Ellipsis)


# ---------------------------
# Runner
# ---------------------------

@dataclass
class Config:
    apply: bool
    paths: list[Path]
    include: str | None
    exclude: str
    backup: bool
    jobs: int


def iter_files(paths: Iterable[Path], include: str | None, exclude: str) -> Iterable[Path]:
    inc_re = re.compile(include) if include else None
    exc_re = re.compile(exclude) if exclude else None

    for base in paths:
        if base.is_file() and base.suffix == ".py":
            rel = base.as_posix()
            if exc_re and exc_re.search(rel):
                continue
            if inc_re and not inc_re.search(rel):
                continue
            yield base
        elif base.is_dir():
            for p in base.rglob("*.py"):
                rel = p.as_posix()
                if exc_re and exc_re.search(rel):
                    continue
                if inc_re and not inc_re.search(rel):
                    continue
                yield p


def rewrite_file(path: Path, cfg: Config, stats: Stats) -> tuple[bool, str]:
    """
    Returns (changed, diff_text)
    """
    src = path.read_text(encoding="utf-8")
    try:
        module = cst.parse_module(src)
    except Exception as e:
        return False, f"# Skipped parse error: {path}: {e}"

    transformer = AnyRewriter(stats)
    new_module = module.visit(transformer)
    new_code = new_module.code

    if new_code == src:
        return False, ""

    diff = "\n".join(
        difflib.unified_diff(
            src.splitlines(),
            new_code.splitlines(),
            fromfile=str(path),
            tofile=str(path),
            lineterm="",
        )
    )

    if cfg.apply:
        if cfg.backup:
            path.with_suffix(path.suffix + ".bak").write_text(src, encoding="utf-8")
        path.write_text(new_code, encoding="utf-8")

    return True, diff


def parse_args(argv: list[str]) -> Config:
    ap = argparse.ArgumentParser(description="Rewrite trivial explicit Any patterns using LibCST.")
    # New interface: --dir, --glob, --include, --exclude, --jobs, --apply
    ap.add_argument("--apply", action="store_true", help="Apply changes in-place (default: dry-run; prints diffs).")
    ap.add_argument("--dir", dest="dirs", action="append", default=None, help="Directory to scan (may be specified multiple times).")
    ap.add_argument("--glob", dest="globs", action="append", default=None, help='fnmatch-style pattern(s), e.g. "src/core/**/*.py".')
    ap.add_argument("--include", default=None, help="Include regex (optional).")
    ap.add_argument("--exclude", default=DEFAULT_EXCLUDE, help=f"Exclude regex. Default: {DEFAULT_EXCLUDE!r}")
    ap.add_argument("--backup", action="store_true", help="Write .bak files when applying changes.")
    ap.add_argument("--jobs", type=int, default=1, help="Parallelism for dry-run scanning (no parallel writes).")
    # Back-compat: allow positional --paths if provided
    ap.add_argument("--paths", nargs="*", default=None, help="Deprecated; prefer --dir/--glob.")
    args = ap.parse_args(argv)

    # Resolve target paths
    paths: list[Path] = []
    if args.paths:
        paths.extend(Path(p) for p in args.paths)
    if args.dirs:
        paths.extend(Path(d) for d in args.dirs)
    if not paths:
        paths = [Path("src")]

    # Expand globs into concrete files and append as synthetic paths list
    if args.globs:
        for pattern in args.globs:
            for p in Path(".").glob(pattern):
                if p.suffix == ".py":
                    paths.append(p)

    return Config(
        apply=bool(args.apply),
        paths=paths,
        include=args.include,
        exclude=args.exclude,
        backup=bool(args.backup),
        jobs=int(args.jobs),
    )


def main(argv: list[str]) -> int:
    cfg = parse_args(argv)
    stats = Stats()

    changed_files = 0
    diffs_out: list[str] = []
    warnings: list[str] = []  # NEW: collect concise warnings for parse failures

    # Parallel dry-run reading; writes remain sequential
    paths = list(iter_files(cfg.paths, cfg.include, cfg.exclude))

    if not cfg.apply:
        # compute per-file to safely aggregate stats
        import concurrent.futures

        def process_one(p: Path) -> tuple[Path, bool, str, dict[str, int]]:
            local_stats = Stats()
            ch, diff = rewrite_file(p, cfg, local_stats)
            return p, ch, diff, local_stats.replacements

        max_workers = cfg.jobs if cfg.jobs and cfg.jobs > 0 else (os.cpu_count() or 4)
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
            for _, ch, diff, repl in ex.map(process_one, paths):
                if ch:
                    changed_files += 1
                    diffs_out.append(diff)
                else:
                    # Capture concise parse warnings (or other non-change notes)
                    if diff.startswith("# Skipped parse error:"):
                        warnings.append(diff)
                for k, v in repl.items():
                    stats.inc(k, v)
    else:
        for f in paths:
            ch, diff = rewrite_file(f, cfg, stats)
            if ch:
                changed_files += 1

    # Reporting
    print("Explicit Any Rewriter Summary")
    print("=============================")
    print(f"Files scanned: {len(paths)}")
    print(f"Files changed: {changed_files}")
    if stats.replacements:
        print("Replacements:")
        for k, v in sorted(stats.replacements.items(), key=lambda kv: (-kv[1], kv[0])):
            print(f"  {k}: {v}")

    # Print concise warnings, if any
    if warnings:
        print("\nWarnings:")
        for w in warnings:
            print(w)

    if not cfg.apply and diffs_out:
        print("\nUnified Diffs (dry-run):")
        print("\n".join(diffs_out))

    # Summary footer (requested by spec)
    print("\n--- Summary Footer ---")
    print(f"Scanned: {len(paths)}, Changed: {changed_files}, Rules: {sum(stats.replacements.values())} total")
    if stats.replacements:
        for k, v in sorted(stats.replacements.items(), key=lambda kv: (-kv[1], kv[0])):
            print(f"  {k}: {v}")

    # Always exit 0; tool is non-fatal
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))