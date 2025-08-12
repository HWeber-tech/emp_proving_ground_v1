# Import rewrite and legacy-import guard: usage

This guide explains how to use the automated import rewrite tool and the legacy-import guard.
The mapping at `docs/development/import_rewrite_map.yaml` is the single source of truth (JSON-as-YAML).

Tools:
- Rewrite tool: `scripts/cleanup/rewrite_imports.py`
- Legacy guard: `scripts/cleanup/check_legacy_imports.py`

CI and pre-commit:
- CI workflow: `.github/workflows/import-hygiene.yml` runs guard and strict dry-run on push/PR
- Local hooks: `.pre-commit-config.yaml` runs the guard and duplicate check before commits


## Requirements

- Python 3.8+
- Optional: `PyYAML` (for YAML parsing if mapping is not valid JSON)
  - Install: `pip install PyYAML`


## Mapping semantics (single source of truth)

File: `docs/development/import_rewrite_map.yaml`

- `sources`: list[str] of legacy module names to catch (supports both `pkg` and `src.pkg` variants).
- `target_module`: canonical module path.
- `symbols`: list[str]
  - `["*"]` means wildcard: rewrite entire module import path.
  - Specific symbols: rewrite only listed names (e.g., `["get_global_cache"]`).
- `rename`: optional map `{old: new}` when moving symbols into canonical API (e.g., `GlobalCache` -> `MarketDataCache`).
- `preserve_alias`: when a rename occurs and the original import has no explicit alias, emit `new as old` to preserve call sites.

Notes:
- The mapping is authored as JSON which is also valid YAML (JSON-as-YAML). The loader tries JSON first, then YAML if `PyYAML` is present.
- Mapping additions should remain minimal and targeted to enable incremental, safe migrations.


## Rewrite tool

Script: `scripts/cleanup/rewrite_imports.py`

Key behavior:
- Walks Python files under `--root` (default: `src`), skipping build/cache folders and `docs/` unless `--include-docs` is set.
- AST-based detection of `import` and `from ... import ...`.
- Applies declarative mappings:
  - Wildcard: rewrites the module path, preserving names and aliases.
  - Symbol-level: rewrites only matched names; supports `rename` and optional `preserve_alias`.
  - Mixed imports are split into multiple lines per `target_module` and an original-line for remaining symbols.
- Star-import handling:
  - Warns only when the star-import targets a legacy module present in mapping `sources`.
- Atomic writes:
  - By default creates a sibling `.orig` backup and writes atomically via a temp file placed alongside.

Exit codes:
- `0` on success.
- Non-zero if `--strict` and unresolved legacy/star imports are present, or on errors.

Common CLI examples:

Dry-run, strict, verbose (no changes are written):
```bash
python scripts/cleanup/rewrite_imports.py \
  --root src \
  --map docs/development/import_rewrite_map.yaml \
  --dry-run --strict --verbose
```

Apply safe rewrites with backups:
```bash
python scripts/cleanup/rewrite_imports.py \
  --root src \
  --map docs/development/import_rewrite_map.yaml \
  --backup --verbose
```

Disable backups (not recommended for large changes):
```bash
python scripts/cleanup/rewrite_imports.py \
  --root src \
  --map docs/development/import_rewrite_map.yaml \
  --no-backup --verbose
```

Exclude generated code (glob patterns; repeatable):
```bash
python scripts/cleanup/rewrite_imports.py \
  --root src \
  --map docs/development/import_rewrite_map.yaml \
  --dry-run --exclude "src/**/generated/*.py" --exclude "src/**/experimental/*.py"
```

Include docs code examples:
```bash
python scripts/cleanup/rewrite_imports.py \
  --root src \
  --map docs/development/import_rewrite_map.yaml \
  --dry-run --include-docs
```


## Legacy-import guard

Script: `scripts/cleanup/check_legacy_imports.py`

Key behavior:
- Loads the same mapping; builds a blacklist from all `sources`.
- Flags any import whose module equals or startswith a blacklisted module (e.g., `legacy` or `legacy.sub`).
- Ignores imports to canonical targets (any `target_module` or its submodules) to avoid false positives.
- Default allow-list for specific façade files; extend with `--allow-file` as needed.

Exit codes:
- `0` if no violations (or `--fail` not set).
- Non-zero if `--fail` and violations were found.

Examples:

Scan and fail on violations (verbose):
```bash
python scripts/cleanup/check_legacy_imports.py \
  --root src \
  --map docs/development/import_rewrite_map.yaml \
  --fail --verbose
```

Allow an additional file (repeatable):
```bash
python scripts/cleanup/check_legacy_imports.py \
  --root src \
  --map docs/development/import_rewrite_map.yaml \
  --allow-file src/some/facade.py --fail
```


## CI integration

Workflow: `.github/workflows/import-hygiene.yml`

- Runs on push and PR:
  - Legacy import guard with `--fail --verbose`
  - Rewrite tool strict dry-run
- Optional dependency: installs `PyYAML` so mapping can be parsed as YAML if ever needed.


## Pre-commit integration

Config: `.pre-commit-config.yaml`

Hooks:
- Duplicate detection (local repo script).
- Legacy-import guard with `--fail`.

Install and enable:
```bash
pip install pre-commit
pre-commit install
# To run on all files once:
pre-commit run --all-files
```


## Safety, rollback, and troubleshooting

- Backups: when applying rewrites with `--backup`, `.orig` files are created next to modified files.
  - To restore a single file: replace the modified file with its `.orig` counterpart.
- Encoding: tools read/write UTF-8 with newline preservation per Python’s default behavior.
- Mapping parse errors:
  - Ensure valid JSON; or install `PyYAML` to allow YAML parsing.
- Strict failures:
  - Use the guard’s output to add or refine mapping entries (prefer targeted entries).
  - For star-import warnings, ensure the imported module truly appears under `sources` before treating as legacy.


## Contribution guidelines for mapping updates

- Keep entries minimal and explicit.
- Prefer symbol-level mappings and renames for carefully curated API shifts.
- Use wildcard entries for whole-module canonicalization only when stable.
- Maintain shims and re-exports; do not remove them as part of mapping expansions.


## Quick reference

- Mapping: `docs/development/import_rewrite_map.yaml`
- Rewrite tool:
  - Dry-run strict: `python scripts/cleanup/rewrite_imports.py --root src --map docs/development/import_rewrite_map.yaml --dry-run --strict --verbose`
  - Apply with backups: `python scripts/cleanup/rewrite_imports.py --root src --map docs/development/import_rewrite_map.yaml --backup --verbose`
- Guard:
  - Scan strict: `python scripts/cleanup/check_legacy_imports.py --root src --map docs/development/import_rewrite_map.yaml --fail --verbose`