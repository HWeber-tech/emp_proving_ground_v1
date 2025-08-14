# PR-2: Trading/Sensory import normalization

Phase 2 validates and normalizes imports under trading.* and sensory.*, applying only targeted configuration adjustments while keeping both the strict rewriter and the guard green: no additional import-rewrite mappings were required after a strict dry-run; trading and sensory public facades require no changes; trading mypy overrides were removed and the trading exclude group was dropped from the mypy regex in [pyproject.toml](pyproject.toml:1); and ruff selectors were expanded globally in [.github/workflows/ci.yml](.github/workflows/ci.yml:1) to include F401, E402, I001.

## Files changed (explicit, with clickable refs)

- Mypy config edits:
  - [pyproject.toml](pyproject.toml:1)
- CI lint selectors (global warning-only baseline now expanded):
  - [.github/workflows/ci.yml](.github/workflows/ci.yml:1)
- Map and facades verified, no changes required:
  - [docs/development/import_rewrite_map.yaml](docs/development/import_rewrite_map.yaml:1)
  - [src/trading/models/__init__.py](src/trading/models/__init__.py:1)
  - [src/sensory/models/__init__.py](src/sensory/models/__init__.py:1)

## What changed (bulleted detail)

- Import rewrite map:
  - No new entries added for trading.* or sensory.*, as strict dry-run surfaced zero plain-root occurrences
- Public facades:
  - Trading: [src/trading/models/__init__.py](src/trading/models/__init__.py:1) remains stable and idempotent
  - Sensory: [src/sensory/models/__init__.py](src/sensory/models/__init__.py:1) remains stable; no speculative re-exports added
- Mypy configuration in [pyproject.toml](pyproject.toml:1):
  - Removed overrides (ignore_errors=true) for:
    - "src.trading.integration.*","src.trading.monitoring.*","src.trading.portfolio.*","src.trading.performance.*","src.trading.strategies.*","src.trading.execution.*"
  - Dropped trading group from exclude regex:
    - `|^src/trading/(?:integration|monitoring|portfolio|performance|strategies)/`
- CI lint in [.github/workflows/ci.yml](.github/workflows/ci.yml:1):
  - Updated ruff selectors:
    - `ruff check . --select E9,F63,F7,F82,F401,E402,I001`

## Commands and observed results (validation)

- Rewriter strict dry-run:
  - Command:
    - python [scripts/cleanup/rewrite_imports.py](scripts/cleanup/rewrite_imports.py:1) --root src --map [docs/development/import_rewrite_map.yaml](docs/development/import_rewrite_map.yaml:1) --dry-run --strict --verbose

    ```
    python scripts/cleanup/rewrite_imports.py --root src --map docs/development/import_rewrite_map.yaml --dry-run --strict --verbose
    ```

  - Output summary (as reported):
    - No files changed
    - Per-mapping hits observed: mapping[5]=3, mapping[12]=3, mapping[13]=2, mapping[14]=43, mapping[15]=13
    - Unresolved star-imports: 0; unresolved legacy plain imports: 0

- Rewriter apply with backups:
  - Command:
    - python [scripts/cleanup/rewrite_imports.py](scripts/cleanup/rewrite_imports.py:1) --root src --map [docs/development/import_rewrite_map.yaml](docs/development/import_rewrite_map.yaml:1) --backup --verbose

    ```
    python scripts/cleanup/rewrite_imports.py --root src --map docs/development/import_rewrite_map.yaml --backup --verbose
    ```

  - Output summary:
    - No files changed; per-mapping hits unchanged

- Guard (strict):
  - Command:
    - python [scripts/cleanup/check_legacy_imports.py](scripts/cleanup/check_legacy_imports.py:1) --root src --map [docs/development/import_rewrite_map.yaml](docs/development/import_rewrite_map.yaml:1) --fail --verbose

    ```
    python scripts/cleanup/check_legacy_imports.py --root src --map docs/development/import_rewrite_map.yaml --fail --verbose
    ```

  - Output summary:
    - “No legacy import violations detected.”

## Acceptance criteria for PR-2 (checklist)

- [x] No unresolved plain-root trading.* or sensory.* imports (guard + rewriter strict green)
- [x] Mypy trading overrides removed and trading exclude group dropped
- [x] Ruff selectors expanded globally to include F401,E402,I001
- [x] No speculative mapping expansions or façade changes

## Risk and rollback

- Low risk: changes limited to mypy config and CI ruff selectors
- Rollback: re-add the removed trading overrides and exclude group in [pyproject.toml](pyproject.toml:1) and restore the previous ruff selector set in [.github/workflows/ci.yml](.github/workflows/ci.yml:1) if needed

## Next steps (Phase 3 preview)

- Normalize ecosystem/thinking/intelligence imports as needed with minimal mapping additions and facades
- Remove domain overrides from [pyproject.toml](pyproject.toml:1) and shrink exclude regex accordingly
- Enable ruff F403/F405 globally and flip Import Linter to mandatory