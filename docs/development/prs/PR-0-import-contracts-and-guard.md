# PR-0: Import Contracts and Guard Baseline

We added Import Linter contracts and introduced a warning-only CI job to surface import hygiene issues without blocking merges. We validated that the legacy import guard and the import rewriter strict dry-run both pass cleanly, establishing a green baseline for Phase 0.

## Files changed

- contracts file: [contracts/importlinter.toml](contracts/importlinter.toml:1)
- workflow updated: [.github/workflows/import-hygiene.yml](.github/workflows/import-hygiene.yml:1)

## Contracts overview

- [contracts/importlinter.toml](contracts/importlinter.toml:1) sets:
  - layers contract across "src" with top→bottom ordering: UI/Operational → Domain → Core
  - independence across domain packages to prevent cross-domain imports
  - root_package="src", include_external_packages=false
- Note: These contracts are non-blocking initially via a separate CI job.

## CI updates

- New (warning-only) job added to [.github/workflows/import-hygiene.yml](.github/workflows/import-hygiene.yml:1):
  - Job name: "Import Linter (warning-only)"
  - Installs import-linter and runs with config [contracts/importlinter.toml](contracts/importlinter.toml:1)
  - continue-on-error: true (non-blocking)
- Existing guard steps still run:
  - "Legacy import guard" using [scripts/cleanup/check_legacy_imports.py](scripts/cleanup/check_legacy_imports.py:1)
  - "Import rewrite dry-run (strict)" using [scripts/cleanup/rewrite_imports.py](scripts/cleanup/rewrite_imports.py:1)

## Commands and observed results (this run)

### Guard strict

- Command:
  - python [scripts/cleanup/check_legacy_imports.py](scripts/cleanup/check_legacy_imports.py:1) --root src --map [docs/development/import_rewrite_map.yaml](docs/development/import_rewrite_map.yaml:1) --fail --verbose

```bash
python scripts/cleanup/check_legacy_imports.py --root src --map docs/development/import_rewrite_map.yaml --fail --verbose
```

- Output summary:
  - Allowed files auto-skipped: [src/phase2d_integration_validator.py](src/phase2d_integration_validator.py:1), [src/core/sensory_organ.py](src/core/sensory_organ.py:1), [src/intelligence/red_team_ai.py](src/intelligence/red_team_ai.py:1), [src/sensory/models/__init__.py](src/sensory/models/__init__.py:1)
  - Result: "No legacy import violations detected."

### Rewriter strict dry-run

- Command:
  - python [scripts/cleanup/rewrite_imports.py](scripts/cleanup/rewrite_imports.py:1) --root src --map [docs/development/import_rewrite_map.yaml](docs/development/import_rewrite_map.yaml:1) --dry-run --strict --verbose

```bash
python scripts/cleanup/rewrite_imports.py --root src --map docs/development/import_rewrite_map.yaml --dry-run --strict --verbose
```

- Output summary:
  - Per-mapping hit counts: mapping[5]=3, mapping[12]=3, mapping[13]=2
  - Unresolved star-imports: 0
  - Unresolved legacy plain imports: 0
  - No files changed (dry-run)

## Acceptance criteria for PR-0

- [x] Import Linter job appears in CI and executes (non-blocking).
- [x] Guard green with "No legacy import violations detected."
- [x] Rewriter strict dry-run green with "Unresolved star-imports: 0" and "Unresolved legacy plain imports: 0".
- [x] No functional code changes in this PR; contracts and CI job only.

## Risk and rollback

- Low risk: contracts job is non-blocking.
- Rollback if needed: remove [contracts/importlinter.toml](contracts/importlinter.toml:1) and revert the changes to [.github/workflows/import-hygiene.yml](.github/workflows/import-hygiene.yml:1).

## Next steps (Phase 1 preview)

- Introduce core facades: [src/core/interfaces/__init__.py](src/core/interfaces/__init__.py:1), [src/core/exceptions.py](src/core/exceptions.py:1).
- Expand rewrite map [docs/development/import_rewrite_map.yaml](docs/development/import_rewrite_map.yaml:1) with:
  - ["core.interfaces"] → "src.core.interfaces" (star)
  - ["core.exceptions"] → "src.core.exceptions" (star)
  - ["trading.models"] → "src.trading.models" (star)
- Remove mypy overrides for: "src.core.population_manager", "src.core.configuration" in [pyproject.toml](pyproject.toml:1).
- Add core-only ruff step in CI for src/core, src/config, src/data_foundation with F401, E402, I001 checks.