# PR-5: Zero mypy suppressions and full-signal posture (probe)

Phase 5 removes remaining mypy suppressions and flips posture toward full signal while keeping CI green. This PR:
- Removes module-level mypy ignores and broad excludes.
- Sets mypy to report missing imports (no silent suppression).
- Adds a separate non-blocking CI probe that runs full mypy (no CLI suppressions).
- Confirms import rewrite/guard passes.

## Files changed
- [pyproject.toml](pyproject.toml:1)
- [.github/workflows/ci.yml](.github/workflows/ci.yml:1) (added mypy-full probe)

## Configuration changes (mypy)
- ignore_missing_imports=false
- disable_error_code removed (including "import-untyped")
- exclude regex removed
- overrides table removed; zero ignored modules remain

## CI updates
- New job "mypy-full" added in [.github/workflows/ci.yml](.github/workflows/ci.yml:1)
  - runs-on: ubuntu-latest
  - continue-on-error: true (warning-only)
  - steps: checkout, setup-python 3.11, pip install requirements (if present) and mypy, then run: mypy src
- Existing "quality" job unchanged and still runs: mypy --ignore-missing-imports src

## Validation: import rewriter (strict dry-run)
- Command:
  - python [scripts/cleanup/rewrite_imports.py](scripts/cleanup/rewrite_imports.py:1) --root src --map [docs/development/import_rewrite_map.yaml](docs/development/import_rewrite_map.yaml:1) --dry-run --strict --verbose
- Result summary:
  - No files changed (dry-run)
  - Per-mapping hit counts (id: count; human label):
    - 5: 3 — trading.models.order[*] -> src.trading.models.order
    - 12: 3 — core.performance.market_data_cache[*] -> src.core.performance.market_data_cache
    - 13: 2 — trading.models.position[*] -> src.trading.models.position
    - 14: 43 — core.interfaces[*] -> src.core.interfaces
    - 15: 13 — core.exceptions[*] -> src.core.exceptions
  - Unresolved star-imports: 0
  - Unresolved legacy plain imports: 0

## Validation: legacy import guard
- Command:
  - python [scripts/cleanup/check_legacy_imports.py](scripts/cleanup/check_legacy_imports.py:1) --root src --map [docs/development/import_rewrite_map.yaml](docs/development/import_rewrite_map.yaml:1) --fail --verbose
- Result summary:
  - No legacy import violations detected.
  - Allowed files skipped per guard defaults: src/phase2d_integration_validator.py, src/core/sensory_organ.py, src/intelligence/red_team_ai.py, src/sensory/models/__init__.py

## Acceptance checklist
- [x] Zero mypy overrides remain
- [x] mypy exclude removed
- [x] ignore_missing_imports=false; disable_error_code "import-untyped" removed
- [x] Import rewriter strict dry-run shows no required edits; unresolved counts are zero
- [x] Legacy import guard green
- [x] mypy-full probe job added (non-blocking)
- [x] Main quality job unchanged

## Risk and rollback
- The probe surfaces typing gaps without blocking merges (continue-on-error). If third-party stub gaps cause noise, keep the probe to monitor trend; no CI breakage expected since the quality job remains suppressed.
- If needed, revert by removing the "mypy-full" job or restoring prior mypy settings in [pyproject.toml](pyproject.toml:1). No code changes were made by the tools in this phase.

## Next steps
- After the probe is consistently green, follow up by making "mypy-full" mandatory and removing "--ignore-missing-imports" from the quality job’s mypy invocation in [.github/workflows/ci.yml](.github/workflows/ci.yml:1).
- Consider incrementally enabling stricter mypy flags per module as readiness improves.