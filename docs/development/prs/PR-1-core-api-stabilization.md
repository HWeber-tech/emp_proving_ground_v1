# PR-1: Core API stabilization and initial mappings

1) Title and summary

This PR stabilizes core import surfaces by introducing minimal, framework-agnostic facades and normalizing initial imports across trading models. It adds minimal stable facades in [src/core/interfaces.py](src/core/interfaces.py:1) and [src/core/exceptions.py](src/core/exceptions.py:1); consolidates trading model re-exports in [src/trading/models/__init__.py](src/trading/models/__init__.py:1); expands the import rewrite map in [docs/development/import_rewrite_map.yaml](docs/development/import_rewrite_map.yaml:1); removes two core mypy overrides in [pyproject.toml](pyproject.toml:1); and introduces a warning-only “core-lint” hygiene job in [.github/workflows/import-hygiene.yml](.github/workflows/import-hygiene.yml:1).

2) Files changed (explicit, with clickable refs)

- Core stable facades:
  - [src/core/interfaces.py](src/core/interfaces.py:1)
  - [src/core/exceptions.py](src/core/exceptions.py:1)
- Trading models façade:
  - [src/trading/models/__init__.py](src/trading/models/__init__.py:1)
- Rewrite map:
  - [docs/development/import_rewrite_map.yaml](docs/development/import_rewrite_map.yaml:1)
- Mypy config edits:
  - [pyproject.toml](pyproject.toml:1)
- CI (warning-only core hygiene job):
  - [.github/workflows/import-hygiene.yml](.github/workflows/import-hygiene.yml:1)

3) What changed (bulleted detail)

- Facades:
  - Introduced minimal, framework-agnostic protocols in [src/core/interfaces.py](src/core/interfaces.py:1) ([python.Cache()](src/core/interfaces.py:5), [python.EventBus()](src/core/interfaces.py:18), [python.Logger()](src/core/interfaces.py:25))
  - Introduced core exception anchors in [src/core/exceptions.py](src/core/exceptions.py:1) ([python.CoreError()](src/core/exceptions.py:12), [python.ConfigurationError()](src/core/exceptions.py:21), [python.DependencyError()](src/core/exceptions.py:26), [python.ValidationError()](src/core/exceptions.py:31))
- Trading models façade re-exports:
  - Re-export Position/Order/OrderStatus from [src/trading/models/__init__.py](src/trading/models/__init__.py:1) using safe try/except ImportError guards and __all__ population (see [src/trading/models/__init__.py](src/trading/models/__init__.py:13), [src/trading/models/__init__.py](src/trading/models/__init__.py:20), [src/trading/models/__init__.py](src/trading/models/__init__.py:27)).
- Rewrite map expansions in [docs/development/import_rewrite_map.yaml](docs/development/import_rewrite_map.yaml:1):
  - { "sources": ["core.interfaces","src.core.interfaces"], "target_module": "src.core.interfaces", "symbols": ["*"] }
  - { "sources": ["core.exceptions","src.core.exceptions"], "target_module": "src.core.exceptions", "symbols": ["*"] }
  - { "sources": ["trading.models","src.trading.models"], "target_module": "src.trading.models", "symbols": ["*"] }
- Mypy override removals in [pyproject.toml](pyproject.toml:1):
  - Removed "src.core.population_manager" and "src.core.configuration" from [[tool.mypy.overrides]] ignore list (preserved valid TOML syntax)
- CI hygiene:
  - Added non-blocking core-lint job to [.github/workflows/import-hygiene.yml](.github/workflows/import-hygiene.yml:1) running:
    - ruff check --select F401,E402,I001 src/core src/config src/data_foundation

4) Commands and observed results (Phase 1 validation)

- Rewriter strict dry-run:
  - Command:
    - python [scripts/cleanup/rewrite_imports.py](scripts/cleanup/rewrite_imports.py:1) --root src --map [docs/development/import_rewrite_map.yaml](docs/development/import_rewrite_map.yaml:1) --dry-run --strict --verbose
  - Output summary (as reported):
    - Intended rewrites in 15 files (list key paths)
    - Per-mapping hits: mapping[5]=3, mapping[12]=3, mapping[13]=2, mapping[14]=43, mapping[15]=13
    - Unresolved star-imports: 0; unresolved legacy plain imports: 0; exit 0
- Apply with backups:
  - Command:
    - python [scripts/cleanup/rewrite_imports.py](scripts/cleanup/rewrite_imports.py:1) --root src --map [docs/development/import_rewrite_map.yaml](docs/development/import_rewrite_map.yaml:1) --backup --verbose
  - Output summary (as reported):
    - Rewrote the same files; .orig backups created; hit counts same as dry-run; unresolved: 0; exit 0
- Guard:
  - Command:
    - python [scripts/cleanup/check_legacy_imports.py](scripts/cleanup/check_legacy_imports.py:1) --root src --map [docs/development/import_rewrite_map.yaml](docs/development/import_rewrite_map.yaml:1) --fail --verbose
  - Output summary:
    - “No legacy import violations detected.”; exit 0

5) Acceptance criteria for PR-1 (checklist)

- [x] Stable facades exist in Core and are importable
- [x] Trading models façade exposes Position, Order, and OrderStatus via guarded re-exports and populates __all__
- [x] Import rewrite map contains entries for core.interfaces, core.exceptions, and trading.models
- [x] Rewriter strict dry-run completes successfully with exit code 0 and expected hit counts
- [x] Rewrite applied with backups (.orig) created; hit counts match dry-run; exit code 0
- [x] Legacy import guard reports “No legacy import violations detected.”; exit code 0
- [x] Mypy overrides for "src.core.population_manager" and "src.core.configuration" removed from [pyproject.toml](pyproject.toml:1)
- [x] Warning-only Ruff core hygiene job present in [.github/workflows/import-hygiene.yml](.github/workflows/import-hygiene.yml:1) and configured to run ruff check --select F401,E402,I001 against src/core src/config src/data_foundation