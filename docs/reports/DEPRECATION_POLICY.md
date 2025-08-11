# Deprecation and Backward-Compatibility Policy — Phase 1

Purpose
- Provide safe, low-risk unification of duplicate definitions by introducing canonical modules and temporary re-export shims.
- Maintain import stability for downstream code during the migration window.
- Prevent reintroduction of duplicates via conventions and CI checks.

Context
- Inputs produced by the scanner:
  - [docs/reports/duplicate_map_classes.csv](docs/reports/duplicate_map_classes.csv)
  - [docs/reports/duplicate_map_functions.csv](docs/reports/duplicate_map_functions.csv)
  - [docs/reports/duplicate_map.json](docs/reports/duplicate_map.json)
- Canonical decisions outlined in:
  - [docs/reports/CANONICALIZATION_PLAN.md](docs/reports/CANONICALIZATION_PLAN.md)
- Scanner implementation:
  - [scripts/cleanup/duplicate_map.py](scripts/cleanup/duplicate_map.py)

Policy Overview
- Definitions (classes, functions, simple data types) must exist in exactly one canonical module.
- Legacy locations keep only a thin re-export (shim) pointing to the canonical module.
- New code imports exclusively from the canonical module.
- No implementations in package-level __init__.py files; only re-exports are allowed there.
- Behavior changes are out-of-scope in Phase 1; this phase is structural only.

Re-export Shim Rules
- Shims must:
  - Import the symbol(s) directly from the canonical module.
  - Avoid additional logic or side effects.
  - Declare a minimal __all__ to advertise the exported surface.
- Example (pattern only):
  ```
  # Legacy module (shim-only)
  from canonical.package.module import Thing as Thing

  __all__ = ["Thing"]
  ```
- For modules re-exporting multiple types, keep them flat and explicit. No wildcard imports.

Public Import Path Policy
- Canonical locations are domain-aligned (see [docs/reports/CANONICALIZATION_PLAN.md](docs/reports/CANONICALIZATION_PLAN.md)).
- New features and refactors must import from canonical modules only.
- Do not add new symbols to legacy modules; they remain shims until removal.

Timeline and Deprecation Markers
- T0 (introduce canonical + shims):
  - Create canonical modules per family.
  - Convert legacy modules to re-exports only.
  - Update documentation and announce canonical paths.
- T0 → T1 (call-site migration window):
  - Gradually update imports across the repo to canonical paths.
  - Keep shims in place to avoid breakage.
- T1 (pre-removal validation):
  - Repo-wide search validates that no references to legacy shims remain (except sanctioned external users, if any).
  - CI guard enforces no new duplicates and no new imports from legacy paths.
- T2 (removal):
  - Remove re-export shims and obsolete files.
  - Update release notes.

Naming and Casing Conventions
- Keep class/function names unchanged when unifying; only move definitions.
- Prefer snake_case modules, PascalCase classes, snake_case functions.
- Avoid duplicate type names across domains unless the concept is domain-specific and disambiguated by module path.

__init__.py Usage
- Allowed:
  - Explicit re-exports of canonical types (for ergonomic imports).
- Not allowed:
  - Implementations (class/function bodies).
  - Side-effectful initialization tied to type definitions.

Exceptions
- Script entrypoints named “main” are not unified (they are separate scripts by design).
- Temporary adapters are permitted only when required to bridge incompatible signatures; these are Phase 2/3 activities and must not be placed in shims.

PR Checklist (Phase 1)
- For each duplicate family:
  - [ ] Canonical module created and contains the only implementation.
  - [ ] Legacy modules reduced to re-export shims (no logic).
  - [ ] References in changed packages point to canonical path.
  - [ ] Update to documentation completed:
        [docs/reports/CANONICALIZATION_PLAN.md](docs/reports/CANONICALIZATION_PLAN.md),
        [docs/reports/CLEANUP_REPORT.md](docs/reports/CLEANUP_REPORT.md)
  - [ ] Verified no new duplicates introduced by running:
        python [duplicate_map.py](../../scripts/cleanup/duplicate_map.py) --root src --out docs/reports --min-count 2
  - [ ] Added or confirmed __all__ surfaces in shims.

Tooling and Automation
- Duplicate scanner: [scripts/cleanup/duplicate_map.py](scripts/cleanup/duplicate_map.py)
- Optional CI guard (Phase 1/2):
  - A job that runs the scanner and fails if:
    - Duplicate group counts increase, or
    - New duplicate families appear over a configured threshold, or
    - Implementations are detected in __init__.py for families listed as canonicalized.
- Optional pre-commit hook (developers):
  - Run the scanner on staged Python files and block commits that introduce new duplicates.

Enforcement
- During the migration window (T0 → T1), PRs adding new code must:
  - Target canonical modules for new definitions.
  - Use canonical imports.
  - Include re-export updates only when removing implementations from legacy modules.

Recordkeeping
- Append resolved families to a “Resolved Duplicates” table in:
  - [docs/reports/CLEANUP_REPORT.md](docs/reports/CLEANUP_REPORT.md)
- Include: concept name, canonical path, legacy shim paths, removal ETA (T2).

End State (post T2)
- Single source-of-truth per concept.
- No legacy re-export shims in the tree.
- CI guard prevents regressions by scanning in every build.