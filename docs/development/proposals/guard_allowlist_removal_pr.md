# Guard Allow‑List Elimination PR Proposal — import guard in strict mode

Objective
- Eliminate the default allow‑list in the legacy‑import guard so all legacy imports are detected and blocked unless rewritten.
- Keep guard/rewriter and Import Linter green; avoid breaking CI by landing this through a phased, observable change.

Key references
- Guard script: [scripts/cleanup/check_legacy_imports.py](scripts/cleanup/check_legacy_imports.py:1)
- Rewriter map (source of truth): [docs/development/import_rewrite_map.yaml](docs/development/import_rewrite_map.yaml:1)
- Import contracts: [contracts/importlinter.toml](contracts/importlinter.toml:1)
- CI workflow (guard job): [.github/workflows/import-hygiene.yml](.github/workflows/import-hygiene.yml:1)
- Mapping telemetry: [scripts/cleanup/log_mapping_hits.py](scripts/cleanup/log_mapping_hits.py:1), [docs/reports/imports_mapping_hits.csv](docs/reports/imports_mapping_hits.csv:1)

Plan (two‑step)

Step 1 — Add a “strict probe” job in CI (no default allow)
- Create a second job in [.github/workflows/import-hygiene.yml](.github/workflows/import-hygiene.yml:1) named import-hygiene-strict that:
  - Runs the same steps as the baseline guard job
  - Passes an environment flag (e.g. IMPORT_GUARD_STRICT=1) to the guard invocation
  - Does NOT fail the pipeline yet (continue-on-error: true) — this is a probe job

- Guard invocation (preferred flags; adjust to actual script interface):
  - python [scripts/cleanup/check_legacy_imports.py](scripts/cleanup/check_legacy_imports.py:1) --root src --map [docs/development/import_rewrite_map.yaml](docs/development/import_rewrite_map.yaml:1) --no-default-allow --fail-on-legacy
  - If the script lacks flags, use an env‑switch in code (fallback in the next section).

- Output handling:
  - Upload CI artifact “import-guard-strict-report” containing:
    - Raw guard log
    - Diff of legacy imports detected (with file:line)
    - Current mapping hits snapshot [docs/reports/imports_mapping_hits.csv](docs/reports/imports_mapping_hits.csv:1)

Step 2 — Remove default allow‑list and make strict the default
- After triaging and remediating findings (below), update the baseline guard job to run strict mode by default:
  - Remove DEFAULT_ALLOWED_FILES in [scripts/cleanup/check_legacy_imports.py](scripts/cleanup/check_legacy_imports.py:1)
  - Delete the probe job; keep a single strict guard job

Script adaptation (if CLI flags don’t exist)
- In [scripts/cleanup/check_legacy_imports.py](scripts/cleanup/check_legacy_imports.py:1):
  - Introduce an env switch:
    - if os.getenv("IMPORT_GUARD_STRICT") == "1": DEFAULT_ALLOWED_FILES = frozenset()
  - Or add a --no-default-allow CLI flag that sets the in‑process allow‑list to empty.
- Keep all logic behind a small function so the strictness is easy to test.

Triage workflow for the strict probe
- Classify each legacy import occurrence:
  1) True legacy path covered by a rewrite rule
     - Add/extend mapping in [docs/development/import_rewrite_map.yaml](docs/development/import_rewrite_map.yaml:1)
  2) Intentional one‑off alias or external boundary
     - Prefer a proper import path or a dedicated allowed‑graph edge via Import Linter; avoid adding permanent allow‑list entries in the guard
  3) False positive or special case
     - Narrow the detector; add a specific exception (regex/file‑line) only if justified and documented in code

- After updating the map:
  - Run [scripts/cleanup/rewrite_imports.py](scripts/cleanup/rewrite_imports.py:1) in dry‑run (and then in non‑dry if green)
  - Log mapping hits: python -u [scripts/cleanup/log_mapping_hits.py](scripts/cleanup/log_mapping_hits.py:1)
  - Re‑run the strict probe job

Acceptance gates
- Guard strict probe finds: 0 unresolved legacy imports (PR turns strict on by default afterwards)
- Import Linter: no new contract violations (config at [contracts/importlinter.toml](contracts/importlinter.toml:1))
- Rewriter: no regressions; mapping hits trend non‑increasing in [docs/reports/imports_mapping_hits.csv](docs/reports/imports_mapping_hits.csv:1)
- Lint hygiene unchanged (ruff CI remains green)
- Minimal code delta in guard script (toggle removal + env/flag support)

PR contents
- Commit 1: Add strict probe job in [.github/workflows/import-hygiene.yml](.github/workflows/import-hygiene.yml:1) (continue-on-error: true), add guard strict flag/env handling
- Commit 2+: Remediate legacy imports via rewrite map and import fixes (small batches)
- Final commit: Remove DEFAULT_ALLOWED_FILES path and delete probe job; make strict default

Rollback
- Revert the final strict commit to restore previous behavior
- Retain the strict probe job (non‑blocking) for observation if needed

Notes
- Keep changes to the guard focused and reversible.
- Prefer rewrite rules + Import Linter contracts over re‑introducing allow‑lists.