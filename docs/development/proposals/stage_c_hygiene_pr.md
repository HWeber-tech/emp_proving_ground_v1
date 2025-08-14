# Stage C Hygiene PR Proposal — targeted high-coupling modules

Scope (files)
- [src/risk/risk_manager_impl.py](src/risk/risk_manager_impl.py:1)
- [src/integration/component_integrator.py](src/integration/component_integrator.py:1)
- [src/sensory/organs/dimensions/anomaly_dimension.py](src/sensory/organs/dimensions/anomaly_dimension.py:1)
- [src/thinking/analysis/market_analyzer.py](src/thinking/analysis/market_analyzer.py:1)
- [src/thinking/phase3_orchestrator.py](src/thinking/phase3_orchestrator.py:1)

Context
- These files are currently excluded in [pyproject.toml](pyproject.toml:1) and generate concentrated hygiene noise (imports).
- Several are high-touch or high-fanout surfaces (integration, orchestrators) and should be cleaned before structural refactors.

Goal
- Pure import hygiene with no behavioral changes:
  - I001: import sorting (isort via Ruff)
  - F401: remove unused imports (or justify with narrow “# noqa: F401” if side-effects intended)
  - E402: ensure imports at top where safe; otherwise use narrow comment/noqa with rationale
  - F403/F405: eliminate star imports; switch to explicit symbols
  - F821: ONLY trivial fixes (e.g., missing obvious import like “random”); defer non-trivial to structural batches
- Keep CI green; do not change runtime semantics.

Inputs
- Ruff excluded-paths report: [docs/reports/ruff_excluded_report.json](docs/reports/ruff_excluded_report.json:1)
- Import contract baseline: [docs/reports/contracts_report.txt](docs/reports/contracts_report.txt:1)
- Hotspots: [docs/development/hotspots.md](docs/development/hotspots.md:1)
- Plan baseline: [docs/development/ruff_fix_plan.md](docs/development/ruff_fix_plan.md:1)

Execution steps

1) Safe autofix (imports only)
- Run import sorting and unused import removal for the exact files in scope:
  - ruff check --fix-only --select I001,F401 src/risk/risk_manager_impl.py src/integration/component_integrator.py src/sensory/organs/dimensions/anomaly_dimension.py src/thinking/analysis/market_analyzer.py src/thinking/phase3_orchestrator.py

2) Manual adjustments (surgical)
- E402: move imports to top if safe; if lazy import is necessary (performance/optional deps), add a one-line comment and a narrow “# noqa: E402” only at the precise line.
- F403/F405: replace wildcard imports with explicit names; if not feasible due to dynamic exposure, define explicit imports and augment __all__ in the relevant package initializer instead of star usage.
- F821: fix only trivial/obvious missing imports; anything deeper (contract mismatches, undefined attributes) is deferred to structural refactors (Phase D).

3) Verification passes
- Ruff against the safety set (includes undefined names for detection but only trivial fixes applied):
  - ruff check --select E9,F63,F7,F82,F401,E402,I001,F403,F405,F821 --isolated src/risk/risk_manager_impl.py src/integration/component_integrator.py src/sensory/organs/dimensions/anomaly_dimension.py src/thinking/analysis/market_analyzer.py src/thinking/phase3_orchestrator.py
- Import Linter:
  - Validate no new contract violations against [contracts/importlinter.toml](contracts/importlinter.toml:1); include summary delta relative to [docs/reports/contracts_report.txt](docs/reports/contracts_report.txt:1)
- Guard/rewriter:
  - Ensure guard remains green and mapping hits do not spike; spot-check [docs/reports/imports_mapping_hits.csv](docs/reports/imports_mapping_hits.csv:1)

Acceptance gates
- Lint clean for the rules: E9, F63, F7, F82, F401, E402, I001, F403, F405, F821 (trivial only)
- Import Linter: no new violations; guard/rewriter green
- No increase in Bandit Medium/High
- No runtime behavior changes (inspection + smoke as applicable)

Risk and rollback
- Low risk; changes confined to import statements and trivial missing imports
- Fast rollback via PR revert; no persistent data or schema changes

Owners
- Risk module: Risk owner(s)
- Integration module: Integration owner(s)
- Sensory anomaly dimension: Sensory owner(s)
- Thinking analyzer and orchestrator: Thinking owner(s)
- DX/Tooling supports lint/CI scripting

Artifacts to attach in PR
- Excerpts from [docs/reports/ruff_excluded_report.json](docs/reports/ruff_excluded_report.json:1) specific to the scope
- Before/after Ruff summaries
- Import Linter result snippet proving no new violations
- Optional mapping hits snapshot delta from [docs/reports/imports_mapping_hits.csv](docs/reports/imports_mapping_hits.csv:1)

Follow-up (post-merge)
- With A/B/C batches green, prepare the excludes reduction PR to trim entries for these files in [pyproject.toml](pyproject.toml:1) as per [docs/development/remediation_plan.md](docs/development/remediation_plan.md:1).