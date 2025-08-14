# Stage B Hygiene PR Proposal — validation suites + ecosystem optimizer

Scope
- Directories:
  - [src/validation](src/validation:1)
- Targeted file (coupled to validation signals and mypy/ruff noise):
  - [src/ecosystem/optimization/ecosystem_optimizer.py](src/ecosystem/optimization/ecosystem_optimizer.py:1)
- Config reference: [pyproject.toml](pyproject.toml:1) current excludes include validation and ecosystem_optimizer.py

Goal
- Pure lint hygiene with minimal/no behavior changes:
  - Fix I001 (import sorting) and F401 (unused imports) via safe autofix
  - Correct obvious E402 (imports not at top) where moving imports is risk-free
  - Replace star imports to avoid F403/F405 where explicit names are clear
  - Address trivial F821 (undefined name) only when it is an unmistakable typo/import omission; otherwise defer to structural PRs
- Keep CI green, preserve runtime behavior for validation utilities and optimizer

Inputs
- Ruff excluded-paths report: [docs/reports/ruff_excluded_report.json](docs/reports/ruff_excluded_report.json:1)
- Import Linter baseline: [docs/reports/contracts_report.txt](docs/reports/contracts_report.txt:1)
- Fix-plan baseline: [docs/development/ruff_fix_plan.md](docs/development/ruff_fix_plan.md:1)
- Hotspots for reference: [docs/development/hotspots.md](docs/development/hotspots.md:1)

Execution steps

1) Safe autofix (imports only)
- Validation directory:
  - ruff check --fix-only --select I001,F401 src/validation
- Optimizer file:
  - ruff check --fix-only --select I001,F401 src/ecosystem/optimization/ecosystem_optimizer.py

2) Manual adjustments (surgical)
- E402: Move imports to file top where it cannot alter lazy-loading behavior or test fixtures. If lazy import is essential, keep as-is and add a narrow comment rationale or a focused “# noqa: E402” at the single offending line.
- F403/F405: Replace star imports with explicit symbols. If a test relies on star semantics or side-effects, restrict the suppression to that exact line with “# noqa: F403,F405” and a short rationale.
- F821: Only fix trivial cases:
  - Add a missing import when the intent is unambiguous (e.g., “random” used without import).
  - Correct obvious typos for local identifiers.
  - Do not introduce stubs or logical refactors here; defer anything non-trivial to the structural phase.

3) Verification passes
- Ruff on the batch with an expanded safety set (includes undefined names in this phase):
  - ruff check --select E9,F63,F7,F82,F401,E402,I001,F403,F405,F821 --isolated src/validation src/ecosystem/optimization/ecosystem_optimizer.py
- Import Linter:
  - Confirm no new contract violations against [contracts/importlinter.toml](contracts/importlinter.toml:1); attach summary delta compared to [docs/reports/contracts_report.txt](docs/reports/contracts_report.txt:1)
- Guard/rewriter:
  - Ensure guard remains green; mapping hits do not regress (see [docs/reports/imports_mapping_hits.csv](docs/reports/imports_mapping_hits.csv:1))

Acceptance gates
- Lint clean for the scope with rules: E9, F63, F7, F82, F401, E402, I001, F403, F405, F821
- Import Linter: no new violations; guard/rewriter green
- No increase in Bandit Medium/High counts (security unchanged)
- No functional changes (inspection + smoke where applicable)

Risk and rollback
- Low risk; changes limited to import statements and trivial undefined name fixes
- Immediate rollback via PR revert; no persisted state

Owners
- Validation/QA: primary ownership for [src/validation](src/validation:1)
- Ecosystem module owners: review [src/ecosystem/optimization/ecosystem_optimizer.py](src/ecosystem/optimization/ecosystem_optimizer.py:1)
- DX/Tooling support for ruff/isort guidance and CI wiring

Artifacts to attach in PR
- Excerpts from [docs/reports/ruff_excluded_report.json](docs/reports/ruff_excluded_report.json:1) that motivate the batch
- Before/after ruff summaries for the exact scope
- Import Linter summary (no new violations)
- Optional: mapping hits snapshot from [docs/reports/imports_mapping_hits.csv](docs/reports/imports_mapping_hits.csv:1)

Follow-up (post-merge)
- If this batch is green and stable, proceed with Stage C targeted modules (see remediation plan [docs/development/remediation_plan.md](docs/development/remediation_plan.md:1))
- Start planning excludes reduction for validation/optimizer entries in [pyproject.toml](pyproject.toml:1) once Stage C concludes cleanly