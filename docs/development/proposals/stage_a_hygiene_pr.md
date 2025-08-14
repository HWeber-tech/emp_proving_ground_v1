# Stage A Hygiene PR Proposal — examples/tools/scripts/tests-legacy

Scope
- Directories: [examples](examples:1), [tools](tools:1), [tests/legacy](tests/legacy:1)
- Files: [scripts/basic_test.py](scripts/basic_test.py:1), [scripts/demo_liquidity_prober.py](scripts/demo_liquidity_prober.py:1), [scripts/place_demo_order.py](scripts/place_demo_order.py:1), [scripts/cleanup/analyze_dependencies.py](scripts/cleanup/analyze_dependencies.py:1)
- Config context: [pyproject.toml](pyproject.toml:1) [tool.ruff.exclude] entries drive this batch.

Goal
- Pure lint hygiene: fix I001 (import sorting) and F401 (unused imports), and correct obvious E402 (imports not at top) and star imports (F403/F405) when safe.
- No behavior changes; keep demos and scripts functional.

Inputs
- Ruff excluded-paths report: [docs/reports/ruff_excluded_report.json](docs/reports/ruff_excluded_report.json:1)
- Import contracts baseline: [docs/reports/contracts_report.txt](docs/reports/contracts_report.txt:1)
- Fix plan reference: [docs/development/ruff_fix_plan.md](docs/development/ruff_fix_plan.md:1)

Execution steps
1) Safe autofix pass (imports only):
   - ruff check --fix-only --select I001,F401 examples tools scripts
   - ruff check --fix-only --select I001,F401 scripts\basic_test.py scripts\demo_liquidity_prober.py scripts\place_demo_order.py scripts\cleanup\analyze_dependencies.py

2) Manual adjustments (surgical):
   - Resolve simple E402 by moving imports to file top when it does not change CLI/runtime behavior.
   - Replace from module import * with explicit names; if demo relies on side-effects, add a narrow '# noqa: F403,F405' with rationale.
   - Do not refactor logic; only import hygiene.

3) Verification passes
   - Ruff (same safety set):
     ruff check --select E9,F63,F7,F82,F401,E402,I001,F403,F405 --isolated examples tools scripts scripts\basic_test.py scripts\demo_liquidity_prober.py scripts\place_demo_order.py scripts\cleanup\analyze_dependencies.py
   - Import Linter: confirm no new violations; see [docs/reports/contracts_report.txt](docs/reports/contracts_report.txt:1).
   - Guard/rewriter: run and ensure no legacy import regressions.

Acceptance gates
- Lint clean on touched scope for: E9,F63,F7,F82,F401,E402,I001,F403,F405.
- Import Linter and guard remain green.
- No increase in Bandit Medium/High counts.
- Zero functional changes (by inspection).

Risk and rollback
- Low risk; changes confined to imports.
- Rollback by reverting the PR; no persistent state changes.

Owners
- DX/Tooling as primary; directory code owners to review.

Artifacts to attach in PR
- Summary from: [docs/reports/ruff_excluded_report.json](docs/reports/ruff_excluded_report.json:1).
- Before/after ruff summaries for the scope.
- Import Linter summary diff.

Follow-up (post-merge)
- If stable, start trimming excludes for examples/tools/scripts in [pyproject.toml](pyproject.toml:1) in a separate PR.