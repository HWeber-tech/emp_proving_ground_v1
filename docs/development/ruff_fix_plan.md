# Ruff Excluded Paths — Per-File Fix Plan

Purpose
- Convert the generated lint report for excluded paths into a concrete, low-risk fix plan that can be executed in small PR batches, keeping CI green.
- Limit to lint hygiene (no functional refactors). Align changes with our import-normalization and Import Linter contracts posture.

Inputs
- Ruff JSON report: [docs/reports/ruff_excluded_report.json](docs/reports/ruff_excluded_report.json:1)
- Current Ruff config: [pyproject.toml](pyproject.toml:1)
- Other baseline audit artifacts (for cross-reference only, do not change in this phase):
  - Bandit: [docs/reports/security_findings.txt](docs/reports/security_findings.txt:1)
  - Vulture (dead code): [docs/reports/deadcode.txt](docs/reports/deadcode.txt:1)
  - Mypy full: [docs/reports/mypy_full.txt](docs/reports/mypy_full.txt:1)
  - Dependency fanin/fanout: [docs/reports/fanin_fanout.csv](docs/reports/fanin_fanout.csv:1)

Scope (as per pyproject excludes)
- Directories:
  - [examples](examples:1)
  - [tests/legacy](tests/legacy:1)
  - [tools](tools:1)
  - [src/validation](src/validation:1)
- Individual files (targeted):
  - [scripts/basic_test.py](scripts/basic_test.py:1)
  - [scripts/cleanup/analyze_dependencies.py](scripts/cleanup/analyze_dependencies.py:1)
  - [scripts/demo_liquidity_prober.py](scripts/demo_liquidity_prober.py:1)
  - [scripts/place_demo_order.py](scripts/place_demo_order.py:1)
  - [src/ecosystem/optimization/ecosystem_optimizer.py](src/ecosystem/optimization/ecosystem_optimizer.py:1)
  - [src/risk/risk_manager_impl.py](src/risk/risk_manager_impl.py:1)
  - [src/integration/component_integrator.py](src/integration/component_integrator.py:1)
  - [src/sensory/organs/dimensions/anomaly_dimension.py](src/sensory/organs/dimensions/anomaly_dimension.py:1)
  - [src/thinking/analysis/market_analyzer.py](src/thinking/analysis/market_analyzer.py:1)
  - [src/thinking/phase3_orchestrator.py](src/thinking/phase3_orchestrator.py:1)

Rules considered in the excluded-paths run
- Safety-focused set + import hygiene: E9, F63, F7, F82, F401, E402, I001, F403, F405

Key categories and how to fix
- I001: Sort imports consistently (isort via Ruff autofix).
- F401: Remove unused imports (Ruff autofix; if import is for side-effects, add “# noqa: F401” with justification).
- E402: Move imports to top of file. If runtime import is intentionally delayed, wrap in function or add clear comments and narrow “# noqa: E402” at the offending line.
- F403/F405: Replace wildcard imports (from x import *) with explicit imports. If public API exposure is desired, switch to explicit imports and/or define __all__ in the package initializer.
- E9/F63/F7/F82 family: Real issues (undefined names, bad unpacking, etc.). For this hygiene phase, only fix trivial, unambiguous cases. Defer larger fixes to the structural remediation batches.

Execution model
- Per directory/file below, perform:
  1) Safe autofix pass: ruff --fix-only --select I001,F401
  2) Manual import normalizations for E402 and F403/F405 (minimally invasive)
  3) Re-run lint targeted rule set; ensure zero regressions
  4) Commit with concise PR scope and acceptance criteria (see “Acceptance gates”)

How to regenerate the JSON report (reference)
- See [docs/reports/ruff_excluded_report.json](docs/reports/ruff_excluded_report.json:1) for the current run; regenerate with the same command captured in CI notes if needed.

Prioritized PR batches (no functional changes)
- Batch A (low risk, broadly safe):
  - [examples](examples:1), [tools](tools:1), [scripts/basic_test.py](scripts/basic_test.py:1), [scripts/demo_liquidity_prober.py](scripts/demo_liquidity_prober.py:1), [scripts/place_demo_order.py](scripts/place_demo_order.py:1), [scripts/cleanup/analyze_dependencies.py](scripts/cleanup/analyze_dependencies.py:1)
  - Scope: I001/F401 autofix; E402 where obvious; replace obvious star imports; do not change runtime behavior.
- Batch B (medium risk in test/support code):
  - [src/validation](src/validation:1)
  - Scope: I001/F401 autofix; E402 where obvious; avoid touching any test logic semantics; prefer localized noqa if import movement is risky.
- Batch C (targeted modules with higher coupling):
  - [src/ecosystem/optimization/ecosystem_optimizer.py](src/ecosystem/optimization/ecosystem_optimizer.py:1)
  - [src/risk/risk_manager_impl.py](src/risk/risk_manager_impl.py:1)
  - [src/integration/component_integrator.py](src/integration/component_integrator.py:1)
  - [src/sensory/organs/dimensions/anomaly_dimension.py](src/sensory/organs/dimensions/anomaly_dimension.py:1)
  - [src/thinking/analysis/market_analyzer.py](src/thinking/analysis/market_analyzer.py:1)
  - [src/thinking/phase3_orchestrator.py](src/thinking/phase3_orchestrator.py:1)
  - Scope: I001/F401 autofix; targeted E402 corrections; replace star imports; leave undefined-name or deeper errors to structural remediation PRs.

Per-directory plan and common patterns

1) examples/ (Batch A)
- Typical issues: I001 (unsorted imports), F401 (unused imports), E402 (imports not at top)
- Plan:
  - Run safe autofix; manually move “import …” to file top unless clearly used for late-binding demo behavior
  - Replace “from foo import *” with explicit imports; if the module is a quick-start demo, allow narrow “# noqa” with rationale
- Acceptance: ruff check examples --select E9,F63,F7,F82,F401,E402,I001,F403,F405 returns 0

2) tools/ (Batch A)
- Typical issues: similar to examples
- Plan:
  - Same as examples; preserve CLI/script ergonomics; prefer not to alter argparse/env assumptions
- Acceptance: ruff clean on tools with the same rule set

3) scripts/ (Batch A)
- Files: [scripts/basic_test.py](scripts/basic_test.py:1), [scripts/demo_liquidity_prober.py](scripts/demo_liquidity_prober.py:1), [scripts/place_demo_order.py](scripts/place_demo_order.py:1), [scripts/cleanup/analyze_dependencies.py](scripts/cleanup/analyze_dependencies.py:1)
- Typical issues: I001, F401, occasional E402 due to inline sys.path manipulation or dynamic imports
- Plan:
  - Safe autofix; for E402 caused by sys.path munging, prefer to keep sys.path edits at top or guard them with comments and minimal noqa
- Acceptance: ruff clean on these files

4) tests/legacy/ (Batch A if any lint present)
- Plan:
  - Safe autofix; tests that rely on asserts or import side effects may need targeted noqa; do not change test logic
- Acceptance: ruff clean on the directory

5) src/validation/ (Batch B)
- Typical issues: I001/F401 widespread, sometimes E402 in module-level fixtures or notebook-origin code
- Plan:
  - Safe autofix; E402 corrections where trivial; extensive import movement or large file reordering deferred; use narrow noqa if necessary
- Acceptance: directory clean on the rule set

Per-file targeted plan (Batch C)

a) [src/ecosystem/optimization/ecosystem_optimizer.py](src/ecosystem/optimization/ecosystem_optimizer.py:1)
- Anticipated: I001/F401; possible E402; ensure required imports (e.g., random) are at top
- Actions:
  - Safe autofix; reorder imports; remove unused (e.g., mutual_info_score if unused per vulture)
  - Replace any star imports
  - Leave undefined names or deeper refactors for structural batch

b) [src/risk/risk_manager_impl.py](src/risk/risk_manager_impl.py:1)
- Anticipated: I001/F401; import ordering; remove unused
- Actions:
  - Safe autofix; manual E402 if trivial; no functional risk logic changes in this PR

c) [src/integration/component_integrator.py](src/integration/component_integrator.py:1)
- Anticipated: I001/F401; potential E402; imports reordering
- Actions:
  - Safe autofix; manually move obvious imports to top; star import cleanup if present

d) [src/sensory/organs/dimensions/anomaly_dimension.py](src/sensory/organs/dimensions/anomaly_dimension.py:1)
- Anticipated: I001/F401; occasional E402; explicit imports
- Actions:
  - Safe autofix; straighten imports; keep dimensional API unchanged

e) [src/thinking/analysis/market_analyzer.py](src/thinking/analysis/market_analyzer.py:1)
- Anticipated: I001/F401; import order; numpy import positions (avoid late imports unless necessary)
- Actions:
  - Safe autofix; replace star imports; keep analysis semantics intact

f) [src/thinking/phase3_orchestrator.py](src/thinking/phase3_orchestrator.py:1)
- Anticipated: I001/F401; potential E402; no star imports recommended
- Actions:
  - Safe autofix; manual import relocation when obvious

Command recipes (local, per batch)
- Safe autofix pass:
  - ruff check --fix-only --select I001,F401 <paths>
- Verification pass:
  - ruff check --select E9,F63,F7,F82,F401,E402,I001,F403,F405 --isolated <paths>
- Example per directory:
  - examples, tools, scripts:
    - ruff check --fix-only --select I001,F401 examples tools scripts
    - ruff check --select E9,F63,F7,F82,F401,E402,I001,F403,F405 --isolated examples tools scripts
  - validation:
    - ruff check --fix-only --select I001,F401 src/validation
    - ruff check --select E9,F63,F7,F82,F401,E402,I001,F403,F405 --isolated src/validation
  - targeted files:
    - ruff check --fix-only --select I001,F401 src/ecosystem/optimization/ecosystem_optimizer.py src/risk/risk_manager_impl.py src/integration/component_integrator.py src/sensory/organs/dimensions/anomaly_dimension.py src/thinking/analysis/market_analyzer.py src/thinking/phase3_orchestrator.py
    - ruff check --select E9,F63,F7,F82,F401,E402,I001,F403,F405 --isolated <same list>

Acceptance gates (each PR)
- Lint: targeted paths clean for the rule set above
- Import Linter: no new contract violations (see [docs/reports/contracts_report.txt](docs/reports/contracts_report.txt:1))
- Guard/rewriter: stays green (no legacy import regressions)
- Zero functional changes (by inspection and CI signal)

Post-success follow-ups
- After Batch A and B are green, propose reducing excludes for:
  - [examples](examples:1), [tools](tools:1), [scripts](scripts:1) segments in [pyproject.toml](pyproject.toml:1)
  - [src/validation](src/validation:1) once stabilized
- After Batch C is green, propose trimming individual file excludes accordingly in [pyproject.toml](pyproject.toml:1)

Notes
- If the Ruff JSON indicates any E9/F7 serious errors in excluded paths, document them inline in the PR as “observed, deferred” unless trivial. Those flow into the structural remediation plan in [docs/development/remediation_plan.md](docs/development/remediation_plan.md:1).

Traceability
- Source report: [docs/reports/ruff_excluded_report.json](docs/reports/ruff_excluded_report.json:1)
- Config driving scope: [pyproject.toml](pyproject.toml:1)