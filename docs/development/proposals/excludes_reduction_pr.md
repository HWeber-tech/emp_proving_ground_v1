# Excludes Reduction PR Proposal — Trim [tool.ruff.exclude] in [pyproject.toml](pyproject.toml:1)

Purpose
- Remove temporary Ruff excludes after hygiene batches A/B/C merge green.
- Expand CI lint coverage without breaking the build.
- Keep Import Linter + guard/rewriter posture unchanged.

Preconditions
- Stage A (examples/tools/scripts/tests-legacy) merged green → see [docs/development/proposals/stage_a_hygiene_pr.md](docs/development/proposals/stage_a_hygiene_pr.md:1)
- Stage B (validation + optimizer) merged green → see [docs/development/proposals/stage_b_hygiene_pr.md](docs/development/proposals/stage_b_hygiene_pr.md:1)
- Stage C (targeted modules) merged green → see [docs/development/proposals/stage_c_hygiene_pr.md](docs/development/proposals/stage_c_hygiene_pr.md:1)
- Import Linter baseline and guard/rewriter are green (reference: [docs/reports/contracts_report.txt](docs/reports/contracts_report.txt:1), mapping telemetry [docs/reports/imports_mapping_hits.csv](docs/reports/imports_mapping_hits.csv:1))

Current excludes (to be trimmed in phases) from [pyproject.toml](pyproject.toml:4)
- Directories
  - "examples"
  - "tests/legacy"
  - "tools"
  - "src/validation"
- Individual files
  - "scripts/basic_test.py"
  - "scripts/cleanup/analyze_dependencies.py"
  - "scripts/demo_liquidity_prober.py"
  - "scripts/place_demo_order.py"
  - "src/ecosystem/optimization/ecosystem_optimizer.py"
  - "src/risk/risk_manager_impl.py"
  - "src/integration/component_integrator.py"
  - "src/sensory/organs/dimensions/anomaly_dimension.py"
  - "src/thinking/analysis/market_analyzer.py"
  - "src/thinking/phase3_orchestrator.py"

Plan — Two-step reduction
1) Step 1 (after A+B green):
   - Remove:
     - "examples"
     - "tests/legacy"
     - "tools"
     - "scripts/basic_test.py"
     - "scripts/cleanup/analyze_dependencies.py"
     - "scripts/demo_liquidity_prober.py"
     - "scripts/place_demo_order.py"
     - "src/validation"
     - "src/ecosystem/optimization/ecosystem_optimizer.py"
2) Step 2 (after C green):
   - Remove:
     - "src/risk/risk_manager_impl.py"
     - "src/integration/component_integrator.py"
     - "src/sensory/organs/dimensions/anomaly_dimension.py"
     - "src/thinking/analysis/market_analyzer.py"
     - "src/thinking/phase3_orchestrator.py"

Change scope
- Only modify [tool.ruff.exclude] array in [pyproject.toml](pyproject.toml:1).
- Do not change [tool.ruff.lint.select] or per-file-ignores in the same PR.

Verification commands
- Lint (safety-focused set used in audit):
  - ruff check --select E9,F63,F7,F82,F401,E402,I001,F403,F405 --output-format text --isolated .
- Optional, focused re-run on previously-excluded items:
  - ruff check --select E9,F63,F7,F82,F401,E402,I001,F403,F405 --isolated examples tools scripts src/validation
  - ruff check --select E9,F63,F7,F82,F401,E402,I001,F403,F405 --isolated src/ecosystem/optimization/ecosystem_optimizer.py
  - ruff check --select E9,F63,F7,F82,F401,E402,I001,F403,F405 --isolated src/risk/risk_manager_impl.py src/integration/component_integrator.py src/sensory/organs/dimensions/anomaly_dimension.py src/thinking/analysis