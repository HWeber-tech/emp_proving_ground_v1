# Comprehensive Remediation Plan (Phased)

Objective
- Convert audit findings into minimally risky, incremental PR batches that improve security, import hygiene, type posture, and architecture conformance while keeping CI green at all times.
- Preserve runtime behavior unless explicitly called out (security P0).

Inputs
- Ruff excluded-paths JSON: [docs/reports/ruff_excluded_report.json](docs/reports/ruff_excluded_report.json:1)
- Security scan (bandit): [docs/reports/security_findings.txt](docs/reports/security_findings.txt:1)
- Mypy full probe: [docs/reports/mypy_full.txt](docs/reports/mypy_full.txt:1)
- Dead code (vulture): [docs/reports/deadcode.txt](docs/reports/deadcode.txt:1)
- Dependency fanin/fanout: [docs/reports/fanin_fanout.csv](docs/reports/fanin_fanout.csv:1)
- Import contracts baseline: [docs/reports/contracts_report.txt](docs/reports/contracts_report.txt:1)
- Excluded-paths Ruff fix plan: [docs/development/ruff_fix_plan.md](docs/development/ruff_fix_plan.md:1)
- Hotspots summary: [docs/development/hotspots.md](docs/development/hotspots.md:1)

Guardrails and Acceptance Gates (apply to every PR)
- Import hygiene gate:
  - Import Linter: no new contract violations (compare against [docs/reports/contracts_report.txt](docs/reports/contracts_report.txt:1))
  - Guard/Rewriter: stays green; no resurgence of legacy imports.
- Lint gate:
  - For files in-scope of the PR, ruff check passes for: E9, F63, F7, F82, F401, E402, I001, F403, F405 (same set used for excluded-paths report).
- Security gate:
  - No increases in bandit Medium/High severity totals; P0 items addressed in dedicated security PRs must hit zero for the targeted files.
- Type gate:
  - Non-blocking: mypy-full job must not regress (errors count does not increase); in typed refactor PRs, a defined subset must be reduced.
- Tests and behavior:
  - No functional changes unless explicitly stated (security P0) and validated with targeted tests or script runs.
- Documentation:
  - Update CHANGELOG section in the PR description summarizing improvements and linking evidence lines in [docs/reports/*](docs/reports/mypy_full.txt:1).

Phasing Overview

Phase 0 — Security P0 Remediations (dedicated, fast-follow PRs)
1) SQL construction hardening (B608)
   - Files/evidence:
     - [src/data_foundation/ingest/yahoo_ingest.py](src/data_foundation/ingest/yahoo_ingest.py:64)
     - [src/data_foundation/ingest/yahoo_ingest.py](src/data_foundation/ingest/yahoo_ingest.py:66)
     - [src/trading/portfolio/real_portfolio_monitor.py](src/trading/portfolio/real_portfolio_monitor.py:454)
     - [src/trading/portfolio/real_portfolio_monitor.py](src/trading/portfolio/real_portfolio_monitor.py:491)
   - Actions:
     - Replace string interpolation/f-strings with parameterized queries or API-native executemany/select constructs.
     - For DuckDB/SQLite, use placeholders (e.g., “?”) and pass parameters list/tuple safely.
   - Acceptance:
     - Bandit: B608 count for these files → 0; CI green.
     - Basic smoke test: run the affected code paths with safe sample inputs.

2) eval replacement (B307)
   - Files/evidence:
     - [src/thinking/adaptation/tactical_adaptation_engine.py](src/thinking/adaptation/tactical_adaptation_engine.py:301)
     - [src/thinking/adversarial/market_gan.py](src/thinking/adversarial/market_gan.py:299)
     - [src/thinking/adversarial/red_team_ai.py](src/thinking/adversarial/red_team_ai.py:702)
     - [src/thinking/competitive/competitive_intelligence_system.py](src/thinking/competitive/competitive_intelligence_system.py:887)
     - [src/thinking/prediction/predictive_market_modeler.py](src/thinking/prediction/predictive_market_modeler.py:453)
   - Actions:
     - Replace eval(...) with ast.literal_eval(...) where JSON-like structures are parsed.
     - If non-literal inputs occur, switch to json.loads(...) or a constrained schema parser; validate types.
   - Acceptance:
     - Bandit: B307 count for these files → 0; CI green.
     - Add narrow unit test(s) for parsing logic to prevent regressions.

3) Hardcoded tmp directory (B108) and try/except/pass (B110-B112)
   - Files (examples):
     - [src/governance/system_config.py](src/governance/system_config.py:28) → use tempfile.gettempdir() or platformdirs
     - Many operational metrics sites (e.g., [src/operational/metrics.py](src/operational/metrics.py:103)) swallow exceptions
   - Actions:
     - Replace os.getenv TMP fallback with stdlib temp APIs.
     - Replace blanket except: with narrow exceptions; log warnings; keep counters.
   - Acceptance:
     - Bandit: B108 down to 0 for the file(s) addressed in this PR.
     - For exception swallowing removals: add minimal logging; verify no noisy failures during smoke run.

Phase A — Hygiene-Only PRs (Excluded-paths Batch A; zero functional changes)
Scope (see plan): [examples](examples:1), [tools](tools:1), [scripts/basic_test.py](scripts/basic_test.py:1), [scripts/demo_liquidity_prober.py](scripts/demo_liquidity_prober.py:1), [scripts/place_demo_order.py](scripts/place_demo_order.py:1), [scripts/cleanup/analyze_dependencies.py](scripts/cleanup/analyze_dependencies.py:1), tests/legacy
- Actions:
  - Run safe autofix: ruff --fix-only --select I001,F401 on the scope.
  - Manually resolve E402 (imports to top) where obvious.
  - Replace star imports; if demo-only side effects are intended, use explicit imports or add narrow # noqa with rationale.
- Acceptance:
  - Lint clean for the listed rules on the scope.
  - No import-linter/guard regressions.

Phase B — Hygiene-Only PRs (Excluded-paths Batch B)
Scope: [src/validation](src/validation:1)
- Actions:
  - Same hygiene pattern as Phase A.
  - Avoid changing test semantics; prefer localized noqa if import movement carries risk.
- Acceptance:
  - Lint clean for the listed rules on the scope.
  - No regression in mypy full count.

Phase C — Hygiene-Only PRs (Excluded-paths Batch C)
Scope: targeted modules
- [src/ecosystem/optimization/ecosystem_optimizer.py](src/ecosystem/optimization/ecosystem_optimizer.py:1)
- [src/risk/risk_manager_impl.py](src/risk/risk_manager_impl.py:1)
- [src/integration/component_integrator.py](src/integration/component_integrator.py:1)
- [src/sensory/organs/dimensions/anomaly_dimension.py](src/sensory/organs/dimensions/anomaly_dimension.py:1)
- [src/thinking/analysis/market_analyzer.py](src/thinking/analysis/market_analyzer.py:1)
- [src/thinking/phase3_orchestrator.py](src/thinking/phase3_orchestrator.py:1)
- Actions:
  - Safe autofix + import reshuffle; star import elimination.
  - Defer undefined names/contract mismatches to Phase D structural.
- Acceptance:
  - Lint clean; no import-linter regressions.

Phase D — Structural Refactors (Contracts, Types, Architecture)
1) Core interface stabilization and consumer alignment
   - Hubs: [src/core/interfaces/__init__.py](src/core/interfaces/__init__.py:1) (fanin 20), [src/core/events.py](src/core/events.py:1) (fanin 13) — see [docs/reports/fanin_fanout.csv](docs/reports/fanin_fanout.csv:22)
   - Mypy signals: many “has no attribute” errors in consumers (e.g., [src/risk/risk_manager_impl.py](src/risk/risk_manager_impl.py:1), [src/integration/component_integrator.py](src/integration/component_integrator.py:1))
   - Actions:
     - Define Protocols/dataclasses in interfaces with minimal surface needed by consumers; add TODO for deeper cleanup.
     - Update consumers to align with Protocols; remove ad-hoc attributes or plug via adapters.
   - Acceptance:
     - Reduce mypy errors for targeted files by N% (target: 20–30% reduction within scope).
     - Keep lint/import contracts green.

2) Sensory “enhanced” module resolution
   - Evidence: import-not-found for src.sensory.enhanced.* (see [docs/reports/mypy_full.txt](docs/reports/mypy_full.txt:1))
   - Options:
     - Introduce no-op facades with typed stubs to satisfy imports.
     - Or feature-flag these paths and remove imports from runtime path.
   - Acceptance:
     - Eliminate import-not-found errors for the chosen subset.
     - Preserve behavior by defaulting to “disabled” mode with clear logs.

3) Exception policy & observability
   - Replace try/except/pass with:
     - Narrow exceptions (ValueError, KeyError, etc.)
     - Structured logs and counters (reusing metrics registry when available)
   - Apply first to operational metrics and mock FIX code: [src/operational/metrics.py](src/operational/metrics.py:1), [src/operational/mock_fix.py](src/operational/mock_fix.py:1)
   - Acceptance:
     - Bandit B110/B112 reduced measurably on touched files; CI green.

4) Dead code reduction (batchable)
   - Use [docs/reports/deadcode.txt](docs/reports/deadcode.txt:1) as source of truth.
   - In each batch:
     - Remove unused functions/classes/vars or mark deprecated with clear comment.
     - Keep public interfaces intact unless unused publicly.
   - Acceptance:
     - Ruff F401/F403 declines for modules touched; maintainability metrics do not regress.

Type Hygiene & Stubs Strategy
- Add first-party stubs where needed and introduce “types-” packages where available:
  - pandas, psutil, PyYAML, sklearn, scipy, yfinance, textblob, feedparser
- Do not broadly enable check_untyped_defs yet.
- Gradually tighten mypy settings for touched modules only after stubs land and interfaces are aligned.
- Re-run mypy-full probe; ensure error count decreases over time (tracked in [docs/reports/mypy_full.txt](docs/reports/mypy_full.txt:1)).

Import Hygiene and Excludes Reduction
- After Phases A–C are green, propose trimming these excludes in [pyproject.toml](pyproject.toml:1):
  - examples, tools, scripts segments
  - src/validation
  - Per-file items covered in Batch C
- Submit separate PR focused only on config change; prove with re-run of Ruff on entire tree.

Guard Allow-List Elimination
- Run strict job (no default allow) in [.github/workflows/import-hygiene.yml](.github/workflows/import-hygiene.yml:1).
- Triages any findings; then remove DEFAULT_ALLOWED_FILES from [scripts/cleanup/check_legacy_imports.py](scripts/cleanup/check_legacy_imports.py:1).
- Prove with green guard runs and zero mapping hits spike (track [docs/reports/imports_mapping_hits.csv](docs/reports/imports_mapping_hits.csv:1)).

Batch Ownership Matrix (suggested)
- Security P0: “Security & Data” with “Trading” support
- Hygiene A: “DX/Tooling”
- Hygiene B: “Validation/QA”
- Hygiene C: Owners of each module (Ecosystem, Risk, Integration, Sensory, Thinking)
- Structural D: “Core & Integration” with module owners

PR Templates (checklist)
- Scope (files/dirs)
- Changes: Hygiene-only vs Structural (and whether P0 security)
- Commands run and outputs:
  - Ruff selected rule set on touched scope (attach summary)
  - Import Linter summary (attach delta)
  - Bandit diff (if security)
  - Optional mypy subset diff
- Risk and validation notes; links to evidence:
  - [docs/reports/ruff_excluded_report.json](docs/reports/ruff_excluded_report.json:1)
  - [docs/reports/security_findings.txt](docs/reports/security_findings.txt:1)
  - [docs/reports/mypy_full.txt](docs/reports/mypy_full.txt:1)
  - [docs/reports/fanin_fanout.csv](docs/reports/fanin_fanout.csv:1)
  - [docs/development/ruff_fix_plan.md](docs/development/ruff_fix_plan.md:1)
  - [docs/development/hotspots.md](docs/development/hotspots.md:1)

Operational Details & Recipes
- Safe autofix (example):
  - ruff check --fix-only --select I001,F401 <paths>
- Verification:
  - ruff check --select E9,F63,F7,F82,F401,E402,I001,F403,F405 --isolated <paths>
- Mapping telemetry:
  - python -u [scripts/cleanup/log_mapping_hits.py](scripts/cleanup/log_mapping_hits.py:1)
- Security rescan (bandit):
  - bandit -r src -q -o [docs/reports/security_findings.txt](docs/reports/security_findings.txt:1) -f txt
- Mypy probe (non-blocking):
  - mypy --config-file pyproject.toml --namespace-packages --cache-dir .mypy_cache | tee [docs/reports/mypy_full.txt](docs/reports/mypy_full.txt:1)

Success Metrics
- Security: Zero Medium/High bandit items in P0 scope; trend down thereafter.
- Lint hygiene: Excluded-paths set reduced by ≥ 50% after Phases A–C.
- Types: ≥ 20% error reduction on targeted modules per structural batch.
- Import architecture: guard/rewriter stable; mapping hits trend down in [docs/reports/imports_mapping_hits.csv](docs/reports/imports_mapping_hits.csv:1).

Rollback & Risk
- Each batch isolated; revertible by PR.
- Structural refactors gated by Protocols and adapters, reducing blast radius.
- Logging added where exceptions were swallowed improves diagnosability post-merge.

Timeline (indicative)
- Week 1: Phase 0 (Security P0), Phase A
- Week 2: Phase B, begin Phase C
- Week 3–4: Phase C completion and Phase D first slice (interfaces + 1–2 modules)
- Week 5+: Continue Phase D, trim excludes, remove guard allow-list

Traceability
- All batch PRs reference this plan; each PR updates the relevant report artifacts. Stakeholder status summarized in the deck at [docs/development/audit_stakeholder_deck.md](docs/development/audit_stakeholder_deck.md:1) (to be produced).