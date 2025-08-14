# Comprehensive Codebase Audit — Stakeholder Deck v1

Executive summary
- Baseline quality and architecture artifacts produced and versioned.
- Security: 0 Highs; 10 Medium issues (SQL construction, eval) to remediate immediately.
- Types: 718 mypy errors across 113 files; plan to reduce in structural batches.
- Dependency posture: 300 modules, 299 edges; high-central modules identified for hardening.
- Import hygiene: excluded-paths Ruff report generated; concrete fix plan prepared; Import Linter baseline in place.
- CI: guard/rewriter stable; telemetry logging for mapping hits operational.

Key artifacts (click to open)
- Ruff excluded report: [docs/reports/ruff_excluded_report.json](docs/reports/ruff_excluded_report.json:1)
- Security findings (bandit): [docs/reports/security_findings.txt](docs/reports/security_findings.txt:1)
- Mypy full probe: [docs/reports/mypy_full.txt](docs/reports/mypy_full.txt:1)
- Dead code scan: [docs/reports/deadcode.txt](docs/reports/deadcode.txt:1)
- Dependency graph (DOT): [docs/reports/dependency_graph.dot](docs/reports/dependency_graph.dot:1)
- Fan-in/Fan-out CSV: [docs/reports/fanin_fanout.csv](docs/reports/fanin_fanout.csv:1)
- Import contracts: [contracts/importlinter.toml](contracts/importlinter.toml:1)
- Ruff fix plan: [docs/development/ruff_fix_plan.md](docs/development/ruff_fix_plan.md:1)
- Hotspots: [docs/development/hotspots.md](docs/development/hotspots.md:1)
- Remediation plan: [docs/development/remediation_plan.md](docs/development/remediation_plan.md:1)
- Mapping hits logger: [scripts/cleanup/log_mapping_hits.py](scripts/cleanup/log_mapping_hits.py:1)
- Mapping hits CSV: [docs/reports/imports_mapping_hits.csv](docs/reports/imports_mapping_hits.csv:1)
- Project lint config: [pyproject.toml](pyproject.toml:1)

Baseline metrics (from reports)
- Bandit totals: Low 259, Medium 10, High 0 (see [docs/reports/security_findings.txt](docs/reports/security_findings.txt:2775)).
- Mypy: 718 errors in 113 files; 247 files checked (see [docs/reports/mypy_full.txt](docs/reports/mypy_full.txt:822)).
- Dependency graph: Modules 300, Edges 299 (generator summary; see [docs/reports/fanin_fanout.csv](docs/reports/fanin_fanout.csv:1)).
- Import mapping hotspots: mapping[14] (core.interfaces) hits 43; mapping[15] (core.exceptions) hits 13 (see [docs/reports/imports_mapping_hits.csv](docs/reports/imports_mapping_hits.csv:5)).

Top hotspots (see detail)
- See [docs/development/hotspots.md](docs/development/hotspots.md:1) for the ranked Top-10 with owners and actions.

Remediation strategy (phases)
- Phase 0 (Security P0): Eliminate B608 (SQL) and B307 (eval) in targeted files; remove B108 hardcoded tmp and reduce exception swallowing in critical paths.
- Phase A–C (Hygiene only): Execute [docs/development/ruff_fix_plan.md](docs/development/ruff_fix_plan.md:1) across excluded paths; no behavior changes.
- Phase D (Structural): Stabilize [src/core/interfaces.py](src/core/interfaces.py:1) contracts; align consumers; introduce typed facades for enhanced sensory modules.
- Excludes reduction: After A–C are green, trim [pyproject.toml](pyproject.toml:1) excludes.
- Guard allow‑list removal: Run strict audit in [.github/workflows/import-hygiene.yml](.github/workflows/import-hygiene.yml:1) then remove allow‑list in [scripts/cleanup/check_legacy_imports.py](scripts/cleanup/check_legacy_imports.py:1).

Acceptance gates (per PR)
- Import architecture: Import Linter baseline steady (see [contracts/importlinter.toml](contracts/importlinter.toml:1)); guard/rewriter green.
- Lint hygiene: ruff clean for E9,F63,F7,F82,F401,E402,I001,F403,F405 on the touched scope.
- Security: Bandit Medium/High do not increase; P0 buckets reach zero in the security PRs.
- Types: mypy-full non‑blocking; error count must not increase and should decrease in structural refactors.

CI workflows and telemetry
- Quality pipeline: [.github/workflows/ci.yml](.github/workflows/ci.yml:1) runs Ruff/Mypy/Tests; mypy-full is non‑blocking probe.
- Import hygiene: [.github/workflows/import-hygiene.yml](.github/workflows/import-hygiene.yml:1) enforces guard/rewriter and Import Linter.
- Mapping hits trend: run [scripts/cleanup/log_mapping_hits.py](scripts/cleanup/log_mapping_hits.py:1); data in [docs/reports/imports_mapping_hits.csv](docs/reports/imports_mapping_hits.csv:1).

Timeline (indicative)
- Week 1: Phase 0 security + Phase A hygiene.
- Week 2: Phase B hygiene; begin Phase C targeted files.
- Week 3–4: Complete Phase C; start Phase D (interfaces + 1–2 consumer modules).
- Week 5+: Continue Phase D; trim excludes; remove guard allow‑list.

Resourcing & ownership
- Security P0: “Security & Data” with “Trading” support.
- Hygiene A: “DX/Tooling”.
- Hygiene B: “Validation/QA”.
- Hygiene C: Module owners (Ecosystem, Risk, Integration, Sensory, Thinking).
- Structural D: “Core & Integration” with module owners.

Risks and mitigations
- Behavior changes during security fixes: mitigate with narrow tests and parameterized SQL fallbacks.
- Import churn causing regressions: guard/rewriter and Import Linter gates prevent legacy patterns; mapping telemetry watched.
- Third‑party typing gaps: add types‑stubs and typed facades before tightening mypy.

Decision & next actions
- Approve Phase 0 + Phase A start immediately.
- Confirm owners for batches and the success metrics above.
- Schedule stakeholder touchpoints at end of each phase; deck will be updated with fresh artifacts.