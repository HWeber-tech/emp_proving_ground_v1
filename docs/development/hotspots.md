# Top-10 Architecture & Quality Hotspots

Sources analyzed
- Ruff excluded-paths report: [docs/reports/ruff_excluded_report.json](docs/reports/ruff_excluded_report.json:1)
- Bandit security: [docs/reports/security_findings.txt](docs/reports/security_findings.txt:1)
- Mypy full probe: [docs/reports/mypy_full.txt](docs/reports/mypy_full.txt:1)
- Dead code (vulture): [docs/reports/deadcode.txt](docs/reports/deadcode.txt:1)
- Dependency fanin/fanout: [docs/reports/fanin_fanout.csv](docs/reports/fanin_fanout.csv:1)

Severity scale
- P0 = must fix before other work; P1 = high priority; P2 = important but can batch

1) SQL construction (B608) in data ingestion and portfolio reporting  [P0]
Owners: Data Foundation, Trading
Evidence:
- [src/data_foundation/ingest/yahoo_ingest.py](src/data_foundation/ingest/yahoo_ingest.py:64)
- [src/data_foundation/ingest/yahoo_ingest.py](src/data_foundation/ingest/yahoo_ingest.py:66)
- [src/trading/portfolio/real_portfolio_monitor.py](src/trading/portfolio/real_portfolio_monitor.py:454)
- [src/trading/portfolio/real_portfolio_monitor.py](src/trading/portfolio/real_portfolio_monitor.py:491)
Action: parameterize SQL, avoid f-strings; prefer placeholders and vetted helpers.

2) Use of eval (B307) on untrusted strings  [P0]
Owners: Thinking
Evidence:
- [src/thinking/adaptation/tactical_adaptation_engine.py](src/thinking/adaptation/tactical_adaptation_engine.py:301)
- [src/thinking/adversarial/market_gan.py](src/thinking/adversarial/market_gan.py:299)
- [src/thinking/adversarial/red_team_ai.py](src/thinking/adversarial/red_team_ai.py:702)
- [src/thinking/competitive/competitive_intelligence_system.py](src/thinking/competitive/competitive_intelligence_system.py:887)
- [src/thinking/prediction/predictive_market_modeler.py](src/thinking/prediction/predictive_market_modeler.py:453)
Action: replace with ast.literal_eval or safe parsers; gate/validate inputs.

3) Core interfaces drift and missing definitions  [P1]
Owners: Core, Integration
Evidence:
- [src/core/interfaces/__init__.py](src/core/interfaces/__init__.py:1) high fanin per deps ([docs/reports/fanin_fanout.csv](docs/reports/fanin_fanout.csv:22))
- Mypy: missing attributes in consumers; see e.g. [src/risk/risk_manager_impl.py](src/risk/risk_manager_impl.py:151), [src/integration/component_integrator.py](src/integration/component_integrator.py:1)
Action: stabilize IRiskManager/IMarketAnalyzer/IExecutionEngine contracts; align implementations; add Protocols where appropriate.

4) High-central modules with many dependents  [P1]
Owners: Core, Integration
Evidence:
- [src/core/interfaces/__init__.py](src/core/interfaces/__init__.py:1) fanin 20 ([docs/reports/fanin_fanout.csv](docs/reports/fanin_fanout.csv:22))
- [src/core/events.py](src/core/events.py:1) fanin 13 ([docs/reports/fanin_fanout.csv](docs/reports/fanin_fanout.csv:14))
- [src/integration/component_integrator_impl.py](src/integration/component_integrator_impl.py:1) fanout 12 ([docs/reports/fanin_fanout.csv](docs/reports/fanin_fanout.csv:90))
Action: add tests around these hubs; enforce import-linter layering; limit public surface.

5) Widespread try/except/pass swallowing telemetry & errors  [P1]
Owners: Operational
Evidence (examples):
- [src/operational/metrics.py](src/operational/metrics.py:103)
- [src/operational/metrics.py](src/operational/metrics.py:125)
- [src/operational/metrics.py](src/operational/metrics.py:132)
- [src/operational/metrics.py](src/operational/metrics.py:139)
Action: replace with narrow exception capturing and structured logging; add counters; avoid blanket pass.

6) Type hygiene gaps and missing stubs across data/science libs  [P1]
Owners: Core, Data Foundation
Evidence:
- Mypy missing stubs: pandas/psutil/sklearn/yfinance/scipy in [docs/reports/mypy_full.txt](docs/reports/mypy_full.txt:1)
- Numerous import-not-found across sensory.* and trading.* packages
Action: add types-* stubs; gate third-party modules behind typed facades; adjust mypy config only after stubs.

7) Undefined-name/contract mismatch in ecosystem optimizer and related  [P1]
Owners: Ecosystem, Thinking
Evidence:
- [src/ecosystem/optimization/ecosystem_optimizer.py](src/ecosystem/optimization/ecosystem_optimizer.py:169)
- [src/thinking/analysis/market_analyzer.py](src/thinking/analysis/market_analyzer.py:126)
Action: import correctness; align signatures; add targeted unit tests.

8) Sensory enhanced-* module references unresolved  [P1]
Owners: Sensory
Evidence:
- [src/sensory/organs/dimensions/integration_orchestrator.py](src/sensory/organs/dimensions/integration_orchestrator.py:20)
- Widespread mypy import-not-found for src.sensory.enhanced.* in [docs/reports/mypy_full.txt](docs/reports/mypy_full.txt:1)
Action: implement or stub enhanced modules; or remove/feature-flag them; update dependencies.

9) Dead code clusters inflating complexity footprint  [P2]
Owners: Governance, Intelligence, Trading
Evidence:
- [src/governance/audit_logger.py](src/governance/audit_logger.py:17)
- [src/intelligence/competitive_intelligence.py](src/intelligence/competitive_intelligence.py:83)
- [src/trading/monitoring/performance_tracker.py](src/trading/monitoring/performance_tracker.py:414)
Action: prune unused classes/methods or mark deprecated; keep public APIs slim.

10) Excluded paths lint hygiene to enable excludes reduction  [P2]
Owners: All
Evidence:
- Report: [docs/reports/ruff_excluded_report.json](docs/reports/ruff_excluded_report.json:1)
- Fix plan: [docs/development/ruff_fix_plan.md](docs/development/ruff_fix_plan.md:1)
Action: execute Batch A→B→C from the plan; then trim [pyproject.toml](pyproject.toml:1) excludes.

Next steps
- Draft remediation batches with owners and acceptance gates in [docs/development/remediation_plan.md](docs/development/remediation_plan.md:1)
- Run mapping hits logger to update [docs/reports/imports_mapping_hits.csv](docs/reports/imports_mapping_hits.csv:1) and feed trends into the plan
- Keep Import Linter and guard/rewriter green at all times