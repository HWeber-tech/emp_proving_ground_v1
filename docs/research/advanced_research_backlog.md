# Advanced research backlog

This document satisfies the Phase 3C deliverables from the high-impact development roadmap by recording the forward-looking research programme beyond the Tier-1 production scope. It is intended to be updated quarterly by the research lead and cross-referenced from the EMP Encyclopedia change log.

## Future genetic algorithm extensions

| Extension | Description | Prerequisites | Operational checkpoint |
| --- | --- | --- | --- |
| Speciation islands | Maintain multiple sub-populations to protect diverse alpha regimes. | Distributed evaluation harness in `orchestration.evolution_cycle.EvolutionCycleOrchestrator` plus per-island seed control. | Demonstrate ≥15% Sharpe uplift versus baseline MA crossover on out-of-sample synthetic baskets.
| Pareto front promotion | Track non-dominated genomes across risk/return objectives and allow operator promotion. | Leaderboard telemetry emitted via `evolution.lineage_telemetry.EvolutionLineageSnapshot` and storage in `artifacts/evolution/`. | Operator runbook capturing promotion criteria signed off by risk.
| Adaptive mutation schedule | Adjust mutation rate based on convergence diagnostics. | Streaming metrics from `operations.evolution_tuning.evaluate_evolution_tuning` and drift thresholds in Prometheus dashboard. | Mutation heatmap included in nightly evolution report.
| Live evolution guardrails | Enable optional live-paper GA refinement gated by risk controls. | Completion of drawdown guard unit tests plus FIX shadow mode in `runtime.fix_dropcopy.FixDropcopyReconciler`. | Signed change-management ticket referencing guard activation parameters.

## NLP and news sentiment ingestion roadmap

1. **Data discovery (Weeks 1–2):** catalogue free/low-cost feeds (e.g., RSS, FinViz, SEC filings) and record licensing notes in the data governance register.
2. **Acquisition pipeline (Weeks 3–4):** extend `data_foundation.ingest.timescale_pipeline` with a textual channel that normalises headlines, timestamps, and publishers. Implement deduplication keyed by ISIN and headline hash.
3. **Feature extraction (Weeks 5–6):** deploy a lightweight transformer (FinBERT or equivalent) behind a batch scoring job, persisting embeddings to Timescale hypertables.
4. **Sentiment/risk wiring (Weeks 7–8):** expose features through `sensory.why.news_sentiment.WhyNewsSentimentSensor` (new module) and integrate with `risk.analytics.volatility_regime` as a volatility regime modifier.
5. **Governance & compliance (ongoing):** record retention limits, PII review, and opt-out workflows in `docs/policies/data_governance.md`; add automated provenance checks to `operations.configuration_audit`.

### Data governance checkpoints

- **Source audit:** verify feed licences and permitted redistribution scope with the compliance team before enabling continuous ingestion.
- **Retention policy:** default to 24-month rolling window stored in encrypted S3-compatible bucket managed by `data_foundation.storage.tiered_storage.MicrostructureTieredArchive`.
- **Incident response:** add sentiment ingestion to `operations.alerts.build_default_alert_manager` with severity mapping for feed failures and classification drifts.

## Success metrics for causal inference and ML classifiers

| Capability | Primary metrics | Guardrail metrics | Promotion threshold |
| --- | --- | --- | --- |
| Treatment effect estimators (causal forests / TMLE) | Average treatment effect confidence interval width ≤ 25 bps, policy value uplift vs baseline ≥ 10%. | Covariate balance (standardised mean difference ≤ 0.1), placebo test p-value ≥ 0.1. | Two consecutive out-of-sample quarters hitting both primary and guardrail metrics.
| Regime classification (HMM / Bayesian changepoint) | Macro-regime F1 ≥ 0.72 against labelled validation set, detection latency ≤ 3 bars. | False positive rate ≤ 8%, stability under rolling retrain (Jensen-Shannon divergence ≤ 0.05). | Promotion after 30-day paper evaluation without alert violations.
| Sentiment-driven alpha filters (transformers) | Precision@K ≥ 0.65 for actionable events, slippage-adjusted return uplift ≥ 5%. | Drift score from `operations.sensory_drift.evaluate_sensory_drift` ≤ 2σ, inference latency ≤ 200 ms. | Integration once risk review confirms capital-at-risk < 2% per deployment.
| Execution optimisation (causal impact of liquidity cues) | Cost saving ≥ 4 bps vs benchmark VWAP, confidence interval covering zero < 5%. | Post-trade slippage outliers ≤ 2 per week, FIX acknowledgement latency impact < 5%. | Requires sign-off in `docs/deployment/ops_command_checklist.md` and automated alert coverage.

## Tier-2/Tier-3 encyclopedia mapping to issue tracker

| Encyclopedia tier | Focus area | Proposed GitHub issue slug | Notes |
| --- | --- | --- | --- |
| Tier-2 | Sensor fusion – multi-venue liquidity heatmaps | `issue/sensory-liquidity-heatmaps` | Requires consolidated order book snapshots and heatmap rendering in observability dashboard.
| Tier-2 | Adaptive portfolio construction (CVaR optimisation) | `issue/risk-cvar-optimizer` | Depends on `risk.analytics.expected_shortfall` enhancements and scenario generator.
| Tier-2 | Multi-agent coordination for execution | `issue/execution-multi-agent` | Builds on `trading.liquidity.smart_routing` reinforcement hooks.
| Tier-3 | Reinforcement learning market-making pilot | `issue/rl-market-making` | Needs simulator hardening and compliance pre-clearance.
| Tier-3 | Cross-asset macro narrative engine | `issue/why-macro-narrative` | Extends WHY sensor family with macroeconomic clustering and textual analytics.
| Tier-3 | Autonomous incident triage | `issue/ops-autonomous-triage` | Leverages anomaly detection outputs plus on-call routing playbooks.

## Research debt register

| Item | Description | Owner | Status | Next review |
| --- | --- | --- | --- | --- |
| Sensor drift false positives | Quantify drift alarms caused by timezone rollovers in `sensory.when.session_calendar`. | Research engineering | Monitoring with temporary suppression rules; need permanent fix. | 2025-02-15 |
| GA compute scalability | Evaluate GPU-backed evolution runners for large genomes. | Quant research | Initial benchmarks captured; awaiting cloud cost envelope. | 2025-03-01 |
| Sentiment model bias | Audit FinBERT outputs for sector-specific skew. | Data science | Collecting validation set from S&P sectors. | 2025-03-10 |
| Microstructure storage cost | Assess tiered archive growth vs budget. | Operations | Weekly delta trending in `docs/runbooks/microstructure_storage.md`. | 2025-02-12 |
| Compliance audit trail gaps | Ensure news ingestion retains evidentiary trail. | Compliance | To design immutable hash chain integrated with `operations.regulatory_telemetry`. | 2025-02-28 |

> **Maintenance cadence:** update this register after each monthly research council meeting and link amended sections in the EMP Encyclopedia appendix.
